using System.Diagnostics;
using System.Text.RegularExpressions;

namespace FunAsrNet;

/// <summary>
/// 一次识别产生的文本、音频时长、推理计时与生成 token 数。
/// </summary>
/// <param name="Text">识别得到的文本。</param>
/// <param name="AudioSeconds">输入音频时长，单位为秒。</param>
/// <param name="EncoderSeconds">编码器与特征阶段计时，单位为秒。</param>
/// <param name="DecoderSeconds">解码器计时，单位为秒。</param>
/// <param name="GeneratedTokens">本次输出的 token 数量。</param>
public sealed record AsrResult(
    string Text,
    double AudioSeconds,
    double EncoderSeconds,
    double DecoderSeconds,
    int GeneratedTokens)
{
    /// <summary>
    /// 提示词与音频嵌入预填充耗时，单位为秒。
    /// </summary>
    public double PrefillSeconds
    {
        get; init;
    }
    /// <summary>
    /// 逐 token 自回归生成耗时，单位为秒。
    /// </summary>
    public double GenerationSeconds
    {
        get; init;
    }
    /// <summary>
    /// 编码与解码阶段的总耗时，单位为秒。
    /// </summary>
    public double TotalSeconds => EncoderSeconds + DecoderSeconds;
    /// <summary>
    /// 识别总耗时除以音频时长；小于一表示计算快于音频播放。
    /// </summary>
    public double RealTimeFactor => TotalSeconds / AudioSeconds;
}

/// <summary>
/// 纯 C# 识别上下文；同一实例须串行使用，不同实例可共享只读模型权重。
/// </summary>
public sealed class FunAsr
{
    private readonly FilterBank _fbank = new();
    private readonly TensorWorkspace _workspace = new();
    private readonly FunEncoder _encoder;
    private readonly QwenDecoder _decoder;
    private readonly TensorMath _math;
    private readonly int[] _prefix, _suffix;
    private bool _prefixCached;
    /// <summary>
    /// 是否缓存固定提示词前缀；应在开始识别前完成配置。
    /// </summary>
    public bool CachePrompt { get; set; } = Environment.GetEnvironmentVariable("FUNASRNET_DISABLE_PROMPT_CACHE") != "1";
    /// <summary>
    /// 是否复用同步识别作用域内的临时张量。
    /// </summary>
    public bool ReuseBuffers { get; set; } = Environment.GetEnvironmentVariable("FUNASRNET_WORKSPACE") != "0";

    /// <summary>可选计算后端，须在首次识别前配置；支持的 Q8 后端保留原权重与分组量化语义。</summary>
    public IFloatComputeBackend? FloatBackend
    {
        get => _math.FloatBackend;
        set => _math.FloatBackend = value;
    }
    /// <summary>
    /// 可选的中间张量观察回调，仅供验证使用；开启后会产生额外快照分配。
    /// </summary>
    public Action<string, float[]>? Trace
    {
        get; set;
    }
    /// <summary>
    /// 可选的线性投影性能观察回调，参数为权重名称和耗时秒数。
    /// </summary>
    public Action<string, double>? LinearMeasured
    {
        get => _math.LinearMeasured; set => _math.LinearMeasured = value;
    }

    /// <summary>
    /// 纯 C# 识别上下文；同一实例须串行使用，不同实例可共享只读模型权重。
    /// </summary>
    /// <param name="encoderPath">F16 编码器 GGUF 文件路径。</param>
    /// <param name="decoderPath">Q8 解码器 GGUF 文件路径。</param>
    /// <param name="threads">算子使用的最大计算线程数。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    public FunAsr(string encoderPath, string decoderPath, int threads = 4, bool simd = true) : this(new Gguf(encoderPath), new Gguf(decoderPath), threads, simd)
    {
    }

    /// <summary>
    /// 纯 C# 识别上下文；同一实例须串行使用，不同实例可共享只读模型权重。
    /// </summary>
    /// <param name="encoderModel">已加载并将在发布前完成布局准备的编码器模型。</param>
    /// <param name="decoderModel">已加载并将在发布前完成布局准备的解码器模型。</param>
    /// <param name="threads">算子使用的最大计算线程数。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    public FunAsr(Gguf encoderModel, Gguf decoderModel, int threads = 4, bool simd = true)
    {
        var math = _math = new TensorMath(threads, simd);
        _encoder = new FunEncoder(encoderModel, math);
        _decoder = new QwenDecoder(decoderModel, math);
        _encoder.Trace = EmitTrace;
        _decoder.Trace = EmitTrace;
        _prefix = _decoder.Tokenizer.Encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：");
        _suffix = _decoder.Tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n");
    }

    /// <summary>
    /// 将 16 kHz 单声道采样编码并贪心解码为文本；该实例不得同时执行另一识别任务。
    /// </summary>
    /// <param name="pcm16k">16 kHz 单声道浮点采样，在同步推理返回前须保持有效。</param>
    /// <param name="maxTokens">本次最多生成的 token 数，用于限制解码时间与缓存占用。</param>
    /// <returns>识别文本、音频时长、编码与解码计时，以及生成 token 数。</returns>
    public AsrResult Transcribe(ReadOnlySpan<float> pcm16k, int maxTokens = 128)
    {
        using var workspace = _workspace.Enter(ReuseBuffers);
        using var affinity = _math.PinCaller();
        var watch = Stopwatch.StartNew();
        var fbank = _fbank.Compute(pcm16k, out int frames);
        EmitTrace("fbank", fbank);
        var audio = _encoder.Encode(fbank, frames);
        double encoderTime = watch.Elapsed.TotalSeconds;
        int ol = 1 + (frames - 1) / 2;
        ol = 1 + (ol - 1) / 2;
        int audioTokens = (ol - 1) / 2 + 1;
        if (CachePrompt && !_prefixCached)
        {
            _decoder.CachePrefix(_prefix);
            _prefixCached = true;
        }

        int prefixLength = CachePrompt ? 0 : _prefix.Length;
        int count = prefixLength + audioTokens + _suffix.Length, width = _decoder.Width;
        var embeddings = TensorWorkspace.Allocate(count * width);
        if (!CachePrompt)
        {
            var prefix = _decoder.Embed(_prefix);
            prefix.CopyTo(embeddings, 0);
            TensorWorkspace.Release(prefix);
        }

        audio.AsSpan(0, audioTokens * width).CopyTo(embeddings.AsSpan(prefixLength * width));
        var suffix = _decoder.Embed(_suffix);
        suffix.CopyTo(embeddings, (prefixLength + audioTokens) * width);
        TensorWorkspace.Release(suffix);
        TensorWorkspace.Release(audio);
        if (CachePrompt)
        {
            _decoder.RestorePrefix();
        }
        else
        {
            _decoder.Reset();
        }

        var logits = _decoder.Decode(embeddings, count);
        List<int> tokens = [];
        double prefillTime = watch.Elapsed.TotalSeconds - encoderTime;
        for (int i = 0; i < maxTokens; ++i)
        {
            int token = QwenDecoder.Greedy(logits);
            if (_decoder.Tokenizer.IsEnd(token))
            {
                break;
            }

            tokens.Add(token);
            TensorWorkspace.Release(logits);
            logits = _decoder.Decode(_decoder.Embed([token]), 1);
        }

        string text = _decoder.Tokenizer.Decode(tokens);
        text = Regex.Replace(text, @"<[^>]*>|\[[^\]]*\]|/sil|endofbreak|FFFF|[\]&|Ｏ＆｜]", "");
        text = Regex.Replace(text, @"\s+", " ").Trim();
        double decoderTime = watch.Elapsed.TotalSeconds - encoderTime;
        return new AsrResult(text, pcm16k.Length / 16000.0, encoderTime, decoderTime, tokens.Count)
        {
            PrefillSeconds = prefillTime,
            GenerationSeconds = decoderTime - prefillTime

        };
    }

    // 观察者可能跨请求保存数据，因此跟踪输出必须使用独立快照。
    /// <summary>
    /// 向跟踪观察者发布独立快照，避免暴露可复用的临时张量。
    /// </summary>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <param name="values">按顺序排列的浮点数据。</param>
    private void EmitTrace(string name, float[] values)
    {
        if (Trace is { } observer)
        {
            observer(name, (float[])values.Clone());
        }
    }
}
