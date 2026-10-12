using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// Qwen3 解码器，包含分组查询注意力、半分割 RoPE、Q/K RMSNorm 与 SwiGLU。
/// </summary>
public sealed class QwenDecoder
{
    private readonly Gguf _model;
    private readonly TensorMath _math;
    private readonly int _width, _heads, _kvHeads, _headSize, _layers, _capacity;
    private readonly float _epsilon, _ropeBase;
    private readonly float[][] _keys, _values;
    private readonly float[] _ropeCos, _ropeSin;
    private readonly Weight[][] _qkvWeights, _gateUpWeights;
    private int _position;
    private int _cachedPrefixLength;
    private int _cacheFrames;
    /// <summary>
    /// 与当前 Qwen 模型配套的字节级分词器。
    /// </summary>
    public ByteBpe Tokenizer
    {
        get;
    }
    /// <summary>
    /// 解码器隐藏层通道数量。
    /// </summary>
    public int Width => _width;
    /// <summary>
    /// 可选的中间张量观察回调，仅供验证使用；开启后会产生额外快照分配。
    /// </summary>
    public Action<string, float[]>? Trace
    {
        get; set;
    }

    /// <summary>
    /// Qwen3 解码器，包含分组查询注意力、半分割 RoPE、Q/K RMSNorm 与 SwiGLU。
    /// </summary>
    /// <param name="model">已加载的模型及其权重；发布给并发上下文后不得修改权重。</param>
    /// <param name="math">用于矩阵与向量运算的托管计算内核。</param>
    /// <param name="capacity">缓存或队列容量上限。</param>
    public QwenDecoder(Gguf model, TensorMath math, int capacity = 2048)
    {
        _model = model;
        _math = math;
        _capacity = capacity;
        // Q8 交错内核依赖 ARM SDOT；其他平台保留行主序，使用已有向量点积。
        if (math.Simd && math.Optimized && Dp.IsSupported && Environment.GetEnvironmentVariable("FUNASRNET_INTERLEAVE") != "0")
        {
            foreach (var w in model.Tensors.Values)
            {
                if (w.Quantized is not null && w.Name != "token_embd.weight")
                {
                    w.PrepareInterleaved();
                }
            }
        }

        _width = model.Int("qwen3.embedding_length", 1024);
        _heads = model.Int("qwen3.attention.head_count", 16);
        _kvHeads = model.Int("qwen3.attention.head_count_kv", 8);
        _headSize = model.Int("qwen3.attention.key_length", 128);
        _layers = model.Int("qwen3.block_count", 28);
        _epsilon = model.Float("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
        _qkvWeights = new Weight[_layers][];
        _gateUpWeights = new Weight[_layers][];
        for (int layer = 0; layer < _layers; ++layer)
        {
            string p = $"blk.{layer}.";
            _qkvWeights[layer] = [model[p + "attn_q.weight"], model[p + "attn_k.weight"], model[p + "attn_v.weight"]];
            _gateUpWeights[layer] = [model[p + "ffn_gate.weight"], model[p + "ffn_up.weight"]];
        }

        _ropeBase = model.Float("qwen3.rope.freq_base", 1000000);
        _ropeCos = new float[capacity * _headSize / 2];
        _ropeSin = new float[_ropeCos.Length];
        for (int t = 0; t < capacity; ++t)
        {
            for (int i = 0; i < _headSize / 2; ++i)
            {
                float angle = t * MathF.Pow(_ropeBase, -2f * i / _headSize);
                _ropeCos[t * (_headSize / 2) + i] = MathF.Cos(angle);
                _ropeSin[t * (_headSize / 2) + i] = MathF.Sin(angle);
            }
        }

        if (_heads % _kvHeads != 0)
        {
            throw new InvalidDataException("Invalid GQA head counts");
        }

        _keys = new float[_layers][];
        _values = new float[_layers][];
        _cacheFrames = Math.Min(capacity, 256);
        for (int i = 0; i < _layers; ++i)
        {
            _keys[i] = new float[_cacheFrames * _kvHeads * _headSize];
            _values[i] = new float[_keys[i].Length];
        }

        Tokenizer = new ByteBpe(model);
    }

    /// <summary>
    /// 重置解码位置和 KV 历史，开始一次独立识别。
    /// </summary>
    public void Reset() => _position = 0;
    /// <summary>
    /// 计算并缓存固定提示词前缀，减少每次识别的重复解码。
    /// </summary>
    /// <param name="tokens">按序排列的 token ID。</param>
    internal void CachePrefix(int[] tokens)
    {
        if (_cachedPrefixLength != 0)
        {
            throw new InvalidOperationException("Prefix is already cached");
        }

        Reset();
        Decode(Embed(tokens), tokens.Length, false);
        _cachedPrefixLength = tokens.Length;
    }

    /// <summary>
    /// 恢复固定前缀对应的 KV 缓存与解码位置。
    /// </summary>
    internal void RestorePrefix() => _position = _cachedPrefixLength;
    /// <summary>
    /// 根据 token ID 读取解码器嵌入向量。
    /// </summary>
    /// <param name="tokens">按序排列的 token ID。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Embed(ReadOnlySpan<int> tokens)
    {
        var x = TensorWorkspace.Allocate(tokens.Length * _width);
        for (int i = 0; i < tokens.Length; ++i)
        {
            _model["token_embd.weight"].CopyRow(tokens[i], x.AsSpan(i * _width, _width));
        }

        return x;
    }

    /// <summary>
    /// 执行一批输入嵌入的自回归解码并更新该上下文的 KV 缓存。
    /// </summary>
    /// <param name="embeddings">按时间排列的解码器输入嵌入。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="outputLogits">是否计算最后一个位置的词表输出分数。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Decode(float[] embeddings, int frames, bool outputLogits = true)
    {
        if (frames < 1 || embeddings.Length != frames * _width || _position + frames > _capacity)
        {
            throw new ArgumentException("Decoder context exhausted or invalid embeddings");
        }

        float[] x = embeddings;
        int kvWidth = _kvHeads * _headSize, qWidth = _heads * _headSize;
        if (_position + frames > _cacheFrames)
        {
            _cacheFrames = Math.Min(_capacity, Math.Max(_cacheFrames * 2, _position + frames));
            for (int i = 0; i < _layers; ++i)
            {
                Array.Resize(ref _keys[i], _cacheFrames * kvWidth);
                Array.Resize(ref _values[i], _cacheFrames * kvWidth);
            }
        }

        for (int layer = 0; layer < _layers; ++layer)
        {
            string p = $"blk.{layer}.";
            float[] h = Norm(x, _width, p + "attn_norm.weight");
            var projections = _math.LinearMany(h, frames, _qkvWeights[layer]);
            TensorWorkspace.Release(h);
            float[] q = projections[0], k = projections[1], v = projections[2];
            var previous = q;
            q = Norm(q, _headSize, p + "attn_q_norm.weight");
            TensorWorkspace.Release(previous);
            previous = k;
            k = Norm(k, _headSize, p + "attn_k_norm.weight");
            TensorWorkspace.Release(previous);
            Rope(q, frames, _heads);
            Rope(k, frames, _kvHeads);
            var kc = _keys[layer];
            var vc = _values[layer];
            long kvStart = StartMeasure();
            // 基线使用 F16 KV 缓存；此处保留完全一致的半精度舍入。
            TensorMath.RoundHalfCopy(k, kc.AsSpan(_position * kvWidth));
            TensorMath.RoundHalfCopy(v, vc.AsSpan(_position * kvWidth));
            EndMeasure($"DecHalf {frames}", kvStart);
            TensorWorkspace.Release(k);
            TensorWorkspace.Release(v);
            var attn = TensorWorkspace.Allocate(frames * qWidth, clear: true);
            long attentionStart = StartMeasure();
            _math.For(
                frames * _heads,
                index =>
                {
                    int t = index / _heads, head = index % _heads, kh = head / (_heads / _kvHeads), end = _position + t + 1;
                    Span<float> scores = stackalloc float[end];
                    int qo = t * qWidth + head * _headSize;
                    for (int s = 0; s < end; ++s)
                    {
                        scores[s] = AudioKernels.Dot(q.AsSpan(qo, _headSize), kc.AsSpan(s * kvWidth + kh * _headSize, _headSize), _math.Simd) / MathF.Sqrt(_headSize);
                    }

                    TensorMath.Softmax(scores);
                    TensorMath.AttentionValues(scores, vc, kh * _headSize, kvWidth, attn.AsSpan(qo, _headSize), _math.Simd);
                });
            EndMeasure($"DecAttention {frames}", attentionStart);
            TensorWorkspace.Release(q);
            var a = Linear(attn, frames, p + "attn_output.weight");
            TensorMath.Add(a, x);
            TensorWorkspace.Release(attn);
            if (!ReferenceEquals(x, embeddings))
            {
                TensorWorkspace.Release(x);
            }

            h = Norm(a, _width, p + "ffn_norm.weight");
            var gateUp = _math.LinearMany(h, frames, _gateUpWeights[layer]);
            var gate = gateUp[0];
            var up = gateUp[1];
            TensorWorkspace.Release(h);
            long activationStart = StartMeasure();
            TensorMath.SiluMultiply(gate, up);
            EndMeasure($"DecSwiGLU {frames}", activationStart);
            TensorWorkspace.Release(up);
            x = Linear(gate, frames, p + "ffn_down.weight");
            TensorMath.Add(x, a);
            TensorWorkspace.Release(gate);
            TensorWorkspace.Release(a);
            if (_position == 0 && layer == 0)
            {
                Trace?.Invoke("qwen_layer0", x);
            }
        }

        _position += frames;
        if (!outputLogits)
        {
            TensorWorkspace.Release(x);
            return [];
        }

        // 贪心识别只需要最后一个 token 位置的词表分数。
        var last = TensorWorkspace.Copy(x.AsSpan((frames - 1) * _width, _width));
        TensorWorkspace.Release(x);
        var lastInput = last;
        last = Norm(last, _width, "output_norm.weight");
        TensorWorkspace.Release(lastInput);
        var logits = Linear(last, 1, _model.Tensors.ContainsKey("output.weight") ? "output.weight" : "token_embd.weight");
        TensorWorkspace.Release(last);
        Trace?.Invoke("logits", logits);
        return logits;
    }

    /// <summary>
    /// 选择概率分数最大的 token，保持确定性的贪心解码。
    /// </summary>
    /// <param name="logits">模型的词表输出分数。</param>
    /// <returns>计算得到的整数结果。</returns>
    public static int Greedy(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; ++i)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary>
    /// 使用模型中命名权重对输入帧执行线性投影。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Linear(float[] x, int frames, string name) => _math.Linear(x, frames, _model[name]);
    /// <summary>
    /// 在性能跟踪开启时取得算子计时起点。
    /// </summary>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private long StartMeasure() => _math.LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
    /// <summary>
    /// 记录指定算子的累计耗时。
    /// </summary>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <param name="started">计时开始的 Stopwatch 时间戳。</param>
    private void EndMeasure(string name, long started) => _math.LinearMeasured?.Invoke(name, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
    /// <summary>
    /// 使用命名缩放参数执行 RMS 归一化。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Norm(float[] x, int width, string name)
    {
        long start = StartMeasure();
        var result = TensorMath.Norm(x, width, _model[name].Values!, null, _epsilon, true);
        EndMeasure($"DecNorm {x.Length / width}x{width}", start);
        return result;
    }

    /// <summary>
    /// 按当前序列位置原位施加半分割旋转位置编码。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="heads">注意力头数量。</param>
    private void Rope(float[] x, int frames, int heads)
    {
        long start = StartMeasure();
        int half = _headSize / 2;
        for (int t = 0; t < frames; ++t)
        {
            for (int h = 0; h < heads; ++h)
            {
                for (int i = 0; i < half; ++i)
                {
                    int position = (_position + t) * half + i;
                    float c = _ropeCos[position], s = _ropeSin[position];
                    int at = (t * heads + h) * _headSize + i;
                    float a = x[at], b = x[at + half];
                    x[at] = a * c - b * s;
                    x[at + half] = a * s + b * c;
                }
            }
        }

        EndMeasure($"DecRope {frames}", start);
    }
}
