using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// Fun-ASR 音频编码器，保留 F16 权重语义并使用托管矩阵内核。
/// </summary>
public sealed class FunEncoder
{
    private readonly Gguf _model;
    private readonly TensorMath _math;
    private readonly int _width, _heads, _kernel, _layers, _tpLayers, _adaptorLayers, _adaptorHeads;
    /// <summary>
    /// 音频编码器输出嵌入的通道数量。
    /// </summary>
    public int OutputWidth
    {
        get;
    }
    /// <summary>
    /// 可选的中间张量观察回调，仅供验证使用；开启后会产生额外快照分配。
    /// </summary>
    public Action<string, float[]>? Trace
    {
        get; set;
    }

    /// <summary>
    /// Fun-ASR 音频编码器，保留 F16 权重语义并使用托管矩阵内核。
    /// </summary>
    /// <param name="model">已加载的模型及其权重；发布给并发上下文后不得修改权重。</param>
    /// <param name="math">用于矩阵与向量运算的托管计算内核。</param>
    public FunEncoder(Gguf model, TensorMath math)
    {
        _model = model;
        _math = math;
        // 交错浮点内核使用 ARM64 NEON；x64 保留行主序以使用 AVX2/FMA，避免标量回退。
        if (math.Simd && math.Optimized && AdvSimd.Arm64.IsSupported
            && Environment.GetEnvironmentVariable("FUNASRNET_INTERLEAVE") != "0" && Environment.GetEnvironmentVariable("FUNASRNET_FLOAT_INTERLEAVE") != "0")
        {
            foreach (var w in model.Tensors.Values)
            {
                if (w.Shape.Length == 2 && w.Columns >= 128 && w.Rows >= 128)
                {
                    w.PrepareInterleaved();
                }
            }
        }

        _width = model.Int("funasr.enc.output_size", 512);
        _heads = model.Int("funasr.enc.attention_heads", 4);
        _kernel = model.Int("funasr.enc.kernel_size", 11);
        _layers = model.Int("funasr.enc.num_blocks", 50);
        _tpLayers = model.Int("funasr.enc.tp_blocks", 20);
        OutputWidth = model.Int("funasr.adp.llm_dim", 1024);
        _adaptorLayers = model.Int("funasr.adp.n_layer", 2);
        _adaptorHeads = model.Int("funasr.adp.attention_heads", 8);
    }

    /// <summary>
    /// 根据模型指定层的参数执行编码器归一化。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="prefix">模型层的名称前缀。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Norm(float[] x, string prefix)
    {
        long start = _math.LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
        var result = TensorMath.Norm(x, _model[prefix + ".weight"].Columns, _model[prefix + ".weight"].Values!, _model[prefix + ".bias"].Values!, 1e-5f);
        _math.LinearMeasured?.Invoke($"EncNorm {x.Length} {prefix}", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return result;
    }

    /// <summary>
    /// 计算多头注意力的分数、归一化概率与加权值。
    /// </summary>
    /// <param name="q">查询张量或分组量化后的激活缓冲区。</param>
    /// <param name="k">按帧和注意力头排列的键张量。</param>
    /// <param name="v">按帧排列的值张量或待归约向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="heads">注意力头数量。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Attention(float[] q, float[] k, float[] v, int frames, int width, int heads)
    {
        long start = _math.LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
        var result = _math.Attention(q, k, v, frames, width, heads);
        _math.LinearMeasured?.Invoke($"EncAttention {frames}x{width}x{heads}", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return result;
    }

    /// <summary>
    /// 根据模型指定层的权重执行编码器线性投影。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="prefix">模型层的名称前缀。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Linear(float[] x, int frames, string prefix) => _math.Linear(x, frames, _model[prefix + ".weight"], _model[prefix + ".bias"]);
    /// <summary>
    /// 将拼接后的滤波器组特征映射为解码器可接受的音频表示。
    /// </summary>
    /// <param name="fbank">按帧排列的滤波器组特征。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Encode(float[] fbank, int frames)
    {
        if (frames < 1 || frames > 1024 || fbank.Length != frames * 560)
        {
            throw new ArgumentException("Encoder accepts up to 61 seconds per segment");
        }

        var x = TensorWorkspace.Copy(fbank);
        float scale = MathF.Sqrt(_width);
        double inc = Math.Log(10000) / 279;
        for (int i = 0; i < x.Length; ++i)
        {
            x[i] *= scale;
        }

        for (int t = 0; t < frames; ++t)
        {
            for (int i = 0; i < 280; ++i)
            {
                double phase = (t + 1) * Math.Exp(-i * inc);
                x[t * 560 + i] += (float)Math.Sin(phase);
                x[t * 560 + 280 + i] += (float)Math.Cos(phase);
            }
        }

        var previous = x;
        x = Sanm(x, frames, "audio_encoder.encoders0.0.", false);
        TensorWorkspace.Release(previous);
        Trace?.Invoke("encoder0", x);
        for (int i = 0; i < _layers - 1; ++i)
        {
            previous = x;
            x = Sanm(x, frames, $"audio_encoder.encoders.{i}.", true);
            TensorWorkspace.Release(previous);
        }

        previous = x;
        x = Norm(x, "audio_encoder.after_norm");
        TensorWorkspace.Release(previous);
        Trace?.Invoke("after_norm", x);
        for (int i = 0; i < _tpLayers; ++i)
        {
            previous = x;
            x = Sanm(x, frames, $"audio_encoder.tp_encoders.{i}.", true);
            TensorWorkspace.Release(previous);
        }

        previous = x;
        x = Norm(x, "audio_encoder.tp_norm");
        TensorWorkspace.Release(previous);
        Trace?.Invoke("tp_norm", x);
        previous = x;
        x = Linear(x, frames, "audio_adaptor.linear1");
        TensorWorkspace.Release(previous);
        Relu(x);
        previous = x;
        x = Linear(x, frames, "audio_adaptor.linear2");
        TensorWorkspace.Release(previous);
        for (int i = 0; i < _adaptorLayers; ++i)
        {
            string p = $"audio_adaptor.blocks.{i}.";
            var h = Norm(x, p + "norm1");
            var q = Linear(h, frames, p + "self_attn.linear_q");
            var k = Linear(h, frames, p + "self_attn.linear_k");
            var v = Linear(h, frames, p + "self_attn.linear_v");
            TensorWorkspace.Release(h);
            var a = Attention(q, k, v, frames, OutputWidth, _adaptorHeads);
            TensorWorkspace.Release(q);
            TensorWorkspace.Release(k);
            TensorWorkspace.Release(v);
            previous = a;
            a = Linear(a, frames, p + "self_attn.linear_out");
            TensorWorkspace.Release(previous);
            TensorMath.Add(x, a);
            TensorWorkspace.Release(a);
            previous = x;
            x = FeedForward(x, frames, p);
            TensorWorkspace.Release(previous);
        }

        Trace?.Invoke("adaptor", x);
        return x;
    }

    /// <summary>
    /// 执行 SANM 层的注意力、记忆与残差计算。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="p">模型层的名称前缀。</param>
    /// <param name="residual">是否叠加残差分支。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Sanm(float[] input, int frames, string p, bool residual)
    {
        var h = Norm(input, p + "norm1");
        var qkv = Linear(h, frames, p + "self_attn.linear_q_k_v");
        TensorWorkspace.Release(h);
        var q = TensorWorkspace.Allocate(frames * _width);
        var k = TensorWorkspace.Allocate(q.Length);
        var v = TensorWorkspace.Allocate(q.Length);
        for (int t = 0; t < frames; ++t)
        {
            qkv.AsSpan(t * _width * 3, _width).CopyTo(q.AsSpan(t * _width));
            qkv.AsSpan((t * 3 + 1) * _width, _width).CopyTo(k.AsSpan(t * _width));
            qkv.AsSpan((t * 3 + 2) * _width, _width).CopyTo(v.AsSpan(t * _width));
        }

        TensorWorkspace.Release(qkv);
        long fsmnStart = _math.LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
        var fsmn = TensorWorkspace.Copy(v);
        var kernel = _model[p + "self_attn.fsmn_block.weight"].Values!;
        _math.For(
            frames,
            t =>
            {
                for (int j = 0; j < _kernel; ++j)
                {
                    int source = t + j - (_kernel - 1) / 2;
                    if (source < 0 || source >= frames)
                    {
                        continue;
                    }

                    int d = 0;
                    for (; _math.Simd && d <= _width - 4; d += 4)
                    {
                        (Vector128.LoadUnsafe(ref fsmn[t * _width + d]) + Vector128.LoadUnsafe(ref v[source * _width + d]) * Vector128.LoadUnsafe(ref kernel[j * _width + d])).StoreUnsafe(ref fsmn[t * _width + d]);
                    }

                    for (; d < _width; ++d)
                    {
                        fsmn[t * _width + d] += v[source * _width + d] * kernel[j * _width + d];
                    }
                }
            });
        _math.LinearMeasured?.Invoke($"EncFsmn {frames}x{_width} {p}", System.Diagnostics.Stopwatch.GetElapsedTime(fsmnStart).TotalMilliseconds);
        var a = Attention(q, k, v, frames, _width, _heads);
        TensorWorkspace.Release(q);
        TensorWorkspace.Release(k);
        TensorWorkspace.Release(v);
        var x = Linear(a, frames, p + "self_attn.linear_out");
        TensorMath.Add(x, fsmn);
        TensorWorkspace.Release(a);
        TensorWorkspace.Release(fsmn);
        if (residual)
        {
            TensorMath.Add(x, input);
        }

        var result = FeedForward(x, frames, p);
        TensorWorkspace.Release(x);
        return result;
    }

    /// <summary>
    /// 执行编码器前馈网络与残差更新。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="p">模型层的名称前缀。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] FeedForward(float[] x, int frames, string p)
    {
        var h = Norm(x, p + "norm2");
        var previous = h;
        h = Linear(h, frames, p + "feed_forward.w_1");
        TensorWorkspace.Release(previous);
        Relu(h);
        previous = h;
        h = Linear(h, frames, p + "feed_forward.w_2");
        TensorWorkspace.Release(previous);
        TensorMath.Add(h, x);
        return h;
    }

    /// <summary>
    /// 原位将负数截断为零，实施 ReLU 激活。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    private static void Relu(float[] x)
    {
        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i <= x.Length - 4; i += 4)
            {
                Vector128.Max(Vector128<float>.Zero, Vector128.LoadUnsafe(ref x[i])).StoreUnsafe(ref x[i]);
            }
        }

        for (; i < x.Length; ++i)
        {
            x[i] = Math.Max(0, x[i]);
        }
    }
}
