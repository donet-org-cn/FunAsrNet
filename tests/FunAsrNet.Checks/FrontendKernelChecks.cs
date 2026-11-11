using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text.Json;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>使用优化前算法独立检查特征提取、半分割旋转和确定性词表扫描。</summary>
internal static class FrontendKernelChecks
{
    /// <summary>覆盖静音、脉冲、随机音频、尾帧、共享实例、向量尾部和非有限分数。</summary>
    /// <param name="check">接收检查条件和中文说明的断言回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var random = new Random(11112026);
        var reference = new ScalarFilterBank();
        var filter = new FilterBank();
        foreach (int length in new[] { 400, 401, 559, 560, 561, 1599, 16000, 96000 })
        {
            for (int kind = 0; kind < 5; ++kind)
            {
                var audio = new float[length + 2];
                for (int i = 1; i <= length; ++i)
                {
                    audio[i] = kind switch
                    {
                        0 => 0,
                        1 => i == 200 ? 1 : 0,
                        2 => MathF.Sin(i * 0.017f) * 0.5f,
                        3 => (float)(random.NextDouble() * 2 - 1),
                        _ => (float)(random.NextDouble() - 0.5) * 1e-15f + 0.125f
                    };
                }
                var input = audio.AsSpan(1, length);
                var expected = reference.Compute80(input, out int expectedFrames);
                var actual = filter.Compute80(input, out int frames);
                check(frames == expectedFrames, "特征帧数与标量参考一致");
                check(MemoryMarshal.AsBytes(actual.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())), "80 维特征逐位一致");
                expected = reference.Compute(input, out expectedFrames);
                actual = filter.Compute(input, out frames);
                check(frames == expectedFrames, "LFR 帧数与标量参考一致");
                check(MemoryMarshal.AsBytes(actual.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())), "LFR 特征逐位一致");
            }
        }
        var sharedInput = Enumerable.Range(0, 16000).Select(i => MathF.Sin(i * 0.1f)).ToArray();
        var sharedExpected = reference.Compute80(sharedInput, out _);
        var results = new float[8][];
        Parallel.For(0, results.Length, i => results[i] = filter.Compute80(sharedInput, out _));
        foreach (var result in results)
        {
            check(result.SequenceEqual(sharedExpected), "共享特征实例的并发计算互不污染");
        }
        foreach (int half in Enumerable.Range(0, 34).Concat(new[] { 63, 64, 65, 128 }))
        {
            var cos = Enumerable.Range(0, half + 2).Select(i => MathF.Cos(i)).ToArray();
            var sin = Enumerable.Range(0, half + 2).Select(i => MathF.Sin(i)).ToArray();
            var source = Enumerable.Range(0, half * 2 + 2).Select(_ => (float)(random.NextDouble() * 20 - 10)).ToArray();
            for (int i = 1; i <= 2 * half; i += 7)
            {
                source[i] = i % 3 == 0 ? -0f : i % 3 == 1 ? float.Epsilon : float.MaxValue;
            }
            var expected = (float[])source.Clone();
            ReferenceRotate(expected.AsSpan(1, half * 2), cos.AsSpan(1, half), sin.AsSpan(1, half));
            foreach (bool simd in new[] { false, true })
            {
                var actual = (float[])source.Clone();
                TensorMath.RotateHalf(actual.AsSpan(1, half * 2), cos.AsSpan(1, half), sin.AsSpan(1, half), simd);
                check(MemoryMarshal.AsBytes(actual.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(expected.AsSpan())), "RoPE 逐位一致且保留切片两端");
            }
        }
        foreach (int length in Enumerable.Range(0, 66).Concat(new[] { 4097, 151936 }))
        {
            for (int kind = 0; kind < 8; ++kind)
            {
                var values = Enumerable.Range(0, length).Select(i => kind switch
                {
                    0 => (float)(random.NextDouble() - 0.5),
                    1 => (float)i,
                    2 => (float)-i,
                    3 => float.NegativeInfinity,
                    4 => i % 3 == 0 ? float.NaN : i % 3 == 1 ? float.PositiveInfinity : 0,
                    5 => i == 0 ? -1 : i % 7 == 0 ? 1 : float.NaN,
                    6 => i % 2 == 0 ? -0f : 0f,
                    _ => i % 3 == 0 ? float.NegativeInfinity : float.NaN
                }).ToArray();
                check(QwenDecoder.Greedy(values) == ReferenceGreedy(values), "Greedy 保留首个最大值、NaN 和无穷规则");
            }
        }
    }

    /// <summary>独立执行旧半分割旋转，不调用待验证的 SIMD 或标量实现。</summary>
    /// <param name="x">单头输入，长度为系数的两倍，原位修改。</param>
    /// <param name="cosine">当前序列位置的余弦系数。</param>
    /// <param name="sine">当前序列位置的正弦系数，与余弦长度相同。</param>
    private static void ReferenceRotate(Span<float> x, ReadOnlySpan<float> cosine, ReadOnlySpan<float> sine)
    {
        int half = cosine.Length;
        for (int i = 0; i < half; ++i)
        {
            float a = x[i], b = x[i + half], c = cosine[i], s = sine[i];
            x[i] = a * c - b * s;
            x[i + half] = a * s + b * c;
        }
    }

    /// <summary>独立保留旧词表扫描规则，严格大于时才更新索引。</summary>
    /// <param name="logits">词表分数，空数组按原接口返回零。</param>
    /// <returns>首个最大值索引；首项为 NaN 时返回零。</returns>
    private static int ReferenceGreedy(float[] logits)
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

    /// <summary>运行不依赖模型的交替微基准，输出全部逐轮时间与当前线程分配量。</summary>
    internal static void Benchmark()
    {
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            kind = "configuration",
            runtime = Environment.Version.ToString(),
            architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            affinity = OperatingSystem.IsWindows() ? Process.GetCurrentProcess().ProcessorAffinity.ToInt64() : (long?)null,
            vector256 = Vector256.IsHardwareAccelerated,
            vector128 = Vector128.IsHardwareAccelerated,
            rounds = 15,
            warmupRounds = 3,
            scope = "synthetic kernels; excludes model inference"
        }));
        var random = new Random(11112026);
        var reference = new ScalarFilterBank();
        var filter = new FilterBank();
        foreach (int seconds in new[] { 1, 6, 28 })
        {
            var audio = Enumerable.Range(0, seconds * 16000).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
            Measure($"fbank-{seconds}s", () => GC.KeepAlive(reference.Compute80(audio, out _)), () => GC.KeepAlive(filter.Compute80(audio, out _)));
        }
        var cos = Enumerable.Range(0, 64).Select(i => MathF.Cos(i)).ToArray();
        var sin = Enumerable.Range(0, 64).Select(i => MathF.Sin(i)).ToArray();
        var source = Enumerable.Range(0, 512 * 128).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        var actual = (float[])source.Clone();
        Measure("rope-512heads-x128", () => RotateBatch(source, cos, sin, false), () => RotateBatch(actual, cos, sin, true));
        var logits = Enumerable.Range(0, 151936).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        Measure("greedy-151936", () => ScanBatch(logits, false), () => ScanBatch(logits, true));
    }

    /// <summary>重复执行旋转以降低时钟采样开销，向量和参考处理相同规模。</summary>
    /// <param name="data">512 个长度为 128 的注意力头，原位修改。</param>
    /// <param name="cos">长度为 64 的余弦系数。</param>
    /// <param name="sin">长度为 64 的正弦系数。</param>
    /// <param name="optimized">为真时调用优化实现，否则调用旧算法。</param>
    private static void RotateBatch(float[] data, float[] cos, float[] sin, bool optimized)
    {
        for (int repeat = 0; repeat < 64; ++repeat)
        {
            for (int head = 0; head < 512; ++head)
            {
                if (optimized)
                {
                    TensorMath.RotateHalf(data.AsSpan(head * 128, 128), cos, sin, true);
                }
                else
                {
                    ReferenceRotate(data.AsSpan(head * 128, 128), cos, sin);
                }
            }
        }
    }

    /// <summary>重复扫描同一词表，保存结果以避免消除计算。</summary>
    /// <param name="logits">长度为 151936 的合成词表分数。</param>
    /// <param name="optimized">是否调用优化后的词表查找。</param>
    private static void ScanBatch(float[] logits, bool optimized)
    {
        int sum = 0;
        for (int i = 0; i < 64; ++i)
        {
            sum += optimized ? QwenDecoder.Greedy(logits) : ReferenceGreedy(logits);
        }
        GC.KeepAlive(sum);
    }

    /// <summary>预热后交替调用两版内核，保留全部慢样本，不将计时输出算入测量。</summary>
    /// <param name="name">包含输入规模的内核名称。</param>
    /// <param name="reference">优化前的独立算法。</param>
    /// <param name="optimized">当前优化算法。</param>
    private static void Measure(string name, Action reference, Action optimized)
    {
        for (int round = -3; round < 15; ++round)
        {
            foreach (bool current in round % 2 == 0 ? new[] { false, true } : new[] { true, false })
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                long started = Stopwatch.GetTimestamp();
                (current ? optimized : reference)();
                double seconds = Stopwatch.GetElapsedTime(started).TotalSeconds;
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                Console.WriteLine(JsonSerializer.Serialize(new
                {
                    kind = "measurement",
                    name,
                    engine = current ? "optimized" : "reference",
                    round,
                    warmup = round < 0,
                    seconds,
                    allocatedBytes = allocated
                }));
            }
        }
    }
}
