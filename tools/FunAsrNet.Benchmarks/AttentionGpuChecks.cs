using System.Diagnostics;
using FunAsrNet;
using FunAsrNet.Vulkan;

namespace FunAsrNet.Benchmarks;

/// <summary>用独立双精度参考验证 GPU 注意力、边界形状与不支持时的回退。</summary>
internal static class AttentionGpuChecks
{
    /// <summary>检查尾帧、尾通道、多个头、数值范围以及重复上传。</summary>
    /// <param name="gpu">已创建的同步后端，测试时最小运算阈值设为零。</param>
    /// <param name="emit">记录误差、形状和时间的输出回调。</param>
    /// <returns>通过的逐元素与契约断言数量，不支持设备返回零。</returns>
    internal static int Run(VulkanFloatBackend gpu, Action<object> emit)
    {
        var random = new Random(170719);
        int checks = 0;
        foreach (var (frames, width, heads) in new[] { (1, 7, 1), (2, 7, 1), (3, 256, 1), (17, 96, 3), (100, 512, 4), (129, 1024, 8), (400, 512, 4), (257, 256, 1), (1024, 16, 2), (1025, 16, 2), (4096, 8, 1) })
        {
            float[] q = Enumerable.Range(0, frames * width).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] k = Enumerable.Range(0, q.Length).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] v = Enumerable.Range(0, q.Length).Select(_ => (float)(random.NextDouble() * 8 - 4)).ToArray();
            if (frames == 17)
            {
                for (int i = 0; i < q.Length; i++)
                {
                    q[i] *= 16;
                    k[i] *= 16;
                }
            }
            var actual = new float[q.Length];
            var watch = Stopwatch.StartNew();
            if (!gpu.TryAttention(q, k, v, frames, width, heads, actual))
            {
                if (checks == 0)
                {
                    emit(new
                    {
                        kind = "attention-check",
                        supported = false
                    });
                    return 0;
                }
                throw new InvalidDataException("GPU 注意力在支持范围内意外回退。");
            }
            double gpuSeconds = watch.Elapsed.TotalSeconds;
            var expected = Reference(q, k, v, frames, width, heads);
            double maxError = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                double error = Math.Abs(actual[i] - expected[i]) / Math.Max(1, Math.Abs(expected[i]));
                maxError = Math.Max(maxError, error);
                if (!float.IsFinite(actual[i]) || error > 0.00005)
                {
                    throw new InvalidDataException($"GPU 注意力误差过大：{frames}x{width}/{heads}, {i}, {expected[i]} / {actual[i]}");
                }
                checks++;
            }
            emit(new
            {
                kind = "attention-check",
                frames,
                width,
                heads,
                maxError,
                gpuSeconds
            });
            Array.Clear(q);
            Array.Clear(k);
            if (!gpu.TryAttention(q, k, v, frames, width, heads, actual))
            {
                throw new InvalidDataException("GPU 注意力重复上传失败。");
            }
            for (int c = 0; c < width; c++)
            {
                double mean = 0;
                for (int t = 0; t < frames; t++)
                {
                    mean += v[t * width + c] / (double)frames;
                }
                for (int t = 0; t < frames; t++)
                {
                    if (Math.Abs(actual[t * width + c] - mean) > 0.00005 * Math.Max(1, Math.Abs(mean)))
                    {
                        throw new InvalidDataException("GPU 注意力复用了过期输入或帧步长错误。");
                    }
                    checks++;
                }
            }
        }
        var sentinel = new float[] { 123 };
        if (gpu.TryAttention([], [], [], 4097, 8, 1, sentinel) || sentinel[0] != 123
            || gpu.TryAttention([], [], [], 1, 257, 1, sentinel) || sentinel[0] != 123)
        {
            throw new InvalidDataException("GPU 注意力不支持的形状修改了输出。");
        }
        bool rejected = false;
        try
        {
            gpu.TryAttention([], [], [], 2, 8, 1, sentinel);
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        if (!rejected)
        {
            throw new InvalidDataException("GPU 注意力未拒绝不完整输入。");
        }
        return checks + 3;
    }

    /// <summary>逐查询用双精度点积、指数、概率和加权值建立独立参考。</summary>
    /// <param name="q">只读查询。</param>
    /// <param name="k">只读键。</param>
    /// <param name="v">只读值。</param>
    /// <param name="frames">完整时间帧数。</param>
    /// <param name="width">每帧总通道数。</param>
    /// <param name="heads">注意力头数。</param>
    /// <returns>与输入布局相同的双精度参考输出。</returns>
    private static double[] Reference(float[] q, float[] k, float[] v, int frames, int width, int heads)
    {
        var output = new double[q.Length];
        int size = width / heads;
        Parallel.For(0, frames * heads, new ParallelOptions { MaxDegreeOfParallelism = 4 }, index =>
        {
            int time = index / heads, head = index % heads, start = time * width + head * size;
            var scores = new double[frames];
            double maximum = double.NegativeInfinity;
            for (int source = 0; source < frames; source++)
            {
                double sum = 0;
                for (int c = 0; c < size; c++)
                {
                    sum += (double)q[start + c] * k[source * width + head * size + c];
                }
                scores[source] = sum / Math.Sqrt(size);
                maximum = Math.Max(maximum, scores[source]);
            }
            double denominator = 0;
            for (int source = 0; source < frames; source++)
            {
                scores[source] = Math.Exp(scores[source] - maximum);
                denominator += scores[source];
            }
            for (int source = 0; source < frames; source++)
            {
                double probability = scores[source] / denominator;
                for (int c = 0; c < size; c++)
                {
                    output[start + c] += probability * v[source * width + head * size + c];
                }
            }
        });
        return output;
    }
}
