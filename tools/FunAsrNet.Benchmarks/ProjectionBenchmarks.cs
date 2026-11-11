using System.Diagnostics;
using FunAsrNet.Vulkan;

namespace FunAsrNet.Benchmarks;

/// <summary>测量独立投影的完整调用，包含 GPU 上传、同步和回读，保留所有样本。</summary>
internal static class ProjectionBenchmarks
{
    /// <summary>使用固定数据验证投影数值并输出预热与正式计时。</summary>
    /// <param name="gpuEnabled">是否启用实际 Vulkan 设备。</param>
    /// <param name="emit">接收每次测量的 JSON 对象。</param>
    internal static void Run(bool gpuEnabled, Action<object> emit)
    {
        using var gpu = gpuEnabled ? new VulkanFloatBackend() { MinimumOperations = 0 } : null;
        var math = new TensorMath(4) { FloatBackend = gpu };
        using var affinity = math.PinCaller();
        foreach (bool quantized in new[] { false, true })
        {
            foreach (var shape in new[] { (M: 1, K: 1024, N: 4096), (M: 1, K: 1024, N: 151936), (M: 3, K: 512, N: 512), (M: 73, K: 1024, N: 3072), (M: 400, K: 512, N: 1536), (M: 400, K: 2048, N: 512) })
            {
                var random = new Random(20261111);
                var input = new float[shape.M * shape.K];
                var values = new float[shape.N * shape.K];
                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = (float)(random.NextDouble() - 0.5);
                }
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(random.NextDouble() - 0.5);
                }
                var weight = new Weight("projection-benchmark", [shape.K, shape.N]);
                if (quantized)
                {
                    weight.Quantized = new sbyte[values.Length];
                    weight.Scales = new float[values.Length / 32];
                    TensorMath.Quantize(values, weight.Quantized, weight.Scales);
                }
                else
                {
                    weight.Values = values;
                }
                var q = new sbyte[input.Length];
                var scales = new float[input.Length / 32];
                if (quantized)
                {
                    TensorMath.Quantize(input, q, scales);
                }
                double error = 0;
                for (int iteration = -3; iteration < 15; iteration++)
                {
                    long allocated = GC.GetTotalAllocatedBytes(false);
                    long start = Stopwatch.GetTimestamp();
                    float[] result = math.Linear(input, shape.M, weight);
                    double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
                    allocated = GC.GetTotalAllocatedBytes(false) - allocated;
                    if (iteration == -3)
                    {
                        for (int sample = 0; sample < 64; sample++)
                        {
                            int time = sample * 7919 % shape.M, row = sample * 3571 % shape.N;
                            double expected = 0;
                            for (int k = 0; k < shape.K; k += quantized ? 32 : 1)
                            {
                                if (quantized)
                                {
                                    int dot = 0;
                                    for (int j = 0; j < 32; j++)
                                    {
                                        dot += weight.Quantized![row * shape.K + k + j] * q[time * shape.K + k + j];
                                    }
                                    expected = (float)expected + dot * weight.Scales![row * (shape.K / 32) + k / 32] * scales[time * (shape.K / 32) + k / 32];
                                }
                                else
                                {
                                    expected += (double)values[row * shape.K + k] * input[time * shape.K + k];
                                }
                            }
                            double difference = Math.Abs(result[time * shape.N + row] - expected) / Math.Max(1, Math.Abs(expected));
                            error = Math.Max(error, difference);
                            if (!double.IsFinite(difference) || difference > 0.0002)
                            {
                                throw new InvalidDataException("投影与独立参考不一致。");
                            }
                        }
                    }
                    emit(new
                    {
                        kind = "projection",
                        engine = gpuEnabled ? "vulkan" : "cpu",
                        precision = quantized ? "Q8_0" : "FP32",
                        shape.M,
                        shape.K,
                        shape.N,
                        iteration,
                        warmup = iteration < 0,
                        seconds,
                        allocatedBytes = allocated,
                        maxRelativeError = error
                    });
                    GC.KeepAlive(result);
                }
            }
        }
        if (gpu is not null)
        {
            emit(gpu.Statistics());
        }
    }
}
