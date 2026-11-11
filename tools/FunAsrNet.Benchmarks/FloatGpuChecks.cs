using FunAsrNet.Vulkan;

namespace FunAsrNet.Benchmarks;

/// <summary>验证原精度分组、专用矩阵内核及 GPU 缓存重新上传的正确性。</summary>
internal static class FloatGpuChecks
{
    /// <summary>与通用 FP32 着色器及独立 CPU 参考比较，包含尾部和缓存淘汰。</summary>
    /// <param name="gpu">调用方拥有的真实 GPU 后端，最低工作量设置为零。</param>
    /// <returns>完成的数值断言数量。</returns>
    internal static int Run(VulkanFloatBackend gpu)
    {
        string? option = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_SPECIALIZED_FLOAT");
        VulkanFloatBackend reference;
        try
        {
            Environment.SetEnvironmentVariable("FUNASRNET_VULKAN_SPECIALIZED_FLOAT", "0");
            reference = new VulkanFloatBackend(4L * 1024 * 1024) { MinimumOperations = 0 };
        }
        finally
        {
            Environment.SetEnvironmentVariable("FUNASRNET_VULKAN_SPECIALIZED_FLOAT", option);
        }
        using var lifetime = reference;
        var math = new TensorMath(4) { FloatBackend = gpu };
        var original = new TensorMath(4) { FloatBackend = reference };
        var random = new Random(211109);
        int checks = 0;
        foreach (int frames in new[] { 1, 3, 17, 33 })
        {
            foreach (int width in new[] { 19, 65, 513, 1024, 1057, 3073 })
            {
                var input = Enumerable.Range(0, width * frames).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                var weights = new[] { 7, 65, 129 }.Select(rows => new Weight("float-gpu-group", [width, rows])
                {
                    Values = Enumerable.Range(0, width * rows).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray()
                }).ToArray();
                float[][] grouped = math.LinearMany(input, frames, weights);
                for (int which = 0; which < weights.Length; which++)
                {
                    float[] separate = math.Linear(input, frames, weights[which]);
                    float[] prior = original.Linear(input, frames, weights[which]);
                    for (int i = 0; i < separate.Length; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(grouped[which][i]) != BitConverter.SingleToInt32Bits(separate[i]))
                        {
                            throw new InvalidDataException("FP32 合并与独立 GPU 投影的位结果不同。");
                        }
                        if (!float.IsFinite(separate[i]) || Math.Abs(separate[i] - prior[i]) > 0.0002 * Math.Max(1, Math.Abs(prior[i])))
                        {
                            throw new InvalidDataException("FP32 专用内核超过原始精度容差。");
                        }
                        if (frames >= 16 && BitConverter.SingleToInt32Bits(separate[i]) != BitConverter.SingleToInt32Bits(prior[i]))
                        {
                            throw new InvalidDataException("FP32 分块矩阵改变了逐列 FMA 顺序。");
                        }
                        checks += 3;
                    }
                }
                var pressure = new Weight("eviction-pressure", [1024, 896]) { Values = new float[1024 * 896] };
                math.Linear(new float[1024], 1, pressure);
                var repeated = math.LinearMany(input, frames, weights);
                for (int which = 0; which < weights.Length; which++)
                {
                    for (int i = 0; i < repeated[which].Length; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(repeated[which][i]) != BitConverter.SingleToInt32Bits(grouped[which][i]))
                        {
                            throw new InvalidDataException("FP32 分组权重淘汰后重新上传结果不同。");
                        }
                        checks++;
                    }
                }
            }
        }
        return checks;
    }
}
