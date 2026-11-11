using FunAsrNet;
using FunAsrNet.Vulkan;

// 此项目只引用实际 NuGet 包，确保公开 API 与传递依赖能够安装和运行。
byte[] encoded = Enumerable.Range(0, 256).Select(value => (byte)value).ToArray();
short[] vector = new short[encoded.Length];
short[] scalar = new short[encoded.Length];
foreach (bool alaw in new[] { false, true })
{
    G711.Decode(encoded, vector, alaw);
    G711.Decode(encoded, scalar, alaw, simd: false);
    for (int i = 0; i < encoded.Length; i++)
    {
        if (vector[i] != scalar[i] || vector[i] != G711.DecodeSample(encoded[i], alaw))
        {
            throw new InvalidOperationException($"Package G.711 mismatch at {i}, A-law={alaw}.");
        }
    }
}

float[] samples = Enumerable.Range(0, 160).Select(i => MathF.Sin(i * 0.1f) * 0.5f).ToArray();
float[] resampled = AudioFile.Resample(samples, 8000, 16000);
if (resampled.Length != 320 || resampled.Any(value => !float.IsFinite(value)))
{
    throw new InvalidOperationException("Package resampling failed.");
}
if (!typeof(IAttentionComputeBackend).IsAssignableFrom(typeof(VulkanFloatBackend)))
{
    throw new InvalidOperationException("The GPU package does not expose the attention backend.");
}
if (typeof(FunAsr).Assembly.GetName().Version != typeof(VulkanFloatBackend).Assembly.GetName().Version)
{
    throw new InvalidOperationException("Core and Vulkan package assembly versions differ.");
}
Console.WriteLine($"PASS: actual NuGet packages, G.711, resampling, GPU API and dependency loading; runtime {Environment.Version}.");
