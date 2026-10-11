using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

Console.OutputEncoding = System.Text.Encoding.UTF8;

#if NATIVE_GPU_REFERENCE
if (args.Length == 4 && args[0] == "prepare-asr-reference")
{
    FunAsrNet.NativeGpuReference.NativeAsrFixture.Prepare(args[1], args[2], args[3]);
    return;
}
#endif

// <summary>
// 读取板端温度，未提供该接口的平台返回空值。
// </summary>
// <returns>摄氏温度，或不可获取时的空值。</returns>
double? Temperature()
{
    const string path = "/sys/class/thermal/thermal_zone0/temp";
    return File.Exists(path) ? double.Parse(File.ReadAllText(path), CultureInfo.InvariantCulture) / 1000 : null;
}

// <summary>
// 读取两个大核簇的瞬时频率；查询开销不计入模型调用。
// </summary>
// <returns>以 kHz 为单位的频率数组，不可获取时为空。</returns>
int[] Frequencies() => new[] { 4, 6 }
    .Select(cluster => $"/sys/devices/system/cpu/cpufreq/policy{cluster}/scaling_cur_freq")
    .Where(File.Exists)
    .Select(path => int.Parse(File.ReadAllText(path), CultureInfo.InvariantCulture)).ToArray();

if (args.Length < 5)
{
    throw new ArgumentException("用法：模型文件 音素ID文件 huayan或aishell3 次数 输出波形文件 [--provider cpu|cuda] [--profile-prefix 路径] [--affinity 0x55]");
}

if (OperatingSystem.IsWindows() && Array.IndexOf(args, "--affinity") is int affinityIndex and >= 0)
{
    Process.GetCurrentProcess().ProcessorAffinity = (nint)Convert.ToInt64(args[affinityIndex + 1].Replace("0x", "", StringComparison.OrdinalIgnoreCase), 16);
}

bool aishell = args[2] == "aishell3";
#if NATIVE_GPU_REFERENCE
string provider = "cuda";
#else
string provider = "cpu";
#endif
int providerIndex = Array.IndexOf(args, "--provider");
if (providerIndex >= 0)
{
    provider = args[providerIndex + 1];
}
string engine = provider switch
{
    "cpu" => "onnxruntime-cpu",
    "cuda" => "onnxruntime-cuda",
    _ => throw new ArgumentException("原生对照仅支持 cpu 或 cuda。")
};
int repeats = int.Parse(args[3], CultureInfo.InvariantCulture);
long[] ids = MemoryMarshal.Cast<byte, long>(File.ReadAllBytes(args[1])).ToArray();
using var options = new SessionOptions
{
    IntraOpNumThreads = 4,
    InterOpNumThreads = 1,
    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
};
if (provider == "cuda")
{
    using var cuda = new OrtCUDAProviderOptions();
    // 保持 FP32 激活精度，与 C# Vulkan 一致；允许原生图保留必要的 CPU 回退节点。
    cuda.UpdateOptions(new Dictionary<string, string> { ["device_id"] = "0", ["use_tf32"] = "0" });
    options.AppendExecutionProvider_CUDA(cuda);
}
int profileIndex = Array.IndexOf(args, "--profile-prefix");
if (profileIndex >= 0)
{
    options.ProfileOutputPathPrefix = args[profileIndex + 1];
    options.EnableProfiling = true;
}
var loading = Stopwatch.StartNew();
using var session = new InferenceSession(args[0], options);
Console.WriteLine(JsonSerializer.Serialize(new { kind = "load", engine, provider, availableProviders = OrtEnv.Instance().GetAvailableProviders(), useTf32 = false, version = OrtEnv.Instance().GetVersionString(), seconds = loading.Elapsed.TotalSeconds }));
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor(aishell ? "tokens" : "input", new DenseTensor<long>(ids, [1, ids.Length])),
    NamedOnnxValue.CreateFromTensor(aishell ? "tokens_lens" : "input_lengths", new DenseTensor<long>(new long[] { ids.Length }, new int[] { 1 }))
};
if (aishell)
{
    inputs.Add(NamedOnnxValue.CreateFromTensor("noise_scale", new DenseTensor<float>(new float[] { 0 }, new int[] { 1 })));
    inputs.Add(NamedOnnxValue.CreateFromTensor("alpha", new DenseTensor<float>(new float[] { 1 }, new int[] { 1 })));
    inputs.Add(NamedOnnxValue.CreateFromTensor("noise_scale_dur", new DenseTensor<float>(new float[] { 0 }, new int[] { 1 })));
    inputs.Add(NamedOnnxValue.CreateFromTensor("speaker", new DenseTensor<long>(new long[] { 50 }, new int[] { 1 })));
}
else
{
    inputs.Add(NamedOnnxValue.CreateFromTensor("scales", new DenseTensor<float>(new float[] { 0, 1, 0 }, new int[] { 3 })));
}

for (int iteration = 0; iteration <= repeats; iteration++)
{
    var cooldown = Stopwatch.StartNew();
    while (Temperature() is > 60)
    {
        if (cooldown.Elapsed.TotalMinutes > 5)
        {
            throw new TimeoutException("五分钟内未能回落到 60°C，停止原生基准。");
        }
        await Task.Delay(2000);
    }
    double cooldownSeconds = cooldown.Elapsed.TotalSeconds;
    double? temperatureBefore = Temperature();
    int[] frequencyBefore = Frequencies();
    var watch = Stopwatch.StartNew();
    using var outputs = session.Run(inputs);
    double seconds = watch.Elapsed.TotalSeconds;
    var audio = outputs.First().AsTensor<float>();
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        kind = "tts",
        engine,
        iteration,
        warmup = iteration == 0,
        seconds,
        samples = audio.Length,
        audioSeconds = audio.Length / (aishell ? 8000.0 : 22050.0),
        temperatureBefore,
        temperatureAfter = Temperature(),
        frequencyBefore,
        frequencyAfter = Frequencies(),
        cooldownSeconds,
        peakRssBytes = Process.GetCurrentProcess().PeakWorkingSet64
    }));
    if (iteration == repeats)
    {
        File.WriteAllBytes(args[4], MemoryMarshal.AsBytes(audio.ToArray().AsSpan()).ToArray());
    }
}
if (profileIndex >= 0)
{
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        kind = "profile-file",
        path = session.EndProfiling()
    }));
}
