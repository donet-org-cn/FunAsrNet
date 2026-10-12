using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using FunAsrNet;
using FunAsrNet.Benchmarks;
using FunAsrNet.Vulkan;

Console.OutputEncoding = Encoding.UTF8;
var jsonOptions = new JsonSerializerOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
if (args.Length < 1)
{
    throw new ArgumentException("用法：prepare 模型目录 词典 样例目录 输出目录 | prepare-tts 模型目录 词典 huayan或aishell3 文本 ID文件 | asr 模型目录 PCM文件 次数 [--vulkan] | http-asr 服务地址 WAV文件 次数 | tts 模型目录 ID文件 huayan或aishell3 次数 输出文件 [--threads 4] [--profile] [--vulkan] | compare 参考波形 实际波形 | gpu-info | gpu-check | vulkan | npu；Windows 可添加 --affinity 0x55。");
}

if (OperatingSystem.IsWindows() && Array.IndexOf(args, "--affinity") is int affinityIndex and >= 0)
{
    Process.GetCurrentProcess().ProcessorAffinity = (nint)Convert.ToInt64(args[affinityIndex + 1].Replace("0x", "", StringComparison.OrdinalIgnoreCase), 16);
}

// <summary>
// 输出一条 JSON 记录，保留全部测量结果，不筛掉慢样本。
// </summary>
// <param name="record">要写入标准输出的测量或清单对象。</param>
void Emit(object record) => Console.WriteLine(JsonSerializer.Serialize(record, jsonOptions));

// <summary>
// 读取板端 CPU 温度；不提供该接口的平台返回空值。
// </summary>
// <returns>摄氏温度，或不可获取时的空值。</returns>
double? Temperature()
{
    const string path = "/sys/class/thermal/thermal_zone0/temp";
    return File.Exists(path) ? double.Parse(File.ReadAllText(path), CultureInfo.InvariantCulture) / 1000 : null;
}

// <summary>
// 在计时区外等待温度不高于 60 摄氏度，超时则中止而非混入不同条件。
// </summary>
// <returns>实际用于降温等待的秒数。</returns>
async Task<double> CoolAsync()
{
    var watch = Stopwatch.StartNew();
    while (Temperature() is > 60)
    {
        if (watch.Elapsed.TotalMinutes > 5)
        {
            throw new TimeoutException("五分钟内未能回落到 60°C，停止本轮基准。");
        }

        await Task.Delay(2000);
    }

    return watch.Elapsed.TotalSeconds;
}

// <summary>
// 读取两个大核簇的瞬时频率；查询开销不计入模型调用。
// </summary>
// <returns>以 kHz 为单位的频率数组，不可获取时为空。</returns>
int[] Frequencies() => new[] { 4, 6 }
    .Select(cluster => $"/sys/devices/system/cpu/cpufreq/policy{cluster}/scaling_cur_freq")
    .Where(File.Exists)
    .Select(path => int.Parse(File.ReadAllText(path), CultureInfo.InvariantCulture)).ToArray();

if (args[0] == "gpu-check")
{
    using var gpu = new VulkanFloatBackend(4L * 1024 * 1024) { MinimumOperations = 0 };
    var random = new Random(6103060);
    int checks = 0;
    foreach (int width in new[] { 32, 96, 1024 })
    {
        foreach (int frames in new[] { 1, 3, 17 })
        {
            int rows = 19;
            var weight = new Weight("gpu-q8-test", [width, rows]) { Quantized = new sbyte[width * rows], Scales = new float[width * rows / 32] };
            var source = Enumerable.Range(0, width * rows).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
            TensorMath.Quantize(source, weight.Quantized, weight.Scales);
            foreach (bool interleaved in new[] { false, true })
            {
                if (interleaved)
                {
                    weight.PrepareInterleaved();
                }

                float[] input = Enumerable.Range(0, frames * width).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                float[] expected = new TensorMath(1, false).Linear(input, frames, weight);
                float[] actual = new TensorMath(4) { FloatBackend = gpu }.Linear(input, frames, weight);
                for (int index = 0; index < actual.Length; index++)
                {
                    if (!float.IsFinite(actual[index]) || Math.Abs(expected[index] - actual[index]) > 0.0002 * Math.Max(1, Math.Abs(expected[index])))
                    {
                        throw new InvalidDataException($"Q8 GPU 参考误差：{index} {expected[index]} / {actual[index]}");
                    }

                    checks++;
                }
            }
        }
    }

    foreach (var shape in new[] { (M: 1, K: 19, N: 7), (M: 33, K: 65, N: 35), (M: 96, K: 512, N: 512) })
    {
        float[] input = Enumerable.Range(0, shape.M * shape.K).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        var weight = new Weight("gpu-test", [shape.K, shape.N]) { Values = Enumerable.Range(0, shape.N * shape.K).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray() };
        var bias = new Weight("bias", [shape.N]) { Values = Enumerable.Range(0, shape.N).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray() };
        float[] expected = new TensorMath(1, false).Linear(input, shape.M, weight, bias);
        float[] actual = new TensorMath(4) { FloatBackend = gpu }.Linear(input, shape.M, weight, bias);
        for (int index = 0; index < actual.Length; index++)
        {
            if (!float.IsFinite(actual[index]) || Math.Abs(expected[index] - actual[index]) > 0.0002 * Math.Max(1, Math.Abs(expected[index])))
            {
                throw new InvalidDataException($"GPU 矩阵参考误差：{index} {expected[index]} / {actual[index]}");
            }

            checks++;
        }

        weight.PrepareInterleaved();
        expected = new TensorMath(1, false).Linear(input, shape.M, weight, bias);
        actual = new TensorMath(4) { FloatBackend = gpu }.Linear(input, shape.M, weight, bias);
        for (int index = 0; index < actual.Length; index++)
        {
            if (!float.IsFinite(actual[index]) || Math.Abs(expected[index] - actual[index]) > 0.0002 * Math.Max(1, Math.Abs(expected[index])))
            {
                throw new InvalidDataException("交错浮点 GPU 参考误差。");
            }

            checks++;
        }
    }

    foreach (bool interleaved in new[] { false, true })
    {
        var weights = new[] { 7, 9, 13 }.Select(rows =>
        {
            var weight = new Weight("group-test", [96, rows]) { Quantized = new sbyte[96 * rows], Scales = new float[3 * rows] };
            var source = Enumerable.Range(0, 96 * rows).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
            TensorMath.Quantize(source, weight.Quantized, weight.Scales);
            if (interleaved)
            {
                weight.PrepareInterleaved();
            }

            return weight;
        }).ToArray();
        float[] input = Enumerable.Range(0, 9 * 96).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
        var expected = new TensorMath(1, false).LinearMany(input, 9, weights);
        var actual = new TensorMath(4) { FloatBackend = gpu }.LinearMany(input, 9, weights);
        for (int projection = 0; projection < actual.Length; projection++)
        {
            for (int index = 0; index < actual[projection].Length; index++)
            {
                if (Math.Abs(actual[projection][index] - expected[projection][index]) > 0.0002 * Math.Max(1, Math.Abs(expected[projection][index])))
                {
                    throw new InvalidDataException("GPU 独立投影合并误差。");
                }

                checks++;
            }
        }
    }

    foreach (int kernel in new[] { 1, 3, 4, 7 })
    {
        foreach (int stride in new[] { 1, 2 })
        {
            foreach (int dilation in new[] { 1, 3 })
            {
                var input = new Tensor(Enumerable.Range(0, 2 * 3 * 257).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray(), [2, 3, 257]);
                var weight = new Tensor(Enumerable.Range(0, 5 * 3 * kernel).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray(), [5, 3, kernel]);
                var node = new OnnxNode("gpu-conv", "Conv", [], [], new Dictionary<string, object> { ["pads"] = new long[] { 2, 4 }, ["strides"] = new long[] { stride }, ["dilations"] = new long[] { dilation } });
                var actual = TensorOps.Conv1D(input, weight, new Tensor([0.1f, 0.2f, 0.3f, 0.4f, 0.5f], [5]), node, new TensorMath(4) { FloatBackend = gpu });
                for (int batch = 0; batch < 2; batch++)
                {
                    for (int channel = 0; channel < 5; channel++)
                    {
                        for (int time = 0; time < actual.Shape[2]; time++)
                        {
                            double expected = (channel + 1) * 0.1f;
                            for (int source = 0; source < 3; source++)
                            {
                                for (int tap = 0; tap < kernel; tap++)
                                {
                                    int at = time * stride + tap * dilation - 2;
                                    if ((uint)at < 257)
                                    {
                                        expected += (double)input.Data[(batch * 3 + source) * 257 + at] * weight.Data[(channel * 3 + source) * kernel + tap];
                                    }
                                }
                            }

                            float value = actual.Data[(batch * 5 + channel) * actual.Shape[2] + time];
                            if (!float.IsFinite(value) || Math.Abs(value - expected) > 0.0001 * Math.Max(1, Math.Abs(expected)))
                            {
                                throw new InvalidDataException("GPU 卷积独立参考误差。");
                            }

                            checks++;
                        }
                    }
                }
            }
        }
    }

    Emit(new
    {
        kind = "gpu-check",
        checks,
        passed = true,
        statistics = gpu.Statistics()
    });
    return;
}

if (args[0] == "cpu-info")
{
    Emit(new
    {
        avx2 = System.Runtime.Intrinsics.X86.Avx2.IsSupported,
        fma = System.Runtime.Intrinsics.X86.Fma.IsSupported,
        avxVnni = System.Runtime.Intrinsics.X86.AvxVnni.IsSupported,
        neon = System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported,
        sdot = System.Runtime.Intrinsics.Arm.Dp.IsSupported,
        threads = 4,
        runtime = Environment.Version.ToString()
    });
    return;
}

if (args[0] == "gpu-info")
{
    using var gpu = new VulkanFloatBackend();
    Emit(gpu.Statistics());
    return;
}

if (args[0] == "vulkan")
{
    VulkanProbe.Run(Emit);
    return;
}

if (args[0] == "npu")
{
    Emit(new
    {
        kind = "npu-configuration",
        precision = "F16 x F16 -> F32",
        runtimeAbi = "RKNN 2.3.2 ARM64",
        productBackend = false
    });
    foreach (var shape in new[] { (M: 256, K: 256, N: 256), (M: 96, K: 512, N: 512), (M: 512, K: 224, N: 32), (M: 1, K: 1024, N: 1024) })
    {
        Emit(new
        {
            kind = "npu-wait",
            shape.M,
            shape.K,
            shape.N,
            temperature = Temperature()
        });
        await CoolAsync();
        RknnProbe.Run(shape.M, shape.K, shape.N, 10, Emit);
    }

    return;
}

if (args[0] == "prepare-tts")
{
    bool aishell = args[3] == "aishell3";
    int[] encoded;
    if (aishell)
    {
        var model = new OnnxModel(Path.Combine(args[1], "aishell3/model.onnx"));
        encoded = new IcefallFrontend(args[2], Path.Combine(args[1], "aishell3/tokens.txt"), model.Metadata).Encode(args[4]);
    }
    else if (args[3] == "huayan")
    {
        encoded = new MandarinFrontend(args[2], Path.Combine(args[1], "zh_CN-huayan-medium.onnx.json")).Encode(args[4]);
    }
    else
    {
        throw new ArgumentException("TTS 模型须为 huayan 或 aishell3。");
    }
    long[] ids = encoded.Select(id => (long)id).ToArray();
    byte[] bytes = MemoryMarshal.AsBytes(ids.AsSpan()).ToArray();
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(args[5]))!);
    File.WriteAllBytes(args[5], bytes);
    Emit(new
    {
        kind = "tts-input",
        name = args[3],
        text = args[4],
        ids,
        sha256 = Convert.ToHexString(SHA256.HashData(bytes)),
        noise = 0,
        durationNoise = 0,
        speed = 1,
        speakerId = aishell ? 50 : 0
    });
    return;
}

if (args[0] == "prepare")
{
    string models = args[1], lexicon = args[2], samples = args[3], output = args[4];
    Directory.CreateDirectory(output);
    foreach (string name in new[] { "sip-short-pcma", "sip-zh-pcmu", "sip-repeated-pcma" })
    {
        string source = Path.Combine(samples, name + ".wav");
        var (audio, rate) = AudioFile.ReadWave(source);
        if (rate != 16000)
        {
            throw new InvalidDataException("识别基准要求已解码为 16kHz 的相同输入。");
        }

        byte[] bytes = MemoryMarshal.AsBytes(audio.AsSpan()).ToArray();
        File.WriteAllBytes(Path.Combine(output, name + ".f32"), bytes);
        Emit(new
        {
            kind = "asr-input",
            name,
            samples = audio.Length,
            audioSeconds = audio.Length / 16000.0,
            sha256 = Convert.ToHexString(SHA256.HashData(bytes))
        });
    }

    const string text = "您好，欢迎致电，请问有什么可以帮您？";
    var huayan = new MandarinFrontend(lexicon, Path.Combine(models, "zh_CN-huayan-medium.onnx.json"));
    var aishellModel = new OnnxModel(Path.Combine(models, "aishell3/model.onnx"));
    var aishell = new IcefallFrontend(Path.Combine(models, "aishell3/lexicon.txt"), Path.Combine(models, "aishell3/tokens.txt"), aishellModel.Metadata);
    foreach (var item in new[] { (Name: "huayan", Ids: huayan.Encode(text)), (Name: "aishell3", Ids: aishell.Encode(text)) })
    {
        long[] ids = item.Ids.Select(id => (long)id).ToArray();
        byte[] bytes = MemoryMarshal.AsBytes(ids.AsSpan()).ToArray();
        File.WriteAllBytes(Path.Combine(output, item.Name + ".i64"), bytes);
        Emit(new
        {
            kind = "tts-input",
            name = item.Name,
            text,
            ids,
            sha256 = Convert.ToHexString(SHA256.HashData(bytes)),
            noise = 0,
            durationNoise = 0,
            speed = 1,
            speakerId = item.Name == "aishell3" ? 50 : 0
        });
    }

    return;
}

if (args[0] == "asr")
{
    float[] audio = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[2])).ToArray();
    int repeats = int.Parse(args[3], CultureInfo.InvariantCulture);
    var watch = Stopwatch.StartNew();
    var engine = new FunAsr(Path.Combine(args[1], "funasr-encoder-f16.gguf"), Path.Combine(args[1], "qwen3-0.6b-q8_0.gguf"), 4);
    using var gpu = args.Contains("--vulkan") ? new VulkanFloatBackend() : null;
    engine.FloatBackend = gpu;
    GC.Collect(2, GCCollectionMode.Forced, true);
    Emit(new
    {
        kind = "load",
        engine = gpu is null ? "csharp-cpu" : "csharp-vulkan",
        seconds = watch.Elapsed.TotalSeconds,
        runtime = Environment.Version.ToString(),
        promptCache = engine.CachePrompt
    });
    for (int iteration = 0; iteration <= repeats; iteration++)
    {
        double cooldownSeconds = await CoolAsync();
        double? before = Temperature();
        int[] frequencyBefore = Frequencies();
        watch.Restart();
        var result = engine.Transcribe(audio, 128);
        double seconds = watch.Elapsed.TotalSeconds;
        Emit(new
        {
            kind = "asr",
            engine = gpu is null ? "csharp-cpu" : "csharp-vulkan",
            iteration,
            warmup = iteration == 0,
            seconds,
            audioSeconds = audio.Length / 16000.0,
            result.Text,
            result.GeneratedTokens,
            result.EncoderSeconds,
            result.DecoderSeconds,
            temperatureBefore = before,
            temperatureAfter = Temperature(),
            frequencyBefore,
            frequencyAfter = Frequencies(),
            cooldownSeconds,
            peakRssBytes = Process.GetCurrentProcess().PeakWorkingSet64
        });
        if (gpu is not null)
        {
            Emit(new
            {
                kind = "gpu-statistics",
                iteration,
                statistics = gpu.Statistics()
            });
        }
    }

    return;
}

if (args[0] == "http-asr")
{
    using var client = new HttpClient { BaseAddress = new Uri(args[1]), Timeout = TimeSpan.FromMinutes(2) };
    byte[] wave = File.ReadAllBytes(args[2]);
    int repeats = int.Parse(args[3], CultureInfo.InvariantCulture);
    Emit(new
    {
        kind = "configuration",
        engine = "csharp-service",
        timing = "requestSeconds: dispatch and full Transcribe call; excludes upload, WAV decoding and resampling",
        maxTokens = 512
    });
    for (int iteration = 0; iteration <= repeats; iteration++)
    {
        double cooldownSeconds = await CoolAsync();
        using var healthBefore = JsonDocument.Parse(await client.GetStringAsync("/health"));
        var health = healthBefore.RootElement;
        if (health.GetProperty("asrBusy").GetBoolean() || health.GetProperty("ttsBusy").GetBoolean()
            || health.GetProperty("activeSessions").GetInt32() != 0
            || health.GetProperty("asrQueued").GetInt32() != 0 || health.GetProperty("ttsQueued").GetInt32() != 0)
        {
            throw new InvalidOperationException("服务有其他请求，停止基准，不中断业务。");
        }

        double? before = Temperature();
        int[] frequencyBefore = Frequencies();
        using var body = new ByteArrayContent(wave);
        body.Headers.ContentType = new("audio/wav");
        var timer = Stopwatch.StartNew();
        using var response = await client.PostAsync("/asr/wav", body);
        response.EnsureSuccessStatusCode();
        using var result = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        double clientSeconds = timer.Elapsed.TotalSeconds;
        var value = result.RootElement;
        Emit(new
        {
            kind = "asr",
            engine = "csharp-service",
            iteration,
            warmup = iteration == 0,
            seconds = value.GetProperty("requestSeconds").GetDouble(),
            inferenceSeconds = value.GetProperty("inferenceSeconds").GetDouble(),
            audioSeconds = value.GetProperty("audioSeconds").GetDouble(),
            Text = value.GetProperty("text").GetString(),
            clientSeconds,
            temperatureBefore = before,
            temperatureAfter = Temperature(),
            frequencyBefore,
            frequencyAfter = Frequencies(),
            cooldownSeconds
        });
    }

    return;
}

if (args[0] == "tts")
{
    bool aishell = args[3] == "aishell3";
    int repeats = int.Parse(args[4], CultureInfo.InvariantCulture);
    int threadsIndex = Array.IndexOf(args, "--threads");
    int threads = threadsIndex < 0 ? 4 : int.Parse(args[threadsIndex + 1], CultureInfo.InvariantCulture);
    if (threads < 1)
    {
        throw new ArgumentOutOfRangeException(nameof(threads), "线程数须为正数。");
    }
    long[] ids = MemoryMarshal.Cast<byte, long>(File.ReadAllBytes(args[2])).ToArray();
    var watch = Stopwatch.StartNew();
    var model = new OnnxModel(Path.Combine(args[1], aishell ? "aishell3/model.onnx" : "zh_CN-huayan-medium.onnx"));
    using var gpu = args.Contains("--vulkan") ? new VulkanFloatBackend() : null;
    var engine = new ManagedOnnx(model, new TensorMath(threads) { Optimized = false, FloatBackend = gpu });
    Dictionary<string, double> profile = [];
    if (args.Contains("--profile"))
    {
        engine.NodeMeasured = (node, tensors, milliseconds) =>
        {
            string key = $"{node.Op} {node.Name} {string.Join(';', tensors.Select(tensor => tensor?.ToString() ?? "none"))}";
            profile[key] = profile.GetValueOrDefault(key) + milliseconds;
        };
    }
    var tokens = new Tensor(ids.Select(id => (float)id).ToArray(), [1, ids.Length], true);
    var length = new Tensor([(float)ids.Length], [1], true);
    Dictionary<string, Tensor> inputs = aishell ? new()
    {
        ["tokens"] = tokens,
        ["tokens_lens"] = length,
        ["noise_scale"] = new([0], [1]),
        ["alpha"] = new([1], [1]),
        ["noise_scale_dur"] = new([0], [1]),
        ["speaker"] = new([50], [1], true)
    } : new()
    {
        ["input"] = tokens,
        ["input_lengths"] = length,
        ["scales"] = new([0, 1, 0], [3])
    };
    GC.Collect(2, GCCollectionMode.Forced, true);
    Emit(new
    {
        kind = "load",
        engine = gpu is null ? "csharp-cpu" : "csharp-vulkan",
        seconds = watch.Elapsed.TotalSeconds,
        runtime = Environment.Version.ToString(),
        threads,
        reuseBuffers = engine.ReuseBuffers
    });
    for (int iteration = 0; iteration <= repeats; iteration++)
    {
        double cooldownSeconds = await CoolAsync();
        double? before = Temperature();
        int[] frequencyBefore = Frequencies();
        long allocatedBefore = GC.GetTotalAllocatedBytes(true);
        watch.Restart();
        profile.Clear();
        float[] audio = engine.Run(inputs, 1).Values.First().Data;
        double seconds = watch.Elapsed.TotalSeconds;
        long allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
        Emit(new
        {
            kind = "tts",
            engine = gpu is null ? "csharp-cpu" : "csharp-vulkan",
            iteration,
            warmup = iteration == 0,
            seconds,
            threads,
            allocatedBytes,
            samples = audio.Length,
            audioSeconds = audio.Length / (aishell ? 8000.0 : 22050.0),
            temperatureBefore = before,
            temperatureAfter = Temperature(),
            frequencyBefore,
            frequencyAfter = Frequencies(),
            cooldownSeconds,
            peakRssBytes = Process.GetCurrentProcess().PeakWorkingSet64
        });
        if (args.Contains("--profile"))
        {
            Emit(new
            {
                kind = "profile",
                iteration,
                operatorMilliseconds = profile.Values.Sum(),
                groups = profile.GroupBy(item => item.Key.Split(' ')[0]).ToDictionary(group => group.Key, group => group.Sum(item => item.Value)),
                slowest = profile.OrderByDescending(item => item.Value).Take(20).ToDictionary()
            });
        }
        if (gpu is not null)
        {
            Emit(new
            {
                kind = "gpu-statistics",
                iteration,
                statistics = gpu.Statistics()
            });
        }
        if (iteration == repeats)
        {
            File.WriteAllBytes(args[5], MemoryMarshal.AsBytes(audio.AsSpan()).ToArray());
        }
    }

    return;
}

if (args[0] == "compare")
{
    float[] reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[1])).ToArray();
    float[] actual = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[2])).ToArray();
    if (reference.Length != actual.Length || reference.Length == 0)
    {
        throw new InvalidDataException($"波形长度不一致：{reference.Length} / {actual.Length}");
    }

    double squaredError = 0, dot = 0, referenceEnergy = 0, actualEnergy = 0, maxAbsoluteError = 0;
    for (int index = 0; index < reference.Length; index++)
    {
        double expected = reference[index], value = actual[index], error = expected - value;
        if (!double.IsFinite(expected) || !double.IsFinite(value))
        {
            throw new InvalidDataException("波形含非有限数值。");
        }

        maxAbsoluteError = Math.Max(maxAbsoluteError, Math.Abs(error));
        squaredError += error * error;
        referenceEnergy += expected * expected;
        actualEnergy += value * value;
        dot += expected * value;
    }

    double rmse = Math.Sqrt(squaredError / reference.Length);
    bool passed = maxAbsoluteError <= 0.001 && rmse <= 0.0001;
    Emit(new
    {
        samples = reference.Length,
        maxAbsoluteError,
        rmse,
        cosine = dot / Math.Sqrt(referenceEnergy * actualEnergy),
        passed
    });
    Environment.ExitCode = passed ? 0 : 1;
    return;
}

throw new ArgumentException("未知基准模式。");
