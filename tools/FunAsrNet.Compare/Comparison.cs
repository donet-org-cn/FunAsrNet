using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace FunAsrNet.Compare;

/// <summary>纯 C# 的串行复测编排，保留原始输出并校验相同输入的结果。</summary>
internal static class Comparison
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    /// <summary>运行 CPU、GPU、AOT 或模型读取对照；子进程之间不并发执行推理。</summary>
    /// <param name="args">模式与显式路径选项。</param>
    /// <returns>全部测量与汇总完成后的任务。</returns>
    internal static async Task RunAsync(string[] args)
    {
        if (args.Length == 0)
        {
            throw new ArgumentException("用法：cpu|gpu|aot|load --output 新目录 [--models 目录] [--inputs 目录] [--before DLL] [--after DLL] [--subjects 名称列表]；load-probe 核心DLL 模型目录");
        }
        if (args[0] == "load-probe")
        {
            ProbeLoad(args[1], args[2]);
            return;
        }
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int index = 1; index < args.Length; index += 2)
        {
            if (index + 1 >= args.Length || !args[index].StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException("每个选项必须具有一个值。");
            }
            options.Add(args[index][2..], args[index + 1]);
        }
        string mode = args[0];
        if (mode is not ("cpu" or "gpu" or "aot" or "load"))
        {
            throw new ArgumentException("未知复测模式。");
        }
        string output = Path.GetFullPath(Value(options, "output", "artifacts/comparison"));
        if (Directory.Exists(output))
        {
            throw new IOException("输出目录已经存在，拒绝覆盖证据。");
        }
        Directory.CreateDirectory(output);
        string models = Path.GetFullPath(Value(options, "models", "artifacts/local-models"));
        string inputs = Path.GetFullPath(Value(options, "inputs", "artifacts/native-gpu/inputs"));
        string before = Path.GetFullPath(Value(options, "before", "artifacts/load-gpu-round6/before/FunAsrNet.Benchmarks.dll"));
        string after = Path.GetFullPath(Value(options, "after", "tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll"));
        var commands = new JsonObject();
        var binaries = new JsonObject();
        var groups = new JsonArray();
        var records = new Dictionary<string, List<JsonObject>>(StringComparer.Ordinal);
        var texts = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        var variantTexts = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        var phases = new Dictionary<string, List<double>>(StringComparer.Ordinal);
        int measured = 0, warmup = 0;
        string[] subjects = mode == "load" ? ["models"] : Value(options, "subjects", "sip-short-pcma,sip-zh-pcmu,sip-repeated-pcma,huayan,aishell3").Split(',');
        string[] order = mode == "aot" ? ["current", "jit", "aot", "native", "native", "aot", "jit", "current"] : mode == "load" ? ["before", "after", "after", "before"] : ["before", "after", "native", "native", "after", "before"];
        if (mode == "load")
        {
            int cycles = int.Parse(Value(options, "repeats", "3"), System.Globalization.CultureInfo.InvariantCulture);
            if (cycles <= 0)
            {
                throw new ArgumentException("测量轮数必须为正。");
            }
            order = Enumerable.Range(0, cycles).SelectMany(_ => new[] { "before", "after", "after", "before" }).ToArray();
        }
        foreach (string subject in subjects)
        {
            string kind = mode == "load" ? "load" : subject is "huayan" or "aishell3" ? "tts" : "asr";
            int repeats = int.Parse(Value(options, "repeats", subject == "sip-repeated-pcma" ? "3" : "5"), System.Globalization.CultureInfo.InvariantCulture);
            if (repeats <= 0)
            {
                throw new ArgumentException("测量轮数必须为正。");
            }
            for (int block = 0; block < order.Length; block++)
            {
                string variant = order[block], name = kind + "-" + subject + "-" + block + "-" + variant;
                string wave = Path.Combine(output, name + ".f32");
                string managed = variant == "before" ? before : variant == "jit" ? Path.GetFullPath(Value(options, "jit", "artifacts/aot-comparison/jit/FunAsrNet.AotBenchmarks.exe")) : variant == "aot" ? Path.GetFullPath(Value(options, "aot", "artifacts/aot-comparison/aot/FunAsrNet.AotBenchmarks.exe")) : after;
                List<string> command;
                if (mode == "load")
                {
                    command = ["dotnet", typeof(Comparison).Assembly.Location, "load-probe", Path.Combine(Path.GetDirectoryName(managed)!, "FunAsrNet.dll"), models];
                }
                else if (variant == "native")
                {
                    if (kind == "asr")
                    {
                        string cpp = Path.GetFullPath(Value(options, "cpp", mode == "gpu" ? "artifacts/native-gpu/build/funasr-native-gpu-reference.exe" : "artifacts/native-cpu/build-vnni/funasr-native-cpu-reference.exe"));
                        command = [cpp, Path.Combine(models, "funasr-encoder-f16.gguf"), Path.Combine(models, "qwen3-0.6b-q8_0.gguf"), Path.Combine(inputs, subject + ".f32"), repeats.ToString(System.Globalization.CultureInfo.InvariantCulture)];
                        if (mode == "gpu")
                        {
                            command.Add("--encoder-gpu");
                        }
                    }
                    else
                    {
                        string ort = Path.GetFullPath(Value(options, "ort", mode == "gpu" ? "tools/FunAsrNet.NativeGpuReference/bin/Release/net10.0/FunAsrNet.NativeGpuReference.dll" : "tools/FunAsrNet.NativeReference/bin/Release/net10.0/FunAsrNet.NativeReference.dll"));
                        command = ["dotnet", ort, Path.Combine(models, subject == "huayan" ? "zh_CN-huayan-medium.onnx" : "aishell3/model.onnx"), Path.Combine(inputs, subject + ".i64"), subject, repeats.ToString(System.Globalization.CultureInfo.InvariantCulture), wave, "--provider", mode == "gpu" ? "cuda" : "cpu", "--threads", "4", "--affinity", "0x55"];
                    }
                }
                else
                {
                    command = managed.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) ? ["dotnet", managed] : [managed];
                    command.AddRange([kind, models, Path.Combine(inputs, subject + (kind == "tts" ? ".i64" : ".f32"))]);
                    command.AddRange(kind == "tts" ? [subject, repeats.ToString(System.Globalization.CultureInfo.InvariantCulture), wave, "--threads", "4"] : [repeats.ToString(System.Globalization.CultureInfo.InvariantCulture)]);
                    command.AddRange(["--affinity", "0x55"]);
                    if (mode == "gpu")
                    {
                        command.Add("--vulkan");
                    }
                }
                commands[name] = new JsonArray(command.Select(part => (JsonNode?)JsonValue.Create(part)).ToArray());
                string executable = command[0] == "dotnet" ? command[1] : command[0];
                binaries[executable] = Digest(executable);
                if (variant != "native")
                {
                    foreach (string dependency in new[] { "FunAsrNet.dll", "FunAsrNet.Vulkan.dll" })
                    {
                        string path = Path.Combine(Path.GetDirectoryName(managed)!, dependency);
                        if (File.Exists(path))
                        {
                            binaries[path] = Digest(path);
                        }
                    }
                }
                Console.WriteLine("Starting " + name);
                await SaveAsync(Path.Combine(output, "commands.json"), commands);
                await ExecuteAsync(command, Path.Combine(output, name + ".jsonl"), Path.Combine(output, name + ".stderr"));
                int samples = 0, warmups = 0;
                double? loadSeconds = null;
                foreach (string line in await File.ReadAllLinesAsync(Path.Combine(output, name + ".jsonl")))
                {
                    var row = JsonNode.Parse(line)!.AsObject();
                    if (kind != "load" && row["kind"]?.GetValue<string>() == "load")
                    {
                        if (loadSeconds is not null)
                        {
                            throw new InvalidDataException("重复的初始化记录：" + name);
                        }
                        loadSeconds = row["seconds"]!.GetValue<double>();
                        AddPhase(phases, subject + "/" + variant + "/load", loadSeconds.Value);
                    }
                    if (row["kind"]?.GetValue<string>() != kind)
                    {
                        continue;
                    }
                    if (kind != "load" && row["warmup"]!.GetValue<bool>())
                    {
                        double first = row["seconds"]!.GetValue<double>();
                        AddPhase(phases, subject + "/" + variant + "/firstInference", first);
                        AddPhase(phases, subject + "/" + variant + "/loadAndFirstInference", (loadSeconds ?? throw new InvalidDataException("首次调用前缺少加载记录：" + name)) + first);
                        warmup++;
                        warmups++;
                        continue;
                    }
                    string key = subject + "/" + variant;
                    if (!records.TryGetValue(key, out var list))
                    {
                        records.Add(key, list = []);
                    }
                    list.Add(row);
                    measured++;
                    samples++;
                    if (kind == "asr")
                    {
                        if (!texts.TryGetValue(subject, out var set))
                        {
                            texts.Add(subject, set = new(StringComparer.Ordinal));
                        }
                        set.Add(row["Text"]!.GetValue<string>());
                        if (!variantTexts.TryGetValue(key, out var variantSet))
                        {
                            variantTexts.Add(key, variantSet = new(StringComparer.Ordinal));
                        }
                        variantSet.Add(row["Text"]!.GetValue<string>());
                    }
                }
                if (samples != (kind == "load" ? 1 : repeats) || warmups != (kind == "load" ? 0 : 1))
                {
                    throw new InvalidDataException("测量数量与协议不一致：" + name);
                }
                if (kind == "tts" && variant == "after")
                {
                    foreach (string reference in Directory.EnumerateFiles(output, "tts-" + subject + "-*-native.f32").Concat(Directory.EnumerateFiles(output, "tts-" + subject + "-*-before.f32")))
                    {
                        string parity = Path.Combine(output, "parity-" + name + "-" + Path.GetFileNameWithoutExtension(reference) + ".jsonl");
                        await ExecuteAsync(["dotnet", after, "compare", reference, wave], parity, parity + ".stderr");
                    }
                }
            }
        }
        foreach (var entry in records)
        {
            double[] seconds = entry.Value.Select(row => row["seconds"]!.GetValue<double>()).Order().ToArray();
            groups.Add(new JsonObject { ["group"] = entry.Key, ["count"] = seconds.Length, ["medianSeconds"] = Median(seconds), ["p95Seconds"] = seconds[(int)Math.Ceiling(seconds.Length * 0.95) - 1], ["minimumSeconds"] = seconds[0], ["maximumSeconds"] = seconds[^1] });
        }
        var textChecks = new JsonObject();
        foreach (var entry in texts)
        {
            var byVariant = new JsonObject();
            foreach (var variant in variantTexts.Where(pair => pair.Key.StartsWith(entry.Key + "/", StringComparison.Ordinal)))
            {
                byVariant[variant.Key.Split('/')[1]] = new JsonArray(variant.Value.Select(value => (JsonNode?)JsonValue.Create(value)).ToArray());
            }
            textChecks[entry.Key] = new JsonObject { ["exactMatch"] = entry.Value.Count == 1, ["textsByVariant"] = byVariant };
        }
        var modelHashes = new JsonObject();
        foreach (string file in new[] { "funasr-encoder-f16.gguf", "qwen3-0.6b-q8_0.gguf", "zh_CN-huayan-medium.onnx", "aishell3/model.onnx" })
        {
            modelHashes[file] = Digest(Path.Combine(models, file));
        }
        var inputHashes = new JsonObject();
        foreach (string file in Directory.EnumerateFiles(inputs).Where(file => subjects.Contains(Path.GetFileNameWithoutExtension(file))))
        {
            inputHashes[Path.GetFileName(file)] = Digest(file);
        }
        await SaveAsync(Path.Combine(output, "commands.json"), commands);
        var phaseGroups = new JsonArray();
        foreach (var entry in phases)
        {
            double[] seconds = entry.Value.Order().ToArray();
            phaseGroups.Add(new JsonObject { ["group"] = entry.Key, ["count"] = seconds.Length, ["medianSeconds"] = Median(seconds), ["p95Seconds"] = seconds[(int)Math.Ceiling(seconds.Length * 0.95) - 1], ["minimumSeconds"] = seconds[0], ["maximumSeconds"] = seconds[^1] });
        }
        await SaveAsync(Path.Combine(output, "summary.json"), new JsonObject { ["mode"] = mode, ["threads"] = 4, ["affinity"] = "0x55", ["measured"] = measured, ["warmup"] = warmup, ["groups"] = groups, ["startupPhases"] = phaseGroups, ["asrTextChecks"] = textChecks, ["binarySha256"] = binaries, ["modelSha256"] = modelHashes, ["inputSha256"] = inputHashes });
        Console.WriteLine(groups.ToJsonString(Json));
        foreach (string subject in subjects.Where(name => name.StartsWith("sip-", StringComparison.Ordinal)))
        {
            if (variantTexts.TryGetValue(subject + "/before", out var original) && (!variantTexts.TryGetValue(subject + "/after", out var updated) || !original.SetEquals(updated) || original.Count != 1))
            {
                throw new InvalidDataException("优化前后转写发生变化，不能接受性能结果：" + subject);
            }
        }
    }

    /// <summary>记录新进程内的加载、首次调用及二者之和，避免与热态样本混合。</summary>
    /// <param name="phases">按项目、版本、阶段分类的计时。</param>
    /// <param name="key">阶段名称。</param>
    /// <param name="seconds">本进程实测秒数。</param>
    private static void AddPhase(Dictionary<string, List<double>> phases, string key, double seconds)
    {
        if (!double.IsFinite(seconds) || seconds < 0)
        {
            throw new InvalidDataException("无效的阶段耗时：" + key);
        }
        if (!phases.TryGetValue(key, out var list))
        {
            phases.Add(key, list = []);
        }
        list.Add(seconds);
    }

    /// <summary>从字典读取选项，缺失时采用固定协议默认值。</summary>
    /// <param name="options">已解析选项。</param>
    /// <param name="key">选项名称。</param>
    /// <param name="fallback">默认值。</param>
    /// <returns>选项值。</returns>
    private static string Value(Dictionary<string, string> options, string key, string fallback) => options.GetValueOrDefault(key, fallback);

    /// <summary>以流读取计算文件散列，避免额外保留大模型副本。</summary>
    /// <param name="path">文件路径。</param>
    /// <returns>小写 SHA-256。</returns>
    private static string Digest(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    /// <summary>计算排序样本的中位数。</summary>
    /// <param name="values">非空且已排序样本。</param>
    /// <returns>中间值或两个中间值的平均。</returns>
    private static double Median(double[] values) => values.Length % 2 == 0 ? (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2 : values[values.Length / 2];

    /// <summary>写入带真实换行的 UTF-8 JSON。</summary>
    /// <param name="path">目标路径。</param>
    /// <param name="value">完整 JSON 数据。</param>
    /// <returns>写入结束的任务。</returns>
    private static Task SaveAsync(string path, JsonNode value) => File.WriteAllTextAsync(path, value.ToJsonString(Json) + "\n");

    /// <summary>使用参数列表运行一个子进程，完整捕获输出并等待退出。</summary>
    /// <param name="command">可执行文件及独立参数。</param>
    /// <param name="output">标准输出目标。</param>
    /// <param name="error">诊断输出目标。</param>
    /// <returns>进程和输出都结束的任务。</returns>
    private static async Task ExecuteAsync(List<string> command, string output, string error)
    {
        var start = new ProcessStartInfo(command[0]) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
        foreach (string argument in command.Skip(1))
        {
            start.ArgumentList.Add(argument);
        }
        foreach (string key in start.Environment.Keys.Where(key => key.StartsWith("FUNASRNET_", StringComparison.OrdinalIgnoreCase)).ToArray())
        {
            start.Environment.Remove(key);
        }
        start.Environment["FUNASRNET_DISABLE_PROMPT_CACHE"] = "1";
        start.Environment["OMP_NUM_THREADS"] = "4";
        using var process = Process.Start(start) ?? throw new InvalidOperationException("无法启动复测进程。");
        await using var stdout = File.Create(output);
        await using var stderr = File.Create(error);
        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(10));
        try
        {
            await Task.WhenAll(process.StandardOutput.BaseStream.CopyToAsync(stdout, timeout.Token), process.StandardError.BaseStream.CopyToAsync(stderr, timeout.Token), process.WaitForExitAsync(timeout.Token));
        }
        catch
        {
            if (!process.HasExited)
            {
                process.Kill(true);
            }
            throw;
        }
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException($"复测退出码 {process.ExitCode}，诊断见 {error}");
        }
    }

    /// <summary>在独立进程中加载指定快照的核心，分开计量编码器和解码器读取。</summary>
    /// <param name="core">被测核心 DLL，允许是上一版独立副本。</param>
    /// <param name="models">固定模型目录。</param>
    private static void ProbeLoad(string core, string models)
    {
        if (OperatingSystem.IsWindows())
        {
            Process.GetCurrentProcess().ProcessorAffinity = (nint)0x55;
        }
        var assembly = System.Runtime.Loader.AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.GetFullPath(core));
        var constructor = assembly.GetType("FunAsrNet.Gguf", true)!.GetConstructor([typeof(string)])!;
        long allocated = GC.GetTotalAllocatedBytes(true);
        var clock = Stopwatch.StartNew();
        object encoder = constructor.Invoke([Path.Combine(models, "funasr-encoder-f16.gguf")]);
        double encoderSeconds = clock.Elapsed.TotalSeconds;
        object decoder = constructor.Invoke([Path.Combine(models, "qwen3-0.6b-q8_0.gguf")]);
        Console.WriteLine(JsonSerializer.Serialize(new { kind = "load", seconds = clock.Elapsed.TotalSeconds, encoderSeconds, decoderSeconds = clock.Elapsed.TotalSeconds - encoderSeconds, allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocated, peakRssBytes = Process.GetCurrentProcess().PeakWorkingSet64, coreSha256 = Digest(core) }));
        GC.KeepAlive(encoder);
        GC.KeepAlive(decoder);
    }
}
