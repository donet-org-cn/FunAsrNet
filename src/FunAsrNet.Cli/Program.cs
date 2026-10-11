using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Text.Json;
using FunAsrNet;

Console.OutputEncoding = System.Text.Encoding.UTF8;
// <summary>
// 读取命令行选项，缺失时使用默认值。
// </summary>
// <param name="key">GGUF 元数据键或命令行选项名。</param>
// <param name="fallback">键或属性不存在时采用的默认值。</param>
// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
string? Option(string key, string? fallback = null)
{
    int i = Array.IndexOf(args, key);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
}

// <summary>
// 读取必填命令行选项，缺失或无值时报告使用错误。
// </summary>
// <param name="key">GGUF 元数据键或命令行选项名。</param>
// <returns>完成处理后的文本。</returns>
string Required(string key) => Option(key) ?? throw new ArgumentException($"Missing {key}");
try
{
    if (args.Length == 0)
    {
        throw new ArgumentException("Usage: inspect MODEL | asr --enc FILE --llm FILE --wav FILE [--threads 4 --repeat 3 --trace DIR]");
    }

    if (args[0] == "inspect-onnx")
    {
        var model = new OnnxModel(args[1]);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            model.Inputs,
            model.Outputs,
            model.Metadata,
            operators = model.Nodes.GroupBy(node => node.Op).ToDictionary(group => group.Key, group => group.Count())
        }, new JsonSerializerOptions { WriteIndented = true }));
        return;
    }

    if (args[0] == "inspect")
    {
        var model = new Gguf(args[1]);
        foreach (var (k, v) in model.Metadata)
        {
            Console.WriteLine($"{k} = {(v is object[] a ? $"[{a.Length} values]" : v)}");
        }

        foreach (var (k, v) in model.Tensors.Take(30))
        {
            Console.WriteLine($"{k} [{string.Join(',', v.Shape)}] {(v.Values is null ? "Q8_0" : "F32")}");
        }

        return;
    }

    if (args[0] == "stream")
    {
        int streamThreads = int.Parse(Option("--threads", "4")!);
        var recognizer = new FunAsr(Required("--enc"), Required("--llm"), streamThreads);
        var vad = new Gguf(Required("--vad"));
        var codec = Enum.Parse<AudioCodec>(Option("--codec", "Pcma")!, true);
        int streamRate = int.Parse(Option("--rate", "8000")!);
        var converter = new PhoneAudioDecoder(codec, streamRate);
        var json = new JsonSerializerOptions
        {
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase

        };
        if (args.Contains("--warmup"))
        {
            recognizer.Transcribe(new float[1600], 1);
        }

        Console.WriteLine(JsonSerializer.Serialize(
            new
            {
                @event = "ready",
                engine = "pure-csharp",
                neon = AdvSimd.IsSupported,
                sdot = Dp.IsSupported

            },
            json));
        await using var stream = new PhoneStream(
            recognizer,
            vad,
            e => Console.Out.WriteLineAsync(JsonSerializer.Serialize(e, json)),
            args.Contains("--partials"),
            int.Parse(Option("--silence-ms", "1200")!));
        using var input = Option("--input") is string source ? File.OpenRead(source) : Console.OpenStandardInput();
        var buffer = new byte[streamRate / 50 * (codec == AudioCodec.Pcm16 ? 2 : 1)];
        var decoded = new float[converter.GetMaxOutput(buffer.Length)];
        var pacing = Stopwatch.StartNew();
        long consumed = 0;
        while (true)
        {
            int count = await input.ReadAsync(buffer);
            if (count == 0)
            {
                break;
            }

            consumed += count;
            if (args.Contains("--realtime"))
            {
                double due = consumed / (double)(streamRate * (codec == AudioCodec.Pcm16 ? 2 : 1));
                double delay = due - pacing.Elapsed.TotalSeconds;
                if (delay > 0)
                {
                    await Task.Delay(TimeSpan.FromSeconds(delay));
                }
            }

            int samples = converter.Decode(buffer.AsSpan(0, count), decoded);
            stream.Feed(decoded.AsSpan(0, samples));
        }

        int tail = converter.Decode([], decoded, true);
        stream.Feed(decoded.AsSpan(0, tail));
        await stream.CompleteAsync();
        return;
    }

    if (args[0] == "vad")
    {
        var (audio, inputRate) = AudioFile.ReadWave(Required("--wav"));
        audio = AudioFile.Resample(audio, inputRate, 16000);
        var vad = new FsmnVad(new Gguf(Required("--model")));
        var endpoint = new PhoneEndpoint();
        int total = 0;
        var timer = Stopwatch.StartNew();
        List<float> probabilities = [];
        for (int offset = 0; offset < audio.Length; offset += 320)
        {
            int count = Math.Min(320, audio.Length - offset);
            var scores = vad.Feed(audio.AsSpan(offset, count), offset + count == audio.Length);
            probabilities.AddRange(scores);
            foreach (float score in scores)
            {
                ++total;
                var segment = endpoint.Step(score);
                if (segment.HasValue)
                {
                    Console.WriteLine(JsonSerializer.Serialize(segment.Value));
                }
            }
        }

        var trailing = endpoint.Flush(audio.Length);
        if (trailing.HasValue)
        {
            Console.WriteLine(JsonSerializer.Serialize(trailing.Value));
        }

        if (Option("--out") is string file)
        {
            File.WriteAllBytes(file, MemoryMarshal.AsBytes(probabilities.ToArray().AsSpan()).ToArray());
        }

        Console.Error.WriteLine(JsonSerializer.Serialize(new
        {
            seconds = timer.Elapsed.TotalSeconds,
            frames = total

        }));
        return;
    }

    if (args[0] == "tts")
    {
        var loadTimer = Stopwatch.StartNew();
        int ttsThreads = int.Parse(Option("--threads", "4")!);
        MandarinTts tts;
        if (Option("--tokens") is string tokensPath)
        {
            var model = new OnnxModel(Required("--model"));
            tts = new MandarinTts(model, new IcefallFrontend(Required("--lexicon"), tokensPath, model.Metadata), ttsThreads);
        }
        else
        {
            tts = new MandarinTts(Required("--model"), Required("--lexicon"), ttsThreads);
        }
        Console.Error.WriteLine(JsonSerializer.Serialize(new
        {
            loadSeconds = loadTimer.Elapsed.TotalSeconds,
            neon = AdvSimd.IsSupported

        }));
        for (int i = 0; i < int.Parse(Option("--repeat", "1")!); ++i)
        {
            var timer = Stopwatch.StartNew();
            float[] audio = tts.Synthesize(Required("--text"), float.Parse(Option("--speed", "1")!), speakerId: int.Parse(Option("--speaker", "0")!));
            double elapsed = timer.Elapsed.TotalSeconds;
            string output = Required("--out");
            int outputRate = int.Parse(Option("--rate", tts.NativeSampleRate.ToString())!);
            AudioFile.WriteWave(output, AudioFile.Resample(audio, tts.NativeSampleRate, outputRate), outputRate);
            Console.WriteLine(JsonSerializer.Serialize(
                new
                {
                    iteration = i,
                    text = Required("--text"),
                    seconds = elapsed,
                    audioSeconds = audio.Length / (double)tts.NativeSampleRate,
                    rtf = elapsed / (audio.Length / (double)tts.NativeSampleRate),
                    peakRss = Process.GetCurrentProcess().PeakWorkingSet64

                },
                new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        }

        return;
    }

    if (args[0] == "tts-ids")
    {
        var model = new OnnxModel(Required("--model"));
        var runtime = new ManagedOnnx(model, new TensorMath(int.Parse(Option("--threads", "4")!)) { Optimized = false });
        int[] ids = JsonSerializer.Deserialize<int[]>(File.ReadAllText(Required("--ids")))!;
        string? traceDir = Option("--trace");
        if (traceDir is not null)
        {
            Directory.CreateDirectory(traceDir);
            runtime.Trace = (i, node, output) =>
            {
                if (i % 100 == 0 || i == model.Nodes.Count - 1)
                {
                    File.WriteAllBytes(Path.Combine(traceDir, $"node-{i}.f32"), MemoryMarshal.AsBytes(output[0].Data.AsSpan()).ToArray());
                    Console.Error.WriteLine($"{i} {node.Op} {node.Name} {output[0]}");
                }
            };
        }

        var timer = Stopwatch.StartNew();
        var outputAudio = runtime.Run(new()
        {
            ["input"] = new(ids.Select(i => (float)i).ToArray(), [1, ids.Length], true),
            ["input_lengths"] = new([(float)ids.Length], [1], true),
            ["scales"] = new([float.Parse(Option("--noise", "0.667")!), float.Parse(Option("--length", "1")!), float.Parse(Option("--noise-w", "0.8")!)], [3])

        }).Values.First();
        AudioFile.WriteWave(Required("--out"), outputAudio.Data, 22050);
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            seconds = timer.Elapsed.TotalSeconds,
            audioSeconds = outputAudio.Count / 22050.0,
            peakRss = Process.GetCurrentProcess().PeakWorkingSet64

        }));
        return;
    }

    if (args[0] != "asr")
    {
        throw new ArgumentException("Unknown command");
    }

    var (pcm, rate) = AudioFile.ReadWave(Required("--wav"));
    pcm = AudioFile.Resample(pcm, rate, 16000);
    int threads = int.Parse(Option("--threads", "4")!);
    var watch = Stopwatch.StartNew();
    var asr = new FunAsr(Required("--enc"), Required("--llm"), threads, !args.Contains("--scalar"));
    // GGUF 解析会分配较大的临时字节缓冲，启动阶段集中回收一次。
    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: false);
    Console.Error.WriteLine(JsonSerializer.Serialize(new
    {
        phase = "loaded",
        seconds = watch.Elapsed.TotalSeconds,
        neon = AdvSimd.IsSupported,
        sdot = Dp.IsSupported,
        threads

    }));
    if (Option("--verify-cache") is string otherWave)
    {
        foreach (string path in new[]
            {
                Required("--wav"),
                otherWave,
                Required("--wav")
            }
        )
        {
            var (samples, sr) = AudioFile.ReadWave(path);
            samples = AudioFile.Resample(samples, sr, 16000);
            float[]? first = null;
            asr.Trace = (name, values) =>
            {
                if (name == "logits")
                {
                    first ??= values;
                }
            };
            asr.CachePrompt = true;
            var cached = asr.Transcribe(samples);
            float[] cachedLogits = first!;
            first = null;
            asr.CachePrompt = false;
            var uncached = asr.Transcribe(samples);
            float error = cachedLogits.Zip(first!).Max(pair => Math.Abs(pair.First - pair.Second));
            if (cached.Text != uncached.Text || error > 1e-4)
            {
                throw new Exception($"Prompt cache mismatch: {error}");
            }

            Console.WriteLine(JsonSerializer.Serialize(new
            {
                path,
                cached.Text,
                maxLogitError = error,
                passed = true

            }));
        }

        return;
    }

    if (Option("--verify-workspace") is string workspaceWave)
    {
        // 比较中间张量前先准备固定前缀，使两条路径仅包含音频计算，
        // 固定前缀的构建只需完成一次。
        asr.Transcribe(pcm, 1);
        foreach (string path in new[]
            {
                Required("--wav"),
                workspaceWave,
                Required("--wav")
            }
        )
        {
            var (samples, sr) = AudioFile.ReadWave(path);
            samples = AudioFile.Resample(samples, sr, 16000);
            List<(string Name, float[] Values)> snapshots = [];
            asr.Trace = (name, values) => snapshots.Add((name, values));
            asr.ReuseBuffers = true;
            var pooled = asr.Transcribe(samples);
            var expected = snapshots.ToArray();
            snapshots.Clear();
            asr.ReuseBuffers = false;
            var fresh = asr.Transcribe(samples);
            bool same = pooled.Text == fresh.Text && expected.Length == snapshots.Count && expected.Zip(snapshots).All(p => p.First.Name == p.Second.Name && p.First.Values.AsSpan().SequenceEqual(p.Second.Values));
            if (!same)
            {
                throw new Exception($"Workspace tensor reuse changed model results: pooled={pooled.Text}; fresh={fresh.Text}; trace counts {expected.Length}/{snapshots.Count}; differences={string.Join(
                        ',',
                        expected.Zip(snapshots).Where(p => p.First.Name != p.Second.Name || !p.First.Values.AsSpan().SequenceEqual(p.Second.Values)).Select(p => p.First.Name + '/' + p.Second.Name))}");
            }

            Console.WriteLine(JsonSerializer.Serialize(new
            {
                path,
                pooled.Text,
                traceTensors = expected.Length,
                exactTensorParity = true,
                passed = true

            }));
        }

        return;
    }

    string? trace = Option("--trace");
    int logitsCount = 0;
    if (trace is not null)
    {
        Directory.CreateDirectory(trace);
        asr.Trace = (name, data) =>
        {
            if (name == "logits")
            {
                name += $"-{logitsCount++}";
            }

            File.WriteAllBytes(Path.Combine(trace, name + ".f32"), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
        };
    }

    for (int i = 0; i < int.Parse(Option("--repeat", "1")!); ++i)
    {
        Dictionary<string, double> profile = [];
        if (args.Contains("--profile"))
        {
            asr.LinearMeasured = (name, ms) => profile[name] = profile.GetValueOrDefault(name) + ms;
        }

        long allocated = GC.GetTotalAllocatedBytes();
        int gc2 = GC.CollectionCount(2);
        var cpuStart = Process.GetCurrentProcess().TotalProcessorTime;
        var wall = Stopwatch.StartNew();
        var result = asr.Transcribe(pcm, int.Parse(Option("--max-tokens", "128")!));
        double wallSeconds = wall.Elapsed.TotalSeconds;
        Console.WriteLine(JsonSerializer.Serialize(
            new
            {
                iteration = i,
                result,
                wallSeconds,
                peakRssBytes = Process.GetCurrentProcess().PeakWorkingSet64

            },
            new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        if (args.Contains("--profile"))
        {
            Console.Error.WriteLine(JsonSerializer.Serialize(new
            {
                iteration = i,
                cpuSeconds = (Process.GetCurrentProcess().TotalProcessorTime - cpuStart).TotalSeconds,
                allocatedBytes = GC.GetTotalAllocatedBytes() - allocated,
                gen2Collections = GC.CollectionCount(2) - gc2,
                operatorMs = profile.Values.Sum(),
                groups = profile.GroupBy(p => p.Key.Split(' ')[0] + " " + p.Key.Split(' ')[1]).ToDictionary(g => g.Key, g => g.Sum(p => p.Value)),
                slowest = profile.OrderByDescending(p => p.Value).Take(15).ToDictionary()

            }));
        }
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    Environment.ExitCode = 1;
}
