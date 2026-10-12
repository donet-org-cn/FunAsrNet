using System.Diagnostics;
using System.Globalization;
using System.Runtime;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace FunAsrNet.AotBenchmarks;

/// <summary>以完全相同的入口比较 JIT 与 Native AOT 的纯 CPU 模型调用。</summary>
internal static class Program
{
    /// <summary>配置线程亲和性并执行指定模型；JSON 写入位于推理计时之外。</summary>
    /// <param name="args">模式、模型目录、输入文件、模型名、次数、波形输出及可选线程参数。</param>
    private static void Main(string[] args)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        int threads = Option(args, "--threads", 4);
        int affinity = Array.IndexOf(args, "--affinity");
        if (OperatingSystem.IsWindows() && affinity >= 0)
        {
            Process.GetCurrentProcess().ProcessorAffinity = (nint)Convert.ToInt64(args[affinity + 1].Replace("0x", "", StringComparison.OrdinalIgnoreCase), 16);
        }
        Emit(writer =>
        {
            writer.WriteString("kind", "configuration");
            writer.WriteString("engine", RuntimeFeature.IsDynamicCodeCompiled ? "jit" : "aot");
            writer.WriteString("runtime", Environment.Version.ToString());
            writer.WriteBoolean("dynamicCodeSupported", RuntimeFeature.IsDynamicCodeSupported);
            writer.WriteBoolean("serverGc", GCSettings.IsServerGC);
            writer.WriteBoolean("avx2", Avx2.IsSupported);
            writer.WriteBoolean("fma", Fma.IsSupported);
            writer.WriteBoolean("avxVnni", AvxVnni.IsSupported);
            writer.WriteNumber("threads", threads);
        });
        if (args.Length == 1 && args[0] == "cpu-info")
        {
            return;
        }
        if (args.Length < 4 || args[0] is not ("asr" or "tts"))
        {
            throw new ArgumentException("用法：cpu-info | asr 模型目录 PCM文件 次数 | tts 模型目录 ID文件 huayan或aishell3 次数 输出文件；可选 --threads 4 --affinity 0x55。");
        }
        if (args[0] == "asr")
        {
            RunAsr(args, threads);
        }
        else
        {
            RunTts(args, threads);
        }
    }

    /// <summary>解析正整数选项并拒绝缺少值或非正数。</summary>
    /// <param name="args">命令行参数。</param>
    /// <param name="name">选项名称。</param>
    /// <param name="fallback">选项未出现时的默认值。</param>
    /// <returns>验证后的选项值。</returns>
    private static int Option(string[] args, string name, int fallback)
    {
        int index = Array.IndexOf(args, name);
        int value = index < 0 ? fallback : int.Parse(args[index + 1], CultureInfo.InvariantCulture);
        return value > 0 ? value : throw new ArgumentOutOfRangeException(name);
    }

    /// <summary>输出一条不依赖反射或动态代码生成的 JSON 记录。</summary>
    /// <param name="write">写入本条记录字段的回调。</param>
    private static void Emit(Action<Utf8JsonWriter> write)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            write(writer);
            writer.WriteEndObject();
        }
        Console.WriteLine(Encoding.UTF8.GetString(stream.GetBuffer(), 0, checked((int)stream.Length)));
    }

    /// <summary>以同一 F16 编码器和 Q8 解码器运行完整识别，首次调用单独标记。</summary>
    /// <param name="args">识别模型、输入与次数参数。</param>
    /// <param name="threads">计算线程数。</param>
    private static void RunAsr(string[] args, int threads)
    {
        float[] audio = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(args[2])).ToArray();
        int repeats = int.Parse(args[3], CultureInfo.InvariantCulture);
        ArgumentOutOfRangeException.ThrowIfNegative(repeats);
        var watch = Stopwatch.StartNew();
        var engine = new FunAsr(Path.Combine(args[1], "funasr-encoder-f16.gguf"), Path.Combine(args[1], "qwen3-0.6b-q8_0.gguf"), threads);
        GC.Collect(2, GCCollectionMode.Forced, true);
        double loadSeconds = watch.Elapsed.TotalSeconds;
        Emit(writer =>
        {
            writer.WriteString("kind", "load");
            writer.WriteNumber("seconds", loadSeconds);
            writer.WriteBoolean("promptCache", engine.CachePrompt);
        });
        for (int iteration = 0; iteration <= repeats; iteration++)
        {
            long allocatedBefore = GC.GetTotalAllocatedBytes(true);
            watch.Restart();
            var result = engine.Transcribe(audio, 128);
            double seconds = watch.Elapsed.TotalSeconds;
            long allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
            Emit(writer =>
            {
                Measurement(writer, "asr", iteration, seconds, allocatedBytes);
                writer.WriteNumber("audioSeconds", audio.Length / 16000.0);
                writer.WriteString("Text", result.Text);
                writer.WriteNumber("GeneratedTokens", result.GeneratedTokens);
                writer.WriteNumber("EncoderSeconds", result.EncoderSeconds);
                writer.WriteNumber("DecoderSeconds", result.DecoderSeconds);
            });
        }
    }

    /// <summary>以与现有基准相同的音素、发音人和零噪声配置运行 TTS。</summary>
    /// <param name="args">模型、音素、次数和最终波形输出路径。</param>
    /// <param name="threads">计算线程数。</param>
    private static void RunTts(string[] args, int threads)
    {
        if (args.Length < 6 || args[3] is not ("huayan" or "aishell3"))
        {
            throw new ArgumentException("TTS 需要 huayan 或 aishell3、次数和输出文件。");
        }
        bool aishell = args[3] == "aishell3";
        int repeats = int.Parse(args[4], CultureInfo.InvariantCulture);
        ArgumentOutOfRangeException.ThrowIfNegative(repeats);
        long[] ids = MemoryMarshal.Cast<byte, long>(File.ReadAllBytes(args[2])).ToArray();
        var watch = Stopwatch.StartNew();
        var model = new OnnxModel(Path.Combine(args[1], aishell ? "aishell3/model.onnx" : "zh_CN-huayan-medium.onnx"));
        var engine = new ManagedOnnx(model, new TensorMath(threads) { Optimized = false });
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
        double loadSeconds = watch.Elapsed.TotalSeconds;
        Emit(writer =>
        {
            writer.WriteString("kind", "load");
            writer.WriteNumber("seconds", loadSeconds);
            writer.WriteBoolean("reuseBuffers", engine.ReuseBuffers);
        });
        for (int iteration = 0; iteration <= repeats; iteration++)
        {
            long allocatedBefore = GC.GetTotalAllocatedBytes(true);
            watch.Restart();
            float[] audio = engine.Run(inputs, 1).Values.First().Data;
            double seconds = watch.Elapsed.TotalSeconds;
            long allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
            string hash = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(audio.AsSpan())));
            Emit(writer =>
            {
                Measurement(writer, "tts", iteration, seconds, allocatedBytes);
                writer.WriteNumber("samples", audio.Length);
                writer.WriteNumber("audioSeconds", audio.Length / (aishell ? 8000.0 : 22050.0));
                writer.WriteString("waveSha256", hash);
            });
            if (iteration == repeats)
            {
                File.WriteAllBytes(args[5], MemoryMarshal.AsBytes(audio.AsSpan()).ToArray());
            }
        }
    }

    /// <summary>在推理停止计时后写入两种模型共用的测量字段。</summary>
    /// <param name="writer">本条记录的 JSON 写入器。</param>
    /// <param name="kind">模型类型。</param>
    /// <param name="iteration">从零开始的调用序号，零为首次预热。</param>
    /// <param name="seconds">模型调用耗时。</param>
    /// <param name="allocatedBytes">本次模型调用期间的托管分配量。</param>
    private static void Measurement(Utf8JsonWriter writer, string kind, int iteration, double seconds, long allocatedBytes)
    {
        writer.WriteString("kind", kind);
        writer.WriteNumber("iteration", iteration);
        writer.WriteBoolean("warmup", iteration == 0);
        writer.WriteNumber("seconds", seconds);
        writer.WriteNumber("allocatedBytes", allocatedBytes);
        writer.WriteNumber("peakRssBytes", Process.GetCurrentProcess().PeakWorkingSet64);
    }
}
