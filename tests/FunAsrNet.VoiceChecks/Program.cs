using System.Diagnostics;
using System.Net.Http.Headers;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using FunAsrNet;

Console.OutputEncoding = Encoding.UTF8;
if (args.Length < 2)
{
    throw new ArgumentException("用法：VoiceChecks AISHELL3目录 输出目录 [ASR服务地址]");
}

string directory = args[0], outputDirectory = args[1];
Directory.CreateDirectory(outputDirectory);
var model = new OnnxModel(Path.Combine(directory, "model.onnx"));
var frontend = new IcefallFrontend(Path.Combine(directory, "lexicon.txt"), Path.Combine(directory, "tokens.txt"), model.Metadata);
var engine = new MandarinTts(model, frontend, 4);
int checks = 0;
// <summary>
// 断言模型或音频满足预期，否则终止验证。
// </summary>
// <param name="condition">必须成立的检查条件。</param>
// <param name="description">失败时显示的中文说明。</param>
void Check(bool condition, string description)
{
    checks++;
    if (!condition)
    {
        throw new InvalidDataException(description);
    }
}

Check(engine.SpeakerCount == 174 && engine.NativeSampleRate == 8000, "模型发音人数与采样率");
Check(frontend.Encode("１２３").SequenceEqual(frontend.Encode("一二三")), "全角数字归一化");
Check(frontend.Encode("123").SequenceEqual(frontend.Encode("一二三")), "电话数字逐位朗读");
Check(frontend.Encode("您好，欢迎致电。").SequenceEqual(frontend.Encode("您好,欢迎致电.")), "中英文标点一致");
foreach (int invalid in new[] { -1, 174 })
{
    bool rejected = false;
    try
    {
        engine.Synthesize("您好", speakerId: invalid);
    }
    catch (ArgumentOutOfRangeException)
    {
        rejected = true;
    }

    Check(rejected, $"越界发音人 {invalid} 被拒绝");
}

const string text = "您好，欢迎致电，请问有什么可以帮您？";
int[] selected = [0, 10, 25, 50, 75, 100, 125, 150, 173];
var results = new List<object>();
var hashes = new HashSet<string>(StringComparer.Ordinal);
var times = new List<double>();
float[]? reference = null;
using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
for (int speaker = 0; speaker < engine.SpeakerCount; speaker++)
{
    var timer = Stopwatch.StartNew();
    float[] audio = engine.Synthesize(text, speakerId: speaker);
    double seconds = timer.Elapsed.TotalSeconds, duration = audio.Length / 8000.0;
    Check(duration is > 0.2 and < 30 && audio.All(float.IsFinite), $"发音人 {speaker} 输出长度及有限性");
    Check(audio.Any(sample => Math.Abs(sample) > 0.001f), $"发音人 {speaker} 不是静音");
    string hash = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(audio.AsSpan())));
    Check(hashes.Add(hash), $"发音人 {speaker} 输出并非另一编号的重复音频");
    if (speaker == 50)
    {
        reference = audio;
    }

    string? transcript = null;
    if (selected.Contains(speaker))
    {
        string path = Path.Combine(outputDirectory, $"speaker-{speaker:D3}.wav");
        AudioFile.WriteWave(path, audio, 8000);
        if (args.Length > 2)
        {
            await using var stream = File.OpenRead(path);
            using var content = new StreamContent(stream);
            content.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
            using var response = await client.PostAsync(new Uri(new Uri(args[2]), "/asr/wav"), content);
            response.EnsureSuccessStatusCode();
            using var json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            transcript = json.RootElement.GetProperty("text").GetString();
        }
    }

    times.Add(seconds);
    var result = new
    {
        speakerId = speaker,
        seconds,
        audioSeconds = duration,
        rtf = seconds / duration,
        sha256 = hash,
        transcript
    };
    results.Add(result);
    Console.WriteLine(JsonSerializer.Serialize(result, new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
}

Check(reference!.SequenceEqual(engine.Synthesize(text, speakerId: 50)), "切换全部发音人后种子与状态可重复");
await using (var pool = new BoundedInferencePool<MandarinTts>([engine, new MandarinTts(model, frontend, 4)], 4))
{
    var outputs = await Task.WhenAll(
        pool.RunAsync(context => context.Synthesize(text, speakerId: 50), CancellationToken.None),
        pool.RunAsync(context => context.Synthesize(text, speakerId: 50), CancellationToken.None));
    Check(outputs.All(output => output.SequenceEqual(reference)), "两个共享权重上下文并发时状态隔离");
}

times.Sort();
var summary = new
{
    checks,
    speakers = engine.SpeakerCount,
    uniqueOutputs = hashes.Count,
    text,
    medianSeconds = times[times.Count / 2],
    peakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64,
    runtime = Environment.Version.ToString(),
    architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    results
};
await File.WriteAllTextAsync(Path.Combine(outputDirectory, "results.json"), JsonSerializer.Serialize(summary, new JsonSerializerOptions
{
    WriteIndented = true,
    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
}));
Console.WriteLine($"PASS {checks} checks; {hashes.Count} 个独立音色输出；合成耗时中位数 {times[times.Count / 2]:F3} 秒");
