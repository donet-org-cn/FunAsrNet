using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using FunAsrNet;

Console.OutputEncoding = Encoding.UTF8;
string url = args.Length > 0 ? args[0] : "http://localhost:18080";
string? fingerprint = args.Length > 1 ? args[1] : null;
using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(4));
using var handler = new HttpClientHandler();
if (fingerprint is not null)
{
    handler.ServerCertificateCustomValidationCallback = (_, certificate, _, _) => certificate?.GetCertHashString(HashAlgorithmName.SHA256).Equals(fingerprint, StringComparison.OrdinalIgnoreCase) == true;
}

using var http = new HttpClient(handler)
{
    BaseAddress = new Uri(url),
    Timeout = TimeSpan.FromMinutes(2)

};
using (var initialHealth = JsonDocument.Parse(await http.GetStringAsync("/health", timeout.Token)))
{
    if (initialHealth.RootElement.GetProperty("asrBusy").GetBoolean() || initialHealth.RootElement.GetProperty("ttsBusy").GetBoolean())
    {
        Console.Error.WriteLine("Service is in use. Finish the active microphone/TTS session before running this probe; it will not interrupt it.");
        Environment.ExitCode = 75;
        return;
    }
}

// <summary>
// 断言测试条件成立，并输出可追踪的检查结果。
// </summary>
// <param name="pass">检查条件是否成立。</param>
// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
void Assert(bool pass, string name)
{
    if (!pass)
    {
        throw new Exception(name);
    }

    Console.WriteLine($"PASS {name}");
}

foreach (string path in new[]
    {
        "/",
        "/app.css",
        "/app.js",
        "/capture.js",
        "/health",
        "/demo/config"
    }
)
{
    using var response = await http.GetAsync(path, timeout.Token);
    Assert(response.IsSuccessStatusCode, $"GET {path}");
}

byte[] wave = await http.GetByteArrayAsync("/sample.wav", timeout.Token);
using var input = new MemoryStream(wave);
var audio = AudioFile.ReadWave(input);
Assert(audio.Rate == 16000 && audio.Samples.Length == 96000 && wave.Length == 192044, "Browser sample: canonical 16 kHz mono PCM16, 6 seconds");
using (var invalid = await http.PostAsync("/asr/wav", new ByteArrayContent([1, 2, 3]), timeout.Token))
{
    Assert(invalid.StatusCode == HttpStatusCode.BadRequest, "Invalid WAV returns 400");
}

// 大文件先等待服务端确认请求头，避免 TLS 在客户端仍写入正文时因提前拒绝而中止。
using (var oversized = new HttpRequestMessage(HttpMethod.Post, "/asr/wav") { Content = new ByteArrayContent(new byte[32 * 1024 * 1024 + 1]) })
{
    oversized.Headers.ExpectContinue = true;
    using var invalid = await http.SendAsync(oversized, timeout.Token);
    Assert(invalid.StatusCode == HttpStatusCode.RequestEntityTooLarge, "Oversized WAV returns 413");
}

using (var response = await http.PostAsync("/asr/wav", new ByteArrayContent(wave), timeout.Token))
{
    response.EnsureSuccessStatusCode();
    var text = await response.Content.ReadAsStringAsync(timeout.Token);
    Console.WriteLine(text);
    using var result = JsonDocument.Parse(text);
    Assert(result.RootElement.GetProperty("text").GetString()!.Contains("滨海新区"), "WAV upload recognized sample");
}

using (var response = await http.PostAsJsonAsync("/tts", new
{
    text = "您好，欢迎致电。",
    sampleRate = 22050,
    format = "wav"

}, timeout.Token))
{
    response.EnsureSuccessStatusCode();
    byte[] bytes = await response.Content.ReadAsByteArrayAsync(timeout.Token);
    using var output = new MemoryStream(bytes);
    var speech = AudioFile.ReadWave(output);
    Assert(speech.Rate == 22050 && speech.Samples.Length > 22050, "TTS generates playable WAV");
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        ttsSeconds = response.Headers.GetValues("X-Inference-Seconds").Single(),
        audioSeconds = speech.Samples.Length / 22050.0,
        bytes = bytes.Length

    }));
}

// 以实时速度验证浏览器 48 kHz 输入路径，并验证上传与会话可以并存。
using var socket = new ClientWebSocket();
if (fingerprint is not null)
{
    socket.Options.RemoteCertificateValidationCallback = (_, certificate, _, _) => certificate?.GetCertHashString(HashAlgorithmName.SHA256).Equals(fingerprint, StringComparison.OrdinalIgnoreCase) == true;
}

var target = new UriBuilder(url)
{
    Scheme = new Uri(url).Scheme == "https" ? "wss" : "ws",
    Path = "/asr/stream",
    Query = "codec=pcm16&rate=48000&partials=true&silenceMs=900"

};
await socket.ConnectAsync(target.Uri, timeout.Token);
var ready = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
var events = new List<JsonElement>();
var receive = Task.Run(
    async () =>
    {
        byte[] buffer = new byte[8192];
        while (true)
        {
            using var memory = new MemoryStream();
            WebSocketReceiveResult message;
            do
            {
                message = await socket.ReceiveAsync(buffer, timeout.Token);
                if (message.MessageType == WebSocketMessageType.Close)
                {
                    await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "ok", timeout.Token);
                    return;
                }

                memory.Write(buffer, 0, message.Count);
            }
            while (!message.EndOfMessage);
            using var doc = JsonDocument.Parse(memory.ToArray());
            var item = doc.RootElement.Clone();
            events.Add(item);
            Console.WriteLine(item.GetRawText());
            if (item.GetProperty("event").GetString() == "ready")
            {
                ready.TrySetResult();
            }
        }
    },
    timeout.Token);
await ready.Task.WaitAsync(TimeSpan.FromSeconds(15), timeout.Token);
using (var concurrent = await http.PostAsync("/asr/wav", new ByteArrayContent(wave), timeout.Token))
{
    Assert(concurrent.IsSuccessStatusCode, "Upload coexists with a connected WebSocket session");
}

float[] resampled = AudioFile.Resample(audio.Samples, 16000, 48000);
var pcm = new short[resampled.Length];
AudioKernels.FloatToPcm16(resampled, pcm);
byte[] raw = System.Runtime.InteropServices.MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
var timer = Stopwatch.StartNew();
for (int i = 0; i < raw.Length; i += 1920)
{
    int n = Math.Min(1920, raw.Length - i);
    double delay = (i + n) / 96000.0 - timer.Elapsed.TotalSeconds;
    if (delay > 0)
    {
        await Task.Delay(TimeSpan.FromSeconds(delay), timeout.Token);
    }

    await socket.SendAsync(raw.AsMemory(i, n), WebSocketMessageType.Binary, true, timeout.Token);
}

double eof = timer.Elapsed.TotalSeconds;
await socket.SendAsync(Encoding.UTF8.GetBytes("{\"event\":\"end\"}"), WebSocketMessageType.Text, true, timeout.Token);
await receive;
string final = string.Join("", events.Where(e => e.GetProperty("event").GetString() == "final").Select(e => e.GetProperty("text").GetString()));
Assert(
    final.Contains("滨海新区") && events.Any(e => e.GetProperty("event").GetString() == "done"),
    "48 kHz browser audio produced final text and done");
Console.WriteLine(JsonSerializer.Serialize(new
{
    afterEofSeconds = timer.Elapsed.TotalSeconds - eof,
    final
}));
await ConcurrentDemoChecks.RunAsync(http, url, fingerprint, audio.Samples, timeout.Token);
await FunAsrNet.DemoChecks.MultivoiceDemoChecks.RunAsync(http, timeout.Token);
// 验证异常请求和会话结束后工作池与会话配额均恢复。
using var health = JsonDocument.Parse(await http.GetStringAsync("/health", timeout.Token));
Assert(!health.RootElement.GetProperty("asrBusy").GetBoolean(), "ASR gate released");
