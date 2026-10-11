using System.Diagnostics;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

// <summary>
// 读取命令行选项，缺失时使用默认值。
// </summary>
// <param name="key">GGUF 元数据键或命令行选项名。</param>
// <param name="fallback">键或属性不存在时采用的默认值。</param>
// <returns>完成处理后的文本。</returns>
string Option(string key, string fallback)
{
    int i = Array.IndexOf(args, key);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
}

string url = Option("--url", "http://127.0.0.1:18080");
string output = Option("--out", "probe-results");
string audioPath = Option("--audio", "");
if (string.IsNullOrWhiteSpace(audioPath) || !File.Exists(audioPath))
{
    throw new ArgumentException("请使用 --audio 指定有效的 8 kHz 单声道裸 PCMA 文件。");
}
Directory.CreateDirectory(output);
using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(3));
using var http = new HttpClient
{
    BaseAddress = new Uri(url),
    Timeout = TimeSpan.FromMinutes(2)

};
Console.WriteLine(await http.GetStringAsync("/health", timeout.Token));
using (var invalid = await http.PostAsJsonAsync("/tts", new
{
    text = "你好",
    sampleRate = 16000,
    format = "pcma"

}, timeout.Token))
{
    if (invalid.StatusCode != System.Net.HttpStatusCode.BadRequest)
    {
        throw new Exception("Invalid G.711 sample rate must be rejected");
    }
}

using (var invalid = await http.PostAsJsonAsync("/tts/stream", new
{
    text = "😀",
    sampleRate = 8000,
    format = "pcma"

}, timeout.Token))
{
    if (invalid.StatusCode != System.Net.HttpStatusCode.BadRequest)
    {
        throw new Exception("Unsupported pronunciation must return 400 before streaming");
    }
}

var stopwatch = Stopwatch.StartNew();
using (var response = await http.PostAsJsonAsync("/tts", new
{
    text = "您好，欢迎致电，请问有什么可以帮您？",
    sampleRate = 8000,
    format = "wav"

}, timeout.Token))
{
    response.EnsureSuccessStatusCode();
    var bytes = await response.Content.ReadAsByteArrayAsync(timeout.Token);
    if (Encoding.ASCII.GetString(bytes, 0, 4) != "RIFF" || BitConverter.ToInt32(bytes, 24) != 8000)
    {
        throw new Exception("Invalid TTS WAV");
    }

    await File.WriteAllBytesAsync(Path.Combine(output, "http-tts.wav"), bytes, timeout.Token);
    Console.WriteLine(JsonSerializer.Serialize(new
    {
        @event = "tts",
        seconds = stopwatch.Elapsed.TotalSeconds,
        bytes = bytes.Length,
        inference = response.Headers.GetValues("X-Inference-Seconds").Single()

    }));
}

stopwatch.Restart();
using (var message = new HttpRequestMessage(HttpMethod.Post, "/tts/stream")
{
    Content = JsonContent.Create(new
    {
        text = "您好，欢迎致电，请问有什么可以帮您？",
        sampleRate = 8000,
        format = "pcma"
    })
}
)
{
    using (var response = await http.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, timeout.Token))
    {
        response.EnsureSuccessStatusCode();
        await using var input = await response.Content.ReadAsStreamAsync(timeout.Token);
        await using var file = File.Create(Path.Combine(output, "http-stream.pcma"));
        byte[] buffer = new byte[2048];
        double? first = null;
        int total = 0;
        while (true)
        {
            int n = await input.ReadAsync(buffer, timeout.Token);
            if (n == 0)
            {
                break;
            }

            first ??= stopwatch.Elapsed.TotalSeconds;
            await file.WriteAsync(buffer.AsMemory(0, n), timeout.Token);
            total += n;
        }

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            @event = "tts-stream",
            firstAudioSeconds = first,
            totalSeconds = stopwatch.Elapsed.TotalSeconds,
            audioSeconds = total / 8000.0

        }));
    }
}

using var socket = new ClientWebSocket();
var target = new UriBuilder(url)
{
    Scheme = new Uri(url).Scheme == "https" ? "wss" : "ws",
    Path = "/asr/stream",
    Query = "codec=pcma&rate=8000&partials=true"

};
await socket.ConnectAsync(target.Uri, timeout.Token);
var ready = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
var events = new List<JsonElement>();
var receive = Task.Run(
    async () =>
    {
        var bytes = new byte[8192];
        while (true)
        {
            using var memory = new MemoryStream();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(bytes, timeout.Token);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "ok", timeout.Token);
                    return;
                }

                memory.Write(bytes, 0, result.Count);
            }
            while (!result.EndOfMessage);
            using var document = JsonDocument.Parse(memory.ToArray());
            var e = document.RootElement.Clone();
            events.Add(e);
            Console.WriteLine(e.GetRawText());
            if (e.GetProperty("event").GetString() == "ready")
            {
                ready.SetResult();
            }
        }
    },
    timeout.Token);
await ready.Task.WaitAsync(timeout.Token);
// 第二条连接拥有独立会话状态，句段通过有界工作池调度。
using (var second = new ClientWebSocket())
{
    await second.ConnectAsync(target.Uri, timeout.Token);
    await second.CloseAsync(WebSocketCloseStatus.NormalClosure, "check-complete", timeout.Token);
}

byte[] audio = await File.ReadAllBytesAsync(audioPath, timeout.Token);
stopwatch.Restart();
for (int i = 0; i < audio.Length; i += 160)
{
    int n = Math.Min(160, audio.Length - i);
    double delay = (i + n) / 8000.0 - stopwatch.Elapsed.TotalSeconds;
    if (delay > 0)
    {
        await Task.Delay(TimeSpan.FromSeconds(delay), timeout.Token);
    }

    await socket.SendAsync(audio.AsMemory(i, n), WebSocketMessageType.Binary, true, timeout.Token);
}

double eof = stopwatch.Elapsed.TotalSeconds;
await socket.SendAsync(Encoding.UTF8.GetBytes("{\"event\":\"end\"}"), WebSocketMessageType.Text, true, timeout.Token);
await receive;
if (!events.Any(e => e.GetProperty("event").GetString() == "final") || !events.Any(e => e.GetProperty("event").GetString() == "done"))
{
    throw new Exception("Stream did not produce final + done");
}

await File.WriteAllTextAsync(Path.Combine(output, "events.json"), JsonSerializer.Serialize(events), timeout.Token);
Console.WriteLine(JsonSerializer.Serialize(new
{
    @event = "probe-passed",
    sentAudioSeconds = audio.Length / 8000.0,
    afterEofSeconds = stopwatch.Elapsed.TotalSeconds - eof
}));
