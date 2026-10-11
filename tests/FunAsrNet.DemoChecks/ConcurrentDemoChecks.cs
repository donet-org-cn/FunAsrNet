using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using FunAsrNet;

/// <summary>
/// 通过实际 HTTP 和 WebSocket 验证多会话隔离、限流以及合成参数。
/// </summary>
internal static class ConcurrentDemoChecks
{
    /// <summary>
    /// 使用长短不同的录音并发转写，并检查发音人和音量参数。
    /// </summary>
    /// <param name="http">已配置基础地址与证书验证的 HTTP 客户端。</param>
    /// <param name="url">服务的 HTTP 或 HTTPS 根地址。</param>
    /// <param name="fingerprint">受信任服务证书的 SHA-256 指纹；HTTP 请求可以为空。</param>
    /// <param name="audio">单声道浮点音频采样。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    internal static async Task RunAsync(
        HttpClient http,
        string url,
        string? fingerprint,
        float[] audio,
        CancellationToken cancellationToken)
    {
        using var metadata = JsonDocument.Parse(await http.GetStringAsync("/tts/voices", cancellationToken));
        var voices = metadata.RootElement.GetProperty("voices");
        Check(voices.GetArrayLength() >= 1 && voices[0].GetProperty("speakerIds").GetArrayLength() == 1, "保留华严单发音人元数据");
        foreach (var request in new object[]
            {
                new
                {
                    text = "你好",
                    speakerId = 1

                },
                new
                {
                    text = "你好",
                    voiceId = "unknown"

                },
                new
                {
                    text = "你好",
                    noiseScale = 3

                },
                new
                {
                    text = "你好",
                    volume = -1

                }
            }
        )
        {
            using var response = await http.PostAsJsonAsync("/tts", request, cancellationToken);
            Check(response.StatusCode == HttpStatusCode.BadRequest, "拒绝不支持的发音人或参数");
        }

        using (var response = await http.PostAsJsonAsync(
            "/tts",
            new
            {
                text = "你好",
                volume = 0,
                sampleRate = 8000,
                format = "wav"

            },
            cancellationToken))
        {
            response.EnsureSuccessStatusCode();
            await using var body = await response.Content.ReadAsStreamAsync(cancellationToken);
            var silent = await AudioFile.ReadWaveAsync(body, cancellationToken: cancellationToken);
            Check(silent.Samples.Length > 0 && silent.Samples.All(value => value == 0), "音量为零生成静音 WAV");
        }

        byte[] first = await SynthesizeAsync(http, cancellationToken);
        byte[] second = await SynthesizeAsync(http, cancellationToken);
        Check(first.AsSpan().SequenceEqual(second), "相同种子和参数生成相同 PCM");
        // 两个连接均收到 ready 后才开始发送，确保实际同时占用独立会话。
        using var a = await ConnectAsync(url, fingerprint, cancellationToken);
        using var b = await ConnectAsync(url, fingerprint, cancellationToken);
        Task<string> shortResult = TranscribeAsync(a, audio.AsMemory(0, 48000), cancellationToken);
        Task<string> longResult = TranscribeAsync(b, audio, cancellationToken);
        string[] results = await Task.WhenAll(shortResult, longResult);
        Check(results[0].Contains("我想") && !results[0].Contains("滨海新区"), "短会话没有混入长会话文本");
        Check(results[1].Contains("滨海新区"), "长会话完整识别");
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            concurrentTexts = results

        }));
        using var health = JsonDocument.Parse(await http.GetStringAsync("/health", cancellationToken));
        int limit = health.RootElement.GetProperty("maxSessions").GetInt32();
        List<ClientWebSocket> held = [];
        try
        {
            for (int i = 0; i < limit; ++i)
            {
                held.Add(await ConnectAsync(url, fingerprint, cancellationToken));
            }

            bool rejected = false;
            try
            {
                using var extra = await ConnectAsync(url, fingerprint, cancellationToken);
            }
            catch (WebSocketException)
            {
                rejected = true;
            }

            Check(rejected, "会话数达到上限时拒绝额外连接");
        }
        finally
        {
            foreach (var socket in held)
            {
                try
                {
                    await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "finished", cancellationToken);
                }
                finally
                {
                    socket.Dispose();
                }
            }
        }
    }

    /// <summary>
    /// 请求固定参数的短句，返回原始 WAV 内容用于确定性比较。
    /// </summary>
    /// <param name="http">已配置基础地址与证书验证的 HTTP 客户端。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>完整 WAV 字节。</returns>
    private static async Task<byte[]> SynthesizeAsync(HttpClient http, CancellationToken cancellationToken)
    {
        using var response = await http.PostAsJsonAsync(
            "/tts",
            new
            {
                text = "你好",
                seed = 7,
                noiseScale = 0.4,
                durationNoise = 0.6,
                sampleRate = 22050,
                format = "wav"

            },
            cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsByteArrayAsync(cancellationToken);
    }

    /// <summary>
    /// 连接转写服务并消费 ready 消息，失败时立即释放套接字。
    /// </summary>
    /// <param name="url">服务的 HTTP 或 HTTPS 根地址。</param>
    /// <param name="fingerprint">受信任服务证书的 SHA-256 指纹；HTTP 请求可以为空。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>已就绪且由调用者负责释放的连接。</returns>
    private static async Task<ClientWebSocket> ConnectAsync(string url, string? fingerprint, CancellationToken cancellationToken)
    {
        var socket = new ClientWebSocket();
        try
        {
            if (fingerprint is not null)
            {
                socket.Options.RemoteCertificateValidationCallback = (_, certificate, _, _) => certificate?.GetCertHashString(HashAlgorithmName.SHA256).Equals(fingerprint, StringComparison.OrdinalIgnoreCase) == true;
            }

            var target = new UriBuilder(url)
            {
                Scheme = new Uri(url).Scheme == "https" ? "wss" : "ws",
                Path = "/asr/stream",
                Query = "codec=pcm16&rate=16000&partials=false"

            };
            await socket.ConnectAsync(target.Uri, cancellationToken);
            var buffer = new byte[4096];
            var ready = await socket.ReceiveAsync(buffer, cancellationToken);
            using var json = JsonDocument.Parse(buffer.AsMemory(0, ready.Count));
            Check(json.RootElement.GetProperty("event").GetString() == "ready", "会话独立就绪");
            return socket;
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    /// <summary>
    /// 发送完整录音并读取最终事件，短音频与长音频可并行运行。
    /// </summary>
    /// <param name="socket">当前会话拥有的 WebSocket 连接。</param>
    /// <param name="audio">单声道浮点音频采样。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>该会话所有最终句段的连接文本。</returns>
    private static async Task<string> TranscribeAsync(ClientWebSocket socket, ReadOnlyMemory<float> audio, CancellationToken cancellationToken)
    {
        short[] pcm = new short[audio.Length];
        AudioKernels.FloatToPcm16(audio.Span, pcm);
        byte[] bytes = MemoryMarshal.AsBytes(pcm.AsSpan()).ToArray();
        for (int i = 0; i < bytes.Length; i += 640)
        {
            await socket.SendAsync(bytes.AsMemory(i, Math.Min(640, bytes.Length - i)), WebSocketMessageType.Binary, true, cancellationToken);
            await Task.Delay(1, cancellationToken);
        }

        await socket.SendAsync("{\"event\":\"end\"}"u8.ToArray(), WebSocketMessageType.Text, true, cancellationToken);
        var text = new StringBuilder();
        var buffer = new byte[16384];
        bool done = false;
        while (true)
        {
            using var message = new MemoryStream();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(buffer, cancellationToken);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "done", cancellationToken);
                    Check(done, "会话收到完成事件");
                    return text.ToString();
                }

                message.Write(buffer, 0, result.Count);
            }
            while (!result.EndOfMessage);
            using var json = JsonDocument.Parse(message.ToArray());
            string? kind = json.RootElement.GetProperty("event").GetString();
            if (kind == "final")
            {
                text.Append(json.RootElement.GetProperty("text").GetString());
            }

            done |= kind == "done";
        }
    }

    /// <summary>
    /// 输出检查结果，失败时中断测试并保留具体条件。
    /// </summary>
    /// <param name="success">检查条件是否成立。</param>
    /// <param name="description">会话关闭原因或检查项目说明。</param>
    private static void Check(bool success, string description)
    {
        if (!success)
        {
            throw new InvalidOperationException(description);
        }

        Console.WriteLine($"PASS {description}");
    }
}
