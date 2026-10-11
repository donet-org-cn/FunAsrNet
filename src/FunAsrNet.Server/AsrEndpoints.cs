using System.Buffers;
using System.Diagnostics;
using System.Net.WebSockets;
using System.Text.Json;

namespace FunAsrNet.Server;

/// <summary>
/// 文件识别与实时识别入口；连接状态隔离，模型计算由有界任务池调度。
/// </summary>
internal static class AsrEndpoints
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };
    /// <summary>
    /// 向应用注册对应的语音接口及处理函数。
    /// </summary>
    /// <param name="app">需要映射语音接口的 Web 应用。</param>
    /// <param name="runtime">已加载模型与并发工作池的服务运行状态。</param>
    internal static void Map(WebApplication app, SpeechRuntime runtime)
    {
        app.MapPost("/asr/wav", (Func<HttpContext, Task<IResult>>)(context => UploadAsync(context, runtime)));
        app.Map("/asr/stream", context => StreamAsync(context, runtime, app.Logger));
    }

    /// <summary>
    /// 逐块读取上传 WAV，通过有界工作池转写并返回计时信息。
    /// </summary>
    /// <param name="context">当前 HTTP 请求或独立推理上下文。</param>
    /// <param name="runtime">已加载模型与并发工作池的服务运行状态。</param>
    /// <returns>包含成功结果或明确错误状态码的 HTTP 响应。</returns>
    private static async Task<IResult> UploadAsync(HttpContext context, SpeechRuntime runtime)
    {
        if (!await runtime.Uploads.WaitAsync(0, context.RequestAborted))
        {
            return Results.Problem("上传处理已满，请稍后重试。", statusCode: 429);
        }

        try
        {
            if (context.Request.ContentLength > SpeechOptions.MaxUploadBytes)
            {
                return Results.StatusCode(413);
            }

            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);
            timeout.CancelAfter(TimeSpan.FromMinutes(2));
            var audio = await AudioFile.ReadWaveAsync(context.Request.Body, SpeechOptions.MaxAudioSeconds, SpeechOptions.MaxUploadBytes, timeout.Token);
            if (audio.Samples.Length < audio.Rate / 10)
            {
                return Results.BadRequest("音频至少需要 0.1 秒。");
            }

            float[] pcm = AudioFile.Resample(audio.Samples, audio.Rate, 16000);
            var timer = Stopwatch.StartNew();
            var result = await runtime.Asr.RunAsync(engine => engine.Transcribe(pcm, 512), context.RequestAborted);
            return Results.Json(new
            {
                result.Text,
                result.AudioSeconds,
                result.EncoderSeconds,
                result.DecoderSeconds,
                inferenceSeconds = result.TotalSeconds,
                result.RealTimeFactor,
                requestSeconds = timer.Elapsed.TotalSeconds

            });
        }
        catch (InferenceQueueFullException exception)
        {
            return Results.Problem(exception.Message, statusCode: 429);
        }
        catch (BadHttpRequestException exception)
        {
            return Results.Problem(exception.Message, statusCode: exception.StatusCode);
        }
        catch (Exception exception) when (exception is InvalidDataException or EndOfStreamException or NotSupportedException or ArgumentException or OverflowException)
        {
            return Results.BadRequest(exception.Message);
        }
        catch (OperationCanceledException) when (!context.RequestAborted.IsCancellationRequested)
        {
            return Results.Problem("上传超时。", statusCode: 408);
        }
        finally
        {
            runtime.Uploads.Release();
        }
    }

    /// <summary>
    /// 接收独立 WebSocket 会话的音频帧，驱动 VAD 与句段识别。
    /// </summary>
    /// <param name="context">当前 HTTP 请求或独立推理上下文。</param>
    /// <param name="runtime">已加载模型与并发工作池的服务运行状态。</param>
    /// <param name="logger">用于记录会话异常和运行状态的日志对象。</param>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    private static async Task StreamAsync(HttpContext context, SpeechRuntime runtime, ILogger logger)
    {
        if (!context.WebSockets.IsWebSocketRequest)
        {
            context.Response.StatusCode = 400;
            await context.Response.WriteAsync("使用 WebSocket 发送单声道 PCM/G.711；文本 {\"event\":\"end\"} 完成转写。");
            return;
        }

        if (!await runtime.Sessions.WaitAsync(0, context.RequestAborted))
        {
            context.Response.StatusCode = 429;
            return;
        }

        WebSocket? socket = null;
        byte[]? input = null;
        float[]? decoded = null;
        try
        {
            string codecText = context.Request.Query["codec"].FirstOrDefault() ?? "pcma";
            if (!Enum.TryParse<AudioCodec>(codecText, true, out var codec))
            {
                throw new ArgumentException("codec 支持 pcma、pcmu、pcm16。");
            }

            int rate = int.Parse(context.Request.Query["rate"].FirstOrDefault() ?? "8000");
            bool partials = bool.Parse(context.Request.Query["partials"].FirstOrDefault() ?? "false");
            int silence = int.Parse(context.Request.Query["silenceMs"].FirstOrDefault() ?? "1200");
            if (silence is < 300 or > 2500)
            {
                throw new ArgumentException("silenceMs 范围为 300–2500。");
            }

            var converter = new PhoneAudioDecoder(codec, rate);
            input = ArrayPool<byte>.Shared.Rent(4096);
            decoded = ArrayPool<float>.Shared.Rent(converter.GetMaxOutput(4096));
            socket = await context.WebSockets.AcceptWebSocketAsync();
            var outgoing = new ArrayBufferWriter<byte>(1024);
            // <summary>
            // 将一个事件直接序列化到复用缓冲区，并发送为单条 WebSocket 文本消息。
            // </summary>
            // <param name="message">需要发送的会话事件。</param>
            // <returns>表示操作完成、取消或失败的异步任务。</returns>
            async Task SendAsync(object message)
            {
                outgoing.Clear();
                using (var writer = new Utf8JsonWriter(outgoing))
                {
                    JsonSerializer.Serialize(writer, message, Json);
                }

                await socket.SendAsync(outgoing.WrittenMemory, WebSocketMessageType.Text, true, context.RequestAborted);
            }

            await SendAsync(new
            {
                @event = "ready",
                codec = codecText,
                rate

            });
            await using var stream = new PhoneStream(
                (pcm, token) => runtime.Asr.RunAsync(engine => engine.Transcribe(pcm.Span), token),
                runtime.Vad,
                message => SendAsync(message),
                partials,
                silence,
                context.RequestAborted);
            using var idle = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);
            while (socket.State == WebSocketState.Open)
            {
                idle.CancelAfter(TimeSpan.FromSeconds(runtime.Options.ReceiveTimeoutSeconds));
                var message = await socket.ReceiveAsync(input.AsMemory(0, 4096), idle.Token);
                if (message.MessageType == WebSocketMessageType.Close)
                {
                    break;
                }

                if (message.MessageType == WebSocketMessageType.Text)
                {
                    if (!message.EndOfMessage)
                    {
                        throw new ArgumentException("控制消息必须完整放在一帧中。");
                    }

                    using var control = JsonDocument.Parse(input.AsMemory(0, message.Count));
                    if (!control.RootElement.TryGetProperty("event", out var eventName) || eventName.GetString() != "end")
                    {
                        throw new ArgumentException("未知的控制消息。");
                    }

                    int count = converter.Decode([], decoded, true);
                    stream.Feed(decoded.AsSpan(0, count));
                    await stream.CompleteAsync();
                    break;
                }

                int samples = converter.Decode(input.AsSpan(0, message.Count), decoded);
                // Feed 同步消费借用视图；异步识别只持有单独租用的句段快照。
                stream.Feed(decoded.AsSpan(0, samples));
            }

            await CloseAsync(socket, WebSocketCloseStatus.NormalClosure, "finished");
        }
        catch (Exception exception) when (exception is ArgumentException or FormatException or OverflowException or InvalidDataException or InvalidOperationException or JsonException or WebSocketException or OperationCanceledException)
        {
            logger.LogWarning("实时转写结束：{Reason}", exception.Message);
            if (socket is null && !context.RequestAborted.IsCancellationRequested)
            {
                context.Response.StatusCode = 400;
                await context.Response.WriteAsync(exception.Message);
            }
            else if (socket is not null)
            {
                await CloseAsync(socket, WebSocketCloseStatus.PolicyViolation, "invalid audio, timeout or queue full");
            }
        }
        finally
        {
            socket?.Dispose();
            if (input is not null)
            {
                ArrayPool<byte>.Shared.Return(input);
            }

            if (decoded is not null)
            {
                ArrayPool<float>.Shared.Return(decoded);
            }

            runtime.Sessions.Release();
        }
    }

    /// <summary>
    /// 在限定时间内关闭 WebSocket，超时后中止连接以回收会话配额。
    /// </summary>
    /// <param name="socket">当前会话拥有的 WebSocket 连接。</param>
    /// <param name="status">发送给对端的 WebSocket 关闭状态。</param>
    /// <param name="description">会话关闭原因或检查项目说明。</param>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    private static async Task CloseAsync(WebSocket socket, WebSocketCloseStatus status, string description)
    {
        if (socket.State is not (WebSocketState.Open or WebSocketState.CloseReceived))
        {
            return;
        }

        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        try
        {
            await socket.CloseAsync(status, description, timeout.Token);
        }
        catch (Exception exception) when (exception is OperationCanceledException or WebSocketException)
        {
            socket.Abort();
        }
    }
}
