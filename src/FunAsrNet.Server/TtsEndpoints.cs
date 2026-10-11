using System.Diagnostics;
using System.Globalization;

namespace FunAsrNet.Server;

/// <summary>
/// 中文 TTS 参数发现、完整音频和分块电话音频接口。
/// </summary>
internal static class TtsEndpoints
{
    /// <summary>
    /// 向应用注册对应的语音接口及处理函数。
    /// </summary>
    /// <param name="app">需要映射语音接口的 Web 应用。</param>
    /// <param name="runtime">已加载模型与并发工作池的服务运行状态。</param>
    internal static void Map(WebApplication app, SpeechRuntime runtime)
    {
        app.MapGet(
            "/tts/voices",
            () => Results.Json(new
            {
                voices = runtime.Voices,
                defaults = new
                {
                    voiceId = "huayan-medium",
                    speakerId = 0,
                    speed = 1,
                    seed = 1,
                    noiseScale = 0.667,
                    durationNoise = 0.8,
                    volume = 1
                },
                formats = new[] { "wav", "pcm16", "pcma", "pcmu" },
                sampleRates = new[] { 8000, 16000, 22050 },
                supportsPitch = false,
                supportsEmotion = false
            }));
        app.MapPost("/tts", (TtsRequest request, HttpContext context) => GenerateAsync(request, context, runtime, false));
        app.MapPost("/tts/stream", (TtsRequest request, HttpContext context) => GenerateAsync(request, context, runtime, true));
    }

    /// <summary>
    /// 验证合成参数，通过有界工作池生成音频并直接写入响应管道。
    /// </summary>
    /// <param name="request">经过模型能力和范围验证的合成请求。</param>
    /// <param name="context">当前 HTTP 请求或独立推理上下文。</param>
    /// <param name="runtime">已加载模型与并发工作池的服务运行状态。</param>
    /// <param name="streaming">是否按句段直接输出原始电话音频。</param>
    /// <returns>包含成功结果或明确错误状态码的 HTTP 响应。</returns>
    private static async Task<IResult> GenerateAsync(TtsRequest request, HttpContext context, SpeechRuntime runtime, bool streaming)
    {
        int nativeSampleRate;
        try
        {
            request.Validate(streaming);
            nativeSampleRate = runtime.ValidateText(request);
        }
        catch (ArgumentException exception)
        {
            return Results.BadRequest(exception.Message);
        }

        if (!await runtime.TtsRequests.WaitAsync(0, context.RequestAborted))
        {
            return Results.Problem("语音合成请求已满，请稍后重试。", statusCode: 429);
        }

        try
        {
            context.Response.ContentType = request.Format == "wav" ? "audio/wav" : "application/octet-stream";
            if (streaming)
            {
                context.Response.Headers["X-Audio-Sample-Rate"] = "8000";
                int index = 0;
                foreach (string text in request.SplitText())
                {
                    int seed = unchecked(request.Seed + index++) & int.MaxValue;
                    float[] samples = await runtime.Tts.RunAsync(engines => engines[request.VoiceId].Synthesize(text, request.Speed, seed, request.NoiseScale, request.DurationNoise, request.SpeakerId), context.RequestAborted);
                    samples = AudioFile.Resample(samples, nativeSampleRate, 8000);
                    await AudioResponse.WriteAsync(context.Response, samples, 8000, request.Format, request.Volume, false, context.RequestAborted);
                }
            }
            else
            {
                var timer = Stopwatch.StartNew();
                float[] audio = await runtime.Tts.RunAsync(engines => engines[request.VoiceId].Synthesize(request.Text, request.Speed, request.Seed, request.NoiseScale, request.DurationNoise, request.SpeakerId), context.RequestAborted);
                audio = AudioFile.Resample(audio, nativeSampleRate, request.SampleRate);
                context.Response.Headers["X-Inference-Seconds"] = timer.Elapsed.TotalSeconds.ToString("F6", CultureInfo.InvariantCulture);
                context.Response.Headers["X-Audio-Seconds"] = (audio.Length / (double)request.SampleRate).ToString("F6", CultureInfo.InvariantCulture);
                context.Response.ContentLength = audio.Length * (request.Format is "pcma" or "pcmu" ? 1L : 2L) + (request.Format == "wav" ? 44 : 0);
                await AudioResponse.WriteAsync(context.Response, audio, request.SampleRate, request.Format, request.Volume, request.Format == "wav", context.RequestAborted);
            }

            return Results.Empty;
        }
        catch (InferenceQueueFullException exception) when (!context.Response.HasStarted)
        {
            return Results.Problem(exception.Message, statusCode: 429);
        }
        catch (Exception exception) when (context.Response.HasStarted && exception is InferenceQueueFullException or IOException or OperationCanceledException)
        {
            context.Abort();
            return Results.Empty;
        }
        finally
        {
            runtime.TtsRequests.Release();
        }
    }
}
