using System.Diagnostics;
using System.Runtime.Intrinsics.Arm;
using System.Text.Encodings.Web;

namespace FunAsrNet.Server;

/// <summary>
/// FunAsrNet HTTP、WebSocket 与浏览器示例宿主。
/// </summary>
public static class Program
{
    /// <summary>
    /// 加载并预热模型，启动示例和语音接口。
    /// </summary>
    /// <param name="args">服务命令行参数。</param>
    /// <returns>服务退出后完成的任务。</returns>
    public static async Task Main(string[] args)
    {
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            Args = args,
            WebRootPath = Path.Combine(AppContext.BaseDirectory, "wwwroot")
        });
        DemoTls.Configure(builder);
        builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = SpeechOptions.MaxUploadBytes);
        builder.Services.ConfigureHttpJsonOptions(options => options.SerializerOptions.Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping);
        SpeechOptions configuration = SpeechOptions.Load(builder.Configuration);
        var timer = Stopwatch.StartNew();
        await using var runtime = new SpeechRuntime(configuration);
        var app = builder.Build();
        app.Logger.LogInformation("FunAsrNet 模型已就绪，加载及预热 {Seconds:F2} 秒，NEON={Neon}，SDOT={Sdot}", timer.Elapsed.TotalSeconds, AdvSimd.IsSupported, Dp.IsSupported);
        app.UseWebSockets(new WebSocketOptions { KeepAliveInterval = TimeSpan.FromSeconds(20) });
        app.UseDefaultFiles();
        app.UseStaticFiles();
        app.MapGet(
            "/health",
            () => Results.Json(new
            {
                status = "ready",
                product = "FunAsrNet",
                version = "1.0.0",
                runtime = Environment.Version.ToString(),
                engine = "pure-csharp",
                neon = AdvSimd.IsSupported,
                sdot = Dp.IsSupported,
                threads = configuration.Threads,
                asrBusy = runtime.Asr.ActiveCount + runtime.Asr.QueuedCount > 0,
                ttsBusy = runtime.Tts.ActiveCount + runtime.Tts.QueuedCount > 0,
                activeSessions = configuration.MaxSessions - runtime.Sessions.CurrentCount,
                maxSessions = configuration.MaxSessions,
                asrWorkers = runtime.Asr.WorkerCount,
                asrQueued = runtime.Asr.QueuedCount,
                ttsWorkers = runtime.Tts.WorkerCount,
                ttsQueued = runtime.Tts.QueuedCount
            }));
        app.MapGet(
            "/demo/config",
            () => Results.Json(new
            {
                httpsPort = builder.Configuration.GetValue<int?>("demoHttpsPort"),
                maxUploadBytes = SpeechOptions.MaxUploadBytes,
                maxAudioSeconds = SpeechOptions.MaxAudioSeconds,
                maxSessions = configuration.MaxSessions
            }));
        AsrEndpoints.Map(app, runtime);
        TtsEndpoints.Map(app, runtime);
        await app.RunAsync();
    }
}
