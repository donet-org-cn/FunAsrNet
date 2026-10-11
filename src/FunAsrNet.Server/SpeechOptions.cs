namespace FunAsrNet.Server;

/// <summary>
/// 配置模型位置、计算线程和有界并发容量；增加工作上下文不会复制模型权重。
/// </summary>
public sealed class SpeechOptions
{
    /// <summary>
    /// 模型文件目录。
    /// </summary>
    public string Models { get; set; } = "models";
    /// <summary>
    /// 中文发音词典的 JSON 路径。
    /// </summary>
    public string Lexicon { get; set; } = "assets/mandarin-lexicon.json";
    /// <summary>
    /// 每个计算上下文的最大算子线程数。
    /// </summary>
    public int Threads { get; set; } = 4;
    /// <summary>
    /// 独立 ASR 解码上下文数量；RK3588 默认一个，多个会话在句段边界交错调度。
    /// </summary>
    public int AsrWorkers { get; set; } = 1;
    /// <summary>
    /// 独立 TTS 执行上下文数量。
    /// </summary>
    public int TtsWorkers { get; set; } = 1;
    /// <summary>
    /// 每种推理任务的排队上限。
    /// </summary>
    public int QueueCapacity { get; set; } = 8;
    /// <summary>
    /// 允许同时接收音频的 WebSocket 会话数。
    /// </summary>
    public int MaxSessions { get; set; } = 4;
    /// <summary>
    /// 允许同时读取或处理的 WAV 上传数。
    /// </summary>
    public int MaxUploads { get; set; } = 2;
    /// <summary>
    /// 允许同时生成的 TTS HTTP 请求数，包含正在等待网络发送的请求。
    /// </summary>
    public int MaxTtsRequests { get; set; } = 4;
    /// <summary>
    /// 超过此秒数未收到音频或控制消息时关闭会话。
    /// </summary>
    public int ReceiveTimeoutSeconds { get; set; } = 120;

    /// <summary>
    /// 上传请求体和 RIFF 文件长度上限，单位为字节。
    /// </summary>
    public const int MaxUploadBytes = 32 * 1024 * 1024;
    /// <summary>
    /// 单个上传录音的最长秒数。
    /// </summary>
    public const int MaxAudioSeconds = 60;
    /// <summary>
    /// 读取配置，并兼容原部署使用的 models、lexicon 和 threads 命令行参数。
    /// </summary>
    /// <param name="configuration">应用合并后的配置。</param>
    /// <returns>已验证有效范围的服务配置。</returns>
    public static SpeechOptions Load(IConfiguration configuration)
    {
        var options = configuration.GetSection("Speech").Get<SpeechOptions>() ?? new();
        options.Models = configuration["models"] ?? options.Models;
        options.Lexicon = configuration["lexicon"] ?? options.Lexicon;
        options.Threads = configuration.GetValue<int?>("threads") ?? options.Threads;
        if (options.Threads is < 1 or > 64
            || options.AsrWorkers is < 1 or > 4
            || options.TtsWorkers is < 1 or > 4
            || options.QueueCapacity is < 1 or > 64
            || options.MaxSessions is < 1 or > 32
            || options.MaxUploads is < 1 or > 8
            || options.MaxTtsRequests is < 1 or > 16
            || options.ReceiveTimeoutSeconds is < 5 or > 3600)
        {
            throw new ArgumentException("推理线程、工作上下文、队列容量或会话数量超出配置范围。");
        }

        return options;
    }
}
