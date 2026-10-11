namespace FunAsrNet.Server;

/// <summary>
/// 拥有共享模型权重、隔离的推理上下文及入口限流器。
/// </summary>
public sealed class SpeechRuntime : IAsyncDisposable
{
    private readonly Dictionary<string, MandarinTts> _textValidators;
    /// <summary>
    /// 加载模型并顺序预热全部上下文，之后才允许并发读取权重。
    /// </summary>
    /// <param name="options">经过范围校验的模型路径、线程和并发容量配置。</param>
    public SpeechRuntime(SpeechOptions options)
    {
        Options = options;
        var encoder = new Gguf(Path.Combine(options.Models, "funasr-encoder-f16.gguf"));
        var decoder = new Gguf(Path.Combine(options.Models, "qwen3-0.6b-q8_0.gguf"));
        var recognizers = new FunAsr[options.AsrWorkers];
        for (int i = 0; i < recognizers.Length; i++)
        {
            // 构造期间进行的权重布局转换必须串行完成，发布后只读共享。
            recognizers[i] = new FunAsr(encoder, decoder, options.Threads);
            recognizers[i].Transcribe(new float[32000], 1);
        }

        Vad = new Gguf(Path.Combine(options.Models, "fsmn-vad.gguf"));
        string ttsPath = Path.Combine(options.Models, "zh_CN-huayan-medium.onnx");
        if (!File.Exists(ttsPath))
        {
            ttsPath = Path.Combine(options.Models, "vits-piper-zh_CN-huayan-medium", "zh_CN-huayan-medium.onnx");
        }

        var ttsModel = new OnnxModel(ttsPath);
        var contexts = Enumerable.Range(0, options.TtsWorkers).Select(_ => new Dictionary<string, MandarinTts>(StringComparer.Ordinal)
        {
            ["huayan-medium"] = new MandarinTts(ttsModel, options.Lexicon, ttsPath + ".json", options.Threads)
        }).ToArray();
        var catalog = new List<TtsVoiceInfo>
        {
            new("huayan-medium", "华严中文", MandarinTts.SampleRate, [new(0, "华严 · 0 号", "huayan")])
        };
        string aishellDirectory = Path.Combine(options.Models, "aishell3");
        if (File.Exists(Path.Combine(aishellDirectory, "model.onnx")))
        {
            var model = new OnnxModel(Path.Combine(aishellDirectory, "model.onnx"));
            var frontend = new IcefallFrontend(Path.Combine(aishellDirectory, "lexicon.txt"), Path.Combine(aishellDirectory, "tokens.txt"), model.Metadata);
            foreach (var context in contexts)
            {
                context.Add("aishell3", new MandarinTts(model, frontend, options.Threads));
            }

            string[] speakers = File.ReadAllLines(Path.Combine(aishellDirectory, "speakers.txt"));
            var engine = contexts[0]["aishell3"];
            if (speakers.Length != engine.SpeakerCount)
            {
                throw new InvalidDataException("AISHELL3 发音人清单数量与模型不一致。");
            }

            catalog.Add(new("aishell3", $"普通话多发音人（{speakers.Length} 位）", engine.NativeSampleRate,
                speakers.Select((source, id) => new TtsSpeakerInfo(id, $"普通话 {id:D3} · {source}", source)).ToArray(), 50));
        }

        Voices = catalog.ToArray();
        _textValidators = contexts[0];
        Asr = new BoundedInferencePool<FunAsr>(recognizers, options.QueueCapacity);
        Tts = new BoundedInferencePool<Dictionary<string, MandarinTts>>(contexts, options.QueueCapacity);
        Sessions = new SemaphoreSlim(options.MaxSessions, options.MaxSessions);
        Uploads = new SemaphoreSlim(options.MaxUploads, options.MaxUploads);
        TtsRequests = new SemaphoreSlim(options.MaxTtsRequests, options.MaxTtsRequests);
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: false);
    }

    /// <summary>
    /// 本进程生效的配置。
    /// </summary>
    public SpeechOptions Options
    {
        get;
    }
    /// <summary>
    /// ASR 任务池；输入数据必须保持有效直到返回任务完成。
    /// </summary>
    public BoundedInferencePool<FunAsr> Asr
    {
        get;
    }
    /// <summary>
    /// TTS 任务池；工作上下文有独立随机数状态。
    /// </summary>
    public BoundedInferencePool<Dictionary<string, MandarinTts>> Tts
    {
        get;
    }
    /// <summary>
    /// 实际加载的音色目录，未安装的可选模型不会出现在列表中。
    /// </summary>
    public IReadOnlyList<TtsVoiceInfo> Voices
    {
        get;
    }
    /// <summary>
    /// 只读的 VAD 权重，每个连接创建独立的 FSMN 缓存。
    /// </summary>
    public Gguf Vad
    {
        get;
    }
    /// <summary>
    /// WebSocket 会话入口容量。
    /// </summary>
    public SemaphoreSlim Sessions
    {
        get;
    }
    /// <summary>
    /// WAV 上传入口容量，限制最终采样数组的同时存活数量。
    /// </summary>
    public SemaphoreSlim Uploads
    {
        get;
    }
    /// <summary>
    /// TTS HTTP 请求入口容量，限制慢客户端持有的音频数量。
    /// </summary>
    public SemaphoreSlim TtsRequests
    {
        get;
    }

    /// <summary>
    /// 只读访问发音词典，提前拒绝无法合成的文本。
    /// </summary>
    /// <param name="request">包含音色、发音人编号与文本的合成请求。</param>
    /// <returns>已校验模型的原生采样率。</returns>
    public int ValidateText(TtsRequest request)
    {
        if (!_textValidators.TryGetValue(request.VoiceId, out var engine))
        {
            throw new ArgumentException("指定音色尚未安装，请从 /tts/voices 获取可用列表。");
        }

        if ((uint)request.SpeakerId >= (uint)engine.SpeakerCount)
        {
            throw new ArgumentException($"该音色的发音人编号须为 0–{engine.SpeakerCount - 1}。");
        }

        engine.ValidateText(request.Text);
        return engine.NativeSampleRate;
    }
    /// <summary>
    /// 等待正在执行的推理结束，再释放调度资源。
    /// </summary>
    /// <returns>所有推理池关闭时完成的任务。</returns>
    public async ValueTask DisposeAsync()
    {
        await Asr.DisposeAsync();
        await Tts.DisposeAsync();
        Sessions.Dispose();
        Uploads.Dispose();
        TtsRequests.Dispose();
    }
}
