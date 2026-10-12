using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace FunAsrNet;

/// <summary>
/// 中文发音词典前端，完成短语匹配、三声变调与音素编码。
/// </summary>
public sealed class MandarinFrontend
{
    private readonly Dictionary<string, string> _lexicon;
    private readonly Dictionary<string, int[]> _symbols;
    private readonly int _longest;
    /// <summary>
    /// 中文发音词典前端，完成短语匹配、三声变调与音素编码。
    /// </summary>
    /// <param name="lexiconPath">中文短语发音词典 JSON 文件路径。</param>
    /// <param name="configPath">模型音素表和配置 JSON 文件路径。</param>
    public MandarinFrontend(string lexiconPath, string configPath)
    {
        _lexicon = JsonSerializer.Deserialize(File.ReadAllText(lexiconPath), MandarinJsonContext.Default.Lexicon)!;
        using var config = JsonDocument.Parse(File.ReadAllText(configPath));
        _symbols = config.RootElement.GetProperty("phoneme_id_map").Deserialize(MandarinJsonContext.Default.Symbols)!;
        _longest = _lexicon.Keys.Max(k => k.Length);
    }

    /// <summary>
    /// 按最长短语匹配生成音素字符串，并处理跨短语三声变调。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    /// <returns>完成处理后的文本。</returns>
    public string Phonemize(string text)
    {
        text = text.Normalize(NormalizationForm.FormKC);
        var normalized = new StringBuilder();
        foreach (char c in text)
        {
            normalized.Append(c is >= '0' and <= '9' ? "零一二三四五六七八九"[c - '0'] : c);
        }

        text = normalized.ToString();
        var pieces = new List<string>();
        for (int i = 0; i < text.Length;)
        {
            char c = text[i];
            if (char.IsWhiteSpace(c))
            {
                ++i;
                continue;
            }

            string punctuation = c switch
            {
                '，' or ',' or '、' => ",",
                '。' or '.' => ".",
                '！' or '!' => "!",
                '？' or '?' => "?",
                '：' or ':' => ":",
                '；' or ';' => ";",
                _ => ""
            };
            if (punctuation.Length != 0)
            {
                pieces.Add(punctuation);
                ++i;
                continue;
            }

            if (c is '“' or '”' or '「' or '」' or '（' or '）' or '(' or ')' or '"' or '\'')
            {
                ++i;
                continue;
            }

            string? phones = null;
            int consumed = 0;
            for (int length = Math.Min(_longest, text.Length - i); length > 0; --length)
            {
                if (_lexicon.TryGetValue(text.Substring(i, length), out phones))
                {
                    consumed = length;
                    break;
                }
            }

            if (phones is null)
            {
                throw new ArgumentException($"Unsupported pronunciation at U+{(int)c:X4} ({c}); add it to the lexicon");
            }

            pieces.Add(phones);
            i += consumed;
        }

        if (pieces.Count == 0)
        {
            throw new ArgumentException("Text contains no pronounceable content");
        }

        // 跨词典词条进行三声变调；短语词条内部的读音已经确定。
        for (int i = 0; i + 1 < pieces.Count; ++i)
        {
            if (Regex.IsMatch(pieces[i], "3[^1-5]*$") && Regex.IsMatch(pieces[i + 1], "^[^1-5]*3"))
            {
                pieces[i] = Regex.Replace(pieces[i], "3([^1-5]*)$", "2$1");
            }
        }

        return string.Join(' ', pieces);
    }

    /// <summary>
    /// 将中文文本转换为模型的音素 ID 序列，含边界与间隔符。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    /// <returns>按输出顺序排列的整数数组。</returns>
    public int[] Encode(string text)
    {
        string phonemes = Phonemize(text);
        var ids = new List<int>();
        ids.AddRange(_symbols["^"]);
        ids.AddRange(_symbols["_"]);
        foreach (var rune in phonemes.EnumerateRunes())
        {
            if (!_symbols.TryGetValue(rune.ToString(), out var values))
            {
                throw new ArgumentException($"Model does not support phoneme {rune}");
            }

            ids.AddRange(values);
            ids.AddRange(_symbols["_"]);
        }

        ids.AddRange(_symbols["$"]);
        return ids.ToArray();
    }
}

/// <summary>
/// 中文语音合成上下文；同一实例须串行调用，不同上下文可共享只读权重。
/// </summary>
public sealed class MandarinTts
{
    private readonly ManagedOnnx _engine;
    private readonly Func<string, int[]> _encode;
    private readonly bool _icefall;
    /// <summary>可选计算后端，须在首次合成前配置，调用方须保持其生命周期覆盖全部合成。</summary>
    public IFloatComputeBackend? FloatBackend
    {
        get => _engine.FloatBackend;
        set => _engine.FloatBackend = value;
    }
    /// <summary>
    /// 当前模型原生采样率，重采样必须使用此值。
    /// </summary>
    public int NativeSampleRate { get; } = SampleRate;
    /// <summary>
    /// 当前模型可选择的发音人数，编号从零开始。
    /// </summary>
    public int SpeakerCount { get; } = 1;
    /// <summary>
    /// Huayan 模型原生输出采样率，固定为 22,050 Hz。
    /// </summary>
    public const int SampleRate = 22050;
    /// <summary>
    /// 单发音人中文语音合成上下文；同一实例须串行调用，输出为 22,050 Hz 浮点音频。
    /// </summary>
    /// <param name="modelPath">Huayan ONNX 模型文件路径。</param>
    /// <param name="lexiconPath">中文短语发音词典 JSON 文件路径。</param>
    /// <param name="threads">算子使用的最大计算线程数。</param>
    public MandarinTts(string modelPath, string lexiconPath, int threads = 4) : this(new OnnxModel(modelPath), lexiconPath, modelPath + ".json", threads)
    {
    }

    /// <summary>
    /// 单发音人中文语音合成上下文；同一实例须串行调用，输出为 22,050 Hz 浮点音频。
    /// </summary>
    /// <param name="model">已加载的模型及其权重；发布给并发上下文后不得修改权重。</param>
    /// <param name="lexiconPath">中文短语发音词典 JSON 文件路径。</param>
    /// <param name="configPath">模型音素表和配置 JSON 文件路径。</param>
    /// <param name="threads">算子使用的最大计算线程数。</param>
    public MandarinTts(OnnxModel model, string lexiconPath, string configPath, int threads = 4)
    {
        // 实测 Huayan 声码器的长窄张量更适合行主序矩阵内核。
        _engine = new ManagedOnnx(model, new TensorMath(threads) { Optimized = false });
        _encode = new MandarinFrontend(lexiconPath, configPath).Encode;
    }

    /// <summary>
    /// 创建 AISHELL3 多发音人推理上下文，共享权重及只读拼音词典。
    /// </summary>
    /// <param name="model">已加载的 Icefall VITS 模型与元数据。</param>
    /// <param name="frontend">与模型配套的只读音素编码器。</param>
    /// <param name="threads">张量计算的最大线程数。</param>
    public MandarinTts(OnnxModel model, IcefallFrontend frontend, int threads = 4)
    {
        if (!model.Inputs.SequenceEqual(new[] { "tokens", "tokens_lens", "noise_scale", "alpha", "noise_scale_dur", "speaker" }))
        {
            throw new InvalidDataException("当前 AISHELL3 适配器不支持该模型的输入契约。");
        }

        _engine = new ManagedOnnx(model, new TensorMath(threads) { Optimized = false });
        _encode = frontend.Encode;
        _icefall = true;
        NativeSampleRate = int.Parse(model.Metadata["sample_rate"], System.Globalization.CultureInfo.InvariantCulture);
        SpeakerCount = int.Parse(model.Metadata["n_speakers"], System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// 按当前模型原生采样率生成单声道语音；长文本按句段处理后合并。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    /// <param name="speed">语速倍率，范围 0.5 到 2，1 为默认语速。</param>
    /// <param name="seed">随机种子；相同模型、文本、参数和运行环境可重复生成。</param>
    /// <param name="noise">潜变量随机度，范围 0 到 2，默认 0.667。</param>
    /// <param name="durationNoise">时长随机度，范围 0 到 2，默认 0.8。</param>
    /// <param name="speakerId">模型内部发音人编号，须在零到发音人数减一之间。</param>
    /// <returns>原生采样率的单声道浮点数组，生命期由调用方管理。</returns>
    public float[] Synthesize(string text, float speed = 1, int seed = 1, float noise = 0.667f, float durationNoise = 0.8f, int speakerId = 0)
    {
        if (string.IsNullOrWhiteSpace(text)
            || text.Length > 200
            || speed is < 0.5f or > 2
            || !float.IsFinite(speed)
            || seed < 0
            || !float.IsFinite(noise)
            || noise is < 0 or > 2
            || !float.IsFinite(durationNoise)
            || durationNoise is < 0 or > 2)
        {
            throw new ArgumentException("文本长度须为 1–200 字符，语速 0.5–2，种子非负，两项随机度须为 0–2 的有限数值");
        }

        if (text.Length > 40)
        {
            return SynthesizeChunks(text, speed, seed, noise, durationNoise, speakerId).SelectMany(x => x).ToArray();
        }

        return SynthesizePart(text, speed, seed, noise, durationNoise, speakerId);
    }

    /// <summary>
    /// 预先验证文本长度与发音词典覆盖，避免发送音频后才发现不支持的字符。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    public void ValidateText(string text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Length > 200)
        {
            throw new ArgumentException("Use 1..200 characters per request");
        }

        _encode(text);
    }

    /// <summary>
    /// 按标点和长度切分文本，逐段生成浮点音频；枚举期间独占该合成上下文。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    /// <param name="speed">语速倍率，范围 0.5 到 2，1 为默认语速。</param>
    /// <param name="seed">随机种子；相同模型、文本、参数和运行环境可重复生成。</param>
    /// <param name="noise">潜变量随机度，范围 0 到 2，默认 0.667。</param>
    /// <param name="durationNoise">时长随机度，范围 0 到 2，默认 0.8。</param>
    /// <param name="speakerId">模型内部发音人编号；所有句段保持同一编号。</param>
    /// <returns>按播放顺序产生的原生采样率单声道音频块。</returns>
    public IEnumerable<float[]> SynthesizeChunks(string text, float speed = 1, int seed = 1, float noise = 0.667f, float durationNoise = 0.8f, int speakerId = 0)
    {
        if (string.IsNullOrWhiteSpace(text)
            || text.Length > 200
            || speed is < 0.5f or > 2
            || !float.IsFinite(speed)
            || seed < 0
            || !float.IsFinite(noise)
            || noise is < 0 or > 2
            || !float.IsFinite(durationNoise)
            || durationNoise is < 0 or > 2)
        {
            throw new ArgumentException("文本长度须为 1–200 字符，语速 0.5–2，种子非负，两项随机度须为 0–2 的有限数值");
        }

        var chunk = new StringBuilder();
        int index = 0;
        foreach (char c in text)
        {
            chunk.Append(c);
            if (chunk.Length >= 24 || c is '，' or '。' or '！' or '？' or '；' or ',' or '.' or '!' or '?' or ';')
            {
                string part = chunk.ToString();
                chunk.Clear();
                if (part.Any(char.IsLetterOrDigit))
                {
                    yield return SynthesizePart(part, speed, unchecked(seed + index++) & int.MaxValue, noise, durationNoise, speakerId);
                }
            }
        }

        if (chunk.ToString().Any(char.IsLetterOrDigit))
        {
            yield return SynthesizePart(chunk.ToString(), speed, unchecked(seed + index) & int.MaxValue, noise, durationNoise, speakerId);
        }
    }

    /// <summary>
    /// 将短句编码为音素，并向当前 VITS 模型传入发音人及合成参数。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    /// <param name="speed">语速倍率，范围 0.5 到 2，1 为默认语速。</param>
    /// <param name="seed">随机种子；相同模型、文本、参数和运行环境可重复生成。</param>
    /// <param name="noise">潜变量随机度，范围 0 到 2，默认 0.667。</param>
    /// <param name="durationNoise">时长随机度，范围 0 到 2，默认 0.8。</param>
    /// <param name="speakerId">模型内部发音人编号。</param>
    /// <returns>当前短句的原生采样率单声道浮点采样。</returns>
    private float[] SynthesizePart(string text, float speed, int seed, float noise, float durationNoise, int speakerId)
    {
        if ((uint)speakerId >= (uint)SpeakerCount)
        {
            throw new ArgumentOutOfRangeException(nameof(speakerId), $"发音人编号须为 0–{SpeakerCount - 1}。");
        }

        int[] ids = _encode(text);
        var tokens = new Tensor(ids.Select(i => (float)i).ToArray(), [1, ids.Length], true);
        var length = new Tensor([(float)ids.Length], [1], true);
        Dictionary<string, Tensor> inputs = _icefall
            ? new()
            {
                ["tokens"] = tokens,
                ["tokens_lens"] = length,
                ["noise_scale"] = new([noise], [1]),
                ["alpha"] = new([1 / speed], [1]),
                ["noise_scale_dur"] = new([durationNoise], [1]),
                ["speaker"] = new([(float)speakerId], [1], true)
            }
            : new()
            {
                ["input"] = tokens,
                ["input_lengths"] = length,
                ["scales"] = new([noise, 1 / speed, durationNoise], [3])
            };
        var outputs = _engine.Run(inputs, seed);
        var audio = outputs.Values.First().Data;
        if (audio.Any(x => !float.IsFinite(x)))
        {
            throw new InvalidDataException("TTS generated non-finite samples");
        }

        return audio;
    }
}
