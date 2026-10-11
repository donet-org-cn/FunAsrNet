using System.Text;

namespace FunAsrNet.Server;

/// <summary>
/// 中文语音合成参数；发音人范围由实际安装的模型目录校验。
/// </summary>
/// <param name="Text">待朗读的中文文本，长度为 1–200 字符。</param>
/// <param name="Speed">语速倍率，范围 0.5–2，默认 1。</param>
/// <param name="SampleRate">输出采样率，支持 8000、16000、22050 Hz。</param>
/// <param name="Format">输出编码，支持 wav、pcm16、pcma、pcmu；G.711 仅支持 8000 Hz。</param>
/// <param name="VoiceId">已安装音色标识，通过 /tts/voices 查询。</param>
/// <param name="SpeakerId">模型内部从零开始的发音人编号。</param>
/// <param name="Seed">非负随机种子，相同参数与环境下可重复生成。</param>
/// <param name="NoiseScale">潜变量随机度，范围 0–2，默认 0.667；不会切换发音人。</param>
/// <param name="DurationNoise">时长随机度，范围 0–2，默认 0.8。</param>
/// <param name="Volume">后处理音量倍率，范围 0–2，默认 1；超出振幅范围的结果会限幅。</param>
public sealed record TtsRequest(
    string Text,
    float Speed = 1,
    int SampleRate = 8000,
    string Format = "wav",
    string VoiceId = "huayan-medium",
    int SpeakerId = 0,
    int Seed = 1,
    float NoiseScale = 0.667f,
    float DurationNoise = 0.8f,
    float Volume = 1)
{
    /// <summary>
    /// 校验参数及模型能力；不支持的发音人不会被静默替换。
    /// </summary>
    /// <param name="streaming">是否按句段直接输出原始电话音频。</param>
    public void Validate(bool streaming = false)
    {
        if (string.IsNullOrWhiteSpace(Text) || Text.Length > 200 || !float.IsFinite(Speed) || Speed is < 0.5f or > 2)
        {
            throw new ArgumentException("文本长度须为 1–200 字，语速须在 0.5–2 倍之间。");
        }

        if (string.IsNullOrWhiteSpace(VoiceId) || SpeakerId < 0)
        {
            throw new ArgumentException("音色标识不能为空，发音人编号须非负。");
        }

        if (Seed < 0
            || !float.IsFinite(NoiseScale)
            || NoiseScale is < 0 or > 2
            || !float.IsFinite(DurationNoise)
            || DurationNoise is < 0 or > 2
            || !float.IsFinite(Volume)
            || Volume is < 0 or > 2)
        {
            throw new ArgumentException("随机种子须非负，随机度与音量须是 0–2 范围的有限数值。");
        }

        if (SampleRate is not (8000 or 16000 or 22050) || Format is not ("wav" or "pcm16" or "pcma" or "pcmu"))
        {
            throw new ArgumentException("采样率支持 8000/16000/22050，格式支持 wav/pcm16/pcma/pcmu。");
        }

        if (Format is "pcma" or "pcmu" && SampleRate != 8000 || streaming && (SampleRate != 8000 || Format == "wav"))
        {
            throw new ArgumentException("G.711 和流式合成需要 8000 Hz；流式输出格式须为 pcm16/pcma/pcmu。");
        }
    }

    /// <summary>
    /// 按标点或长度切分短句；各段可独立排队，不长期占用一个 TTS 上下文。
    /// </summary>
    /// <returns>依次产生的可朗读短句。</returns>
    public IEnumerable<string> SplitText()
    {
        var chunk = new StringBuilder();
        foreach (char value in Text)
        {
            chunk.Append(value);
            if (chunk.Length >= 24 || value is '，' or '。' or '！' or '？' or '；' or ',' or '.' or '!' or '?' or ';')
            {
                string part = chunk.ToString();
                chunk.Clear();
                if (part.Any(char.IsLetterOrDigit))
                {
                    yield return part;
                }
            }
        }

        if (chunk.Length > 0 && chunk.ToString().Any(char.IsLetterOrDigit))
        {
            yield return chunk.ToString();
        }
    }
}
