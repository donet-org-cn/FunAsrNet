namespace FunAsrNet.Server;

/// <summary>
/// 实际加载的语音模型及其可用发音人信息。
/// </summary>
/// <param name="Id">稳定的模型标识，用作请求中的 voiceId。</param>
/// <param name="Name">供示例页面显示的中文名称。</param>
/// <param name="SampleRate">模型原生采样率。</param>
/// <param name="Speakers">按模型内部编号排列的发音人清单。</param>
/// <param name="DefaultSpeakerId">示例页面首次选择该音色库时使用的发音人编号。</param>
public sealed record TtsVoiceInfo(string Id, string Name, int SampleRate, TtsSpeakerInfo[] Speakers, int DefaultSpeakerId = 0)
{
    /// <summary>
    /// 模型支持的语言标识。
    /// </summary>
    public string Language => "zh-CN";
    /// <summary>
    /// 兼容现有客户端的发音人编号列表。
    /// </summary>
    public int[] SpeakerIds { get; } = Speakers.Select(speaker => speaker.Id).ToArray();
}

/// <summary>
/// 发音人的稳定编号与来源标识；没有依据时不标注性别或情绪。
/// </summary>
/// <param name="Id">模型内部从零开始的编号。</param>
/// <param name="Name">中文显示名称。</param>
/// <param name="SourceId">上游语料的匿名发音人标识。</param>
public sealed record TtsSpeakerInfo(int Id, string Name, string SourceId);
