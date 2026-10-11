# 中文语音合成参数

`GET /tts/voices` 返回实际安装的音色、发音人编号、原生采样率和支持能力。示例页面从该接口加载音色列表。

支持两个模型音色库，共 175 位发音人：

| voiceId | speakerId | 原生采样率 | 说明 |
|---|---|---:|---|
| `huayan-medium` | `0` | 22050 Hz | 保留原音色和 API 默认行为 |
| `aishell3` | `0`–`173` | 8000 Hz | 174 位普通话发音人，共用一套权重；页面默认 50 号 |

服务只列出已安装模型；未安装音色和越界编号返回 400。`speakers` 包含编号、中文显示名和语料匿名标识，`speakerIds` 保持原接口兼容，`defaultSpeakerId` 供页面选择初始编号。没有依据时不标注男女或情绪。不同编号的清晰度存在差异，应使用实际业务文本试听验收。全部编号验证命令见 [开发说明](development.md)。

浏览器示例提供可滚动的发音人单选列表，支持按名称、编号或语料标识搜索。切换音色库会恢复库内最近选择；筛选无匹配时保留原选择。生成过程中仍可挑选下一次使用的发音人，但生成按钮保持禁用，当前请求继续使用提交时的编号。

| JSON 参数 | 默认值 | 范围与意义 |
|---|---|---|
| `text` | 必填 | 1–200 个字符，推荐简体中文和常用标点 |
| `voiceId` | `huayan-medium` | 已加载的模型音色标识 |
| `speakerId` | `0` | 模型内部编号；AISHELL3 推荐显式选择 `50` |
| `speed` | `1` | 0.5–2 倍；传入模型的长度尺度为 `1 / speed` |
| `volume` | `1` | 0–2 倍，输出编码前调整并限幅；0 为静音 |
| `sampleRate` | `8000` | 8000、16000、22050 Hz；由原生采样率转换 |
| `format` | `wav` | `wav`、`pcm16`、`pcma`、`pcmu` |
| `seed` | `1` | 0–2147483647；控制本次合成的随机源 |
| `noiseScale` | `0.667` | 0–2；模型潜变量随机度，不等于切换发音人 |
| `durationNoise` | `0.8` | 0–2；音素时长的随机变化 |

所有浮点参数要求有限数值。`pcma/pcmu` 必须选择 8000 Hz。`POST /tts/stream` 只接受 8000 Hz 的 `pcm16/pcma/pcmu` 裸音频，按短句产生和发送；它不是逐采样自回归生成，也不是 RTP 数据包。

完整 WAV 示例：

```json
{
  "text": "您好，欢迎致电，请问有什么可以帮您？",
  "voiceId": "aishell3",
  "speakerId": 50,
  "speed": 1,
  "volume": 1,
  "sampleRate": 8000,
  "format": "wav",
  "seed": 1,
  "noiseScale": 0.667,
  "durationNoise": 0.8
}
```

C# 调用示例：

```csharp
using System.Net.Http.Json;

using var client = new HttpClient
{
    BaseAddress = new Uri("http://localhost:18080"),
    Timeout = TimeSpan.FromMinutes(2)
};
using var response = await client.PostAsJsonAsync("/tts", new
{
    text = "您好，欢迎致电。",
    voiceId = "aishell3",
    speakerId = 50,
    speed = 1.0,
    volume = 1.0,
    seed = 1,
    noiseScale = 0.667,
    durationNoise = 0.8,
    sampleRate = 8000,
    format = "wav"
});
response.EnsureSuccessStatusCode();
await using var file = File.Create("speech.wav");
await response.Content.CopyToAsync(file);
```

相同模型、文本、参数、运行时与硬件环境下可以重复生成。跨 CPU/运行时不承诺逐字节一致。长文本和流式接口会按句段增加随机种子；切句不同也会改变结果。

目前没有独立音高、情绪、角色模仿或 SSML 控制。中文前端按词典最长匹配，数字逐位读；Huayan 支持有限三声变调，AISHELL3 使用模型配套拼音词典，不调用上游的 FST 文本规范化器。日期、金额请先转为明确中文读法，多音字、人名和业务词需要扩充及验收词典。不支持的文字明确返回错误。

AISHELL3 原生为 8kHz，电话输出无需重采样；选择 16kHz 或 22.05kHz 只转换采样率，不增加原始声音带宽。不同音色和编号共用一个有界 TTS 工作池，各工作上下文拥有独立随机状态。
