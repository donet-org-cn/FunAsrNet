# HTTP、WebSocket 与 PJSIP 接入

服务地址默认 `http://localhost:18080`。网页与接口共用宿主。此示例服务未提供账号鉴权，面向受控局域网；通过网关对外提供时由业务系统负责访问控制。

## 路由

| 路由 | 请求 | 响应 |
|---|---|---|
| `GET /health` | 无 | 产品、版本、SIMD、计算忙闲、队列与会话计数 |
| `GET /demo/config` | 无 | HTTPS 端口、上传长度与会话限制 |
| `POST /asr/wav` | 请求体直接为 WAV 文件，不是 multipart | 文本、音频长度与计算计时 |
| `WS /asr/stream` | 音频二进制帧和结束控制消息 | ready、partial、final、done JSON 事件 |
| `GET /tts/voices` | 无 | 实际音色、发音人和参数默认值 |
| `POST /tts` | [合成 JSON 参数](tts.md) | 完整 WAV 或裸音频 |
| `POST /tts/stream` | 8000 Hz 的合成请求 | 按句段生成的裸 PCM16/PCMA/PCMU |

## WAV 上传

支持 PCM16、Float32、G.711 A-law/μ-law 的 WAV，8000–96000 Hz、1–8 声道；至少 0.1 秒、最多 60 秒、32 MiB。多声道混合为单声道后转换到 16 kHz。上传内容不落盘。

```csharp
using var http = new HttpClient { BaseAddress = new Uri("http://localhost:18080") };
await using var file = File.OpenRead("input.wav");
using var body = new StreamContent(file);
body.Headers.ContentType = new("audio/wav");
using var response = await http.PostAsync("/asr/wav", body);
response.EnsureSuccessStatusCode();
Console.WriteLine(await response.Content.ReadAsStringAsync());
```

响应字段：`text`、`audioSeconds`、`encoderSeconds`、`decoderSeconds`、`inferenceSeconds`、`realTimeFactor`、`requestSeconds`。`inferenceSeconds` 是编码与解码计算之和；`requestSeconds` 包含排队及调用耗时，从重采样完成后开始计时，不包含上传、WAV 解码和重采样。完整端到端延迟应由客户端自行计时。

## 实时电话音频

```text
ws://localhost:18080/asr/stream?codec=pcma&rate=8000&partials=true&silenceMs=900
ws://localhost:18080/asr/stream?codec=pcm16&rate=48000&partials=true&silenceMs=900
```

- `codec`：`pcma`、`pcmu`、`pcm16`；默认 pcma。
- `rate`：G.711 必须为 8000；PCM16 支持 8000、16000、22050、24000、32000、44100、48000、96000 Hz。
- `partials`：默认 false，开启后返回可修订预览。
- `silenceMs`：300–2500，默认 1200，表示结束一句所需静音长度。

收到 `{"event":"ready",...}` 后开始发送单声道音频。推荐每 20 ms 一帧：8 kHz PCMA/PCMU 为 160 字节，8 kHz PCM16 为 320 字节，16 kHz PCM16 为 640 字节。PCM16 使用小端格式。

发送的音频必须是媒体载荷，不包含 RTP 头、WAV 头或 SIP 信令。结束时发送一条完整文本消息 `{"event":"end"}`，继续读取直到 `done`，再关闭连接。文本控制消息不可跨多个 WebSocket 消息片段；音频可以分片，跨帧的单个 PCM 字节会保留到下一次输入。

事件字段包含 `event`、`text`、`elapsedSeconds`，有文本的句段还包含 `audioStartSeconds`、`audioEndSeconds`、`inferenceSeconds`。`partial` 可被后续预览修订；`final` 是闭合句段，应追加保存。错误关闭且没有 `done` 表示录音未完整处理。

这是 VAD 切句与滑动预览窗口上的推理，不是模型编码器逐帧增量计算。静音阈值、排队、预览频率、设备温度均会影响用户看到结果的时间。

自研 PJSIP 层应为每通电话建立独立连接，将媒体端口输出的 PCM16 或解包后的 G.711 帧送入服务；上、下行若要分别识别，应分别建立会话。RTP 重排、丢包补偿、抖动缓冲、AEC 和 SIP 注册由现有电话系统负责。服务本身不是 SIP UA，也没有实现呼叫控制。

## TTS 输出与计时

`/tts` 返回 `X-Inference-Seconds` 和 `X-Audio-Seconds`。前者包含推理排队、合成和输出重采样，不包括网络发送；后者是输出采样数除以输出采样率。

`/tts/stream` 返回 `X-Audio-Sample-Rate: 8000`，正文没有 WAV 头，按生成进度写出。HTTP 分块边界不是 20 ms 媒体边界：SIP 层必须先缓存，再按时钟取出 160 个 8 kHz 采样送入电话媒体口。禁止把一次 HTTP 读取当成一个 RTP 包。

## 错误与限流

| 状态 | 原因 |
|---|---|
| 400 | WAV/音频格式、文本或参数无效；不支持的音色/发音人 |
| 408 | WAV 上传读取超过两分钟 |
| 413 | 上传请求体超过 32 MiB |
| 429 | HTTP 配额、实时会话数或推理队列已满 |

已开始发送的合成流若发生失败会中断连接；客户端应将截断流视为失败。连接终止和取消的资源生命期见 [并发与内存](concurrency-and-memory.md)。
