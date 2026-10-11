# FunASR.NET

**用 .NET 10 和纯 C# 推理实现普通话识别、语音合成与电话音频处理。** 核心命名空间 `FunAsrNet`，支持 CPU SIMD 和显式启用的 Vulkan GPU 混合后端。

基于 [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) 固定版本架构的独立社区实现，**不是官方 .NET SDK，也不是整个 FunASR 工具箱的移植**。围绕固定 F16 编码器、Qwen3 Q8_0 解码器、FSMN VAD 与两种中文 VITS 模型验证，不承诺兼容上游所有模型及语言。

## 纯 .NET 的实现范围

- 默认 CPU 的模型读取、特征、神经网络算子、解码、VAD、中文前端和 TTS 均由 C# 实现，不是给原生引擎包一层接口。
- 核心、CLI 和 Web 服务不依赖 Python、ONNX Runtime、ggml、sherpa-onnx 或 RKNN 推理库。ONNX/GGUF 在这里是文件格式；自有 ONNX 解释器只支持固定模型需要的子集。
- 使用 .NET SIMD、ARM64 NEON/SDOT、池缓冲、共享只读权重和有界队列。各平台优化程度不同，Windows x64 ASR CPU 的交错布局仍有标量瓶颈。
- 可选 Vulkan 用 C# 调度系统原生 GPU 驱动，执行嵌入 SPIR-V。仓库保留 GLSL 源码字符串；“纯 .NET”描述默认 CPU 推理，不表示 GPU 无需原生驱动。
- `tools/FunAsrNet.NativeReference` 单独引用原生 ORT，只作开发对照，不在主解决方案中，不是产品依赖。

详见 [架构](doc/architecture.md) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 功能

| 功能 | 当前实现 |
|---|---|
| ASR | WAV 转写；固定 F16 编码器、Q8_0 解码器 |
| 实时转写 | WebSocket、VAD 切句、可修订预览；不是逐帧增量编码 |
| 中文 TTS | Huayan 1 位、AISHELL3 174 位发音人，安装两模型后共 175 位 |
| 浏览器 Demo | 麦克风实时转写、上传 WAV、TTS 播放/下载，发音人搜索列表 |
| 电话音频 | PCM16、G.711 PCMA/PCMU、重采样，可接自研 SIP/PJSIP 媒体层 |
| 并发 | 独立会话状态、只读权重共享、队列/配额/取消和缓冲所有权管理 |
| 可选 GPU | Vulkan FP32 矩阵/卷积、原 Q8_0 投影；剩余算子在 CPU |

TTS 可设置语速、音量、采样率、格式、发音人、种子和随机度；尚无独立音高/情绪控制。服务不实现 SIP 信令、RTP 抖动缓冲或 AEC。

## 构建与运行

安装 .NET 10 SDK（`global.json` 最低 10.0.300），在仓库根目录执行：

```shell
dotnet build FunAsrNet.sln -c Release
dotnet run --project tests/FunAsrNet.Checks -c Release --no-build
dotnet run --project tools/FunAsrNet.CodeStyle -c Release --no-build -- .
```

**权重不在仓库内。** 按 [模型清单](doc/models.md) 准备 `models/` 并校验 SHA-256。ASR/VAD 需要兼容 GGUF，暂未提供一键下载的公开 GGUF 镜像或转换工具。克隆代码可运行无模型数值检查，不能直接启动完整服务。

```shell
dotnet publish src/FunAsrNet.Server -c Release -o artifacts/server
dotnet artifacts/server/FunAsrNet.Server.dll --urls http://localhost:18080 --models models --lexicon assets/mandarin-lexicon.json --threads 4
```

打开 `http://localhost:18080` 使用三个 Demo。远程麦克风需 HTTPS，见 [部署](doc/deployment.md)。默认 Web 服务用 CPU；Vulkan 通过代码或基准工具显式选择，服务暂未提供 GPU 配置开关。

```shell
dotnet run --project src/FunAsrNet.Cli -c Release -- asr --enc models/funasr-encoder-f16.gguf --llm models/qwen3-0.6b-q8_0.gguf --wav input.wav --threads 4 --repeat 3
dotnet run --project src/FunAsrNet.Cli -c Release -- tts --model models/aishell3/model.onnx --lexicon models/aishell3/lexicon.txt --tokens models/aishell3/tokens.txt --speaker 50 --text "您好，欢迎致电。" --rate 8000 --out speech.wav
```

## C# 调用

引用 `src/FunAsrNet/FunAsrNet.csproj`，尚未发布 NuGet 包。同一推理实例须串行使用。

```csharp
using FunAsrNet;

var asr = new FunAsr("models/funasr-encoder-f16.gguf", "models/qwen3-0.6b-q8_0.gguf", threads: 4);
var (samples, rate) = AudioFile.ReadWave("input.wav");
var result = asr.Transcribe(AudioFile.Resample(samples, rate, 16000), maxTokens: 128);
Console.WriteLine(result.Text);

var tts = new MandarinTts("models/zh_CN-huayan-medium.onnx", "assets/mandarin-lexicon.json", threads: 4);
var speech = tts.Synthesize("您好，欢迎致电。", speed: 1, seed: 1);
AudioFile.WriteWave("speech.wav", speech, tts.NativeSampleRate);
```

GPU 另外引用 `src/FunAsrNet.Vulkan/FunAsrNet.Vulkan.csproj`，首次推理前设置：

```csharp
using FunAsrNet.Vulkan;

using var gpu = new VulkanFloatBackend();
var gpuAsr = new FunAsr("models/funasr-encoder-f16.gguf", "models/qwen3-0.6b-q8_0.gguf", threads: 4)
{
    FloatBackend = gpu
};
var gpuTts = new MandarinTts("models/zh_CN-huayan-medium.onnx", "assets/mandarin-lexicon.json", threads: 4)
{
    FloatBackend = gpu
};
// 所有推理须在 gpu 释放前完成；模型实例仍串行使用。
```

## 与原版的真实速度对比

**2026-10-12 TTS CPU 优化实测：** 相同 i7-12700、4 线程和模型输入，Huayan 从 0.975 秒降到 0.317 秒，AISHELL3 从 0.407 秒降到 0.132 秒，均提速约 3.07 倍；同机 ORT CPU 分别为 0.102 秒和 0.043 秒。RK3588 上，相对旧托管部署，Huayan 从 2.142 秒降到 1.234 秒、AISHELL3 从 1.149 秒降到 0.742 秒，分别提速 1.74 倍和 1.55 倍；同机 ORT CPU 为 0.437 秒和 0.234 秒。仍比 ORT 慢约 2.83–3.18 倍。模型与精度保持不变，两端波形校验和数值回归通过，详见 [Windows 实测](doc/tts-cpu-optimization.md) 与 [ARM64 更新和实测](doc/rk3588-update-20261012.md)。下列表格保留此前历史结果。

RK3588 新旧托管 ASR 服务也已复测并完成 HTTP/HTTPS 验收：两份短句基本持平，28 秒重复样例中位数由 8.237 秒降到 5.169 秒，本次耗时减少 37.2%。每边仅三次正式样本，不能作为自然长通话或新一轮 C++ 原版对比结论，边界和原始结果见上述 ARM64 记录。

### 同一 RTX 3060：当前 C# Vulkan 对原生 GPU（2026-10-12）

4 线程、相同模型和输入，计预热后完整推理调用中位数，包含 GPU 传输及同步。TTS 对照为原生 ORT CUDA 1.23.2，关闭 TF32、保持 FP32；已用剖析确认 CUDA 节点实际执行。

| TTS 模型 | 当前 C# Vulkan | 原生 ORT CUDA | C# / 原生耗时 |
|---|---:|---:|---:|
| Huayan 0 号 | 0.296 秒 | 0.020 秒 | 14.55 倍 |
| AISHELL3 50 号 | 0.174 秒 | 0.018 秒 | 9.63 倍 |

| ASR 输入 | 当前 C# Vulkan | C++ CPU 编码器 + CUDA 解码器 | C++ 原图 CUDA 适配 |
|---|---:|---:|---:|
| 6.000 秒 PCMA | 0.564 秒 | 0.312 秒 | 0.096 秒 |
| 5.616 秒 PCMU | 0.713 秒 | 0.308 秒 | 0.109 秒 |
| 28.000 秒重复样例 | 2.474 秒 | 1.485 秒 | 0.254 秒 |

**当前 C# Vulkan 没有同卡 GPU 速度优势。** 固定上游 C++ 的 Windows 入口原本默认 CPU；这里显式启用官方 llama.cpp CUDA 解码器，最后一列另将原编码计算图和权重放入 CUDA，属于明确标注的设备适配，不是原版开箱即用入口。ASR 保持 F16/Q8_0 权重，但 CUDA 内部计算精度可能与 Vulkan FP32 不同；双方都关闭提示词缓存，128 token 上限。短 PCMA 的原生输出少句末句号，另外两组文本完全相同；TTS 波形误差通过。不能将差异解释为语言优势。

GPU 对比共 105 次正式样本、23 次预热，保留全部慢样本。版本、适配范围、计时边界、可复现的基准生成器和原始记录见 [原生 GPU 对比](doc/gpu-native-comparison-20261012.md)。

以下 RK3588 CPU 和 Windows ASR GPU/CPU 表格为重命名前实现的**历史实测**。RK3588 来自 2026-09-30；Windows 来自 2026-10-10，不与上述当前 GPU 数据混算。原始记录、散列、版本与边界见 [性能比较](doc/performance.md)。

### RK3588：纯 C# CPU 对原版 C++ CPU

相同 F16/Q8_0、4 个 A76 核、4 线程；排除预热，单位秒。

| 音频 | C# 中位 / P95 | C++ 中位 / P95 | C# RTF |
|---|---:|---:|---:|
| 6.000 秒 PCMA | 1.137 / 1.229 | 1.160 / 1.910 | 0.189 |
| 5.616 秒 PCMU | 1.252 / 1.349 | 1.280 / 2.659 | 0.223 |
| 28.000 秒重复样例 | 5.241 / 5.662 | 6.077 / 6.286 | 0.187 |

短句基本持平，约 2% 差异不能证明稳定领先。长样例本次耗时少 13.8%，但输入是人工重复拼接。三组文本一致，未测可靠 CER/WER。在线设备有背景负载、调频和换页；不是多路电话持续实时容量保证。

### Windows：Vulkan ASR 对原版 CPU（2026-10-10 历史测量）

i7-12700、RTX 3060 12 GiB、.NET 10.0.12、CPU 4 线程/亲和性 `0x55`。以下为预热后完整调用中位数，包含上传、下载、同步及剩余 CPU 计算。

| ASR 音频 | 当时 C# CPU | 当时 C# Vulkan | 原版 C++ CPU | Vulkan 耗时减少 |
|---|---:|---:|---:|---:|
| 6.000 秒 PCMA | 9.689 秒 | 0.487 秒 | 0.749 秒 | 34.9% |
| 5.616 秒 PCMU | 9.958 秒 | 0.554 秒 | 0.856 秒 | 35.3% |
| 28.000 秒重复样例 | 39.879 秒 | 2.345 秒 | 2.913 秒 | 19.5% |

这是当时 GPU 混合路径与原版 CPU 的比较；那次实验未测 C++ GPU。新的原生 GPU 对照已列在上段，不能把这张旧 GPU/CPU 表格解释为当前的 GPU 竞争力或语言优势。ASR 的 C# x64 交错布局仍有标量瓶颈，当时 17–20 倍的 GPU/CPU 差异不能推广到充分优化的 x64 SIMD。

2026-10-10 Windows 历史实验共 108 次正式样本、30 次预热，包含当时的 ASR 和 TTS，详细旧表保留在 [历史 Vulkan 记录](doc/vulkan-local-validation.md)。表格排除模型读取和冷启动，不能当作首次请求延迟。RTX 3060 结果不能套用 RK3588 Mali；NPU 尚未接入正式模型。

## 结构与平台

```text
src/FunAsrNet/              C# 模型、算子、音频、队列
src/FunAsrNet.Vulkan/       可选 GPU、着色器、API 绑定
src/FunAsrNet.Server/       ASP.NET Core 与浏览器 Demo
src/FunAsrNet.Cli/          离线识别/合成
src/FunAsrNet.Probe/        电话音频验证客户端
tests/                     数值、接口与发音人检查
tools/                     规范、测速、隔离原生对照
doc/                       中文说明与历史测量证据
deploy/                    systemd 模板
assets/                    发音词典及来源
testdata/                  固定参考数据
.github/workflows/         无模型 CI
```

历史实现验证了 Windows x64 和 Linux ARM64，不绑定 RK。其他平台仍需验收。默认 4 个实时连接、1 个 ASR 上下文；不能理解为四路都已满足实时延迟。使用 Span/Memory、池缓冲、分块 WAV 和 BodyWriter 减少拷贝；异步句段与 GPU 传输仍有复制，**不宣称全链路零拷贝**。

## 文档与许可

- [架构与托管推理范围](doc/architecture.md)
- [模型及校验值](doc/models.md)
- [HTTP/WebSocket 与 PJSIP](doc/api.md)
- [TTS 参数](doc/tts.md)
- [部署和 HTTPS](doc/deployment.md)
- [并发与内存](doc/concurrency-and-memory.md)
- [性能比较及原始记录](doc/performance.md)
- [Vulkan 详细验证](doc/vulkan-local-validation.md)
- [代码规范](doc/development.md)
- [公开版整理与验证](doc/public-edition.md)

代码使用 [Apache-2.0](LICENSE)。模型和发音资产不随代码重新授权；Huayan 数据许可为 Unknown，AISHELL3 归档无独立模型 LICENSE。详见 [第三方说明](THIRD_PARTY_NOTICES.md)。
