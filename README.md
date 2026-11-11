# FunAsrNet

简体中文 · [English](README.en.md) · .NET 10+ · [发布说明](doc/releases/1.0.1.md) · [Apache-2.0](LICENSE)

**在 .NET 应用中使用普通话识别、中文语音合成和电话音频处理，默认推理由纯 C# 完成。**

FunAsrNet 是基于固定 [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) 架构的独立社区实现，提供 CPU SIMD、可选 Vulkan GPU 后端、C# API、CLI 和 ASP.NET Core 示例。适合希望将已验证语音模型直接集成到 .NET 服务，并能查看、调试和调整推理代码的开发者。

当前范围是指定的 Fun-ASR、FSMN VAD 和中文 VITS 模型；项目不是官方 FunASR SDK，也不覆盖整个 FunASR 工具箱或任意 ONNX 模型。

## 功能

| 功能 | 当前支持 |
|---|---|
| 语音识别 | WAV/PCM 普通话转写；固定 F16/FP32 编码器与 Qwen3 Q8_0/FP32 解码器 |
| 实时展示 | WebSocket、VAD 切句、滑动窗口预览、结果修订和过期预览取消 |
| 中文语音合成 | Huayan 1 位、AISHELL3 174 位发音人；两模型安装后合计 175 位；支持语速、音量、采样率和格式 |
| 电话音频 | PCM16、G.711 PCMA/PCMU、重采样，可接入自有 SIP/PJSIP 媒体层 |
| CPU 加速 | x64 AVX2/FMA/AVX-VNNI、ARM64 NEON/SDOT，以及不支持对应指令时的回退 |
| 可选 GPU | Vulkan FP32 矩阵、卷积和长序列注意力，保留原 Q8_0 投影；按形状和设备特性选择内核 |
| 服务与示例 | HTTP/WebSocket 接口、浏览器麦克风转写、上传 WAV、TTS 播放与下载 |
| 会话管理 | 只读权重共享、独立会话工作区、有界队列、配额、取消和池缓冲 |

## 优势与限制

| 维度 | 优势 | 当前限制 |
|---|---|---|
| .NET 集成 | CPU 模型读取、特征、算子和解码均为 C#，核心包没有原生推理引擎依赖 | 模型与发音资产单独准备；Vulkan 需要系统 GPU 驱动 |
| 可维护性 | 推理路径、内存布局、SIMD 和着色器源码可直接检查和修改 | 支持固定模型及必要算子子集，模型覆盖面有限 |
| 性能 | 已优化矩阵、特征、解码、权重加载、分组投影与缓冲复用 | 最近同批 GPU/TTS 原版对照仍有差距；首次请求、长语料与多路负载需单独测量 |
| 精度 | 支持 BF16/FP32 原始权重无损展开为 FP32 GGUF；性能对照保持权重量化级别 | 沿用 F16 KV 缓存；GPU 浮点归约不保证与 CPU 逐位相同；FP32 权重占用更多内存 |
| 实时与电话集成 | 提供 VAD、预览、队列及音频编解码接口 | 实时识别基于切句与重新编码；未实现逐帧增量编码、SIP 信令、RTP 抖动缓冲或 AEC |
| 平台与语言 | Windows x64 和 Linux ARM64 有实机记录；Windows/Linux 有无模型 CI | 普通话是当前验证范围；其他语言、macOS、其他 GPU 需独立验收；NPU 未接入正式模型 |

TTS 暂无独立音高、情绪或声音克隆接口。多音字、人名、真实通话准确率和持续并发容量需要用业务语料验收。一个模型实例须串行调用；增加会话数量不代表吞吐线性增加。

## 快速开始

### 安装与模型

核心包：[FunAsrNet](https://www.nuget.org/packages/FunAsrNet/1.0.1)。可选 GPU 包：[FunAsrNet.Vulkan](https://www.nuget.org/packages/FunAsrNet.Vulkan/1.0.1)。

```shell
dotnet add package FunAsrNet --version 1.0.1
```

最低运行时为 .NET 10，已做 .NET 11 正确性回归。**模型不包含在 NuGet 包内。** 按[模型清单与 SHA-256](doc/models.md)准备权重和资产；原始权重导出见[精度与转换说明](doc/cpu-gpu-original-20261111.md)。

### 语音识别

```csharp
using FunAsrNet;

var asr = new FunAsr(
    "models/funasr-encoder-f16.gguf",
    "models/qwen3-0.6b-q8_0.gguf",
    threads: 4);
var (samples, rate) = AudioFile.ReadWave("input.wav");
var result = asr.Transcribe(
    AudioFile.Resample(samples, rate, 16000),
    maxTokens: 128);
Console.WriteLine(result.Text);
```

### 启用 GPU

安装扩展包，并在首次推理前为上面的实例设置后端：

```shell
dotnet add package FunAsrNet.Vulkan --version 1.0.1
```

```csharp
using FunAsrNet.Vulkan;

using var gpu = new VulkanFloatBackend();
asr.FloatBackend = gpu;
```

所有使用该后端的推理须在它释放前完成。GPU 只承接部分算子，剩余计算仍在 CPU；当前 Web 服务默认使用 CPU，尚无服务端 GPU 配置开关。

### 中文语音合成

```csharp
var tts = new MandarinTts(
    "models/zh_CN-huayan-medium.onnx",
    "assets/mandarin-lexicon.json",
    threads: 4);
var speech = tts.Synthesize("您好，欢迎致电。", speed: 1, seed: 1);
AudioFile.WriteWave("speech.wav", speech, tts.NativeSampleRate);
```

AISHELL3 多发音人配置、输出格式及参数见 [TTS 文档](doc/tts.md)。

## 与原版的性能对比

以下为同机、同批、预热后测量，单位秒、中位数，**越小越快**。完整条件和所有慢样本均保留在报告中，不能跨日期拼接速度比。

### 最近一次 C++ CPU 同批复核：2026-11-11

i7-12700、四线程、相同 F16/Q8 权重、关闭提示词缓存，每格 6 次正式调用。ASR 计完整转写：

| 输入 | FunAsrNet CPU | 原版 C++ CPU |
|---|---:|---:|
| 6 秒 | 0.749 | 0.857 |
| 24 秒重复样本 | 2.920 | 3.022 |

这组复核在本版最后一轮内核优化之前完成。桌面负载有明显波动，样本范围有限；结果不代表语言优势或广泛语料识别率。来源：[同批 C++ 复核](doc/cpu-gpu-original-20261111.md)。

### 最近完整 CPU/GPU 原版对照快照：2026-10-12

i7-12700、RTX 3060、四线程，每格 10 次正式调用。ASR 为完整转写；TTS 为相同音素输入的模型执行，排除文本前端、播放与重采样：

| 项目 | FunAsrNet CPU | 原版 CPU | FunAsrNet Vulkan | 原版 GPU |
|---|---:|---:|---:|---:|
| ASR 6 秒原始 PCM | 0.670 | 0.730 | 0.346 | 0.098 |
| ASR 28 秒重复样本 | 2.881 | 3.039 | 1.344 | 0.263 |
| TTS Huayan 0 号 | 0.141 | 0.108 | 0.154 | 0.028 |
| TTS AISHELL3 50 号 | 0.067 | 0.046 | 0.087 | 0.026 |

ASR 原版为固定 C++ CPU / 原计算图 CUDA 适配，TTS 原版为同模型 ONNX Runtime CPU / CUDA。GPU 时间包含传输与同步；ASR CUDA 的内部计算精度可能与 Vulkan FP32 不同。**该快照中 Vulkan 与 TTS 尚未追平原生后端，TTS 的 Vulkan 也慢于自身 CPU。** 初始化、首次请求、精度范围和完整数据见[同轮报告](doc/gpu-round7-20261012.md)。

### 本版后续优化

2026-11-11 的最新 24 秒样本中，CPU F16/Q8、原始权重路径分别为 **2.406 / 6.460 秒**；GPU 扩大样本复核分别为 **0.865 / 0.980 秒**。相对该轮各自对照，CPU 耗时观察值减少约 4.0% / 8.4%，GPU 减少约 7.3% / 10.8%。

CPU 每格 6 次，GPU 每格 16 次交替复核、4 GiB 权重缓存，且仅将 GPU 基准进程临时设为 AboveNormal 优先级。普通桌面批次仍有反向波动，Q8 短句基本持平；本轮未重跑原生 CUDA 或 TTS 完整模型，不能据此给出最新原生 GPU 差距。320 次正式模型调用中，同一输入文本和 token 数一致。详见[最新优化报告](doc/cpu-gpu-round2-20261111.md)。

全部历史记录、加载与首次请求、ARM64、Native AOT 和原始证据统一见[性能索引](doc/performance.md)。

## 从源码运行

源码构建遵循 global.json，需要 .NET 10 SDK 10.0.300 或同主版本内符合回退规则的更新 SDK：

```shell
dotnet build FunAsrNet.sln -c Release
dotnet run --project tests/FunAsrNet.Checks -c Release --no-build
dotnet run --project tools/FunAsrNet.CodeStyle -c Release --no-build -- .
```

准备模型后启动浏览器 Demo：

```shell
dotnet run --project src/FunAsrNet.Server -c Release -- --urls http://localhost:18080 --models models --lexicon assets/mandarin-lexicon.json --threads 4
```

打开 http://localhost:18080。远程麦克风访问需 HTTPS；部署与证书配置见[部署文档](doc/deployment.md)。也可使用 [CLI](src/FunAsrNet.Cli) 进行离线转写与合成。

## 文档与许可

| 主题 | 文档 |
|---|---|
| 模型与精度 | [模型清单](doc/models.md) · [原始权重](doc/cpu-gpu-original-20261111.md) |
| 使用与部署 | [HTTP/WebSocket 与 PJSIP](doc/api.md) · [TTS](doc/tts.md) · [部署与 HTTPS](doc/deployment.md) |
| 实现与容量 | [架构](doc/architecture.md) · [并发与内存](doc/concurrency-and-memory.md) |
| 性能与开发 | [测量与证据](doc/performance.md) · [代码规范](doc/development.md) · [NuGet/Release 流程](doc/releases.md) |

代码采用 [Apache-2.0](LICENSE)。模型与发音资产按各自上游许可使用；Huayan 训练数据许可为 Unknown，AISHELL3 归档没有独立模型 LICENSE，本项目不对这些资产重新授权。来源和 Vulkan 绑定许可见[第三方说明](THIRD_PARTY_NOTICES.md)。
