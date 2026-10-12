# FunASR.NET

**用 .NET 10 和纯 C# 推理实现普通话识别、语音合成与电话音频处理。** 核心命名空间 `FunAsrNet`，支持 CPU SIMD 和显式启用的 Vulkan GPU 混合后端。

基于 [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) 固定版本架构的独立社区实现，**不是官方 .NET SDK，也不是整个 FunASR 工具箱的移植**。围绕固定 F16 编码器、Qwen3 Q8_0 解码器、FSMN VAD 与两种中文 VITS 模型验证，不承诺兼容上游所有模型及语言。

## 纯 .NET 的实现范围

- 默认 CPU 的模型读取、特征、神经网络算子、解码、VAD、中文前端和 TTS 均由 C# 实现，不是给原生引擎包一层接口。
- 核心、CLI 和 Web 服务不依赖 Python、ONNX Runtime、ggml、sherpa-onnx 或 RKNN 推理库。ONNX/GGUF 在这里是文件格式；自有 ONNX 解释器只支持固定模型需要的子集。
- 使用 .NET SIMD、ARM64 NEON/SDOT、池缓冲、共享只读权重和有界队列。ASR 默认布局按硬件选择，x64 用行主序 SIMD，ARM64 用 NEON/SDOT 交错内核；各平台优化程度不同。
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

### Windows：第六轮模型加载与 GPU 优化（2026-10-12）

已移除当前版本的六个 Python 复测/汇总脚本，统一使用 [C# 对比入口](tools/FunAsrNet.Compare/README.md)；历史原始数据保留。默认 CPU 推理仍不依赖原生引擎。

两份 ASR GGUF 的读取与解包中位数从 **3.203 秒降到 0.723 秒，耗时减少 77.4%**；托管分配从 3.076 GB 降到 1.812 GB。12 个新进程样本，未清空文件缓存，不含推理或 GPU 首次上传。短 ASR 的 CPU 模型初始化与首次转写之和从 4.061 秒降到 1.606 秒，同轮原版为 1.168 秒；仍不能当作完整服务冷启动。

RTX 3060、相同模型/输入、四线程；每格十次预热后的正式样本：

| 项目 | 上一版 C# Vulkan | 本次 C# Vulkan | 同轮原版 GPU | 本轮耗时减少 |
|---|---:|---:|---:|---:|
| ASR 6.000 秒 PCMA | 0.495 秒 | 0.427 秒 | 0.107 秒 | 13.8% |
| ASR 5.616 秒 PCMU | 0.553 秒 | 0.480 秒 | 0.125 秒 | 13.2% |
| ASR 28.000 秒重复样例 | 1.926 秒 | 1.640 秒 | 0.275 秒 | 14.8% |
| TTS Huayan 0 号 | 0.229 秒 | 0.175 秒 | 0.029 秒 | 23.3% |
| TTS AISHELL3 50 号 | 0.130 秒 | 0.095 秒 | 0.025 秒 | 27.0% |

原版 ASR 为 **C++ CUDA 编码器和解码器**，TTS 为同模型 **ORT CUDA**。GPU 三方 ASR 文本一致，新旧 C# GPU TTS 波形相同；未降低计算精度。**当前仍为原版 GPU 的 3.84–6.10 倍耗时，尚未追平。** Vulkan 混合路径仍有大量 CPU 中间处理和同步，不能解释为语言优劣；本机结果不适用于 RK3588。

同轮 CPU 回归的 ASR 仍略快于原版，热态相对上一版约 ±2%，没有进一步的整体提升；TTS CPU 尚未追平 ORT CPU。GPU 150 次正式调用/30 次预热，CPU 126 次/42 次，加载、首次调用、P95、内存和正确性边界见 [完整第六轮记录](doc/load-gpu-round6-20261012.md)。

### Windows：历史第五轮 ASR CPU 优化（2026-10-12）

保持纯 C# 推理，同模型、四线程、亲和性 `0x55`，双方关闭提示词缓存，预热后本轮中位数：

| ASR 输入 | 上一版 C# CPU | 本次 C# CPU | 同轮 C++ CPU | 本次 / 原版耗时 |
|---|---:|---:|---:|---:|
| 6.000 秒 PCMA | 0.743 秒 | 0.691 秒 | 0.749 秒 | 0.923 倍 |
| 5.616 秒 PCMU | 0.861 秒 | 0.825 秒 | 0.879 秒 | 0.939 倍 |
| 28.000 秒重复样例 | 2.939 秒 | 2.764 秒 | 2.976 秒 | 0.929 倍 |
| 6.000 秒原始 PCM | 0.680 秒 | 0.642 秒 | 0.708 秒 | 0.906 倍 |
| 官方样例 PCMA | 0.729 秒 | 0.682 秒 | 0.730 秒 | 0.934 倍 |

**这五份 ASR 热态样例比上一版减少 4%–7% 耗时，比同轮原版少约 6%–9%，转写逐字一致。** 新增两份是同一句话的格式变体，不是独立语料。共享 Q8 输入准备和 FP32 循环展开保持原数值次序；会减少输出 token 的分段实验已撤回。另修复实时预览排队取消，以及尾帧发送后等待设备释放才发送 `end` 的延迟。258 次正式样本、42 次预热，所有慢样本保留；P95、撤回方案及实时优化边界见 [本机 ASR 第五轮记录](doc/cpu-round5-20261012.md)。模型读取仍约 3.2 秒、原版约 0.3 秒，热态结果不能解释为首次请求也持平。仅验证 Windows x64，不能推广到其他语料或 RK3588。

同轮 TTS Huayan 为 0.138 / 0.103 秒、AISHELL3 为 0.065 / 0.044 秒（C# / ORT CPU），仍未追平。以下表格为各自版本的历史结果。

### Windows：历史第四轮纯 C# CPU（2026-10-12）

同模型、四线程、亲和性 `0x55`，双方预热并关闭提示词缓存，本轮中位数：

| 项目 | 上一版 C# CPU | 本次 C# CPU | 同轮原版 CPU | 本次 / 原版耗时 |
|---|---:|---:|---:|---:|
| ASR 6.000 秒 | 0.965 秒 | 0.749 秒 | 0.744 秒 | 1.006 倍 |
| ASR 5.616 秒 | 1.017 秒 | 0.854 秒 | 0.881 秒 | 0.969 倍 |
| ASR 28.000 秒 | 3.501 秒 | 3.082 秒 | 2.995 秒 | 1.029 倍 |
| TTS Huayan | 0.154 秒 | 0.152 秒 | 0.104 秒 | 1.463 倍 |
| TTS AISHELL3 | 0.069 秒 | 0.070 秒 | 0.048 秒 | 1.456 倍 |

**这三条 ASR 样例已基本追平原版 CPU，比上一版减少 12%–22% 耗时；TTS 仍约慢 46%，尚未追平。** TTS 本轮中位数分别减少 1.8% / 增加 2.4%，没有明确速度收益；Huayan 的每次托管分配减少 79.7%。ASR 计完整推理调用，TTS 只计相同音素输入的模型执行，不含模型加载、前端或重采样。258 次正式测量、42 次预热，包括针对 TTS 波动追加的固定轮数确认组，全部样本保留。三方转写一致、波形误差通过。P95、内存代价、纯 C# SIMD 改动和完整证据见 [本机 CPU 第四轮记录](doc/cpu-round4-20261012.md)。仅为 Windows x64 固定样例，不能推广到其他语料、尾延迟、RK3588 或真实麦克风识别率。

下方 AOT、CPU 和 GPU 表格保留为各自版本的历史结果。SimdPaddleOCR 的 `.comp` / `.spv` 属于 GPU 着色器链路，含义及已有引用见 [架构说明](doc/architecture.md)。

### Windows：前轮 Native AOT 编译对比（2026-10-12）

已增加独立 CPU AOT 基准入口，使用同一份源码分别发布 JIT/AOT。四线程、相同模型和输入、预热后的本轮中位数：

| 项目 | 同入口 JIT | Native AOT | 同轮原生 CPU |
|---|---:|---:|---:|
| ASR 6.000 秒 | 0.909 秒 | 0.914 秒 | 0.736 秒 |
| ASR 5.616 秒 | 0.971 秒 | 0.971 秒 | 0.837 秒 |
| ASR 28.000 秒 | 3.456 秒 | 3.626 秒 | 2.939 秒 |
| TTS Huayan | 0.159 秒 | 0.156 秒 | 0.102 秒 |
| TTS AISHELL3 | 0.071 秒 | 0.067 秒 | 0.045 秒 |

**AOT 主要改善启动，连续推理没有整体提速。** 同入口对比，短 ASR 基本持平、长 ASR 慢 4.9%，两种 TTS 中位数减少 2.3%/6.4%；小样本及背景负载下不能宣称微小差异显著。新进程至短 ASR 首个结果约从 4.7–4.9 秒降到 2.3 秒，未清空文件缓存；不能套用完整服务冷启动。184 次正式模型测量、40 次预热，JIT/AOT 各 1,796,447 项检查通过，ASR 文本与两版 TTS 波形一致。**仅 CPU 基准完成 AOT，Web 服务和 RK3588 尚未做 AOT 验证。** 编译命令、原有入口控制数据、内存与完整证据见 [AOT 对比报告](doc/aot-comparison-20261012.md)。

### Windows：历史第三轮纯 CPU 优化（2026-10-12）

同模型、四线程、亲和性 `0x55`，双方预热后测量。ASR 原版 C++ 启用 AVX2/FMA/AVX-VNNI；TTS 对照相同模型的 ORT 1.23.2 CPU。

| 项目 | 上一版 C# CPU | 本次 C# CPU | 原版 CPU | 本次 / 原版耗时 | 本轮耗时减少 |
|---|---:|---:|---:|---:|---:|
| ASR 6.000 秒 PCMA | 0.904 秒 | 0.877 秒 | 0.743 秒 | 1.18 倍 | 3.0% |
| ASR 5.616 秒 PCMU | 0.994 秒 | 0.964 秒 | 0.838 秒 | 1.15 倍 | 3.0% |
| ASR 28.000 秒重复样例 | 3.771 秒 | 3.317 秒 | 2.863 秒 | 1.16 倍 | 12.0% |
| TTS Huayan | 0.181 秒 | 0.148 秒 | 0.101 秒 | 1.47 倍 | 18.2% |
| TTS AISHELL3 | 0.072 秒 | 0.066 秒 | 0.044 秒 | 1.51 倍 | 9.0% |

本次 ASR 为原版 CPU 的 **1.15–1.18 倍耗时**，TTS 为 ORT CPU 的 **1.47–1.51 倍**。仍有项目超过原版 1.5 倍耗时。**尚未全面持平，也未达到原版一半耗时。** “慢 50% 以内”只是本次小样本中位数的口径，不能作为尾延迟或其他输入的保证。 本轮长音频与 TTS 有明确收益，短句提升有限。138 次正式样本、30 次预热，三方转写相同、TTS 波形误差通过；P95、优化范围与原始数据见 [本机 CPU 第三轮记录](doc/cpu-round3-20261012.md)。仅验证 Windows x64，未将新数据用于 RK3588 或真实麦克风识别率结论。

上一轮 Q8 和 TTS 的优化记录保留在 [前轮 CPU 对比](doc/cpu-followup-20261012.md)。下方表格是更早的历史快照。

### Windows：前轮默认布局修正记录（2026-10-12）

本次修正 x64 误用 ARM 交错布局而落入标量循环的问题，默认 CPU 使用已有 SIMD 内核。i7-12700、4 线程、亲和性 `0x55`，相同 F16/Q8_0 模型与输入，双方关闭提示词缓存、最大 128 token，排除加载和预热，计完整 ASR 调用中位数：

| ASR 输入 | 当时修正前 C# CPU | 当时修正后 C# CPU | 原版 C++ CPU | 修正后 / 原版耗时 |
|---|---:|---:|---:|---:|
| 6.000 秒 PCMA | 12.795 秒 | 1.423 秒 | 0.775 秒 | 1.84 倍 |
| 5.616 秒 PCMU | 12.720 秒 | 1.502 秒 | 0.886 秒 | 1.70 倍 |
| 28.000 秒重复样例 | 42.830 秒 | 5.018 秒 | 2.952 秒 | 1.70 倍 |

**默认 ASR CPU 提速 8.47–8.99 倍，仍慢于原版 CPU 1.70–1.84 倍。** 三组转写严格一致，模型和权重量化不变，没有依靠 GPU。

TTS 当前实现保持此前 CPU 优化，本次同模型 ORT CPU 复测合并两批全部样本，每模型、每引擎二十次；只计模型 Run，语速 1、随机度 0：

| TTS 模型 | C# CPU 中位 / P95 | 原生 ORT CPU 中位 / P95 | C# / ORT 耗时 |
|---|---:|---:|---:|
| Huayan 0 号 | 0.321 / 0.365 秒 | 0.106 / 0.111 秒 | 3.03 倍 |
| AISHELL3 50 号 | 0.138 / 0.299 秒 | 0.045 / 0.050 秒 | 3.10 倍 |

TTS 波形长度和误差通过；AISHELL3 的块间波动与慢样本全部保留。Windows CPU 共 184 次正式测量、40 次预热，另有单独布局诊断。P95、版本、计时边界、散列和原始逐轮数据见 [纯 CPU 对比](doc/cpu-native-comparison-20261012.md)。这些热态结果不能当作首次请求延迟或并发容量。

此前 TTS 优化相对初始实现提速约三倍的结论继续成立，记录见 [TTS CPU 优化](doc/tts-cpu-optimization.md)。RK3588 上，相对旧托管部署，Huayan 从 2.142 秒降到 1.234 秒、AISHELL3 从 1.149 秒降到 0.742 秒，分别提速 1.74 倍和 1.55 倍；同机 ORT CPU 为 0.437 秒和 0.234 秒，仍慢约 2.83–3.18 倍。ARM64 波形与服务验收见 [更新和实测](doc/rk3588-update-20261012.md)。

### RK3588：纯 CPU 原版对照快照（2026-10-12，`ab83e5b`）

CPU 4–7 的四个大核、四线程，双方关闭提示词缓存、128 token，直接计完整推理调用；52 次正式样本、12 次预热：

| ASR 输入 | C# CPU 中位 / P95 | 原版 C++ CPU 中位 / P95 | C# 本轮耗时减少 |
|---|---:|---:|---:|
| 6.000 秒 PCMA | 1.271 / 1.333 秒 | 1.545 / 2.007 秒 | 17.8% |
| 5.616 秒 PCMU | 1.351 / 1.454 秒 | 1.604 / 1.851 秒 | 15.8% |
| 28.000 秒重复样例 | 5.339 / 5.518 秒 | 7.351 / 7.411 秒 | 27.4% |

本轮 C# 更快，但在线机器有明显换页与动态调频，不能当作稳定领先或持续多路容量保证。短 PCMA 的原生输出缺句末句号，另两组文本严格一致；模型、输入、散列、温度策略和原始数据见 [纯 CPU 对比](doc/cpu-native-comparison-20261012.md)。正式比较后已更新远端程序集并通过 HTTP/HTTPS 验收，RK3588 的 NEON/SDOT 内核选择保持原有行为。

### 同一 RTX 3060：C# Vulkan 对原生 GPU（2026-10-12 测量快照）

4 线程、相同模型和输入，计预热后完整推理调用中位数，包含 GPU 传输及同步。TTS 对照为原生 ORT CUDA 1.23.2，关闭 TF32、保持 FP32；已用剖析确认 CUDA 节点实际执行。

| TTS 模型 | 测量快照 C# Vulkan | 原生 ORT CUDA | C# / 原生耗时 |
|---|---:|---:|---:|
| Huayan 0 号 | 0.296 秒 | 0.020 秒 | 14.55 倍 |
| AISHELL3 50 号 | 0.174 秒 | 0.018 秒 | 9.63 倍 |

| ASR 输入 | 测量快照 C# Vulkan | C++ CPU 编码器 + CUDA 解码器 | C++ 原图 CUDA 适配 |
|---|---:|---:|---:|
| 6.000 秒 PCMA | 0.564 秒 | 0.312 秒 | 0.096 秒 |
| 5.616 秒 PCMU | 0.713 秒 | 0.308 秒 | 0.109 秒 |
| 28.000 秒重复样例 | 2.474 秒 | 1.485 秒 | 0.254 秒 |

**这组 C# Vulkan 测量没有同卡 GPU 速度优势。** 固定上游 C++ 的 Windows 入口原本默认 CPU；这里显式启用官方 llama.cpp CUDA 解码器，最后一列另将原编码计算图和权重放入 CUDA，属于明确标注的设备适配，不是原版开箱即用入口。ASR 保持 F16/Q8_0 权重，但 CUDA 内部计算精度可能与 Vulkan FP32 不同；双方都关闭提示词缓存，128 token 上限。短 PCMA 的原生输出少句末句号，另外两组文本完全相同；TTS 波形误差通过。不能将差异解释为语言优势。

GPU 对比共 105 次正式样本、23 次预热，保留全部慢样本。版本、适配范围、计时边界、可复现的基准生成器和原始记录见 [原生 GPU 对比](doc/gpu-native-comparison-20261012.md)。

GPU 表测于本次 CPU 默认布局修正前的 `1f817be` 推理实现；修正后只做 GPU 数值与转写回归，未重跑整套 GPU 性能统计，不能把新的 CPU 耗时拼入该 GPU 实验。

2026-09-30 RK3588 与 2026-10-10 Windows 的历史表格保留在 [性能记录](doc/performance.md) 和 [历史 Vulkan 记录](doc/vulkan-local-validation.md)，不与本次测量混算。RTX 3060 结果不能套用 RK3588 Mali；NPU 尚未接入正式模型。

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
