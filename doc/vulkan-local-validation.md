# Windows 本机 Vulkan 移植与原版对照（2026-10-10）

**本页为历史归档。** 本页 TTS 数字和倍数不代表 CPU 优化后的当前版本；2026-10-12 已补充同卡原生 ORT CUDA 和 C++ CUDA 对照，当前 C# Vulkan 没有速度优势，见 [新的 GPU 对比](gpu-native-comparison-20261012.md)。本页保留旧记录及当时未测 C++ GPU 的事实。

以下为 2026-10-10 重命名前实现的本机历史测量；公开版 `FunAsrNet.Vulkan` 保留同一计算内核。不是以新名称重新测得的全套结果。默认纯 C# CPU 路径保留；GPU 结果不能移用为 RK3588 Mali 的性能或兼容性结论。

## 本机与对照对象

| 项目 | 条件 |
|---|---|
| 系统 | Windows 11 专业工作站版，10.0.28000，x64 |
| CPU | Intel Core i7-12700，12 核/20 线程 |
| GPU | NVIDIA GeForce RTX 3060，12 GiB 显存，驱动 610.88 |
| 托管环境 | .NET SDK 10.0.401，运行时 10.0.12，Release JIT |
| 计算线程 | CPU 算子统一 4 线程；进程亲和性 `0x55`，即逻辑处理器 0、2、4、6 |
| ASR 权重 | 相同 `funasr-encoder-f16.gguf` / `qwen3-0.6b-q8_0.gguf` |
| TTS 权重 | 相同 Huayan / AISHELL3 ONNX 文件 |
| ASR 原版 | Fun-ASR `0339018ba74a7defa3b6b6a96718d17b816be77b` + llama.cpp `8086439a4cea94c71a5dfb8fe4ad1546aebd640f` |
| 原版 ASR 编译 | MSVC 19.51 Release，Ninja，AVX2/FMA/F16C、OpenMP，GPU 后端关闭 |
| TTS 原生参考 | ONNX Runtime 1.23.2 CPU，4 intra-op / 1 inter-op，顺序执行，全部图优化 |

此处“原版 ASR”是重新编译的 C++ 原版 CPU 路径；TTS 是同模型原生 ORT 算子参考，未恢复已删除的 sherpa-onnx/Python 完整服务。没有比较 C++ Vulkan/CUDA 或 ORT GPU，因此不能把 GPU 对 CPU 的差异说成 C# 相对 C++ 的语言优势。

Vulkan 参考基础设施来自 [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR)，固定提交 `dbbc21c535c6b2794ed1f217400d28596e8be091`。仅复用 C# API、设备、缓冲与提交基础设施；原许可和适配说明在 `src/FunAsrNet.Vulkan/Vendor`。该项目 OCR 图与 F16 着色器没有移植。

本机没有常见名称的 64 位系统加载器，通过 `FUNASRNET_VULKAN_LIBRARY` 显式指向已安装 NVIDIA DriverStore 中的 `vulkan-1-x64.dll`。实际枚举和计算均为 RTX 3060，不接受软件设备假装 GPU。没有替换、安装或重启显卡驱动。

## 实际结果

所有版本合计 **108 次正式测量，另有 30 次预热**。原始 JSONL、环境及统计在 [本机记录目录](benchmarks/20261010-vulkan-local/)。完整位数见 [summary.json](benchmarks/20261010-vulkan-local/summary.json)。

| ASR 样例 | C# CPU 中位数（秒） | Vulkan 中位数 / P95（秒） | C++ CPU 中位数 / P95（秒） | 相对 C++ 耗时减少 |
|---|---:|---:|---:|---:|
| 6.000 秒 PCMA | 9.689 | 0.487 / 0.489 | 0.749 / 0.760 | 34.9% |
| 5.616 秒 PCMU | 9.958 | 0.554 / 0.586 | 0.856 / 0.860 | 35.3% |
| 28.000 秒重复样例 | 39.879 | 2.345 / 2.545 | 2.913 / 3.060 | 19.5% |

三组样例全部正式转写文本相同，包括句末标点。Vulkan 的 RTF 分别为 0.081、0.099、0.084。这仅证明本机、这组样例、所述版本与条件下 Vulkan 混合路径快于原版 CPU，不保证所有文本或显卡都更快。

本机 C# CPU ASR 基线明显慢于 C++：现有代码在 x64 对交错布局使用通用标量分支，而 ARM64 有 NEON/SDOT 专用路径。这个问题仍保留，不通过故意禁用 CPU 功能制造对照优势；CPU 基线如实使用默认配置。该 17–20 倍的 Vulkan/当前 C# CPU 差异不能推广为 Vulkan 对充分优化的 x64 SIMD 的普遍收益。原版 C++ 本机已启用 AVX2/FMA/F16C。

| TTS 模型 | C# CPU 中位数（秒） | Vulkan 中位数 / P95（秒） | 原生 ORT CPU 中位数 / P95（秒） | Vulkan 相对 C# CPU |
|---|---:|---:|---:|---:|
| Huayan | 0.961 | 0.618 / 0.695 | 0.115 / 0.171 | 1.56 倍 |
| AISHELL3 50 号 | 0.409 | 0.220 / 0.242 | 0.045 / 0.057 | 1.86 倍 |

**TTS 尚未追平原生 ORT**，Vulkan 中位耗时仍是其 5.36 倍和 4.88 倍。混合模式中的逐节点传输、同步以及 CPU 剩余算子仍是限制；未实现完整 GPU 常驻图，不把局部 GEMM 或卷积的速度冒充完整 TTS 速度。

Vulkan 与 ORT 最终波形：Huayan 长度 69376，最大绝对误差 `1.85×10⁻⁵`、RMSE `9.47×10⁻⁷`；AISHELL3 长度 28928，最大绝对误差 `2.04×10⁻⁵`、RMSE `1.32×10⁻⁶`，均通过阈值。CPP 原始记录没有采集峰值 RSS，summary 对应值为 null，不表示零内存。

## 已实现的 GPU 路径

- FP32 分块矩阵乘法；F16 编码器权重按原值展开并上传，不另把 FP32 激活压成 F16，也不使用 TF32。
- FP32 一维卷积使用隐式展开，GPU 直接读取 NCL 输入和 OCK 权重，直接生成 NCL 输出，避免 CPU 全量 im2col 与结果转置。分组不为一的卷积仍使用原 CPU 内核。
- Q8_0 投影直接使用原整数权重与分组缩放，激活仍调用原 `TensorMath.Quantize`，每 32 个元素一组。整数点积与 FP32 缩放在 GPU 完成，保留原始/交错路径的缩放乘法次序。
- 合并 Q/K/V 与门控的独立 Q8 投影，一次量化与 GPU 同步后拆分 CPU 结果，减少设备往返。
- 只读权重缓存上限默认 2 GiB，采用最近使用淘汰；输入、GPU 输出和回读缓冲按容量复用。实例内串行保护原生池与单个提交状态，释放时等待 GPU 空闲并销毁相关资源。

其余算子仍由 C# 执行，包括部分注意力、归一化、形状与索引操作、逐元素以及转置卷积结果整理。大规模矩阵有实际 GPU 调用，小规模投影按阈值回退 CPU。不是整图全 GPU，也不宣称全链路零拷贝。

GPU 权重与 CPU 权重同时存在；Q8 投影合并目前还准备只读 CPU 行布局副本。每次跨设备数据上传、下载、缓存同步和栅栏等待都计入完整调用。冷启动、首次上传、管线初始化及驱动首次编译可能明显慢于预热后的轮次，原始数据分别保留 `load` 和 `warmup`，不能只把预热中位数当作第一次请求延迟。

## 精度与正确性

主算子保持 FP32 / F16 原权重值 / Q8_0 原权重与激活分组语义。不同硬件的浮点求和和乘加可能引入舍入差异，不承诺逐位相同。调试过程中曾发现 GPU 编码器的融合乘加与 Windows 原交错内核的分步乘加导致短句句末标点不同；已为该路径保留分步乘加和偏置次序，不把早期不同文本的试验纳入正式结果。

已有纯 C# 数值回归 **317843 项通过**，默认 CPU 路径没有引入 Vulkan 初始化。真实 RTX 3060 额外执行 **134474 项 GPU 参考检查通过**，覆盖 Q8 单行/批次/交错布局、投影合并、FP32 非整块尺寸、偏置、多批卷积、非对称填充、步长、膨胀和尾部。

完整 TTS 与同次 CPU、原生 ORT 的 float32 波形对照，要求样本数相同、全部有限值，并保持原阈值：最大绝对误差 ≤0.001、RMSE ≤0.0001。ASR 保留每轮文本和 token 数，核对这三份固定电话样例；没有足够标注真实 SIP 通话测 CER/WER，不把样例一致性宣传为所有文本准确率保证。

## 测量方法

三条路径按 C# CPU、Vulkan、原生 CPU、原生 CPU、Vulkan、C# CPU 的六块顺序执行，全部顺序运行。TTS 每块一次预热、五次正式测量，每种实现每个模型 10 次；两份短 ASR 每块一次预热、三次正式测量，每种实现各 6 次；28 秒长样例每块两次正式测量，每种实现各 4 次。保留慢样本，P95 使用最近秩，小样本下等于最大值。

ASR 固定 16 kHz float32 输入，由同一电话 WAV 解码，三份分别为 6.000 秒 PCMA、5.616 秒 PCMU、28.000 秒 PCMA 重复拼接。最大输出 token 均为 128，原版上下文/batch/ubatch 均为 2048，重复惩罚 1.0。C# 保留默认固定提示词缓存，C++ 保留上游每段清理 KV 的行为。计时完整 `Transcribe` / `asr_decode_window`，包含特征、编码、解码、必要复制与同步，不含读取输入和加载模型。长样例是人工重复拼接，不等同于自然 28 秒通话。

TTS 使用与板端一致的 int64 音素 ID，文本“您好，欢迎致电，请问有什么可以帮您？”，Huayan 0 号 / 22050 Hz、AISHELL3 50 号 / 8000 Hz、语速 1、两项随机度 0。计时完整 `ManagedOnnx.Run` / `InferenceSession.Run`，包含后端同步和托管调度，不含模型加载、文本前端、写波形和重采样。关闭随机度用于确定性数值校验，不代表页面默认参数性能。

Windows 无板端 thermal_zone 温度接口，GPU 温度与利用率由 NVIDIA 工具在前后记录，没有锁频、改电源策略或清文件缓存。正常桌面与其他应用仍存在，测试不是独占实验室环境，也不是长期并发容量验收。环境快照的 `runtime` 字段来自 SDK 版本命令；真正运行时版本以每块 `load.runtime` 为准。

## 使用方式

核心不引用 GPU 项目；只需为 GPU 使用方增加项目引用。后端由调用方创建和释放，须在开始推理前配置：

```csharp
using FunAsrNet;
using FunAsrNet.Vulkan;

using var gpu = new VulkanFloatBackend();
var asr = new FunAsr("models/funasr-encoder-f16.gguf", "models/qwen3-0.6b-q8_0.gguf", 4)
{
    FloatBackend = gpu
};
var result = asr.Transcribe(pcm, 128);

var model = new OnnxModel("models/zh_CN-huayan-medium.onnx");
var executor = new ManagedOnnx(model, new TensorMath(4)
{
    Optimized = false,
    FloatBackend = gpu
});
// inputs 按 Huayan 或 AISHELL3 的模型输入契约准备。
var audio = executor.Run(inputs, 1).Values.First().Data;

// 高层 TTS 同样可显式选择 GPU，仍由调用方管理后端资源。
var tts = new MandarinTts("models/zh_CN-huayan-medium.onnx", "assets/mandarin-lexicon.json", 4)
{
    FloatBackend = gpu
};
var speech = tts.Synthesize("您好，欢迎致电。");
```

本机复测工具：

```text
dotnet build tools/FunAsrNet.Benchmarks -c Release
dotnet tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll gpu-info
dotnet tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll gpu-check
dotnet tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll asr models input.f32 3 --vulkan --affinity 0x55
dotnet tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll tts models ids.i64 huayan 5 output.f32 --vulkan --affinity 0x55
dotnet build tools/FunAsrNet.NativeReference -c Release
```

不传 `--vulkan` 或不设置 `FloatBackend` 就保持 CPU。显式选择 Vulkan 时初始化错误会报告，不会把回退 CPU 的时间冒充 GPU 结果。统计记录中的真实设备、投影/卷积调用数及上传下载字节用于核验 GPU 确实执行。

GPU 程序来源嵌入 C#，glslang 16.6.0 将其编译为 Vulkan 1.1 SPIR-V，再嵌入 C# 字节码；运行时不需要编译器或原生自编译桥接。原生 C++ 参考只在临时目录构建，结果收集后移除，不作为产品依赖。

原始逐轮 JSONL 在公开版原样保存，散列见记录目录 `SHA256SUMS.txt`，Git 禁用证据文件的换行转换。原测量后仅补充高层 TTS 后端配置属性，公开版仅改命名和发布资料；计算内核未变化。含私人工作目录的原程序集来源日志不发布；模型与输入散列见 [模型](models.md) 和 [证据范围](performance.md)。`cpp-build-summary.txt` 保留实际平台配置；原版只统一线程与输出上限，并作 Windows NOMINMAX/UTF-8 编译适配，未重写推理算法。
