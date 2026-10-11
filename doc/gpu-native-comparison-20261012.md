# 同卡原生 GPU 对比（2026-10-12）

在同一 RTX 3060 上，当前 C# Vulkan 没有速度优势。两种 TTS 的原生 ORT CUDA 更快；ASR 的上游计算图 CUDA 适配也更快。此前 GPU 对原版 CPU 的表格只能描述历史实验，不能替代这次 GPU/GPU 对比。

## 条件与测量

- Windows x64、i7-12700、RTX 3060 12 GiB、驱动 610.88、.NET 10.0.12；CPU 4 线程，进程亲和性 `0x55`。
- C# 为本仓库 `1f817be` 的推理实现；没有为这次测量改动产品内核。两边使用相同固定模型与输入，模型和输入 SHA-256 保留在 [summary.json](benchmarks/20261012-gpu-native/summary.json)。
- TTS：Huayan 0 号与 AISHELL3 50 号，文本“您好，欢迎致电，请问有什么可以帮您？”，语速 1、两项随机度 0。
- ASR：相同 F16 编码器、Q8_0 解码器和三份 16 kHz f32 输入，最大 128 token。双方都禁用提示词缓存，每次重新预填充。不是旧 HTTP 服务的 512 token 对比。
- 计时包含完整模型执行、必要的 GPU 上传/下载和同步；ASR 还包含特征、编码和解码。模型加载、文件读取、输出文件写入、文本前端与音频封装不计入。

主比较按 C# Vulkan、原生 CUDA、原生 CUDA、C# Vulkan 的四个进程块交替运行，每块先预热一次。TTS 与两份短 ASR 每块测五次，长 ASR 每块测三次；另将 CPU 编码器/CUDA 解码器模式单独测量五/五/三次。共 **105 次正式测量、23 次预热**，保留全部慢样本。中位数取排序中间值，偶数取均值；P95 最近秩，在这些样本量下等于最大值。正式测量关闭剖析回调。

## TTS：原生 ORT CUDA

使用官方 `Microsoft.ML.OnnxRuntime.Gpu.Windows` **1.23.2**，CUDA Execution Provider，device 0，4 intra-op / 1 inter-op，顺序执行、全部图优化。保留必要 CPU 回退，关闭 TF32，保持 FP32 对照；TensorRT 未启用。CUDA/cuDNN DLL 使用本机已有 CUDA 12.9/cuDNN 9.10 运行库，没有改变系统驱动。TF32 的精度与配置含义见 [官方 CUDA EP 文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_tf32)。

完整 `Session.Run` 返回 CPU 输出后计时结束，包含最终 GPU 到 CPU 的复制，没有只计 CUDA 核函数或提前结束异步计时。

| 模型 | C# Vulkan 中位 / P95 | 原生 ORT CUDA 中位 / P95 | C# / 原生耗时 |
|---|---:|---:|---:|
| Huayan | 0.296 / 0.353 秒 | 0.020 / 0.025 秒 | 14.55 倍 |
| AISHELL3 50 号 | 0.174 / 0.297 秒 | 0.018 / 0.019 秒 | 9.63 倍 |

独立剖析验证 GPU 实际执行，而不是仅检测到 CUDA 库。两次诊断调用中，Huayan 有 3,796 个 CUDA 节点事件与 1,332 个 CPU 节点事件，AISHELL3 有 4,000 个 CUDA 与 766 个 CPU 节点事件。CPU 中包含形状等辅助操作。原始 ORT 剖析以 `ort-cuda-*-profile.json.gz` 保留；解压后 SHA-256 与 summary 一致。诊断剖析没有计入上述正式性能样本。

两边输出长度相同且全部有限，沿用最大绝对误差 `0.001`、RMSE `0.0001` 阈值：

| 模型 | 样本点数 | 最大绝对误差 | RMSE | 结果 |
|---|---:|---:|---:|---|
| Huayan | 69,376 | 2.25 × 10⁻⁵ | 1.18 × 10⁻⁶ | 通过 |
| AISHELL3 | 28,928 | 2.35 × 10⁻⁵ | 1.55 × 10⁻⁶ | 通过 |

这组结果表明当前 Vulkan 的逐节点传输、CPU 往返及图调度仍有较大差距。CPU 优化约三倍的结果继续成立，但不能据此宣称 GPU 也领先原生实现。

## ASR：上游 C++ 与 CUDA 的适配范围

固定 [Fun-ASR `0339018`](https://github.com/QwenAudio/Fun-ASR/tree/0339018ba74a7defa3b6b6a96718d17b816be77b) 和 [llama.cpp `8086439`](https://github.com/ggml-org/llama.cpp/tree/8086439a4cea94c71a5dfb8fe4ad1546aebd640f)，MSVC 19.44、CUDA Toolkit 12.6.20、Release、SM 86、CPU AVX2/FMA/F16C、OpenMP。

**固定上游 Windows 入口并没有开箱即用的 CUDA 编码器路径。** 其编码器直接初始化 CPU 后端，Windows 解码器的 `n_gpu_layers` 也是 0。因而本次明确区分：

1. **CPU 编码器 + CUDA 解码器**：原编码器保持 CPU，统一四线程，解码器通过官方 llama.cpp 参数设置 `n_gpu_layers=99`。加载日志确认所有 29 层卸载到 CUDA0，KV 和计算缓冲也在 GPU。
2. **原图 CUDA 适配**：保持上游 fbank、SAN-M/adaptor 计算图、提示词、采样和解码算法；将命名编码权重放到 CUDA 缓冲，切换计算后端与分配器，输入/输出显式传输。每个编码节点检查 CUDA 支持，无法支持则失败，不悄悄退回 CPU。记录确认包含 6,588 个节点的编码图交由 CUDA 后端执行；计数包含布局/视图元操作，不是 CUDA 核函数发射次数。音频特征与位置编码的准备仍在 CPU。

第二种是原计算图的设备适配，**不能标作原版未经修改的 Windows 默认程序**。可复现生成器位于 [NativeAsrFixture.cs](../tools/FunAsrNet.NativeGpuReference/NativeAsrFixture.cs)，参数/设备改动及计时包装可直接审阅；原始上游文件保留不变。原文件、适配文件、包装与二进制的散列记录在 summary。

| 输入 | C# Vulkan 中位 / P95 | 原图 CUDA 适配中位 / P95 | CPU 编码器 + CUDA 解码器中位 |
|---|---:|---:|---:|
| 6.000 秒 PCMA | 0.564 / 0.719 秒 | 0.096 / 0.104 秒 | 0.312 秒 |
| 5.616 秒 PCMU | 0.713 / 0.897 秒 | 0.109 / 0.135 秒 | 0.308 秒 |
| 28.000 秒重复样例 | 2.474 / 2.751 秒 | 0.254 / 0.264 秒 | 1.485 秒 |

ASR 的权重格式和量化级别相同，但上游 CUDA 默认算子可能使用 F16 激活/累积，C# Vulkan 使用 FP32；没有将两种内部浮点运算宣称为完全一致。PCMA 短句正文一致，CUDA 输出少一个句末句号；PCMU 与重复长样例全部正式文本完全一致。原始文本逐轮公开，未测可靠 CER/WER，也不以忽略标点后的匹配代替严格文本一致。

原图 CUDA 适配比 C# Vulkan 快约 5.87、6.52、9.73 倍。包括只卸载解码器的模式也更快。这是具体后端、图驻留、调度和内部计算精度的对比，不能简化为语言优劣或泛化到所有 GPU、自然长通话与并发容量。

## 复现与依赖隔离

`tools/FunAsrNet.NativeGpuReference` 不进入产品解决方案，GPU ORT 包不传递给核心或服务，也不部署到 RK3588。准备本机所需 CUDA/cuDNN DLL 后独立构建：

```powershell
dotnet build tools/FunAsrNet.NativeGpuReference -c Release
$gpuRef = 'tools/FunAsrNet.NativeGpuReference/bin/Release/net10.0/FunAsrNet.NativeGpuReference.dll'
dotnet $gpuRef models/zh_CN-huayan-medium.onnx results/huayan.i64 huayan 5 results/huayan-cuda.f32 --affinity 0x55
dotnet $gpuRef models/aishell3/model.onnx results/aishell3.i64 aishell3 5 results/aishell3-cuda.f32 --affinity 0x55
```

使用 `--profile-prefix results/profile` 单独确认节点分配。正式测量不加该参数。音素 ID 通过现有基准工具 `prepare-tts` 生成，见 [CPU 实测](tts-cpu-optimization.md)。

C++ 对照需上述固定 Fun-ASR 和 llama.cpp 源码。生成开发目录中的设备适配与包装，再在 Visual Studio x64 开发者终端构建：

```powershell
dotnet $gpuRef prepare-asr-reference artifacts/native-reference /path/to/Fun-ASR/runtime/llama.cpp/funasr-cli/funasr-cli.cpp /path/to/llama.cpp
cmake -S artifacts/native-reference -B artifacts/native-reference/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build artifacts/native-reference/build --target funasr-native-gpu-reference --parallel 8
artifacts/native-reference/build/funasr-native-gpu-reference.exe models/funasr-encoder-f16.gguf models/qwen3-0.6b-q8_0.gguf results/sip-short-pcma.f32 5 --encoder-gpu
```

不加 `--encoder-gpu` 即单独的 CPU 编码器/CUDA 解码器模式。环境 `FUNASRNET_DISABLE_PROMPT_CACHE=1` 用于 C# ASR 的相同缓存条件。模型、输入音频及原生构建产物不入库；正式 JSONL、完整统计、剖析、CUDA 卸载日志和散列见 [本次证据目录](benchmarks/20261012-gpu-native/)。首次加载和 cuDNN 预热显著慢于热态数值，不能用表格当作首次请求延迟。
