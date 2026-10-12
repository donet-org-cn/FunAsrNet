# C# 对比入口

此工具用 .NET 子进程串行执行、记录和汇总对照，不依赖 Python，不被服务或核心引用。先准备固定模型、f32/i64 输入、原版对照工具，以及优化前的独立基准目录。模型和原生二进制不随仓库分发；参照构建见 [GPU](../../doc/gpu-native-comparison-20261012.md) 和 [CPU](../../doc/cpu-followup-20261012.md)。

```powershell
dotnet build tools/FunAsrNet.Compare -c Release
dotnet build tools/FunAsrNet.Benchmarks -c Release
dotnet tools/FunAsrNet.Compare/bin/Release/net10.0/FunAsrNet.Compare.dll load --output artifacts/comparison-load --before artifacts/load-gpu-round6/before/FunAsrNet.Benchmarks.dll
dotnet tools/FunAsrNet.Compare/bin/Release/net10.0/FunAsrNet.Compare.dll cpu --output artifacts/comparison-cpu --before artifacts/load-gpu-round6/before/FunAsrNet.Benchmarks.dll
dotnet tools/FunAsrNet.Compare/bin/Release/net10.0/FunAsrNet.Compare.dll gpu --output artifacts/comparison-gpu --before artifacts/load-gpu-round6/before/FunAsrNet.Benchmarks.dll
```

- `--models`：默认 `artifacts/local-models`；`--inputs`：默认 `artifacts/native-gpu/inputs`。
- `--before` / `--after`：两版基准 DLL；默认后者为当前 Release 构建。`load` 使用同目录的核心 DLL。
- `--subjects`：逗号分隔，默认三份 ASR 加 `huayan,aishell3`。`load` 固定两份 GGUF。
- `--repeats`：每进程预热一次后的正式轮数，默认短 ASR/TTS 为 5，长 ASR 为 3；`load` 中为 ABBA 循环数，默认 3，每个样本均是新进程。
- `--cpp` / `--ort`：覆盖原版 ASR 可执行文件和 ORT 基准 DLL。GPU 模式要求原版编码器/解码器 CUDA 构建及 ORT CUDA 依赖可用。
- `aot` 模式接受 `--jit` / `--aot` 发布路径，顺序为 current/JIT/AOT/native/native/AOT/JIT/current。它用于新一轮热态对照，不重建历史报告中的额外启动及 prime 实验。

CPU/GPU 顺序为 before/after/native/native/after/before，统一四线程、Windows 亲和性 `0x55`、关闭 ASR 提示词缓存。输入输出路径用参数列表传递。输出目录必须不存在，十分钟子进程超时会终止该进程并报错。

`commands.json`、逐次 UTF-8 JSONL/stderr 和 `summary.json` 保留轮数、散列、中位数、最近秩 P95、各版本转写。轮数不符、优化前后 ASR 转写变化、TTS 波形校验不通过均报错；原版与 C# 的转写差异单独展示。GPU/CPU 计算精度及剩余 CPU 操作应以对应报告为准。

CPU/GPU 的 `summary.json` 还包含 `startupPhases`：`load` 是基准内模型/后端初始化，`firstInference` 是每个新进程的首次调用，`loadAndFirstInference` 先对同一进程的二者求和再统计。每版本通常只有两个启动样本，不能把热态调用数当成启动样本数。没有清空文件缓存或驱动缓存，不含进程启动和 Web 服务就绪；Vulkan 延迟上传的权重计入首次调用，原版按各自初始化边界记录。两种后端的加载数字不能脱离首次调用单独解读。

`load` 计时为新进程内两个 `Gguf` 构造器，不含进程启动、模型执行、GPU 首次上传或服务就绪；不清空操作系统文件缓存。完整 ASR 基准另记录 `load` 和预热记录，不能混用这些边界。`FUNASRNET_PROFILE_LOAD=1` 可在直接运行基准时诊断 GGUF 元数据、读取和 Q8 展开耗时；对比入口会清除继承的 `FUNASRNET_*` 实验选项。

旧报告中的 Python 编排已移除，历史原始测量和 source manifest 保留。当前工具产生新的汇总格式；复核历史数值直接使用原始 JSONL，不将今天的新测量冒充历史复现。
