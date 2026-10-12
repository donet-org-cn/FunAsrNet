# TTS CPU 优化与实测（2026-10-12）

后续本机 CPU 算子优化已将 Huayan 降到 0.184 秒、AISHELL3 降到 0.075 秒，同模型 ORT CPU 为 0.103、0.044 秒，仍慢 1.79、1.71 倍。完整前/后/原生交替比较见 [后续记录](cpu-followup-20261012.md)，本页保留首次三倍优化的实验。

本次直接优化纯 C# 的模型执行，没有增加原生推理依赖，也没有更换模型、缩短文本或改变权重精度。Windows x64 的两个固定模型均提速约 3 倍，与原生 ORT CPU 的差距从约 9 倍缩小到约 3 倍，仍未追平 ORT。

随后完成的 [纯 CPU 复测](cpu-native-comparison-20261012.md) 合并两批 TTS 全部样本，测得 Windows Huayan 0.321 秒、AISHELL3 0.138 秒，仍比各自 ORT CPU 慢约 3.03–3.10 倍；本页保留首次优化前/后的三方实验。该后续实验还修正并复测了 x64 ASR 的默认布局选择，不改变这里的 TTS 算法。

## 相同模型、输入与线程的对比

环境为 Intel Core i7-12700、Windows x64、.NET SDK 10.0.401、运行时 10.0.12，CPU 4 线程、进程亲和性 `0x55`；对照 ORT 为 1.23.2 CPU Provider、intra-op=4、inter-op=1、顺序执行、全部图优化。优化前使用本仓库提交 `fcfb20af9599383e03d19835dd083acfbe1b850a` 的独立 Release 构建，优化后使用本次工作区构建。

下表为模型 `Run` 的热态中位数 / P95，单位秒。每个模型、每种实现各测 10 次；最近秩 P95 在该样本量下为最大值。

| 模型 | 优化前 C# CPU | 优化后 C# CPU | ORT CPU | C# 提速 | 优化后 / ORT |
|---|---:|---:|---:|---:|---:|
| Huayan 0 号 | 0.975 / 1.048 | 0.317 / 0.422 | 0.102 / 0.107 | 3.08 倍 | 3.12 倍 |
| AISHELL3 50 号 | 0.407 / 0.491 | 0.132 / 0.153 | 0.043 / 0.045 | 3.07 倍 | 3.09 倍 |

文本固定为“您好，欢迎致电，请问有什么可以帮您？”，语速 1、两项随机度 0。Huayan 输出 69,376 点、22,050 Hz；AISHELL3 输出 28,928 点、8,000 Hz。输入 ID 的 SHA-256 与历史基准一致，模型文件散列也已核对。每个模型按“优化前、优化后、ORT、ORT、优化后、优化前”顺序运行六个独立进程块，各块先预热一次，再测五次。共 60 次正式测量、12 次预热；不删除慢样本，不在正式测量中开启节点性能回调。

计时包含完整模型执行和托管张量调度；不含模型加载、音素编码、文件写入、音频封装与重采样。加载和首轮预热另有记录。Windows 无可用温度/频率接口，未锁频，不能把小样本 P95 当作长期尾延迟或并发服务容量。

原始逐轮 JSONL、输入 ID 清单和完整统计见 [本次记录](benchmarks/20261012-tts-cpu/summary.json)。本次为 CPU 测量，不能与历史 Vulkan 结果直接拼成同一次实验。另已在 RK3588 完成与旧托管部署及原生 ORT 的相同条件复测，结果见 [ARM64 更新与实测](rk3588-update-20261012.md)。两台机器的数字分开统计。

## 主要改动

- 长的一维卷积沿连续时间维使用 AVX2/FMA，四个输出通道复用十六个时间点，避免展开整段 `im2col` 和转置结果。分组、膨胀、填充及通道/时间尾部保留边界检查。
- 短卷积、矩阵乘法及转置卷积的线性投影使用 AVX2 激活面板，复用输入读取，保持一份原始 FP32 权重。内积按 32 项分段累加，通过原有数值容差。
- 时间掩码、通道常量及普通广播按连续末维处理，减少逐元素除余索引。连续切片与末维填充按整行复制，末两维转置分块处理，单轴归约保留原累加顺序。
- TTS 的同步执行器复用有界临时工作区，每实例最多保留 64 MiB。按底层数组的命名引用数处理 `Reshape`、`Identity` 等视图，返回结果脱离回收集合；开启 `Trace` 时禁用复用。模型权重和外部输入不归该工作区所有。
- AVX2 CPU 默认复用同步线程组；空闲超过 5 ms 后休眠，多个上下文按线程数共享线程组。
- ARM64 的休眠通知握手使用完整的存储/加载屏障，避免漏掉唤醒；固定计算核心时，TTS 调用线程也绑定到线程组剩余的核心，并在调用结束后恢复亲和性。

诊断开关：`FUNASRNET_LEGACY_KERNELS=1` 禁用新增 AVX2 矩阵内核，`FUNASRNET_LEGACY_CONV=1` 禁用直接卷积，`FUNASRNET_SPIN_WORKERS=0` 使用 `Parallel.For`，`FUNASRNET_REUSE_TTS_BUFFERS=0` 禁用 TTS 临时工作区。它们只用于定位各项收益；这些开关不会撤销广播、切片、填充等改动，不能当作整个旧版本的性能基线。`ManagedOnnx.ReuseBuffers` 也可按执行器设置。

## 正确性

优化后与原生 ORT 的最终波形长度相同、全部为有限数值。预先沿用历史标准：最大绝对误差不超过 `0.001`、RMSE 不超过 `0.0001`。

| 模型 | 最大绝对误差 | RMSE | 结果 |
|---|---:|---:|---|
| Huayan | 1.69 × 10⁻⁵ | 9.26 × 10⁻⁷ | 通过 |
| AISHELL3 | 3.30 × 10⁻⁵ | 1.83 × 10⁻⁶ | 通过 |

优化前/后波形比较也通过。Windows x64 和 Linux ARM64 的 Release 数值回归均通过 685,701 项检查，包含分组卷积独立参考、矩阵尾部、广播、裁剪、转置、归约、数组别名、跨调用输出、观察回调、异常后的工作区恢复、独立上下文并发和线程组休眠唤醒压力测试。未以放宽原有矩阵误差阈值换取通过。数值接近不替代人工听感评估。

临时张量复用也影响 Vulkan 的 CPU 回退部分，因此补做了 134,474 项 GPU 数值检查，并核对两种模型的 Vulkan 输出与 ORT 波形，全部通过；该检查不属于上表的正式 CPU 性能样本。

## 复测命令

按 [模型清单](models.md) 准备模型，独立构建原生对照工具。以下 PowerShell 示例不需要 ASR 权重或 WAV 文件：

```powershell
dotnet build FunAsrNet.sln -c Release
dotnet build tools/FunAsrNet.NativeReference -c Release
$runner = 'tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll'
$native = 'tools/FunAsrNet.NativeReference/bin/Release/net10.0/FunAsrNet.NativeReference.dll'
dotnet $runner prepare-tts models assets/mandarin-lexicon.json huayan '您好，欢迎致电，请问有什么可以帮您？' results/huayan.i64
dotnet $runner prepare-tts models models/aishell3/lexicon.txt aishell3 '您好，欢迎致电，请问有什么可以帮您？' results/aishell3.i64
dotnet $runner tts models results/huayan.i64 huayan 5 results/huayan.f32 --threads 4 --affinity 0x55
dotnet $native models/zh_CN-huayan-medium.onnx results/huayan.i64 huayan 5 results/huayan-ort.f32 --affinity 0x55
dotnet $runner compare results/huayan-ort.f32 results/huayan.f32
dotnet $runner tts models results/aishell3.i64 aishell3 5 results/aishell3.f32 --threads 4 --affinity 0x55
dotnet $native models/aishell3/model.onnx results/aishell3.i64 aishell3 5 results/aishell3-ort.f32 --affinity 0x55
dotnet $runner compare results/aishell3-ort.f32 results/aishell3.f32
```

`--profile` 用于寻找节点热点，带回调的耗时应与正式测量分开。TTS 基准输出 `allocatedBytes`，统计当前进程托管分配，不包含原生 ORT 的内部内存，不能用该字段跨引擎比较内存。此次原生工具从已有包构建时采用 `-p:NuGetAudit=false`，因为漏洞查询的 TLS 连接失败；原生包版本保持 1.23.2，产品项目仍不引用该包。
