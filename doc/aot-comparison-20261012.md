# Windows CPU：JIT 与 Native AOT 对比（2026-10-12）

已经编译并运行一版 Windows x64 Native AOT CPU 基准。**本轮主要收益是启动与首个结果时间，连续推理没有整体提速。** 相同入口下，两条短 ASR 基本持平，长 ASR 的 AOT 中位数慢 4.9%；Huayan、AISHELL3 的 AOT 中位数分别减少 2.3%、6.4%。这些是小样本描述，不能宣称统计显著或推广到其他输入。

本次新增 `tools/FunAsrNet.AotBenchmarks`，同一份入口分别发布 JIT 与 AOT，两者都调用现有核心算法。核心改动仅为中文词典和音素表生成 JSON 元数据，没有修改模型、量化、算子或线程调度。**这是 CPU 模型基准的 AOT 验证，Web 服务尚未发布为 AOT；没有 GPU、RK3588 或远端部署测试。**

## 预热后的模型调用

| 项目 | 原有入口 JIT | 相同入口 JIT | AOT | 原生 CPU | AOT / 同入口 JIT | AOT / 原生 |
|---|---:|---:|---:|---:|---:|---:|
| ASR 6.000 秒 | 0.975 秒 | 0.909 秒 | 0.914 秒 | 0.736 秒 | 1.005 | 1.241 |
| ASR 5.616 秒 | 0.991 秒 | 0.971 秒 | 0.971 秒 | 0.837 秒 | 1.000 | 1.160 |
| ASR 28.000 秒 | 3.338 秒 | 3.456 秒 | 3.626 秒 | 2.939 秒 | 1.049 | 1.234 |
| TTS Huayan | 0.156 秒 | 0.159 秒 | 0.156 秒 | 0.102 秒 | 0.977 | 1.524 |
| TTS AISHELL3 | 0.071 秒 | 0.071 秒 | 0.067 秒 | 0.045 秒 | 0.936 | 1.480 |

全部为同轮中位数；比值小于 1 表示 AOT 更快。“相同入口 JIT”是判断发布方式影响的主要对照。“原有入口”保留第三轮基准二进制，作为额外检查，散列见 [manifest.json](benchmarks/20261012-aot/manifest.json)。没有拿前轮的原生秒数与本轮 AOT 拼表。

本机存在背景负载且未锁定频率：原有入口 6 秒 ASR 两个进程块中位数分别为 1.094、0.884 秒；AOT 长 ASR 两块为 3.449、3.745 秒，均保留。这些波动限制了小幅收益的结论。AOT 的 ASR 仍为同轮原生 CPU 的 1.16–1.24 倍耗时，TTS 为 ORT CPU 的 1.48–1.52 倍，**没有全面追平原生 CPU**。

## 新进程、加载与首次结果

不加载模型、仅输出 `cpu-info` 的进程，从父进程启动计时到收到配置记录，中位数为 **JIT 51.4 ms、AOT 8.9 ms**。每种实现先运行一个不计入统计的进程，再按 JIT/AOT/AOT/JIT 顺序重复五组，每种各十次正式样本。此项没有设置模型基准的 CPU 亲和性。

带模型的结果如下，每格是两个新进程的中位数：

| 项目 | JIT 加载 | AOT 加载 | JIT 进程至首个结果 | AOT 进程至首个结果 |
|---|---:|---:|---:|---:|
| ASR 6.000 秒 | 3.742 秒 | 1.314 秒 | 4.886 秒 | 2.253 秒 |
| ASR 5.616 秒 | 3.477 秒 | 1.295 秒 | 4.671 秒 | 2.296 秒 |
| ASR 28.000 秒 | 3.498 秒 | 1.343 秒 | 7.129 秒 | 4.769 秒 |
| TTS Huayan | 0.095 秒 | 0.074 秒 | 0.457 秒 | 0.252 秒 |
| TTS AISHELL3 | 0.064 秒 | 0.037 秒 | 0.336 秒 | 0.131 秒 |

“加载”是基准内部的引擎构造、权重准备和强制 GC 时间，排除输入文件读取；“进程至首个结果”由父进程计时，包含进程启动、输入读取、加载、首次模型调用以及输出记录。不等于完整 HTTP 服务首个请求，也不等于浏览器开始录音至识别文字的时间。**未清空操作系统文件缓存，因此不能称为磁盘冷启动。** 两进程样本不足以估计稳定启动尾延迟，完整观测见 [process-observations.json](benchmarks/20261012-aot/process-observations.json)。

## 内存与产物

下表取各实现两个进程在首次及正式推理记录中的峰值 RSS 最大值，包含模型和初始化内存，不能当作单次推理新增内存：

| 项目 | JIT 峰值 RSS | AOT 峰值 RSS |
|---|---:|---:|
| ASR 6.000 秒 | 2562.0 MiB | 2283.2 MiB |
| ASR 5.616 秒 | 2558.5 MiB | 2305.5 MiB |
| ASR 28.000 秒 | 2579.8 MiB | 2633.9 MiB |
| TTS Huayan | 341.2 MiB | 335.1 MiB |
| TTS AISHELL3 | 149.2 MiB | 124.2 MiB |

短 ASR 和 AISHELL3 的峰值较低，长 ASR 的 AOT 峰值反而较高，不能声称所有场景都省内存。AOT 可执行文件为 3,485,696 字节，不含 PDB、XML 文档和模型；没有将它与不包含共享运行时的 JIT 文件体积直接比较。

## 条件、顺序与正确性

- Windows x64、Intel Core i7-12700、SDK 10.0.401、运行时/ILCompiler 10.0.12；四线程、模型测试亲和性 `0x55`，全部关闭 GPU。JIT 保持项目现有 `TieredCompilation=false`，不启用 ReadyToRun。两版均为工作站 GC。
- AOT 设置 `PublishAot=true`、`OptimizationPreference=Speed`、`IlcInstructionSet=native`。运行记录确认 `dynamicCodeSupported=false`，AVX2、FMA、AVX-VNNI 全部为 true；同入口 JIT 的三项指令也全部启用。未压掉 AOT/裁剪告警。
- 每个输入顺序为原有入口、同入口 JIT、AOT、原生、原生、AOT、同入口 JIT、原有入口。每个独立进程预热一次；短 ASR/TTS 每块正式五次，长 ASR 每块正式三次。合计 **184 次正式模型测量、40 次首次调用/预热**，另有前述二十次正式启动测试及两次启动预运行。
- 每实现短输入十次、长输入六次，P95 采用最近秩，此样本量下等于最大值。全部慢样本保留；没有与数值检查、构建或另一推理进程并行跑基准。
- ASR 采用同一 F16/Q8_0 模型和 16 kHz 浮点 PCM，关闭提示词缓存、最大 128 token；计完整 Transcribe，包括特征、编码和解码，排除模型加载和输入读取。
- ASR 原生参照沿用上一轮同一二进制：Fun-ASR `0339018ba74a7defa3b6b6a96718d17b816be77b`、llama.cpp `8086439a4cea94c71a5dfb8fe4ad1546aebd640f`，MSVC 19.44、AVX2/FMA/AVX-VNNI、OpenMP，GPU 层数零。
- TTS 使用相同音素 ID、Huayan 0 号/AISHELL3 50 号、语速 1、两项随机度 0、种子 1；计模型 Run，不含文本前端、封装及重采样。对照 ORT 1.23.2 CPU，intra-op 4、inter-op 1、顺序执行、全部图优化。
- 四方全部 ASR 调用的文本严格相同；两种模型 JIT/AOT 每次波形散列稳定，保存的波形逐样本相同。与 ORT 比较长度一致且全部有限，最大绝对误差至多 `2.42e-05`，通过既有 `maxAbs <= 0.001`、`RMSE <= 0.0001` 阈值。
- JIT 与 AOT 分别通过 **1,796,447 项数值及前端检查**。新增前端检查覆盖中文/转义词典键、变调和多 ID 音素表，确认新 JSON 源生成在真正的 AOT 进程中可运行。完整解决方案构建零告警、零错误，中文注释和格式检查通过。

## 本机复建

Windows 需 .NET 10 SDK 和 C++ 本机构建工具，详见 [Native AOT 官方要求](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/)。AOT 的 [Speed 优化偏好](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/optimizing) 不保证每个工作负载都更快；[`native` 指令集目标](https://github.com/dotnet/runtime/blob/main/src/coreclr/nativeaot/docs/optimizing.md) 针对构建机 CPU，产物不能直接用于 RK3588 或缺少这些指令的 x64 机器。

```powershell
dotnet publish tools/FunAsrNet.AotBenchmarks -c Release -r win-x64 --self-contained false -p:PublishAot=false -p:PublishReadyToRun=false -o artifacts/aot-comparison/jit
dotnet publish tools/FunAsrNet.AotBenchmarks -c Release -r win-x64 -p:PublishAot=true -p:IlcInstructionSet=native -o artifacts/aot-comparison/aot

& artifacts/aot-comparison/aot/FunAsrNet.AotBenchmarks.exe cpu-info
$env:FUNASRNET_DISABLE_PROMPT_CACHE = '1'
& artifacts/aot-comparison/aot/FunAsrNet.AotBenchmarks.exe asr artifacts/local-models artifacts/native-gpu/inputs/sip-short-pcma.f32 5 --threads 4 --affinity 0x55
& artifacts/aot-comparison/aot/FunAsrNet.AotBenchmarks.exe tts artifacts/local-models artifacts/native-gpu/inputs/huayan.i64 huayan 5 artifacts/aot-comparison/huayan.f32 --threads 4 --affinity 0x55
```

本机首次恢复因审计源不可达失败；正式构建使用已缓存的 10.0.12 编译包，并在命令上加 `-p:NuGetAudit=false`，未改全局 NuGet 设置。普通 JIT 和 AOT 发布目录独立；解决方案默认发布方式不变。

完整交错协议脚本为 [run-comparison.py](../tools/FunAsrNet.AotBenchmarks/run-comparison.py)，从仓库根目录运行：

```powershell
python tools/FunAsrNet.AotBenchmarks/run-comparison.py --output artifacts/aot-comparison/repeat-01
```

脚本需要本机模型、相同 f32/i64 输入和已构建的原生参照；输出目录必须尚不存在，避免覆盖证据。可用参数指定路径。原生构建方法见 [CPU 后续对比](cpu-followup-20261012.md)。输入与权重未公开分发，不能声称公开仓库一键复现全部实验。

源基础提交为 `6fa904a`，本轮源码、模型、输入和二进制散列见 [manifest.json](benchmarks/20261012-aot/manifest.json)，统计、各次调用和正确性见 [summary.json](benchmarks/20261012-aot/summary.json)，证据校验见 [SHA256SUMS.txt](benchmarks/20261012-aot/SHA256SUMS.txt)。现有服务和持续推理优化仍应以各自实测为准。
