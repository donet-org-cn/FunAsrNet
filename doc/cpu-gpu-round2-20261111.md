# CPU/GPU 后续优化（2026-11-11，第二轮）

继续在[上一轮版本](cpu-gpu-original-20261111.md)上优化。CPU 六行 FP32 分块和 Q8 整数归约已落地；GPU 增加长序列 FP32 注意力与适合单 token 投影的宽子组。权重精度不变，CPU 改动保持严格位一致，GPU 的新浮点归约按独立数值参考检查。

## CPU 完整调用

i7-12700，四线程、亲和性 0x55，.NET 10.0.15，SDK 10.0.204，关闭提示词缓存，最多 128 token。前后版本按 ABBA 启动进程，每进程一次预热、三次正式调用，每格六次正式调用中位数，单位秒。

| 权重与输入 | 本轮之前 | 本轮之后 | 耗时减少 |
|---|---:|---:|---:|
| 原始权重 FP32，6 秒 | 1.769104 | 1.731897 | 2.1% |
| 原始权重 FP32，24 秒 | 7.056175 | 6.460459 | 8.4% |
| F16/Q8，6 秒 | 0.642504 | 0.644612 | -0.3% |
| F16/Q8，24 秒 | 2.505340 | 2.406200 | 4.0% |

CPU 编码器阶段在四组中减少约 8%–12%；Q8 短句完整调用基本持平。原始权重长句总耗时的 8.4% 是该批观察值，生成阶段也有背景波动，不能全部归因于本轮矩阵改动。桌面仍有其他任务运行；不把几个百分点的变化推广到所有环境。

## GPU 完整调用与复核

RTX 3060 12 GiB，双方 GPU 权重缓存均为 4 GiB。最初将所有大小的注意力都迁入 GPU，短句变慢，因此最终默认将不足 160 帧的注意力留在 CPU。三十二通道的 Q8 候选没有获得收益，已撤回；最终仅 FP32 矩阵向量选择宽子组。

为了处理普通桌面负载下的波动，保留以下三批结果，单位秒，各格均为对照 → 新路径的中位数：

| 权重与输入 | 跨进程版本对照，各 6 次 | 同进程开关复核，各 6 次 | 扩大样本复核，各 16 次 |
|---|---:|---:|---:|
| 原始权重 FP32，6 秒 | 0.328048 → 0.321667 | 0.332468 → 0.306579 | 0.342581 → 0.315674 |
| 原始权重 FP32，24 秒 | 1.027819 → 1.054303 | 1.088950 → 1.016385 | 1.099159 → 0.979950 |
| F16/Q8，6 秒 | 0.281419 → 0.283744 | 0.275882 → 0.288572 | 0.287019 → 0.281464 |
| F16/Q8，24 秒 | 0.914984 → 0.851568 | 0.971993 → 1.009438 | 0.932825 → 0.864969 |

跨进程组比较冻结的前后程序集，普通优先级。两批同进程组使用同一个模型实例、相同最终 CPU 库、两个 GPU 后端，分别关闭/开启本轮 GPU 特性，每个后端各有 4 GiB 缓存。普通同进程组采用一次预热、ABBA 三轮；扩大样本组采用两次预热、ABBA 八轮，并仅把基准进程临时设为 AboveNormal 优先级，产品的默认优先级没有改变。

扩大样本组的 Q8 长句耗时减少 **7.3%**，原始权重短句/长句减少 **7.9%/10.8%**；长句编码器阶段分别减少约 15.5%/18.4%。原始权重 GPU 在普通同进程组也有收益，但跨进程长句一批变慢 2.6%；Q8 长句在普通同进程组变慢 3.9%。这些不利结果全部保留，不把某一批的百分比当作稳定保证。Q8 短句的两个新 GPU 内核调用计数均为零，仍出现 -4.6% 到 +1.9% 的变化，可见小幅差异处在桌面测量噪声范围内。

[主版本对照统计](benchmarks/20261111-cpu-gpu-round2/summary.json)、[普通同进程复核](benchmarks/20261111-cpu-gpu-round2/gpu-paired/summary.json)、[扩大样本复核](benchmarks/20261111-cpu-gpu-round2/gpu-control/summary.json) 包含各组最小值、最大值和分阶段计时。没有将不同批次的前后秒数拼接计算收益。

## 实现

- CPU FP32：一次计算六个输出行和十六帧，使用十二个向量累加器共享激活加载，保留原有 128 项分段及 FMA 次序；尾行只使用有效地址并只写有效输出。已有四行路径作为回退保留，不增加常驻模型副本。
- CPU Q8：四行整数点积先在两个 128 位半区归约，再合并半区，减少提取与加法指令；浮点缩放、逐块相加及整数结果均不变，覆盖 AVX-VNNI 和普通 AVX2。
- GPU 注意力：非因果 QK、稳定 Softmax、加权值在一个计算内核完成。至多 1,024 帧时四个查询共享键和值；1,025–4,096 帧采用单查询内核。只上传 Q/K/V 和回读最终结果，不在全局显存中建立完整的平方分数矩阵。设备不支持子组或共享内存不足时回退，默认小于 160 帧保留 CPU。
- GPU FP32 矩阵向量：一至四帧、内积至少 1,024 项且设备支持时，三十二通道分担一行，提高小矩阵的工作组数量。其他形状和设备保留八通道或通用管线，调度数量限定在通用设备保证的范围内。

新的注意力经 IAttentionComputeBackend 接入既有 TensorMath.Attention，使用 Vulkan 后端时自动选择。单独的新内核没有要求调用方改变模型或推理 API。子组同步与交换依据 [Khronos subgroup 规范](https://raw.githubusercontent.com/KhronosGroup/GLSL/main/extensions/khr/GL_KHR_shader_subgroup.txt)，全部着色器保留 GLSL 及对应的嵌入 SPIR-V。

## 精度和验证

原始 BF16/FP32 checkpoint 无损展开为 FP32 GGUF，沿用既有 F16 KV 缓存；未新增权重量化、FP16 激活或 TF32。模型来源、导出方式及 SHA-256 见[上一轮精度说明](cpu-gpu-original-20261111.md)。这仍不是 PyTorch/BF16 原始计算图对照。

CPU 六行矩阵与原实现逐位比较，覆盖所有六行余数、尾帧、奇数内积及分段边界；Q8 继续通过含 -128 权重的严格整数参考。GPU 注意力使用独立双精度点积、指数、概率及加权值参考，容差为 5e-5 × max(1, |参考值|)，该误差指标的实测最大值约 4.91e-6。FP32 GPU 矩阵向量沿用 2e-4 混合容差，批量矩阵与分组/独立投影的既有严格检查保留。新 FP32 归约与 Softmax 不宣称和 CPU/旧 GPU 中间结果逐位相同。

| 验证路径 | 通过数量 |
|---|---:|
| CPU .NET 10 / .NET 11 / 禁用 AVX-VNNI，各自 | 9,552,345 |
| 禁用 AVX2 / 禁用全部硬件 intrinsic，各自 | 9,179,904 |
| GPU 默认 / 无整数点积 / 单查询与窄子组 / .NET 11，各自 | 1,453,760 |
| 禁用 GPU 注意力 | 410,035 |

Release 全解决方案编译零警告、零错误；147 个文件、475 个方法的中文注释与格式检查通过，九个着色器变体与源码离线重编译结果逐字节一致。注意力测试包含 1、2、3、17、100、129、257、400、1,024、1,025、4,096 帧，尾通道、尾查询、重复上传、形状拒绝和浮点权重逐出重传。

四批合计 **320 次正式模型调用、72 次预热**，同一输入的文本和 token 数全部一致，详见[逐批数量与输出核对](benchmarks/20261111-cpu-gpu-round2/output-consistency.json)。输入为同一 6 秒样本及其四次重复的 24 秒版本，不代表多语料字错误率。全部性能在 .NET 10 测量；.NET 11 本轮用于正确性回归。未重新测试 C++、CUDA、TTS 完整模型、ARM64 或其他显卡，不能从这些数据得出相应速度结论。

## 复现和回退

原始记录、脚本、相同计时入口、程序集散列、基线和最终计算源码归档见[证据目录](benchmarks/20261111-cpu-gpu-round2)。初轮 96 次包含最终保留的 48 次 CPU 测量及 48 次 GPU 候选测量；随后 48 次 GPU 版本复测、48 次普通同进程复核、128 次扩大样本复核，均保留慢样本。最终 CPU 程序集 SHA-256 与 CPU 正式计时使用的程序集相同。

[baseline-source.zip](benchmarks/20261111-cpu-gpu-round2/baseline-source.zip) 和 [final-source.zip](benchmarks/20261111-cpu-gpu-round2/final-source.zip) 包含核心、Vulkan、基准工具源码及构建配置，可用其中的基准项目复建；模型和输入文件需另行准备。基线为本轮开始时含前轮优化的工作区源码，并非 Git HEAD。归档文件校验见 [SHA256SUMS.txt](benchmarks/20261111-cpu-gpu-round2/SHA256SUMS.txt)。初轮中间 GPU 候选保留日志与冻结程序集散列，完整二进制仍在本机 artifacts。GPU 微基准 lanes32 日志包含后来撤回的 Q8 宽子组，不能把该候选当作当前默认实现。

本机安装的 SDK 为 10.0.204，低于 global.json 指定的 10.0.300，因此测量构建显式调用该 SDK；满足仓库 SDK 要求的环境也可直接使用 dotnet build。下面对应本机的构建、检查和运行命令：

```powershell
dotnet 'C:/Program Files/dotnet/sdk/10.0.204/dotnet.dll' build FunAsrNet.sln -c Release
dotnet tests/FunAsrNet.Checks/bin/Release/net10.0/FunAsrNet.Checks.dll
dotnet tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll gpu-check
$env:FUNASRNET_DISABLE_PROMPT_CACHE = '1'
dotnet tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll asr models input.f32 5 --vulkan --gpu-cache-mib 4096 --affinity 0x55
Remove-Item Env:FUNASRNET_DISABLE_PROMPT_CACHE
./tools/CompileVulkanShaders.ps1 -Compiler path/to/glslang.exe -VerifyOnly
```

原始权重另用 --encoder 和 --decoder 指定两份 FP32 GGUF，CPU 测量去掉 --vulkan。input.f32 为 16 kHz、单声道 little-endian float32。着色器编译器为 glslang 16.6.0，仅离线维护使用。

诊断开关应在创建模型/后端之前设置：FUNASRNET_AVX_ROWS6=0 回到四行 CPU 矩阵；FUNASRNET_VULKAN_ATTENTION=0 关闭 GPU 注意力；FUNASRNET_VULKAN_ATTENTION_TILE=0 使用单查询内核；FUNASRNET_VULKAN_WIDE=0 关闭 FP32 宽子组。硬件回退使用 DOTNET_EnableAVXVNNI=0、DOTNET_EnableAVX2=0 或 DOTNET_EnableHWIntrinsic=0，名称来自 [运行时配置定义](https://github.com/dotnet/runtime/blob/release/10.0/src/coreclr/inc/clrconfigvalues.h)。
