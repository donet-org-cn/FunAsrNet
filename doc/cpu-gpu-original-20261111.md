# CPU、GPU 与原始权重优化（2026-11-11）

本轮在已经包含前序 FFT、RoPE、词表扫描及四帧 AVX-VNNI 优化的工作区上继续优化。RTX 3060 的完整 ASR 调用中，Q8 路径耗时减少 **15.4%–21.4%**，原始权重路径减少 **59.8%–66.3%**。CPU 的 Q8 长句在本批减少 7.0%，短句与原始权重整体接近持平；不能据此宣称 CPU 全面提速。

## 完整调用结果

Windows、i7-12700、RTX 3060 12 GiB、驱动 610.88；.NET 10.0.15，SDK 10.0.204。四线程、亲和性 0x55、最大 128 token、关闭固定提示词缓存；前后版本均使用 4 GiB GPU 权重缓存。每组按前/后、后/前顺序启动进程，每进程 1 次预热、3 次正式调用，表中每格为 6 次正式调用中位数，单位秒。

| 后端与权重 | 输入 | 优化前 | 优化后 | 耗时减少 |
|---|---|---:|---:|---:|
| CPU，F16 编码器 / Q8 解码器 | 6 秒 | 0.711828 | 0.715051 | -0.5% |
| CPU，F16 编码器 / Q8 解码器 | 24 秒 | 2.832290 | 2.634353 | 7.0% |
| GPU，F16 编码器 / Q8 解码器 | 6 秒 | 0.342357 | 0.289661 | 15.4% |
| GPU，F16 编码器 / Q8 解码器 | 24 秒 | 1.175621 | 0.924282 | 21.4% |
| CPU，原始权重展开为 FP32 | 6 秒 | 1.987721 | 1.994239 | -0.3% |
| CPU，原始权重展开为 FP32 | 24 秒 | 7.238452 | 7.064376 | 2.4% |
| GPU，原始权重展开为 FP32 | 6 秒 | 0.824171 | 0.331045 | 59.8% |
| GPU，原始权重展开为 FP32 | 24 秒 | 3.107507 | 1.045741 | 66.3% |

96 次正式、32 次预热，保留全部慢样本。计时包含特征、编码、解码、CPU 算子、GPU 上传、同步及回读，排除模型加载和输入文件读取。24 秒输入是同一 6 秒样本重复四次；不是独立长语料。前后各精度、各后端的文本和生成 token 数一致，分别为 7 和 28 token。文本一致属于样例回归，不代表完整字错误率评估。

GPU 原始权重的热态速度为此前的 2.49/2.97 倍，主要来自逐 token 解码。其短句生成阶段中位数从 0.575 降到 0.122 秒，长句从 2.446 降到 0.461 秒。首次推理仍需编译管线和上传权重：短句首次调用中位数为 3.712 → 3.605 秒，长句为 6.454 → 4.178 秒。热态倍数不能用于描述冷启动；模型加载也受文件缓存影响。

详见 [逐组统计及阶段计时](benchmarks/20261111-cpu-gpu-original/formal/summary.json)、[原始逐轮记录](benchmarks/20261111-cpu-gpu-original/formal)、[环境、输入及构建散列](benchmarks/20261111-cpu-gpu-original/provenance.json)。GPU 温度记录为 54–55°C，CPU 温度未取得。本轮“优化前”是本轮开始时冻结的程序集，包含此前未提交的优化，不是仓库 HEAD。冻结的 before/after 二进制留在本机 artifacts；公开记录提供散列，未保存完整的优化前源码快照，不能仅从 HEAD 重建该基线。

## 原始权重的定义

来源为 [FunAudioLLM/Fun-ASR-Nano-2512](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)，修订 db823224e5c2510001b4a3d3ed97f430bd52ac88。checkpoint 中 1,347 个张量包含 1,261 个 BF16 和 86 个 FP32 张量。工具提取固定 ASR 路径所需的 950 个编码器张量和 311 个解码器张量，将原值无损展开为 FP32 GGUF，不从 Q8 权重反量化来冒充原始权重。

GPU 权重与投影运算使用 FP32，没有新增 F16/TF32 或低位量化。现有解码器的 F16 KV 缓存舍入保持不变，因此此处验证的是原始权重在本实现中的推理，不是全程 FP32，也不是 PyTorch/BF16 原始计算图对照。新 GPU 单帧归约树改变浮点加法顺序，采用误差检查及真实文本回归；不声称所有中间结果与旧 GPU 逐位相同。

FP32 GGUF 总计约 3.95 GB，热态 GPU 权重驻留约 3.32 GB。本轮前后均显式配置 4 GiB 缓存；产品默认仍是 2 GiB，使用原始权重时应按下方示例调整，否则可能反复逐出和上传。原始权重 GPU 进程峰值 RSS 中位数约 4.80 → 4.84 GB（短句）、4.82 → 4.88 GB（长句），内存数字为十进制 GB，非显存总占用。

[模型散列与张量数量](benchmarks/20261111-cpu-gpu-original/original-models.json)、[导出工具重建校验](benchmarks/20261111-cpu-gpu-original/export-tool-verification.json) 已归档。两个导出结果均与正式基准使用的模型逐字节一致。

## 落地的改动

- Q8 GPU：每八个 subgroup lane 分别计算一个完整量化块，使用整数点积后按原块顺序累加，减少 shuffle 和空闲 lane。保留无整数点积的整数回退；两者均与旧逐行内核逐位一致。
- FP32 GPU：增加 32×64 分块矩阵管线，以及适合一到四帧的 subgroup 矩阵向量管线。按形状及设备能力自动选择，卷积和不适用形状保留通用管线。批量矩阵保持原累加顺序。
- FP32 GPU：合并 Q/K/V、gate/up 投影，减少输入上传及提交等待。原始权重短句每次计算提交从 1,797 降至 1,125，长句从 5,934 降至 3,498。合并权重只长期保存在 GPU，CPU 临时拼接使用池，避免额外常驻约 1.2 GB 的模型副本。
- CPU：单帧 Q8 量化暂存改用 ArrayPool；FP32 单帧一次处理四个输出行，共享输入加载；单 token、两查询头共享一 KV 头的 GQA 使用现有成对注意力内核。CPU 改动保持原乘加和归约顺序。
- 分配：FP32/Q8 GPU 分组输出接入 TensorWorkspace，避免每个投影创建托管数组。

每次完整调用的托管分配中位数如下，单位十进制 MB；包括工作线程的分配，不等于进程常驻内存：

| 路径 | 输入 | 优化前 | 优化后 |
|---|---|---:|---:|
| CPU Q8 | 6 秒 | 8.271 | 1.317 |
| CPU Q8 | 24 秒 | 20.407 | 4.053 |
| GPU Q8 | 6 秒 | 99.946 | 1.382 |
| GPU Q8 | 24 秒 | 235.766 | 4.385 |
| GPU 原始权重 | 6 秒 | 1.417 | 1.313 |
| GPU 原始权重 | 24 秒 | 4.508 | 4.136 |

八行打包 AVX-VNNI 的候选在部分合成矩阵上更快，但完整 ASR 变慢，已撤回；双累加器点积改变严格注意力参考结果，也未保留。新增指令或布局只作为候选，最终按完整调用与精度检查决定是否采用。

## 同批 C++ CPU 复核

使用固定 Fun-ASR 0339018ba74a7defa3b6b6a96718d17b816be77b、llama.cpp 8086439a4cea94c71a5dfb8fe4ad1546aebd640f，MSVC 19.44.35230，开启 AVX2/FMA/F16C/AVX-VNNI/OpenMP，纯 CPU。双方相同 F16/Q8 权重、四线程、亲和性、输入与 token 限制，关闭提示词缓存。顺序为 C++、C#、C#、C++，每进程一次预热和三次正式调用。

| 输入 | C++ CPU 中位数 | 本轮 C# CPU 中位数 | C# 耗时减少 |
|---|---:|---:|---:|
| 6 秒 | 0.856841 秒 | 0.748786 秒 | 12.6% |
| 24 秒 | 3.022118 秒 | 2.919829 秒 | 3.4% |

另计 24 次正式、8 次预热，文本一致。[原始数据与统计](benchmarks/20261111-cpu-gpu-original/native-confirmation/summary.json) 保留全部样本。该批 C# 长句范围为 2.606–2.991 秒，前一批为 2.582–3.141 秒，CPU 波动明显；不要把前一批的 2.634 秒与这里的 C++ 3.022 秒拼成优势。本轮未重测原生 GPU 和 C++ 原始 FP32 权重，不能据此声称超过 CUDA 或 C++ GPU。

## 正确性与回退

| 检查 | 结果 |
|---|---|
| .NET 10 默认 CPU | 9,500,865 项通过 |
| 同一程序集在 .NET 11.0.0 | 9,500,865 项通过 |
| 禁用 AVX2 | 9,179,904 项通过 |
| 禁用全部硬件 intrinsic | 9,179,904 项通过 |
| .NET 10 Vulkan | 276,595 项通过 |
| Vulkan 禁用整数点积 | 276,595 项通过 |
| .NET 11 Vulkan | 276,595 项通过 |

FP32 矩阵向量检查采用 2e-4 × max(1, |参考值|) 容差；批量 FP32、分组/独立投影、Q8 和 CPU 成对 GQA 的相关一致性检查使用严格位比较。GPU 检查包含小缓存下逐出和重新上传。Release 全解决方案编译零警告、零错误，中文注释检查通过；六种 GLSL 编译结果与嵌入 SPIR-V 逐字节一致。

.NET 11 此处验证兼容和正确性，表中性能均在 .NET 10.0.15 测量，未把切换运行时的收益混入本轮。GPU 本轮只在这张 RTX 3060 上实测；TTS、ARM64 和其他显卡的完整性能未重跑。完整日志见本报告的 [证据目录](benchmarks/20261111-cpu-gpu-original)。

## 使用与复现

主项目维持 .NET 10，不强制升级到 .NET 11。准备匹配的官方 checkpoint 及 GGUF 模板后，在新的输出目录导出原始权重：

```powershell
python tools/ExportOriginalAsr.py --checkpoint artifacts/cpu-gpu-20261111/original/model.pt --templates artifacts/cpp-asr-20261111/models --output artifacts/original-f32
```

此离线工具需要 Python 3.11+、PyTorch 与 NumPy，使用 weights_only 加载并逐张量验证形状；它不增加生产推理依赖，也不覆盖已有模型。模板元数据与分词器必须与固定模型匹配，来源和 checkpoint SHA-256 在 provenance.json 中。

显式指定原始权重运行完整 ASR，PCM 为 16 kHz 单声道 little-endian float32：

```powershell
$env:FUNASRNET_DISABLE_PROMPT_CACHE = '1'
dotnet run --project tools/FunAsrNet.Benchmarks -c Release -- asr . artifacts/cpp-asr-20261111/inputs/sample-6s.f32 5 --encoder artifacts/original-f32/funasr-encoder-original-f32.gguf --decoder artifacts/original-f32/qwen3-original-f32.gguf --gpu-cache-mib 4096 --vulkan --affinity 0x55
Remove-Item Env:FUNASRNET_DISABLE_PROMPT_CACHE
```

去掉 --vulkan 测 CPU；改用相同 F16/Q8 文件测量化版本。实际应用按需启用默认提示词缓存。本轮还单独验证了原始权重、4 GiB 缓存和默认提示词缓存同时开启的运行。

原始权重 GPU 调用示例：

```csharp
using FunAsrNet;
using FunAsrNet.Vulkan;

using var gpu = new VulkanFloatBackend(4096L * 1024 * 1024);
var asr = new FunAsr(
    "models/funasr-encoder-original-f32.gguf",
    "models/qwen3-original-f32.gguf", threads: 4)
{
    FloatBackend = gpu
};
// 在 gpu 释放前串行调用 asr.Transcribe(samples)。
```

无需模型的算子检查、微基准与离线着色器验证：

```powershell
dotnet run --project tools/FunAsrNet.Benchmarks -c Release -- gpu-check
dotnet run --project tools/FunAsrNet.Benchmarks -c Release -- kernels --vulkan --affinity 0x55
./tools/CompileVulkanShaders.ps1 -Compiler artifacts/cpu-gpu-20261111/glslang/bin/glslang.exe -VerifyOnly
```

着色器使用 [glslang 16.6.0](https://github.com/KhronosGroup/glslang/releases/tag/16.6.0)，参数为 -V --target-env vulkan1.1 -S comp；整数点积变体加 -DINTEGER_DOT=1。VerifyOnly 只核对，不改源码；不传该开关才会重建嵌入二进制。生产运行无需着色器编译器。

正式测量使用同一个 AsrRunner.dll 分别加载冻结的前后库，避免入口差异；[入口源码](benchmarks/20261111-cpu-gpu-original/AsrRunner.cs.txt)、[项目文件](benchmarks/20261111-cpu-gpu-original/AsrRunner.csproj.txt)、[正式测量脚本](benchmarks/20261111-cpu-gpu-original/run-formal.ps1)、[C++ 复核脚本](benchmarks/20261111-cpu-gpu-original/run-native-confirmation.ps1) 均已保留。脚本中的本机路径需按部署调整，已有正式输出目录会被拒绝，避免覆盖测量记录。权重及冻结程序在本机 artifacts 中，不随证据目录发布；SHA256SUMS.txt 覆盖归档证据本身。
