# 纯 CPU 原生对照复测（2026-10-12）

**本页为前轮历史快照。** 当前 Windows CPU 的最新优化和原版对照见 [本机 ASR 第五轮记录](cpu-round5-20261012.md)；本页数字不代表最新构建。

本次单独复测 Windows x64 与 RK3588 的 CPU 路径：ASR 对照固定上游 C++/ggml/llama.cpp CPU，TTS 对照同模型原生 ONNX Runtime CPU。所有正式样本均关闭 GPU。

发现并修正了 x64 默认权重布局选择：ARM 用的浮点/Q8 交错布局在 x64 上落入标量循环，挡住已有 SIMD 内核。现在只在 ARM64 NEON/SDOT 可用时准备对应交错布局；x64 保留行主序，使用已有 AVX2/FMA 浮点矩阵和向量 Q8 点积。没有改变模型、权重量化、输入或 token 上限，RK3588 的 NEON/SDOT 选择保持原有行为。

## Windows ASR：默认 CPU 修正前后

下表为热态中位数 / P95，单位秒。短句每边十次正式样本，长样例每边六次。原版 CPU 一列取修正后的交替对照批次；修正前批次的 C++ 结果也完整保留。

| 输入 | 修正前 C# CPU | 修正后默认 C# CPU | 原版 C++ CPU | C# 提速 | 修正后 / 原版 |
|---|---:|---:|---:|---:|---:|
| 6.000 秒 PCMA | 12.795 / 13.353 | 1.423 / 1.482 | 0.775 / 0.875 | 8.99 倍 | 1.84 倍 |
| 5.616 秒 PCMU | 12.720 / 12.796 | 1.502 / 1.534 | 0.886 / 1.079 | 8.47 倍 | 1.70 倍 |
| 28.000 秒重复样例 | 42.830 / 42.952 | 5.018 / 5.197 | 2.952 / 3.451 | 8.54 倍 | 1.70 倍 |

默认 x64 CPU 已不再是原版十几倍的差距，**仍慢约 1.70–1.84 倍**。三份样例的修正前、修正后、C++ CPU 正式文本严格一致。修正后编码与特征阶段中位数约 0.280、0.249、1.782 秒；剩余差距主要在解码阶段，单独阶段计时与完整调用统计见 [summary.json](benchmarks/20261012-cpu-native/summary.json)。阶段中位数的和不保证等于完整调用中位数。

关闭交错布局的九次诊断样本单独保存；正式“修正后”数据使用默认环境，不依赖 `FUNASRNET_INTERLEAVE=0`。Release 数值回归通过 685,701 项检查，原有误差阈值没有放宽。

修正后本机默认 CPU 的完整 DemoChecks 通过 46 项接口验收（按 PASS 项计数，已更正此前把 61 行日志当作检查数的表述）。因为布局选择也影响 x64 GPU 实例的权重准备，另通过 134,474 项 GPU 数值检查并保留三份 ASR 诊断：短 PCMA 的 Vulkan 输出较旧 GPU 快照少句末句号，另外两份文本严格一致。短句正文一致不替代严格文本一致；这些诊断没有计入 CPU 正式性能或重标为新的整套 GPU 性能数据。

## Windows TTS：CPU 对 CPU

本次 ASR 默认布局修正没有修改 TTS 实现。为复核首批 AISHELL3 的块间波动，按相同条件再运行一批，合并两批全部样本，每模型、每引擎各二十次。没有删除首批慢样本。

| 模型 | C# CPU 中位 / P95 | 原生 ORT CPU 中位 / P95 | C# / ORT |
|---|---:|---:|---:|
| Huayan 0 号 | 0.321 / 0.365 秒 | 0.106 / 0.111 秒 | 3.03 倍 |
| AISHELL3 50 号 | 0.138 / 0.299 秒 | 0.045 / 0.050 秒 | 3.10 倍 |

Huayan 输出 69,376 点，AISHELL3 输出 28,928 点，长度一致、全部有限；两批波形比较均通过最大绝对误差 `0.001`、RMSE `0.0001` 阈值。本次最大误差分别为 1.69 × 10⁻⁵、3.30 × 10⁻⁵。AISHELL3 的 C# 正式耗时跨 0.126–0.304 秒，尾部波动真实保留，不能用中位数代替稳定尾延迟。

## 条件与计时

- Intel Core i7-12700、Windows x64、.NET 10.0.12，Release；CPU 四线程、进程亲和性 `0x55`。
- 默认 C# 基线为本仓库 `6c589f4`，包含此前 TTS CPU 优化。原生 ASR 固定 [Fun-ASR `0339018`](https://github.com/QwenAudio/Fun-ASR/tree/0339018ba74a7defa3b6b6a96718d17b816be77b) 与 [llama.cpp `8086439`](https://github.com/ggml-org/llama.cpp/tree/8086439a4cea94c71a5dfb8fe4ad1546aebd640f)。
- ASR 两边均使用相同 F16 编码器、Q8_0 解码器和三份 16 kHz f32 输入，最大 128 token，关闭提示词缓存，每轮重新预填充。计完整特征、编码和解码调用。
- 原生 C++ 用 MSVC 19.44、AVX2、OpenMP；CUDA/Vulkan/BLAS 关闭，`n_gpu_layers=0`，KV 与计算缓冲在 CPU。包装只统一四线程、输入、重复调用与计时，保留上游计算图和算法。生成器见 [NativeAsrCpuFixture.cs](../tools/FunAsrNet.NativeReference/NativeAsrCpuFixture.cs)。
- TTS 使用 ORT 1.23.2 CPU 包，intra-op=4、inter-op=1、顺序执行、全部图优化；不是 GPU 包回退。文本“您好，欢迎致电，请问有什么可以帮您？”，Huayan 0 号、AISHELL3 50 号，语速 1、两项随机度 0。
- TTS 计完整模型 Run；音素前端、音频封装和重采样不计入。模型加载、输入文件读取、输出文件写入均在两类测量的计时区外。加载和预热另行保留，不代表首次请求延迟。

默认路径按 C#、原生、原生、C# 四个独立进程块测量；每块预热一次，TTS 和短 ASR 各测五次，长 ASR 各测三次。共 92 次正式样本、20 次预热，保留全部慢样本。中位数取排序中间值，偶数取均值；P95 最近秩，在这组样本量下等于最大值。Windows 未锁频，无法读取温度和实时频率，在线机器有背景负载。

随后用修正后的默认实现重跑 ASR，共 52 次正式样本、12 次预热；TTS 复核增加 40 次正式样本、8 次预热。Windows 总计 **184 次正式测量、40 次预热**，另有九次布局诊断和三次诊断预热。TTS 合并后二十次样本的最近秩 P95 为排序第十九项，不再等于最大值。修正前、修正后和 TTS 复核的原始记录分开保存；summary 记录各构建 DLL 的 SHA-256。

## RK3588：新的原版 CPU 对照

在原部署机器上重新构建固定原版 CPU，使用 GCC 10.5、CMake 3.31.6、ARMv8.2-A/SDOT、OpenMP，CUDA/Vulkan/BLAS 关闭。两边直接运行基准工具，限定 CPU 4–7、四线程、关闭提示词缓存、128 token；计完整特征、编码和解码，不使用 HTTP 内部 requestSeconds。按相同 ABBA 顺序，短句每边十次、长样例每边六次，共 **52 次正式样本、12 次预热**。

| 输入 | C# CPU 中位 / P95 | 原版 C++ CPU 中位 / P95 | C# 本轮耗时减少 |
|---|---:|---:|---:|
| 6.000 秒 PCMA | 1.271 / 1.333 秒 | 1.545 / 2.007 秒 | 17.8% |
| 5.616 秒 PCMU | 1.351 / 1.454 秒 | 1.604 / 1.851 秒 | 15.8% |
| 28.000 秒重复样例 | 5.339 / 5.518 秒 | 7.351 / 7.411 秒 | 27.4% |

短 PCMA 的 C# 输出含句末句号，原生缺句号；另两份文本严格一致。不能将忽略标点后的正文匹配宣称为严格文本一致，也没有测自然通话 CER/WER。

两边每次调用前均在计时区外等待温度不高于 60°C，最多等待五分钟。最初进程块另等待到 55°C；后续板端空闲温度持续高于 55°C，将进程块等待放宽至同样的 60°C，保留已经完成的样本，逐调用 60°C 条件保持一致。调用结束温度可能超过 60°C，保留系统原有动态调频与在线背景服务，没有锁频。CPU 内核不同、环境有换页，**本轮更快不等于长期稳定领先，也不能解释为语言优势**。

本轮系统 `pswpin` 从 374,392 到 375,862 页，`pswpout` 从 800,803 到 1,049,996 页，与此前 TTS 实验的零新增换页不同；不将这轮 ASR 结论与旧 TTS 拼成同一次实验。每块前后的内存、交换计数和频率见 [ARM64 summary 与环境](benchmarks/20261012-rk3588-cpu-native/summary.json)。这台机器的最新 TTS CPU 对 ORT 仍为 Huayan 1.234 / 0.437 秒、AISHELL3 0.742 / 0.234 秒，属于此前独立 [TTS 实验](rk3588-update-20261012.md)，本轮未重测。

ARM64 正式比较使用部署中 `1f817be` 构建（散列以 summary 为准）；本次新增硬件条件在 RK3588 上仍选择相同 NEON/SDOT 内核。正式比较结束后部署修正后的程序集，补做数值、HTTP/HTTPS 以及三份输入的转写验证。这些部署验收不计入性能样本。Linux 计时包装和 CMake 文件与原始逐轮日志一起公开，C++ 参数与算法适配和 Windows 相同，另加与 C# 一致的温度等待。

## 复测命令

准备模型和输入见 [模型清单](models.md) 与 [TTS CPU 优化](tts-cpu-optimization.md)。固定上游源码和匹配的 llama.cpp 源码准备在开发目录，不增加产品依赖。

```powershell
dotnet build tools/FunAsrNet.NativeReference -c Release
dotnet build tools/FunAsrNet.Benchmarks -c Release
dotnet tools/FunAsrNet.NativeReference/bin/Release/net10.0/FunAsrNet.NativeReference.dll prepare-asr-cpu-reference artifacts/native-cpu upstream/runtime/llama.cpp/funasr-cli/funasr-cli.cpp llama-source
```

在 VS 2022 x64 Native Tools 命令提示符中构建原生 CPU 对照：

```text
cmake -S artifacts/native-cpu -B artifacts/native-cpu/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build artifacts/native-cpu/build --target funasr-native-cpu-reference --parallel 8
```

然后在 PowerShell 中交替运行。对其余两份 ASR 输入替换 f32 文件名；28 秒样例使用三次正式重复：

```powershell
$env:FUNASRNET_DISABLE_PROMPT_CACHE = '1'
$runner = 'tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll'
$native = 'artifacts/native-cpu/build/funasr-native-cpu-reference.exe'
dotnet $runner asr models results/sip-short-pcma.f32 5 --affinity 0x55
& $native models/funasr-encoder-f16.gguf models/qwen3-0.6b-q8_0.gguf results/sip-short-pcma.f32 5
```

原生包装是 Windows 开发基准，固定亲和性 `0x55`，不是跨平台服务程序。公开仓库保存输入散列，但未发布三份私人录音与模型；可以复建计时包装，不能从仓库一键获取完全相同的测试输入。
