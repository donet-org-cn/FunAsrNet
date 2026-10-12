# 性能比较与证据范围

本机后续 CPU 优化已完成 138 次正式样本、30 次预热：C# ASR 比上一版耗时减少 19–33%，仍慢于同指令集原版 CPU 21–40%；TTS 减少 41–44%，仍慢于 ORT CPU 71–79%。目标尚未全面达到持平或原版一半耗时，数据及正确性见 [本机后续对比](cpu-followup-20261012.md) 和 [原始统计](benchmarks/20261012-cpu-next/summary.json)。这轮只测 Windows x64。

2026-10-12 已补做 [纯 CPU 原生对照](cpu-native-comparison-20261012.md)：修正 x64 默认交错布局的标量回退后，ASR 提速 8.47–8.99 倍，仍慢于原版 C++ CPU 1.70–1.84 倍。TTS 两批复测保留全部样本，仍慢于 ORT CPU 3.03–3.10 倍。原始记录、修正前/后构建散列与分阶段计时见 [CPU summary.json](benchmarks/20261012-cpu-native/summary.json)。下文历史结果不与新实验混算。

同日新的 RK3588 CPU 对 C++ CPU 实验中，三组 C# 中位数为 1.271、1.351、5.339 秒，C++ 为 1.545、1.604、7.351 秒。该在线环境期间有明显换页，短句还存在句末标点差异，不能宣称稳定语言优势；52 次正式样本、温度和交换计数见 [ARM64 CPU summary](benchmarks/20261012-rk3588-cpu-native/summary.json)。

2026-10-12 另已完成同一 RTX 3060 上的原生 GPU 对比：TTS 的 C# Vulkan 比 ORT CUDA 慢 9.63–14.55 倍；ASR 的 C++ 原计算图 CUDA 适配也更快。当前 C# GPU 路径没有速度优势，不能用历史 GPU/CPU 表格代替 GPU/GPU 结论。适配范围、精度差异和原始数据见 [原生 GPU 对比](gpu-native-comparison-20261012.md)。

2026-10-12 新的纯 C# TTS CPU 优化已完成两端同机复测：Windows x64 两种模型均提速约 3.07 倍；RK3588 相对旧托管部署，Huayan 提速 1.74 倍、AISHELL3 提速 1.55 倍。仍比各自环境的 ORT CPU 慢约 2.83–3.18 倍。条件、正确性、命令与原始数据见 [Windows TTS CPU 优化](tts-cpu-optimization.md) 和 [ARM64 更新与实测](rk3588-update-20261012.md)。下文保留历史记录，不将新旧实验混算。

文档保留的历史表格来自真实历史测量。首次公开版没有把这些结果重新标记为新名称下的全套实测，也没有只留下有利数据。两个环境和代码快照不同，不能将时间拼成一次实验。

| 记录 | 历史实现快照 | 范围 |
|---|---|---|
| 2026-09-30 RK3588 | `1d046c175cd144dfd0c094bf670b2b9e1913434c` | CPU，92 次正式、20 次预热 |
| 2026-10-10 Windows GPU | 内核在 `c8ede4ce4c43f9ba035597134ce30e5ffee16e5e` 中归档 | CPU/GPU/原生，108 次正式、30 次预热 |

历史提交号是来源标识，不是此新仓库的提交；公开仓库没有复制旧仓库历史。历史模型和输入的 SHA-256 固定，首次公开版计算源码与来源仅命名变化。首次整理验证见 [公开版记录](public-edition.md)，后续 CPU 改动见本页开头的新测量。

## 固定原版

ASR 对照为 Fun-ASR `0339018ba74a7defa3b6b6a96718d17b816be77b`，llama.cpp `8086439a4cea94c71a5dfb8fe4ad1546aebd640f`，不是今天上游 main 的速度。TTS 对照为 ONNX Runtime 1.23.2 CPU 的相同模型执行，不是 sherpa-onnx/Python 整套服务。

下述历史 ASR 保持 F16 编码器和 Q8_0 解码器，线程 4。历史 C# 使用默认固定提示词缓存，C++ 保留每段清理 KV 行为；不是逐条指令完全相同的程序。历史实验没有测原版 GPU；页首所链的新 CPU 实验双方都关闭提示词缓存，新的 GPU 实验也已另有原生对照。

## 计时与统计

Windows ASR 计完整 Transcribe/解码调用：特征、编码、解码、复制及 GPU 同步包含在内；模型加载、输入读取和 WAV 解码排除。RK3588 的 C# 来自在线 HTTP 服务内部 requestSeconds，包含调度，原版为完整解码调用。TTS 计模型 Run，排除文本前端、音频封装及重采样；两项随机度 0、语速 1、文本“您好，欢迎致电，请问有什么可以帮您？”。

中位数为排序中间值，偶数取均值；P95 最近秩，小样本时等于最大值。只排除标记为 warmup 的轮次，保留全部慢样本。冷启动不能用热态中位数代替。详细轮数、环境、温度/频率策略及命令见 [RK3588](benchmark-20260930.md) 和 [Windows Vulkan](vulkan-local-validation.md)。

## 公开证据

- [RK3588 summary.json](benchmarks/20260930/summary.json)、runs 逐轮 JSONL、输入散列和波形误差结果。
- [Windows summary.json](benchmarks/20261010-vulkan-local/summary.json)、逐轮 JSONL、CPU/GPU/原生波形校验、GPU 数值检查、环境快照与 C++ 编译特性。
- [2026-10-12 TTS CPU summary.json](benchmarks/20261012-tts-cpu/summary.json)、优化前/后与 ORT 的 60 次正式测量、输入 ID 清单和波形校验。
- [2026-10-12 RK3588 summary.json](benchmarks/20261012-rk3588-cpu/summary.json)、旧托管部署/新版本与 ORT 的 60 次正式测量、温度/频率和波形校验。
- [2026-10-12 原生 GPU summary.json](benchmarks/20261012-gpu-native/summary.json)、同卡 GPU 对比的 105 次正式测量、CUDA 节点剖析和转写/波形校验。
- [2026-10-12 新纯 CPU summary.json](benchmarks/20261012-cpu-native/summary.json)、Windows 默认布局修正前/后及 TTS 复核的 184 次正式测量；布局与 GPU 回归单独保存。
- [2026-10-12 新 ARM64 CPU summary.json](benchmarks/20261012-rk3588-cpu-native/summary.json)、与固定 C++ CPU 的 52 次正式 ASR 测量、温度/频率/换页和 Linux 计时包装。
- [2026-10-12 本机 CPU 后续 summary.json](benchmarks/20261012-cpu-next/summary.json)、进一步优化前/后及同指令集原版 CPU 的 138 次正式样本。
- 每个记录目录的 `SHA256SUMS.txt` 对应本仓库证据；历史归档已确认逐字节与历史来源相同，保留原 CRLF 与历史引擎标签。新实验的 JSONL 保留全部逐轮记录；已注明的原生日志路径脱敏与换行整理不改变数值。

未复制私人部署/健康快照、内部 Git 地址、证书或含个人工作路径的构建日志。Windows 证据保留系统 Vulkan 加载器路径，便于确认真实驱动；它不是通用安装说明。

测试输入 WAV/f32/i64 与模型未发布，只有散列，没有可下载同一语料，因此不能声称从公开仓库一键复现全套历史实验。历史 C++ 临时计时包装没有随历史归档发布；2026-10-12 新 CPU/GPU 实验已经公开对应生成器与 Linux CPU 包装，可按各实验文档复建对照。

输入 f32 散列：

```text
8ebc139144c67bba23110e445db36b0cb9da70eef32e5e8ad0d9dda06ce2322  sip-short-pcma（95999 采样）
fb705a8368d62b25bfec86b28b897e94c61173ca5ba6c0490b5556e078471432  sip-zh-pcmu（89855 采样）
839f53e709e87ed6c74b645c5e2f871fd4af68e84b4b09698372f2d065c5b048  sip-repeated-pcma（447996 采样）
feb05cdbb64fdf8045f0c08e42701ee5e24a72f7feee73c2858abce65f760569  huayan.i64
66ac60d96f74dd848d71580982ac6dd0227ce53ec80fac3afca25cfa2dc012ee  aishell3.i64
```

## 结论的实际限制

历史 RK3588 CPU 短句基本持平；重复长样例 C# 更快，但不代表自然通话。历史 Windows Vulkan ASR 比原版 CPU 快，不能视为语言优势；当时的 x64 默认布局标量回退现已修正，新 CPU 结果见页首。TTS 在两种环境都没有追平原生 ORT。文本相同是固定样例回归，不是字错误率保证；波形误差接近不是人工听感评价；4 个会话配额不是四路持续实时容量。
