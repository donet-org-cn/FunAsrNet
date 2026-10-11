# 性能比较与证据范围

README 的数字来自真实历史测量。公开版没有把这些结果重新标记为新名称下的全套实测，也没有只留下有利数据。两个环境和代码快照不同，不能将时间拼成一次实验。

| 记录 | 历史实现快照 | 范围 |
|---|---|---|
| 2026-09-30 RK3588 | `1d046c175cd144dfd0c094bf670b2b9e1913434c` | CPU，92 次正式、20 次预热 |
| 2026-10-10 Windows GPU | 内核在 `c8ede4ce4c43f9ba035597134ce30e5ffee16e5e` 中归档 | CPU/GPU/原生，108 次正式、30 次预热 |

历史提交号是来源标识，不是此新仓库的提交；公开仓库没有复制旧仓库历史。历史模型和输入的 SHA-256 固定，公开版计算源码与来源仅命名变化。新的整理验证见 [公开版记录](public-edition.md)。

## 固定原版

ASR 对照为 Fun-ASR `0339018ba74a7defa3b6b6a96718d17b816be77b`，llama.cpp `8086439a4cea94c71a5dfb8fe4ad1546aebd640f`，不是今天上游 main 的速度。TTS 对照为 ONNX Runtime 1.23.2 CPU 的相同模型执行，不是 sherpa-onnx/Python 整套服务。

ASR 保持 F16 编码器和 Q8_0 解码器，线程 4。C# 使用默认固定提示词缓存，C++ 保留每段清理 KV 行为；不是逐条指令完全相同的程序。GPU 没有和原版 GPU 比。

## 计时与统计

Windows ASR 计完整 Transcribe/解码调用：特征、编码、解码、复制及 GPU 同步包含在内；模型加载、输入读取和 WAV 解码排除。RK3588 的 C# 来自在线 HTTP 服务内部 requestSeconds，包含调度，原版为完整解码调用。TTS 计模型 Run，排除文本前端、音频封装及重采样；两项随机度 0、语速 1、文本“您好，欢迎致电，请问有什么可以帮您？”。

中位数为排序中间值，偶数取均值；P95 最近秩，小样本时等于最大值。只排除标记为 warmup 的轮次，保留全部慢样本。冷启动不能用热态中位数代替。详细轮数、环境、温度/频率策略及命令见 [RK3588](benchmark-20260930.md) 和 [Windows Vulkan](vulkan-local-validation.md)。

## 公开证据

- [RK3588 summary.json](benchmarks/20260930/summary.json)、runs 逐轮 JSONL、输入散列和波形误差结果。
- [Windows summary.json](benchmarks/20261010-vulkan-local/summary.json)、逐轮 JSONL、CPU/GPU/原生波形校验、GPU 数值检查、环境快照与 C++ 编译特性。
- 每个记录目录的 `SHA256SUMS.txt` 对应本仓库证据；已确认逐字节与历史来源相同，保留原 CRLF。原始数据中的历史引擎标签不重命名。

未复制私人部署/健康快照、内部 Git 地址、证书或含个人工作路径的构建日志。Windows 证据保留系统 Vulkan 加载器路径，便于确认真实驱动；它不是通用安装说明。

测试输入 WAV/f32/i64 与模型未发布，只有散列，没有可下载同一语料，因此不能声称从公开仓库一键复现全套历史实验。C++ 临时计时包装也未发布；可按固定版本和边界重建对照，但不承诺一条命令复现全部原生包装。

输入 f32 散列：

```text
8ebc139144c67bba23110e445db36b0cb9da70eef32e5e8ad0d9dda06ce2322  sip-short-pcma（95999 采样）
fb705a8368d62b25bfec86b28b897e94c61173ca5ba6c0490b5556e078471432  sip-zh-pcmu（89855 采样）
839f53e709e87ed6c74b645c5e2f871fd4af68e84b4b09698372f2d065c5b048  sip-repeated-pcma（447996 采样）
feb05cdbb64fdf8045f0c08e42701ee5e24a72f7feee73c2858abce65f760569  huayan.i64
66ac60d96f74dd848d71580982ac6dd0227ce53ec80fac3afca25cfa2dc012ee  aishell3.i64
```

## 结论的实际限制

RK3588 CPU 短句基本持平；重复长样例 C# 更快，但不代表自然通话。Windows Vulkan ASR 比原版 CPU 快，不能视为语言优势，x64 CPU 标量瓶颈仍需优化。TTS 在两种环境都没有追平原生 ORT。文本相同是固定样例回归，不是字错误率保证；波形误差接近不是人工听感评价；4 个会话配额不是四路持续实时容量。
