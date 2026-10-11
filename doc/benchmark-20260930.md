# RK3588 在线复测（2026-09-30）

以下是重命名前实现于 2026-09-30 的历史实测，并非 FunASR.NET 公开版新测。结果摘要见 [README](../README.md)，逐轮数值与统计见 [历史记录](benchmarks/20260930/)。公开版保留数值证据，未复制私人部署与服务快照。来源范围见 [性能说明](performance.md)。

## 本次结论

正式记录共 **92 次测量，另有 20 次预热**，全部完成。ASR 两份短样例中位耗时与 C++ 接近；28 秒重复样例的 C# 中位耗时为 5.241 秒，C++ 为 6.077 秒。两种 TTS 的 C# 中位耗时为原生 ORT 的 5.98 倍与 6.45 倍，尚未达到原生执行速度。完整精度的统计值见 [summary.json](benchmarks/20260930/summary.json)。

正式样本调用前温度为 52.692–60.076°C，调用后最高 72.076°C。整次测试的系统换入计数增加 6174 页、换出增加 64010 页，记录是系统总量，不能直接归因于某一实现或某一轮。应结合原始长尾耗时评估在线环境干扰，不能将百分之几的差异视为稳定优势。测试前后服务 PID 均为 4176283，保持运行；健康采样未发现其他语音请求。

## 对照对象

| 项目 | 版本与配置 |
|---|---|
| 产品代码 | `1d046c175cd144dfd0c094bf670b2b9e1913434c`，重命名前实现 1.1.0；非本公开版版本号 |
| 托管环境 | .NET SDK 10.0.302，运行时 10.0.10，Linux ARM64 Release |
| ASR 原版 | [Fun-ASR](https://github.com/QwenAudio/Fun-ASR/tree/0339018ba74a7defa3b6b6a96718d17b816be77b)，提交 `0339018ba74a7defa3b6b6a96718d17b816be77b` |
| ASR 原生依赖 | [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/8086439a4cea94c71a5dfb8fe4ad1546aebd640f)，提交 `8086439a4cea94c71a5dfb8fe4ad1546aebd640f` |
| TTS 原生参考 | [ONNX Runtime 1.23.2](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2)，官方 Linux ARM64 CPU 包，通过 C++ API 调用 |
| 原生编译 | GCC 10.5.0，CMake 3.16.3，Release，`armv8.2-a+fp16+dotprod`，OpenMP |
| 处理器 | RK3588，4 个 A76 核（CPU 4–7），4 算子线程，未使用 NPU/GPU |
| 权重 | 相同 F16 编码器、Q8_0 解码器及两个 ONNX 文件；SHA-256 见 [模型清单](models.md) 和本次原始记录 |

ASR C++ 保留上游计算、提示词与采样逻辑，只把编码器和解码器的计算线程统一设为 4；测量包装调用上游 `load_decoder`、`asr_decode_window` 和 `free_decoder`。上下文、batch、ubatch 均保留上游的 2048，最大输出 token 为 512，与 HTTP 转写接口一致。C# 保留产品默认的固定提示词缓存，C++ 每次按上游逻辑清理 KV 缓存。这是两种实际实现的对照，不要求内部调度和浮点舍入逐指令相同。

TTS 是**相同模型、相同音素 ID 的算子执行对照**。旧 sherpa-onnx/Python 完整服务已删除，本次没有恢复其文本前端、HTTP 或音频封装，因此不能把 ONNX Runtime 数值称为旧 TTS 服务的端到端速度。

## 输入与计时边界

ASR 使用固定的三份 16 kHz 单声道 PCM16 WAV：6 秒 PCMA 电话编解码样例、5.616 秒 PCMU 样例，以及 28 秒 PCMA 重复拼接样例。最后一份是合成长样例，不代表自然连续通话。原生输入由同一 WAV 解码为 float32，散列记录在 `inputs.jsonl`。文件名中的 PCMA/PCMU 表示样例此前经过的电话编解码流程；计时区不再执行 G.711 编解码或重采样。

- C# ASR：向本机现有服务 `/asr/wav` 请求，使用响应的 `requestSeconds`，包含队列调度和完整 `Transcribe` 调用；不含上传、WAV 解码、重采样与响应网络传输。完整客户端耗时另存为 `clientSeconds`。没有直接使用更短的编码器、解码器分项合计。
- C++ ASR：模型已加载，计时完整 `asr_decode_window`，包含特征计算、编码器和解码器，不含文件读取和加载权重。
- C# / 原生 TTS：计时 `ManagedOnnx.Run` / `Ort::Session::Run`。模型、输入 Tensor 与音素 ID 在计时前准备，输出复制、写文件、文本前端和重采样不计入。

ASR 两条路径的差异是 C# 多了服务内部调度，且模型驻留方式不同；不把这组数据宣传为完全相同的离线执行环境。原生程序的首次模型加载单独记录，当前服务已常驻，不能用两者作冷启动比较。

TTS 文本为“您好，欢迎致电，请问有什么可以帮您？”。Huayan 为 0 号、22050 Hz；AISHELL3 为 50 号、8000 Hz。两边使用同一份 `int64` ID 文件，语速 1、随机度 0、时长随机度 0。关闭随机度是为使波形误差可复核，**不等于页面默认随机参数**。不使用缩短文本、降低权重量化精度或更换模型来提速。

## 重复与环境控制

每个样例按 C#、原生、原生、C# 的四块顺序交替测试。每块先运行一次预热并明确标为 `warmup: true`，其余全部保留。两份短 ASR 和每个 TTS 模型每块测 5 次，即每种实现各 10 次；长 ASR 每块 3 次，即各 6 次。中位数使用排序后的中间值（偶数取两数均值）；P95 使用最近秩 `ceil(0.95 × n)`。当前样本量下 P95 就是最大值，不能代表长期尾延迟分布。RTF = 耗时 / 音频时长，小于 1 表示该次计算快于音频播放。

每次调用前以 thermal_zone0 不高于 60°C 为降温目标，最多等待 5 分钟，降温时间不计入推理耗时。温度查询、HTTP 空闲检查与实际调用并非原子操作，后续采样可能短暂超过目标；以逐轮 `temperatureBefore` / `temperatureAfter` 为准，不删除这些样本。CPU 4/6 两簇频率、降温等待也逐轮保存。保留系统原有 `interactive` 调频策略，不锁频、不清文件缓存；降温目标相同不保证每轮温度轨迹完全一致。

服务持续在线，未重启或停服。C# ASR 客户端限制到 CPU 0–3，服务与原生计算都限制到 4–7；其他业务进程正常运行。C# 每次请求前检查服务空闲；运行独立原生/C# TTS 进程时每秒检查 `/health`，如发现其他语音请求则停止后续测量并保留记录。这只能发现采样时刻的活动，不能证明绝无短暂请求或其他系统负载。系统已有交换空间使用，内存、换页计数和负载保存在前后环境快照中。

这些是在线设备的顺序单请求数据，不是独占机器的实验室微基准，也不是 SIP 端到端延迟、字错误率或多路持续实时容量保证。原始记录中的峰值 RSS 是各独立进程口径；HTTP 客户端不代表服务内存，因此未用客户端 RSS 参与 ASR 内存比较。

## 正确性校验

保留每次 ASR 的完整文本，核对重复运行和两种实现的输出。文本一致仅代表这组样例的回归一致性，没有足够人工标注语料计算可靠 CER/WER。

两种 TTS 输出均保留 float32 波形供核验。校验要求长度相同、全部为有限数值，并在测量前确定阈值：最大绝对误差不超过 `0.001`、RMSE 不超过 `0.0001`。波形校验结果另存 `huayan-parity.json` 和 `aishell3-parity.json`；数值接近不替代人工听感评价。

| 模型 | 样本数（两边一致） | 最大绝对误差 | RMSE | 结果 |
|---|---:|---:|---:|---|
| Huayan | 69376 | 7.60 × 10⁻⁶ | 4.54 × 10⁻⁷ | 通过 |
| AISHELL3 | 28928 | 1.64 × 10⁻⁵ | 1.16 × 10⁻⁶ | 通过 |

误差表比较最后一块 C# 与最后一块原生的波形。原始音频和输入 float32/int64 不随公开版发布；逐轮 JSONL 原样保存，校验值见该目录 `SHA256SUMS.txt`。未发布个人设备的健康快照与部署日志。

## 使用 C# 工具复测

基准工具是仓库内的纯 C# 项目，不增加产品运行依赖：

```sh
dotnet build tools/FunAsrNet.Benchmarks -c Release
runner=tools/FunAsrNet.Benchmarks/bin/Release/net10.0/FunAsrNet.Benchmarks.dll
dotnet "$runner" prepare models assets/mandarin-lexicon.json /path/to/samples results/inputs
taskset -c 0-3 dotnet "$runner" http-asr http://127.0.0.1:18080 /path/to/samples/sip-short-pcma.wav 5
FUNASRNET_PIN_THREADS=1 taskset -c 4-7 dotnet "$runner" tts models results/inputs/huayan.i64 huayan 5 results/huayan.f32
FUNASRNET_PIN_THREADS=1 taskset -c 4-7 dotnet "$runner" tts models results/inputs/aishell3.i64 aishell3 5 results/aishell3.f32
dotnet "$runner" compare /path/to/reference.f32 results/huayan.f32
```

`prepare` 需自行准备 WAV；历史测试音频不提交 Git，不能声称公开仓库可一键复现同一语料。核对 `inputs.jsonl` 中的长度和散列。工具离线 `asr` 默认 128 token，而本次 HTTP 正式数据使用 512 上限，不能混用。

重建原版时使用上表固定提交，CMake 设置 `CMAKE_BUILD_TYPE=Release`、`GGML_NATIVE=OFF`、`GGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod`、`GGML_OPENMP=ON`；将 `FETCHCONTENT_SOURCE_DIR_LLAMA` 指向固定 llama.cpp 源码。原生 TTS 设置 intra-op=4、inter-op=1、顺序执行、全部图优化、CPU Provider。输入/输出名称及形状与 C# `tts` 模式一致。

按只保留 C# 代码的要求，原生对照源码、编译产物与下载包仅在临时目录使用，结果收集后删除，不入库、不作为产品依赖。保留版本、下载散列、构建日志和测量边界供审计；仓库不保留临时 C++ 计时包装源码，因此不会声称从仓库可一键重建整个原生测量环境。
