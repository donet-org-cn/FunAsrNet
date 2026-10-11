# 上游与第三方说明

Fun-ASR 编码器、滤波器组与 FSMN 架构移植自 QwenAudio/Fun-ASR，固定提交 `0339018ba74a7defa3b6b6a96718d17b816be77b`，Apache-2.0，许可见 LICENSE。新的 C# 实现调整了执行方式、内存布局、流式缓冲与 CPU 内核。本项目不是上游官方 .NET SDK。

Qwen3 架构与 GGUF 布局参照 llama.cpp `8086439a4cea94c71a5dfb8fe4ad1546aebd640f`（MIT）核对。默认 .NET 推理未链接 llama.cpp/ggml 代码。模型许可仍遵循发布方声明。

Huayan Piper VITS 直接读取既有 ONNX。托管解释器依照 ONNX 算子规范实现固定模型需要的子集，不是通用 ONNX Runtime 替代品。

中文发音资产由 piper-phonemize 1.1.0 与 espeak cmn 音色离线生成。Piper/espeak 二进制仅作开发参考，不被 C# 应用加载。运行前端使用词典最长匹配、电话号码逐位读法和有限三声变调，不支持的音素明确拒绝。数据来源见 assets/lexicon-provenance.json，离线生成不代表重新授权上游资产。

Huayan 原模型卡标注训练数据许可为 Unknown，本项目保留该状态，不将其解释为 Apache-2.0。

AISHELL3 多发音人模型来自 k2-fsa/sherpa-onnx 官方 tts-models 发布的 `vits-icefall-zh-aishell3.tar.bz2`，元数据声明作者为 k2-fsa、174 位发音人和 8000 Hz。模型、词典、音素表和匿名发音人列表是外部数据，不提交到本仓库；固定校验值见 doc/models.md。AISHELL-3 数据集发布页声明 Apache License v2.0：https://www.openslr.org/93/ 。该数据集许可声明不应解释为本项目重新许可模型；归档没有独立模型 LICENSE，部署方应保留上游来源说明。

本项目根据 ONNX 算子规范和公开的模型输入契约新增纯 C# Tile、拼音词典与发音人选择逻辑。没有链接 sherpa-onnx、ONNX Runtime、原生拼音或 FST 运行库，也没有复制它们的实现代码。模型文档：https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html 。

参考来源：[Fun-ASR](https://github.com/QwenAudio/Fun-ASR)、[llama.cpp](https://github.com/ggml-org/llama.cpp)、[ONNX 算子规范](https://onnx.ai/onnx/operators/)、[piper-phonemize](https://github.com/rhasspy/piper-phonemize)。

可选 `FunAsrNet.Vulkan` 项目的 C# Vulkan API 与设备基础设施来自 sdcb/SimdPaddleOCR，提交 `dbbc21c535c6b2794ed1f217400d28596e8be091`，Apache-2.0。原许可保留在 `src/FunAsrNet.Vulkan/Vendor/LICENSE`，适配范围见同目录 README。没有复制该项目的 OCR 图执行器、模型或 F16 着色器；本项目 FP32/Q8_0 GPU 程序、语音调度和数值验证独立实现。

着色器开发时使用 Khronos glslang 16.6.0；编译器只用于生成 SPIR-V，不作为运行时依赖。GPU 程序来源及字节码嵌入 C#。正式可选 GPU 后端只加载系统 Vulkan 驱动，不随代码嵌入原生 CPU 二进制。

`tools/FunAsrNet.NativeReference` 是隔离的开发对照工具，通过 Microsoft.ML.OnnxRuntime 1.23.2 调用原生 CPU 算子以复测 VITS。`tools/FunAsrNet.NativeGpuReference` 独立引用官方 Microsoft.ML.OnnxRuntime.Gpu.Windows 1.23.2，复测 CUDA GPU 路径；ONNX Runtime 为 MIT，CUDA/cuDNN 运行库遵循 NVIDIA 的许可。本仓库不重新分发 NVIDIA 运行库。

两种原生对照工具均不是核心库或 Web 服务的依赖，没有替换默认纯 C# 推理。C++ GPU 基准使用上述固定 Fun-ASR（Apache-2.0）与 llama.cpp/ggml（MIT），保留原计算图，设备/参数适配与计时包装由 C# 生成器产生。原生源码、构建物与模型仅放在被 Git 忽略的开发目录，不提交，也不作为产品依赖；结果、来源散列和适配边界见 doc/gpu-native-comparison-20261012.md。
