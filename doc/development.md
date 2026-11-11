# 代码规范与验证

## 源码规范

- C# 命名空间以 `FunAsrNet` 开头；核心库为 `FunAsrNet`，Web 服务为 `FunAsrNet.Server`。
- UTF-8、LF 换行，C# 四空格缩进，Web/JSON 两空格缩进，不保留行尾空白。
- C# 大括号独立成行；if/for/foreach/while/using/lock 即使只有一条语句也写完整块。
- 类型和方法提供中文说明，参数说明包含单位、范围或所有权；公开返回值说明结果与生命期。复杂内核注释解释布局、舍入顺序及边界条件。
- 普通方法使用 XML `summary/param/returns`。局部函数的 XML 结构放在 `//` 注释内，因为 C# 编译器不将局部函数作为 XML 文档成员；检查工具同样验证这些说明。
- `Directory.Build.props` 启用 XML 文档生成和警告视为错误；`.editorconfig` 定义格式。
- 托管推理保留所选权重精度，支持固定 F16/Q8 与原始权重的 FP32 展开；同一性能对照组不改变权重量化级别。新增浮点归约顺序必须注明并验证误差，严格位一致的已有检查不得放宽来迁就优化。

## 本地检查

从仓库根目录运行：

```text
dotnet build FunAsrNet.sln -c Release
dotnet format whitespace FunAsrNet.sln --verify-no-changes --no-restore
dotnet run --project tools/FunAsrNet.CodeStyle -c Release --no-build -- .
dotnet run --project tests/FunAsrNet.Checks -c Release --no-build
```

规范工具仅引用 SDK 自带的 Roslyn，不为推理程序增加 NuGet 或原生依赖。它检查方法及参数是否有中文说明、控制流大括号、UTF-8 和 LF。它不能代替审阅注释含义，也不能证明算法正确。

修改共享算子线程组时，也运行休眠/连续任务和多个调用方的压力检查，并在 ARM64 上验证；仅在 x64 上通过不足以证明存储/加载通知握手正确。固定线程亲和性时，调用线程与后台计算线程应分占配置的核心，结束后恢复调用方状态。

数值测试覆盖 G.711 全量编码/解码参考、SIMD 尾部和未对齐输入、矩阵/注意力/归一化、全部半精度舍入边界、重采样、流式 WAV 短读和坏输入，以及有界推理队列的取消和上下文隔离。固定参考数据在 `testdata/`，不会在测试时运行旧 Python/C++ 工具。

服务运行且没有业务请求时执行接口验收：

```text
dotnet run --project tests/FunAsrNet.DemoChecks -c Release --no-build -- http://localhost:18080
```

HTTPS 验收的第二个参数为预先确认的服务证书 SHA-256 指纹，不使用“接受所有证书”的回调：

```text
dotnet run --project tests/FunAsrNet.DemoChecks -c Release --no-build -- https://服务器地址:18443 证书SHA256指纹
```

验收包含 HTTP/WAV/TTS、48 kHz 实时输入、同时存在的不同长度会话、单发音人元数据、参数拒绝、静音音量、随机种子确定性及会话上限。测试会短时占用模型和全部会话配额，不要在业务繁忙时执行。

安装 AISHELL3 后，接口验收还检查 174 个编号、非法编号拒绝、三位发音人并发生成、跨编号确定性以及 PCM16/PCMA/PCMU 流式输出。全部发音人的离线有效性和双上下文并发检查使用：

```text
dotnet run --project tests/FunAsrNet.VoiceChecks -c Release -- models/aishell3 results/voices
```

增加第三个参数 `http://localhost:18080` 可对九位发音人的样例进行 ASR 回读抽测。结果写入 `results.json`，选定编号的 WAV 保存在输出目录；数据和生成音频不提交 Git。回读不等于人工听感评分，也不能证明所有文本的清晰度。

浏览器文件使用固定版本 Prettier 3.6.2 整理。可选维护命令：

```text
npx --yes prettier@3.6.2 --check "src/FunAsrNet.Server/wwwroot/*.{js,css,html}"
node --check src/FunAsrNet.Server/wwwroot/app.js
node --check src/FunAsrNet.Server/wwwroot/capture.js
```

Node/Prettier 仅用于编辑时检查，不是构建 C# 服务或运行推理的依赖。

## 性能测量

无需模型的特征、RoPE 和 Greedy 微基准：

```text
dotnet run --project tests/FunAsrNet.Checks -c Release --no-build -- --frontend-benchmark
```

该入口使用固定随机种子，交替输出参考与优化实现的 JSONL，含 3 轮预热、15 轮正式时间和分配量；它不执行全量检查，也不代表完整 ASR/TTS 的耗时。亲和性与多进程复现命令见 [2026-11-11 内核记录](frontend-kernels-20261111.md)。

固定模型、输入采样率、线程、CPU 亲和性和量化精度。将冷启动、模型预热、单请求计算、排队与网络延迟分别报告。持续测试保留慢样本并记录温度；不要只挑最快一轮，也不要用协议功能测试替代真实 PJSIP 通话准确率和并发容量评估。

`tools/FunAsrNet.Benchmarks` 提供相同输入准备、在线 ASR 内部计时、离线 TTS 算子计时和波形误差校验。历史原版对照条件见 [性能比较](performance.md)。原始 JSONL 显式标记预热，保留温度、频率和等待时间；统计时只排除预热，不删除慢样本。

CPU/GPU 最新测量、全部批次及精度边界见 [2026-11-11 第二轮报告](cpu-gpu-round2-20261111.md)，原始权重来源见 [前轮报告](cpu-gpu-original-20261111.md)。基准入口可显式指定模型，避免用文件名推断精度：

```text
dotnet run --project tools/FunAsrNet.Benchmarks -c Release -- asr . input.f32 5 --encoder models/funasr-encoder-original-f32.gguf --decoder models/qwen3-original-f32.gguf --gpu-cache-mib 4096 --vulkan --affinity 0x55
dotnet run --project tools/FunAsrNet.Benchmarks -c Release -- kernels --vulkan --affinity 0x55
dotnet run --project tools/FunAsrNet.Benchmarks -c Release -- gpu-check
```

`input.f32` 为 16 kHz、单声道、little-endian float32。完整 ASR 默认沿用提示词缓存；对照时双方统一设置 `FUNASRNET_DISABLE_PROMPT_CACHE=1`。`kernels` 使用固定种子检查 FP32/Q8 投影并报告 3 轮预热、15 轮正式微基准，不代表模型总耗时。`gpu-check` 验证形状边界、分组结果及缓存逐出重传；用 `FUNASRNET_VULKAN_INTEGER_DOT=0` 验证整数点积回退，`FUNASRNET_VULKAN_SPECIALIZED_FLOAT=0` 可禁用浮点专用管线。

`gpu-check` 还以独立双精度参考验证 FP32 注意力，覆盖 1–4,096 帧、尾通道、尾查询及高对比输入。新增诊断开关须在创建模型/后端之前设置：`FUNASRNET_AVX_ROWS6=0` 回到 CPU 四行矩阵；`FUNASRNET_VULKAN_ATTENTION=0` 关闭 GPU 注意力；`FUNASRNET_VULKAN_ATTENTION_TILE=0` 使用单查询内核；`FUNASRNET_VULKAN_WIDE=0` 关闭 FP32 宽子组。运行时回退检查使用 `DOTNET_EnableAVXVNNI=0`、`DOTNET_EnableAVX2=0` 或 `DOTNET_EnableHWIntrinsic=0`；每次分别在新进程设置。

`tools/ExportOriginalAsr.py` 仅用于固定官方 checkpoint 的 BF16/FP32 张量无损导出，需要 Python 3.11+、PyTorch、NumPy 和匹配的 GGUF 模板，不是生产依赖。`tools/CompileVulkanShaders.ps1 -Compiler <glslang路径> -VerifyOnly` 使用 glslang 16.6.0 核对九种嵌入 SPIR-V；去掉 `-VerifyOnly` 才会重新生成源码中的二进制，运行推理不需要编译器。

可选 Vulkan 源码位于 `src/FunAsrNet.Vulkan`。项目自有代码仍执行中文注释检查；`Vendor` 中保留来源与许可的第三方 Vulkan ABI 绑定不重写为本项目风格，临时 `artifacts` 不参与源码检查。来源和改动在 Vendor README 中明确列出。原生对照工具独立构建，核心服务不会传递引用其 ONNX Runtime 包。
