# 代码规范与验证

## 源码规范

- C# 命名空间以 `FunAsrNet` 开头；核心库为 `FunAsrNet`，Web 服务为 `FunAsrNet.Server`。
- UTF-8、LF 换行，C# 四空格缩进，Web/JSON 两空格缩进，不保留行尾空白。
- C# 大括号独立成行；if/for/foreach/while/using/lock 即使只有一条语句也写完整块。
- 类型和方法提供中文说明，参数说明包含单位、范围或所有权；公开返回值说明结果与生命期。复杂内核注释解释布局、舍入顺序及边界条件。
- 普通方法使用 XML `summary/param/returns`。局部函数的 XML 结构放在 `//` 注释内，因为 C# 编译器不将局部函数作为 XML 文档成员；检查工具同样验证这些说明。
- `Directory.Build.props` 启用 XML 文档生成和警告视为错误；`.editorconfig` 定义格式。
- 托管推理保持原 F16/Q8 精度，不通过改变权重量化级别取得速度提升。

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

固定模型、输入采样率、线程、CPU 亲和性和量化精度。将冷启动、模型预热、单请求计算、排队与网络延迟分别报告。持续测试保留慢样本并记录温度；不要只挑最快一轮，也不要用协议功能测试替代真实 PJSIP 通话准确率和并发容量评估。

`tools/FunAsrNet.Benchmarks` 提供相同输入准备、在线 ASR 内部计时、离线 TTS 算子计时和波形误差校验。历史原版对照条件见 [性能比较](performance.md)。原始 JSONL 显式标记预热，保留温度、频率和等待时间；统计时只排除预热，不删除慢样本。

可选 Vulkan 源码位于 `src/FunAsrNet.Vulkan`。项目自有代码仍执行中文注释检查；`Vendor` 中保留来源与许可的第三方 Vulkan ABI 绑定不重写为本项目风格，临时 `artifacts` 不参与源码检查。来源和改动在 Vendor README 中明确列出。原生对照工具独立构建，核心服务不会传递引用其 ONNX Runtime 包。
