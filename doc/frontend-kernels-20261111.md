# 特征提取与解码内核优化（2026-11-11）

本轮在提交 `96a2ec9902efda3326e5bc44ffb085f6b21a228e` 的基础上补齐仍走标量的特征提取、RoPE 与词表查找。已有 AVX2/FMA、AVX-VNNI、ARM64 NEON/SDOT、Q8 分组投影和计算线程组继续沿用。本轮没有改变 F16/Q8 精度，也没有新增原生依赖。

## 实现

- 固定 512 点 FFT 的旋转因子按原来的浮点递推顺序生成一次，各级和各帧共享只读表。蝶形与功率谱使用 `Vector256` / `Vector128`，硬件不支持时回退标量。独立乘法与加减保持原中间舍入，不引入 FMA。
- 每帧的 257 点功率谱仅计算一次；80 个 Mel 滤波器只扫描各自连续的非零范围，并保持频点累加次序。`Compute80` 的固定工作区使用约 6.6 KiB 的调用栈空间，实例和线程之间没有共享可写工作缓冲。
- Qwen 半分割 RoPE 使用 256/128 位 SIMD；同一帧的各注意力头复用系数切片。`TensorMath.Simd=false` 时执行标量旋转。
- Greedy 用向量比较跳过不可能更新最大值的词表块；需要更新时仍顺序比较，保留最早并列项、NaN、正负无穷和正负零的旧行为。实际收益与分数分布有关，表中的词表是确定性随机合成数据。

## 同机微基准

Windows x64、Intel Core i7-12700、.NET Runtime 10.0.15、Release、关闭分层编译；单调用线程固定亲和性 `0x1`，没有锁频，存在背景调度波动。参考算法独立保留在测试中：特征提取来自优化前源码，RoPE 和 Greedy 直接保留旧公式与扫描规则。

三个独立进程块；每块、每个内核、每个实现预热 3 次、正式 15 次，两版每轮交替先后。每个实现每个内核共 45 个正式样本，合计 **450 次正式测量、90 次预热**。统计只排除标记的预热，保留所有慢样本。P95 为最近秩，即每组排序后的第 43 项。

| 合成输入 / 操作 | 参考中位数 | 本次中位数 | 加速倍数 | 参考 / 本次 P95 |
|---|---:|---:|---:|---:|
| 1 秒音频，Compute80 | 2.048 ms | 0.389 ms | 5.26× | 2.567 / 0.448 ms |
| 6 秒音频，Compute80 | 13.367 ms | 2.385 ms | 5.60× | 24.750 / 5.815 ms |
| 28 秒音频，Compute80 | 59.381 ms | 10.635 ms | 5.58× | 99.846 / 17.852 ms |
| 512 头 × 128 通道 RoPE，重复 64 次 | 1.690 ms | 0.283 ms | 5.97× | 1.779 / 0.352 ms |
| 151936 项 Greedy，重复 64 次 | 4.374 ms | 0.579 ms | 7.55× | 5.241 / 0.648 ms |

Compute80 每次托管分配减少 **5,768 字节**，来自三个临时数组的消除；结果数组仍按输入帧数分配。6 秒输入由 197,152 减至 191,384 字节。静态旋转表初始化与滤波器构造在预热和计时之外，不把初始化分配算作稳态每次调用。RoPE 与 Greedy 的测量循环均零托管分配。

这些是组合内核优化的收益，不能单独归因于 SIMD，也不能直接推算完整模型的加速倍数。基准音频与分数由固定种子 11112026 生成；不含模型加载、编码器/解码器矩阵计算、服务排队或网络。仓库没有模型权重，本轮未测完整 ASR/TTS 转写、波形与端到端耗时，也未在真实 ARM64 硬件上验证。

原始逐轮记录为 [第一块](benchmarks/20261111-frontend-kernels/kernels.jsonl)、[第二块](benchmarks/20261111-frontend-kernels/kernels-block2.jsonl)、[第三块](benchmarks/20261111-frontend-kernels/kernels-block3.jsonl)，机器可读统计见 [summary.json](benchmarks/20261111-frontend-kernels/summary.json)。

## 正确性与复现

- 默认 SIMD：**9,373,901 项**通过。
- `COMPlus_EnableAVX=0`：**9,179,220 项**通过，覆盖新内核的 128 位回退。
- `COMPlus_EnableHWIntrinsic=0`：**9,179,220 项**通过，覆盖纯标量回退。
- 新增检查包含静音、脉冲、正弦、随机音频、直流输入、帧边界和 LFR；80 维与 LFR 特征逐位对照优化前算法。共享 FilterBank 的并发调用相互隔离。
- RoPE 检查未对齐切片、向量尾部、零长度、正负零、非正规数、大数和切片两端保护，结果逐位一致。Greedy 检查小数组、真实词表规模、递增/递减、并列、NaN 和无穷。
- 解决方案 Release 构建零警告、零错误；中文注释检查与修改文件的空白格式检查通过。

本机安装了 SDK 10.0.204，而仓库 `global.json` 要求最低 10.0.300；验证直接调用已安装的 SDK 驱动，未修改仓库 SDK 配置。实际构建命令：

```powershell
$env:DOTNET_CLI_TELEMETRY_OPTOUT='1'
dotnet 'C:/Program Files/dotnet/sdk/10.0.204/dotnet.dll' build FunAsrNet.sln -c Release
```

安装符合 global.json 的 SDK 后，可使用通常的 `dotnet build FunAsrNet.sln -c Release`。运行数值检查与微基准无需再次解析 SDK：

```powershell
dotnet tests/FunAsrNet.Checks/bin/Release/net10.0/FunAsrNet.Checks.dll
$env:COMPlus_EnableAVX='0'
dotnet tests/FunAsrNet.Checks/bin/Release/net10.0/FunAsrNet.Checks.dll
Remove-Item Env:COMPlus_EnableAVX
$env:COMPlus_EnableHWIntrinsic='0'
dotnet tests/FunAsrNet.Checks/bin/Release/net10.0/FunAsrNet.Checks.dll
Remove-Item Env:COMPlus_EnableHWIntrinsic
[System.Diagnostics.Process]::GetCurrentProcess().ProcessorAffinity = [IntPtr]1
dotnet tests/FunAsrNet.Checks/bin/Release/net10.0/FunAsrNet.Checks.dll --frontend-benchmark
```

每次运行微基准输出一组 JSONL，含硬件路径、亲和性、预热标记、逐轮耗时与分配字节数。源码与证据的 SHA-256 索引保存在同目录。
