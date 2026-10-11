using System.Runtime.CompilerServices;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
public sealed partial class TensorMath
{
    /// <summary>
    /// 复用输入准备工作，依次计算多个独立线性投影。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public float[][] LinearMany(float[] input, int frames, Weight[] weights)
    {
        if (weights.Length == 0)
        {
            return [];
        }

        if (FloatBackend is IGroupedComputeBackend groupedBackend
            && groupedBackend.TryLinearMany(input, frames, weights, out var groupedOutputs))
        {
            return groupedOutputs!;
        }

        if (!Simd || !Dp.IsSupported || weights.Any(w => !w.QuantizedInterleaved))
        {
            return weights.Select(w => Linear(input, frames, w)).ToArray();
        }

        int n = weights[0].Columns;
        if (input.Length != frames * n || weights.Any(w => w.Columns != n))
        {
            throw new ArgumentException("Grouped projection dimensions differ");
        }

        long started = LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
        var outputs = weights.Select(w => TensorWorkspace.Allocate(frames * w.Rows)).ToArray();
        var blocks = weights.Select(w => (w.Rows + 31) / 32).ToArray();
        var q = new sbyte[input.Length];
        var scales = new float[input.Length / 32];
        Quantize(input, q, scales);
        For(
            blocks.Sum(),
            index =>
            {
                int which = 0;
                while (index >= blocks[which])
                {
                    index -= blocks[which++];
                }

                var w = weights[which];
                var output = outputs[which];
                int m = w.Rows;
                for (int row = index * 32; row < Math.Min(m, index * 32 + 32); row += 4)
                {
                    int t = 0, valid = Math.Min(4, m - row);
                    for (; t + 3 < frames; t += 4)
                    {
                        Q8Interleaved4(w.Quantized!, w.Scales!, row * n, q, scales, t * n, output, t * m + row, n, m, null, row, valid);
                    }

                    for (; t < frames; ++t)
                    {
                        Q8Interleaved1(w.Quantized!, w.Scales!, row * n, q, scales, t * n, output, t * m + row, n, m, null, row, valid);
                    }
                }
            });
        LinearMeasured?.Invoke(
            $"Q8Group {frames}x{n}x{weights.Sum(w => w.Rows)} {weights[0].Name}",
            System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        return outputs;
    }

    /// <summary>
    /// 使用交错浮点权重布局执行线性投影。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    private void InterleavedFloatLinear(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows;
        if (!Simd || !AdvSimd.Arm64.IsSupported)
        {
            For(
                m,
                row =>
                {
                    for (int t = 0; t < frames; ++t)
                    {
                        float sum = 0;
                        for (int k = 0; k < n; ++k)
                        {
                            sum += input[t * n + k] * w.Values![row / 8 * 8 * n + k * 8 + row % 8];
                        }

                        output[t * m + row] = sum + (bias?.Values?[row] ?? 0);
                    }
                });
            return;
        }

        int full = frames / 8 * 8;
        var packed = ArrayPool<float>.Shared.Rent(Math.Max(1, full * n));
        try
        {
            PackEightRows(input, packed, full, n);
            For(
                (m + 31) / 32,
                block =>
                {
                    for (int row = block * 32; row < Math.Min(m, block * 32 + 32); row += 8)
                    {
                        int t = 0, valid = Math.Min(8, m - row);
                        for (; t + 7 < frames; t += 8)
                        {
                            FloatInterleavedPacked8(w.Values!, row * n, packed, t * n, output, t * m + row, n, m, bias?.Values, row, valid);
                        }

                        for (; t + 3 < frames; t += 4)
                        {
                            FloatInterleaved4(w.Values!, row * n, input, t * n, output, t * m + row, n, m, bias?.Values, row, valid);
                        }

                        for (; t < frames; ++t)
                        {
                            FloatInterleaved1(w.Values!, row * n, input, t * n, output, t * m + row, n, m, bias?.Values, row, valid);
                        }
                    }
                });
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
        }
    }

    /// <summary>
    /// 将有效通道写入输出区，不写入尾部填充位置。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="at">目标数组的起始元素位置。</param>
    /// <param name="count">需要处理的元素或任务数量。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreChannels(Vector128<float> a, float[] output, int at, int count)
    {
        if (count >= 4)
        {
            a.StoreUnsafe(ref output[at]);
        }
        else
        {
            for (int i = 0; i < count; ++i)
            {
                output[at + i] = a.GetElement(i);
            }
        }
    }

    /// <summary>
    /// 加载有效通道偏置，不足一组的尾部用零补齐。
    /// </summary>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="row">第一个输出通道的编号。</param>
    /// <param name="valid">当前尾块中实际有效的通道数量。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> LoadBias(float[]? bias, int row, int valid)
    {
        if (bias is null)
        {
            return Vector128<float>.Zero;
        }

        if (valid >= 4)
        {
            return Vector128.LoadUnsafe(ref bias[row]);
        }

        return Vector128.Create(valid > 0 ? bias[row] : 0, valid > 1 ? bias[row + 1] : 0, valid > 2 ? bias[row + 2] : 0, 0);
    }

    /// <summary>
    /// 计算四个时间位置的交错浮点矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="row">第一个输出通道的编号。</param>
    /// <param name="valid">当前尾块中实际有效的通道数量。</param>
    private static void FloatInterleaved4(
        float[] weights,
        int wo,
        float[] input,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row,
        int valid)
    {
        ref float wp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weights), wo);
        ref float x = ref MemoryMarshal.GetArrayDataReference(input);
        ref float x0p = ref Unsafe.Add(ref x, xo + 0 * n);
        var a0l = Vector128<float>.Zero;
        var a0h = a0l;
        ref float x1p = ref Unsafe.Add(ref x, xo + 1 * n);
        var a1l = Vector128<float>.Zero;
        var a1h = a1l;
        ref float x2p = ref Unsafe.Add(ref x, xo + 2 * n);
        var a2l = Vector128<float>.Zero;
        var a2h = a2l;
        ref float x3p = ref Unsafe.Add(ref x, xo + 3 * n);
        var a3l = Vector128<float>.Zero;
        var a3h = a3l;
        int k = 0;
        for (; k <= n - 4; k += 4)
        {
            var x0 = Vector128.LoadUnsafe(ref x0p, (nuint)k);
            var x1 = Vector128.LoadUnsafe(ref x1p, (nuint)k);
            var x2 = Vector128.LoadUnsafe(ref x2p, (nuint)k);
            var x3 = Vector128.LoadUnsafe(ref x3p, (nuint)k);
            {
                var wl = Vector128.LoadUnsafe(ref wp, 0);
                var wh = Vector128.LoadUnsafe(ref wp, 4);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 0);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 0);
                a1l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1l, wl, x1, 0);
                a1h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1h, wh, x1, 0);
                a2l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2l, wl, x2, 0);
                a2h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2h, wh, x2, 0);
                a3l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3l, wl, x3, 0);
                a3h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3h, wh, x3, 0);
            }

            {
                var wl = Vector128.LoadUnsafe(ref wp, 8);
                var wh = Vector128.LoadUnsafe(ref wp, 12);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 1);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 1);
                a1l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1l, wl, x1, 1);
                a1h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1h, wh, x1, 1);
                a2l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2l, wl, x2, 1);
                a2h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2h, wh, x2, 1);
                a3l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3l, wl, x3, 1);
                a3h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3h, wh, x3, 1);
            }

            {
                var wl = Vector128.LoadUnsafe(ref wp, 16);
                var wh = Vector128.LoadUnsafe(ref wp, 20);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 2);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 2);
                a1l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1l, wl, x1, 2);
                a1h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1h, wh, x1, 2);
                a2l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2l, wl, x2, 2);
                a2h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2h, wh, x2, 2);
                a3l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3l, wl, x3, 2);
                a3h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3h, wh, x3, 2);
            }

            {
                var wl = Vector128.LoadUnsafe(ref wp, 24);
                var wh = Vector128.LoadUnsafe(ref wp, 28);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 3);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 3);
                a1l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1l, wl, x1, 3);
                a1h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1h, wh, x1, 3);
                a2l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2l, wl, x2, 3);
                a2h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2h, wh, x2, 3);
                a3l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3l, wl, x3, 3);
                a3h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3h, wh, x3, 3);
            }

            wp = ref Unsafe.Add(ref wp, 32);
        }

        for (; k < n; ++k)
        {
            var wl = Vector128.LoadUnsafe(ref wp);
            var wh = Vector128.LoadUnsafe(ref wp, 4);
            var x0 = Vector128.Create(Unsafe.Add(ref x0p, k));
            a0l = AdvSimd.FusedMultiplyAdd(a0l, wl, x0);
            a0h = AdvSimd.FusedMultiplyAdd(a0h, wh, x0);
            var x1 = Vector128.Create(Unsafe.Add(ref x1p, k));
            a1l = AdvSimd.FusedMultiplyAdd(a1l, wl, x1);
            a1h = AdvSimd.FusedMultiplyAdd(a1h, wh, x1);
            var x2 = Vector128.Create(Unsafe.Add(ref x2p, k));
            a2l = AdvSimd.FusedMultiplyAdd(a2l, wl, x2);
            a2h = AdvSimd.FusedMultiplyAdd(a2h, wh, x2);
            var x3 = Vector128.Create(Unsafe.Add(ref x3p, k));
            a3l = AdvSimd.FusedMultiplyAdd(a3l, wl, x3);
            a3h = AdvSimd.FusedMultiplyAdd(a3h, wh, x3);
            wp = ref Unsafe.Add(ref wp, 8);
        }

        var bl = LoadBias(bias, row, valid);
        var bh = LoadBias(bias, row + 4, valid - 4);
        StoreChannels(a0l + bl, output, yo + 0 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a0h + bh, output, yo + 0 * stride + 4, valid - 4);
        }

        StoreChannels(a1l + bl, output, yo + 1 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a1h + bh, output, yo + 1 * stride + 4, valid - 4);
        }

        StoreChannels(a2l + bl, output, yo + 2 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a2h + bh, output, yo + 2 * stride + 4, valid - 4);
        }

        StoreChannels(a3l + bl, output, yo + 3 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a3h + bh, output, yo + 3 * stride + 4, valid - 4);
        }
    }

    /// <summary>
    /// 计算单个时间位置的交错浮点矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="row">第一个输出通道的编号。</param>
    /// <param name="valid">当前尾块中实际有效的通道数量。</param>
    private static void FloatInterleaved1(
        float[] weights,
        int wo,
        float[] input,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row,
        int valid)
    {
        ref float wp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weights), wo);
        ref float x = ref MemoryMarshal.GetArrayDataReference(input);
        ref float x0p = ref Unsafe.Add(ref x, xo + 0 * n);
        var a0l = Vector128<float>.Zero;
        var a0h = a0l;
        int k = 0;
        for (; k <= n - 4; k += 4)
        {
            var x0 = Vector128.LoadUnsafe(ref x0p, (nuint)k);
            {
                var wl = Vector128.LoadUnsafe(ref wp, 0);
                var wh = Vector128.LoadUnsafe(ref wp, 4);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 0);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 0);
            }

            {
                var wl = Vector128.LoadUnsafe(ref wp, 8);
                var wh = Vector128.LoadUnsafe(ref wp, 12);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 1);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 1);
            }

            {
                var wl = Vector128.LoadUnsafe(ref wp, 16);
                var wh = Vector128.LoadUnsafe(ref wp, 20);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 2);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 2);
            }

            {
                var wl = Vector128.LoadUnsafe(ref wp, 24);
                var wh = Vector128.LoadUnsafe(ref wp, 28);
                a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, x0, 3);
                a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, x0, 3);
            }

            wp = ref Unsafe.Add(ref wp, 32);
        }

        for (; k < n; ++k)
        {
            var wl = Vector128.LoadUnsafe(ref wp);
            var wh = Vector128.LoadUnsafe(ref wp, 4);
            var x0 = Vector128.Create(Unsafe.Add(ref x0p, k));
            a0l = AdvSimd.FusedMultiplyAdd(a0l, wl, x0);
            a0h = AdvSimd.FusedMultiplyAdd(a0h, wh, x0);
            wp = ref Unsafe.Add(ref wp, 8);
        }

        var bl = LoadBias(bias, row, valid);
        var bh = LoadBias(bias, row + 4, valid - 4);
        StoreChannels(a0l + bl, output, yo + 0 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a0h + bh, output, yo + 0 * stride + 4, valid - 4);
        }
    }

    /// <summary>
    /// 使用交错 Q8 权重和分组缩放执行线性投影。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    private void InterleavedQuantizedLinear(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows;
        var q = new sbyte[input.Length];
        var scales = new float[input.Length / 32];
        Quantize(input, q, scales);
        if (!Simd || !Dp.IsSupported)
        {
            For(
                m,
                row =>
                {
                    for (int t = 0; t < frames; ++t)
                    {
                        float sum = 0;
                        for (int k = 0; k < n; k += 32)
                        {
                            int dot = 0;
                            for (int j = 0; j < 32; ++j)
                            {
                                dot += q[t * n + k + j] * w.Quantized![row / 4 * 4 * n + (k + j) / 4 * 16 + row % 4 * 4 + j % 4];
                            }

                            sum += dot * (w.Scales![row / 4 * 4 * (n / 32) + k / 32 * 4 + row % 4] * scales[(t * n + k) / 32]);
                        }

                        output[t * m + row] = sum + (bias?.Values?[row] ?? 0);
                    }
                });
            return;
        }

        For(
            (m + 31) / 32,
            block =>
            {
                for (int row = block * 32; row < Math.Min(m, block * 32 + 32); row += 4)
                {
                    int t = 0, valid = Math.Min(4, m - row);
                    for (; t + 3 < frames; t += 4)
                    {
                        Q8Interleaved4(w.Quantized!, w.Scales!, row * n, q, scales, t * n, output, t * m + row, n, m, bias?.Values, row, valid);
                    }

                    for (; t < frames; ++t)
                    {
                        Q8Interleaved1(w.Quantized!, w.Scales!, row * n, q, scales, t * n, output, t * m + row, n, m, bias?.Values, row, valid);
                    }
                }
            });
    }

    /// <summary>
    /// 计算四个时间位置的交错 Q8 矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="weightScales">Q8 权重每 32 元素分组的缩放因子。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="inputScales">Q8 激活每 32 元素分组的缩放因子。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="row">第一个输出通道的编号。</param>
    /// <param name="valid">当前尾块中实际有效的通道数量。</param>
    private static void Q8Interleaved4(
        sbyte[] weights,
        float[] weightScales,
        int wo,
        sbyte[] input,
        float[] inputScales,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row,
        int valid)
    {
        ref sbyte wp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weights), wo);
        ref float sp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weightScales), wo / 32);
        ref sbyte xp = ref MemoryMarshal.GetArrayDataReference(input);
        ref float sx = ref MemoryMarshal.GetArrayDataReference(inputScales);
        ref sbyte x0p = ref Unsafe.Add(ref xp, xo + 0 * n);
        var a0 = Vector128<float>.Zero;
        ref sbyte x1p = ref Unsafe.Add(ref xp, xo + 1 * n);
        var a1 = Vector128<float>.Zero;
        ref sbyte x2p = ref Unsafe.Add(ref xp, xo + 2 * n);
        var a2 = Vector128<float>.Zero;
        ref sbyte x3p = ref Unsafe.Add(ref xp, xo + 3 * n);
        var a3 = Vector128<float>.Zero;
        for (int k = 0; k < n; k += 32)
        {
            var x0l = Vector128.LoadUnsafe(ref x0p, (nuint)k);
            var x0h = Vector128.LoadUnsafe(ref x0p, (nuint)(k + 16));
            var d0 = Vector128<int>.Zero;
            var x1l = Vector128.LoadUnsafe(ref x1p, (nuint)k);
            var x1h = Vector128.LoadUnsafe(ref x1p, (nuint)(k + 16));
            var d1 = Vector128<int>.Zero;
            var x2l = Vector128.LoadUnsafe(ref x2p, (nuint)k);
            var x2h = Vector128.LoadUnsafe(ref x2p, (nuint)(k + 16));
            var d2 = Vector128<int>.Zero;
            var x3l = Vector128.LoadUnsafe(ref x3p, (nuint)k);
            var x3h = Vector128.LoadUnsafe(ref x3p, (nuint)(k + 16));
            var d3 = Vector128<int>.Zero;
            {
                var w = Vector128.LoadUnsafe(ref wp, 0);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0l, 0);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1l, 0);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2l, 0);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3l, 0);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 16);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0l, 1);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1l, 1);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2l, 1);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3l, 1);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 32);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0l, 2);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1l, 2);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2l, 2);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3l, 2);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 48);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0l, 3);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1l, 3);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2l, 3);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3l, 3);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 64);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0h, 0);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1h, 0);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2h, 0);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3h, 0);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 80);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0h, 1);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1h, 1);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2h, 1);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3h, 1);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 96);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0h, 2);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1h, 2);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2h, 2);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3h, 2);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 112);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0h, 3);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x1h, 3);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x2h, 3);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x3h, 3);
            }

            var sw = Vector128.LoadUnsafe(ref sp);
            a0 = AdvSimd.FusedMultiplyAdd(a0, Vector128.ConvertToSingle(d0), sw * Vector128.Create(Unsafe.Add(ref sx, (xo + 0 * n + k) / 32)));
            a1 = AdvSimd.FusedMultiplyAdd(a1, Vector128.ConvertToSingle(d1), sw * Vector128.Create(Unsafe.Add(ref sx, (xo + 1 * n + k) / 32)));
            a2 = AdvSimd.FusedMultiplyAdd(a2, Vector128.ConvertToSingle(d2), sw * Vector128.Create(Unsafe.Add(ref sx, (xo + 2 * n + k) / 32)));
            a3 = AdvSimd.FusedMultiplyAdd(a3, Vector128.ConvertToSingle(d3), sw * Vector128.Create(Unsafe.Add(ref sx, (xo + 3 * n + k) / 32)));
            wp = ref Unsafe.Add(ref wp, 128);
            sp = ref Unsafe.Add(ref sp, 4);
        }

        var b = LoadBias(bias, row, valid);
        StoreChannels(a0 + b, output, yo + 0 * stride, valid);
        StoreChannels(a1 + b, output, yo + 1 * stride, valid);
        StoreChannels(a2 + b, output, yo + 2 * stride, valid);
        StoreChannels(a3 + b, output, yo + 3 * stride, valid);
    }

    /// <summary>
    /// 计算单个时间位置的交错 Q8 矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="weightScales">Q8 权重每 32 元素分组的缩放因子。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="inputScales">Q8 激活每 32 元素分组的缩放因子。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="row">第一个输出通道的编号。</param>
    /// <param name="valid">当前尾块中实际有效的通道数量。</param>
    private static void Q8Interleaved1(
        sbyte[] weights,
        float[] weightScales,
        int wo,
        sbyte[] input,
        float[] inputScales,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row,
        int valid)
    {
        ref sbyte wp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weights), wo);
        ref float sp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weightScales), wo / 32);
        ref sbyte xp = ref MemoryMarshal.GetArrayDataReference(input);
        ref float sx = ref MemoryMarshal.GetArrayDataReference(inputScales);
        ref sbyte x0p = ref Unsafe.Add(ref xp, xo + 0 * n);
        var a0 = Vector128<float>.Zero;
        for (int k = 0; k < n; k += 32)
        {
            var x0l = Vector128.LoadUnsafe(ref x0p, (nuint)k);
            var x0h = Vector128.LoadUnsafe(ref x0p, (nuint)(k + 16));
            var d0 = Vector128<int>.Zero;
            var d1 = d0;
            var d2 = d0;
            var d3 = d0;
            {
                var w = Vector128.LoadUnsafe(ref wp, 0);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0l, 0);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 16);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x0l, 1);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 32);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x0l, 2);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 48);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x0l, 3);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 64);
                d0 = Dp.DotProductBySelectedQuadruplet(d0, w, x0h, 0);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 80);
                d1 = Dp.DotProductBySelectedQuadruplet(d1, w, x0h, 1);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 96);
                d2 = Dp.DotProductBySelectedQuadruplet(d2, w, x0h, 2);
            }

            {
                var w = Vector128.LoadUnsafe(ref wp, 112);
                d3 = Dp.DotProductBySelectedQuadruplet(d3, w, x0h, 3);
            }

            var sw = Vector128.LoadUnsafe(ref sp);
            a0 = AdvSimd.FusedMultiplyAdd(a0, Vector128.ConvertToSingle((d0 + d1) + (d2 + d3)), sw * Vector128.Create(Unsafe.Add(ref sx, (xo + 0 * n + k) / 32)));
            wp = ref Unsafe.Add(ref wp, 128);
            sp = ref Unsafe.Add(ref sp, 4);
        }

        var b = LoadBias(bias, row, valid);
        StoreChannels(a0 + b, output, yo + 0 * stride, valid);
    }
}
