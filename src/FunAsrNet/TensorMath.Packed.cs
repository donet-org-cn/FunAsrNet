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
    // 仅打包较小的激活矩阵，模型权重保持行主序，
    // 避免重复保留约 0.9 GB 的编码器权重；十六个时间位置共享权重加载。
    /// <summary>
    /// 仅打包激活矩阵，复用行主序权重完成浮点线性投影。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    private void PackedFloatLinear(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows, full = frames / 16 * 16;
        var packed = ArrayPool<float>.Shared.Rent(full * n);
        try
        {
            for (int t = 0; t < full; t += 16)
            {
                for (int k = 0; k < n; ++k)
                {
                    for (int j = 0; j < 16; ++j)
                    {
                        packed[t * n + k * 16 + j] = input[(t + j) * n + k];
                    }
                }
            }

            int rowsPerBlock = m < 128 ? 4 : 16;
            For(
                (m + rowsPerBlock - 1) / rowsPerBlock,
                block =>
                {
                    int end = Math.Min(m, (block + 1) * rowsPerBlock), row = block * rowsPerBlock;
                    for (; row + 3 < end; row += 4)
                    {
                        int t = 0;
                        for (; t < full; t += 16)
                        {
                            Float16x4(w.Values!, row * n, packed, t * n, output, t * m + row, n, m, bias?.Values, row);
                        }

                        for (; t + 3 < frames; t += 4)
                        {
                            Dot4x4(w.Values!, row * n, input, t * n, output, t * m + row, n, m, bias?.Values, row);
                        }

                        for (; t < frames; ++t)
                        {
                            for (int j = 0; j < 4; ++j)
                            {
                                output[t * m + row + j] = (bias?.Values?[row + j] ?? 0) + AudioKernels.Dot(w.Values!.AsSpan((row + j) * n, n), input.AsSpan(t * n, n));
                            }
                        }
                    }

                    for (; row < end; ++row)
                    {
                        for (int t = 0; t < frames; ++t)
                        {
                            output[t * m + row] = (bias?.Values?[row] ?? 0) + AudioKernels.Dot(w.Values!.AsSpan(row * n, n), input.AsSpan(t * n, n));
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
    /// 计算十六个时间位置与四个输出通道的浮点矩阵小块。
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
    private static void Float16x4(
        float[] weights,
        int wo,
        float[] input,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row)
    {
        ref float w = ref MemoryMarshal.GetArrayDataReference(weights);
        ref float x = ref MemoryMarshal.GetArrayDataReference(input);
        ref float wr0 = ref Unsafe.Add(ref w, wo + 0 * n);
        ref float wr1 = ref Unsafe.Add(ref w, wo + 1 * n);
        ref float wr2 = ref Unsafe.Add(ref w, wo + 2 * n);
        ref float wr3 = ref Unsafe.Add(ref w, wo + 3 * n);
        ref float xp = ref Unsafe.Add(ref x, xo);
        var a00 = Vector128<float>.Zero;
        var a01 = Vector128<float>.Zero;
        var a02 = Vector128<float>.Zero;
        var a03 = Vector128<float>.Zero;
        var a10 = Vector128<float>.Zero;
        var a11 = Vector128<float>.Zero;
        var a12 = Vector128<float>.Zero;
        var a13 = Vector128<float>.Zero;
        var a20 = Vector128<float>.Zero;
        var a21 = Vector128<float>.Zero;
        var a22 = Vector128<float>.Zero;
        var a23 = Vector128<float>.Zero;
        var a30 = Vector128<float>.Zero;
        var a31 = Vector128<float>.Zero;
        var a32 = Vector128<float>.Zero;
        var a33 = Vector128<float>.Zero;
        int k = 0;
        for (; k <= n - 4; k += 4)
        {
            var w0 = Vector128.LoadUnsafe(ref wr0, (nuint)k);
            var w1 = Vector128.LoadUnsafe(ref wr1, (nuint)k);
            var w2 = Vector128.LoadUnsafe(ref wr2, (nuint)k);
            var w3 = Vector128.LoadUnsafe(ref wr3, (nuint)k);
            {
                var x0 = Vector128.LoadUnsafe(ref xp, 0);
                var x1 = Vector128.LoadUnsafe(ref xp, 4);
                var x2 = Vector128.LoadUnsafe(ref xp, 8);
                var x3 = Vector128.LoadUnsafe(ref xp, 12);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, x0, w0, 0);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, x1, w0, 0);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, x2, w0, 0);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, x3, w0, 0);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, x0, w1, 0);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, x1, w1, 0);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, x2, w1, 0);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, x3, w1, 0);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, x0, w2, 0);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, x1, w2, 0);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, x2, w2, 0);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, x3, w2, 0);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, x0, w3, 0);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, x1, w3, 0);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, x2, w3, 0);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, x3, w3, 0);
            }

            {
                var x0 = Vector128.LoadUnsafe(ref xp, 16);
                var x1 = Vector128.LoadUnsafe(ref xp, 20);
                var x2 = Vector128.LoadUnsafe(ref xp, 24);
                var x3 = Vector128.LoadUnsafe(ref xp, 28);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, x0, w0, 1);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, x1, w0, 1);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, x2, w0, 1);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, x3, w0, 1);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, x0, w1, 1);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, x1, w1, 1);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, x2, w1, 1);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, x3, w1, 1);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, x0, w2, 1);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, x1, w2, 1);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, x2, w2, 1);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, x3, w2, 1);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, x0, w3, 1);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, x1, w3, 1);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, x2, w3, 1);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, x3, w3, 1);
            }

            {
                var x0 = Vector128.LoadUnsafe(ref xp, 32);
                var x1 = Vector128.LoadUnsafe(ref xp, 36);
                var x2 = Vector128.LoadUnsafe(ref xp, 40);
                var x3 = Vector128.LoadUnsafe(ref xp, 44);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, x0, w0, 2);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, x1, w0, 2);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, x2, w0, 2);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, x3, w0, 2);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, x0, w1, 2);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, x1, w1, 2);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, x2, w1, 2);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, x3, w1, 2);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, x0, w2, 2);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, x1, w2, 2);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, x2, w2, 2);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, x3, w2, 2);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, x0, w3, 2);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, x1, w3, 2);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, x2, w3, 2);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, x3, w3, 2);
            }

            {
                var x0 = Vector128.LoadUnsafe(ref xp, 48);
                var x1 = Vector128.LoadUnsafe(ref xp, 52);
                var x2 = Vector128.LoadUnsafe(ref xp, 56);
                var x3 = Vector128.LoadUnsafe(ref xp, 60);
                a00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a00, x0, w0, 3);
                a01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a01, x1, w0, 3);
                a02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a02, x2, w0, 3);
                a03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a03, x3, w0, 3);
                a10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a10, x0, w1, 3);
                a11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a11, x1, w1, 3);
                a12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a12, x2, w1, 3);
                a13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a13, x3, w1, 3);
                a20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a20, x0, w2, 3);
                a21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a21, x1, w2, 3);
                a22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a22, x2, w2, 3);
                a23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a23, x3, w2, 3);
                a30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a30, x0, w3, 3);
                a31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a31, x1, w3, 3);
                a32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a32, x2, w3, 3);
                a33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a33, x3, w3, 3);
            }

            xp = ref Unsafe.Add(ref xp, 64);
        }

        for (; k < n; ++k)
        {
            var x0 = Vector128.LoadUnsafe(ref x, (nuint)(xo + k * 16 + 0));
            var x1 = Vector128.LoadUnsafe(ref x, (nuint)(xo + k * 16 + 4));
            var x2 = Vector128.LoadUnsafe(ref x, (nuint)(xo + k * 16 + 8));
            var x3 = Vector128.LoadUnsafe(ref x, (nuint)(xo + k * 16 + 12));
            var w0 = Vector128.Create(Unsafe.Add(ref w, wo + 0 * n + k));
            a00 = AdvSimd.FusedMultiplyAdd(a00, x0, w0);
            a01 = AdvSimd.FusedMultiplyAdd(a01, x1, w0);
            a02 = AdvSimd.FusedMultiplyAdd(a02, x2, w0);
            a03 = AdvSimd.FusedMultiplyAdd(a03, x3, w0);
            var w1 = Vector128.Create(Unsafe.Add(ref w, wo + 1 * n + k));
            a10 = AdvSimd.FusedMultiplyAdd(a10, x0, w1);
            a11 = AdvSimd.FusedMultiplyAdd(a11, x1, w1);
            a12 = AdvSimd.FusedMultiplyAdd(a12, x2, w1);
            a13 = AdvSimd.FusedMultiplyAdd(a13, x3, w1);
            var w2 = Vector128.Create(Unsafe.Add(ref w, wo + 2 * n + k));
            a20 = AdvSimd.FusedMultiplyAdd(a20, x0, w2);
            a21 = AdvSimd.FusedMultiplyAdd(a21, x1, w2);
            a22 = AdvSimd.FusedMultiplyAdd(a22, x2, w2);
            a23 = AdvSimd.FusedMultiplyAdd(a23, x3, w2);
            var w3 = Vector128.Create(Unsafe.Add(ref w, wo + 3 * n + k));
            a30 = AdvSimd.FusedMultiplyAdd(a30, x0, w3);
            a31 = AdvSimd.FusedMultiplyAdd(a31, x1, w3);
            a32 = AdvSimd.FusedMultiplyAdd(a32, x2, w3);
            a33 = AdvSimd.FusedMultiplyAdd(a33, x3, w3);
        }

        var b0 = Vector128.Create(bias?[row + 0] ?? 0);
        a00 += b0;
        a01 += b0;
        a02 += b0;
        a03 += b0;
        var b1 = Vector128.Create(bias?[row + 1] ?? 0);
        a10 += b1;
        a11 += b1;
        a12 += b1;
        a13 += b1;
        var b2 = Vector128.Create(bias?[row + 2] ?? 0);
        a20 += b2;
        a21 += b2;
        a22 += b2;
        a23 += b2;
        var b3 = Vector128.Create(bias?[row + 3] ?? 0);
        a30 += b3;
        a31 += b3;
        a32 += b3;
        a33 += b3;
        StoreTransposed(a00, a10, a20, a30, output, yo + 0 * stride, stride);
        StoreTransposed(a01, a11, a21, a31, output, yo + 4 * stride, stride);
        StoreTransposed(a02, a12, a22, a32, output, yo + 8 * stride, stride);
        StoreTransposed(a03, a13, a23, a33, output, yo + 12 * stride, stride);
    }

    /// <summary>
    /// 将四个向量转置后按输出行步长保存。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <param name="c">第三个输入操作数。</param>
    /// <param name="d">第四个输入向量。</param>
    /// <param name="y">当前操作的输出存储区。</param>
    /// <param name="at">目标数组的起始元素位置。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreTransposed(
        Vector128<float> a,
        Vector128<float> b,
        Vector128<float> c,
        Vector128<float> d,
        float[] y,
        int at,
        int stride)
    {
        var ab0 = AdvSimd.Arm64.ZipLow(a, b);
        var ab1 = AdvSimd.Arm64.ZipHigh(a, b);
        var cd0 = AdvSimd.Arm64.ZipLow(c, d);
        var cd1 = AdvSimd.Arm64.ZipHigh(c, d);
        AdvSimd.Arm64.ZipLow(ab0.AsInt64(), cd0.AsInt64()).AsSingle().StoreUnsafe(ref y[at]);
        AdvSimd.Arm64.ZipHigh(ab0.AsInt64(), cd0.AsInt64()).AsSingle().StoreUnsafe(ref y[at + stride]);
        AdvSimd.Arm64.ZipLow(ab1.AsInt64(), cd1.AsInt64()).AsSingle().StoreUnsafe(ref y[at + 2 * stride]);
        AdvSimd.Arm64.ZipHigh(ab1.AsInt64(), cd1.AsInt64()).AsSingle().StoreUnsafe(ref y[at + 3 * stride]);
    }

    /// <summary>
    /// 将 Q8 矩阵乘法分为时间与通道小块进行计算。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    private void TiledQuantizedLinear(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows;
        var q = new sbyte[input.Length];
        var scales = new float[input.Length / 32];
        Quantize(input, q, scales);
        For(
            (m + 31) / 32,
            block =>
            {
                int end = Math.Min(m, block * 32 + 32), row = block * 32;
                for (; row + 3 < end; row += 4)
                {
                    int t = 0;
                    for (; t + 1 < frames; t += 2)
                    {
                        Quantized2x4(w.Quantized!, w.Scales!, row * n, q, scales, t * n, output, t * m + row, n, m);
                    }

                    for (; t < frames; ++t)
                    {
                        Quantized1x4(w.Quantized!, w.Scales!, row * n, q, scales, t * n, output, t * m + row, n);
                    }

                    if (bias?.Values is not null)
                    {
                        for (t = 0; t < frames; ++t)
                        {
                            for (int j = 0; j < 4; ++j)
                            {
                                output[t * m + row + j] += bias.Values[row + j];
                            }
                        }
                    }
                }

                for (; row < end; ++row)
                {
                    for (int t = 0; t < frames; ++t)
                    {
                        output[t * m + row] = (bias?.Values?[row] ?? 0) + QuantizedDot(w.Quantized!, w.Scales!, row * n, q, scales, t * n, n);
                    }
                }
            });
    }

    /// <summary>
    /// 计算两个时间位置与四个输出通道的 Q8 矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="ws">Q8 权重的分组缩放因子。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="xs">Q8 激活的分组缩放因子。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    private static void Quantized2x4(
        sbyte[] weights,
        float[] ws,
        int wo,
        sbyte[] input,
        float[] xs,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride)
    {
        ref sbyte w = ref MemoryMarshal.GetArrayDataReference(weights);
        ref sbyte x = ref MemoryMarshal.GetArrayDataReference(input);
        ref float sw = ref MemoryMarshal.GetArrayDataReference(ws);
        ref float sx = ref MemoryMarshal.GetArrayDataReference(xs);
        var a00 = Vector128<float>.Zero;
        var a01 = Vector128<float>.Zero;
        var a10 = Vector128<float>.Zero;
        var a11 = Vector128<float>.Zero;
        var a20 = Vector128<float>.Zero;
        var a21 = Vector128<float>.Zero;
        var a30 = Vector128<float>.Zero;
        var a31 = Vector128<float>.Zero;
        for (int k = 0; k < n; k += 32)
        {
            var x0l = Vector128.LoadUnsafe(ref x, (nuint)(xo + 0 * n + k));
            var x0h = Vector128.LoadUnsafe(ref x, (nuint)(xo + 0 * n + k + 16));
            float s0 = Unsafe.Add(ref sx, (xo + 0 * n + k) / 32);
            var x1l = Vector128.LoadUnsafe(ref x, (nuint)(xo + 1 * n + k));
            var x1h = Vector128.LoadUnsafe(ref x, (nuint)(xo + 1 * n + k + 16));
            float s1 = Unsafe.Add(ref sx, (xo + 1 * n + k) / 32);
            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 0 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 0 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 0 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a00 = AdvSimd.FusedMultiplyAdd(a00, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
                var d1 = Dp.DotProduct(Vector128<int>.Zero, wl, x1l);
                d1 = Dp.DotProduct(d1, wh, x1h);
                a01 = AdvSimd.FusedMultiplyAdd(a01, Vector128.ConvertToSingle(d1), Vector128.Create(scale * s1));
            }

            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 1 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 1 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 1 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a10 = AdvSimd.FusedMultiplyAdd(a10, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
                var d1 = Dp.DotProduct(Vector128<int>.Zero, wl, x1l);
                d1 = Dp.DotProduct(d1, wh, x1h);
                a11 = AdvSimd.FusedMultiplyAdd(a11, Vector128.ConvertToSingle(d1), Vector128.Create(scale * s1));
            }

            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 2 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 2 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 2 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a20 = AdvSimd.FusedMultiplyAdd(a20, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
                var d1 = Dp.DotProduct(Vector128<int>.Zero, wl, x1l);
                d1 = Dp.DotProduct(d1, wh, x1h);
                a21 = AdvSimd.FusedMultiplyAdd(a21, Vector128.ConvertToSingle(d1), Vector128.Create(scale * s1));
            }

            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 3 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 3 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 3 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a30 = AdvSimd.FusedMultiplyAdd(a30, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
                var d1 = Dp.DotProduct(Vector128<int>.Zero, wl, x1l);
                d1 = Dp.DotProduct(d1, wh, x1h);
                a31 = AdvSimd.FusedMultiplyAdd(a31, Vector128.ConvertToSingle(d1), Vector128.Create(scale * s1));
            }
        }

        output[yo + 0] = Vector128.Sum(a00);
        output[yo + stride + 0] = Vector128.Sum(a01);
        output[yo + 1] = Vector128.Sum(a10);
        output[yo + stride + 1] = Vector128.Sum(a11);
        output[yo + 2] = Vector128.Sum(a20);
        output[yo + stride + 2] = Vector128.Sum(a21);
        output[yo + 3] = Vector128.Sum(a30);
        output[yo + stride + 3] = Vector128.Sum(a31);
    }

    /// <summary>
    /// 计算单个时间位置与四个输出通道的 Q8 矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="ws">Q8 权重的分组缩放因子。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="xs">Q8 激活的分组缩放因子。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    private static void Quantized1x4(
        sbyte[] weights,
        float[] ws,
        int wo,
        sbyte[] input,
        float[] xs,
        int xo,
        float[] output,
        int yo,
        int n)
    {
        ref sbyte w = ref MemoryMarshal.GetArrayDataReference(weights);
        ref sbyte x = ref MemoryMarshal.GetArrayDataReference(input);
        ref float sw = ref MemoryMarshal.GetArrayDataReference(ws);
        ref float sx = ref MemoryMarshal.GetArrayDataReference(xs);
        var a00 = Vector128<float>.Zero;
        var a10 = Vector128<float>.Zero;
        var a20 = Vector128<float>.Zero;
        var a30 = Vector128<float>.Zero;
        for (int k = 0; k < n; k += 32)
        {
            var x0l = Vector128.LoadUnsafe(ref x, (nuint)(xo + 0 * n + k));
            var x0h = Vector128.LoadUnsafe(ref x, (nuint)(xo + 0 * n + k + 16));
            float s0 = Unsafe.Add(ref sx, (xo + 0 * n + k) / 32);
            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 0 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 0 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 0 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a00 = AdvSimd.FusedMultiplyAdd(a00, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
            }

            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 1 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 1 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 1 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a10 = AdvSimd.FusedMultiplyAdd(a10, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
            }

            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 2 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 2 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 2 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a20 = AdvSimd.FusedMultiplyAdd(a20, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
            }

            {
                var wl = Vector128.LoadUnsafe(ref w, (nuint)(wo + 3 * n + k));
                var wh = Vector128.LoadUnsafe(ref w, (nuint)(wo + 3 * n + k + 16));
                float scale = Unsafe.Add(ref sw, (wo + 3 * n + k) / 32);
                var d0 = Dp.DotProduct(Vector128<int>.Zero, wl, x0l);
                d0 = Dp.DotProduct(d0, wh, x0h);
                a30 = AdvSimd.FusedMultiplyAdd(a30, Vector128.ConvertToSingle(d0), Vector128.Create(scale * s0));
            }
        }

        output[yo + 0] = Vector128.Sum(a00);
        output[yo + 1] = Vector128.Sum(a10);
        output[yo + 2] = Vector128.Sum(a20);
        output[yo + 3] = Vector128.Sum(a30);
    }
}
