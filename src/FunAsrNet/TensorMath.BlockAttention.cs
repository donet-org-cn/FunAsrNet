using System.Runtime.CompilerServices;
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
    /// 按帧块计算注意力，减少中间读取并保留原有数值顺序。
    /// </summary>
    /// <param name="q">查询张量或分组量化后的激活缓冲区。</param>
    /// <param name="k">按帧和注意力头排列的键张量。</param>
    /// <param name="v">按帧排列的值张量或待归约向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="heads">注意力头数量。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] BlockAttention(float[] q, float[] k, float[] v, int frames, int width, int heads)
    {
        int size = width / heads;
        float divisor = MathF.Sqrt(size);
        var output = TensorWorkspace.Allocate(q.Length);
        var keys = TensorWorkspace.Allocate(k.Length);
        var values = TensorWorkspace.Allocate(v.Length);
        for (int h = 0; h < heads; ++h)
        {
            for (int t = 0; t < frames; ++t)
            {
                k.AsSpan(t * width + h * size, size).CopyTo(keys.AsSpan((h * frames + t) * size));
                v.AsSpan(t * width + h * size, size).CopyTo(values.AsSpan((h * frames + t) * size));
            }
        }

        For(
            heads * ((frames + 1) / 2),
            index =>
            {
                int groups = (frames + 1) / 2;
                int h = index / groups, t = index % groups * 2, at = t * width + h * size;
                Span<float> scores = stackalloc float[frames * 2];
                int source = 0;
                if (t + 1 < frames)
                {
                    for (; source + 3 < frames; source += 4)
                    {
                        AttentionScores2x4(q, at, keys, (h * frames + source) * size, width, size, size, scores.Slice(source), frames, divisor);
                    }

                    for (; source < frames; ++source)
                    {
                        scores[source] = AudioKernels.Dot(q.AsSpan(at, size), keys.AsSpan((h * frames + source) * size, size)) / divisor;
                        scores[frames + source] = AudioKernels.Dot(q.AsSpan(at + width, size), keys.AsSpan((h * frames + source) * size, size)) / divisor;
                    }

                    Softmax(scores.Slice(0, frames));
                    Softmax(scores.Slice(frames, frames));
                    AttentionValues2(scores, values, h * frames * size, size, output, at, width, size, frames);
                }
                else
                {
                    for (; source < frames; ++source)
                    {
                        scores[source] = AudioKernels.Dot(q.AsSpan(at, size), keys.AsSpan((h * frames + source) * size, size)) / divisor;
                    }

                    Softmax(scores.Slice(0, frames));
                    AttentionValues(scores.Slice(0, frames), values, h * frames * size, size, output.AsSpan(at, size), true);
                }
            });
        TensorWorkspace.Release(keys);
        TensorWorkspace.Release(values);
        return output;
    }

    // 同时计算八个点积，每个点积保留原来的双向量累加顺序。
    /// <summary>
    /// 并行计算两条查询与四条键的注意力分数。
    /// </summary>
    /// <param name="query">按帧排列的查询矩阵。</param>
    /// <param name="qo">查询数组的起始元素偏移。</param>
    /// <param name="key">按帧排列的键矩阵。</param>
    /// <param name="ko">键数组的起始元素偏移。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="keyStride">键矩阵的行步长，单位为元素。</param>
    /// <param name="size">参与当前块计算的元素数量。</param>
    /// <param name="scores">注意力分数或 VAD 静音概率的存储区。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="divisor">注意力分数的缩放除数。</param>
    private static void AttentionScores2x4(
        float[] query,
        int qo,
        float[] key,
        int ko,
        int stride,
        int keyStride,
        int size,
        Span<float> scores,
        int frames,
        float divisor)
    {
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            AttentionScores2x4Avx(query, qo, key, ko, stride, keyStride, size, scores, frames, divisor);
            return;
        }
        ref float q = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(query), qo);
        ref float k = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(key), ko);
        var a00 = Vector128<float>.Zero;
        var b00 = a00;
        var a01 = Vector128<float>.Zero;
        var b01 = a01;
        var a02 = Vector128<float>.Zero;
        var b02 = a02;
        var a03 = Vector128<float>.Zero;
        var b03 = a03;
        var a10 = Vector128<float>.Zero;
        var b10 = a10;
        var a11 = Vector128<float>.Zero;
        var b11 = a11;
        var a12 = Vector128<float>.Zero;
        var b12 = a12;
        var a13 = Vector128<float>.Zero;
        var b13 = a13;
        int channel = 0;
        for (; channel <= size - 8; channel += 8)
        {
            var q0a = Vector128.LoadUnsafe(ref q, (nuint)(channel + 0));
            var q1a = Vector128.LoadUnsafe(ref q, (nuint)(stride + channel + 0));
            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(0 * keyStride + channel + 0));
                a00 = AdvSimd.FusedMultiplyAdd(a00, q0a, value);
                a10 = AdvSimd.FusedMultiplyAdd(a10, q1a, value);
            }

            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(1 * keyStride + channel + 0));
                a01 = AdvSimd.FusedMultiplyAdd(a01, q0a, value);
                a11 = AdvSimd.FusedMultiplyAdd(a11, q1a, value);
            }

            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(2 * keyStride + channel + 0));
                a02 = AdvSimd.FusedMultiplyAdd(a02, q0a, value);
                a12 = AdvSimd.FusedMultiplyAdd(a12, q1a, value);
            }

            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(3 * keyStride + channel + 0));
                a03 = AdvSimd.FusedMultiplyAdd(a03, q0a, value);
                a13 = AdvSimd.FusedMultiplyAdd(a13, q1a, value);
            }

            var q0b = Vector128.LoadUnsafe(ref q, (nuint)(channel + 4));
            var q1b = Vector128.LoadUnsafe(ref q, (nuint)(stride + channel + 4));
            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(0 * keyStride + channel + 4));
                b00 = AdvSimd.FusedMultiplyAdd(b00, q0b, value);
                b10 = AdvSimd.FusedMultiplyAdd(b10, q1b, value);
            }

            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(1 * keyStride + channel + 4));
                b01 = AdvSimd.FusedMultiplyAdd(b01, q0b, value);
                b11 = AdvSimd.FusedMultiplyAdd(b11, q1b, value);
            }

            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(2 * keyStride + channel + 4));
                b02 = AdvSimd.FusedMultiplyAdd(b02, q0b, value);
                b12 = AdvSimd.FusedMultiplyAdd(b12, q1b, value);
            }

            {
                var value = Vector128.LoadUnsafe(ref k, (nuint)(3 * keyStride + channel + 4));
                b03 = AdvSimd.FusedMultiplyAdd(b03, q0b, value);
                b13 = AdvSimd.FusedMultiplyAdd(b13, q1b, value);
            }
        }

        float s00 = Vector128.Sum(a00 + b00);
        for (int c = channel; c < size; ++c)
        {
            s00 += query[qo + 0 * stride + c] * key[ko + 0 * keyStride + c];
        }

        scores[0 * frames + 0] = s00 / divisor;
        float s01 = Vector128.Sum(a01 + b01);
        for (int c = channel; c < size; ++c)
        {
            s01 += query[qo + 0 * stride + c] * key[ko + 1 * keyStride + c];
        }

        scores[0 * frames + 1] = s01 / divisor;
        float s02 = Vector128.Sum(a02 + b02);
        for (int c = channel; c < size; ++c)
        {
            s02 += query[qo + 0 * stride + c] * key[ko + 2 * keyStride + c];
        }

        scores[0 * frames + 2] = s02 / divisor;
        float s03 = Vector128.Sum(a03 + b03);
        for (int c = channel; c < size; ++c)
        {
            s03 += query[qo + 0 * stride + c] * key[ko + 3 * keyStride + c];
        }

        scores[0 * frames + 3] = s03 / divisor;
        float s10 = Vector128.Sum(a10 + b10);
        for (int c = channel; c < size; ++c)
        {
            s10 += query[qo + 1 * stride + c] * key[ko + 0 * keyStride + c];
        }

        scores[1 * frames + 0] = s10 / divisor;
        float s11 = Vector128.Sum(a11 + b11);
        for (int c = channel; c < size; ++c)
        {
            s11 += query[qo + 1 * stride + c] * key[ko + 1 * keyStride + c];
        }

        scores[1 * frames + 1] = s11 / divisor;
        float s12 = Vector128.Sum(a12 + b12);
        for (int c = channel; c < size; ++c)
        {
            s12 += query[qo + 1 * stride + c] * key[ko + 2 * keyStride + c];
        }

        scores[1 * frames + 2] = s12 / divisor;
        float s13 = Vector128.Sum(a13 + b13);
        for (int c = channel; c < size; ++c)
        {
            s13 += query[qo + 1 * stride + c] * key[ko + 3 * keyStride + c];
        }

        scores[1 * frames + 3] = s13 / divisor;
    }

    /// <summary>
    /// 同时计算两条查询对应的值向量加权和。
    /// </summary>
    /// <param name="scores">注意力分数或 VAD 静音概率的存储区。</param>
    /// <param name="values">按顺序排列的浮点数据。</param>
    /// <param name="start">起始采样位置或输入偏移。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="dest">输出数组的起始元素偏移。</param>
    /// <param name="outputStride">输出的行步长，单位为元素。</param>
    /// <param name="size">参与当前块计算的元素数量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    private static void AttentionValues2(
        ReadOnlySpan<float> scores,
        float[] values,
        int start,
        int stride,
        float[] output,
        int dest,
        int outputStride,
        int size,
        int frames)
    {
        int channel = 0;
        ref float data = ref MemoryMarshal.GetArrayDataReference(values);
        if (Vector256.IsHardwareAccelerated)
        {
            channel = AttentionValues2Avx(scores, values, start, stride, output, dest, outputStride, size, frames);
        }
        for (; channel <= size - 16; channel += 16)
        {
            var a00 = Vector128<float>.Zero;
            var a01 = Vector128<float>.Zero;
            var a02 = Vector128<float>.Zero;
            var a03 = Vector128<float>.Zero;
            var a10 = Vector128<float>.Zero;
            var a11 = Vector128<float>.Zero;
            var a12 = Vector128<float>.Zero;
            var a13 = Vector128<float>.Zero;
            ref float source = ref Unsafe.Add(ref data, start + channel);
            for (int t = 0; t < frames; ++t)
            {
                var s0 = Vector128.Create(scores[t]);
                var s1 = Vector128.Create(scores[frames + t]);
                {
                    var value = Vector128.LoadUnsafe(ref source, 0);
                    a00 += value * s0;
                    a10 += value * s1;
                }

                {
                    var value = Vector128.LoadUnsafe(ref source, 4);
                    a01 += value * s0;
                    a11 += value * s1;
                }

                {
                    var value = Vector128.LoadUnsafe(ref source, 8);
                    a02 += value * s0;
                    a12 += value * s1;
                }

                {
                    var value = Vector128.LoadUnsafe(ref source, 12);
                    a03 += value * s0;
                    a13 += value * s1;
                }

                source = ref Unsafe.Add(ref source, stride);
            }

            a00.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 0]);
            a01.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 4]);
            a02.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 8]);
            a03.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 12]);
            a10.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 0]);
            a11.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 4]);
            a12.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 8]);
            a13.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 12]);
        }

        for (; channel < size; ++channel)
        {
            float a = 0, b = 0;
            for (int t = 0; t < frames; ++t)
            {
                float value = values[start + t * stride + channel];
                a += value * scores[t];
                b += value * scores[frames + t];
            }

            output[dest + channel] = a;
            output[dest + outputStride + channel] = b;
        }
    }
}
