using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>在 x64 上复用键和值，保持原双向量点积与逐帧累加顺序。</summary>
public sealed partial class TensorMath
{
    /// <summary>同时计算两条查询和四条键的点积，不改变每八项的累加分组。</summary>
    /// <param name="query">帧主序查询矩阵。</param>
    /// <param name="qo">第一条查询的起始偏移。</param>
    /// <param name="key">按注意力头连续打包的键。</param>
    /// <param name="ko">第一条键的起始偏移。</param>
    /// <param name="stride">查询行间隔。</param>
    /// <param name="keyStride">键行间隔。</param>
    /// <param name="size">点积宽度。</param>
    /// <param name="scores">两条查询的分数存储区。</param>
    /// <param name="frames">每条查询的分数数量。</param>
    /// <param name="divisor">模型指定的缩放除数。</param>
    private static void AttentionScores2x4Avx(float[] query, int qo, float[] key, int ko, int stride, int keyStride, int size, Span<float> scores, int frames, float divisor)
    {
        ref float q = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(query), qo);
        ref float k = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(key), ko);
        var a00 = Vector256<float>.Zero;
        var a01 = Vector256<float>.Zero;
        var a02 = Vector256<float>.Zero;
        var a03 = Vector256<float>.Zero;
        var a10 = Vector256<float>.Zero;
        var a11 = Vector256<float>.Zero;
        var a12 = Vector256<float>.Zero;
        var a13 = Vector256<float>.Zero;
        int channel = 0;
        for (; channel <= size - 8; channel += 8)
        {
            var q0 = Vector256.LoadUnsafe(ref q, (nuint)channel);
            var q1 = Vector256.LoadUnsafe(ref q, (nuint)(stride + channel));
            {
                var value = Vector256.LoadUnsafe(ref k, (nuint)(0 * keyStride + channel));
                a00 += q0 * value;
                a10 += q1 * value;
            }
            {
                var value = Vector256.LoadUnsafe(ref k, (nuint)(1 * keyStride + channel));
                a01 += q0 * value;
                a11 += q1 * value;
            }
            {
                var value = Vector256.LoadUnsafe(ref k, (nuint)(2 * keyStride + channel));
                a02 += q0 * value;
                a12 += q1 * value;
            }
            {
                var value = Vector256.LoadUnsafe(ref k, (nuint)(3 * keyStride + channel));
                a03 += q0 * value;
                a13 += q1 * value;
            }
        }
        float s00 = Vector128.Sum(a00.GetLower() + a00.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s00 += query[qo + 0 * stride + c] * key[ko + 0 * keyStride + c];
        }
        scores[0 * frames + 0] = s00 / divisor;
        float s01 = Vector128.Sum(a01.GetLower() + a01.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s01 += query[qo + 0 * stride + c] * key[ko + 1 * keyStride + c];
        }
        scores[0 * frames + 1] = s01 / divisor;
        float s02 = Vector128.Sum(a02.GetLower() + a02.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s02 += query[qo + 0 * stride + c] * key[ko + 2 * keyStride + c];
        }
        scores[0 * frames + 2] = s02 / divisor;
        float s03 = Vector128.Sum(a03.GetLower() + a03.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s03 += query[qo + 0 * stride + c] * key[ko + 3 * keyStride + c];
        }
        scores[0 * frames + 3] = s03 / divisor;
        float s10 = Vector128.Sum(a10.GetLower() + a10.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s10 += query[qo + 1 * stride + c] * key[ko + 0 * keyStride + c];
        }
        scores[1 * frames + 0] = s10 / divisor;
        float s11 = Vector128.Sum(a11.GetLower() + a11.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s11 += query[qo + 1 * stride + c] * key[ko + 1 * keyStride + c];
        }
        scores[1 * frames + 1] = s11 / divisor;
        float s12 = Vector128.Sum(a12.GetLower() + a12.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s12 += query[qo + 1 * stride + c] * key[ko + 2 * keyStride + c];
        }
        scores[1 * frames + 2] = s12 / divisor;
        float s13 = Vector128.Sum(a13.GetLower() + a13.GetUpper());
        for (int c = channel; c < size; c++)
        {
            s13 += query[qo + 1 * stride + c] * key[ko + 3 * keyStride + c];
        }
        scores[1 * frames + 3] = s13 / divisor;
    }

    /// <summary>共享值加载并计算两条查询的加权和，保持各通道独立的原累加顺序。</summary>
    /// <param name="scores">两条查询的归一化概率。</param>
    /// <param name="values">按头连续的值矩阵。</param>
    /// <param name="start">当前头的首元素偏移。</param>
    /// <param name="stride">值行间隔。</param>
    /// <param name="output">完整结果数组。</param>
    /// <param name="dest">第一条查询的输出起点。</param>
    /// <param name="outputStride">输出行间隔。</param>
    /// <param name="size">当前头的通道数。</param>
    /// <param name="frames">值的时间帧数。</param>
    /// <returns>已写入的通道数，尾部交给通用内核。</returns>
    private static int AttentionValues2Avx(ReadOnlySpan<float> scores, float[] values, int start, int stride, float[] output, int dest, int outputStride, int size, int frames)
    {
        int channel = 0;
        ref float data = ref MemoryMarshal.GetArrayDataReference(values);
        for (; channel <= size - 32; channel += 32)
        {
            var a00 = Vector256<float>.Zero;
            var a01 = Vector256<float>.Zero;
            var a02 = Vector256<float>.Zero;
            var a03 = Vector256<float>.Zero;
            var a10 = Vector256<float>.Zero;
            var a11 = Vector256<float>.Zero;
            var a12 = Vector256<float>.Zero;
            var a13 = Vector256<float>.Zero;
            ref float source = ref Unsafe.Add(ref data, start + channel);
            for (int t = 0; t < frames; t++)
            {
                var s0 = Vector256.Create(scores[t]);
                var s1 = Vector256.Create(scores[frames + t]);
                {
                    var value = Vector256.LoadUnsafe(ref source, 0);
                    a00 += value * s0;
                    a10 += value * s1;
                }
                {
                    var value = Vector256.LoadUnsafe(ref source, 8);
                    a01 += value * s0;
                    a11 += value * s1;
                }
                {
                    var value = Vector256.LoadUnsafe(ref source, 16);
                    a02 += value * s0;
                    a12 += value * s1;
                }
                {
                    var value = Vector256.LoadUnsafe(ref source, 24);
                    a03 += value * s0;
                    a13 += value * s1;
                }
                source = ref Unsafe.Add(ref source, stride);
            }
            a00.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 0]);
            a01.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 8]);
            a02.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 16]);
            a03.StoreUnsafe(ref output[dest + 0 * outputStride + channel + 24]);
            a10.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 0]);
            a11.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 8]);
            a12.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 16]);
            a13.StoreUnsafe(ref output[dest + 1 * outputStride + channel + 24]);
        }
        return channel;
    }
}
