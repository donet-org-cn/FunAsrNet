using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
public sealed partial class TensorMath
{
    // 遍历全部源帧时将加权值累加器保持在寄存器内。
    // 每个输出元素的乘加顺序与 AddScaled 完全一致。
    /// <summary>
    /// 使用注意力概率对值向量加权累加，支持 SIMD 与标量路径。
    /// </summary>
    /// <param name="scores">注意力分数或 VAD 静音概率的存储区。</param>
    /// <param name="values">按顺序排列的浮点数据。</param>
    /// <param name="start">起始采样位置或输入偏移。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    internal static void AttentionValues(ReadOnlySpan<float> scores, float[] values, int start, int stride, Span<float> output, bool simd)
    {
        if (scores.Length == 0)
        {
            output.Clear();
            return;
        }

        int channel = 0;
        ref float data = ref MemoryMarshal.GetArrayDataReference(values);
        ref float dest = ref MemoryMarshal.GetReference(output);
        if (simd && Vector128.IsHardwareAccelerated)
        {
            for (; channel <= output.Length - 16; channel += 16)
            {
                var a0 = Vector128<float>.Zero;
                var a1 = a0;
                var a2 = a0;
                var a3 = a0;
                ref float source = ref Unsafe.Add(ref data, start + channel);
                for (int t = 0; t < scores.Length; ++t)
                {
                    var scale = Vector128.Create(scores[t]);
                    a0 += Vector128.LoadUnsafe(ref source) * scale;
                    a1 += Vector128.LoadUnsafe(ref source, 4) * scale;
                    a2 += Vector128.LoadUnsafe(ref source, 8) * scale;
                    a3 += Vector128.LoadUnsafe(ref source, 12) * scale;
                    source = ref Unsafe.Add(ref source, stride);
                }

                a0.StoreUnsafe(ref dest, (nuint)channel);
                a1.StoreUnsafe(ref dest, (nuint)(channel + 4));
                a2.StoreUnsafe(ref dest, (nuint)(channel + 8));
                a3.StoreUnsafe(ref dest, (nuint)(channel + 12));
            }
        }

        for (; channel < output.Length; ++channel)
        {
            float sum = 0;
            for (int t = 0; t < scores.Length; ++t)
            {
                sum += values[start + t * stride + channel] * scores[t];
            }

            output[channel] = sum;
        }
    }
}
