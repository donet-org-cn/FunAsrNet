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
    /// 使用 NEON 对激活执行分组 Q8 量化及缩放计算。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="q">查询张量或分组量化后的激活缓冲区。</param>
    /// <param name="scales">每 32 个元素一组的量化缩放因子。</param>
    private static void QuantizeNeon(ReadOnlySpan<float> input, Span<sbyte> q, Span<float> scales)
    {
        ref float x = ref MemoryMarshal.GetReference(input);
        ref sbyte y = ref MemoryMarshal.GetReference(q);
        for (int b = 0; b < scales.Length; ++b)
        {
            nuint at = (nuint)(b * 32);
            var x0 = Vector128.LoadUnsafe(ref x, at + 0);
            var x1 = Vector128.LoadUnsafe(ref x, at + 4);
            var x2 = Vector128.LoadUnsafe(ref x, at + 8);
            var x3 = Vector128.LoadUnsafe(ref x, at + 12);
            var x4 = Vector128.LoadUnsafe(ref x, at + 16);
            var x5 = Vector128.LoadUnsafe(ref x, at + 20);
            var x6 = Vector128.LoadUnsafe(ref x, at + 24);
            var x7 = Vector128.LoadUnsafe(ref x, at + 28);
            var max = Vector128.Max(Vector128.Abs(x0), Vector128.Abs(x1));
            max = Vector128.Max(max, Vector128.Abs(x2));
            max = Vector128.Max(max, Vector128.Abs(x3));
            max = Vector128.Max(max, Vector128.Abs(x4));
            max = Vector128.Max(max, Vector128.Abs(x5));
            max = Vector128.Max(max, Vector128.Abs(x6));
            max = Vector128.Max(max, Vector128.Abs(x7));
            float d = AdvSimd.Arm64.MaxAcross(max).ToScalar() / 127f;
            scales[b] = d;
            var inv = Vector128.Create(d == 0 ? 0 : 1f / d);
            var q0 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x0 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q1 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x1 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q2 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x2 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q3 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x3 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q4 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x4 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q5 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x5 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q6 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x6 * inv), Vector128.Create(-127), Vector128.Create(127));
            var q7 = Vector128.Clamp(AdvSimd.ConvertToInt32RoundAwayFromZero(x7 * inv), Vector128.Create(-127), Vector128.Create(127));
            Vector128.Narrow(Vector128.Narrow(q0, q1), Vector128.Narrow(q2, q3)).StoreUnsafe(ref y, at + 0);
            Vector128.Narrow(Vector128.Narrow(q4, q5), Vector128.Narrow(q6, q7)).StoreUnsafe(ref y, at + 16);
        }
    }
}
