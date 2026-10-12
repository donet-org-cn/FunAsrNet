using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>保持远离零舍入语义的 x64 分组量化，避免逐元素调用标量舍入。</summary>
public sealed partial class TensorMath
{
    /// <summary>每次量化三十二项；非有限值与倒数溢出的块回退到原标量语义。</summary>
    /// <param name="input">经过完整分组长度验证的只读输入。</param>
    /// <param name="q">与输入等长的量化结果。</param>
    /// <param name="scales">每三十二项一个缩放值。</param>
    private static void QuantizeAvx(ReadOnlySpan<float> input, Span<sbyte> q, Span<float> scales)
    {
        ref float source = ref MemoryMarshal.GetReference(input);
        ref sbyte destination = ref MemoryMarshal.GetReference(q);
        var mask = Vector256.Create(int.MaxValue);
        for (int block = 0; block < scales.Length; block++)
        {
            nuint at = (nuint)(block * 32);
            var a = Vector256.LoadUnsafe(ref source, at);
            var b = Vector256.LoadUnsafe(ref source, at + 8);
            var c = Vector256.LoadUnsafe(ref source, at + 16);
            var d = Vector256.LoadUnsafe(ref source, at + 24);
            // 非负浮点位模式的整数顺序等于数值顺序，且任何 NaN 都排在无穷之后。
            var maximum = Vector256.Max(Vector256.Max(a.AsInt32() & mask, b.AsInt32() & mask), Vector256.Max(c.AsInt32() & mask, d.AsInt32() & mask));
            var narrow = Vector128.Max(maximum.GetLower(), maximum.GetUpper());
            narrow = Vector128.Max(narrow, Sse2.Shuffle(narrow, 0x4e));
            narrow = Vector128.Max(narrow, Sse2.Shuffle(narrow, 0xb1));
            float scale = narrow.AsSingle().ToScalar() / 127f;
            float inverse = scale == 0 ? 0 : 1f / scale;
            if (!float.IsFinite(scale) || !float.IsFinite(inverse))
            {
                float scalarMax = 0;
                for (int index = 0; index < 32; index++)
                {
                    scalarMax = Math.Max(scalarMax, Math.Abs(input[block * 32 + index]));
                }
                scale = scalarMax / 127f;
                inverse = scale == 0 ? 0 : 1f / scale;
                for (int index = 0; index < 32; index++)
                {
                    q[block * 32 + index] = (sbyte)Math.Clamp((int)MathF.Round(input[block * 32 + index] * inverse, MidpointRounding.AwayFromZero), -127, 127);
                }
            }
            else
            {
                var factor = Vector256.Create(inverse);
                var qa = RoundQuantizedAvx(a * factor);
                var qb = RoundQuantizedAvx(b * factor);
                var qc = RoundQuantizedAvx(c * factor);
                var qd = RoundQuantizedAvx(d * factor);
                Vector256.Narrow(Vector256.Narrow(qa, qb), Vector256.Narrow(qc, qd)).StoreUnsafe(ref destination, at);
            }
            scales[block] = scale;
        }
    }

    /// <summary>用截断后的小数部分判断半整数，避免加半再截断把半整数前一浮点数误进位。</summary>
    /// <param name="value">有限的缩放激活，位于量化范围附近。</param>
    /// <returns>与标量远离零舍入及截断范围相同的八项整数。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> RoundQuantizedAvx(Vector256<float> value)
    {
        var integral = Avx.ConvertToVector256Int32WithTruncation(value);
        var fraction = Vector256.Abs(value - Avx.ConvertToVector256Single(integral));
        var increment = Avx.CompareGreaterThanOrEqual(fraction, Vector256.Create(0.5f)).AsInt32();
        var sign = Avx2.ShiftRightArithmetic(value.AsInt32(), 31) | Vector256.Create(1);
        return Vector256.Clamp(integral + (increment & sign), Vector256.Create(-127), Vector256.Create(127));
    }
}
