using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>行主序矩阵与向量算子。</summary>
public sealed partial class TensorMath
{
    /// <summary>原位施加半分割 RoPE，保持独立乘法、加减和标量尾部的计算顺序。</summary>
    /// <param name="x">单个注意力头，长度必须为旋转因子长度的两倍，结果原位写回。</param>
    /// <param name="cosine">当前序列位置的余弦系数，不得与输出重叠。</param>
    /// <param name="sine">当前序列位置的正弦系数，长度与余弦相同，不得与输出重叠。</param>
    /// <param name="simd">是否启用本机支持的 256 位或 128 位 SIMD。</param>
    internal static void RotateHalf(Span<float> x, ReadOnlySpan<float> cosine, ReadOnlySpan<float> sine, bool simd)
    {
        int half = cosine.Length;
        if (x.Length != 2 * half || sine.Length != half)
        {
            throw new ArgumentException("Invalid RoPE dimensions");
        }
        ref float data = ref MemoryMarshal.GetReference(x);
        ref float cos = ref MemoryMarshal.GetReference(cosine);
        ref float sin = ref MemoryMarshal.GetReference(sine);
        int i = 0;
        if (simd && Vector256.IsHardwareAccelerated)
        {
            for (; i <= half - 8; i += 8)
            {
                var a = Vector256.LoadUnsafe(ref data, (nuint)i);
                var b = Vector256.LoadUnsafe(ref data, (nuint)(i + half));
                var c = Vector256.LoadUnsafe(ref cos, (nuint)i);
                var s = Vector256.LoadUnsafe(ref sin, (nuint)i);
                (a * c - b * s).StoreUnsafe(ref data, (nuint)i);
                (a * s + b * c).StoreUnsafe(ref data, (nuint)(i + half));
            }
        }
        if (simd && Vector128.IsHardwareAccelerated)
        {
            for (; i <= half - 4; i += 4)
            {
                var a = Vector128.LoadUnsafe(ref data, (nuint)i);
                var b = Vector128.LoadUnsafe(ref data, (nuint)(i + half));
                var c = Vector128.LoadUnsafe(ref cos, (nuint)i);
                var s = Vector128.LoadUnsafe(ref sin, (nuint)i);
                (a * c - b * s).StoreUnsafe(ref data, (nuint)i);
                (a * s + b * c).StoreUnsafe(ref data, (nuint)(i + half));
            }
        }
        for (; i < half; ++i)
        {
            float a = x[i], b = x[i + half], c = cosine[i], s = sine[i];
            x[i] = a * c - b * s;
            x[i + half] = a * s + b * c;
        }
    }
}
