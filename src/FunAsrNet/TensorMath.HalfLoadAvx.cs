using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>在读取 F16 权重时向量化展开为 FP32，不改变原始模型权重的精度语义。</summary>
public sealed partial class TensorMath
{
    /// <summary>把连续 binary16 存储精确展开为 binary32，保留负零、符号、NaN 载荷和静默化语义。</summary>
    /// <param name="input">只读的 F16 源数据，调用期间不能与输出存储重叠。</param>
    /// <param name="output">至少与源数据等长的 FP32 目标，超出源长度的尾部保持不变。</param>
    internal static void ConvertHalfToSingle(ReadOnlySpan<Half> input, Span<float> output)
    {
        if (output.Length < input.Length)
        {
            throw new ArgumentException("Half conversion output too short");
        }
        int index = 0;
        if (Avx2.IsSupported && Environment.GetEnvironmentVariable("FUNASRNET_SCALAR_HALF_LOAD") != "1")
        {
            ref ushort source = ref MemoryMarshal.GetReference(MemoryMarshal.Cast<Half, ushort>(input));
            ref float destination = ref MemoryMarshal.GetReference(output);
            for (; index <= input.Length - 16; index += 16)
            {
                var raw = Vector256.LoadUnsafe(ref source, (nuint)index);
                ConvertHalfBitsAvx(Vector256.WidenLower(raw).AsInt32()).StoreUnsafe(ref destination, (nuint)index);
                ConvertHalfBitsAvx(Vector256.WidenUpper(raw).AsInt32()).StoreUnsafe(ref destination, (nuint)(index + 8));
            }
        }
        for (; index < input.Length; index++)
        {
            output[index] = (float)input[index];
        }
    }

    /// <summary>转换八个已零扩展的 F16 位模式，非正规数通过精确二进制缩放得到正规 Single 值。</summary>
    /// <param name="raw">每通道存放一个十六位 F16 位模式，其余高位为零。</param>
    /// <returns>与逐元素 .NET Half 转换一致的 FP32 位模式。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ConvertHalfBitsAvx(Vector256<int> raw)
    {
        var absolute = raw & Vector256.Create(0x7fff);
        var exponent = raw & Vector256.Create(0x7c00);
        var mantissa = raw & Vector256.Create(0x03ff);
        var normal = (absolute << 13) + Vector256.Create(0x38000000);
        // Half 非正规值为 mantissa * 2^-24，Single 可以精确表示所有这些结果。
        var subnormal = Vector256.ConvertToSingle(mantissa) * Vector256.Create(5.9604644775390625E-8f);
        var bits = Vector256.ConditionalSelect(Vector256.Equals(exponent, Vector256<int>.Zero), subnormal.AsInt32(), normal);
        var quiet = Vector256.GreaterThan(absolute, Vector256.Create(0x7c00)) & Vector256.Create(0x00400000);
        var special = Vector256.Create(0x7f800000) | (mantissa << 13) | quiet;
        bits = Vector256.ConditionalSelect(Vector256.Equals(exponent, Vector256.Create(0x7c00)), special, bits);
        var sign = (raw & Vector256.Create(0x8000)) << 16;
        return (bits | sign).AsSingle();
    }
}
