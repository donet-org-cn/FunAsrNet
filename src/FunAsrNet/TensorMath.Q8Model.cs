using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace FunAsrNet;

/// <summary>读取 Q8_0 的原始块，不重新量化或改变 F16 缩放值。</summary>
public sealed partial class TensorMath
{
    /// <summary>受检后按固定三十四字节块解包，消除每三十二项的通用 Span 搬运分派。</summary>
    /// <param name="input">连续的小端 F16 缩放与三十二项有符号字节块。</param>
    /// <param name="quantized">恰好容纳完整块的字节目标。</param>
    /// <param name="scales">恰好一个缩放值对应一个块。</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void UnpackQ8Model(ReadOnlySpan<byte> input, Span<sbyte> quantized, Span<float> scales)
    {
        if (input.Length % 34 != 0 || scales.Length != input.Length / 34 || quantized.Length != checked(scales.Length * 32))
        {
            throw new ArgumentException("Q8 model block dimensions differ");
        }
        ref byte source = ref MemoryMarshal.GetReference(input);
        ref byte target = ref Unsafe.As<sbyte, byte>(ref MemoryMarshal.GetReference(quantized));
        ref float scale = ref MemoryMarshal.GetReference(scales);
        for (int block = 0; block < scales.Length; block++)
        {
            ref byte current = ref Unsafe.Add(ref source, block * 34);
            ushort bits = Unsafe.ReadUnaligned<ushort>(ref current);
            if (!BitConverter.IsLittleEndian)
            {
                bits = BinaryPrimitives.ReverseEndianness(bits);
            }
            uint exponent = (uint)bits & 0x7c00;
            // 模型绝大多数缩放是正规数，直接扩展指数与尾数；特殊值保留 .NET 的完整语义。
            Unsafe.Add(ref scale, block) = exponent is > 0 and < 0x7c00
                ? BitConverter.UInt32BitsToSingle((((uint)bits & 0x7fff) << 13) + 0x38000000 | (((uint)bits & 0x8000) << 16))
                : (float)BitConverter.UInt16BitsToHalf(bits);
            Unsafe.CopyBlockUnaligned(ref Unsafe.Add(ref target, block * 32), ref Unsafe.Add(ref current, 2), 32);
        }
    }
}
