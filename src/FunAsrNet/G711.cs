using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// 电话音频编码类型。
/// </summary>
public enum AudioCodec
{
    /// <summary>
    /// 小端有符号 16 位线性 PCM。
    /// </summary>
    Pcm16,
    /// <summary>
    /// G.711 A-law 压扩编码。
    /// </summary>
    Pcma,
    /// <summary>
    /// G.711 μ-law 压扩编码。
    /// </summary>
    Pcmu
}

/// <summary>
/// G.711 A-law 与 μ-law 编解码工具，提供 NEON 与标量路径。
/// </summary>
public static class G711
{
    private static readonly byte[] AlawEncode = BuildEncoder(true);
    private static readonly byte[] UlawEncode = BuildEncoder(false);
    /// <summary>
    /// 将单个 G.711 压扩字节转换为 PCM16 采样。
    /// </summary>
    /// <param name="sample">单个压扩音频字节。</param>
    /// <param name="alaw">为 true 时使用 A-law，否则使用 μ-law。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public static short DecodeSample(byte sample, bool alaw)
    {
        if (alaw)
        {
            int a = sample ^ 0x55, exponent = (a >> 4) & 7;
            int magnitude = (a & 15) << 4;
            magnitude = exponent == 0 ? magnitude + 8 : (magnitude + 264) << (exponent - 1);
            return (short)((a & 128) != 0 ? magnitude : -magnitude);
        }

        int u = sample ^ 255;
        int t = (((u & 15) << 3) + 132) << ((u >> 4) & 7);
        return (short)((u & 128) != 0 ? 132 - t : t - 132);
    }

    // 每次 NEON 迭代处理 16 个采样，避免逐采样分支和离散查表。
    /// <summary>
    /// 将 G.711 字节序列解码到调用方 PCM16 缓冲区。
    /// </summary>
    /// <param name="source">在调用期间保持有效的只读输入视图。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    /// <param name="alaw">为 true 时使用 A-law，否则使用 μ-law。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    public static void Decode(ReadOnlySpan<byte> source, Span<short> destination, bool alaw, bool simd = true)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination too small");
        }

        int i = 0;
        if (simd && AdvSimd.IsSupported)
        {
            ref byte input = ref MemoryMarshal.GetReference(source);
            ref short output = ref MemoryMarshal.GetReference(destination);
            for (; i <= source.Length - 16; i += 16)
            {
                var bytes = Vector128.LoadUnsafe(ref input, (nuint)i) ^ Vector128.Create((byte)(alaw ? 85 : 255));
                var (lo, hi) = Vector128.Widen(bytes);
                DecodeNeon(lo, alaw).StoreUnsafe(ref output, (nuint)i);
                DecodeNeon(hi, alaw).StoreUnsafe(ref output, (nuint)(i + 8));
            }
        }

        for (; i < source.Length; ++i)
        {
            destination[i] = DecodeSample(source[i], alaw);
        }
    }

    /// <summary>
    /// 使用 NEON 同时解码一组 G.711 压扩字节。
    /// </summary>
    /// <param name="value">待编码、填充或处理的值。</param>
    /// <param name="alaw">为 true 时使用 A-law，否则使用 μ-law。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static Vector128<short> DecodeNeon(Vector128<ushort> value, bool alaw)
    {
        var exponent = (value >> 4) & Vector128.Create((ushort)7);
        var mantissa = value & Vector128.Create((ushort)15);
        Vector128<short> magnitude;
        if (alaw)
        {
            var low = (mantissa << 4) + Vector128.Create((ushort)8);
            var high = AdvSimd.ShiftLogical((mantissa << 4) + Vector128.Create((ushort)264), (exponent - Vector128.Create((ushort)1)).AsInt16());
            magnitude = Vector128.ConditionalSelect(Vector128.Equals(exponent, Vector128<ushort>.Zero), low, high).AsInt16();
        }
        else
        {
            magnitude = (AdvSimd.ShiftLogical((mantissa << 3) + Vector128.Create((ushort)132), exponent.AsInt16()) - Vector128.Create((ushort)132)).AsInt16();
        }

        var sign = Vector128.Equals(value & Vector128.Create((ushort)128), Vector128<ushort>.Zero).AsInt16();
        return alaw ? Vector128.ConditionalSelect(sign, -magnitude, magnitude) : Vector128.ConditionalSelect(sign, magnitude, -magnitude);
    }

    /// <summary>
    /// 将 PCM16 采样压扩编码到调用方字节缓冲区。
    /// </summary>
    /// <param name="source">在调用期间保持有效的只读输入视图。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    /// <param name="alaw">为 true 时使用 A-law，否则使用 μ-law。</param>
    public static void Encode(ReadOnlySpan<short> source, Span<byte> destination, bool alaw)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination too small");
        }

        var lut = alaw ? AlawEncode : UlawEncode;
        for (int i = 0; i < source.Length; ++i)
        {
            destination[i] = lut[(ushort)source[i]];
        }
    }

    /// <summary>
    /// 按 G.711 量化边界构建完整 PCM16 编码查找表。
    /// </summary>
    /// <param name="alaw">为 true 时使用 A-law，否则使用 μ-law。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static byte[] BuildEncoder(bool alaw)
    {
        var result = new byte[65536];
        for (int n = 0; n < result.Length; ++n)
        {
            int value = (short)n;
            if (alaw)
            {
                // 采用 ITU G.711 A-law 的量化规则，包含负数的量化边界。
                int mask = value >= 0 ? 0xD5 : 0x55;
                value >>= 3;
                if (value < 0)
                {
                    value = -value - 1;
                }

                int segment = value <= 31 ? 0 : 27 - BitOperations.LeadingZeroCount((uint)value);
                int compressed = segment >= 8 ? 0x7F : (segment << 4) | ((value >> (segment < 2 ? 1 : segment)) & 15);
                result[n] = (byte)(compressed ^ mask);
            }
            else
            {
                int mask = value < 0 ? 0x7F : 0xFF;
                // 参考线性编码器在压扩前先量化到 14 位。
                value >>= 2;
                if (value < 0)
                {
                    value = -value;
                }

                value = Math.Min(value, 8159) + 33;
                int segment = value <= 63 ? 0 : 26 - BitOperations.LeadingZeroCount((uint)value);
                int compressed = segment >= 8 ? 0x7F : (segment << 4) | ((value >> (segment + 1)) & 15);
                result[n] = (byte)(compressed ^ mask);
            }
        }

        return result;
    }
}
