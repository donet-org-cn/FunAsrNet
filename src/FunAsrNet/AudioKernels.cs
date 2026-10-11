using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// 音频采样转换和点积内核；支持向量化及标量回退。
/// </summary>
public static class AudioKernels
{
    /// <summary>
    /// 当前运行环境是否支持 ARM Advanced SIMD。
    /// </summary>
    public static bool ArmNeon => AdvSimd.IsSupported;

    /// <summary>
    /// 将有符号 PCM16 采样归一化到浮点范围，结果写入调用方缓冲区。
    /// </summary>
    /// <param name="source">在调用期间保持有效的只读输入视图。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    public static void Pcm16ToFloat(ReadOnlySpan<short> source, Span<float> destination, bool simd = true)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination too small");
        }

        int i = 0;
        if (simd && Vector128.IsHardwareAccelerated)
        {
            ref short input = ref MemoryMarshal.GetReference(source);
            ref float output = ref MemoryMarshal.GetReference(destination);
            var scale = Vector128.Create(1f / 32768);
            for (; i <= source.Length - 8; i += 8)
            {
                var (lo, hi) = Vector128.Widen(Vector128.LoadUnsafe(ref input, (nuint)i));
                (Vector128.ConvertToSingle(lo) * scale).StoreUnsafe(ref output, (nuint)i);
                (Vector128.ConvertToSingle(hi) * scale).StoreUnsafe(ref output, (nuint)(i + 4));
            }
        }

        for (; i < source.Length; ++i)
        {
            destination[i] = source[i] * (1f / 32768);
        }
    }

    /// <summary>
    /// 将浮点采样限幅并量化为 PCM16，写入调用方提供的输出视图。
    /// </summary>
    /// <param name="source">在调用期间保持有效的只读输入视图。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    public static void FloatToPcm16(ReadOnlySpan<float> source, Span<short> destination, bool simd = true)
    {
        if (destination.Length < source.Length)
        {
            throw new ArgumentException("Destination too small");
        }

        int i = 0;
        if (simd && Vector128.IsHardwareAccelerated)
        {
            ref float input = ref MemoryMarshal.GetReference(source);
            ref short output = ref MemoryMarshal.GetReference(destination);
            var min = Vector128.Create(-1f);
            var max = Vector128.Create(1f);
            var scale = Vector128.Create(32767f);
            for (; i <= source.Length - 8; i += 8)
            {
                var a = Vector128.Min(max, Vector128.Max(min, Vector128.LoadUnsafe(ref input, (nuint)i)));
                var b = Vector128.Min(max, Vector128.Max(min, Vector128.LoadUnsafe(ref input, (nuint)(i + 4))));
                Vector128.Narrow(Vector128.ConvertToInt32(a * scale), Vector128.ConvertToInt32(b * scale)).StoreUnsafe(ref output, (nuint)i);
            }
        }

        for (; i < source.Length; ++i)
        {
            destination[i] = (short)(Math.Clamp(source[i], -1, 1) * 32767);
        }
    }

    /// <summary>
    /// 计算两个等长向量的点积，支持 SIMD 与标量路径。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    /// <returns>计算得到的标量数值。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b, bool simd = true)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Dot-product lengths differ");
        }

        int i = 0;
        float result = 0;
        if (simd && Vector128.IsHardwareAccelerated)
        {
            ref float x = ref MemoryMarshal.GetReference(a);
            ref float y = ref MemoryMarshal.GetReference(b);
            var acc0 = Vector128<float>.Zero;
            var acc1 = Vector128<float>.Zero;
            for (; i <= a.Length - 8; i += 8)
            {
                var x0 = Vector128.LoadUnsafe(ref x, (nuint)i);
                var y0 = Vector128.LoadUnsafe(ref y, (nuint)i);
                var x1 = Vector128.LoadUnsafe(ref x, (nuint)(i + 4));
                var y1 = Vector128.LoadUnsafe(ref y, (nuint)(i + 4));
                if (AdvSimd.IsSupported)
                {
                    acc0 = AdvSimd.FusedMultiplyAdd(acc0, x0, y0);
                    acc1 = AdvSimd.FusedMultiplyAdd(acc1, x1, y1);
                }
                else
                {
                    acc0 += x0 * y0;
                    acc1 += x1 * y1;
                }
            }

            result = Vector128.Sum(acc0 + acc1);
        }

        for (; i < a.Length; ++i)
        {
            result += a[i] * b[i];
        }

        return result;
    }
}
