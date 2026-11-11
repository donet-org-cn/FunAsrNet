using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>固定尺寸 FFT 的共享旋转因子与保持舍入顺序的 SIMD 蝶形运算。</summary>
public sealed partial class FilterBank
{
    private static readonly (float[] Real, float[] Imaginary) Twiddles = CreateTwiddles();

    /// <summary>按旧 FFT 的逐次乘法顺序生成各级旋转因子，仅在类型初始化时计算一次。</summary>
    /// <returns>按级拼接的只读实部、虚部数组，共 FftSize 减一项。</returns>
    private static (float[] Real, float[] Imaginary) CreateTwiddles()
    {
        var real = new float[FftSize - 1];
        var imaginary = new float[FftSize - 1];
        for (int length = 2; length <= FftSize; length <<= 1)
        {
            float angle = (float)(-2 * Math.PI / length), wr = MathF.Cos(angle), wi = MathF.Sin(angle);
            float cr = 1, ci = 0;
            int start = length / 2 - 1;
            for (int k = 0; k < length / 2; ++k)
            {
                real[start + k] = cr;
                imaginary[start + k] = ci;
                float next = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next;
            }
        }
        return (real, imaginary);
    }

    /// <summary>原位执行固定 512 点 FFT；向量内的每个蝶形保持旧算法独立乘法和加减顺序。</summary>
    /// <param name="re">长度固定为 512 的实部工作数组，结果原位写回。</param>
    /// <param name="im">长度固定为 512 的虚部工作数组，结果原位写回。</param>
    private static void Fft(Span<float> re, Span<float> im)
    {
        for (int i = 1, j = 0; i < FftSize; ++i)
        {
            int bit = FftSize >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        var (real, imaginary) = Twiddles;
        for (int length = 2; length <= FftSize; length <<= 1)
        {
            int half = length / 2, start = half - 1;
            for (int i = 0; i < FftSize; i += length)
            {
                int k = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    for (; k <= half - 8; k += 8)
                    {
                        int a = i + k, b = a + half;
                        var cr = Vector256.LoadUnsafe(ref real[start + k]);
                        var ci = Vector256.LoadUnsafe(ref imaginary[start + k]);
                        var ur = Vector256.LoadUnsafe(ref re[a]);
                        var ui = Vector256.LoadUnsafe(ref im[a]);
                        var br = Vector256.LoadUnsafe(ref re[b]);
                        var bi = Vector256.LoadUnsafe(ref im[b]);
                        // 不使用 FMA，保留与原标量 FFT 一致的中间舍入。
                        var vr = br * cr - bi * ci;
                        var vi = br * ci + bi * cr;
                        (ur + vr).StoreUnsafe(ref re[a]);
                        (ui + vi).StoreUnsafe(ref im[a]);
                        (ur - vr).StoreUnsafe(ref re[b]);
                        (ui - vi).StoreUnsafe(ref im[b]);
                    }
                }
                if (Vector128.IsHardwareAccelerated)
                {
                    for (; k <= half - 4; k += 4)
                    {
                        int a = i + k, b = a + half;
                        var cr = Vector128.LoadUnsafe(ref real[start + k]);
                        var ci = Vector128.LoadUnsafe(ref imaginary[start + k]);
                        var ur = Vector128.LoadUnsafe(ref re[a]);
                        var ui = Vector128.LoadUnsafe(ref im[a]);
                        var br = Vector128.LoadUnsafe(ref re[b]);
                        var bi = Vector128.LoadUnsafe(ref im[b]);
                        var vr = br * cr - bi * ci;
                        var vi = br * ci + bi * cr;
                        (ur + vr).StoreUnsafe(ref re[a]);
                        (ui + vi).StoreUnsafe(ref im[a]);
                        (ur - vr).StoreUnsafe(ref re[b]);
                        (ui - vi).StoreUnsafe(ref im[b]);
                    }
                }
                for (; k < half; ++k)
                {
                    int a = i + k, b = a + half;
                    float cr = real[start + k], ci = imaginary[start + k];
                    float ur = re[a], ui = im[a], vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
                    re[a] = ur + vr;
                    im[a] = ui + vi;
                    re[b] = ur - vr;
                    im[b] = ui - vi;
                }
            }
        }
    }

    /// <summary>每帧仅计算一次功率谱，供全部 Mel 滤波器复用。</summary>
    /// <param name="re">FFT 实部，至少包含功率输出所需的频点。</param>
    /// <param name="im">FFT 虚部，与实部数组长度相同。</param>
    /// <param name="power">接收前 257 个频点的功率，调用内完整覆盖。</param>
    private static void PowerSpectrum(ReadOnlySpan<float> re, ReadOnlySpan<float> im, Span<float> power)
    {
        int k = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; k <= power.Length - 8; k += 8)
            {
                var r = Vector256.LoadUnsafe(in re[k]);
                var i = Vector256.LoadUnsafe(in im[k]);
                (r * r + i * i).StoreUnsafe(ref power[k]);
            }
        }
        if (Vector128.IsHardwareAccelerated)
        {
            for (; k <= power.Length - 4; k += 4)
            {
                var r = Vector128.LoadUnsafe(in re[k]);
                var i = Vector128.LoadUnsafe(in im[k]);
                (r * r + i * i).StoreUnsafe(ref power[k]);
            }
        }
        for (; k < power.Length; ++k)
        {
            power[k] = re[k] * re[k] + im[k] * im[k];
        }
    }
}
