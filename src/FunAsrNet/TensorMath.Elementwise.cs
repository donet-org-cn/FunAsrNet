using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
public sealed partial class TensorMath
{
    private static readonly bool VectorElementwise = Environment.GetEnvironmentVariable("FUNASRNET_SCALAR_ELEMENTWISE") != "1";
    // x64 使用八路向量指数；Cortex-A76 默认保留较快的标量指数。
    private static readonly bool VectorTranscendentals = VectorElementwise
        && (Environment.GetEnvironmentVariable("FUNASRNET_VECTOR_EXP") == "1"
            || System.Runtime.Intrinsics.X86.Avx2.IsSupported && Environment.GetEnvironmentVariable("FUNASRNET_VECTOR_EXP") != "0");
    /// <summary>
    /// 使用 SIMD 执行按行归一化与仿射变换。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="scale">仿射变换或加权累加使用的缩放参数。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="epsilon">用于避免归一化除零的正数稳定项。</param>
    /// <param name="rms">为 true 时执行 RMSNorm，否则执行 LayerNorm。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private static float[] VectorNorm(float[] x, int width, float[] scale, float[]? bias, float epsilon, bool rms)
    {
        if (width <= 0 || x.Length % width != 0 || scale.Length != width || (bias is not null && bias.Length != width))
        {
            throw new ArgumentException("Invalid normalization dimensions");
        }

        var y = TensorWorkspace.Allocate(x.Length);
        for (int offset = 0; offset < x.Length; offset += width)
        {
            double mean = 0;
            if (!rms)
            {
                var sum0 = Vector128<double>.Zero;
                var sum1 = sum0;
                int j = 0;
                for (; j <= width - 4; j += 4)
                {
                    var (lo, hi) = Vector128.Widen(Vector128.LoadUnsafe(ref x[offset + j]));
                    sum0 += lo;
                    sum1 += hi;
                }

                mean = Vector128.Sum(sum0 + sum1);
                for (; j < width; ++j)
                {
                    mean += x[offset + j];
                }

                mean /= width;
            }

            var mu = Vector128.Create(mean);
            var var0 = Vector128<double>.Zero;
            var var1 = var0;
            int k = 0;
            for (; k <= width - 4; k += 4)
            {
                var (lo, hi) = Vector128.Widen(Vector128.LoadUnsafe(ref x[offset + k]));
                lo -= mu;
                hi -= mu;
                var0 += lo * lo;
                var1 += hi * hi;
            }

            double variance = Vector128.Sum(var0 + var1);
            for (; k < width; ++k)
            {
                double z = x[offset + k] - mean;
                variance += z * z;
            }

            float inverse = 1 / MathF.Sqrt((float)(variance / width) + epsilon);
            var meanVector = Vector128.Create((float)mean);
            var inverseVector = Vector128.Create(inverse);
            k = 0;
            for (; k <= width - 4; k += 4)
            {
                var result = (Vector128.LoadUnsafe(ref x[offset + k]) - meanVector) * inverseVector * Vector128.LoadUnsafe(ref scale[k]);
                if (bias is not null)
                {
                    result += Vector128.LoadUnsafe(ref bias[k]);
                }

                result.StoreUnsafe(ref y[offset + k]);
            }

            for (; k < width; ++k)
            {
                y[offset + k] = (x[offset + k] - (float)mean) * inverse * scale[k] + (bias?[k] ?? 0);
            }
        }

        return y;
    }

    /// <summary>
    /// 使用向量化归约和标量指数执行原位 Softmax。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    private static void VectorSoftmax(Span<float> x)
    {
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            VectorSoftmaxAvx(x);
            return;
        }
        ref float p = ref MemoryMarshal.GetReference(x);
        var maxima = Vector128.Create(float.NegativeInfinity);
        int i = 0;
        for (; i <= x.Length - 4; i += 4)
        {
            maxima = Vector128.Max(maxima, Vector128.LoadUnsafe(ref p, (nuint)i));
        }

        float max = Vector128.Max(maxima, Vector128.Shuffle(maxima, Vector128.Create(2, 3, 0, 1))).GetElement(0);
        max = Math.Max(max, Math.Max(maxima.GetElement(1), maxima.GetElement(3)));
        for (; i < x.Length; ++i)
        {
            max = Math.Max(max, x[i]);
        }

        var maxVector = Vector128.Create(max);
        var sum0 = Vector128<double>.Zero;
        var sum1 = sum0;
        i = 0;
        for (; i <= x.Length - 4; i += 4)
        {
            var e = Vector128.Exp(Vector128.LoadUnsafe(ref p, (nuint)i) - maxVector);
            e.StoreUnsafe(ref p, (nuint)i);
            var (lo, hi) = Vector128.Widen(e);
            sum0 += lo;
            sum1 += hi;
        }

        double sum = Vector128.Sum(sum0 + sum1);
        for (; i < x.Length; ++i)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }

        float inv = (float)(1 / sum);
        var inverse = Vector128.Create(inv);
        i = 0;
        for (; i <= x.Length - 4; i += 4)
        {
            (Vector128.LoadUnsafe(ref p, (nuint)i) * inverse).StoreUnsafe(ref p, (nuint)i);
        }

        for (; i < x.Length; ++i)
        {
            x[i] *= inv;
        }
    }

    /// <summary>
    /// 计算门控向量的 SiLU 激活，并逐元素乘以上投影向量。
    /// </summary>
    /// <param name="gate">门控投影向量，结果写回该数组。</param>
    /// <param name="up">与门控激活逐元素相乘的上投影向量。</param>
    public static void SiluMultiply(float[] gate, float[] up)
    {
        if (gate.Length != up.Length)
        {
            throw new ArgumentException("SwiGLU dimensions differ");
        }

        int i = 0;
        if (Vector256.IsHardwareAccelerated && VectorTranscendentals)
        {
            for (; i <= gate.Length - 8; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref gate[i]);
                (g / (Vector256.Create(1f) + Vector256.Exp(-g)) * Vector256.LoadUnsafe(ref up[i])).StoreUnsafe(ref gate[i]);
            }
        }
        if (Vector128.IsHardwareAccelerated && VectorTranscendentals)
        {
            for (; i <= gate.Length - 4; i += 4)
            {
                var g = Vector128.LoadUnsafe(ref gate[i]);
                (g / (Vector128.Create(1f) + Vector128.Exp(-g)) * Vector128.LoadUnsafe(ref up[i])).StoreUnsafe(ref gate[i]);
            }
        }

        for (; i < gate.Length; ++i)
        {
            gate[i] = gate[i] / (1 + MathF.Exp(-gate[i])) * up[i];
        }
    }
}
