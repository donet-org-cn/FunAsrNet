using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>使用 x64 八路指数与双精度分母累加计算稳定的 Softmax。</summary>
public sealed partial class TensorMath
{
    /// <summary>先减最大值，再批量求指数；尾部与向量部分共用双精度分母。</summary>
    /// <param name="x">原位覆盖的连续分数。</param>
    private static void VectorSoftmaxAvx(Span<float> x)
    {
        ref float data = ref MemoryMarshal.GetReference(x);
        var maximum = Vector256.Create(float.NegativeInfinity);
        int index = 0;
        for (; index <= x.Length - 8; index += 8)
        {
            maximum = Vector256.Max(maximum, Vector256.LoadUnsafe(ref data, (nuint)index));
        }
        float max = float.NegativeInfinity;
        for (int lane = 0; lane < 8; lane++)
        {
            max = Math.Max(max, maximum.GetElement(lane));
        }
        for (; index < x.Length; index++)
        {
            max = Math.Max(max, x[index]);
        }
        var shift = Vector256.Create(max);
        var low = Vector256<double>.Zero;
        var high = low;
        index = 0;
        for (; index <= x.Length - 8; index += 8)
        {
            var value = Vector256.Exp(Vector256.LoadUnsafe(ref data, (nuint)index) - shift);
            value.StoreUnsafe(ref data, (nuint)index);
            var (left, right) = Vector256.Widen(value);
            low += left;
            high += right;
        }
        double sum = Vector256.Sum(low + high);
        for (; index < x.Length; index++)
        {
            x[index] = MathF.Exp(x[index] - max);
            sum += x[index];
        }
        float inverse = (float)(1 / sum);
        var scale = Vector256.Create(inverse);
        index = 0;
        for (; index <= x.Length - 8; index += 8)
        {
            (Vector256.LoadUnsafe(ref data, (nuint)index) * scale).StoreUnsafe(ref data, (nuint)index);
        }
        for (; index < x.Length; index++)
        {
            x[index] *= inverse;
        }
    }
}
