using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>使用运行时向量指数实现 x64 激活函数，其他平台保留原标量实现。</summary>
public static partial class TensorOps
{
    /// <summary>批量计算双曲正切，小输入使用奇多项式以避免指数相减损失精度。</summary>
    /// <param name="x">只读浮点输入张量。</param>
    /// <returns>保留输入形状、非数和零符号的结果。</returns>
    public static Tensor Tanh(Tensor x)
    {
        if (!Avx2.IsSupported)
        {
            return Unary(x, MathF.Tanh);
        }
        var result = Tensor.ForOverwrite(x.Shape);
        int index = 0;
        var one = Vector256.Create(1f);
        for (; index <= x.Count - 8; index += 8)
        {
            var value = Vector256.LoadUnsafe(ref x.Data[index]);
            var magnitude = Vector256.Abs(value);
            var exponential = Vector256.Exp(Vector256.Create(-2f) * magnitude);
            var regular = Vector256.CopySign((one - exponential) / (one + exponential), value);
            var small = Vector256.Min(magnitude, Vector256.Create(0.125f));
            var square = small * small;
            var polynomial = value * (one + square * (Vector256.Create(-1f / 3) + square * (Vector256.Create(2f / 15) + square * (Vector256.Create(-17f / 315) + square * Vector256.Create(62f / 2835)))));
            Vector256.ConditionalSelect(Vector256.LessThan(magnitude, Vector256.Create(0.125f)), polynomial, regular).StoreUnsafe(ref result.Data[index]);
        }
        for (; index < x.Count; index++)
        {
            result.Data[index] = MathF.Tanh(x.Data[index]);
        }
        return result;
    }

    /// <summary>逐元素计算 sigmoid，使用运行时向量指数并保持 FP32 存储和原函数定义。</summary>
    /// <param name="x">只读输入张量。</param>
    /// <returns>具有相同形状的独立结果。</returns>
    public static Tensor Sigmoid(Tensor x)
    {
        if (!Avx2.IsSupported)
        {
            return Unary(x, value => 1 / (1 + MathF.Exp(-value)));
        }
        var result = Tensor.ForOverwrite(x.Shape);
        int index = 0;
        var one = Vector256.Create(1f);
        for (; index <= x.Count - 8; index += 8)
        {
            var value = Vector256.LoadUnsafe(ref x.Data[index]);
            (one / (one + Vector256.Exp(-value))).StoreUnsafe(ref result.Data[index]);
        }
        for (; index < x.Count; index++)
        {
            result.Data[index] = 1 / (1 + MathF.Exp(-x.Data[index]));
        }
        return result;
    }

    /// <summary>保留原误差函数的双精度系数、运算顺序和舍入，仅批量计算指数。</summary>
    /// <param name="x">只读 FP32 输入。</param>
    /// <returns>与原函数相同形状的 FP32 结果。</returns>
    public static Tensor ErrorFunction(Tensor x)
    {
        var result = Tensor.ForOverwrite(x.Shape);
        int index = 0;
        if (Avx2.IsSupported)
        {
            for (; index <= x.Count - 8; index += 8)
            {
                var (left, right) = Vector256.Widen(Vector256.LoadUnsafe(ref x.Data[index]));
                Vector256.Narrow(ErrorFunctionBlock(left), ErrorFunctionBlock(right)).StoreUnsafe(ref result.Data[index]);
            }
        }
        for (; index < x.Count; index++)
        {
            double value = x.Data[index], a = Math.Abs(value), t = 1 / (1 + 0.3275911 * a);
            double p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
            result.Data[index] = (float)Math.CopySign(1 - p * Math.Exp(-a * a), value);
        }
        return result;
    }

    /// <summary>按四个双精度元素计算原误差函数表达式，避免改变乘加结合方式。</summary>
    /// <param name="value">原输入，含用于恢复结果符号的符号位。</param>
    /// <returns>未窄化的双精度计算结果。</returns>
    private static Vector256<double> ErrorFunctionBlock(Vector256<double> value)
    {
        var a = Vector256.Abs(value);
        var t = Vector256.Create(1d) / (Vector256.Create(1d) + Vector256.Create(0.3275911) * a);
        var p = (((((Vector256.Create(1.061405429) * t - Vector256.Create(1.453152027)) * t) + Vector256.Create(1.421413741)) * t - Vector256.Create(0.284496736)) * t + Vector256.Create(0.254829592)) * t;
        var magnitude = Vector256.Create(1d) - p * Vector256.Exp(-a * a);
        return Vector256.CopySign(magnitude, value);
    }
}
