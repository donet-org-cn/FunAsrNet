using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>在卷积寄存器结果上执行精确顺序的残差相加与激活。</summary>
public static partial class TensorOps
{
    /// <summary>不支持直接融合的卷积路径复用自己的输出进行后处理，广播保持独立算子语义。</summary>
    /// <param name="value">卷积刚分配并完整写入的自有结果。</param>
    /// <param name="residual">在偏置之后相加的可选张量。</param>
    /// <param name="leakyAlpha">相加完成后可选的激活系数。</param>
    /// <param name="math">当前同步计算工作线程组。</param>
    /// <returns>后处理完成的张量，不覆盖调用方输入。</returns>
    private static Tensor FinishConvEpilogue(Tensor value, Tensor? residual, float? leakyAlpha, TensorMath math)
    {
        if (residual is not null)
        {
            Tensor? reuse = TensorWorkspace.Owns(value.Data) && value.Shape.SequenceEqual(BroadcastShape(value.Shape, residual.Shape)) ? value : null;
            var result = ArithmeticOwned(value, residual, "Add", math, reuse);
            if (!ReferenceEquals(value.Data, result.Data))
            {
                TensorWorkspace.Release(value.Data);
            }
            value = result;
        }
        if (leakyAlpha is { } alpha)
        {
            var result = LeakyReluOwned(value, alpha, math, value);
            value = result;
        }
        return value;
    }

    /// <summary>在偏置之后按原顺序加入残差并激活，只读取当前通道内有效尾部。</summary>
    /// <param name="value">已完成原点积和偏置相加的八元素向量。</param>
    /// <param name="destination">当前输出通道的起始引用。</param>
    /// <param name="offset">相对输出通道的写入元素偏移。</param>
    /// <param name="count">当前时间块剩余的有效元素数，可小于等于零。</param>
    /// <param name="residual">与输出同形的只读残差，或空。</param>
    /// <param name="residualOffset">当前向量在残差数组中的绝对偏移。</param>
    /// <param name="leakyAlpha">可选的 LeakyRelu 负值缩放系数。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreConvTime(Vector256<float> value, ref float destination, int offset, int count, Tensor? residual, int residualOffset, float? leakyAlpha)
    {
        if (residual is null && leakyAlpha is null)
        {
            StoreTime(value, ref destination, offset, count);
            return;
        }
        if (count <= 0)
        {
            return;
        }
        if (residual is not null)
        {
            if (count >= 8)
            {
                value += Vector256.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(residual.Data), (nuint)residualOffset);
            }
            else
            {
                value += ReadConvResidualTail(residual.Data, residualOffset, count);
            }
        }
        if (leakyAlpha is { } alpha)
        {
            value = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(value, Vector256<float>.Zero), value, value * Vector256.Create(alpha));
        }
        StoreTime(value, ref destination, offset, count);
    }

    /// <summary>读取卷积残差的不足八元素尾部，将栈分配限制在少数边界向量中。</summary>
    /// <param name="data">与输出同形的残差数组。</param>
    /// <param name="offset">当前通道尾部的绝对元素偏移。</param>
    /// <param name="count">必须位于一到七之间的有效元素数。</param>
    /// <returns>仅有效元素非零的八元素残差向量。</returns>
    private static Vector256<float> ReadConvResidualTail(float[] data, int offset, int count)
    {
        Span<float> padded = stackalloc float[8];
        padded.Clear();
        for (int index = 0; index < count; index++)
        {
            padded[index] = data[offset + index];
        }
        return Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(padded));
    }
}
