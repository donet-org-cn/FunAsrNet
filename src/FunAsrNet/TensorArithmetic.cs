using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>保留 FP32 和广播语义的连续逐元素向量算子。</summary>
public static partial class TensorOps
{
    /// <summary>向量化加减乘除，复杂广播和整数除法继续使用通用实现。</summary>
    /// <param name="a">左操作数，只读且保持原始存储精度。</param>
    /// <param name="b">右操作数，只读；可与左侧形状相同或广播。</param>
    /// <param name="operation">运算名称，仅接受 Add、Sub、Mul、Div。</param>
    /// <returns>拥有独立存储的结果张量，不覆盖任何共享输入。</returns>
    public static Tensor Arithmetic(Tensor a, Tensor b, string operation)
    {
        Func<float, float, float> scalar = operation switch
        {
            "Add" => (left, right) => left + right,
            "Sub" => (left, right) => left - right,
            "Mul" => (left, right) => left * right,
            "Div" when a.Integer && b.Integer => (left, right) => MathF.Truncate(left / right),
            "Div" => (left, right) => left / right,
            _ => throw new ArgumentException("只支持 Add、Sub、Mul、Div。", nameof(operation))
        };
        if (!Vector128.IsHardwareAccelerated || operation == "Div" && a.Integer && b.Integer)
        {
            return Binary(a, b, scalar, a.Integer && b.Integer);
        }

        int[] shape = BroadcastShape(a.Shape, b.Shape);
        if (a.Shape.SequenceEqual(b.Shape) || a.Count == 1 || b.Count == 1)
        {
            var y = Tensor.Zeros(shape, a.Integer && b.Integer);
            ArithmeticBlock(a.Data, b.Data, y.Data, operation, scalar);
            return y;
        }

        // NCL 通道常量广播：每个时间块只查询一次通道值，不对每个采样点做除余索引。
        bool leftFull = a.Shape.SequenceEqual(shape);
        Tensor full = leftFull ? a : b, channel = leftFull ? b : a;
        int[] channelShape = Enumerable.Repeat(1, 3 - Math.Min(3, channel.Rank)).Concat(channel.Shape).ToArray();
        if (shape.Length == 3 && full.Shape.SequenceEqual(shape) && channelShape.Length == 3
            && channelShape[0] == 1 && channelShape[1] == shape[1] && channelShape[2] == 1)
        {
            var y = Tensor.Zeros(shape, a.Integer && b.Integer);
            for (int block = 0; block < shape[0] * shape[1]; block++)
            {
                var values = full.Data.AsSpan(block * shape[2], shape[2]);
                var value = channel.Data.AsSpan(block % shape[1], 1);
                ArithmeticBlock(leftFull ? values : value, leftFull ? value : values, y.Data.AsSpan(block * shape[2], shape[2]), operation, scalar);
            }

            return y;
        }

        return Binary(a, b, scalar, a.Integer && b.Integer);
    }

    /// <summary>计算连续块，长度为一的操作数按标量广播。</summary>
    /// <param name="left">左侧块，长度为输出长度或一。</param>
    /// <param name="right">右侧块，长度为输出长度或一。</param>
    /// <param name="output">调用方提供的独立结果区间。</param>
    /// <param name="operation">已验证的运算名称。</param>
    /// <param name="scalar">处理不足四个元素的尾部所用参考运算。</param>
    private static void ArithmeticBlock(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output, string operation, Func<float, float, float> scalar)
    {
        int index = 0;
        for (; index <= output.Length - 4; index += 4)
        {
            var a = left.Length == 1 ? Vector128.Create(left[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(left), (nuint)index);
            var b = right.Length == 1 ? Vector128.Create(right[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(right), (nuint)index);
            var value = operation switch
            {
                "Add" => a + b,
                "Sub" => a - b,
                "Mul" => a * b,
                "Div" => a / b,
                _ => throw new InvalidOperationException("未验证的逐元素运算。")
            };
            value.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
        }

        for (; index < output.Length; index++)
        {
            output[index] = scalar(left[left.Length == 1 ? 0 : index], right[right.Length == 1 ? 0 : index]);
        }
    }

    /// <summary>向量化 LeakyRelu，保留负零和 NaN 的标量分支语义。</summary>
    /// <param name="x">只读 FP32 输入张量。</param>
    /// <param name="alpha">负值分支的乘数，与模型属性一致。</param>
    /// <returns>具有相同形状和独立存储的结果。</returns>
    public static Tensor LeakyRelu(Tensor x, float alpha)
    {
        var y = Tensor.Zeros(x.Shape);
        int index = 0;
        var scale = Vector128.Create(alpha);
        if (Vector128.IsHardwareAccelerated)
        {
            for (; index <= x.Count - 4; index += 4)
            {
                var value = Vector128.LoadUnsafe(ref x.Data[index]);
                Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(value, Vector128<float>.Zero), value, value * scale).StoreUnsafe(ref y.Data[index]);
            }
        }

        for (; index < x.Count; index++)
        {
            float value = x.Data[index];
            y.Data[index] = value >= 0 ? value : value * alpha;
        }

        return y;
    }
}
