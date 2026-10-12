using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>保留 FP32 和广播语义的连续逐元素向量算子。</summary>
public static partial class TensorOps
{
    /// <summary>按连续末维选择输入，标量条件及操作数广播，保留 NaN 非零判定和有符号零。</summary>
    /// <param name="condition">条件块，长度为一或输出长度。</param>
    /// <param name="left">条件非零时采用的输入。</param>
    /// <param name="right">条件为零时采用的输入。</param>
    /// <param name="output">调用方提供的完整结果区间。</param>
    private static void SelectBlock(ReadOnlySpan<float> condition, ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        int index = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; index <= output.Length - 8; index += 8)
            {
                var c = condition.Length == 1 ? Vector256.Create(condition[0]) : Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(condition), (nuint)index);
                var a = left.Length == 1 ? Vector256.Create(left[0]) : Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(left), (nuint)index);
                var b = right.Length == 1 ? Vector256.Create(right[0]) : Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(right), (nuint)index);
                Vector256.ConditionalSelect(Vector256.Equals(c, Vector256<float>.Zero), b, a).StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
            }
        }
        if (Vector128.IsHardwareAccelerated)
        {
            for (; index <= output.Length - 4; index += 4)
            {
                var c = condition.Length == 1 ? Vector128.Create(condition[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(condition), (nuint)index);
                var a = left.Length == 1 ? Vector128.Create(left[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(left), (nuint)index);
                var b = right.Length == 1 ? Vector128.Create(right[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(right), (nuint)index);
                Vector128.ConditionalSelect(Vector128.Equals(c, Vector128<float>.Zero), b, a).StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
            }
        }
        for (; index < output.Length; index++)
        {
            output[index] = condition[condition.Length == 1 ? 0 : index] != 0 ? left[left.Length == 1 ? 0 : index] : right[right.Length == 1 ? 0 : index];
        }
    }

    /// <summary>按连续末维分块向量化加减乘除，整数除法保留截断语义。</summary>
    /// <param name="a">左操作数，只读且保持原始存储精度。</param>
    /// <param name="b">右操作数，只读；可与左侧形状相同或广播。</param>
    /// <param name="operation">运算名称，仅接受 Add、Sub、Mul、Div。</param>
    /// <param name="math">可选同步计算线程组；大连续结果按独立区间分配，不改变逐元素顺序。</param>
    /// <returns>拥有独立存储的结果张量，不覆盖任何共享输入。</returns>
    public static Tensor Arithmetic(Tensor a, Tensor b, string operation, TensorMath? math = null)
        => ArithmeticOwned(a, b, operation, math, null);

    /// <summary>在解释器证明输入已经结束生命期后，允许复用其完整存储作为结果。</summary>
    /// <param name="a">只读语义的左操作数。</param>
    /// <param name="b">只读语义的右操作数。</param>
    /// <param name="operation">加减乘除算子名称。</param>
    /// <param name="math">可选同步计算调度器。</param>
    /// <param name="reuse">已确认没有其他观察者且形状等于输出的自有临时张量。</param>
    /// <returns>完整写入的输出张量。</returns>
    internal static Tensor ArithmeticOwned(Tensor a, Tensor b, string operation, TensorMath? math, Tensor? reuse)
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
            var y = reuse is null ? Tensor.ForOverwrite(shape, a.Integer && b.Integer) : new Tensor(reuse.Data, shape, a.Integer && b.Integer);
            if (math is not null && y.Count >= 65536 && Environment.GetEnvironmentVariable("FUNASRNET_SERIAL_ELEMENTWISE") != "1")
            {
                math.For((y.Count + 65535) / 65536, block =>
                {
                    int start = block * 65536, count = Math.Min(65536, y.Count - start);
                    ArithmeticBlock(a.Data.AsSpan(a.Count == 1 ? 0 : start, a.Count == 1 ? 1 : count), b.Data.AsSpan(b.Count == 1 ? 0 : start, b.Count == 1 ? 1 : count), y.Data.AsSpan(start, count), operation, scalar);
                });
            }
            else
            {
                ArithmeticBlock(a.Data, b.Data, y.Data, operation, scalar);
            }
            return y;
        }

        // 时间掩码和通道常量都按行广播，只在每行开始计算一次外层坐标。
        // 最后一维的访问步长只能为零或一，可直接使用连续向量内核。
        var result = Tensor.ForOverwrite(shape, a.Integer && b.Integer);
        if (result.Count != 0)
        {
            int width = shape[^1];
            int[] outer = shape[..^1], sa = BroadcastStrides(a, shape.Length), sb = BroadcastStrides(b, shape.Length);
            int leftWidth = sa[^1] == 0 ? 1 : width, rightWidth = sb[^1] == 0 ? 1 : width;
            int rows = result.Count / width;
            int rowsPerTask = Math.Max(1, 65536 / width);
            // 大时间轴的通道偏置和掩码同样需要并行，避免广播分支退回单核。
            if (math is not null && result.Count >= 65536 && Environment.GetEnvironmentVariable("FUNASRNET_SERIAL_ELEMENTWISE") != "1")
            {
                math.For((rows + rowsPerTask - 1) / rowsPerTask, task =>
                {
                    for (int block = task * rowsPerTask; block < Math.Min(rows, (task + 1) * rowsPerTask); block++)
                    {
                        var left = a.Data.AsSpan(Offset(block, outer, sa), leftWidth);
                        var right = b.Data.AsSpan(Offset(block, outer, sb), rightWidth);
                        ArithmeticBlock(left, right, result.Data.AsSpan(block * width, width), operation, scalar);
                    }
                });
            }
            else
            {
                for (int block = 0; block < rows; block++)
                {
                    var left = a.Data.AsSpan(Offset(block, outer, sa), leftWidth);
                    var right = b.Data.AsSpan(Offset(block, outer, sb), rightWidth);
                    ArithmeticBlock(left, right, result.Data.AsSpan(block * width, width), operation, scalar);
                }
            }
        }
        return result;
    }

    /// <summary>计算连续块，长度为一的操作数按标量广播。</summary>
    /// <param name="left">左侧块，长度为输出长度或一。</param>
    /// <param name="right">右侧块，长度为输出长度或一。</param>
    /// <param name="output">调用方提供的独立结果区间。</param>
    /// <param name="operation">已验证的运算名称。</param>
    /// <param name="scalar">处理不足四个元素的尾部所用参考运算。</param>
    private static void ArithmeticBlock(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output, string operation, Func<float, float, float> scalar)
    {
        switch (operation)
        {
            case "Add": ArithmeticBlockCore<AddOperation>(left, right, output, scalar); break;
            case "Sub": ArithmeticBlockCore<SubtractOperation>(left, right, output, scalar); break;
            case "Mul": ArithmeticBlockCore<MultiplyOperation>(left, right, output, scalar); break;
            case "Div": ArithmeticBlockCore<DivideOperation>(left, right, output, scalar); break;
            default: throw new InvalidOperationException("未验证的逐元素运算。");
        }
    }

    /// <summary>通过值类型专用化在循环外选定运算，消除每个向量的字符串分派。</summary>
    /// <param name="left">连续左侧输入或标量。</param>
    /// <param name="right">连续右侧输入或标量。</param>
    /// <param name="output">完整覆盖的独立结果区间。</param>
    /// <param name="scalar">不足一个向量时的参考运算。</param>
    private static void ArithmeticBlockCore<TOperation>(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output, Func<float, float, float> scalar) where TOperation : struct
    {
        int index = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; index <= output.Length - 8; index += 8)
            {
                var a = left.Length == 1 ? Vector256.Create(left[0]) : Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(left), (nuint)index);
                var b = right.Length == 1 ? Vector256.Create(right[0]) : Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(right), (nuint)index);
                var value = ArithmeticValue<TOperation>(a, b);
                value.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
            }
        }
        for (; index <= output.Length - 4; index += 4)
        {
            var a = left.Length == 1 ? Vector128.Create(left[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(left), (nuint)index);
            var b = right.Length == 1 ? Vector128.Create(right[0]) : Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(right), (nuint)index);
            var value = ArithmeticValue<TOperation>(a, b);
            value.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
        }

        for (; index < output.Length; index++)
        {
            output[index] = scalar(left[left.Length == 1 ? 0 : index], right[right.Length == 1 ? 0 : index]);
        }
    }

    /// <summary>由即时编译器按类型常量选定单个向量的算术运算。</summary>
    /// <param name="a">左输入向量。</param>
    /// <param name="b">右输入向量。</param>
    /// <returns>保持逐元素浮点语义的运算结果。</returns>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ArithmeticValue<TOperation>(Vector256<float> a, Vector256<float> b) where TOperation : struct
    {
        if (typeof(TOperation) == typeof(AddOperation))
        {
            return a + b;
        }
        if (typeof(TOperation) == typeof(SubtractOperation))
        {
            return a - b;
        }
        return typeof(TOperation) == typeof(MultiplyOperation) ? a * b : a / b;
    }

    /// <summary>由即时编译器按类型常量选定单个向量的算术运算。</summary>
    /// <param name="a">左输入向量。</param>
    /// <param name="b">右输入向量。</param>
    /// <returns>保持逐元素浮点语义的运算结果。</returns>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ArithmeticValue<TOperation>(Vector128<float> a, Vector128<float> b) where TOperation : struct
    {
        if (typeof(TOperation) == typeof(AddOperation))
        {
            return a + b;
        }
        if (typeof(TOperation) == typeof(SubtractOperation))
        {
            return a - b;
        }
        return typeof(TOperation) == typeof(MultiplyOperation) ? a * b : a / b;
    }

    /// <summary>加法内核的编译期类型标记。</summary>
    private readonly struct AddOperation { }

    /// <summary>减法内核的编译期类型标记。</summary>
    private readonly struct SubtractOperation { }

    /// <summary>乘法内核的编译期类型标记。</summary>
    private readonly struct MultiplyOperation { }

    /// <summary>除法内核的编译期类型标记。</summary>
    private readonly struct DivideOperation { }

    /// <summary>向量化 LeakyRelu，保留负零和 NaN 的标量分支语义。</summary>
    /// <param name="x">只读 FP32 输入张量。</param>
    /// <param name="alpha">负值分支的乘数，与模型属性一致。</param>
    /// <param name="math">可选线程组，大张量分块并行。</param>
    /// <returns>具有相同形状和独立存储的结果。</returns>
    public static Tensor LeakyRelu(Tensor x, float alpha, TensorMath? math = null)
        => LeakyReluOwned(x, alpha, math, null);

    /// <summary>对解释器独占且不再读取的临时数组执行可原地的激活。</summary>
    /// <param name="x">输入张量。</param>
    /// <param name="alpha">负值缩放系数。</param>
    /// <param name="math">可选同步计算调度器。</param>
    /// <param name="reuse">与输入同一存储且已通过生命期检查的张量。</param>
    /// <returns>激活后的张量，数值语义与独立分配相同。</returns>
    internal static Tensor LeakyReluOwned(Tensor x, float alpha, TensorMath? math, Tensor? reuse)
    {
        var y = reuse is null ? Tensor.ForOverwrite(x.Shape) : new Tensor(reuse.Data, x.Shape);
        if (math is not null && x.Count >= 65536 && Environment.GetEnvironmentVariable("FUNASRNET_SERIAL_ELEMENTWISE") != "1")
        {
            math.For((x.Count + 65535) / 65536, block =>
            {
                int start = block * 65536, count = Math.Min(65536, x.Count - start);
                LeakyReluBlock(x.Data.AsSpan(start, count), y.Data.AsSpan(start, count), alpha);
            });
        }
        else
        {
            LeakyReluBlock(x.Data, y.Data, alpha);
        }
        return y;
    }

    /// <summary>处理独立的连续激活区间，保留负零和非数分支。</summary>
    /// <param name="input">只读输入区间。</param>
    /// <param name="output">待完整覆盖的输出区间。</param>
    /// <param name="alpha">模型定义的负数缩放系数。</param>
    private static void LeakyReluBlock(ReadOnlySpan<float> input, Span<float> output, float alpha)
    {
        int index = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var wideScale = Vector256.Create(alpha);
            for (; index <= input.Length - 8; index += 8)
            {
                var value = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(input), (nuint)index);
                Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(value, Vector256<float>.Zero), value, value * wideScale).StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
            }
        }
        var scale = Vector128.Create(alpha);
        if (Vector128.IsHardwareAccelerated)
        {
            for (; index <= input.Length - 4; index += 4)
            {
                var value = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(input), (nuint)index);
                Vector128.ConditionalSelect(Vector128.GreaterThanOrEqual(value, Vector128<float>.Zero), value, value * scale).StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)index);
            }
        }

        for (; index < input.Length; index++)
        {
            float value = input[index];
            output[index] = value >= 0 ? value : value * alpha;
        }

    }
}
