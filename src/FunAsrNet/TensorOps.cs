namespace FunAsrNet;

/// <summary>
/// Huayan 模型需要的张量形状、索引、逐元素与卷积算子。
/// </summary>
public static partial class TensorOps
{
    /// <summary>
    /// 将负轴编号规范化为有效的非负轴编号。
    /// </summary>
    /// <param name="axis">操作轴，允许使用从末尾计数的负索引。</param>
    /// <param name="rank">张量维数。</param>
    /// <returns>计算得到的整数结果。</returns>
    public static int Axis(int axis, int rank) => axis < 0 ? axis + rank : axis;
    /// <summary>
    /// 按连续行主序形状生成元素步长。
    /// </summary>
    /// <param name="shape">按行主序排列的目标维度数组。</param>
    /// <returns>按输出顺序排列的整数数组。</returns>
    public static int[] Strides(int[] shape)
    {
        int[] strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride = checked(stride * shape[i]);
        }

        return strides;
    }

    /// <summary>
    /// 合并兼容的输入形状，遇到不兼容维度时拒绝计算。
    /// </summary>
    /// <param name="shapes">待广播合并的输入形状。</param>
    /// <returns>按输出顺序排列的整数数组。</returns>
    public static int[] BroadcastShape(params int[][] shapes)
    {
        int rank = shapes.Max(x => x.Length);
        int[] result = Enumerable.Repeat(1, rank).ToArray();
        foreach (var shape in shapes)
        {
            for (int i = 0; i < shape.Length; ++i)
            {
                int d = rank - shape.Length + i;
                if (result[d] == 1)
                {
                    result[d] = shape[i];
                }
                else if (shape[i] != 1 && shape[i] != result[d])
                {
                    throw new ArgumentException("Broadcast shape mismatch");
                }
            }
        }

        return result;
    }

    /// <summary>
    /// 生成广播后的访问步长，扩展维度使用零步长。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="rank">张量维数。</param>
    /// <returns>按输出顺序排列的整数数组。</returns>
    private static int[] BroadcastStrides(Tensor x, int rank)
    {
        var source = Strides(x.Shape);
        int[] strides = new int[rank];
        for (int i = 0; i < x.Rank; ++i)
        {
            strides[rank - x.Rank + i] = x.Shape[i] == 1 ? 0 : source[i];
        }

        return strides;
    }

    /// <summary>
    /// 将输出线性索引映射到输入张量的实际元素偏移。
    /// </summary>
    /// <param name="index">从零开始的行、元素或任务索引。</param>
    /// <param name="shape">按行主序排列的目标维度数组。</param>
    /// <param name="strides">按元素计数的访问步长。</param>
    /// <returns>计算得到的整数结果。</returns>
    private static int Offset(int index, int[] shape, int[] strides)
    {
        int offset = 0;
        for (int i = shape.Length - 1; i >= 0; --i)
        {
            offset += index % shape[i] * strides[i];
            index /= shape[i];
        }

        return offset;
    }

    /// <summary>
    /// 逐元素应用一元运算并保留张量形状。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="operation">应用于输入元素的运算函数。</param>
    /// <param name="integer">是否将张量视为模型的整数形状或索引数据。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Unary(Tensor x, Func<float, float> operation, bool integer = false)
    {
        var y = Tensor.ForOverwrite(x.Shape, integer);
        for (int i = 0; i < x.Count; ++i)
        {
            y.Data[i] = operation(x.Data[i]);
        }

        return y;
    }

    /// <summary>
    /// 按广播规则逐元素应用二元运算。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <param name="operation">应用于输入元素的运算函数。</param>
    /// <param name="integer">是否将张量视为模型的整数形状或索引数据。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Binary(Tensor a, Tensor b, Func<float, float, float> operation, bool integer = false)
    {
        var shape = BroadcastShape(a.Shape, b.Shape);
        var y = Tensor.Zeros(shape, integer);
        if (a.Shape.SequenceEqual(b.Shape))
        {
            for (int i = 0; i < y.Count; ++i)
            {
                y.Data[i] = operation(a.Data[i], b.Data[i]);
            }
        }
        else if (a.Count == 1)
        {
            for (int i = 0; i < y.Count; ++i)
            {
                y.Data[i] = operation(a.Data[0], b.Data[i]);
            }
        }
        else if (b.Count == 1)
        {
            for (int i = 0; i < y.Count; ++i)
            {
                y.Data[i] = operation(a.Data[i], b.Data[0]);
            }
        }
        else
        {
            var sa = BroadcastStrides(a, shape.Length);
            var sb = BroadcastStrides(b, shape.Length);
            for (int i = 0; i < y.Count; ++i)
            {
                y.Data[i] = operation(a.Data[Offset(i, shape, sa)], b.Data[Offset(i, shape, sb)]);
            }
        }

        return y;
    }

    /// <summary>
    /// 按广播后的条件张量选择两个输入中的元素。
    /// </summary>
    /// <param name="condition">用于逐元素选择的条件张量。</param>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Where(Tensor condition, Tensor a, Tensor b)
    {
        var shape = BroadcastShape(condition.Shape, a.Shape, b.Shape);
        var y = Tensor.ForOverwrite(shape, a.Integer);
        var sc = BroadcastStrides(condition, shape.Length);
        var sa = BroadcastStrides(a, shape.Length);
        var sb = BroadcastStrides(b, shape.Length);
        if (y.Count == 0)
        {
            return y;
        }
        int width = shape.Length == 0 ? 1 : shape[^1];
        var outer = shape.Length == 0 ? [] : shape[..^1];
        int conditionWidth = shape.Length == 0 || sc[^1] == 0 ? 1 : width;
        int leftWidth = shape.Length == 0 || sa[^1] == 0 ? 1 : width;
        int rightWidth = shape.Length == 0 || sb[^1] == 0 ? 1 : width;
        for (int row = 0; row < y.Count / width; row++)
        {
            int conditionOffset = Offset(row, outer, sc), leftOffset = Offset(row, outer, sa), rightOffset = Offset(row, outer, sb);
            SelectBlock(condition.Data.AsSpan(conditionOffset, conditionWidth), a.Data.AsSpan(leftOffset, leftWidth), b.Data.AsSpan(rightOffset, rightWidth), y.Data.AsSpan(row * width, width));
        }

        return y;
    }

    /// <summary>
    /// 根据广播规则将张量扩展到目标形状。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="target">目标数组或目标形状。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Expand(Tensor x, int[] target)
    {
        var shape = BroadcastShape(x.Shape, target);
        var y = Tensor.Zeros(shape, x.Integer);
        var strides = BroadcastStrides(x, shape.Length);
        for (int i = 0; i < y.Count; ++i)
        {
            y.Data[i] = x.Data[Offset(i, shape, strides)];
        }

        return y;
    }

    /// <summary>
    /// 根据目标形状和零维规则重新解释张量存储。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="shape">按行主序排列的目标维度数组。</param>
    /// <param name="allowZero">是否将重塑形状中的零保留为零长度维度。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Reshape(Tensor x, int[] shape, bool allowZero = false)
    {
        shape = (int[])shape.Clone();
        int infer = -1, count = 1;
        for (int i = 0; i < shape.Length; ++i)
        {
            if (shape[i] == -1)
            {
                if (infer != -1)
                {
                    throw new ArgumentException("Multiple inferred dimensions");
                }

                infer = i;
            }
            else
            {
                if (shape[i] == 0 && !allowZero)
                {
                    shape[i] = x.Shape[i];
                }

                count = checked(count * shape[i]);
            }
        }

        if (infer != -1)
        {
            shape[infer] = x.Count / count;
        }

        return x.View(shape);
    }

    /// <summary>
    /// 在指定轴插入长度为一的维度，不复制张量数据。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="axes">需要参与操作的轴编号数组。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Unsqueeze(Tensor x, int[] axes)
    {
        int rank = x.Rank + axes.Length;
        var positions = axes.Select(a => Axis(a, rank)).ToHashSet();
        int source = 0;
        return x.View(Enumerable.Range(0, rank).Select(i => positions.Contains(i) ? 1 : x.Shape[source++]).ToArray());
    }

    /// <summary>
    /// 移除指定的单位维度，不复制张量数据。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="axes">需要参与操作的轴编号数组。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Squeeze(Tensor x, int[]? axes)
    {
        var positions = (axes ?? Enumerable.Range(0, x.Rank).Where(i => x.Shape[i] == 1).ToArray()).Select(a => Axis(a, x.Rank)).ToHashSet();
        if (positions.Any(i => x.Shape[i] != 1))
        {
            throw new ArgumentException("Cannot squeeze non-unit dimension");
        }

        return x.View(x.Shape.Where((_, i) => !positions.Contains(i)).ToArray());
    }

    /// <summary>
    /// 按维度置换重新排列连续张量元素。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="permutation">输出各轴对应的输入轴编号。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Transpose(Tensor x, int[]? permutation = null)
    {
        permutation ??= Enumerable.Range(0, x.Rank).Reverse().ToArray();
        var shape = permutation.Select(i => x.Shape[i]).ToArray();
        var source = Strides(x.Shape);
        var strides = permutation.Select(i => source[i]).ToArray();
        var y = Tensor.Zeros(shape, x.Integer);
        if (x.Rank >= 2 && permutation.Take(x.Rank - 2).SequenceEqual(Enumerable.Range(0, x.Rank - 2))
            && permutation[^2] == x.Rank - 1 && permutation[^1] == x.Rank - 2)
        {
            int rows = x.Shape[^2], columns = x.Shape[^1], batches = Tensor.Size(x.Shape[..^2]);
            for (int batch = 0; batch < batches; batch++)
            {
                int offset = batch * rows * columns;
                for (int rowBlock = 0; rowBlock < rows; rowBlock += 32)
                {
                    for (int columnBlock = 0; columnBlock < columns; columnBlock += 32)
                    {
                        for (int column = columnBlock; column < Math.Min(columns, columnBlock + 32); column++)
                        {
                            for (int row = rowBlock; row < Math.Min(rows, rowBlock + 32); row++)
                            {
                                y.Data[offset + column * rows + row] = x.Data[offset + row * columns + column];
                            }
                        }
                    }
                }
            }
            return y;
        }
        for (int i = 0; i < y.Count; ++i)
        {
            y.Data[i] = x.Data[Offset(i, shape, strides)];
        }

        return y;
    }

    /// <summary>
    /// 沿指定轴根据索引张量收集数据。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="indices">索引张量。</param>
    /// <param name="axis">操作轴，允许使用从末尾计数的负索引。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Gather(Tensor x, Tensor indices, int axis)
    {
        axis = Axis(axis, x.Rank);
        int inner = Tensor.Size(x.Shape.Skip(axis + 1)), outer = Tensor.Size(x.Shape.Take(axis)), dim = x.Shape[axis];
        var y = Tensor.Zeros(x.Shape.Take(axis).Concat(indices.Shape).Concat(x.Shape.Skip(axis + 1)).ToArray(), x.Integer);
        for (int o = 0; o < outer; ++o)
        {
            for (int i = 0; i < indices.Count; ++i)
            {
                int at = (int)indices.Data[i];
                if (at < 0)
                {
                    at += dim;
                }

                if ((uint)at >= dim)
                {
                    throw new IndexOutOfRangeException("Gather index");
                }

                x.Data.AsSpan((o * dim + at) * inner, inner).CopyTo(y.Data.AsSpan((o * indices.Count + i) * inner));
            }
        }

        return y;
    }

    /// <summary>
    /// 按逐元素索引沿指定轴收集数据。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="indices">索引张量。</param>
    /// <param name="axis">操作轴，允许使用从末尾计数的负索引。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor GatherElements(Tensor x, Tensor indices, int axis)
    {
        axis = Axis(axis, x.Rank);
        var strides = Strides(x.Shape);
        var y = Tensor.Zeros(indices.Shape, x.Integer);
        for (int i = 0; i < y.Count; ++i)
        {
            int n = i, at = 0;
            for (int d = x.Rank - 1; d >= 0; --d)
            {
                int coordinate = d == axis ? (int)indices.Data[i] : n % indices.Shape[d];
                n /= indices.Shape[d];
                if (coordinate < 0)
                {
                    coordinate += x.Shape[d];
                }

                at += coordinate * strides[d];
            }

            y.Data[i] = x.Data[at];
        }

        return y;
    }

    /// <summary>
    /// 根据多维索引元组收集对应的数据切片。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="indices">索引张量。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor GatherND(Tensor x, Tensor indices)
    {
        int k = indices.Shape[^1], tuples = Tensor.Size(indices.Shape[..^1]), inner = Tensor.Size(x.Shape.Skip(k));
        var strides = Strides(x.Shape);
        var y = Tensor.Zeros(indices.Shape[..^1].Concat(x.Shape.Skip(k)).ToArray(), x.Integer);
        for (int t = 0; t < tuples; ++t)
        {
            int offset = 0;
            for (int d = 0; d < k; ++d)
            {
                int at = (int)indices.Data[t * k + d];
                if (at < 0)
                {
                    at += x.Shape[d];
                }

                offset += at * strides[d];
            }

            x.Data.AsSpan(offset, inner).CopyTo(y.Data.AsSpan(t * inner));
        }

        return y;
    }

    /// <summary>
    /// 将更新切片写入索引指定的位置，保留未覆盖的数据。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="indices">索引张量。</param>
    /// <param name="updates">需要写入索引位置的数据张量。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor ScatterND(Tensor x, Tensor indices, Tensor updates)
    {
        int k = indices.Shape[^1], tuples = Tensor.Size(indices.Shape[..^1]), inner = Tensor.Size(x.Shape.Skip(k));
        var strides = Strides(x.Shape);
        var y = new Tensor((float[])x.Data.Clone(), x.Shape, x.Integer);
        for (int t = 0; t < tuples; ++t)
        {
            int offset = 0;
            for (int d = 0; d < k; ++d)
            {
                int at = (int)indices.Data[t * k + d];
                if (at < 0)
                {
                    at += x.Shape[d];
                }

                offset += at * strides[d];
            }

            updates.Data.AsSpan(t * inner, inner).CopyTo(y.Data.AsSpan(offset));
        }

        return y;
    }

    /// <summary>
    /// 按起止、轴与步幅截取张量。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="starts">各参与轴的起始索引。</param>
    /// <param name="ends">各参与轴的结束索引，不包含该位置。</param>
    /// <param name="axes">需要参与操作的轴编号数组。</param>
    /// <param name="steps">各参与轴的切片步幅。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Slice(Tensor x, Tensor starts, Tensor ends, int[]? axes, int[]? steps)
    {
        axes ??= Enumerable.Range(0, starts.Count).ToArray();
        steps ??= Enumerable.Repeat(1, starts.Count).ToArray();
        int[] shape = (int[])x.Shape.Clone(), begin = new int[x.Rank], increment = Enumerable.Repeat(1, x.Rank).ToArray();
        for (int i = 0; i < axes.Length; ++i)
        {
            int axis = Axis(axes[i], x.Rank), dim = x.Shape[axis], step = steps[i];
            if (step == 0)
            {
                throw new ArgumentException("Zero slice step");
            }

            double b = starts.Data[i], e = ends.Data[i];
            if (b < 0)
            {
                b += dim;
            }

            if (e < 0)
            {
                e += dim;
            }

            b = Math.Clamp(b, step > 0 ? 0 : -1, step > 0 ? dim : dim - 1);
            e = Math.Clamp(e, step > 0 ? 0 : -1, step > 0 ? dim : dim - 1);
            begin[axis] = (int)b;
            increment[axis] = step;
            shape[axis] = Math.Max(0, (int)Math.Ceiling((e - b) / step));
        }

        var y = Tensor.Zeros(shape, x.Integer);
        var original = Strides(x.Shape);
        int initial = begin.Zip(original, (a, b) => a * b).Sum();
        var strides = original.Zip(increment, (a, b) => a * b).ToArray();
        int inner = 1, first = shape.Length;
        while (first > 0 && strides[first - 1] == inner)
        {
            inner *= shape[--first];
        }
        if (inner > 1)
        {
            int[] outerShape = shape[..first], outerStrides = strides[..first];
            for (int block = 0; block < y.Count / inner; block++)
            {
                x.Data.AsSpan(initial + Offset(block, outerShape, outerStrides), inner).CopyTo(y.Data.AsSpan(block * inner, inner));
            }
            return y;
        }
        for (int i = 0; i < y.Count; ++i)
        {
            y.Data[i] = x.Data[initial + Offset(i, shape, strides)];
        }

        return y;
    }

    /// <summary>
    /// 沿指定轴连接形状兼容的张量。
    /// </summary>
    /// <param name="tensors">按连接顺序排列的输入张量。</param>
    /// <param name="axis">操作轴，允许使用从末尾计数的负索引。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Concat(Tensor[] tensors, int axis)
    {
        axis = Axis(axis, tensors[0].Rank);
        var shape = (int[])tensors[0].Shape.Clone();
        shape[axis] = tensors.Sum(x => x.Shape[axis]);
        var y = Tensor.Zeros(shape, tensors[0].Integer);
        int inner = Tensor.Size(shape.Skip(axis + 1)), outer = Tensor.Size(shape.Take(axis)), at = 0;
        for (int o = 0; o < outer; ++o)
        {
            foreach (var x in tensors)
            {
                int length = x.Shape[axis] * inner;
                x.Data.AsSpan(o * length, length).CopyTo(y.Data.AsSpan(at));
                at += length;
            }
        }

        return y;
    }

    /// <summary>
    /// 按各轴前后长度填充指定常量。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="pads">每个轴前后的填充长度。</param>
    /// <param name="value">待编码、填充或处理的值。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Pad(Tensor x, int[] pads, float value)
    {
        var shape = x.Shape.Select((n, i) => checked(n + pads[i] + pads[i + x.Rank])).ToArray();
        var y = Tensor.Zeros(shape, x.Integer);
        var strides = Strides(x.Shape);
        // 只填充最后一维时，按整行复制；同时处理负填充造成的裁剪。
        if (x.Rank > 0 && Enumerable.Range(0, x.Rank - 1).All(axis => pads[axis] == 0 && pads[axis + x.Rank] == 0))
        {
            if (BitConverter.SingleToInt32Bits(value) != 0)
            {
                Array.Fill(y.Data, value);
            }
            int width = x.Shape[^1], outputWidth = shape[^1], left = pads[x.Rank - 1];
            int sourceStart = Math.Max(0, -left), destinationStart = Math.Max(0, left);
            int count = Math.Max(0, Math.Min(width - sourceStart, outputWidth - destinationStart));
            if (count > 0)
            {
                int rows = Tensor.Size(x.Shape[..^1]);
                for (int row = 0; row < rows; row++)
                {
                    x.Data.AsSpan(row * width + sourceStart, count).CopyTo(y.Data.AsSpan(row * outputWidth + destinationStart, count));
                }
            }
            return y;
        }
        for (int i = 0; i < y.Count; ++i)
        {
            int n = i, at = 0;
            bool outside = false;
            for (int d = shape.Length - 1; d >= 0; --d)
            {
                int coordinate = n % shape[d] - pads[d];
                n /= shape[d];
                if (coordinate < 0 || coordinate >= x.Shape[d])
                {
                    outside = true;
                    break;
                }

                at += coordinate * strides[d];
            }

            y.Data[i] = outside ? value : x.Data[at];
        }

        return y;
    }

    /// <summary>
    /// 沿指定轴执行求和或均值等归约，并按配置保留维度。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="axes">需要参与操作的轴编号数组。</param>
    /// <param name="keep">是否保留被归约的单位维度。</param>
    /// <param name="op">归约运算名称。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Reduce(Tensor x, int[]? axes, bool keep, string op)
    {
        var set = (axes ?? Enumerable.Range(0, x.Rank).ToArray()).Select(a => Axis(a, x.Rank)).ToHashSet();
        var keptShape = x.Shape.Select((n, i) => set.Contains(i) ? 1 : n).ToArray();
        var y = Tensor.Zeros(keptShape, x.Integer);
        if (set.Count == 1)
        {
            int axis = set.Single(), width = x.Shape[axis];
            int inner = Tensor.Size(x.Shape[(axis + 1)..]), outer = Tensor.Size(x.Shape[..axis]);
            // 逐轴累加的先后顺序与通用实现一致，只消除每个元素的多维除余索引。
            for (int block = 0; block < outer; block++)
            {
                for (int index = 0; index < inner; index++)
                {
                    float sum = op == "ReduceMax" ? float.NegativeInfinity : 0;
                    for (int item = 0; item < width; item++)
                    {
                        float value = x.Data[(block * width + item) * inner + index];
                        sum = op == "ReduceMax" ? Math.Max(sum, value) : sum + value;
                    }
                    y.Data[block * inner + index] = op == "ReduceMean" ? sum / width : sum;
                }
            }
            return keep ? y : y.View(x.Shape.Where((_, i) => i != axis).ToArray());
        }
        var strides = Strides(keptShape);
        for (int d = 0; d < x.Rank; ++d)
        {
            if (set.Contains(d))
            {
                strides[d] = 0;
            }
        }

        if (op == "ReduceMax")
        {
            Array.Fill(y.Data, float.NegativeInfinity);
        }

        for (int i = 0; i < x.Count; ++i)
        {
            int at = Offset(i, x.Shape, strides);
            y.Data[at] = op == "ReduceMax" ? Math.Max(y.Data[at], x.Data[i]) : y.Data[at] + x.Data[i];
        }

        if (op == "ReduceMean")
        {
            int divisor = Tensor.Size(x.Shape.Where((_, i) => set.Contains(i)));
            for (int i = 0; i < y.Count; ++i)
            {
                y.Data[i] /= divisor;
            }
        }

        return keep ? y : y.View(x.Shape.Where((_, i) => !set.Contains(i)).ToArray());
    }

    /// <summary>
    /// 沿指定轴执行累积求和，支持排除当前位置和反向遍历。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="axis">操作轴，允许使用从末尾计数的负索引。</param>
    /// <param name="exclusive">是否排除当前位置的输入值。</param>
    /// <param name="reverse">是否从轴的末尾反向累计。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor CumSum(Tensor x, int axis, bool exclusive, bool reverse)
    {
        axis = Axis(axis, x.Rank);
        int inner = Tensor.Size(x.Shape.Skip(axis + 1)), outer = Tensor.Size(x.Shape.Take(axis)), length = x.Shape[axis];
        var y = Tensor.Zeros(x.Shape, x.Integer);
        for (int o = 0; o < outer; ++o)
        {
            for (int j = 0; j < inner; ++j)
            {
                float sum = 0;
                for (int i = 0; i < length; ++i)
                {
                    int at = (o * length + (reverse ? length - 1 - i : i)) * inner + j;
                    if (exclusive)
                    {
                        y.Data[at] = sum;
                        sum += x.Data[at];
                    }
                    else
                    {
                        sum += x.Data[at];
                        y.Data[at] = sum;
                    }
                }
            }
        }

        return y;
    }

    /// <summary>
    /// 沿指定轴执行数值稳定的归一化指数运算。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="axis">操作轴，允许使用从末尾计数的负索引。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Softmax(Tensor x, int axis)
    {
        axis = Axis(axis, x.Rank);
        int inner = Tensor.Size(x.Shape.Skip(axis + 1)), outer = Tensor.Size(x.Shape.Take(axis)), length = x.Shape[axis];
        var y = Tensor.Zeros(x.Shape);
        var temp = new float[length];
        for (int o = 0; o < outer; ++o)
        {
            for (int j = 0; j < inner; ++j)
            {
                for (int i = 0; i < length; ++i)
                {
                    temp[i] = x.Data[(o * length + i) * inner + j];
                }

                TensorMath.Softmax(temp);
                for (int i = 0; i < length; ++i)
                {
                    y.Data[(o * length + i) * inner + j] = temp[i];
                }
            }
        }

        return y;
    }

    /// <summary>
    /// 返回所有非零元素的多维坐标。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor NonZero(Tensor x)
    {
        int count = x.Data.Count(v => v != 0);
        var y = Tensor.Zeros([x.Rank, count], true);
        int at = 0;
        for (int i = 0; i < x.Count; ++i)
        {
            if (x.Data[i] != 0)
            {
                int n = i;
                for (int d = x.Rank - 1; d >= 0; --d)
                {
                    y.Data[d * count + at] = n % x.Shape[d];
                    n /= x.Shape[d];
                }

                ++at;
            }
        }

        return y;
    }
}
