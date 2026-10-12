namespace FunAsrNet;

/// <summary>把通用多维填充转换为连续行的复制，避免逐元素计算高维坐标。</summary>
public static partial class TensorOps
{
    /// <summary>每个输出行只计算一次外层坐标，填充区与源数据区分别完整写入。</summary>
    /// <param name="input">保持只读的原始张量。</param>
    /// <param name="pads">各轴起始与末尾的填充长度，可为负以表示裁剪。</param>
    /// <param name="shape">已受检计算的输出形状。</param>
    /// <param name="value">保持原位模式的填充值。</param>
    /// <param name="math">可选同步线程调度器。</param>
    /// <returns>完整覆盖且没有借用输入存储的输出。</returns>
    private static Tensor PadRows(Tensor input, int[] pads, int[] shape, float value, TensorMath? math)
    {
        var output = Tensor.ForOverwrite(shape, input.Integer);
        if (output.Count == 0)
        {
            return output;
        }
        int rank = input.Rank, width = input.Shape[^1], outWidth = shape[^1];
        int sourceStart = Math.Max(0, -pads[rank - 1]), destinationStart = Math.Max(0, pads[rank - 1]);
        int count = Math.Max(0, Math.Min(width - sourceStart, outWidth - destinationStart));
        int rows = output.Count / outWidth, rowsPerTask = Math.Max(1, 65536 / outWidth);
        int[] strides = Strides(input.Shape);
        // <summary>写入一组互不重叠的输出行。</summary>
        // <param name="task">按连续输出行分组的任务编号。</param>
        void CopyRows(int task)
        {
            for (int row = task * rowsPerTask; row < Math.Min(rows, (task + 1) * rowsPerTask); row++)
            {
                var destination = output.Data.AsSpan(row * outWidth, outWidth);
                int index = row, offset = 0;
                bool outside = count == 0;
                for (int axis = rank - 2; axis >= 0 && !outside; axis--)
                {
                    int coordinate = index % shape[axis] - pads[axis];
                    index /= shape[axis];
                    if ((uint)coordinate >= (uint)input.Shape[axis])
                    {
                        outside = true;
                    }
                    else
                    {
                        offset += coordinate * strides[axis];
                    }
                }
                if (outside)
                {
                    destination.Fill(value);
                }
                else
                {
                    destination[..destinationStart].Fill(value);
                    input.Data.AsSpan(offset + sourceStart, count).CopyTo(destination.Slice(destinationStart, count));
                    destination[(destinationStart + count)..].Fill(value);
                }
            }
        }
        int tasks = (rows + rowsPerTask - 1) / rowsPerTask;
        if (math is not null && output.Count >= 65536)
        {
            math.For(tasks, CopyRows);
        }
        else
        {
            for (int task = 0; task < tasks; task++)
            {
                CopyRows(task);
            }
        }
        return output;
    }
}
