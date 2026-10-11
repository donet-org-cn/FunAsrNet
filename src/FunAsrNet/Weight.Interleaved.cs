namespace FunAsrNet;

/// <summary>
/// 模型权重及其计算布局；加载完成后只允许在发布前准备布局。
/// </summary>
public sealed partial class Weight
{
    /// <summary>
    /// 权重是否已经转换为浮点交错通道布局。
    /// </summary>
    public bool FloatInterleaved
    {
        get; private set;
    }
    /// <summary>
    /// 权重是否已经转换为 Q8 交错通道布局。
    /// </summary>
    public bool QuantizedInterleaved
    {
        get; private set;
    }

    // 只允许在模型准备阶段、发布到推理线程之前调用。
    // 替换原数组，使进程仅常驻一份权重布局。
    /// <summary>
    /// 在发布模型前将权重改为交错布局，替换原数组以避免常驻双份权重。
    /// </summary>
    internal void PrepareInterleaved()
    {
        if (Shape.Length != 2 || FloatInterleaved || QuantizedInterleaved)
        {
            return;
        }

        int n = Columns, m = Rows;
        if (Values is not null)
        {
            var packed = new float[checked((m + 7) / 8 * 8 * n)];
            for (int row = 0; row < m; ++row)
            {
                for (int k = 0; k < n; ++k)
                {
                    packed[row / 8 * 8 * n + k * 8 + row % 8] = Values[row * n + k];
                }
            }

            Values = packed;
            FloatInterleaved = true;
        }
        else
        {
            if (n % 32 != 0)
            {
                throw new ArgumentException("Q8 interleaving requires complete blocks");
            }

            var packed = new sbyte[checked((m + 3) / 4 * 4 * n)];
            var scales = new float[packed.Length / 32];
            for (int row = 0; row < m; ++row)
            {
                for (int k = 0; k < n; ++k)
                {
                    packed[row / 4 * 4 * n + k / 4 * 16 + row % 4 * 4 + k % 4] = Quantized![row * n + k];
                }

                for (int block = 0; block < n / 32; ++block)
                {
                    scales[row / 4 * 4 * (n / 32) + block * 4 + row % 4] = Scales![row * (n / 32) + block];
                }
            }

            Quantized = packed;
            Scales = scales;
            QuantizedInterleaved = true;
        }
    }

    /// <summary>
    /// 从交错布局还原指定权重行到连续输出区。
    /// </summary>
    /// <param name="index">从零开始的行、元素或任务索引。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    private void CopyInterleavedRow(int index, Span<float> destination)
    {
        if (index < 0 || index >= Rows || destination.Length < Columns)
        {
            throw new ArgumentException("Invalid weight row");
        }

        if (FloatInterleaved)
        {
            for (int k = 0; k < Columns; ++k)
            {
                destination[k] = Values![index / 8 * 8 * Columns + k * 8 + index % 8];
            }
        }
        else
        {
            for (int k = 0; k < Columns; ++k)
            {
                destination[k] = Quantized![index / 4 * 4 * Columns + k / 4 * 16 + index % 4 * 4 + k % 4] * Scales![index / 4 * 4 * (Columns / 32) + k / 32 * 4 + index % 4];
            }
        }
    }
}
