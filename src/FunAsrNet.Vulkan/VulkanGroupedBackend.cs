using System.Runtime.CompilerServices;

namespace FunAsrNet.Vulkan;

/// <summary>将 Q/K/V 或门控投影的只读原精度行合并，减少输入准备和设备同步。</summary>
public sealed unsafe partial class VulkanFloatBackend
{
    private readonly ConditionalWeakTable<Weight[], Weight> _groups = new();
    private readonly ConditionalWeakTable<Weight, object> _combinedScaleFirst = new();

    /// <summary>合并兼容的 Q8_0 或 FP32 投影，不支持的布局交由通用路径处理。</summary>
    /// <param name="input">共享的连续 FP32 输入数组。</param>
    /// <param name="frames">连续输入行数。</param>
    /// <param name="weights">稳定且只读的列数一致权重数组。</param>
    /// <param name="outputs">与原权重顺序一致的独立结果数组。</param>
    /// <returns>完整处理时为 true，不支持时为 false。</returns>
    public bool TryLinearMany(float[] input, int frames, Weight[] weights, out float[][]? outputs)
    {
        outputs = null;
        if (weights.Length >= 2 && frames > 0 && weights.All(weight => weight.Values is not null && weight.Quantized is null
            && !weight.FloatInterleaved && !weight.QuantizedInterleaved && !weight.IsTransient
            && weight.Columns == weights[0].Columns && weight.Columns > 0 && weight.Rows > 0))
        {
            return TryFloatLinearMany(input, frames, weights, out outputs);
        }
        if (weights.Length < 2 || frames <= 0 || weights.Any(weight => weight.Quantized is null || weight.Scales is null
            || weight.Columns != weights[0].Columns || weight.Columns % 32 != 0
            || weight.QuantizedInterleaved != weights[0].QuantizedInterleaved))
        {
            return false;
        }

        if ((long)frames * weights[0].Columns * weights.Sum(weight => weight.Rows) < MinimumOperations)
        {
            return false;
        }

        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var grouped = _groups.GetValue(weights, Combine);
            var result = TensorWorkspace.Allocate(checked(frames * grouped.Rows));
            if (!TryQuantizedLinear(input, frames, grouped, null, result))
            {
                TensorWorkspace.Release(result);
                return false;
            }

            var split = new float[weights.Length][];
            int offset = 0;
            for (int index = 0; index < weights.Length; index++)
            {
                int rows = weights[index].Rows;
                split[index] = TensorWorkspace.Allocate(checked(frames * rows));
                for (int time = 0; time < frames; time++)
                {
                    result.AsSpan(time * grouped.Rows + offset, rows).CopyTo(split[index].AsSpan(time * rows));
                }

                offset += rows;
            }

            TensorWorkspace.Release(result);
            outputs = split;
            return true;
        }
    }

    /// <summary>按原 Q8 字节和缩放拼接只读行，不解量化或重新量化权重。</summary>
    /// <param name="weights">列数一致的原始或交错布局 Q8 权重。</param>
    /// <returns>按原投影顺序合并的行主序权重。</returns>
    private Weight Combine(Weight[] weights)
    {
        int columns = weights[0].Columns, rows = weights.Sum(weight => weight.Rows), blocks = columns / 32;
        var result = new Weight("Vulkan Q8 group " + weights[0].Name, [columns, rows])
        {
            Quantized = new sbyte[checked(columns * rows)],
            Scales = new float[checked(rows * blocks)]
        };
        int offset = 0;
        foreach (var weight in weights)
        {
            for (int row = 0; row < weight.Rows; row++)
            {
                for (int column = 0; column < columns; column++)
                {
                    int source = weight.QuantizedInterleaved ? row / 4 * 4 * columns + column / 4 * 16 + row % 4 * 4 + column % 4 : row * columns + column;
                    result.Quantized[(offset + row) * columns + column] = weight.Quantized![source];
                }

                for (int block = 0; block < blocks; block++)
                {
                    int source = weight.QuantizedInterleaved ? row / 4 * 4 * blocks + block * 4 + row % 4 : row * blocks + block;
                    result.Scales[(offset + row) * blocks + block] = weight.Scales![source];
                }
            }

            offset += weight.Rows;
        }

        if (weights.All(weight => weight.QuantizedInterleaved))
        {
            _combinedScaleFirst.Add(result, new object());
        }

        return result;
    }
}
