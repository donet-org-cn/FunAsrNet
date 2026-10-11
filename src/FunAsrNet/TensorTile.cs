namespace FunAsrNet;

/// <summary>
/// 多发音人模型使用的张量平铺操作。
/// </summary>
public static partial class TensorOps
{
    /// <summary>
    /// 沿每个轴重复输入张量，按行主序生成独立输出。
    /// </summary>
    /// <param name="input">待重复的输入张量，方法不会修改其存储。</param>
    /// <param name="repeats">各轴非负重复次数，数量必须等于输入维数。</param>
    /// <returns>各轴尺寸乘以重复次数后的张量。</returns>
    public static Tensor Tile(Tensor input, int[] repeats)
    {
        if (repeats.Length != input.Rank || repeats.Any(repeat => repeat < 0))
        {
            throw new ArgumentException("平铺次数必须非负且与张量维数一致。");
        }

        int[] shape = input.Shape.Select((size, axis) => checked(size * repeats[axis])).ToArray();
        var output = Tensor.Zeros(shape, input.Integer);
        for (int index = 0; index < output.Count; index++)
        {
            int remaining = index, source = 0, stride = 1;
            for (int axis = shape.Length - 1; axis >= 0; axis--)
            {
                source += remaining % shape[axis] % input.Shape[axis] * stride;
                remaining /= shape[axis];
                stride *= input.Shape[axis];
            }

            output.Data[index] = input.Data[source];
        }

        return output;
    }
}
