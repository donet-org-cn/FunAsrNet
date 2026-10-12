using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>用独立坐标参考验证转置和多维填充的边界与原始位模式。</summary>
internal static class LayoutChecks
{
    /// <summary>覆盖 SIMD 边界、批次、非有限值，以及多轴填充和裁剪。</summary>
    /// <param name="check">断言记录回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var random = new Random(12105);
        foreach (int rows in new[] { 0, 1, 7, 8, 9, 31, 32, 33, 65 })
        {
            foreach (int columns in new[] { 0, 3, 7, 8, 17, 32, 63 })
            {
                float[] data = Enumerable.Range(0, 2 * rows * columns).Select(_ => BitConverter.Int32BitsToSingle(random.Next(int.MinValue, int.MaxValue))).ToArray();
                var actual = TensorOps.Transpose(new Tensor(data, [2, rows, columns], true), [0, 2, 1]);
                check(actual.Integer && actual.Shape.SequenceEqual(new[] { 2, columns, rows }), "Transpose preserves shape and integer tag");
                for (int b = 0; b < 2; b++)
                {
                    for (int r = 0; r < rows; r++)
                    {
                        for (int c = 0; c < columns; c++)
                        {
                            check(BitConverter.SingleToInt32Bits(actual.Data[(b * columns + c) * rows + r]) == BitConverter.SingleToInt32Bits(data[(b * rows + r) * columns + c]), "Transpose exact bit permutation including vector and cache tails");
                        }
                    }
                }
            }
        }
        foreach (var test in new[]
        {
            (Shape: new[] { 2, 3, 9 }, Pads: new[] { 1, 2, 3, 0, 1, 4 }),
            (Shape: new[] { 2, 3, 9 }, Pads: new[] { -1, 1, -2, 2, -1, 3 }),
            (Shape: new[] { 2, 0, 7 }, Pads: new[] { 0, 1, 1, 0, 1, 2 }),
            (Shape: new[] { 7 }, Pads: new[] { -9, 5 }),
            (Shape: new[] { 2, 17, 4097 }, Pads: new[] { 1, -2, -1, 0, 2, 3 })
        })
        {
            int size = Tensor.Size(test.Shape);
            float[] data = Enumerable.Range(0, size).Select(i => i % 7 == 0 ? -0f : i * 0.25f).ToArray();
            foreach (float fill in new[] { -0f, -3.25f, float.NaN })
            {
                var actual = TensorOps.Pad(new Tensor(data, test.Shape), test.Pads, fill, new TensorMath(4));
                for (int i = 0; i < actual.Count; i++)
                {
                    int index = i, source = 0, stride = 1;
                    bool outside = false;
                    for (int axis = test.Shape.Length - 1; axis >= 0; axis--)
                    {
                        int coordinate = index % actual.Shape[axis] - test.Pads[axis];
                        index /= actual.Shape[axis];
                        outside |= coordinate < 0 || coordinate >= test.Shape[axis];
                        source += coordinate * stride;
                        stride *= test.Shape[axis];
                    }
                    float expected = outside ? fill : data[source];
                    check(BitConverter.SingleToInt32Bits(actual.Data[i]) == BitConverter.SingleToInt32Bits(expected), "Multiaxis pad exact independent coordinates and parallel tails");
                }
            }
        }
    }
}
