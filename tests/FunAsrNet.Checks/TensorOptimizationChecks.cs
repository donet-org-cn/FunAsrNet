using FunAsrNet;

/// <summary>独立验证优化卷积和连续张量路径的分组、裁剪、广播及尾部语义。</summary>
internal static class TensorOptimizationChecks
{
    /// <summary>以数学参考值校验向量化路径，不依赖被测试内核生成预期结果。</summary>
    /// <param name="check">接收断言结果与检查名称的测试回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var random = new Random(121026);
        foreach (var shape in new[]
        {
            (Length: 4097, Groups: 1, Channels: 3, Outputs: 5, Kernel: 7, Stride: 1, Dilation: 3, Left: 5, Right: 9),
            (Length: 4103, Groups: 2, Channels: 6, Outputs: 10, Kernel: 4, Stride: 1, Dilation: 2, Left: 7, Right: 3),
            (Length: 33, Groups: 2, Channels: 6, Outputs: 10, Kernel: 4, Stride: 2, Dilation: 3, Left: 2, Right: 5),
            (Length: 4097, Groups: 3, Channels: 3, Outputs: 3, Kernel: 3, Stride: 1, Dilation: 1, Left: 2, Right: 4)
        })
        {
            int inGroup = shape.Channels / shape.Groups, outGroup = shape.Outputs / shape.Groups;
            var input = new Tensor(Enumerable.Range(0, 2 * shape.Channels * shape.Length).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray(), [2, shape.Channels, shape.Length]);
            var weight = new Tensor(Enumerable.Range(0, shape.Outputs * inGroup * shape.Kernel).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray(), [shape.Outputs, inGroup, shape.Kernel]);
            var node = new OnnxNode("grouped-tail", "Conv", [], [], new()
            {
                ["group"] = (long)shape.Groups,
                ["strides"] = new long[] { shape.Stride },
                ["dilations"] = new long[] { shape.Dilation },
                ["pads"] = new long[] { shape.Left, shape.Right }
            });
            foreach (bool withBias in new[] { false, true })
            {
                Tensor? bias = withBias ? new Tensor(Enumerable.Range(0, shape.Outputs).Select(index => index * 0.03f).ToArray(), [shape.Outputs]) : null;
                var actual = TensorOps.Conv1D(input, weight, bias, node, new TensorMath(4));
                double maximum = 0;
                for (int batch = 0; batch < 2; batch++)
                {
                    for (int output = 0; output < shape.Outputs; output++)
                    {
                        for (int time = 0; time < actual.Shape[2]; time++)
                        {
                            double expected = bias?.Data[output] ?? 0;
                            for (int channel = 0; channel < inGroup; channel++)
                            {
                                for (int tap = 0; tap < shape.Kernel; tap++)
                                {
                                    int at = time * shape.Stride + tap * shape.Dilation - shape.Left;
                                    if ((uint)at < (uint)shape.Length)
                                    {
                                        expected += (double)input.Data[(batch * shape.Channels + output / outGroup * inGroup + channel) * shape.Length + at] * weight.Data[(output * inGroup + channel) * shape.Kernel + tap];
                                    }
                                }
                            }
                            float value = actual.Data[(batch * shape.Outputs + output) * actual.Shape[2] + time];
                            check(float.IsFinite(value), "Grouped convolution produces finite values");
                            maximum = Math.Max(maximum, Math.Abs(expected - value));
                        }
                    }
                }
                check(maximum < 1e-5, "Grouped convolution independent oracle: batch, bias, dilation and channel/time tails");
            }
        }
        var matrix = new Tensor(Enumerable.Range(0, 2 * 33 * 35).Select(value => (float)value).ToArray(), [2, 33, 35]);
        var transposed = TensorOps.Transpose(matrix, [0, 2, 1]);
        for (int batch = 0; batch < 2; batch++)
        {
            for (int row = 0; row < 33; row++)
            {
                for (int column = 0; column < 35; column++)
                {
                    check(transposed.Data[(batch * 35 + column) * 33 + row] == matrix.Data[(batch * 33 + row) * 35 + column], "Tiled transpose batches and partial tiles");
                }
            }
        }
        var slice = TensorOps.Slice(matrix, new Tensor([2, 3], [2]), new Tensor([31, 32], [2]), [1, 2], [1, 1]);
        check(slice.Shape.SequenceEqual(new[] { 2, 29, 29 }), "Contiguous slice shape");
        for (int batch = 0; batch < 2; batch++)
        {
            for (int row = 0; row < 29; row++)
            {
                check(slice.Data.AsSpan((batch * 29 + row) * 29, 29).SequenceEqual(matrix.Data.AsSpan((batch * 33 + row + 2) * 35 + 3, 29)), "Contiguous slice copies only selected rows and columns");
            }
        }
        var padded = TensorOps.Pad(new Tensor([1, 2, 3, 4, 5, 6], [2, 3]), [0, -1, 0, 2], -0f);
        check(padded.Data.SequenceEqual(new float[] { 2, 3, 0, 0, 5, 6, 0, 0 }), "Last-axis padding crops before filling");
        check(BitConverter.SingleToInt32Bits(padded.Data[2]) == BitConverter.SingleToInt32Bits(-0f), "Padding retains signed zero");
        foreach (int axis in new[] { 0, 1, 2 })
        {
            foreach (string operation in new[] { "ReduceMean", "ReduceSum", "ReduceMax" })
            {
                var reduced = TensorOps.Reduce(matrix, [axis], true, operation);
                int inner = Tensor.Size(matrix.Shape[(axis + 1)..]), outer = Tensor.Size(matrix.Shape[..axis]);
                for (int block = 0; block < outer; block++)
                {
                    for (int index = 0; index < inner; index++)
                    {
                        float expected = operation == "ReduceMax" ? float.NegativeInfinity : 0;
                        for (int value = 0; value < matrix.Shape[axis]; value++)
                        {
                            float source = matrix.Data[(block * matrix.Shape[axis] + value) * inner + index];
                            expected = operation == "ReduceMax" ? Math.Max(expected, source) : expected + source;
                        }
                        expected = operation == "ReduceMean" ? expected / matrix.Shape[axis] : expected;
                        check(reduced.Data[block * inner + index] == expected, "Single-axis reduction preserves scalar accumulation order");
                    }
                }
            }
        }
    }
}
