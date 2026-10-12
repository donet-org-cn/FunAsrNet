using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>独立验证采样相位卷积、双曲正切和并行激活的边界与数值误差。</summary>
internal static class CpuFollowupChecks
{
    /// <summary>用双精度散射卷积及标准库函数生成参考结果。</summary>
    /// <param name="check">断言结果及说明回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var random = new Random(12103);
        var math = new TensorMath(4);
        foreach (var shape in new[]
        {
            (Length: 17, Kernel: 7, Stride: 4, Left: 2, Right: 1, Extra: 3, Dilation: 1),
            (Length: 31, Kernel: 16, Stride: 8, Left: 4, Right: 4, Extra: 0, Dilation: 1),
            (Length: 33, Kernel: 8, Stride: 4, Left: 9, Right: 0, Extra: 2, Dilation: 1),
            (Length: 271, Kernel: 4, Stride: 4, Left: 0, Right: 0, Extra: 3, Dilation: 1),
            (Length: 16, Kernel: 3, Stride: 2, Left: 0, Right: 1, Extra: 1, Dilation: 2),
            (Length: 15, Kernel: 3, Stride: 1, Left: 1, Right: 2, Extra: 0, Dilation: 1)
        })
        {
            const int batches = 2, inputs = 3, outputs = 5;
            var input = new Tensor(Enumerable.Range(0, batches * inputs * shape.Length).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray(), [batches, inputs, shape.Length]);
            var weight = new Tensor(Enumerable.Range(0, inputs * outputs * shape.Kernel).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray(), [inputs, outputs, shape.Kernel]);
            var node = new OnnxNode("phase-oracle", "ConvTranspose", [], [], new()
            {
                ["strides"] = new long[] { shape.Stride }, ["pads"] = new long[] { shape.Left, shape.Right },
                ["dilations"] = new long[] { shape.Dilation }, ["output_padding"] = new long[] { shape.Extra }
            });
            foreach (bool withBias in new[] { false, true })
            {
                Tensor? bias = withBias ? new Tensor([0.1f, -0.2f, 0.3f, 0f, -0.1f], [outputs]) : null;
                var actual = TensorOps.ConvTranspose1D(input, weight, bias, node, math);
                int length = shape.Stride * (shape.Length - 1) + shape.Dilation * (shape.Kernel - 1) + 1 - shape.Left - shape.Right + shape.Extra;
                check(actual.Shape.SequenceEqual(new[] { batches, outputs, length }), "Transpose convolution independent output shape");
                var expected = new double[batches * outputs * length];
                for (int batch = 0; batch < batches; batch++)
                {
                    for (int output = 0; output < outputs; output++)
                    {
                        expected.AsSpan((batch * outputs + output) * length, length).Fill(bias?.Data[output] ?? 0);
                        for (int inputChannel = 0; inputChannel < inputs; inputChannel++)
                        {
                            for (int time = 0; time < shape.Length; time++)
                            {
                                for (int tap = 0; tap < shape.Kernel; tap++)
                                {
                                    int at = time * shape.Stride + tap * shape.Dilation - shape.Left;
                                    if ((uint)at < (uint)length)
                                    {
                                        expected[(batch * outputs + output) * length + at] += (double)input.Data[(batch * inputs + inputChannel) * shape.Length + time] * weight.Data[(inputChannel * outputs + output) * shape.Kernel + tap];
                                    }
                                }
                            }
                        }
                    }
                }
                for (int index = 0; index < expected.Length; index++)
                {
                    check(Math.Abs(expected[index] - actual.Data[index]) < 2e-6, "Transpose convolution double scatter oracle: batches, odd kernels, asymmetric padding and tails");
                }
            }
        }
        float[] points = Enumerable.Range(-10000, 20001).Select(index => index / 1000f)
            .Concat(Enumerable.Range(-149, 150).SelectMany(exponent => new[] { MathF.ScaleB(1, exponent), -MathF.ScaleB(1, exponent) }))
            .Concat(new[] { 0.125f, MathF.BitDecrement(0.125f), MathF.BitIncrement(0.125f), -0f, 0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity }).ToArray();
        var tanh = TensorOps.Tanh(new Tensor(points, [points.Length]));
        for (int index = 0; index < points.Length; index++)
        {
            float expected = (float)Math.Tanh(points[index]);
            check(float.IsNaN(expected) ? float.IsNaN(tanh.Data[index]) : Math.Abs(tanh.Data[index] - expected) <= 1.8e-7, "Vector tanh independent double reference including tiny and special values");
            if (points[index] == 0)
            {
                check(BitConverter.SingleToInt32Bits(points[index]) == BitConverter.SingleToInt32Bits(tanh.Data[index]), "Vector tanh preserves signed zero");
            }
        }
        var values = Enumerable.Range(0, 131079).Select(index => index % 17 == 0 ? -0f : (index - 65536) * 0.1f).ToArray();
        values[65535] = float.NaN;
        var leaky = TensorOps.LeakyRelu(new Tensor(values, [values.Length]), 0.03f, math);
        for (int index = 0; index < values.Length; index++)
        {
            float expected = values[index] >= 0 ? values[index] : values[index] * 0.03f;
            check(float.IsNaN(expected) ? float.IsNaN(leaky.Data[index]) : BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(leaky.Data[index]), "Parallel LeakyRelu exact reference and block tails");
        }
    }
}
