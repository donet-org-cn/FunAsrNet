namespace FunAsrNet;

/// <summary>按输出采样相位计算转置卷积，避免完整展开矩阵。</summary>
public static partial class TensorOps
{
    /// <summary>每个相位等价于短的一维卷积，最后交错写入时间轴。</summary>
    /// <param name="x">连续的批次、通道、时间输入。</param>
    /// <param name="w">输入通道、输出通道、核位置权重。</param>
    /// <param name="bias">可选的输出通道偏置。</param>
    /// <param name="y">完整覆盖的输出张量。</param>
    /// <param name="stride">相邻输入之间的输出采样间隔。</param>
    /// <param name="padLeft">左侧被裁剪的采样数。</param>
    /// <param name="math">当前同步计算线程组。</param>
    private static void PolyphaseConvTransposeAvx(Tensor x, Tensor w, Tensor? bias, Tensor y, int stride, int padLeft, TensorMath math)
    {
        int batches = x.Shape[0], inputs = x.Shape[1], outputs = w.Shape[1], kernel = w.Shape[2], length = y.Shape[2];
        for (int phase = 0; phase < Math.Min(stride, length); phase++)
        {
            int firstTap = (phase + padLeft) % stride, taps = (kernel - 1 - firstTap) / stride + 1;
            int phaseLength = (length - 1 - phase) / stride + 1;
            var weight = Tensor.ForOverwrite([outputs, inputs, taps]);
            for (int output = 0; output < outputs; output++)
            {
                for (int input = 0; input < inputs; input++)
                {
                    for (int tap = 0; tap < taps; tap++)
                    {
                        weight.Data[(output * inputs + input) * taps + tap] = w.Data[(input * outputs + output) * kernel + firstTap + (taps - 1 - tap) * stride];
                    }
                }
            }
            var result = Tensor.ForOverwrite([batches, outputs, phaseLength]);
            int padding = taps - 1 - (phase + padLeft) / stride;
            DirectConv1DAvx(x, weight, bias, result, 1, 1, padding, math);
            math.For(batches * outputs, channel =>
            {
                for (int time = 0; time < phaseLength; time++)
                {
                    y.Data[channel * length + time * stride + phase] = result.Data[channel * phaseLength + time];
                }
            });
            TensorWorkspace.Release(result.Data);
            TensorWorkspace.Release(weight.Data);
        }
    }
}
