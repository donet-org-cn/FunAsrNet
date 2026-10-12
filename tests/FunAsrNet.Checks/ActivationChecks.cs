using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>以双精度标量及独立数值积分验证批量激活的精度、特殊值和向量尾部。</summary>
internal static class ActivationChecks
{
    /// <summary>验证 sigmoid 的完整范围和 erf 的数学定义，不以优化函数生成预期值。</summary>
    /// <param name="check">断言和检查名称的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        float[] values = Enumerable.Range(-10000, 20001).Select(index => index / 100f).Concat(new[] { float.NegativeInfinity, float.PositiveInfinity, float.NaN, -0f, 0f }).ToArray();
        var actual = TensorOps.Sigmoid(new Tensor(values, [values.Length]));
        for (int i = 0; i < values.Length; i++)
        {
            float expected = (float)(1 / (1 + Math.Exp(-(double)values[i])));
            check(float.IsNaN(expected) ? float.IsNaN(actual.Data[i]) : Math.Abs(expected - actual.Data[i]) <= 1.5e-7f, "Vector sigmoid independent double reference and tails");
        }
        float[] points = Enumerable.Range(-40, 81).Select(index => index / 8f).ToArray();
        var erf = TensorOps.ErrorFunction(new Tensor(points, [points.Length]));
        for (int i = 0; i < points.Length; i++)
        {
            double step = Math.Abs(points[i]) / 1024, sum = 1 + Math.Exp(-(double)points[i] * points[i]);
            for (int k = 1; k < 1024; k++)
            {
                double t = k * step;
                sum += (k % 2 == 0 ? 2 : 4) * Math.Exp(-t * t);
            }
            double expected = Math.CopySign(sum * step / 3 * 2 / Math.Sqrt(Math.PI), points[i]);
            check(Math.Abs(expected - erf.Data[i]) <= 2e-7, "Vector erf independent Simpson integration");
        }
        var exceptional = TensorOps.ErrorFunction(new Tensor([float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, 1f, -1f, float.NaN, 2f], [9]));
        check(float.IsNaN(exceptional.Data[0]) && float.IsNaN(exceptional.Data[7]), "Vector erf propagates NaN");
        check(exceptional.Data[1] == 1 && exceptional.Data[2] == -1, "Vector erf infinite limits");
        check(BitConverter.SingleToInt32Bits(exceptional.Data[3]) >= 0 && BitConverter.SingleToInt32Bits(exceptional.Data[4]) < 0, "Vector erf preserves zero input sign");
    }
}
