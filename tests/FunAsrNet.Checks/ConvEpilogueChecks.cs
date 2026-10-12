using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>对照独立卷积及逐元素算子检查融合后的数值顺序、边界与输入所有权。</summary>
internal static class ConvEpilogueChecks
{
    /// <summary>覆盖 AVX 四通道、八通道和尾块，分组、膨胀、广播以及无 SIMD 回退。</summary>
    /// <param name="check">记录断言及说明的测试回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var cases = new (int Channels, int Outputs, int Length, int Kernel, int Dilation, int Groups, int Stride)[]
        {
            (4, 9, 129, 3, 1, 1, 1),
            (4, 191, 137, 5, 3, 1, 1),
            (4, 192, 129, 3, 3, 1, 1),
            (4, 193, 263, 7, 1, 1, 1),
            (6, 10, 137, 3, 2, 2, 1),
            (6, 384, 137, 3, 2, 2, 1),
            (3, 4, 129, 1, 1, 1, 1),
            (3, 8, 137, 2, 1, 1, 1),
            (3, 12, 129, 7, 3, 1, 1),
            (3, 9, 127, 3, 1, 1, 2),
            (4, 4, 129, 3, 1, 4, 1)
        };
        foreach (bool simd in new[] { true, false })
        {
            foreach (var item in cases)
            {
                using var scope = new TensorWorkspace().Enter();
                var math = new TensorMath(4, simd);
                var x = new Tensor(Enumerable.Range(0, 2 * item.Channels * item.Length).Select(index => (index % 23 - 11) * 0.03125f).ToArray(), [2, item.Channels, item.Length]);
                var w = new Tensor(Enumerable.Range(0, item.Outputs * item.Channels / item.Groups * item.Kernel).Select(index => (index % 17 - 8) * 0.0234375f).ToArray(), [item.Outputs, item.Channels / item.Groups, item.Kernel]);
                Tensor? bias = item.Outputs % 2 == 0 ? new Tensor(Enumerable.Range(0, item.Outputs).Select(index => (index % 5 - 2) * 0.015625f).ToArray(), [item.Outputs]) : null;
                int totalPadding = item.Dilation * (item.Kernel - 1);
                var node = new OnnxNode("conv-epilogue", "Conv", ["x", "w", "b"], ["y"], new()
                {
                    ["group"] = item.Groups,
                    ["strides"] = new long[] { item.Stride },
                    ["dilations"] = new long[] { item.Dilation },
                    ["pads"] = new long[] { totalPadding / 2, totalPadding - totalPadding / 2 }
                });
                var convolution = TensorOps.Conv1D(x, w, bias, node, math);
                var residual = new Tensor(Enumerable.Range(0, convolution.Count).Select(ResidualValue).ToArray(), convolution.Shape);
                var expectedAdd = TensorOps.Arithmetic(convolution, residual, "Add", math);
                var expectedLeaky = TensorOps.LeakyRelu(convolution, 0.1f, math);
                var expectedBoth = TensorOps.LeakyRelu(expectedAdd, 0.1f, math);
                float[] savedInput = (float[])x.Data.Clone(), savedWeight = (float[])w.Data.Clone(), savedResidual = (float[])residual.Data.Clone();
                Compare(TensorOps.Conv1DFused(x, w, bias, node, math, residual, null), expectedAdd, check, "Conv fused residual preserves separate FP32 addition");
                Compare(TensorOps.Conv1DFused(x, w, bias, node, math, null, 0.1f), expectedLeaky, check, "Conv fused LeakyRelu preserves separate activation");
                Compare(TensorOps.Conv1DFused(x, w, bias, node, math, residual, 0.1f), expectedBoth, check, "Conv fused residual then activation preserves operation order");
                var scalar = new Tensor([0.125f], []);
                Compare(TensorOps.Conv1DFused(x, w, bias, node, math, scalar, 0.1f),
                    TensorOps.LeakyRelu(TensorOps.Arithmetic(convolution, scalar, "Add", math), 0.1f, math), check, "Conv fused scalar broadcasting fallback");
                var channel = new Tensor(Enumerable.Range(0, item.Outputs).Select(index => (index % 7 - 3) * 0.0625f).ToArray(), [1, item.Outputs, 1]);
                Compare(TensorOps.Conv1DFused(x, w, bias, node, math, channel, null),
                    TensorOps.Arithmetic(convolution, channel, "Add", math), check, "Conv fused channel broadcasting fallback");
                check(x.Data.AsSpan().SequenceEqual(savedInput) && w.Data.AsSpan().SequenceEqual(savedWeight) && residual.Data.AsSpan().SequenceEqual(savedResidual), "Conv fused epilogue keeps caller input, weights and residual unchanged");
            }
        }
        CheckSpecialConvolution(check);
    }

    /// <summary>构造含负零、无穷与非数的残差，验证向量尾部和非数分支。</summary>
    /// <param name="index">当前元素的绝对索引。</param>
    /// <returns>用于有限值或非有限值语义检查的 FP32 残差。</returns>
    private static float ResidualValue(int index) => (index % 67) switch
    {
        0 => BitConverter.Int32BitsToSingle(unchecked((int)0x7fc12345)),
        1 => float.PositiveInfinity,
        2 => float.NegativeInfinity,
        3 => -0f,
        _ => (index % 19 - 9) * 0.046875f
    };

    /// <summary>检查非有限卷积结果及非有限激活系数，数值特殊值不作为优化前提。</summary>
    /// <param name="check">接收比较结果的断言回调。</param>
    private static void CheckSpecialConvolution(Action<bool, string> check)
    {
        using var scope = new TensorWorkspace().Enter();
        var math = new TensorMath(4);
        var x = new Tensor(Enumerable.Range(0, 129).Select(index => (index % 13) switch { 0 => float.NaN, 1 => float.PositiveInfinity, 2 => float.NegativeInfinity, 3 => -0f, _ => (float)(index % 7 - 3) }).ToArray(), [1, 1, 129]);
        var w = new Tensor([1, -1, 0, -0f], [4, 1, 1]);
        var bias = new Tensor([0, -0f, float.PositiveInfinity, float.NaN], [4]);
        var node = new OnnxNode("special-epilogue", "Conv", ["x", "w", "b"], ["y"], []);
        var convolution = TensorOps.Conv1D(x, w, bias, node, math);
        foreach (float alpha in new[] { 0.1f, -0f, float.PositiveInfinity, float.NaN })
        {
            var expected = TensorOps.LeakyRelu(convolution, alpha, math);
            Compare(TensorOps.Conv1DFused(x, w, bias, node, math, null, alpha), expected, check, "Conv fused activation retains NaN, infinity and signed-zero semantics");
        }
    }

    /// <summary>有限值和有符号零逐位比较，非数要求双方都保持非数。</summary>
    /// <param name="actual">融合实现结果。</param>
    /// <param name="expected">独立卷积加独立逐元素算子的结果。</param>
    /// <param name="check">接收断言的回调。</param>
    /// <param name="name">当前数值检查的说明。</param>
    private static void Compare(Tensor actual, Tensor expected, Action<bool, string> check, string name)
    {
        check(actual.Shape.SequenceEqual(expected.Shape), "Conv fused epilogue output shape");
        for (int index = 0; index < expected.Count; index++)
        {
            float a = actual.Data[index], b = expected.Data[index];
            check(float.IsNaN(a) && float.IsNaN(b) || BitConverter.SingleToInt32Bits(a) == BitConverter.SingleToInt32Bits(b), name);
        }
    }
}
