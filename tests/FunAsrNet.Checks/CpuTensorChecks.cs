using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证大张量并行区间和广播选择的特殊值、末维尾部及只读输入所有权。</summary>
internal static class CpuTensorChecks
{
    /// <summary>独立计算标量参考，覆盖跨六万五千元素的并行切分以及条件广播。</summary>
    /// <param name="check">记录断言的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        const int count = 131079;
        var left = new Tensor(Enumerable.Range(0, count).Select(index => (index % 101 - 50) * 0.03125f).ToArray(), [count]);
        var right = new Tensor(Enumerable.Range(0, count).Select(index => (index % 17 + 1) * 0.125f).ToArray(), [count]);
        float[] saved = (float[])left.Data.Clone();
        foreach (string operation in new[] { "Add", "Sub", "Mul", "Div" })
        {
            var actual = TensorOps.Arithmetic(left, right, operation, new TensorMath(4));
            for (int i = 0; i < count; i++)
            {
                float a = left.Data[i], b = right.Data[i];
                float expected = operation switch
                {
                    "Add" => a + b,
                    "Sub" => a - b,
                    "Mul" => a * b,
                    _ => a / b
                };
                check(BitConverter.SingleToInt32Bits(actual.Data[i]) == BitConverter.SingleToInt32Bits(expected), "Parallel tensor arithmetic exact scalar parity across block tails");
            }
        }
        check(left.Data.AsSpan().SequenceEqual(saved), "Parallel arithmetic keeps shared input read-only");
        var broadcastInput = new Tensor(left.Data, [3, 43693]);
        var channelBias = new Tensor([1f, -2f, 0.125f], [3, 1]);
        var broadcast = TensorOps.Arithmetic(broadcastInput, channelBias, "Add", new TensorMath(4));
        for (int i = 0; i < count; i++)
        {
            check(broadcast.Data[i] == left.Data[i] + channelBias.Data[i / 43693], "Parallel channel broadcast crosses task and vector tails");
        }
        float[] conditions = [0f, -0f, float.NaN, 1f, -1f, float.PositiveInfinity, 0f, 1f, 0f];
        var condition = new Tensor(conditions, [1, 1, 9]);
        var input = new Tensor(Enumerable.Range(0, 3 * 9).Select(index => index % 2 == 0 ? -0f : index * 0.25f).ToArray(), [1, 3, 9]);
        var fallback = new Tensor([-7f], []);
        var result = TensorOps.Where(condition, input, fallback);
        for (int i = 0; i < result.Count; i++)
        {
            float expected = conditions[i % 9] != 0 ? input.Data[i] : -7f;
            check(BitConverter.SingleToInt32Bits(result.Data[i]) == BitConverter.SingleToInt32Bits(expected), "Where NaN condition, scalar broadcast and signed-zero selection");
        }
    }
}
