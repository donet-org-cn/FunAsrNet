using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证短帧 FP32 投影与向量点积的累加顺序一致。</summary>
internal static class FloatRowsAvxChecks
{
    /// <summary>覆盖未对齐列数、尾行和一到三帧。</summary>
    /// <param name="check">接收断言和说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var random = new Random(20261112);
        foreach (int frames in new[] { 1, 2, 3 })
        {
            foreach (int width in new[] { 7, 16, 17, 32, 513, 1024 })
            {
                var input = Enumerable.Range(0, width * frames).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                var weight = new Weight("float-short", [width, 19]) { Values = Enumerable.Range(0, width * 19).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray() };
                var actual = new TensorMath(4).Linear(input, frames, weight);
                for (int time = 0; time < frames; time++)
                {
                    for (int row = 0; row < 19; row++)
                    {
                        float expected = 0f + AudioKernels.Dot(input.AsSpan(time * width, width), weight.Values.AsSpan(row * width, width));
                        check(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual[time * 19 + row]), "FP32 four-row kernel preserves SIMD dot order");
                    }
                }
            }
        }
    }
}
