using System.Runtime.Intrinsics.X86;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证展开的 FP32 微内核不改变既有分段 FMA 运算次序和尾帧结果。</summary>
internal static class UnrolledFloatAvxChecks
{
    /// <summary>在同一输入上切换新旧内核，并用独立标量分段 FMA 建立逐位预期结果。</summary>
    /// <param name="check">记录断言及其说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
        {
            return;
        }
        string? previousUnroll = Environment.GetEnvironmentVariable("FUNASRNET_AVX_UNROLL");
        string? previousSegment = Environment.GetEnvironmentVariable("FUNASRNET_AVX_SEGMENT128");
        try
        {
            var random = new Random(121005);
            var math = new TensorMath(4);
            foreach (string segmentMode in new[] { "0", "1" })
            {
                Environment.SetEnvironmentVariable("FUNASRNET_AVX_SEGMENT128", segmentMode);
                foreach (int width in new[] { 17, 31, 32, 127, 128, 129, 257, 511, 560, 1024, 2048 })
                {
                    foreach (int frames in new[] { 16, 17, 31, 33 })
                    {
                        foreach (int rows in new[] { 4, 12 })
                        {
                            var input = Enumerable.Range(0, width * frames).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                            var weights = new Weight("unrolled-fp32", [width, rows]) { Values = Enumerable.Range(0, width * rows).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray() };
                            var bias = new Weight("unrolled-bias", [rows]) { Values = Enumerable.Range(0, rows).Select(row => row * -0.0125f).ToArray() };
                            var savedInput = input.ToArray();
                            var savedWeights = weights.Values.ToArray();
                            Environment.SetEnvironmentVariable("FUNASRNET_AVX_UNROLL", "0");
                            var original = math.Linear(input, frames, weights, bias);
                            Environment.SetEnvironmentVariable("FUNASRNET_AVX_UNROLL", "1");
                            var actual = math.Linear(input, frames, weights, bias);
                            int segment = width <= 256 || segmentMode == "1" ? 128 : 32;
                            for (int time = 0; time < frames; time++)
                            {
                                for (int row = 0; row < rows; row++)
                                {
                                    float expected = 0;
                                    for (int start = 0; start < width; start += segment)
                                    {
                                        float part = 0;
                                        for (int k = start; k < Math.Min(width, start + segment); k++)
                                        {
                                            part = MathF.FusedMultiplyAdd(input[time * width + k], weights.Values[row * width + k], part);
                                        }
                                        expected += part;
                                    }
                                    expected += bias.Values[row];
                                    int at = time * rows + row;
                                    check(BitConverter.SingleToInt32Bits(actual[at]) == BitConverter.SingleToInt32Bits(expected), "Unrolled FP32 exact independent segmented FMA oracle");
                                    check(BitConverter.SingleToInt32Bits(actual[at]) == BitConverter.SingleToInt32Bits(original[at]), "Unrolled FP32 exact original kernel parity");
                                }
                            }
                            check(input.SequenceEqual(savedInput) && weights.Values.SequenceEqual(savedWeights), "Unrolled FP32 preserves caller input and model weights");
                        }
                    }
                }
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("FUNASRNET_AVX_UNROLL", previousUnroll);
            Environment.SetEnvironmentVariable("FUNASRNET_AVX_SEGMENT128", previousSegment);
        }
    }
}
