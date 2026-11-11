using System.Runtime.Intrinsics.X86;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>逐位比较六行矩阵块与已有四行实现，覆盖 K 分段和两个维度的尾部。</summary>
internal static class FloatTileChecks
{
    /// <summary>保持 FMA 次序、偏置、输入和权重不变，验证不同任务块大小。</summary>
    /// <param name="check">记录布尔断言和说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
        {
            return;
        }
        string? previous = Environment.GetEnvironmentVariable("FUNASRNET_AVX_ROWS6");
        var random = new Random(611120);
        try
        {
            foreach (int frames in new[] { 16, 17, 33 })
            {
                foreach (int width in new[] { 17, 127, 128, 129, 513 })
                {
                    foreach (int rows in new[] { 12, 13, 14, 15, 16, 17, 18, 20, 31 })
                    {
                        var input = Enumerable.Range(0, frames * width).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
                        var weight = new Weight("six-row-check", [width, rows])
                        {
                            Values = Enumerable.Range(0, rows * width).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray()
                        };
                        var bias = new Weight("six-row-bias", [rows])
                        {
                            Values = Enumerable.Range(0, rows).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray()
                        };
                        var math = new TensorMath(4);
                        Environment.SetEnvironmentVariable("FUNASRNET_AVX_ROWS6", "0");
                        var expected = math.Linear(input, frames, weight, bias);
                        Environment.SetEnvironmentVariable("FUNASRNET_AVX_ROWS6", "1");
                        var actual = math.Linear(input, frames, weight, bias);
                        for (int i = 0; i < actual.Length; i++)
                        {
                            check(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                                "Six-row FP32 preserves segmented FMA order and tail bounds");
                        }
                    }
                }
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("FUNASRNET_AVX_ROWS6", previous);
        }
    }
}
