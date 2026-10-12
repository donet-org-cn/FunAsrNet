using System.Runtime.Intrinsics.X86;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证 AVX2 Q8 字节乘加的饱和边界、带符号缩放和严格逐块数值一致性。</summary>
internal static class QuantizedAvxChecks
{
    /// <summary>对包含 -128 的权重和极值激活逐项比较独立标量结果，不放宽浮点误差。</summary>
    /// <param name="check">记录断言及其说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        if (!Avx2.IsSupported)
        {
            return;
        }
        foreach (int frames in new[] { 1, 3, 17 })
        {
            foreach (int width in new[] { 32, 96, 1024 })
            {
                foreach (int rows in new[] { 1, 3, 19, 32 })
                {
                    var values = Enumerable.Range(0, frames * width).Select(index => (index % 5) switch { 0 => -1f, 1 => 1f, 2 => 0f, 3 => -0.5f, _ => 0.1f }).ToArray();
                    var q = new sbyte[values.Length];
                    var inputScales = new float[values.Length / 32];
                    TensorMath.Quantize(values, q, inputScales);
                    var weight = new Weight("avx-extremes", [width, rows])
                    {
                        Quantized = Enumerable.Range(0, width * rows).Select(index => (index % 4) switch { 0 => (sbyte)-128, 1 => (sbyte)127, 2 => (sbyte)0, _ => (sbyte)-127 }).ToArray(),
                        Scales = Enumerable.Range(0, width * rows / 32).Select(index => index % 2 == 0 ? 0.013f : -0.007f).ToArray()
                    };
                    var bias = new Weight("bias", [rows]) { Values = Enumerable.Range(0, rows).Select(index => index * 0.125f).ToArray() };
                    float[] actual = new TensorMath(4).Linear(values, frames, weight, bias);
                    for (int time = 0; time < frames; time++)
                    {
                        for (int row = 0; row < rows; row++)
                        {
                            float sum = 0;
                            for (int start = 0; start < width; start += 32)
                            {
                                int dot = 0;
                                for (int k = 0; k < 32; k++)
                                {
                                    dot += weight.Quantized[row * width + start + k] * q[time * width + start + k];
                                }
                                sum += dot * weight.Scales[row * width / 32 + start / 32] * inputScales[time * width / 32 + start / 32];
                            }
                            float expected = bias.Values[row] + sum;
                            check(BitConverter.SingleToInt32Bits(actual[time * rows + row]) == BitConverter.SingleToInt32Bits(expected), "AVX Q8 exact scalar parity including -128 weights");
                        }
                    }
                }
            }
        }
        var invalid = new Weight("truncated", [32, 1]) { Quantized = new sbyte[16], Scales = [1] };
        bool rejected = false;
        try
        {
            new TensorMath(4).Linear(new float[32], 1, invalid);
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        check(rejected, "AVX Q8 rejects truncated weights before unsafe reads");
    }
}
