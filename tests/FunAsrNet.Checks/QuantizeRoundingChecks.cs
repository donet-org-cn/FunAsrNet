namespace FunAsrNet.Checks;

/// <summary>用独立标量参考检查量化的半整数边界、极小数及非有限块。</summary>
internal static class QuantizeRoundingChecks
{
    /// <summary>逐位比较缩放值并逐项比较量化字节，不能只比较最终识别文本。</summary>
    /// <param name="check">断言结果及说明回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            return;
        }
        var random = new Random(12104);
        var values = new List<float>();
        for (int value = -126; value <= 126; value++)
        {
            float halfway = value + 0.5f;
            values.AddRange(new[] { 127f, halfway, MathF.BitDecrement(halfway), MathF.BitIncrement(halfway) }.Concat(new float[28]));
        }
        for (int block = 0; block < 2048; block++)
        {
            float scale = MathF.ScaleB(1, block % 250 - 149);
            values.AddRange(Enumerable.Range(0, 32).Select(_ => (float)(random.NextDouble() - 0.5) * scale));
        }
        foreach (float special in new[] { 0f, -0f, float.Epsilon, float.MaxValue, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            values.AddRange(Enumerable.Range(0, 32).Select(index => index == 17 ? special : 0f));
        }
        float[] input = values.ToArray();
        var q = new sbyte[input.Length];
        var scales = new float[input.Length / 32];
        TensorMath.Quantize(input, q, scales);
        for (int block = 0; block < scales.Length; block++)
        {
            float maximum = 0;
            for (int index = 0; index < 32; index++)
            {
                maximum = Math.Max(maximum, Math.Abs(input[block * 32 + index]));
            }
            float scale = maximum / 127f, inverse = scale == 0 ? 0 : 1f / scale;
            check(float.IsNaN(scale) ? float.IsNaN(scales[block]) : BitConverter.SingleToInt32Bits(scale) == BitConverter.SingleToInt32Bits(scales[block]), "AVX quantization exact block scale");
            for (int index = 0; index < 32; index++)
            {
                sbyte expected = (sbyte)Math.Clamp((int)MathF.Round(input[block * 32 + index] * inverse, MidpointRounding.AwayFromZero), -127, 127);
                check(q[block * 32 + index] == expected, "AVX quantization exact away-from-zero rounding");
            }
        }
    }
}
