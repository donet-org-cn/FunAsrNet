using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>穷举 F16 的全部位模式，验证加载展开的位级精度、尾部和目标边界。</summary>
internal static class HalfLoadAvxChecks
{
    /// <summary>使用独立 .NET Half cast 建立 oracle，不忽略 NaN 载荷或负零。</summary>
    /// <param name="check">记录断言及其说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        var input = Enumerable.Range(0, 65536).Select(bits => BitConverter.UInt16BitsToHalf((ushort)bits)).ToArray();
        var original = input.Select(BitConverter.HalfToUInt16Bits).ToArray();
        var output = new float[input.Length];
        TensorMath.ConvertHalfToSingle(input, output);
        for (int index = 0; index < input.Length; index++)
        {
            check(BitConverter.SingleToInt32Bits(output[index]) == BitConverter.SingleToInt32Bits((float)input[index]), "F16 load exhaustive .NET cast bits including all NaN payloads");
            check(BitConverter.HalfToUInt16Bits(input[index]) == original[index], "F16 load preserves original weight bits");
        }
        foreach (int offset in new[] { 0, 1, 3, 7, 15 })
        {
            for (int length = 0; length <= 65; length++)
            {
                var source = Enumerable.Range(0, offset + length + 3).Select(index => BitConverter.UInt16BitsToHalf((ushort)((index * 3739 + 0x7bfe) & 0xffff))).ToArray();
                var destination = Enumerable.Repeat(-123.5f, length + 5).ToArray();
                TensorMath.ConvertHalfToSingle(source.AsSpan(offset, length), destination.AsSpan(2, length + 1));
                for (int index = 0; index < length; index++)
                {
                    check(BitConverter.SingleToInt32Bits(destination[index + 2]) == BitConverter.SingleToInt32Bits((float)source[offset + index]), "F16 load unaligned slice and every vector tail");
                }
                check(destination[0] == -123.5f && destination[1] == -123.5f && destination[length + 2] == -123.5f && destination[length + 3] == -123.5f && destination[length + 4] == -123.5f, "F16 load does not overwrite destination prefix or extra tail");
            }
        }
        float[] shortOutput = Enumerable.Repeat(-77f, 15).ToArray();
        bool rejected = false;
        try
        {
            TensorMath.ConvertHalfToSingle(input.AsSpan(0, 16), shortOutput);
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        check(rejected && shortOutput.All(value => value == -77f), "F16 load rejects short output before writing");
        TensorMath.ConvertHalfToSingle([], []);
        check(true, "F16 load supports empty spans");
    }
}
