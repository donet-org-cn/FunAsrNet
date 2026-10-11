using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
public sealed partial class TensorMath
{
    // 在 KV 缓存写入边界按 binary16 舍入，底层仍使用 binary32 存储。
    // 普通有限数值使用整数 SIMD，采用最近舍入、正中取偶。
    // 非规格化数、溢出、无穷和 NaN 使用 System.Half 的精确转换。
    /// <summary>
    /// 按 IEEE binary16 舍入后写回浮点存储，保持 F16 KV 缓存精度语义。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    internal static void RoundHalfCopy(ReadOnlySpan<float> input, Span<float> output)
    {
        if (output.Length < input.Length)
        {
            throw new ArgumentException("Half output too short");
        }

        int i = 0;
        if (Vector128.IsHardwareAccelerated)
        {
            for (; i <= input.Length - 4; i += 4)
            {
                var bits = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(input), (nuint)i).AsUInt32();
                var absolute = bits & Vector128.Create(0x7fffffffU);
                var normal = Vector128.GreaterThanOrEqual(absolute, Vector128.Create(0x38800000U)) & Vector128.LessThan(absolute, Vector128.Create(0x477fe000U));
                var rounded = (bits + Vector128.Create(0xfffU) + ((bits >> 13) & Vector128.Create(1U))) & Vector128.Create(0xffffe000U);
                rounded.AsSingle().StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output), (nuint)i);
                uint mask = normal.ExtractMostSignificantBits();
                if (mask != 15)
                {
                    for (int lane = 0; lane < 4; ++lane)
                    {
                        if ((mask & (1U << lane)) == 0)
                        {
                            output[i + lane] = (float)(Half)input[i + lane];
                        }
                    }
                }
            }
        }

        for (; i < input.Length; ++i)
        {
            output[i] = (float)(Half)input[i];
        }
    }
}
