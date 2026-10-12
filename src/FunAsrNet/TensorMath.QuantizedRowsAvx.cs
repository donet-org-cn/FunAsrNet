using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>在输出通道间共享 Q8 激活加载与整数归约，保持逐块浮点累加顺序。</summary>
public sealed partial class TensorMath
{
    /// <summary>同时计算四个输出行，在 SIMD 通道内分别缩放并累加。</summary>
    /// <param name="q">已量化且不含负一百二十八的单帧输入。</param>
    /// <param name="scales">输入每三十二项的缩放值。</param>
    /// <param name="w">经过尺寸检查的行主序 Q8 权重。</param>
    /// <param name="bias">可选输出偏置。</param>
    /// <param name="output">具有完整输出行数的结果。</param>
    private void QuantizedRows4Avx(sbyte[] q, float[] scales, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows, blocks = n / 32;
        For((m + 15) / 16, task =>
        {
            ref sbyte weights = ref MemoryMarshal.GetArrayDataReference(w.Quantized!);
            ref sbyte input = ref MemoryMarshal.GetArrayDataReference(q);
            ref float ws = ref MemoryMarshal.GetArrayDataReference(w.Scales!);
            ref float xs = ref MemoryMarshal.GetArrayDataReference(scales);
            int end = Math.Min(m, (task + 1) * 16), row = task * 16;
            for (; row + 3 < end; row += 4)
            {
                var sum = Vector128<float>.Zero;
                for (int block = 0; block < blocks; block++)
                {
                    int at = row * n + block * 32, scaleAt = row * blocks + block;
                    var x = Vector256.LoadUnsafe(ref input, (nuint)(block * 32));
                    var a = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)at), x);
                    var b = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)(at + n)), x);
                    var c = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)(at + n * 2)), x);
                    var d = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)(at + n * 3)), x);
                    var ab = Ssse3.HorizontalAdd(a.GetLower() + a.GetUpper(), b.GetLower() + b.GetUpper());
                    var cd = Ssse3.HorizontalAdd(c.GetLower() + c.GetUpper(), d.GetLower() + d.GetUpper());
                    var dot = Vector128.ConvertToSingle(Ssse3.HorizontalAdd(ab, cd));
                    var scale = Vector128.Create(Unsafe.Add(ref ws, scaleAt), Unsafe.Add(ref ws, scaleAt + blocks), Unsafe.Add(ref ws, scaleAt + blocks * 2), Unsafe.Add(ref ws, scaleAt + blocks * 3));
                    sum += dot * scale * Vector128.Create(Unsafe.Add(ref xs, block));
                }
                if (bias?.Values is not null)
                {
                    sum = Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(bias.Values), (nuint)row) + sum;
                }
                sum.StoreUnsafe(ref MemoryMarshal.GetArrayDataReference(output), (nuint)row);
            }
            for (; row < end; row++)
            {
                float sum = 0;
                for (int block = 0; block < blocks; block++)
                {
                    var weight = Vector256.LoadUnsafe(ref weights, (nuint)(row * n + block * 32));
                    var x = Vector256.LoadUnsafe(ref input, (nuint)(block * 32));
                    sum += QuantizedBlockAvx(Avx2.Abs(weight), weight, x) * Unsafe.Add(ref ws, row * blocks + block) * Unsafe.Add(ref xs, block);
                }
                output[row] = (bias?.Values?[row] ?? 0) + sum;
            }
        });
    }

    /// <summary>计算八组四项整数点积，推迟归约以便多个输出行共享后续 SIMD 运算。</summary>
    /// <param name="weight">含原始符号的权重，可含负一百二十八。</param>
    /// <param name="input">只包含负一百二十七至一百二十七的激活。</param>
    /// <returns>每四项一组的精确整数部分和。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> QuantizedPartialAvx(Vector256<sbyte> weight, Vector256<sbyte> input)
    {
        var absolute = Avx2.Abs(weight);
        var signed = Avx2.Sign(input, weight);
        return AvxVnni.IsSupported
            ? AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, absolute, signed)
            : Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(absolute, signed), Vector256.Create((short)1));
    }
}
