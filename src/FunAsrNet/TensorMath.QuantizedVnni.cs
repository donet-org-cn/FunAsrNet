using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>四帧共用行主序权重的 AVX-VNNI 投影。</summary>
public sealed partial class TensorMath
{
    /// <summary>四帧同一量化块的交错激活、无符号权重偏移修正及原始缩放。</summary>
    private struct VnniInputBlock
    {
        internal Vector256<sbyte> K04, K15, K26, K37;
        internal Vector128<int> Offset;
        internal Vector128<float> Scales;
    }

    /// <summary>将四帧的量化字节按四项小组转置，在同一向量的两个半区分别处理量化块的前后十六项。</summary>
    /// <param name="q">按帧排列的有符号量化激活。</param>
    /// <param name="scales">每帧每三十二项对应的原始缩放因子。</param>
    /// <param name="frames">有效帧数，只打包完整的四帧组。</param>
    /// <param name="blocks">每帧的三十二项量化块数。</param>
    /// <param name="packed">由调用方租借的打包缓冲区。</param>
    private static void PackVnniInput(sbyte[] q, float[] scales, int frames, int blocks, VnniInputBlock[] packed)
    {
        ref int values = ref Unsafe.As<sbyte, int>(ref MemoryMarshal.GetArrayDataReference(q));
        var zeroPoint = Vector256.Create((byte)128);
        int index = 0;
        for (int time = 0; time + 3 < frames; time += 4)
        {
            int start = time * blocks;
            for (int k = 0; k < blocks; ++k)
            {
                int at = start + k;
                var a = Vector256.LoadUnsafe(ref values, (nuint)(at * 8));
                var b = Vector256.LoadUnsafe(ref values, (nuint)((at + blocks) * 8));
                var c = Vector256.LoadUnsafe(ref values, (nuint)((at + blocks * 2) * 8));
                var d = Vector256.LoadUnsafe(ref values, (nuint)((at + blocks * 3) * 8));
                var ab = Avx2.UnpackLow(a, b).AsInt64();
                var cd = Avx2.UnpackLow(c, d).AsInt64();
                var k04 = Avx2.UnpackLow(ab, cd).AsSByte();
                var k15 = Avx2.UnpackHigh(ab, cd).AsSByte();
                ab = Avx2.UnpackHigh(a, b).AsInt64();
                cd = Avx2.UnpackHigh(c, d).AsInt64();
                var k26 = Avx2.UnpackLow(ab, cd).AsSByte();
                var k37 = Avx2.UnpackHigh(ab, cd).AsSByte();

                // VNNI 计算 (w + 128) * q；每个激活块的修正项对所有权重行相同。
                var correction = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, zeroPoint, k04);
                correction = AvxVnni.MultiplyWideningAndAdd(correction, zeroPoint, k15);
                correction = AvxVnni.MultiplyWideningAndAdd(correction, zeroPoint, k26);
                correction = AvxVnni.MultiplyWideningAndAdd(correction, zeroPoint, k37);
                packed[index++] = new VnniInputBlock
                {
                    K04 = k04,
                    K15 = k15,
                    K26 = k26,
                    K37 = k37,
                    Offset = correction.GetLower() + correction.GetUpper(),
                    Scales = Vector128.Create(scales[at], scales[at + blocks], scales[at + blocks * 2], scales[at + blocks * 3])
                };
            }
        }
    }

    /// <summary>计算独立输出行，整数结果修正后再按原次序逐块缩放和累加。</summary>
    /// <param name="q">未打包的有符号量化激活，供帧尾使用。</param>
    /// <param name="scales">原始激活缩放因子，供帧尾使用。</param>
    /// <param name="packed">只读的完整四帧激活块。</param>
    /// <param name="frames">有效输入帧数。</param>
    /// <param name="w">尺寸已验证的行主序 Q8 权重。</param>
    /// <param name="bias">可选的输出偏置。</param>
    /// <param name="output">各行任务互不重叠的帧主序结果。</param>
    /// <param name="row">任务的第一个输出行。</param>
    /// <param name="end">任务输出行的排他上界。</param>
    private static void QuantizedVnniRows(sbyte[] q, float[] scales, VnniInputBlock[] packed, int frames, Weight w, Weight? bias, float[] output, int row, int end)
    {
        int n = w.Columns, m = w.Rows, blocks = n / 32;
        ref sbyte weights = ref MemoryMarshal.GetArrayDataReference(w.Quantized!);
        ref sbyte input = ref MemoryMarshal.GetArrayDataReference(q);
        ref float ws = ref MemoryMarshal.GetArrayDataReference(w.Scales!);
        ref VnniInputBlock panel = ref MemoryMarshal.GetArrayDataReference(packed);
        var flipSign = Vector256.Create(unchecked((int)0x80808080));
        for (; row < end; ++row)
        {
            ref sbyte weightRow = ref Unsafe.Add(ref weights, row * n);
            ref float weightScales = ref Unsafe.Add(ref ws, row * blocks);
            float b = bias?.Values?[row] ?? 0f;
            int time = 0, start = 0;
            for (; time + 3 < frames; time += 4, start += blocks)
            {
                var sum = Vector128<float>.Zero;
                ref VnniInputBlock group = ref Unsafe.Add(ref panel, start);
                for (int block = 0; block < blocks; ++block)
                {
                    ref readonly var x = ref Unsafe.Add(ref group, block);
                    var weight = Vector256.LoadUnsafe(ref weightRow, (nuint)(block * 32)).AsInt32() ^ flipSign;
                    var a = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, Avx2.Shuffle(weight, 0x00).AsByte(), x.K04);
                    var c = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, Avx2.Shuffle(weight, 0xaa).AsByte(), x.K26);
                    a = AvxVnni.MultiplyWideningAndAdd(a, Avx2.Shuffle(weight, 0x55).AsByte(), x.K15);
                    c = AvxVnni.MultiplyWideningAndAdd(c, Avx2.Shuffle(weight, 0xff).AsByte(), x.K37);
                    var pair = a + c;
                    var dot = Vector128.ConvertToSingle(pair.GetLower() + pair.GetUpper() - x.Offset);
                    sum += dot * Vector128.Create(Unsafe.Add(ref weightScales, block)) * x.Scales;
                }
                sum = Vector128.Create(b) + sum;
                output[time * m + row] = sum.GetElement(0);
                output[(time + 1) * m + row] = sum.GetElement(1);
                output[(time + 2) * m + row] = sum.GetElement(2);
                output[(time + 3) * m + row] = sum.GetElement(3);
            }
            for (; time < frames; ++time)
            {
                float sum = 0;
                for (int block = 0; block < blocks; ++block)
                {
                    int k = block * 32;
                    var weight = Vector256.LoadUnsafe(ref weightRow, (nuint)k);
                    var activation = Vector256.LoadUnsafe(ref input, (nuint)(time * n + k));
                    sum += QuantizedBlockAvx(Avx2.Abs(weight), weight, activation) * Unsafe.Add(ref weightScales, block) * scales[time * blocks + block];
                }
                output[time * m + row] = b + sum;
            }
        }
    }
}
