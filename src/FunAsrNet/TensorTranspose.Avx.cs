using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>按缓存块和八乘八寄存器块交换末两维，保留所有浮点位模式。</summary>
public static partial class TensorOps
{
    /// <summary>在三十二乘三十二缓存块内转置完整 SIMD 块，边缘只读取有效元素。</summary>
    /// <param name="input">连续的只读源矩阵。</param>
    /// <param name="output">独立且完整覆盖的目的矩阵。</param>
    /// <param name="rows">源矩阵行数。</param>
    /// <param name="columns">源矩阵列数。</param>
    /// <param name="batches">前导批维的矩阵数量。</param>
    private static void TransposeLastTwoAvx(float[] input, float[] output, int rows, int columns, int batches)
    {
        ref float source = ref MemoryMarshal.GetArrayDataReference(input);
        ref float destination = ref MemoryMarshal.GetArrayDataReference(output);
        for (int batch = 0; batch < batches; batch++)
        {
            int offset = batch * rows * columns;
            for (int rb = 0; rb < rows; rb += 32)
            {
                int rowEnd = Math.Min(rows, rb + 32);
                for (int cb = 0; cb < columns; cb += 32)
                {
                    int columnEnd = Math.Min(columns, cb + 32), row = rb;
                    for (; row + 7 < rowEnd; row += 8)
                    {
                        int column = cb;
                        for (; column + 7 < columnEnd; column += 8)
                        {
                            int at = offset + row * columns + column;
                            var r0 = Vector256.LoadUnsafe(ref source, (nuint)at);
                            var r1 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns));
                            var r2 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns * 2));
                            var r3 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns * 3));
                            var r4 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns * 4));
                            var r5 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns * 5));
                            var r6 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns * 6));
                            var r7 = Vector256.LoadUnsafe(ref source, (nuint)(at + columns * 7));
                            var t0 = Avx.UnpackLow(r0, r1);
                            var t1 = Avx.UnpackHigh(r0, r1);
                            var t2 = Avx.UnpackLow(r2, r3);
                            var t3 = Avx.UnpackHigh(r2, r3);
                            var t4 = Avx.UnpackLow(r4, r5);
                            var t5 = Avx.UnpackHigh(r4, r5);
                            var t6 = Avx.UnpackLow(r6, r7);
                            var t7 = Avx.UnpackHigh(r6, r7);
                            var s0 = Avx.Shuffle(t0, t2, 0x44);
                            var s1 = Avx.Shuffle(t0, t2, 0xee);
                            var s2 = Avx.Shuffle(t1, t3, 0x44);
                            var s3 = Avx.Shuffle(t1, t3, 0xee);
                            var s4 = Avx.Shuffle(t4, t6, 0x44);
                            var s5 = Avx.Shuffle(t4, t6, 0xee);
                            var s6 = Avx.Shuffle(t5, t7, 0x44);
                            var s7 = Avx.Shuffle(t5, t7, 0xee);
                            int dest = offset + column * rows + row;
                            Avx.Permute2x128(s0, s4, 0x20).StoreUnsafe(ref destination, (nuint)dest);
                            Avx.Permute2x128(s1, s5, 0x20).StoreUnsafe(ref destination, (nuint)(dest + rows));
                            Avx.Permute2x128(s2, s6, 0x20).StoreUnsafe(ref destination, (nuint)(dest + rows * 2));
                            Avx.Permute2x128(s3, s7, 0x20).StoreUnsafe(ref destination, (nuint)(dest + rows * 3));
                            Avx.Permute2x128(s0, s4, 0x31).StoreUnsafe(ref destination, (nuint)(dest + rows * 4));
                            Avx.Permute2x128(s1, s5, 0x31).StoreUnsafe(ref destination, (nuint)(dest + rows * 5));
                            Avx.Permute2x128(s2, s6, 0x31).StoreUnsafe(ref destination, (nuint)(dest + rows * 6));
                            Avx.Permute2x128(s3, s7, 0x31).StoreUnsafe(ref destination, (nuint)(dest + rows * 7));
                        }
                        for (; column < columnEnd; column++)
                        {
                            for (int r = row; r < row + 8; r++)
                            {
                                output[offset + column * rows + r] = input[offset + r * columns + column];
                            }
                        }
                    }
                    for (; row < rowEnd; row++)
                    {
                        for (int column = cb; column < columnEnd; column++)
                        {
                            output[offset + column * rows + row] = input[offset + row * columns + column];
                        }
                    }
                }
            }
        }
    }
}
