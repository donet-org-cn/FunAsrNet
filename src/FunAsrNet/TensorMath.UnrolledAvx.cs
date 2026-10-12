using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>通过双项展开和预先准备的行地址减少 FP32 矩阵内核的整数地址计算。</summary>
public sealed partial class TensorMath
{
    /// <summary>仅处理完整四输出行块，保留现有十六帧布局、内积分段和 FMA 累加次序。</summary>
    /// <param name="input">只读帧主序 FP32 输入。</param>
    /// <param name="frames">输入有效帧数，尾帧在打包面板中补零。</param>
    /// <param name="w">行主序 FP32 权重，输出行数必须为四的倍数。</param>
    /// <param name="bias">可选输出偏置，在内积完成后叠加。</param>
    /// <param name="output">完整帧主序输出，各任务写入独立区间。</param>
    private void PackedFloatLinearUnrolledAvx(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows, tiles = (frames + 15) / 16, blocks = m / 4;
        if ((m & 3) != 0 || input.Length != checked(frames * n) || w.Values!.Length < checked(m * n)
            || output.Length != checked(frames * m) || (bias?.Values is not null && bias.Values.Length < m))
        {
            throw new ArgumentException("Invalid unrolled FP32 projection dimensions");
        }
        int segment = FloatAvxSegment(n);
        float[] packed = ArrayPool<float>.Shared.Rent(checked(tiles * 16 * n));
        try
        {
            For(tiles, tile =>
            {
                int first = tile * 16, valid = Math.Min(16, frames - first);
                for (int k = 0; k < n; k++)
                {
                    int offset = (tile * n + k) * 16;
                    for (int time = 0; time < valid; time++)
                    {
                        packed[offset + time] = input[(first + time) * n + k];
                    }
                    packed.AsSpan(offset + valid, 16 - valid).Clear();
                }
            });
            For(blocks * tiles, task =>
            {
                int row = task / tiles * 4, tile = task % tiles;
                ref float inputPanel = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(packed), tile * n * 16);
                ref float firstRow = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(w.Values!), row * n);
                Span<float> sums = stackalloc float[64];
                sums.Clear();
                for (int start = 0; start < n; start += segment)
                {
                    int count = Math.Min(segment, n - start), k = 0;
                    ref float x = ref Unsafe.Add(ref inputPanel, start * 16);
                    ref float w0 = ref Unsafe.Add(ref firstRow, start);
                    ref float w1 = ref Unsafe.Add(ref firstRow, n + start);
                    ref float w2 = ref Unsafe.Add(ref firstRow, n * 2 + start);
                    ref float w3 = ref Unsafe.Add(ref firstRow, n * 3 + start);
                    var a0 = Vector256<float>.Zero;
                    var a1 = a0;
                    var b0 = a0;
                    var b1 = a0;
                    var c0 = a0;
                    var c1 = a0;
                    var d0 = a0;
                    var d1 = a0;
                    for (; k + 1 < count; k += 2)
                    {
                        var x0 = Vector256.LoadUnsafe(ref x);
                        var x1 = Vector256.LoadUnsafe(ref x, 8);
                        var scale = Vector256.Create(w0);
                        a0 = Fma.MultiplyAdd(x0, scale, a0);
                        a1 = Fma.MultiplyAdd(x1, scale, a1);
                        scale = Vector256.Create(w1);
                        b0 = Fma.MultiplyAdd(x0, scale, b0);
                        b1 = Fma.MultiplyAdd(x1, scale, b1);
                        scale = Vector256.Create(w2);
                        c0 = Fma.MultiplyAdd(x0, scale, c0);
                        c1 = Fma.MultiplyAdd(x1, scale, c1);
                        scale = Vector256.Create(w3);
                        d0 = Fma.MultiplyAdd(x0, scale, d0);
                        d1 = Fma.MultiplyAdd(x1, scale, d1);
                        x0 = Vector256.LoadUnsafe(ref x, 16);
                        x1 = Vector256.LoadUnsafe(ref x, 24);
                        scale = Vector256.Create(Unsafe.Add(ref w0, 1));
                        a0 = Fma.MultiplyAdd(x0, scale, a0);
                        a1 = Fma.MultiplyAdd(x1, scale, a1);
                        scale = Vector256.Create(Unsafe.Add(ref w1, 1));
                        b0 = Fma.MultiplyAdd(x0, scale, b0);
                        b1 = Fma.MultiplyAdd(x1, scale, b1);
                        scale = Vector256.Create(Unsafe.Add(ref w2, 1));
                        c0 = Fma.MultiplyAdd(x0, scale, c0);
                        c1 = Fma.MultiplyAdd(x1, scale, c1);
                        scale = Vector256.Create(Unsafe.Add(ref w3, 1));
                        d0 = Fma.MultiplyAdd(x0, scale, d0);
                        d1 = Fma.MultiplyAdd(x1, scale, d1);
                        x = ref Unsafe.Add(ref x, 32);
                        w0 = ref Unsafe.Add(ref w0, 2);
                        w1 = ref Unsafe.Add(ref w1, 2);
                        w2 = ref Unsafe.Add(ref w2, 2);
                        w3 = ref Unsafe.Add(ref w3, 2);
                    }
                    if (k < count)
                    {
                        var x0 = Vector256.LoadUnsafe(ref x);
                        var x1 = Vector256.LoadUnsafe(ref x, 8);
                        var scale = Vector256.Create(w0);
                        a0 = Fma.MultiplyAdd(x0, scale, a0);
                        a1 = Fma.MultiplyAdd(x1, scale, a1);
                        scale = Vector256.Create(w1);
                        b0 = Fma.MultiplyAdd(x0, scale, b0);
                        b1 = Fma.MultiplyAdd(x1, scale, b1);
                        scale = Vector256.Create(w2);
                        c0 = Fma.MultiplyAdd(x0, scale, c0);
                        c1 = Fma.MultiplyAdd(x1, scale, c1);
                        scale = Vector256.Create(w3);
                        d0 = Fma.MultiplyAdd(x0, scale, d0);
                        d1 = Fma.MultiplyAdd(x1, scale, d1);
                    }
                    ref float previous = ref MemoryMarshal.GetReference(sums);
                    (Vector256.LoadUnsafe(ref previous) + a0).StoreUnsafe(ref previous);
                    (Vector256.LoadUnsafe(ref previous, 8) + a1).StoreUnsafe(ref previous, 8);
                    (Vector256.LoadUnsafe(ref previous, 16) + b0).StoreUnsafe(ref previous, 16);
                    (Vector256.LoadUnsafe(ref previous, 24) + b1).StoreUnsafe(ref previous, 24);
                    (Vector256.LoadUnsafe(ref previous, 32) + c0).StoreUnsafe(ref previous, 32);
                    (Vector256.LoadUnsafe(ref previous, 40) + c1).StoreUnsafe(ref previous, 40);
                    (Vector256.LoadUnsafe(ref previous, 48) + d0).StoreUnsafe(ref previous, 48);
                    (Vector256.LoadUnsafe(ref previous, 56) + d1).StoreUnsafe(ref previous, 56);
                }
                for (int time = 0; time < Math.Min(16, frames - tile * 16); time++)
                {
                    for (int channel = 0; channel < 4; channel++)
                    {
                        output[(tile * 16 + time) * m + row + channel] = sums[channel * 16 + time] + (bias?.Values?[row + channel] ?? 0);
                    }
                }
            });
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
        }
    }
}
