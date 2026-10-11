using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>使用 AVX2/FMA 复用十六帧激活，保持行主序 FP32 权重。</summary>
public sealed partial class TensorMath
{
    /// <summary>仅打包激活面板，避免常驻第二份模型权重；尾部补零并只写有效输出。</summary>
    /// <param name="input">只读帧主序输入，长度为帧数乘内积宽度。</param>
    /// <param name="frames">实际有效帧数，至少为十六。</param>
    /// <param name="w">只读行主序 FP32 权重。</param>
    /// <param name="bias">可选输出通道偏置，在内积完成后叠加。</param>
    /// <param name="output">调用方提供的帧主序结果，各任务写入独立区间。</param>
    private void PackedFloatLinearAvx(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows, tiles = (frames + 15) / 16, blocks = (m + 3) / 4;
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
                int row = task / tiles * 4, tile = task % tiles, valid = Math.Min(4, m - row);
                ref float xp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(packed), tile * n * 16);
                ref float wp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(w.Values!), row * n);
                Span<float> sums = stackalloc float[64];
                sums.Clear();
                // 内积按三十二项分段，减少长链 FP32 累加误差；不改变模型或激活精度。
                for (int start = 0; start < n; start += 32)
                {
                    var a0 = Vector256<float>.Zero;
                    var a1 = a0;
                    var b0 = a0;
                    var b1 = a0;
                    var c0 = a0;
                    var c1 = a0;
                    var d0 = a0;
                    var d1 = a0;
                    for (int k = start; k < Math.Min(n, start + 32); k++)
                    {
                        var x0 = Vector256.LoadUnsafe(ref xp, (nuint)(k * 16));
                        var x1 = Vector256.LoadUnsafe(ref xp, (nuint)(k * 16 + 8));
                        var scale = Vector256.Create(Unsafe.Add(ref wp, k));
                        a0 = Fma.MultiplyAdd(x0, scale, a0);
                        a1 = Fma.MultiplyAdd(x1, scale, a1);
                        scale = Vector256.Create(valid > 1 ? Unsafe.Add(ref wp, n + k) : 0);
                        b0 = Fma.MultiplyAdd(x0, scale, b0);
                        b1 = Fma.MultiplyAdd(x1, scale, b1);
                        scale = Vector256.Create(valid > 2 ? Unsafe.Add(ref wp, 2 * n + k) : 0);
                        c0 = Fma.MultiplyAdd(x0, scale, c0);
                        c1 = Fma.MultiplyAdd(x1, scale, c1);
                        scale = Vector256.Create(valid > 3 ? Unsafe.Add(ref wp, 3 * n + k) : 0);
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
                    for (int channel = 0; channel < valid; channel++)
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
