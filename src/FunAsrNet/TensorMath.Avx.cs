using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>使用 AVX2/FMA 复用十六帧激活，保持行主序 FP32 权重。</summary>
public sealed partial class TensorMath
{
    /// <summary>使用 AVX2 的字节乘加计算行主序 Q8 投影，保留逐三十二项缩放与累加顺序。</summary>
    /// <param name="input">只读帧主序浮点输入。</param>
    /// <param name="frames">有效输入帧数。</param>
    /// <param name="w">列数为三十二倍数的行主序 Q8 权重。</param>
    /// <param name="bias">可选输出通道偏置。</param>
    /// <param name="output">调用方持有的输出，各行任务写入独立区域。</param>
    private void QuantizedLinearAvx(float[] input, int frames, Weight w, Weight? bias, float[] output)
    {
        int n = w.Columns, m = w.Rows, blocks = n / 32;
        if ((n & 31) != 0 || w.Quantized!.Length < checked(m * n) || w.Scales is null || w.Scales.Length < checked(m * blocks))
        {
            throw new ArgumentException("Invalid row-major Q8 dimensions");
        }
        var q = new sbyte[input.Length];
        var scales = new float[input.Length / 32];
        Quantize(input, q, scales);
        For((m + 15) / 16, block =>
        {
            ref sbyte weights = ref MemoryMarshal.GetArrayDataReference(w.Quantized!);
            ref sbyte activations = ref MemoryMarshal.GetArrayDataReference(q);
            for (int row = block * 16; row < Math.Min(m, (block + 1) * 16); row++)
            {
                int time = 0;
                for (; time + 3 < frames; time += 4)
                {
                    float a = 0, b = 0, c = 0, d = 0;
                    for (int k = 0; k < n; k += 32)
                    {
                        var weight = Vector256.LoadUnsafe(ref weights, (nuint)(row * n + k));
                        var absolute = Avx2.Abs(weight);
                        float scale = w.Scales[row * blocks + k / 32];
                        a += QuantizedBlockAvx(absolute, weight, Vector256.LoadUnsafe(ref activations, (nuint)(time * n + k))) * scale * scales[time * blocks + k / 32];
                        b += QuantizedBlockAvx(absolute, weight, Vector256.LoadUnsafe(ref activations, (nuint)((time + 1) * n + k))) * scale * scales[(time + 1) * blocks + k / 32];
                        c += QuantizedBlockAvx(absolute, weight, Vector256.LoadUnsafe(ref activations, (nuint)((time + 2) * n + k))) * scale * scales[(time + 2) * blocks + k / 32];
                        d += QuantizedBlockAvx(absolute, weight, Vector256.LoadUnsafe(ref activations, (nuint)((time + 3) * n + k))) * scale * scales[(time + 3) * blocks + k / 32];
                    }
                    float offset = bias?.Values?[row] ?? 0;
                    output[time * m + row] = offset + a;
                    output[(time + 1) * m + row] = offset + b;
                    output[(time + 2) * m + row] = offset + c;
                    output[(time + 3) * m + row] = offset + d;
                }
                for (; time < frames; time++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k += 32)
                    {
                        var weight = Vector256.LoadUnsafe(ref weights, (nuint)(row * n + k));
                        var activation = Vector256.LoadUnsafe(ref activations, (nuint)(time * n + k));
                        // 激活量化严格限制到 [-127,127]，权重可含 -128。
                        // 取权重绝对值并把符号给激活，避免对 -128 求负溢出；
                        // 两项乘积和最大为 2*128*127=32512，不触发短整数饱和。
                        int dot = QuantizedBlockAvx(Avx2.Abs(weight), weight, activation);
                        sum += dot * w.Scales![row * blocks + k / 32] * scales[time * blocks + k / 32];
                    }
                    output[time * m + row] = (bias?.Values?[row] ?? 0) + sum;
                }
            }
        });
    }

    /// <summary>对三十二项作精确整数点积；激活来自限制为 [-127,127] 的量化器。</summary>
    /// <param name="absolute">权重绝对值，可包含字节值一百二十八。</param>
    /// <param name="weight">保留符号的原权重。</param>
    /// <param name="activation">不包含负一百二十八的量化激活。</param>
    /// <returns>完整整数点积，不发生短整数饱和。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int QuantizedBlockAvx(Vector256<byte> absolute, Vector256<sbyte> weight, Vector256<sbyte> activation)
    {
        if (AvxVnni.IsSupported)
        {
            return Vector256.Sum(AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, absolute, Avx2.Sign(activation, weight)));
        }
        var products = Avx2.MultiplyAddAdjacent(absolute, Avx2.Sign(activation, weight));
        return Vector256.Sum(Avx2.MultiplyAddAdjacent(products, Vector256.Create((short)1)));
    }

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
                // 短内积减少写回，长内积保持三十二项分段；不改变模型或激活精度。
                int segment = n <= 256 ? 128 : 32;
                for (int start = 0; start < n; start += segment)
                {
                    var a0 = Vector256<float>.Zero;
                    var a1 = a0;
                    var b0 = a0;
                    var b1 = a0;
                    var c0 = a0;
                    var c1 = a0;
                    var d0 = a0;
                    var d1 = a0;
                    for (int k = start; k < Math.Min(n, start + segment); k++)
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
