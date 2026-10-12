using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>直接读取 NCL 布局的 AVX2 卷积，避免整段 im2col 和结果转置。</summary>
public static partial class TensorOps
{
    /// <summary>每次复用十六个时间点的输入，计算同组内四个输出通道。</summary>
    /// <param name="x">只读 NCL 输入，时间维连续。</param>
    /// <param name="w">只读输出通道、组内输入通道、核位置布局的 FP32 权重。</param>
    /// <param name="bias">可选输出通道偏置，在点积完成后叠加。</param>
    /// <param name="y">调用方拥有的输出，各任务写入独立的通道与时间块。</param>
    /// <param name="groups">输入与输出通道的分组数。</param>
    /// <param name="dilation">相邻卷积核位置的采样间隔，须为正数。</param>
    /// <param name="padLeft">输入左侧零填充的采样点数。</param>
    /// <param name="math">同步计算任务调度器。</param>
    private static void DirectConv1DAvx(Tensor x, Tensor w, Tensor? bias, Tensor y, int groups, int dilation, int padLeft, TensorMath math)
    {
        if (w.Shape[0] / groups >= 192 && Environment.GetEnvironmentVariable("FUNASRNET_AVX_CONV8") != "0"
            && Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_AVX_CONV_INNER") != "1")
        {
            DirectConv1DEightAvx(x, w, bias, y, groups, dilation, padLeft, math);
            return;
        }
        int channels = x.Shape[1], length = x.Shape[2], outputs = w.Shape[0], kernel = w.Shape[2], outLength = y.Shape[2];
        int inGroup = channels / groups, outGroup = outputs / groups, width = inGroup * kernel;
        int blocks = (outGroup + 3) / 4, tiles = (outLength + 255) / 256;
        bool specialize = Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_AVX_CONV_INNER") != "1";
        math.For(x.Shape[0] * groups * blocks * tiles, task =>
        {
            int tile = task % tiles, block = task / tiles % blocks, group = task / (tiles * blocks) % groups;
            int batch = task / (tiles * blocks * groups), output = group * outGroup + block * 4;
            int valid = Math.Min(4, outGroup - block * 4), end = Math.Min(outLength, (tile + 1) * 256);
            ref float source = ref MemoryMarshal.GetArrayDataReference(x.Data);
            ref float weight = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(w.Data), output * width);
            ref float destination = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(y.Data), (batch * outputs + output) * outLength);
            int time = tile * 256;
            for (; time < end; time += 16)
            {
                int first = time - padLeft;
                bool interior = first >= 0 && first + 15 + (kernel - 1) * dilation < length;
                var a0 = Vector256<float>.Zero;
                var a1 = a0;
                var b0 = a0;
                var b1 = a0;
                var c0 = a0;
                var c1 = a0;
                var d0 = a0;
                var d1 = a0;
                if (specialize && interior && valid == 4)
                {
                    // 模型长卷积的大多数时间点和输出块都是完整内部块；
                    // 把通道尾部及填充分支移到内积之外，保持原 FMA 顺序。
                    for (int channel = 0; channel < inGroup; channel++)
                    {
                        int offset = (batch * channels + group * inGroup + channel) * length + first;
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            int at = offset + tap * dilation, k = channel * kernel + tap;
                            var x0 = Vector256.LoadUnsafe(ref source, (nuint)at);
                            var x1 = Vector256.LoadUnsafe(ref source, (nuint)(at + 8));
                            var scale = Vector256.Create(Unsafe.Add(ref weight, k));
                            a0 = Fma.MultiplyAdd(x0, scale, a0);
                            a1 = Fma.MultiplyAdd(x1, scale, a1);
                            scale = Vector256.Create(Unsafe.Add(ref weight, width + k));
                            b0 = Fma.MultiplyAdd(x0, scale, b0);
                            b1 = Fma.MultiplyAdd(x1, scale, b1);
                            scale = Vector256.Create(Unsafe.Add(ref weight, 2 * width + k));
                            c0 = Fma.MultiplyAdd(x0, scale, c0);
                            c1 = Fma.MultiplyAdd(x1, scale, c1);
                            scale = Vector256.Create(Unsafe.Add(ref weight, 3 * width + k));
                            d0 = Fma.MultiplyAdd(x0, scale, d0);
                            d1 = Fma.MultiplyAdd(x1, scale, d1);
                        }
                    }
                }
                else
                {
                    for (int channel = 0; channel < inGroup; channel++)
                    {
                        int offset = (batch * channels + group * inGroup + channel) * length + first;
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            int at = offset + tap * dilation, k = channel * kernel + tap;
                            int channelStart = offset - first, position = first + tap * dilation;
                            var x0 = interior ? Vector256.LoadUnsafe(ref source, (nuint)at) : ReadPaddedTime(x.Data, channelStart, position, length);
                            var x1 = interior ? Vector256.LoadUnsafe(ref source, (nuint)(at + 8)) : ReadPaddedTime(x.Data, channelStart, position + 8, length);
                            var scale = Vector256.Create(Unsafe.Add(ref weight, k));
                            a0 = Fma.MultiplyAdd(x0, scale, a0);
                            a1 = Fma.MultiplyAdd(x1, scale, a1);
                            if (valid > 1)
                            {
                                scale = Vector256.Create(Unsafe.Add(ref weight, width + k));
                                b0 = Fma.MultiplyAdd(x0, scale, b0);
                                b1 = Fma.MultiplyAdd(x1, scale, b1);
                            }
                            if (valid > 2)
                            {
                                scale = Vector256.Create(Unsafe.Add(ref weight, 2 * width + k));
                                c0 = Fma.MultiplyAdd(x0, scale, c0);
                                c1 = Fma.MultiplyAdd(x1, scale, c1);
                            }
                            if (valid > 3)
                            {
                                scale = Vector256.Create(Unsafe.Add(ref weight, 3 * width + k));
                                d0 = Fma.MultiplyAdd(x0, scale, d0);
                                d1 = Fma.MultiplyAdd(x1, scale, d1);
                            }
                        }
                    }

                }
                var offsetBias = Vector256.Create(bias?.Data[output] ?? 0);
                StoreTime(a0 + offsetBias, ref destination, time, end - time);
                StoreTime(a1 + offsetBias, ref destination, time + 8, end - time - 8);
                if (valid > 1)
                {
                    offsetBias = Vector256.Create(bias?.Data[output + 1] ?? 0);
                    StoreTime(b0 + offsetBias, ref destination, outLength + time, end - time);
                    StoreTime(b1 + offsetBias, ref destination, outLength + time + 8, end - time - 8);
                }
                if (valid > 2)
                {
                    offsetBias = Vector256.Create(bias?.Data[output + 2] ?? 0);
                    StoreTime(c0 + offsetBias, ref destination, 2 * outLength + time, end - time);
                    StoreTime(c1 + offsetBias, ref destination, 2 * outLength + time + 8, end - time - 8);
                }
                if (valid > 3)
                {
                    offsetBias = Vector256.Create(bias?.Data[output + 3] ?? 0);
                    StoreTime(d0 + offsetBias, ref destination, 3 * outLength + time, end - time);
                    StoreTime(d1 + offsetBias, ref destination, 3 * outLength + time + 8, end - time - 8);
                }
            }
        });
    }

    /// <summary>读取边界处的八个时间点，在通道之外补零而不跨行加载。</summary>
    /// <param name="data">调用期间有效的输入数组。</param>
    /// <param name="offset">当前批次和通道的起始元素偏移。</param>
    /// <param name="time">通道内时间位置，允许落在填充区。</param>
    /// <param name="length">当前通道的实际采样点数。</param>
    /// <returns>含有效采样与零填充的八元素向量。</returns>
    private static Vector256<float> ReadPaddedTime(float[] data, int offset, int time, int length)
    {
        Span<float> padded = stackalloc float[8];
        padded.Clear();
        int first = Math.Max(0, -time), end = Math.Min(8, length - time);
        for (int index = first; index < end; index++)
        {
            padded[index] = data[offset + time + index];
        }
        return Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(padded));
    }

    /// <summary>写入完整时间向量或有效尾部，避免覆盖相邻通道。</summary>
    /// <param name="value">待写入的八元素计算结果。</param>
    /// <param name="destination">当前输出通道起始元素的引用。</param>
    /// <param name="offset">相对于引用的元素偏移。</param>
    /// <param name="count">当前时间块剩余的有效元素数，可小于等于零。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreTime(Vector256<float> value, ref float destination, int offset, int count)
    {
        if (count >= 8)
        {
            value.StoreUnsafe(ref destination, (nuint)offset);
        }
        else
        {
            for (int index = 0; index < count; index++)
            {
                Unsafe.Add(ref destination, offset + index) = value.GetElement(index);
            }
        }
    }
}
