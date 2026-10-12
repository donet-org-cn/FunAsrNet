using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>在八个输出通道间共享时间向量的 x64 卷积。</summary>
public static partial class TensorOps
{
    /// <summary>使用八通道乘八采样点的寄存器块，减少输入重复加载。</summary>
    /// <param name="x">连续输入张量。</param>
    /// <param name="w">连续卷积权重。</param>
    /// <param name="bias">可选输出偏置。</param>
    /// <param name="y">完整覆盖的结果。</param>
    /// <param name="groups">卷积分组数。</param>
    /// <param name="dilation">核采样间隔。</param>
    /// <param name="padLeft">左侧零填充量。</param>
    /// <param name="math">同步工作线程组。</param>
    /// <param name="residual">偏置之后相加的可选同形残差。</param>
    /// <param name="leakyAlpha">残差相加后可选的 LeakyRelu 缩放系数。</param>
    private static void DirectConv1DEightAvx(Tensor x, Tensor w, Tensor? bias, Tensor y, int groups, int dilation, int padLeft, TensorMath math, Tensor? residual, float? leakyAlpha)
    {
        int channels = x.Shape[1], length = x.Shape[2], outputs = w.Shape[0], kernel = w.Shape[2], outLength = y.Shape[2];
        int inGroup = channels / groups, outGroup = outputs / groups, width = inGroup * kernel;
        int blocks = (outGroup + 7) / 8, tiles = (outLength + 255) / 256;
        math.For(x.Shape[0] * groups * blocks * tiles, task =>
        {
            int tile = task % tiles, block = task / tiles % blocks, group = task / (tiles * blocks) % groups;
            int batch = task / (tiles * blocks * groups), output = group * outGroup + block * 8;
            int valid = Math.Min(8, outGroup - block * 8), end = Math.Min(outLength, (tile + 1) * 256);
            ref float source = ref MemoryMarshal.GetArrayDataReference(x.Data);
            ref float weight = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(w.Data), output * width);
            int destinationOffset = (batch * outputs + output) * outLength;
            ref float destination = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(y.Data), destinationOffset);
            for (int time = tile * 256; time < end; time += 8)
            {
                int first = time - padLeft;
                bool interior = first >= 0 && first + 7 + (kernel - 1) * dilation < length;
                var a0 = Vector256<float>.Zero;
                var a1 = Vector256<float>.Zero;
                var a2 = Vector256<float>.Zero;
                var a3 = Vector256<float>.Zero;
                var a4 = Vector256<float>.Zero;
                var a5 = Vector256<float>.Zero;
                var a6 = Vector256<float>.Zero;
                var a7 = Vector256<float>.Zero;
                if (interior && valid == 8)
                {
                    for (int channel = 0; channel < inGroup; channel++)
                    {
                        int offset = (batch * channels + group * inGroup + channel) * length + first;
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            int at = offset + tap * dilation, k = channel * kernel + tap;
                            var value = Vector256.LoadUnsafe(ref source, (nuint)at);
                            a0 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 0 * width + k)), a0);
                            a1 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 1 * width + k)), a1);
                            a2 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 2 * width + k)), a2);
                            a3 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 3 * width + k)), a3);
                            a4 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 4 * width + k)), a4);
                            a5 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 5 * width + k)), a5);
                            a6 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 6 * width + k)), a6);
                            a7 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 7 * width + k)), a7);
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
                            var value = interior ? Vector256.LoadUnsafe(ref source, (nuint)at) : ReadPaddedTime(x.Data, offset - first, first + tap * dilation, length);
                            a0 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 0 * width + k)), a0);
                            if (valid > 1)
                            {
                                a1 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 1 * width + k)), a1);
                            }
                            if (valid > 2)
                            {
                                a2 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 2 * width + k)), a2);
                            }
                            if (valid > 3)
                            {
                                a3 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 3 * width + k)), a3);
                            }
                            if (valid > 4)
                            {
                                a4 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 4 * width + k)), a4);
                            }
                            if (valid > 5)
                            {
                                a5 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 5 * width + k)), a5);
                            }
                            if (valid > 6)
                            {
                                a6 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 6 * width + k)), a6);
                            }
                            if (valid > 7)
                            {
                                a7 = Fma.MultiplyAdd(value, Vector256.Create(Unsafe.Add(ref weight, 7 * width + k)), a7);
                            }
                        }
                    }
                }
                StoreConvTime(a0 + Vector256.Create(bias?.Data[output + 0] ?? 0), ref destination, 0 * outLength + time, end - time, residual, destinationOffset + 0 * outLength + time, leakyAlpha);
                if (valid > 1)
                {
                    StoreConvTime(a1 + Vector256.Create(bias?.Data[output + 1] ?? 0), ref destination, 1 * outLength + time, end - time, residual, destinationOffset + 1 * outLength + time, leakyAlpha);
                }
                if (valid > 2)
                {
                    StoreConvTime(a2 + Vector256.Create(bias?.Data[output + 2] ?? 0), ref destination, 2 * outLength + time, end - time, residual, destinationOffset + 2 * outLength + time, leakyAlpha);
                }
                if (valid > 3)
                {
                    StoreConvTime(a3 + Vector256.Create(bias?.Data[output + 3] ?? 0), ref destination, 3 * outLength + time, end - time, residual, destinationOffset + 3 * outLength + time, leakyAlpha);
                }
                if (valid > 4)
                {
                    StoreConvTime(a4 + Vector256.Create(bias?.Data[output + 4] ?? 0), ref destination, 4 * outLength + time, end - time, residual, destinationOffset + 4 * outLength + time, leakyAlpha);
                }
                if (valid > 5)
                {
                    StoreConvTime(a5 + Vector256.Create(bias?.Data[output + 5] ?? 0), ref destination, 5 * outLength + time, end - time, residual, destinationOffset + 5 * outLength + time, leakyAlpha);
                }
                if (valid > 6)
                {
                    StoreConvTime(a6 + Vector256.Create(bias?.Data[output + 6] ?? 0), ref destination, 6 * outLength + time, end - time, residual, destinationOffset + 6 * outLength + time, leakyAlpha);
                }
                if (valid > 7)
                {
                    StoreConvTime(a7 + Vector256.Create(bias?.Data[output + 7] ?? 0), ref destination, 7 * outLength + time, end - time, residual, destinationOffset + 7 * outLength + time, leakyAlpha);
                }
            }
        });
    }
}
