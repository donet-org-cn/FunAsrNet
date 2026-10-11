using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>一维卷积的连续时间向量内核，复用输入读取并避免展开整段音频。</summary>
public static partial class TensorOps
{
    /// <summary>沿连续时间位置计算 FP32 卷积，当前仅由单输出声码器末层调用。</summary>
    /// <param name="x">连续 NCL 输入张量，调用期间只读。</param>
    /// <param name="w">按输出通道、输入通道、卷积核排列的只读 FP32 权重。</param>
    /// <param name="bias">可选的输出通道偏置，最后叠加。</param>
    /// <param name="y">由调用方提供的 NCL 输出张量，各任务写入独立区间。</param>
    /// <param name="dilation">卷积核相邻采样点的正整数间隔。</param>
    /// <param name="padLeft">左侧零填充的采样点数。</param>
    /// <param name="math">用于分配独立计算任务的线程调度器。</param>
    private static void DirectConv1D(Tensor x, Tensor w, Tensor? bias, Tensor y, int dilation, int padLeft, TensorMath math)
    {
        int channels = x.Shape[1], length = x.Shape[2], outputs = w.Shape[0], kernel = w.Shape[2], outLength = y.Shape[2];
        int pairs = (outputs + 1) / 2, tiles = (outLength + 511) / 512;
        math.For(x.Shape[0] * pairs * tiles, task =>
        {
            int tile = task % tiles, pair = task / tiles % pairs, batch = task / (tiles * pairs);
            int output = pair * 2, end = Math.Min(outLength, (tile + 1) * 512);
            bool second = output + 1 < outputs;
            int y0 = (batch * outputs + output) * outLength, y1 = y0 + outLength;
            var b0 = Vector128.Create(bias?.Data[output] ?? 0);
            var b1 = Vector128.Create(second ? bias?.Data[output + 1] ?? 0 : 0);
            int time = tile * 512;
            for (; time < end;)
            {
                int source = time - padLeft;
                if (time + 15 < end && source >= 0 && source + 15 + (kernel - 1) * dilation < length)
                {
                    var a0 = Vector128<float>.Zero;
                    var a1 = a0;
                    var a2 = a0;
                    var a3 = a0;
                    var c0 = a0;
                    var c1 = a0;
                    var c2 = a0;
                    var c3 = a0;
                    for (int channel = 0; channel < channels; channel++)
                    {
                        int offset = (batch * channels + channel) * length + source;
                        int weight = (output * channels + channel) * kernel;
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            int at = offset + tap * dilation;
                            var v0 = Vector128.LoadUnsafe(ref x.Data[at]);
                            var v1 = Vector128.LoadUnsafe(ref x.Data[at + 4]);
                            var v2 = Vector128.LoadUnsafe(ref x.Data[at + 8]);
                            var v3 = Vector128.LoadUnsafe(ref x.Data[at + 12]);
                            var scale = Vector128.Create(w.Data[weight + tap]);
                            a0 = AdvSimd.FusedMultiplyAdd(a0, v0, scale);
                            a1 = AdvSimd.FusedMultiplyAdd(a1, v1, scale);
                            a2 = AdvSimd.FusedMultiplyAdd(a2, v2, scale);
                            a3 = AdvSimd.FusedMultiplyAdd(a3, v3, scale);
                            if (second)
                            {
                                var next = Vector128.Create(w.Data[weight + channels * kernel + tap]);
                                c0 = AdvSimd.FusedMultiplyAdd(c0, v0, next);
                                c1 = AdvSimd.FusedMultiplyAdd(c1, v1, next);
                                c2 = AdvSimd.FusedMultiplyAdd(c2, v2, next);
                                c3 = AdvSimd.FusedMultiplyAdd(c3, v3, next);
                            }
                        }
                    }

                    (a0 + b0).StoreUnsafe(ref y.Data[y0 + time]);
                    (a1 + b0).StoreUnsafe(ref y.Data[y0 + time + 4]);
                    (a2 + b0).StoreUnsafe(ref y.Data[y0 + time + 8]);
                    (a3 + b0).StoreUnsafe(ref y.Data[y0 + time + 12]);
                    if (second)
                    {
                        (c0 + b1).StoreUnsafe(ref y.Data[y1 + time]);
                        (c1 + b1).StoreUnsafe(ref y.Data[y1 + time + 4]);
                        (c2 + b1).StoreUnsafe(ref y.Data[y1 + time + 8]);
                        (c3 + b1).StoreUnsafe(ref y.Data[y1 + time + 12]);
                    }

                    time += 16;
                }
                else
                {
                    // 填充边界与不足十六点的尾部使用有界标量索引，不读取填充区之外。
                    for (int destination = output; destination < Math.Min(output + 2, outputs); destination++)
                    {
                        float sum = 0;
                        for (int channel = 0; channel < channels; channel++)
                        {
                            for (int tap = 0; tap < kernel; tap++)
                            {
                                int at = source + tap * dilation;
                                if ((uint)at < (uint)length)
                                {
                                    sum = MathF.FusedMultiplyAdd(x.Data[(batch * channels + channel) * length + at], w.Data[(destination * channels + channel) * kernel + tap], sum);
                                }
                            }
                        }

                        y.Data[(batch * outputs + destination) * outLength + time] = sum + (bias?.Data[destination] ?? 0);
                    }

                    time++;
                }
            }
        });
    }
}
