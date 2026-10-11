using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
public sealed partial class TensorMath
{
    // 两个面板均按内积维优先排列，覆盖八个输出通道和八个时间位置。
    // 使用二十个向量寄存器，避免寄存器溢出及二次幂行步长的缓存冲突。
    /// <summary>
    /// 复用打包的八帧激活，计算交错浮点矩阵小块。
    /// </summary>
    /// <param name="weights">已经按相应内核布局准备的模型权重。</param>
    /// <param name="wo">权重数组的起始元素偏移。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="xo">输入数组的起始元素偏移。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="yo">输出数组的起始元素偏移。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="row">第一个输出通道的编号。</param>
    /// <param name="valid">当前尾块中实际有效的通道数量。</param>
    private static void FloatInterleavedPacked8(
        float[] weights,
        int wo,
        float[] input,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row,
        int valid)
    {
        ref float wp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(weights), wo);
        ref float xp = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(input), xo);
        var a0l = Vector128<float>.Zero;
        var a0h = a0l;
        var a1l = Vector128<float>.Zero;
        var a1h = a1l;
        var a2l = Vector128<float>.Zero;
        var a2h = a2l;
        var a3l = Vector128<float>.Zero;
        var a3h = a3l;
        var a4l = Vector128<float>.Zero;
        var a4h = a4l;
        var a5l = Vector128<float>.Zero;
        var a5h = a5l;
        var a6l = Vector128<float>.Zero;
        var a6h = a6l;
        var a7l = Vector128<float>.Zero;
        var a7h = a7l;
        for (int k = 0; k < n; ++k)
        {
            var wl = Vector128.LoadUnsafe(ref wp);
            var wh = Vector128.LoadUnsafe(ref wp, 4);
            var xl = Vector128.LoadUnsafe(ref xp);
            var xh = Vector128.LoadUnsafe(ref xp, 4);
            a0l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0l, wl, xl, 0);
            a0h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a0h, wh, xl, 0);
            a1l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1l, wl, xl, 1);
            a1h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a1h, wh, xl, 1);
            a2l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2l, wl, xl, 2);
            a2h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a2h, wh, xl, 2);
            a3l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3l, wl, xl, 3);
            a3h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a3h, wh, xl, 3);
            a4l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4l, wl, xh, 0);
            a4h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a4h, wh, xh, 0);
            a5l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5l, wl, xh, 1);
            a5h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a5h, wh, xh, 1);
            a6l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a6l, wl, xh, 2);
            a6h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a6h, wh, xh, 2);
            a7l = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a7l, wl, xh, 3);
            a7h = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(a7h, wh, xh, 3);
            wp = ref Unsafe.Add(ref wp, 8);
            xp = ref Unsafe.Add(ref xp, 8);
        }

        var bl = LoadBias(bias, row, valid);
        var bh = LoadBias(bias, row + 4, valid - 4);
        StoreChannels(a0l + bl, output, yo + 0 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a0h + bh, output, yo + 0 * stride + 4, valid - 4);
        }

        StoreChannels(a1l + bl, output, yo + 1 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a1h + bh, output, yo + 1 * stride + 4, valid - 4);
        }

        StoreChannels(a2l + bl, output, yo + 2 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a2h + bh, output, yo + 2 * stride + 4, valid - 4);
        }

        StoreChannels(a3l + bl, output, yo + 3 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a3h + bh, output, yo + 3 * stride + 4, valid - 4);
        }

        StoreChannels(a4l + bl, output, yo + 4 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a4h + bh, output, yo + 4 * stride + 4, valid - 4);
        }

        StoreChannels(a5l + bl, output, yo + 5 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a5h + bh, output, yo + 5 * stride + 4, valid - 4);
        }

        StoreChannels(a6l + bl, output, yo + 6 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a6h + bh, output, yo + 6 * stride + 4, valid - 4);
        }

        StoreChannels(a7l + bl, output, yo + 7 * stride, Math.Min(4, valid));
        if (valid > 4)
        {
            StoreChannels(a7h + bh, output, yo + 7 * stride + 4, valid - 4);
        }
    }
}
