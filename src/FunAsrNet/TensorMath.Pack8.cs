using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
public sealed partial class TensorMath
{
    /// <summary>
    /// 以八行一组重排激活，为矩阵小块计算提供连续读取。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="packed">用于保存打包激活的目标数组。</param>
    /// <param name="full">可按完整分组处理的行数。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    private static void PackEightRows(float[] input, float[] packed, int full, int width)
    {
        for (int t = 0; t < full; t += 8)
        {
            int k = 0;
            for (; k <= width - 4; k += 4)
            {
                TransposeFour(input, t * width + k, width, packed, t * width + k * 8);
                TransposeFour(input, (t + 4) * width + k, width, packed, t * width + k * 8 + 4);
            }

            for (; k < width; ++k)
            {
                for (int j = 0; j < 8; ++j)
                {
                    packed[t * width + k * 8 + j] = input[(t + j) * width + k];
                }
            }
        }
    }

    /// <summary>
    /// 将四行激活转置为连续交错布局。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="start">起始采样位置或输入偏移。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="dest">输出数组的起始元素偏移。</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void TransposeFour(float[] input, int start, int stride, float[] output, int dest)
    {
        var a = Vector128.LoadUnsafe(ref input[start]);
        var b = Vector128.LoadUnsafe(ref input[start + stride]);
        var c = Vector128.LoadUnsafe(ref input[start + 2 * stride]);
        var d = Vector128.LoadUnsafe(ref input[start + 3 * stride]);
        var ab0 = AdvSimd.Arm64.ZipLow(a, b).AsDouble();
        var ab1 = AdvSimd.Arm64.ZipHigh(a, b).AsDouble();
        var cd0 = AdvSimd.Arm64.ZipLow(c, d).AsDouble();
        var cd1 = AdvSimd.Arm64.ZipHigh(c, d).AsDouble();
        AdvSimd.Arm64.ZipLow(ab0, cd0).AsSingle().StoreUnsafe(ref output[dest]);
        AdvSimd.Arm64.ZipHigh(ab0, cd0).AsSingle().StoreUnsafe(ref output[dest + 8]);
        AdvSimd.Arm64.ZipLow(ab1, cd1).AsSingle().StoreUnsafe(ref output[dest + 16]);
        AdvSimd.Arm64.ZipHigh(ab1, cd1).AsSingle().StoreUnsafe(ref output[dest + 24]);
    }
}
