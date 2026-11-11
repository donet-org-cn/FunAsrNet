using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>原精度浮点短输入内核，在多个输出通道间共享激活加载。</summary>
public sealed partial class TensorMath
{
    /// <summary>同时累加四个输出行，保持原 AVX 点积的乘加与归约顺序。</summary>
    /// <param name="input">只读帧主序 FP32 输入。</param>
    /// <param name="frames">有效帧数。</param>
    /// <param name="weight">行主序只读 FP32 权重，F16 来源已按原值展开。</param>
    /// <param name="bias">可选输出通道偏置。</param>
    /// <param name="output">调用方提供的独立输出数组。</param>
    private void FloatRows4Avx(float[] input, int frames, Weight weight, Weight? bias, float[] output)
    {
        int width = weight.Columns, rows = weight.Rows;
        For((rows + 15) / 16, task =>
        {
            int end = Math.Min(rows, task * 16 + 16), row = task * 16;
            ref float x = ref MemoryMarshal.GetArrayDataReference(input);
            ref float w = ref MemoryMarshal.GetArrayDataReference(weight.Values!);
            for (; row + 3 < end; row += 4)
            {
                for (int time = 0; time < frames; time++)
                {
                    var a = Vector256<float>.Zero;
                    var b = a;
                    var c = a;
                    var d = a;
                    int column = 0;
                    for (; column <= width - 8; column += 8)
                    {
                        var value = Vector256.LoadUnsafe(ref x, (nuint)(time * width + column));
                        a += value * Vector256.LoadUnsafe(ref w, (nuint)(row * width + column));
                        b += value * Vector256.LoadUnsafe(ref w, (nuint)((row + 1) * width + column));
                        c += value * Vector256.LoadUnsafe(ref w, (nuint)((row + 2) * width + column));
                        d += value * Vector256.LoadUnsafe(ref w, (nuint)((row + 3) * width + column));
                    }
                    float sa = Vector128.Sum(a.GetLower() + a.GetUpper());
                    float sb = Vector128.Sum(b.GetLower() + b.GetUpper());
                    float sc = Vector128.Sum(c.GetLower() + c.GetUpper());
                    float sd = Vector128.Sum(d.GetLower() + d.GetUpper());
                    for (; column < width; column++)
                    {
                        float value = input[time * width + column];
                        sa += value * weight.Values![row * width + column];
                        sb += value * weight.Values![(row + 1) * width + column];
                        sc += value * weight.Values![(row + 2) * width + column];
                        sd += value * weight.Values![(row + 3) * width + column];
                    }
                    output[time * rows + row] = (bias?.Values?[row] ?? 0) + sa;
                    output[time * rows + row + 1] = (bias?.Values?[row + 1] ?? 0) + sb;
                    output[time * rows + row + 2] = (bias?.Values?[row + 2] ?? 0) + sc;
                    output[time * rows + row + 3] = (bias?.Values?[row + 3] ?? 0) + sd;
                }
            }
            for (; row < end; row++)
            {
                for (int time = 0; time < frames; time++)
                {
                    output[time * rows + row] = (bias?.Values?[row] ?? 0) + AudioKernels.Dot(weight.Values.AsSpan(row * width, width), input.AsSpan(time * width, width));
                }
            }
        });
    }
}
