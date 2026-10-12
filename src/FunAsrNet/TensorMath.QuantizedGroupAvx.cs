using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>共享行主序 Q8 多投影的输入量化与线程组，保持单投影的整数和浮点运算次序。</summary>
public sealed partial class TensorMath
{
    /// <summary>一次量化公共输入，并把所有投影的输出行分配到同一批独立任务。</summary>
    /// <param name="input">只读帧主序输入，所有投影具有相同的内积宽度。</param>
    /// <param name="frames">输入有效帧数。</param>
    /// <param name="weights">只读行主序 Q8 权重，不允许交错布局或不完整的量化块。</param>
    /// <returns>与权重顺序一致且互不共享存储的投影结果。</returns>
    private float[][] QuantizedLinearManyAvx(float[] input, int frames, Weight[] weights)
    {
        int n = weights[0].Columns, blocks = n / 32, totalTasks = 0, totalRows = 0;
        if (frames < 0 || n < 0 || input.Length != checked(frames * n))
        {
            throw new ArgumentException("Grouped projection dimensions differ");
        }
        var tasks = new int[weights.Length];
        for (int which = 0; which < weights.Length; which++)
        {
            var w = weights[which];
            int m = w.Rows;
            if (w.Columns != n || m < 0 || w.Quantized!.Length < checked(m * n)
                || w.Scales is null || w.Scales.Length < checked(m * blocks))
            {
                throw new ArgumentException("Invalid grouped row-major Q8 dimensions");
            }
            tasks[which] = checked((m + 15) / 16);
            totalTasks = checked(totalTasks + tasks[which]);
            totalRows = checked(totalRows + m);
        }
        long started = LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
        var outputs = new float[weights.Length][];
        for (int which = 0; which < weights.Length; which++)
        {
            outputs[which] = TensorWorkspace.Allocate(checked(frames * weights[which].Rows));
        }
        var q = ArrayPool<sbyte>.Shared.Rent(Math.Max(1, input.Length));
        try
        {
            var scales = ArrayPool<float>.Shared.Rent(Math.Max(1, input.Length / 32));
            try
            {
                Quantize(input, q.AsSpan(0, input.Length), scales.AsSpan(0, input.Length / 32));
                bool rows4 = frames == 1 && Environment.GetEnvironmentVariable("FUNASRNET_Q8_ROWS4") != "0";
                For(totalTasks, task =>
                {
                    int which = 0;
                    while (task >= tasks[which])
                    {
                        task -= tasks[which++];
                    }
                    int row = task * 16, end = Math.Min(weights[which].Rows, row + 16);
                    if (rows4)
                    {
                        QuantizedGroupedRows4Avx(q, scales, weights[which], outputs[which], row, end);
                    }
                    else
                    {
                        QuantizedGroupedFrames4Avx(q, scales, frames, weights[which], outputs[which], row, end);
                    }
                });
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scales);
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(q);
        }
        LinearMeasured?.Invoke($"Q8Group {frames}x{n}x{totalRows} {weights[0].Name}", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        return outputs;
    }

    /// <summary>计算一个任务的单帧输出行，以四路归约保持单投影内核的逐块运算次序。</summary>
    /// <param name="q">公共输入的量化字节，只有输入长度范围内的元素有效。</param>
    /// <param name="scales">公共输入每三十二项的缩放因子。</param>
    /// <param name="w">经过完整尺寸检查的行主序 Q8 权重。</param>
    /// <param name="output">本投影的独立输出存储。</param>
    /// <param name="row">本任务的第一个输出行。</param>
    /// <param name="end">本任务输出行的排他上界。</param>
    private static void QuantizedGroupedRows4Avx(sbyte[] q, float[] scales, Weight w, float[] output, int row, int end)
    {
        int n = w.Columns, blocks = n / 32;
        ref sbyte weights = ref MemoryMarshal.GetArrayDataReference(w.Quantized!);
        ref sbyte input = ref MemoryMarshal.GetArrayDataReference(q);
        ref float ws = ref MemoryMarshal.GetArrayDataReference(w.Scales!);
        ref float xs = ref MemoryMarshal.GetArrayDataReference(scales);
        for (; row + 3 < end; row += 4)
        {
            var sum = Vector128<float>.Zero;
            for (int block = 0; block < blocks; block++)
            {
                int at = row * n + block * 32, scaleAt = row * blocks + block;
                var x = Vector256.LoadUnsafe(ref input, (nuint)(block * 32));
                var a = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)at), x);
                var b = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)(at + n)), x);
                var c = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)(at + n * 2)), x);
                var d = QuantizedPartialAvx(Vector256.LoadUnsafe(ref weights, (nuint)(at + n * 3)), x);
                var ab = Ssse3.HorizontalAdd(a.GetLower() + a.GetUpper(), b.GetLower() + b.GetUpper());
                var cd = Ssse3.HorizontalAdd(c.GetLower() + c.GetUpper(), d.GetLower() + d.GetUpper());
                var dot = Vector128.ConvertToSingle(Ssse3.HorizontalAdd(ab, cd));
                var scale = Vector128.Create(Unsafe.Add(ref ws, scaleAt), Unsafe.Add(ref ws, scaleAt + blocks), Unsafe.Add(ref ws, scaleAt + blocks * 2), Unsafe.Add(ref ws, scaleAt + blocks * 3));
                sum += dot * scale * Vector128.Create(Unsafe.Add(ref xs, block));
            }
            sum.StoreUnsafe(ref MemoryMarshal.GetArrayDataReference(output), (nuint)row);
        }
        for (; row < end; row++)
        {
            float sum = 0;
            for (int block = 0; block < blocks; block++)
            {
                var weight = Vector256.LoadUnsafe(ref weights, (nuint)(row * n + block * 32));
                var x = Vector256.LoadUnsafe(ref input, (nuint)(block * 32));
                sum += QuantizedBlockAvx(Avx2.Abs(weight), weight, x) * Unsafe.Add(ref ws, row * blocks + block) * Unsafe.Add(ref xs, block);
            }
            output[row] = 0f + sum;
        }
    }

    /// <summary>计算一个任务的多帧输出行，四帧并行且只在帧尾使用标量归约。</summary>
    /// <param name="q">公共输入的量化字节，只有输入长度范围内的元素有效。</param>
    /// <param name="scales">公共输入每三十二项的缩放因子。</param>
    /// <param name="frames">输入有效帧数。</param>
    /// <param name="w">经过完整尺寸检查的行主序 Q8 权重。</param>
    /// <param name="output">本投影的独立帧主序输出存储。</param>
    /// <param name="row">本任务的第一个输出行。</param>
    /// <param name="end">本任务输出行的排他上界。</param>
    private static void QuantizedGroupedFrames4Avx(sbyte[] q, float[] scales, int frames, Weight w, float[] output, int row, int end)
    {
        int n = w.Columns, m = w.Rows, blocks = n / 32;
        ref sbyte weights = ref MemoryMarshal.GetArrayDataReference(w.Quantized!);
        ref sbyte input = ref MemoryMarshal.GetArrayDataReference(q);
        ref float ws = ref MemoryMarshal.GetArrayDataReference(w.Scales!);
        ref float xs = ref MemoryMarshal.GetArrayDataReference(scales);
        for (; row < end; row++)
        {
            int time = 0;
            for (; time + 3 < frames; time += 4)
            {
                var sum = Vector128<float>.Zero;
                for (int block = 0; block < blocks; block++)
                {
                    int at = block * 32;
                    var weight = Vector256.LoadUnsafe(ref weights, (nuint)(row * n + at));
                    var a = QuantizedPartialAvx(weight, Vector256.LoadUnsafe(ref input, (nuint)(time * n + at)));
                    var b = QuantizedPartialAvx(weight, Vector256.LoadUnsafe(ref input, (nuint)((time + 1) * n + at)));
                    var c = QuantizedPartialAvx(weight, Vector256.LoadUnsafe(ref input, (nuint)((time + 2) * n + at)));
                    var d = QuantizedPartialAvx(weight, Vector256.LoadUnsafe(ref input, (nuint)((time + 3) * n + at)));
                    var ab = Ssse3.HorizontalAdd(a.GetLower() + a.GetUpper(), b.GetLower() + b.GetUpper());
                    var cd = Ssse3.HorizontalAdd(c.GetLower() + c.GetUpper(), d.GetLower() + d.GetUpper());
                    var dot = Vector128.ConvertToSingle(Ssse3.HorizontalAdd(ab, cd));
                    var scale = Vector128.Create(Unsafe.Add(ref xs, time * blocks + block), Unsafe.Add(ref xs, (time + 1) * blocks + block), Unsafe.Add(ref xs, (time + 2) * blocks + block), Unsafe.Add(ref xs, (time + 3) * blocks + block));
                    sum += dot * Vector128.Create(Unsafe.Add(ref ws, row * blocks + block)) * scale;
                }
                sum = Vector128<float>.Zero + sum;
                output[time * m + row] = sum.GetElement(0);
                output[(time + 1) * m + row] = sum.GetElement(1);
                output[(time + 2) * m + row] = sum.GetElement(2);
                output[(time + 3) * m + row] = sum.GetElement(3);
            }
            for (; time < frames; time++)
            {
                float sum = 0;
                for (int block = 0; block < blocks; block++)
                {
                    var weight = Vector256.LoadUnsafe(ref weights, (nuint)(row * n + block * 32));
                    var activation = Vector256.LoadUnsafe(ref input, (nuint)(time * n + block * 32));
                    sum += QuantizedBlockAvx(Avx2.Abs(weight), weight, activation) * Unsafe.Add(ref ws, row * blocks + block) * Unsafe.Add(ref xs, time * blocks + block);
                }
                output[time * m + row] = 0f + sum;
            }
        }
    }
}
