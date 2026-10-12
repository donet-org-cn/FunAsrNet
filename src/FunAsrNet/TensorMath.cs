using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace FunAsrNet;

/// <summary>
/// 行主序矩阵算子；Q8 路径按 32 个元素缩放，并在支持时使用 ARM SDOT。
/// </summary>
/// <param name="threads">算子使用的最大计算线程数。</param>
/// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
public sealed partial class TensorMath(int threads = 4, bool simd = true)
{
    private readonly ParallelOptions _parallel = new()
    {
        MaxDegreeOfParallelism = Math.Max(1, threads)
    };
    private readonly CpuKernelWorkers? _workers = threads > 1 && (AdvSimd.Arm64.IsSupported || System.Runtime.Intrinsics.X86.Avx2.IsSupported || Environment.GetEnvironmentVariable("FUNASRNET_SPIN_WORKERS") == "1") && Environment.GetEnvironmentVariable("FUNASRNET_SPIN_WORKERS") != "0" ? CpuKernelWorkers.Get(threads) : null;
    /// <summary>
    /// 当前计算实例是否允许使用硬件 SIMD 路径。
    /// </summary>
    public bool Simd { get; } = simd;
    /// <summary>
    /// 是否启用行主序权重的 ARM 专用打包内核；交错权重和 AVX2 路径按布局与硬件选择。
    /// </summary>
    public bool Optimized { get; set; } = Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_KERNELS") != "1";

    /// <summary>可选 FP32 后端；默认空值保持纯 C# CPU 推理，须在计算开始前配置。</summary>
    public IFloatComputeBackend? FloatBackend
    {
        get; set;
    }
    /// <summary>
    /// 可选的线性投影性能观察回调，参数为权重名称和耗时秒数。
    /// </summary>
    public Action<string, double>? LinearMeasured
    {
        get; set;
    }

    /// <summary>
    /// 在配置允许时把当前调用线程绑定到算子线程组对应的 CPU。
    /// </summary>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    internal IDisposable? PinCaller() => _workers?.PinCaller();
    /// <summary>
    /// 按线程分配索引区间，并等待该批索引上的操作全部完成。
    /// </summary>
    /// <param name="count">需要处理的元素或任务数量。</param>
    /// <param name="body">对分配到的索引执行的同步操作。</param>
    public void For(int count, Action<int> body)
    {
        if (_parallel.MaxDegreeOfParallelism == 1 || count < 4)
        {
            for (int i = 0; i < count; ++i)
            {
                body(i);
            }
        }
        else if (_workers is not null)
        {
            _workers.For(count, body);
        }
        else
        {
            Parallel.For(0, count, _parallel, body);
        }
    }

    /// <summary>
    /// 执行带可选偏置的线性投影，按权重布局选择向量化内核。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Linear(float[] input, int frames, Weight w, Weight? bias = null)
    {
        long started = LinearMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
        int n = w.Columns, m = w.Rows;
        if (input.Length != frames * n)
        {
            throw new ArgumentException($"{w.Name}: input shape mismatch");
        }

        float[] output = TensorWorkspace.Allocate(frames * m);
        if (w.Values is not null && FloatBackend?.TryLinear(input, frames, w, bias, output) == true)
        {
            // 后端负责完整计算与同步，Q8 路径不经此分支。
        }
        else if (w.Quantized is not null && FloatBackend is IQuantizedComputeBackend quantizedBackend
            && quantizedBackend.TryQuantizedLinear(input, frames, w, bias, output))
        {
            // 保留原始 Q8_0 与每三十二元素的激活量化语义。
        }
        else if (w.FloatInterleaved)
        {
            InterleavedFloatLinear(input, frames, w, bias, output);
        }
        else if (w.QuantizedInterleaved)
        {
            InterleavedQuantizedLinear(input, frames, w, bias, output);
        }
        else if (Simd && System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported
            && w.Values is not null && frames >= 16 && n >= 16 && m >= 4
            && Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_KERNELS") != "1")
        {
            PackedFloatLinearAvx(input, frames, w, bias, output);
        }
        else if (Optimized && Simd && AdvSimd.Arm64.IsSupported && w.Values is not null && frames >= 16 && n >= 128 && m >= 4)
        {
            PackedFloatLinear(input, frames, w, bias, output);
        }
        else if (Optimized && Simd && Dp.IsSupported && w.Quantized is not null && frames >= 2)
        {
            TiledQuantizedLinear(input, frames, w, bias, output);
        }
        else if (Optimized && Simd && System.Runtime.Intrinsics.X86.Avx2.IsSupported && w.Quantized is not null && (n & 31) == 0
            && Environment.GetEnvironmentVariable("FUNASRNET_AVX2_Q8") != "0")
        {
            QuantizedLinearAvx(input, frames, w, bias, output);
        }
        else if (w.Values is not null)
        {
            int rowsPerBlock = m < 128 ? 4 : 16;
            For(
                (m + rowsPerBlock - 1) / rowsPerBlock,
                block =>
                {
                    int end = Math.Min(m, (block + 1) * rowsPerBlock);
                    int row = block * rowsPerBlock;
                    if (Simd && Vector128.IsHardwareAccelerated)
                    {
                        for (; row + 3 < end; row += 4)
                        {
                            int t = 0;
                            for (; t + 3 < frames; t += 4)
                            {
                                Dot4x4(w.Values, row * n, input, t * n, output, t * m + row, n, m, bias?.Values, row);
                            }

                            for (; t < frames; ++t)
                            {
                                for (int j = 0; j < 4; ++j)
                                {
                                    output[t * m + row + j] = (bias?.Values?[row + j] ?? 0) + AudioKernels.Dot(w.Values.AsSpan((row + j) * n, n), input.AsSpan(t * n, n), Simd);
                                }
                            }
                        }
                    }

                    for (; row < end; ++row)
                    {
                        var weights = w.Values.AsSpan(row * n, n);
                        float b = bias?.Values?[row] ?? 0;
                        int t = 0;
                        if (Simd && Vector128.IsHardwareAccelerated)
                        {
                            for (; t + 3 < frames; t += 4)
                            {
                                Dot4(weights, input, t * n, output, t * m + row, n, m, b);
                            }
                        }

                        for (; t < frames; ++t)
                        {
                            output[t * m + row] = b + AudioKernels.Dot(weights, input.AsSpan(t * n, n), Simd);
                        }
                    }
                });
        }
        else
        {
            var q = new sbyte[input.Length];
            var scales = new float[input.Length / 32];
            Quantize(input, q, scales);
            For(
                (m + 31) / 32,
                block =>
                {
                    for (int row = block * 32; row < Math.Min(m, block * 32 + 32); ++row)
                    {
                        for (int t = 0; t < frames; ++t)
                        {
                            output[t * m + row] = (bias?.Values?[row] ?? 0) + QuantizedDot(w.Quantized!, w.Scales!, row * n, q, scales, t * n, n, Simd);
                        }
                    }
                });
        }

        LinearMeasured?.Invoke($"{(w.Values is null ? "Q8" : "F32")} {frames}x{n}x{m} {w.Name}", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        return output;
    }

    /// <summary>
    /// 执行向量乘加，在支持的硬件上使用融合乘加指令。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <param name="c">第三个输入操作数。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Mac(Vector128<float> a, Vector128<float> b, Vector128<float> c) => AdvSimd.IsSupported ? AdvSimd.FusedMultiplyAdd(a, b, c) : a + b * c;
    // 一次计算四个时间位置和四个输出通道，每次向量加载参与四次融合乘加。
    // 用十六个独立累加器隐藏 Cortex-A76 的乘加延迟，内层循环避免数组边界检查。
    /// <summary>
    /// 计算四个时间位置与四个输出通道的矩阵小块。
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
    private static void Dot4x4(
        float[] weights,
        int wo,
        float[] input,
        int xo,
        float[] output,
        int yo,
        int n,
        int stride,
        float[]? bias,
        int row)
    {
        ref float w = ref MemoryMarshal.GetArrayDataReference(weights);
        ref float x = ref MemoryMarshal.GetArrayDataReference(input);
        var a00 = Vector128<float>.Zero;
        var a01 = a00;
        var a02 = a00;
        var a03 = a00;
        var a10 = a00;
        var a11 = a00;
        var a12 = a00;
        var a13 = a00;
        var a20 = a00;
        var a21 = a00;
        var a22 = a00;
        var a23 = a00;
        var a30 = a00;
        var a31 = a00;
        var a32 = a00;
        var a33 = a00;
        int i = 0;
        for (; i <= n - 4; i += 4)
        {
            var w0 = Vector128.LoadUnsafe(ref w, (nuint)(wo + i));
            var w1 = Vector128.LoadUnsafe(ref w, (nuint)(wo + n + i));
            var w2 = Vector128.LoadUnsafe(ref w, (nuint)(wo + 2 * n + i));
            var w3 = Vector128.LoadUnsafe(ref w, (nuint)(wo + 3 * n + i));
            var x0 = Vector128.LoadUnsafe(ref x, (nuint)(xo + i));
            var x1 = Vector128.LoadUnsafe(ref x, (nuint)(xo + n + i));
            var x2 = Vector128.LoadUnsafe(ref x, (nuint)(xo + 2 * n + i));
            var x3 = Vector128.LoadUnsafe(ref x, (nuint)(xo + 3 * n + i));
            a00 = Mac(a00, w0, x0);
            a01 = Mac(a01, w1, x0);
            a02 = Mac(a02, w2, x0);
            a03 = Mac(a03, w3, x0);
            a10 = Mac(a10, w0, x1);
            a11 = Mac(a11, w1, x1);
            a12 = Mac(a12, w2, x1);
            a13 = Mac(a13, w3, x1);
            a20 = Mac(a20, w0, x2);
            a21 = Mac(a21, w1, x2);
            a22 = Mac(a22, w2, x2);
            a23 = Mac(a23, w3, x2);
            a30 = Mac(a30, w0, x3);
            a31 = Mac(a31, w1, x3);
            a32 = Mac(a32, w2, x3);
            a33 = Mac(a33, w3, x3);
        }

        output[yo] = Vector128.Sum(a00);
        output[yo + 1] = Vector128.Sum(a01);
        output[yo + 2] = Vector128.Sum(a02);
        output[yo + 3] = Vector128.Sum(a03);
        output[yo + stride] = Vector128.Sum(a10);
        output[yo + stride + 1] = Vector128.Sum(a11);
        output[yo + stride + 2] = Vector128.Sum(a12);
        output[yo + stride + 3] = Vector128.Sum(a13);
        output[yo + 2 * stride] = Vector128.Sum(a20);
        output[yo + 2 * stride + 1] = Vector128.Sum(a21);
        output[yo + 2 * stride + 2] = Vector128.Sum(a22);
        output[yo + 2 * stride + 3] = Vector128.Sum(a23);
        output[yo + 3 * stride] = Vector128.Sum(a30);
        output[yo + 3 * stride + 1] = Vector128.Sum(a31);
        output[yo + 3 * stride + 2] = Vector128.Sum(a32);
        output[yo + 3 * stride + 3] = Vector128.Sum(a33);
        for (int t = 0; t < 4; ++t)
        {
            for (int j = 0; j < 4; ++j)
            {
                float sum = output[yo + t * stride + j] + (bias?[row + j] ?? 0);
                for (int k = i; k < n; ++k)
                {
                    sum += weights[wo + j * n + k] * input[xo + t * n + k];
                }

                output[yo + t * stride + j] = sum;
            }
        }
    }

    /// <summary>
    /// 计算四个时间位置对同一权重行的点积。
    /// </summary>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="offset">输入数据的起始元素偏移。</param>
    /// <param name="y">当前操作的输出存储区。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="n">节点、向量长度或内积维度，具体含义由所在算子定义。</param>
    /// <param name="stride">输入或输出的行步长，单位为元素。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    private static void Dot4(ReadOnlySpan<float> w, float[] x, int offset, float[] y, int output, int n, int stride, float bias)
    {
        var a = Vector128<float>.Zero;
        var b = a;
        var c = a;
        var d = a;
        int i = 0;
        for (; i <= n - 4; i += 4)
        {
            var v = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(w), (nuint)i);
            var x0 = Vector128.LoadUnsafe(ref x[offset + i]);
            var x1 = Vector128.LoadUnsafe(ref x[offset + n + i]);
            var x2 = Vector128.LoadUnsafe(ref x[offset + 2 * n + i]);
            var x3 = Vector128.LoadUnsafe(ref x[offset + 3 * n + i]);
            if (AdvSimd.IsSupported)
            {
                a = AdvSimd.FusedMultiplyAdd(a, v, x0);
                b = AdvSimd.FusedMultiplyAdd(b, v, x1);
                c = AdvSimd.FusedMultiplyAdd(c, v, x2);
                d = AdvSimd.FusedMultiplyAdd(d, v, x3);
            }
            else
            {
                a += v * x0;
                b += v * x1;
                c += v * x2;
                d += v * x3;
            }
        }

        float aa = Vector128.Sum(a) + bias, bb = Vector128.Sum(b) + bias, cc = Vector128.Sum(c) + bias, dd = Vector128.Sum(d) + bias;
        for (; i < n; ++i)
        {
            aa += w[i] * x[offset + i];
            bb += w[i] * x[offset + n + i];
            cc += w[i] * x[offset + 2 * n + i];
            dd += w[i] * x[offset + 3 * n + i];
        }

        y[output] = aa;
        y[output + stride] = bb;
        y[output + 2 * stride] = cc;
        y[output + 3 * stride] = dd;
    }

    /// <summary>
    /// 按 32 个元素一组量化激活到有符号八位整数，并保存缩放因子。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="q">查询张量或分组量化后的激活缓冲区。</param>
    /// <param name="scales">每 32 个元素一组的量化缩放因子。</param>
    public static void Quantize(ReadOnlySpan<float> input, Span<sbyte> q, Span<float> scales)
    {
        if (input.Length % 32 != 0 || q.Length != input.Length || scales.Length != input.Length / 32)
        {
            throw new ArgumentException("Quantization needs matching complete blocks");
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            QuantizeNeon(input, q, scales);
            return;
        }

        for (int b = 0; b < scales.Length; ++b)
        {
            float max = 0;
            for (int i = 0; i < 32; ++i)
            {
                max = Math.Max(max, Math.Abs(input[b * 32 + i]));
            }

            float d = max / 127f, inv = d == 0 ? 0 : 1f / d;
            scales[b] = d;
            for (int i = 0; i < 32; ++i)
            {
                q[b * 32 + i] = (sbyte)Math.Clamp((int)MathF.Round(input[b * 32 + i] * inv, MidpointRounding.AwayFromZero), -127, 127);
            }
        }
    }

    /// <summary>
    /// 计算带分组缩放的 Q8 点积，支持 ARM SDOT 和精确边界回退。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="sa">第一个量化输入的分组缩放因子。</param>
    /// <param name="ao">第一个量化输入的元素偏移。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <param name="sb">第二个量化输入的分组缩放因子。</param>
    /// <param name="bo">第二个量化输入的元素偏移。</param>
    /// <param name="count">需要处理的元素或任务数量。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    /// <returns>计算得到的标量数值。</returns>
    public static float QuantizedDot(sbyte[] a, float[] sa, int ao, sbyte[] b, float[] sb, int bo, int count, bool simd = true)
    {
        if (ao < 0
            || bo < 0
            || count < 0
            || ((ao | bo | count) & 31) != 0
            || ao > a.Length - count
            || bo > b.Length - count
            || (ao + count) / 32 > sa.Length
            || (bo + count) / 32 > sb.Length)
        {
            throw new ArgumentException("Invalid quantized dot-product range");
        }

        if (simd && Dp.IsSupported)
        {
            ref sbyte av = ref MemoryMarshal.GetArrayDataReference(a);
            ref sbyte bv = ref MemoryMarshal.GetArrayDataReference(b);
            ref float ascale = ref MemoryMarshal.GetArrayDataReference(sa);
            ref float bscale = ref MemoryMarshal.GetArrayDataReference(sb);
            var sum = Vector128<float>.Zero;
            for (int i = 0; i < count; i += 32)
            {
                var acc = Dp.DotProduct(Vector128<int>.Zero, Vector128.LoadUnsafe(ref av, (nuint)(ao + i)), Vector128.LoadUnsafe(ref bv, (nuint)(bo + i)));
                acc = Dp.DotProduct(acc, Vector128.LoadUnsafe(ref av, (nuint)(ao + i + 16)), Vector128.LoadUnsafe(ref bv, (nuint)(bo + i + 16)));
                float scale = Unsafe.Add(ref ascale, (ao + i) / 32) * Unsafe.Add(ref bscale, (bo + i) / 32);
                sum = AdvSimd.FusedMultiplyAdd(sum, Vector128.ConvertToSingle(acc), Vector128.Create(scale));
            }

            return Vector128.Sum(sum);
        }

        float result = 0;
        for (int i = 0; i < count; i += 32)
        {
            int sum;
            if (simd && Dp.IsSupported)
            {
                var acc = Dp.DotProduct(Vector128<int>.Zero, Vector128.LoadUnsafe(ref a[ao + i]), Vector128.LoadUnsafe(ref b[bo + i]));
                acc = Dp.DotProduct(acc, Vector128.LoadUnsafe(ref a[ao + i + 16]), Vector128.LoadUnsafe(ref b[bo + i + 16]));
                sum = Vector128.Sum(acc);
            }
            else if (simd && Vector128.IsHardwareAccelerated)
            {
                var (a0, a1) = Vector128.Widen(Vector128.LoadUnsafe(ref a[ao + i]));
                var (b0, b1) = Vector128.Widen(Vector128.LoadUnsafe(ref b[bo + i]));
                var (a2, a3) = Vector128.Widen(Vector128.LoadUnsafe(ref a[ao + i + 16]));
                var (b2, b3) = Vector128.Widen(Vector128.LoadUnsafe(ref b[bo + i + 16]));
                // 求和前先扩展位宽，两个乘积相加可能溢出有符号 16 位整数。
                sum = SumWide(a0 * b0) + SumWide(a1 * b1) + SumWide(a2 * b2) + SumWide(a3 * b3);
            }
            else
            {
                sum = 0;
                for (int j = 0; j < 32; ++j)
                {
                    sum += a[ao + i + j] * b[bo + i + j];
                }
            }

            result += sum * sa[(ao + i) / 32] * sb[(bo + i) / 32];
        }

        return result;
    }

    /// <summary>
    /// 先扩展整数累加器再求和，避免短整数溢出。
    /// </summary>
    /// <param name="v">按帧排列的值张量或待归约向量。</param>
    /// <returns>计算得到的整数结果。</returns>
    private static int SumWide(Vector128<short> v)
    {
        var (lo, hi) = Vector128.Widen(v);
        return Vector128.Sum(lo + hi);
    }

    /// <summary>
    /// 将另一数组逐元素累加到目标数组。
    /// </summary>
    /// <param name="target">目标数组或目标形状。</param>
    /// <param name="other">需要累加到目标区的等长输入。</param>
    public static void Add(float[] target, float[] other)
    {
        if (target.Length != other.Length)
        {
            throw new ArgumentException("Residual shape mismatch");
        }

        int i = 0;
        for (; i <= target.Length - 4; i += 4)
        {
            (Vector128.LoadUnsafe(ref target[i]) + Vector128.LoadUnsafe(ref other[i])).StoreUnsafe(ref target[i]);
        }

        for (; i < target.Length; ++i)
        {
            target[i] += other[i];
        }
    }

    /// <summary>
    /// 按行执行 LayerNorm 或 RMSNorm，原位更新输入。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="scale">仿射变换或加权累加使用的缩放参数。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="epsilon">用于避免归一化除零的正数稳定项。</param>
    /// <param name="rms">为 true 时执行 RMSNorm，否则执行 LayerNorm。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public static float[] Norm(float[] x, int width, float[] scale, float[]? bias, float epsilon, bool rms = false)
    {
        if (Vector128.IsHardwareAccelerated && VectorElementwise)
        {
            return VectorNorm(x, width, scale, bias, epsilon, rms);
        }

        var y = TensorWorkspace.Allocate(x.Length);
        for (int offset = 0; offset < x.Length; offset += width)
        {
            double mean = 0;
            if (!rms)
            {
                for (int j = 0; j < width; ++j)
                {
                    mean += x[offset + j];
                }

                mean /= width;
            }

            double variance = 0;
            for (int j = 0; j < width; ++j)
            {
                double z = x[offset + j] - mean;
                variance += z * z;
            }

            float inverse = 1 / MathF.Sqrt((float)(variance / width) + epsilon);
            for (int j = 0; j < width; ++j)
            {
                y[offset + j] = (x[offset + j] - (float)mean) * inverse * scale[j] + (bias?[j] ?? 0);
            }
        }

        return y;
    }

    /// <summary>
    /// 原位执行数值稳定的 Softmax。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    public static void Softmax(Span<float> x)
    {
        if (Vector128.IsHardwareAccelerated && x.Length >= 8 && VectorTranscendentals)
        {
            VectorSoftmax(x);
            return;
        }

        float max = float.NegativeInfinity;
        foreach (float z in x)
        {
            max = Math.Max(max, z);
        }

        double sum = 0;
        for (int i = 0; i < x.Length; ++i)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }

        float inv = (float)(1 / sum);
        for (int i = 0; i < x.Length; ++i)
        {
            x[i] *= inv;
        }
    }

    /// <summary>
    /// 计算多头注意力的分数、归一化概率与加权值。
    /// </summary>
    /// <param name="q">查询张量或分组量化后的激活缓冲区。</param>
    /// <param name="k">按帧和注意力头排列的键张量。</param>
    /// <param name="v">按帧排列的值张量或待归约向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="heads">注意力头数量。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Attention(float[] q, float[] k, float[] v, int frames, int width, int heads)
    {
        if (Simd && (AdvSimd.IsSupported || System.Runtime.Intrinsics.X86.Avx2.IsSupported) && frames >= 16 && Environment.GetEnvironmentVariable("FUNASRNET_BLOCK_ATTENTION") != "0")
        {
            return BlockAttention(q, k, v, frames, width, heads);
        }

        return AttentionReference(q, k, v, frames, width, heads);
    }

    /// <summary>
    /// 通过直接分数与加权累加实现注意力，作为优化路径的数值参考。
    /// </summary>
    /// <param name="q">查询张量或分组量化后的激活缓冲区。</param>
    /// <param name="k">按帧和注意力头排列的键张量。</param>
    /// <param name="v">按帧排列的值张量或待归约向量。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <param name="width">每一输入行的元素数量。</param>
    /// <param name="heads">注意力头数量。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    internal float[] AttentionReference(float[] q, float[] k, float[] v, int frames, int width, int heads)
    {
        var output = TensorWorkspace.Allocate(q.Length, clear: true);
        int size = width / heads;
        For(
            heads * frames,
            index =>
            {
                int h = index % heads, t = index / heads, at = t * width + h * size;
                Span<float> scores = stackalloc float[frames];
                for (int s = 0; s < frames; ++s)
                {
                    scores[s] = AudioKernels.Dot(q.AsSpan(at, size), k.AsSpan(s * width + h * size, size), Simd) / MathF.Sqrt(size);
                }

                Softmax(scores);
                AttentionValues(scores, v, h * size, width, output.AsSpan(at, size), Simd);
            });
        return output;
    }

    /// <summary>
    /// 将缩放后的输入向量累加到输出区。
    /// </summary>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="scale">仿射变换或加权累加使用的缩放参数。</param>
    public static void AddScaled(Span<float> output, ReadOnlySpan<float> x, float scale)
    {
        var s = Vector128.Create(scale);
        int i = 0;
        ref float o = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(output);
        ref float v = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x);
        for (; i <= x.Length - 4; i += 4)
        {
            (Vector128.LoadUnsafe(ref o, (nuint)i) + Vector128.LoadUnsafe(ref v, (nuint)i) * s).StoreUnsafe(ref o, (nuint)i);
        }

        for (; i < x.Length; ++i)
        {
            output[i] += x[i] * scale;
        }
    }
}
