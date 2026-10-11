using System.Diagnostics;
using System.Runtime.InteropServices;
using FunAsrNet;

namespace FunAsrNet.Benchmarks;

/// <summary>独立 NPU 矩阵能力探针，仅显式调用时加载 RKNN，不接入产品默认推理。</summary>
internal static unsafe class RknnProbe
{
    /// <summary>与 RKNN 2.3.2 ARM64 矩阵配置 ABI 对应的值类型。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MatrixInfo
    {
        public int M, K, N, Type;
        public short BLayout, BQuantType, ACLayout, ACQuantType;
        public int Domain;
        public short GroupSize;
        public fixed byte Reserved[34];
    }

    /// <summary>矩阵输入或输出的尺寸、类型与布局信息。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MatrixAttribute
    {
        public fixed byte Name[256];
        public uint DimensionCount;
        public fixed uint Dimensions[16];
        public uint Size;
        public int Type;
    }

    /// <summary>三份矩阵属性，由运行库填充。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct MatrixAttributes
    {
        public MatrixAttribute A, B, C;
    }

    /// <summary>RKNN 分配的内存描述符，所有权由运行库管理。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct TensorMemory
    {
        public nint Address;
        public ulong PhysicalAddress;
        public int FileDescriptor, Offset;
        public uint Size, Flags;
        public nint PrivateData;
    }

    /// <summary>创建一个 F16 输入、F32 输出的矩阵上下文。</summary>
    /// <param name="context">返回的 ARM64 上下文句柄，成功后须销毁。</param>
    /// <param name="info">待计算的尺寸和布局。</param>
    /// <param name="attributes">返回的三个矩阵属性。</param>
    /// <returns>零表示成功，其他值为 RKNN 错误码。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int rknn_matmul_create(out ulong context, ref MatrixInfo info, out MatrixAttributes attributes);

    /// <summary>分配与当前上下文关联的设备内存。</summary>
    /// <param name="context">已创建的上下文。</param>
    /// <param name="size">请求分配的字节数。</param>
    /// <returns>内存描述符指针，失败时为零。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern nint rknn_create_mem(ulong context, uint size);

    /// <summary>把设备内存绑定到一个矩阵输入或输出。</summary>
    /// <param name="context">已创建的矩阵上下文。</param>
    /// <param name="memory">仍由运行库管理的内存描述符。</param>
    /// <param name="attribute">所绑定输入或输出的属性。</param>
    /// <returns>零表示成功，其他值为 RKNN 错误码。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int rknn_matmul_set_io_mem(ulong context, nint memory, ref MatrixAttribute attribute);

    /// <summary>同步 CPU 与设备缓存，确保复制内容可见。</summary>
    /// <param name="context">与内存关联的矩阵上下文。</param>
    /// <param name="memory">要同步的内存描述符。</param>
    /// <param name="mode">一表示同步至设备，二表示同步回 CPU。</param>
    /// <returns>零表示成功，其他值为 RKNN 错误码。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int rknn_mem_sync(ulong context, nint memory, int mode);

    /// <summary>阻塞运行矩阵乘法，默认使用厂商自动核心选择。</summary>
    /// <param name="context">已绑定三个矩阵的上下文。</param>
    /// <returns>零表示成功，其他值为 RKNN 错误码。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int rknn_matmul_run(ulong context);

    /// <summary>释放运行库分配的设备内存。</summary>
    /// <param name="context">内存所属的上下文。</param>
    /// <param name="memory">待释放的内存描述符，释放后不再访问。</param>
    /// <returns>零表示成功，其他值为 RKNN 错误码。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int rknn_destroy_mem(ulong context, nint memory);

    /// <summary>销毁矩阵上下文，调用前先释放其内存。</summary>
    /// <param name="context">待销毁的上下文句柄。</param>
    /// <returns>零表示成功，其他值为 RKNN 错误码。</returns>
    [DllImport("librknnrt.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern int rknn_matmul_destroy(ulong context);

    /// <summary>检查运行库返回值，失败时停止探针。</summary>
    /// <param name="code">RKNN 返回的错误码。</param>
    /// <param name="operation">便于诊断的操作名称。</param>
    private static void Success(int code, string operation)
    {
        if (code != 0)
        {
            throw new InvalidOperationException($"{operation}: RKNN error {code}");
        }
    }

    /// <summary>运行一个确定性的矩阵样例，包含半精度转换与传输成本。</summary>
    /// <param name="m">输入时间行数，须为正整数。</param>
    /// <param name="k">矩阵内维，须按 RK3588 要求对齐。</param>
    /// <param name="n">输出通道数，须按 RK3588 要求对齐。</param>
    /// <param name="repeats">预热之外的测量次数。</param>
    /// <param name="emit">接收逐轮测量及误差记录的输出回调。</param>
    internal static void Run(int m, int k, int n, int repeats, Action<object> emit)
    {
        if (!OperatingSystem.IsLinux() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
        {
            throw new PlatformNotSupportedException("RKNN 探针仅支持 Linux ARM64。");
        }

        if (sizeof(MatrixInfo) != 64 || sizeof(MatrixAttribute) != 332 || sizeof(TensorMemory) != 40)
        {
            throw new InvalidOperationException("RKNN 2.3.2 ABI 布局检查失败。");
        }

        var random = new Random(20261010);
        float[] input = Enumerable.Range(0, checked(m * k)).Select(_ => (float)(random.NextDouble() * 0.2 - 0.1)).ToArray();
        float[] weights = Enumerable.Range(0, checked(n * k)).Select(_ => (float)(random.NextDouble() * 0.2 - 0.1)).ToArray();
        Half[] inputHalf = new Half[input.Length], weightHalf = new Half[weights.Length];
        for (int channel = 0; channel < n; channel++)
        {
            for (int column = 0; column < k; column++)
            {
                weightHalf[column * n + channel] = (Half)weights[channel * k + column];
            }
        }

        var math = new TensorMath(4) { Optimized = false };
        var cpuWeight = new Weight("npu-probe-f32", [k, n]) { Values = weights };
        float[] exact = math.Linear(input, m, cpuWeight);
        float[] roundedInput = input.Select(value => (float)(Half)value).ToArray();
        var roundedWeight = new Weight("npu-probe-rounded", [k, n]) { Values = weights.Select(value => (float)(Half)value).ToArray() };
        float[] rounded = math.Linear(roundedInput, m, roundedWeight);
        var info = new MatrixInfo { M = m, K = k, N = n, Type = 1 };
        ulong context = 0;
        nint a = 0, b = 0, c = 0;
        try
        {
            var loading = Stopwatch.StartNew();
            Success(rknn_matmul_create(out context, ref info, out var attributes), "create");
            if (attributes.A.Size < inputHalf.Length * 2 || attributes.B.Size < weightHalf.Length * 2 || attributes.C.Size < exact.Length * 4)
            {
                throw new InvalidOperationException("运行库返回了小于逻辑矩阵的内存尺寸。");
            }

            a = rknn_create_mem(context, attributes.A.Size);
            b = rknn_create_mem(context, attributes.B.Size);
            c = rknn_create_mem(context, attributes.C.Size);
            if (a == 0 || b == 0 || c == 0)
            {
                throw new InvalidOperationException("NPU 内存分配失败。");
            }

            Success(rknn_matmul_set_io_mem(context, c, ref attributes.C), "bind C");
            weightHalf.AsSpan().CopyTo(new Span<Half>((void*)((TensorMemory*)b)->Address, weightHalf.Length));
            Success(rknn_mem_sync(context, b, 1), "upload weights");
            // 普通布局的权重可能在绑定时转换，因此先写入和同步，再绑定只读权重。
            Success(rknn_matmul_set_io_mem(context, b, ref attributes.B), "bind B");
            emit(new
            {
                kind = "npu-load",
                m,
                k,
                n,
                seconds = loading.Elapsed.TotalSeconds,
                precision = "F16 x F16 -> F32",
                coreMask = "auto",
                productBackend = false
            });
            float[] result = new float[exact.Length];
            for (int iteration = 0; iteration <= repeats; iteration++)
            {
                var watch = Stopwatch.StartNew();
                for (int index = 0; index < input.Length; index++)
                {
                    inputHalf[index] = (Half)input[index];
                }

                inputHalf.AsSpan().CopyTo(new Span<Half>((void*)((TensorMemory*)a)->Address, inputHalf.Length));
                Success(rknn_mem_sync(context, a, 1), "upload input");
                Success(rknn_matmul_set_io_mem(context, a, ref attributes.A), "bind A");
                long started = Stopwatch.GetTimestamp();
                Success(rknn_matmul_run(context), "run");
                double deviceSeconds = Stopwatch.GetElapsedTime(started).TotalSeconds;
                Success(rknn_mem_sync(context, c, 2), "download output");
                new ReadOnlySpan<float>((void*)((TensorMemory*)c)->Address, result.Length).CopyTo(result);
                double seconds = watch.Elapsed.TotalSeconds;
                watch.Restart();
                _ = math.Linear(input, m, cpuWeight);
                double cpuSeconds = watch.Elapsed.TotalSeconds;
                emit(new
                {
                    kind = "npu-matmul",
                    m,
                    k,
                    n,
                    iteration,
                    warmup = iteration == 0,
                    seconds,
                    deviceSeconds,
                    cpuSeconds
                });
            }

            double maxVsRounded = 0, errorVsRounded = 0, maxVsF32 = 0, errorVsF32 = 0;
            for (int index = 0; index < result.Length; index++)
            {
                if (!float.IsFinite(result[index]))
                {
                    throw new InvalidDataException("NPU 输出包含非有限值。");
                }

                double aError = result[index] - rounded[index], bError = result[index] - exact[index];
                maxVsRounded = Math.Max(maxVsRounded, Math.Abs(aError));
                maxVsF32 = Math.Max(maxVsF32, Math.Abs(bError));
                errorVsRounded += aError * aError;
                errorVsF32 += bError * bError;
            }

            double rmseVsRounded = Math.Sqrt(errorVsRounded / result.Length);
            bool passed = maxVsRounded <= 0.001 && rmseVsRounded <= 0.0001;
            emit(new
            {
                kind = "npu-parity",
                m,
                k,
                n,
                maxVsRounded,
                rmseVsRounded,
                maxVsF32,
                rmseVsF32 = Math.Sqrt(errorVsF32 / result.Length),
                introducesHalfRounding = true,
                passed
            });
            if (!passed)
            {
                throw new InvalidDataException("NPU 结果与半精度输入 CPU 参考误差超限，不接受其速度结论。");
            }
        }
        finally
        {
            foreach (nint memory in new[] { c, b, a })
            {
                if (memory != 0)
                {
                    _ = rknn_destroy_mem(context, memory);
                }
            }

            if (context != 0)
            {
                _ = rknn_matmul_destroy(context);
            }
        }
    }
}
