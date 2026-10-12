using System.Diagnostics;
using FunAsrNet.Vulkan.Native;

namespace FunAsrNet.Vulkan;

/// <summary>使用系统 Vulkan 驱动执行 FP32 大矩阵与一维卷积，实例内同步且缓存容量有界。</summary>
public sealed unsafe partial class VulkanFloatBackend : IFloatComputeBackend, IQuantizedComputeBackend, IGroupedComputeBackend, IReluProjectionBackend, IDisposable
{
    private readonly object _gate = new();
    private readonly VkDevice _device;
    private readonly VkPipeline _pipeline;
    private readonly nint _shader, _set, _command, _fence;
    private readonly long _cacheLimit;
    private readonly Dictionary<Array, Resident> _weights = new(ReferenceEqualityComparer.Instance);
    private readonly float[] _zero = [0];
    private VkBuffer? _input, _output, _readback, _transientWeight;
    private long _sequence, _cacheBytes, _hits, _uploads, _downloads, _linearCalls, _convolutionCalls;
    private double _seconds;
    private bool _disposed;
    private readonly int _fenceSpinMicroseconds = int.TryParse(Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_SPIN_US"), out int spin)
        ? Math.Clamp(spin, 0, 500) : OperatingSystem.IsWindows() ? 100 : 0;

    /// <summary>只读权重的驻留缓冲和最近使用顺序。</summary>
    /// <param name="buffer">已经完成上传的设备缓冲。</param>
    /// <param name="used">最近使用的递增序号。</param>
    private sealed class Resident(VkBuffer buffer, long used)
    {
        public readonly VkBuffer Buffer = buffer;
        public long Used = used;
    }

    /// <summary>后端及实际选中的物理设备名称。</summary>
    public string Name => "vulkan-fp32 / " + _device.DeviceName;

    /// <summary>最小乘加次数；低于此值的线性或卷积留在 CPU。</summary>
    public long MinimumOperations { get; set; } = 250000;

    /// <summary>创建独立 Vulkan 设备和可复用计算状态，不自动替代默认 CPU 路径。</summary>
    /// <param name="cacheBytes">只读权重缓存字节上限，至少 4 MiB，默认 2 GiB。</param>
    /// <param name="deviceIndex">按独立显卡优先枚举的设备索引，从零开始。</param>
    public VulkanFloatBackend(long cacheBytes = 2048L * 1024 * 1024, uint deviceIndex = 0)
    {
        if (cacheBytes < 4L * 1024 * 1024 || !Environment.Is64BitProcess)
        {
            throw new ArgumentException("Vulkan 后端要求 64 位进程及至少 4 MiB 权重缓存。");
        }

        _cacheLimit = cacheBytes;
        _device = VkDevice.Create(deviceIndex);
        try
        {
            _shader = _device.NewShaderModule(SpeechShader.Binary());
            _pipeline = _device.NewPipeline(_shader, 4, 52);
            _set = _device.NewDescriptorSet(_pipeline.SetLayout);
            _command = _device.NewCommandBuffer();
            _fence = _device.NewFence();
            InitializeQuantized();
        }
        catch
        {
            DisposeQuantized();
            _device.DestroyFence(_fence);
            _device.FreeCommandBuffer(_command);
            if (_set != 0)
            {
                _device.FreeDescriptorSet(_set);
            }

            if (_pipeline is not null)
            {
                Vk.vkDestroyPipeline(_device.Device, _pipeline.Pipeline, null);
                Vk.vkDestroyPipelineLayout(_device.Device, _pipeline.Layout, null);
                Vk.vkDestroyDescriptorSetLayout(_device.Device, _pipeline.SetLayout, null);
            }

            if (_shader != 0)
            {
                Vk.vkDestroyShaderModule(_device.Device, _shader, null);
            }

            _device.Dispose();
            throw;
        }
    }

    /// <summary>复制当前计数器快照，区分真实 GPU 调用与 CPU 回退。</summary>
    /// <returns>设备、调用数、缓存及包含数据传输和同步的累计时间。</returns>
    public object Statistics()
    {
        lock (_gate)
        {
            return new
            {
                device = _device.DeviceName,
                loader = Vk.LoadedLibrary,
                precision = "FP32 + original Q8_0",
                linearCalls = _linearCalls,
                convolutionCalls = _convolutionCalls,
                reluProjectionCalls = _reluProjectionCalls,
                computeSubmissions = _computeSubmissions,
                quantizedCalls = _quantizedCalls,
                quantizedKernel = _integerDotQ8 ? "subgroup-eight-lane-integer-dot" : _cooperativeQ8 ? "subgroup-eight-lane" : "scalar-row",
                subgroupSize = _device.SubgroupSize,
                fenceSpinMicroseconds = _fenceSpinMicroseconds,
                cacheBytes = _cacheBytes,
                cacheHits = _hits,
                uploadedBytes = _uploads,
                downloadedBytes = _downloads,
                endToEndSeconds = _seconds,
                maxStorageRange = _device.MaxStorageRange
            };
        }
    }

    /// <summary>执行浮点线性投影；量化权重和小矩阵返回 CPU 计算。</summary>
    /// <param name="input">连续的时间行输入，调用期间只读。</param>
    /// <param name="frames">输入时间行数。</param>
    /// <param name="weight">原始或交错布局的只读 FP32 权重，F16 来源按原值展开。</param>
    /// <param name="bias">可选只读偏置。</param>
    /// <param name="output">独立的输出数组，完成后可由 CPU 立即读取。</param>
    /// <returns>完整计算成功为 true；不适合 GPU 时返回 false。</returns>
    public bool TryLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output)
    {
        if (frames <= 0 || weight.Columns <= 0 || weight.Rows <= 0
            || weight.Values is null || weight.Quantized is not null || weight.QuantizedInterleaved
            || (long)frames * weight.Columns * weight.Rows < MinimumOperations)
        {
            return false;
        }

        if (input.Length != checked(frames * weight.Columns) || output.Length != checked(frames * weight.Rows))
        {
            throw new ArgumentException("GPU 线性投影尺寸不一致。");
        }

        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long started = Stopwatch.GetTimestamp();
            float[]? ordered = null;
            if (weight.FloatInterleaved && !_weights.ContainsKey(weight.Values))
            {
                ordered = new float[checked(weight.Columns * weight.Rows)];
                for (int row = 0; row < weight.Rows; row++)
                {
                    weight.CopyRow(row, ordered.AsSpan(row * weight.Columns, weight.Columns));
                }
            }

            uint* parameters = stackalloc uint[13];
            new Span<uint>(parameters, 13).Clear();
            parameters[0] = (uint)frames;
            parameters[1] = (uint)weight.Rows;
            parameters[2] = (uint)weight.Columns;
            parameters[12] = bias?.Values is null ? 0u : 1u;
            if (weight.FloatInterleaved)
            {
                // 交错内核最后添加偏置；x64 原路径按列累加且不使用 ARM 融合乘加。
                if (!System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported)
                {
                    parameters[12] |= 4;
                }
            }
            Execute(input, weight.Values, ordered ?? weight.Values, bias?.Values ?? _zero, output, parameters, weight.IsTransient);
            _linearCalls++;
            _seconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
            return true;
        }
    }

    /// <summary>用隐式展开直接计算分组为一的一维卷积，不在 CPU 创建完整 im2col。</summary>
    /// <param name="input">只读 NCL 输入张量。</param>
    /// <param name="weight">只读 OCK 权重张量。</param>
    /// <param name="bias">可选输出通道偏置。</param>
    /// <param name="node">ONNX 卷积节点及步长、填充、膨胀属性。</param>
    /// <param name="output">成功时返回 NCL 输出，不支持时为空。</param>
    /// <returns>成功完成 GPU 计算时为 true。</returns>
    public bool TryConvolution(Tensor input, Tensor weight, Tensor? bias, OnnxNode node, out Tensor? output)
    {
        output = null;
        if (input.Rank != 3 || weight.Rank != 3 || node.Int("group", 1) != 1 || input.Integer || weight.Integer)
        {
            return false;
        }

        int batches = input.Shape[0], channels = input.Shape[1], length = input.Shape[2];
        int outputs = weight.Shape[0], kernel = weight.Shape[2];
        if (batches <= 0 || channels <= 0 || length <= 0 || outputs <= 0 || kernel <= 0)
        {
            return false;
        }
        int stride = node.Ints("strides", [1])[0], dilation = node.Ints("dilations", [1])[0];
        int[] pads = node.Ints("pads", [0, 0]);
        if (stride <= 0 || dilation <= 0 || weight.Shape[1] != channels)
        {
            throw new ArgumentException("GPU 卷积属性或通道不一致。");
        }

        int outLength = (length + pads[0] + pads[1] - dilation * (kernel - 1) - 1) / stride + 1;
        if (outLength <= 0 || (long)batches * outLength * outputs * channels * kernel < MinimumOperations)
        {
            return false;
        }

        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long started = Stopwatch.GetTimestamp();
            var result = Tensor.Zeros([batches, outputs, outLength]);
            uint* parameters = stackalloc uint[13]
            {
                (uint)(batches * outLength), (uint)outputs, (uint)(channels * kernel), 1,
                (uint)batches, (uint)channels, (uint)length, (uint)outLength,
                (uint)kernel, (uint)stride, (uint)dilation, unchecked((uint)pads[0]), bias is null ? 0u : 1u
            };
            Execute(input.Data, weight.Data, weight.Data, bias?.Data ?? _zero, result.Data, parameters);
            _convolutionCalls++;
            _seconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
            output = result;
            return true;
        }
    }

    /// <summary>按上限缓存只读权重，首次上传完成后才发布缓存项。</summary>
    /// <param name="key">标识不可变原数组的引用。</param>
    /// <param name="values">供首次上传的行主序数据，不长期保留布局转换副本。</param>
    /// <returns>有效的设备驻留缓冲；只在实例锁内使用。</returns>
    /// <param name="protectedKey">本次提交已引用的权重，分配偏置时不得淘汰。</param>
    private VkBuffer ResidentWeight(float[] key, float[] values, Array? protectedKey = null)
        => ResidentRaw(key, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()), protectedKey is null ? [] : [protectedKey]);

    /// <summary>上传任意只读字节布局，并保护当前提交已经取得的缓冲不被淘汰。</summary>
    /// <param name="key">不可变原数组的引用标识。</param>
    /// <param name="values">需要上传的字节范围。</param>
    /// <param name="protectedKeys">当前提交仍需要的数组标识。</param>
    /// <returns>完成上传的设备缓冲。</returns>
    private VkBuffer ResidentRaw(Array key, ReadOnlySpan<byte> values, Array[] protectedKeys)
    {
        if (_weights.TryGetValue(key, out var cached))
        {
            cached.Used = ++_sequence;
            _hits++;
            return cached.Buffer;
        }

        long bytes = values.Length;
        if (bytes > _cacheLimit || (ulong)bytes > _device.MaxStorageRange)
        {
            throw new NotSupportedException("单份 GPU 权重超过缓存或绑定范围。");
        }

        while (_cacheBytes + bytes > _cacheLimit)
        {
            var oldest = _weights.Where(item => !protectedKeys.Any(protectedKey => ReferenceEquals(item.Key, protectedKey))).MinBy(item => item.Value.Used);
            oldest.Value.Buffer.Free();
            _cacheBytes -= (long)oldest.Value.Buffer.Size;
            _weights.Remove(oldest.Key);
        }

        var buffer = _device.NewStorageBuffer((ulong)bytes, hostVisible: false);
        try
        {
            fixed (byte* source = values)
            {
                _device.Upload(buffer, source, (ulong)bytes);
            }

            _weights.Add(key, new Resident(buffer, ++_sequence));
            _cacheBytes += bytes;
            _uploads += bytes;
            return buffer;
        }
        catch
        {
            buffer.Free();
            throw;
        }
    }

    /// <summary>复用输入、输出或回读缓冲，增大时释放旧分配。</summary>
    /// <param name="buffer">由实例独占管理的缓冲引用。</param>
    /// <param name="bytes">本次至少需要的字节数。</param>
    /// <param name="hostVisible">是否需要 CPU 映射。</param>
    /// <param name="preferHost">回读时优先使用缓存的系统内存。</param>
    private void EnsureBuffer(ref VkBuffer? buffer, ulong bytes, bool hostVisible, bool preferHost = false)
    {
        if (bytes > _device.MaxStorageRange)
        {
            throw new NotSupportedException("张量超过 Vulkan 存储缓冲绑定范围。");
        }

        if (buffer is null || buffer.Size < bytes)
        {
            buffer?.Free();
            buffer = null;
            buffer = _device.NewStorageBuffer(bytes, hostVisible, preferHost);
            buffer.KeepMapped = hostVisible;
        }
    }

    /// <summary>上传激活，执行计算并同步回 CPU；所有传输与栅栏等待均在调用内。</summary>
    /// <param name="input">输入数组，本次调用内只读。</param>
    /// <param name="weightKey">原始只读权重数组引用，用于驻留缓存。</param>
    /// <param name="weightValues">供首次上传的权重布局。</param>
    /// <param name="bias">只读偏置或单元素零数组。</param>
    /// <param name="output">CPU 结果数组，在栅栏完成后写入。</param>
    /// <param name="parameters">十三个 uint 推送参数，布局与 SPIR-V 一致。</param>
    /// <param name="transientWeight">动态矩阵直接上传到可复用缓冲，不占用只读权重缓存。</param>
    private void Execute(float[] input, float[] weightKey, float[] weightValues, float[] bias, float[] output, uint* parameters, bool transientWeight = false)
    {
        ulong inputBytes = checked((ulong)input.Length * sizeof(float));
        ulong outputBytes = checked((ulong)output.Length * sizeof(float));
        if (checked(((long)weightValues.Length + bias.Length) * sizeof(float)) > _cacheLimit)
        {
            throw new NotSupportedException("单次提交的权重与偏置超过缓存上限。");
        }

        VkBuffer weights;
        if (transientWeight)
        {
            ulong weightBytes = checked((ulong)weightValues.Length * sizeof(float));
            EnsureBuffer(ref _transientWeight, weightBytes, true);
            fixed (float* source = weightValues)
            {
                _device.Upload(_transientWeight!, source, weightBytes);
            }
            weights = _transientWeight!;
            _uploads += (long)weightBytes;
        }
        else
        {
            weights = ResidentWeight(weightKey, weightValues);
        }
        var biases = ResidentWeight(bias, bias, weightKey);
        EnsureBuffer(ref _input, inputBytes, true);
        EnsureBuffer(ref _output, outputBytes, false);
        EnsureBuffer(ref _readback, outputBytes, true, true);
        fixed (float* source = input)
        {
            _device.Upload(_input!, source, inputBytes);
        }

        _uploads += (long)inputBytes;
        _device.BindBuffer(_set, 0, _input!);
        _device.BindBuffer(_set, 1, weights);
        _device.BindBuffer(_set, 2, biases);
        _device.BindBuffer(_set, 3, _output!);
        DispatchAndRead(_pipeline, _set, parameters, 52, (parameters[0] + 31) / 32, (parameters[1] + 31) / 32, output);
    }

    /// <summary>记录、提交并等待一批计算与回读命令。</summary>
    /// <param name="pipeline">当前使用的计算管线。</param>
    /// <param name="descriptorSet">在本次提交结束前保持不变的描述符集合。</param>
    /// <param name="parameters">与着色器布局一致的参数指针。</param>
    /// <param name="parameterBytes">参数区的字节数量。</param>
    /// <param name="groupsX">第一维工作组数。</param>
    /// <param name="groupsY">第二维工作组数。</param>
    /// <param name="output">同步后的 CPU 输出数组。</param>
    private void DispatchAndRead(VkPipeline pipeline, nint descriptorSet, uint* parameters, uint parameterBytes, uint groupsX, uint groupsY, float[] output)
    {
        BeginCompute();
        RecordDispatch(pipeline, descriptorSet, parameters, parameterBytes, groupsX, groupsY);
        FinishRead(output);
    }

    /// <summary>开始记录计算命令，保证此前上传的输入可供着色器读取。</summary>
    private void BeginCompute()
    {
        Vk.Check(Vk.vkResetCommandBuffer(_command, 0), "reset compute command");
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_command, &begin), "begin compute command");
        var uploadBarrier = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier, SrcAccessMask = VkConst.AccessHostWrite | VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead };
        Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageHost | VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0, 1, &uploadBarrier, 0, null, 0, null);
    }

    /// <summary>在当前命令内记录一次计算，描述符在提交完成前不得修改。</summary>
    /// <param name="pipeline">计算管线。</param>
    /// <param name="descriptorSet">本次计算独立使用的描述符集合。</param>
    /// <param name="parameters">推送参数指针。</param>
    /// <param name="parameterBytes">推送参数字节数。</param>
    /// <param name="groupsX">第一维工作组数。</param>
    /// <param name="groupsY">第二维工作组数。</param>
    private void RecordDispatch(VkPipeline pipeline, nint descriptorSet, uint* parameters, uint parameterBytes, uint groupsX, uint groupsY)
    {
        Vk.vkCmdBindPipeline(_command, VkConst.BindPointCompute, pipeline.Pipeline);
        Vk.vkCmdBindDescriptorSets(_command, VkConst.BindPointCompute, pipeline.Layout, 0, 1, &descriptorSet, 0, null);
        Vk.vkCmdPushConstants(_command, pipeline.Layout, VkConst.StageComputeShader, 0, parameterBytes, parameters);
        Vk.vkCmdDispatch(_command, groupsX, groupsY, 1);
    }

    /// <summary>提交已记录的计算，等待最终结果回读；中间张量无需经过 CPU。</summary>
    /// <param name="output">最后一个计算写入的 CPU 输出数组。</param>
    private void FinishRead(float[] output)
    {
        ulong outputBytes = checked((ulong)output.Length * sizeof(float));
        var computeBarrier = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier, SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead };
        Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0, 1, &computeBarrier, 0, null, 0, null);
        var copy = new Vk.VkBufferCopy { Size = outputBytes };
        Vk.vkCmdCopyBuffer(_command, _output!.Buffer, _readback!.Buffer, 1, &copy);
        var readBarrier = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier, SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessHostRead };
        Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageTransfer, VkConst.PipelineStageHost, 0, 1, &readBarrier, 0, null, 0, null);
        Vk.Check(Vk.vkEndCommandBuffer(_command), "end compute command");
        _device.Submit(_command, _fence);
        if (_fenceSpinMicroseconds > 0)
        {
            long deadline = Stopwatch.GetTimestamp() + Stopwatch.Frequency * _fenceSpinMicroseconds / 1000000;
            while (Vk.vkGetFenceStatus(_device.Device, _fence) == VkResult.NotReady && Stopwatch.GetTimestamp() < deadline)
            {
                Thread.SpinWait(8);
            }
        }
        _device.WaitFence(_fence);
        void* address = _readback.Map();
        try
        {
            _readback.Invalidate(0, outputBytes);
            new ReadOnlySpan<float>(address, output.Length).CopyTo(output);
        }
        finally
        {
            _readback.Unmap();
        }

        _downloads += (long)outputBytes;
        _computeSubmissions++;
    }

    /// <summary>等待 GPU 空闲并释放本实例拥有的缓存、管线和设备；可重复调用。</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            Vk.Check(Vk.vkDeviceWaitIdle(_device.Device), "device idle before disposal");
            foreach (var value in _weights.Values)
            {
                value.Buffer.Free();
            }

            _weights.Clear();
            _groups.Clear();
            _combinedScaleFirst.Clear();
            _cacheBytes = 0;
            _input?.Free();
            _transientWeight?.Free();
            DisposeReluProjection();
            _output?.Free();
            _readback?.Free();
            _device.FreeDescriptorSet(_set);
            _device.FreeCommandBuffer(_command);
            _device.DestroyFence(_fence);
            Vk.vkDestroyPipeline(_device.Device, _pipeline.Pipeline, null);
            Vk.vkDestroyPipelineLayout(_device.Device, _pipeline.Layout, null);
            Vk.vkDestroyDescriptorSetLayout(_device.Device, _pipeline.SetLayout, null);
            Vk.vkDestroyShaderModule(_device.Device, _shader, null);
            DisposeQuantized();
            _device.Dispose();
        }
    }
}
