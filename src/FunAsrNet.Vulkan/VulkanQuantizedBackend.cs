using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using FunAsrNet.Vulkan.Native;

namespace FunAsrNet.Vulkan;

/// <summary>保留原 Q8_0 分组与激活舍入规则的 Vulkan 投影路径。</summary>
public sealed unsafe partial class VulkanFloatBackend
{
    private VkPipeline? _quantizedPipeline;
    private nint _quantizedShader, _quantizedSet;
    private VkBuffer? _inputScales;
    private long _quantizedCalls;
    private bool _cooperativeQ8;

    /// <summary>创建 Q8_0 管线与描述符，FP32 管线和栅栏由主实例管理。</summary>
    private void InitializeQuantized()
    {
        uint size = _device.SubgroupSize;
        _cooperativeQ8 = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_SUBGROUP_Q8") != "0"
            && _device.SubgroupSizeControl && _device.ComputeSubgroupSize && _device.ComputeFullSubgroups
            && (_device.SubgroupOps & 0x10) != 0 && (_device.SubgroupStages & 0x20) != 0
            && size >= 8 && size <= 128 && (size & (size - 1)) == 0
            && size >= _device.SubgroupMin && size <= _device.SubgroupMax;
        _quantizedShader = _device.NewShaderModule(_cooperativeQ8 ? CooperativeQuantizedShader.Binary() : QuantizedShader.Binary());
        _quantizedPipeline = _device.NewPipeline(_quantizedShader, 6, 16, _cooperativeQ8 ? size : 0);
        _quantizedSet = _device.NewDescriptorSet(_quantizedPipeline.SetLayout);
    }

    /// <summary>按原每三十二元素量化规则执行 Q8_0 线性投影。</summary>
    /// <param name="input">只读 FP32 输入数组。</param>
    /// <param name="frames">连续输入时间行数。</param>
    /// <param name="weight">只读原 Q8_0 权重及缩放，允许交错存储。</param>
    /// <param name="bias">可选只读 FP32 偏置。</param>
    /// <param name="output">调用方提供的独立 FP32 结果数组。</param>
    /// <returns>成功处理返回 true；小矩阵或不支持的类型返回 false。</returns>
    public bool TryQuantizedLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output)
    {
        if (frames <= 0 || weight.Columns <= 0 || weight.Rows <= 0 || weight.Columns % 32 != 0
            || weight.Quantized is null || weight.Scales is null
            || (long)frames * weight.Columns * weight.Rows < MinimumOperations)
        {
            return false;
        }

        if (input.Length != checked(frames * weight.Columns) || output.Length != checked(frames * weight.Rows))
        {
            throw new ArgumentException("Q8 GPU 投影尺寸不一致。");
        }

        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long started = Stopwatch.GetTimestamp();
            int columns = weight.Columns, rows = weight.Rows, blocks = columns / 32;
            sbyte[] weights = weight.Quantized;
            float[] scales = weight.Scales;
            if (weight.QuantizedInterleaved && (!_weights.ContainsKey(weights) || !_weights.ContainsKey(scales)))
            {
                weights = new sbyte[checked(rows * columns)];
                scales = new float[checked(rows * blocks)];
                for (int row = 0; row < rows; row++)
                {
                    for (int column = 0; column < columns; column++)
                    {
                        weights[row * columns + column] = weight.Quantized[row / 4 * 4 * columns + column / 4 * 16 + row % 4 * 4 + column % 4];
                    }

                    for (int block = 0; block < blocks; block++)
                    {
                        scales[row * blocks + block] = weight.Scales[row / 4 * 4 * blocks + block * 4 + row % 4];
                    }
                }
            }

            float[] biasValues = bias?.Values ?? _zero;
            if ((long)weights.Length + (long)(scales.Length + biasValues.Length) * sizeof(float) > _cacheLimit)
            {
                throw new NotSupportedException("Q8 权重和缩放超过驻留上限。");
            }

            var w = ResidentRaw(weight.Quantized, MemoryMarshal.AsBytes(weights.AsSpan()), []);
            var ws = ResidentRaw(weight.Scales, MemoryMarshal.AsBytes(scales.AsSpan()), [weight.Quantized]);
            var b = ResidentRaw(biasValues, MemoryMarshal.AsBytes(biasValues.AsSpan()), [weight.Quantized, weight.Scales]);
            var q = ArrayPool<sbyte>.Shared.Rent(input.Length);
            var inputScale = ArrayPool<float>.Shared.Rent(input.Length / 32);
            try
            {
                TensorMath.Quantize(input, q.AsSpan(0, input.Length), inputScale.AsSpan(0, input.Length / 32));
                EnsureBuffer(ref _input, (ulong)input.Length, true);
                EnsureBuffer(ref _inputScales, checked((ulong)(input.Length / 32) * sizeof(float)), true);
                ulong outputBytes = checked((ulong)output.Length * sizeof(float));
                EnsureBuffer(ref _output, outputBytes, false);
                EnsureBuffer(ref _readback, outputBytes, true, true);
                fixed (sbyte* source = q)
                {
                    _device.Upload(_input!, source, (ulong)input.Length);
                }

                fixed (float* source = inputScale)
                {
                    _device.Upload(_inputScales!, source, (ulong)(input.Length / 32) * sizeof(float));
                }

                _uploads += input.Length + (long)(input.Length / 32) * sizeof(float);
                _device.BindBuffer(_quantizedSet, 0, _input!);
                _device.BindBuffer(_quantizedSet, 1, w);
                _device.BindBuffer(_quantizedSet, 2, _inputScales!);
                _device.BindBuffer(_quantizedSet, 3, ws);
                _device.BindBuffer(_quantizedSet, 4, b);
                _device.BindBuffer(_quantizedSet, 5, _output!);
                uint* parameters = stackalloc uint[4] { (uint)frames, (uint)rows, (uint)columns, bias?.Values is null ? 0u : 1u };
                if (weight.QuantizedInterleaved || _combinedScaleFirst.TryGetValue(weight, out _))
                {
                    parameters[3] |= 2;
                }
                uint rowsPerGroup = _cooperativeQ8 ? 16u : 128u;
                DispatchAndRead(_quantizedPipeline!, _quantizedSet, parameters, 16, ((uint)rows + rowsPerGroup - 1) / rowsPerGroup, (uint)frames, output);
                _quantizedCalls++;
                _seconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
            }
            finally
            {
                ArrayPool<sbyte>.Shared.Return(q);
                ArrayPool<float>.Shared.Return(inputScale);
            }

            return true;
        }
    }

    /// <summary>释放 Q8 管线资源，调用时 GPU 已经空闲。</summary>
    private void DisposeQuantized()
    {
        _inputScales?.Free();
        if (_quantizedSet != 0)
        {
            _device.FreeDescriptorSet(_quantizedSet);
        }

        if (_quantizedPipeline is not null)
        {
            Vk.vkDestroyPipeline(_device.Device, _quantizedPipeline.Pipeline, null);
            Vk.vkDestroyPipelineLayout(_device.Device, _quantizedPipeline.Layout, null);
            Vk.vkDestroyDescriptorSetLayout(_device.Device, _quantizedPipeline.SetLayout, null);
        }

        if (_quantizedShader != 0)
        {
            Vk.vkDestroyShaderModule(_device.Device, _quantizedShader, null);
        }
    }
}
