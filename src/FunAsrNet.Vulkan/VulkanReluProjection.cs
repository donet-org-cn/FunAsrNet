using System.Diagnostics;
using System.Runtime.InteropServices;
using FunAsrNet.Vulkan.Native;

namespace FunAsrNet.Vulkan;

/// <summary>在一次 GPU 提交内执行编码器的两层前馈投影，保留原有 FP32 累加顺序。</summary>
public sealed unsafe partial class VulkanFloatBackend
{
    private VkBuffer? _reluIntermediate;
    private nint _reluSecondSet;
    private long _reluProjectionCalls, _computeSubmissions;
    private readonly bool _reluProjectionEnabled = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_RELU_CHAIN") != "0";

    /// <summary>执行线性、ReLU、线性；不支持的布局返回 false，由调用方使用原有路径。</summary>
    /// <param name="input">按帧连续排列的输入。</param>
    /// <param name="frames">输入帧数。</param>
    /// <param name="first">第一层行优先只读权重。</param>
    /// <param name="firstBias">第一层偏置，可为空。</param>
    /// <param name="second">第二层行优先只读权重。</param>
    /// <param name="secondBias">第二层偏置，可为空。</param>
    /// <param name="output">最终输出；返回 false 时保持不变。</param>
    /// <returns>是否已完成整个投影链。</returns>
    public bool TryReluProjection(float[] input, int frames, Weight first, Weight? firstBias, Weight second, Weight? secondBias, float[] output)
    {
        if (!_reluProjectionEnabled || frames <= 0 || first.Values is null || second.Values is null
            || first.FloatInterleaved || second.FloatInterleaved || first.IsTransient || second.IsTransient
            || first.Columns <= 0 || first.Rows <= 0 || second.Rows <= 0 || first.Rows != second.Columns
            || first.Rows % 4 != 0 || (firstBias is not null && firstBias.Values is null)
            || (secondBias is not null && secondBias.Values is null)
            || (long)frames * first.Rows * first.Columns < MinimumOperations
            || (long)frames * second.Rows * second.Columns < MinimumOperations)
        {
            return false;
        }

        if (input.Length != checked(frames * first.Columns) || output.Length != checked(frames * second.Rows)
            || first.Values.Length != checked(first.Rows * first.Columns) || second.Values.Length != checked(second.Rows * second.Columns)
            || (firstBias is not null && firstBias.Values!.Length != first.Rows)
            || (secondBias is not null && secondBias.Values!.Length != second.Rows))
        {
            throw new ArgumentException("前馈投影的输入、权重或偏置形状不匹配。");
        }

        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long started = Stopwatch.GetTimestamp();
            float[] bias1 = firstBias?.Values ?? _zero;
            float[] bias2 = secondBias?.Values ?? _zero;
            Array[] keys = [first.Values, bias1, second.Values, bias2];
            if (keys.Distinct<Array>(ReferenceEqualityComparer.Instance).Sum(key => (long)key.Length * sizeof(float)) > _cacheLimit)
            {
                return false;
            }

            var weights1 = ResidentRaw(first.Values, MemoryMarshal.AsBytes(first.Values.AsSpan()), keys);
            var biases1 = ResidentRaw(bias1, MemoryMarshal.AsBytes(bias1.AsSpan()), keys);
            var weights2 = ResidentRaw(second.Values, MemoryMarshal.AsBytes(second.Values.AsSpan()), keys);
            var biases2 = ResidentRaw(bias2, MemoryMarshal.AsBytes(bias2.AsSpan()), keys);
            ulong inputBytes = checked((ulong)input.Length * sizeof(float));
            ulong outputBytes = checked((ulong)output.Length * sizeof(float));
            EnsureBuffer(ref _input, inputBytes, true);
            EnsureBuffer(ref _reluIntermediate, checked((ulong)frames * (ulong)first.Rows * sizeof(float)), false);
            EnsureBuffer(ref _output, outputBytes, false);
            EnsureBuffer(ref _readback, outputBytes, true, true);
            if (_reluSecondSet == 0)
            {
                _reluSecondSet = _device.NewDescriptorSet(_pipeline.SetLayout);
            }

            fixed (float* source = input)
            {
                _device.Upload(_input!, source, inputBytes);
            }

            _uploads += (long)inputBytes;
            _device.BindBuffer(_set, 0, _input!);
            _device.BindBuffer(_set, 1, weights1);
            _device.BindBuffer(_set, 2, biases1);
            _device.BindBuffer(_set, 3, _reluIntermediate!);
            _device.BindBuffer(_reluSecondSet, 0, _reluIntermediate!);
            _device.BindBuffer(_reluSecondSet, 1, weights2);
            _device.BindBuffer(_reluSecondSet, 2, biases2);
            _device.BindBuffer(_reluSecondSet, 3, _output!);
            uint* parameters = stackalloc uint[13];
            new Span<uint>(parameters, 13).Clear();
            parameters[0] = (uint)frames;
            parameters[1] = (uint)first.Rows;
            parameters[2] = (uint)first.Columns;
            parameters[12] = firstBias is null ? 8u : 9u;
            BeginCompute();
            RecordDispatch(_pipeline, _set, parameters, 52, ((uint)frames + 31) / 32, ((uint)first.Rows + 31) / 32);
            _device.CmdComputeBarrier(_command);
            parameters[1] = (uint)second.Rows;
            parameters[2] = (uint)second.Columns;
            parameters[12] = secondBias is null ? 0u : 1u;
            RecordDispatch(_pipeline, _reluSecondSet, parameters, 52, ((uint)frames + 31) / 32, ((uint)second.Rows + 31) / 32);
            FinishRead(output);
            _linearCalls += 2;
            _reluProjectionCalls++;
            _seconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
            return true;
        }
    }

    /// <summary>释放中间张量和第二层专用描述符。</summary>
    private void DisposeReluProjection()
    {
        _reluIntermediate?.Free();
        if (_reluSecondSet != 0)
        {
            _device.FreeDescriptorSet(_reluSecondSet);
            _reluSecondSet = 0;
        }
    }
}
