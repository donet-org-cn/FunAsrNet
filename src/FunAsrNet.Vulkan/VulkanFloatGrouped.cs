using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using FunAsrNet.Vulkan.Native;

namespace FunAsrNet.Vulkan;

/// <summary>合并原精度 FP32 投影，只在 GPU 驻留合并权重，CPU 不保留模型副本。</summary>
public sealed unsafe partial class VulkanFloatBackend
{
    private long _floatGroupCalls;

    /// <summary>共享输入上传与同步，合并只读行主序浮点矩阵。</summary>
    /// <param name="input">全部投影共享的只读帧主序输入。</param>
    /// <param name="frames">有效输入帧数。</param>
    /// <param name="weights">列数一致且发布后不变的权重数组。</param>
    /// <param name="outputs">按原投影顺序拆分的独立结果，不支持时为空。</param>
    /// <returns>是否完成所有投影及 GPU 同步。</returns>
    private bool TryFloatLinearMany(float[] input, int frames, Weight[] weights, out float[][]? outputs)
    {
        outputs = null;
        int rows = weights.Sum(weight => weight.Rows), columns = weights[0].Columns;
        if ((long)frames * columns * rows < MinimumOperations || (long)rows * columns * sizeof(float) + sizeof(float) > _cacheLimit)
        {
            return false;
        }
        if (input.Length != checked(frames * columns) || weights.Any(weight => weight.Values!.Length != checked(weight.Rows * columns)))
        {
            throw new ArgumentException("浮点分组投影尺寸不一致。");
        }
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long started = Stopwatch.GetTimestamp();
            var resident = ResidentFloatGroup(weights, columns, rows);
            var bias = ResidentRaw(_zero, MemoryMarshal.AsBytes(_zero.AsSpan()), [weights]);
            var result = TensorWorkspace.Allocate(checked(frames * rows));
            try
            {
                ulong inputBytes = checked((ulong)input.Length * sizeof(float));
                ulong outputBytes = checked((ulong)result.Length * sizeof(float));
                EnsureBuffer(ref _input, inputBytes, true);
                EnsureBuffer(ref _output, outputBytes, false);
                EnsureBuffer(ref _readback, outputBytes, true, true);
                fixed (float* source = input)
                {
                    _device.Upload(_input!, source, inputBytes);
                }
                _uploads += (long)inputBytes;
                _device.BindBuffer(_set, 0, _input!);
                _device.BindBuffer(_set, 1, resident);
                _device.BindBuffer(_set, 2, bias);
                _device.BindBuffer(_set, 3, _output!);
                uint* parameters = stackalloc uint[13];
                new Span<uint>(parameters, 13).Clear();
                parameters[0] = (uint)frames;
                parameters[1] = (uint)rows;
                parameters[2] = (uint)columns;
                DispatchAndRead(_pipeline, _set, parameters, 52, ((uint)frames + 31) / 32, ((uint)rows + 31) / 32, result);
                outputs = new float[weights.Length][];
                int offset = 0;
                for (int which = 0; which < weights.Length; which++)
                {
                    int count = weights[which].Rows;
                    outputs[which] = TensorWorkspace.Allocate(checked(frames * count));
                    for (int time = 0; time < frames; time++)
                    {
                        result.AsSpan(time * rows + offset, count).CopyTo(outputs[which].AsSpan(time * count));
                    }
                    offset += count;
                }
                _linearCalls++;
                _floatGroupCalls++;
                _seconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
                return true;
            }
            finally
            {
                TensorWorkspace.Release(result);
            }
        }
    }

    /// <summary>首次上传时租借一个合并缓冲，缓存命中后只使用 GPU 驻留副本。</summary>
    /// <param name="weights">用作稳定缓存键的只读权重数组。</param>
    /// <param name="columns">所有投影共有的列数。</param>
    /// <param name="rows">合并后的总输出行数。</param>
    /// <returns>由本后端拥有的设备缓冲，只能在实例锁内使用。</returns>
    private VkBuffer ResidentFloatGroup(Weight[] weights, int columns, int rows)
    {
        if (_weights.TryGetValue(weights, out var cached))
        {
            cached.Used = ++_sequence;
            _hits++;
            return cached.Buffer;
        }
        int length = checked(columns * rows);
        var values = ArrayPool<float>.Shared.Rent(length);
        try
        {
            int offset = 0;
            foreach (var weight in weights)
            {
                weight.Values!.CopyTo(values, offset);
                offset += weight.Values.Length;
            }
            return ResidentRaw(weights, MemoryMarshal.AsBytes(values.AsSpan(0, length)), []);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }
}
