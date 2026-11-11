using System.Diagnostics;
using FunAsrNet.Vulkan.Native;

namespace FunAsrNet.Vulkan;

/// <summary>用单次提交执行编码器注意力，复用上传暂存且不占用只读权重缓存。</summary>
public sealed unsafe partial class VulkanFloatBackend
{
    private VkPipeline? _attentionPipeline, _attentionTiledPipeline;
    private nint _attentionShader, _attentionTiledShader, _attentionSet;
    private VkBuffer? _attentionStaging, _attentionData;
    private long _attentionCalls;
    private readonly bool _attentionTiledEnabled = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_ATTENTION_TILE") != "0";
    private readonly bool _attentionEnabled = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_ATTENTION") != "0";

    /// <summary>计算原精度非因果注意力，形状或设备能力不支持时保留 CPU 回退。</summary>
    /// <param name="query">帧主序只读查询。</param>
    /// <param name="key">与查询形状相同的只读键。</param>
    /// <param name="value">与查询形状相同的只读值。</param>
    /// <param name="frames">有效帧数，一至四千零九十六。</param>
    /// <param name="width">每帧的总通道数。</param>
    /// <param name="heads">注意力头数，每头最多二百五十六通道。</param>
    /// <param name="output">成功后完整写入的独立 FP32 输出。</param>
    /// <returns>已完整处理时返回真，否则输出保持不变。</returns>
    public bool TryAttention(float[] query, float[] key, float[] value, int frames, int width, int heads, float[] output)
    {
        uint subgroup = _device.SubgroupSize;
        if (!_attentionEnabled || frames < 1 || frames > 4096 || (MinimumOperations > 0 && frames < 160) || heads < 1 || heads > 65535 || width < heads || width % heads != 0
            || width / heads > 256 || (long)frames * frames * width < MinimumOperations
            || !_device.SubgroupSizeControl || !_device.ComputeSubgroupSize || !_device.ComputeFullSubgroups
            || (_device.SubgroupOps & 0x10) == 0 || (_device.SubgroupStages & 0x20) == 0
            || subgroup < 8 || subgroup > 128 || (subgroup & (subgroup - 1)) != 0
            || subgroup < _device.SubgroupMin || subgroup > _device.SubgroupMax || _device.MaxSharedMemory < 17920)
        {
            return false;
        }
        int count = checked(frames * width);
        if (query.Length != count || key.Length != count || value.Length != count || output.Length != count)
        {
            throw new ArgumentException("GPU 注意力的输入、输出形状不一致。");
        }
        ulong bytes = checked((ulong)count * sizeof(float)), inputBytes = checked(bytes * 3);
        if (inputBytes > _device.MaxStorageRange)
        {
            return false;
        }
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long started = Stopwatch.GetTimestamp();
            bool tiled = frames <= 1024 && _device.MaxSharedMemory >= 22528
                && _attentionTiledEnabled;
            if (tiled && _attentionTiledPipeline is null)
            {
                _attentionTiledShader = _device.NewShaderModule(AttentionTiledShader.Binary());
                _attentionTiledPipeline = _device.NewPipeline(_attentionTiledShader, 2, 16, subgroup);
            }
            else if (!tiled && _attentionPipeline is null)
            {
                _attentionShader = _device.NewShaderModule(AttentionShader.Binary());
                _attentionPipeline = _device.NewPipeline(_attentionShader, 2, 16, subgroup);
            }
            var pipeline = tiled ? _attentionTiledPipeline! : _attentionPipeline!;
            if (_attentionSet == 0)
            {
                _attentionSet = _device.NewDescriptorSet(pipeline.SetLayout);
            }
            EnsureBuffer(ref _attentionStaging, inputBytes, true, true);
            EnsureBuffer(ref _attentionData, inputBytes, false);
            EnsureBuffer(ref _output, bytes, false);
            EnsureBuffer(ref _readback, bytes, true, true);
            void* mapped = _attentionStaging!.Map();
            try
            {
                var target = new Span<float>(mapped, checked(count * 3));
                query.CopyTo(target);
                key.CopyTo(target.Slice(count));
                value.CopyTo(target.Slice(count * 2));
                _attentionStaging.Flush(0, inputBytes);
            }
            finally
            {
                _attentionStaging.Unmap();
            }
            _uploads += (long)inputBytes;
            _device.BindBuffer(_attentionSet, 0, _attentionData!);
            _device.BindBuffer(_attentionSet, 1, _output!);
            BeginCompute();
            var hostBarrier = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier, SrcAccessMask = VkConst.AccessHostWrite, DstAccessMask = VkConst.AccessTransferRead };
            Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageHost, VkConst.PipelineStageTransfer, 0, 1, &hostBarrier, 0, null, 0, null);
            var copy = new Vk.VkBufferCopy { Size = inputBytes };
            Vk.vkCmdCopyBuffer(_command, _attentionStaging.Buffer, _attentionData!.Buffer, 1, &copy);
            var copiedBarrier = new Vk.VkMemoryBarrier { SType = VkConst.StMemoryBarrier, SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead };
            Vk.vkCmdPipelineBarrier(_command, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0, 1, &copiedBarrier, 0, null, 0, null);
            uint* parameters = stackalloc uint[4] { (uint)frames, (uint)width, (uint)heads, BitConverter.SingleToUInt32Bits(MathF.Sqrt(width / heads)) };
            RecordDispatch(pipeline, _attentionSet, parameters, 16, tiled ? ((uint)frames + 3) / 4 : (uint)frames, (uint)heads);
            FinishRead(output);
            _attentionCalls++;
            _seconds += Stopwatch.GetElapsedTime(started).TotalSeconds;
            return true;
        }
    }

    /// <summary>设备空闲后释放注意力管线与临时缓冲。</summary>
    private void DisposeAttention()
    {
        _attentionStaging?.Free();
        _attentionData?.Free();
        if (_attentionSet != 0)
        {
            _device.FreeDescriptorSet(_attentionSet);
        }
        if (_attentionPipeline is not null)
        {
            Vk.vkDestroyPipeline(_device.Device, _attentionPipeline.Pipeline, null);
            Vk.vkDestroyPipelineLayout(_device.Device, _attentionPipeline.Layout, null);
            Vk.vkDestroyDescriptorSetLayout(_device.Device, _attentionPipeline.SetLayout, null);
        }
        if (_attentionTiledPipeline is not null)
        {
            Vk.vkDestroyPipeline(_device.Device, _attentionTiledPipeline.Pipeline, null);
            Vk.vkDestroyPipelineLayout(_device.Device, _attentionTiledPipeline.Layout, null);
            Vk.vkDestroyDescriptorSetLayout(_device.Device, _attentionTiledPipeline.SetLayout, null);
        }
        if (_attentionTiledShader != 0)
        {
            Vk.vkDestroyShaderModule(_device.Device, _attentionTiledShader, null);
        }
        if (_attentionShader != 0)
        {
            Vk.vkDestroyShaderModule(_device.Device, _attentionShader, null);
        }
    }
}
