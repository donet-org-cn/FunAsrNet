using FunAsrNet.Vulkan.Native;

namespace FunAsrNet.Vulkan;

/// <summary>按输入帧数选择原精度 FP32 矩阵或矩阵向量管线，保留通用卷积回退。</summary>
public sealed unsafe partial class VulkanFloatBackend
{
    private VkPipeline? _floatLinearPipeline, _floatVectorPipeline, _floatWideVectorPipeline;
    private nint _floatLinearShader, _floatVectorShader, _floatWideVectorShader;
    private long _floatLinearDispatches, _floatVectorDispatches, _floatWideVectorDispatches;
    private readonly bool _wideVectors = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_WIDE") != "0";
    private readonly bool _specializedFloat = Environment.GetEnvironmentVariable("FUNASRNET_VULKAN_SPECIALIZED_FLOAT") != "0";

    /// <summary>为线性计算选择与通用描述符布局兼容的管线。</summary>
    /// <param name="pipeline">输入的通用管线，按形状替换为专用管线。</param>
    /// <param name="parameters">十三个只读推送参数，零号模式表示线性投影。</param>
    /// <param name="groupsX">输出的第一维工作组数量。</param>
    /// <param name="groupsY">输出的第二维工作组数量。</param>
    private void SelectFloatPipeline(ref VkPipeline pipeline, uint* parameters, ref uint groupsX, ref uint groupsY)
    {
        if (!_specializedFloat || !ReferenceEquals(pipeline, _pipeline) || parameters[3] != 0 || (parameters[12] & 4) != 0)
        {
            return;
        }
        uint size = _device.SubgroupSize;
        if (parameters[0] <= 4 && _device.SubgroupSizeControl && _device.ComputeSubgroupSize && _device.ComputeFullSubgroups
            && (_device.SubgroupOps & 0x10) != 0 && (_device.SubgroupStages & 0x20) != 0
            && size >= 8 && size <= 128 && (size & (size - 1)) == 0 && size >= _device.SubgroupMin && size <= _device.SubgroupMax)
        {
            bool wide = size >= 32 && parameters[2] >= 1024 && parameters[1] <= 262140 && _wideVectors;
            if (wide && _floatWideVectorPipeline is null)
            {
                _floatWideVectorShader = _device.NewShaderModule(FloatVectorShader.WideBinary());
                _floatWideVectorPipeline = _device.NewPipeline(_floatWideVectorShader, 4, 52, size);
            }
            else if (!wide && _floatVectorPipeline is null)
            {
                _floatVectorShader = _device.NewShaderModule(FloatVectorShader.Binary());
                _floatVectorPipeline = _device.NewPipeline(_floatVectorShader, 4, 52, size);
            }
            pipeline = wide ? _floatWideVectorPipeline! : _floatVectorPipeline!;
            groupsX = wide ? (parameters[1] + 3) / 4 : (parameters[1] + 15) / 16;
            if (wide)
            {
                _floatWideVectorDispatches++;
            }
            groupsY = parameters[0];
            _floatVectorDispatches++;
        }
        else if (parameters[0] >= 16 && parameters[1] >= 64 && parameters[2] >= 32)
        {
            if (_floatLinearPipeline is null)
            {
                _floatLinearShader = _device.NewShaderModule(FloatLinearShader.Binary());
                _floatLinearPipeline = _device.NewPipeline(_floatLinearShader, 4, 52);
            }
            pipeline = _floatLinearPipeline;
            groupsX = (parameters[0] + 31) / 32;
            groupsY = (parameters[1] + 63) / 64;
            _floatLinearDispatches++;
        }
    }

    /// <summary>在设备空闲后释放按需创建的浮点专用管线及模块。</summary>
    private void DisposeFloatPipelines()
    {
        foreach (var item in new[] { (_floatLinearPipeline, _floatLinearShader), (_floatVectorPipeline, _floatVectorShader), (_floatWideVectorPipeline, _floatWideVectorShader) })
        {
            if (item.Item1 is { } pipeline)
            {
                Vk.vkDestroyPipeline(_device.Device, pipeline.Pipeline, null);
                Vk.vkDestroyPipelineLayout(_device.Device, pipeline.Layout, null);
                Vk.vkDestroyDescriptorSetLayout(_device.Device, pipeline.SetLayout, null);
            }
            if (item.Item2 != 0)
            {
                Vk.vkDestroyShaderModule(_device.Device, item.Item2, null);
            }
        }
    }
}
