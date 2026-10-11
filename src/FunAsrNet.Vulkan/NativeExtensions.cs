using System.Runtime.InteropServices;

namespace FunAsrNet.Vulkan.Native;

/// <summary>补充本项目生命周期管理需要的 Vulkan 销毁接口。</summary>
internal static unsafe partial class Vk
{
    /// <summary>释放计算管线布局，调用前确保相关提交已经完成。</summary>
    /// <param name="device">布局所属的逻辑设备。</param>
    /// <param name="layout">待销毁的布局句柄。</param>
    /// <param name="allocator">与创建时一致的分配器，为空使用默认分配器。</param>
    [LibraryImport(LibName)]
    internal static partial void vkDestroyPipelineLayout(nint device, nint layout, void* allocator);

    /// <summary>释放描述符集合布局。</summary>
    /// <param name="device">集合布局所属的逻辑设备。</param>
    /// <param name="layout">待销毁的布局句柄。</param>
    /// <param name="allocator">与创建时一致的分配器，为空使用默认分配器。</param>
    [LibraryImport(LibName)]
    internal static partial void vkDestroyDescriptorSetLayout(nint device, nint layout, void* allocator);
}
