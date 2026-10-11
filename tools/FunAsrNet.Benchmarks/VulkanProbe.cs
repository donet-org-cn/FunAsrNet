using System.Runtime.InteropServices;

namespace FunAsrNet.Benchmarks;

/// <summary>查询系统 Vulkan 加载器和计算队列，不把软件设备自动当作可用 GPU 后端。</summary>
internal static class VulkanProbe
{
    /// <summary>Vulkan 1.0 实例创建结构，Linux ARM64 使用自然指针对齐。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct InstanceInfo
    {
        public int Type;
        public nint Next;
        public uint Flags;
        public nint Application;
        public uint LayerCount;
        public nint LayerNames;
        public uint ExtensionCount;
        public nint ExtensionNames;
    }

    /// <summary>计算队列属性及最小图像传输粒度。</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct QueueFamily
    {
        public uint Flags, Count, TimestampBits, Width, Height, Depth;
    }

    /// <summary>向系统加载器请求创建 Vulkan 实例。</summary>
    /// <param name="info">未启用可选层或扩展的实例配置。</param>
    /// <param name="allocator">为零时使用系统默认分配器。</param>
    /// <param name="instance">返回的实例句柄，成功后须销毁。</param>
    /// <returns>零表示成功，负值为驱动或加载器错误码。</returns>
    [DllImport("libvulkan.so.1", CallingConvention = CallingConvention.Cdecl)]
    private static extern int vkCreateInstance(ref InstanceInfo info, nint allocator, out nint instance);

    /// <summary>枚举加载器提供的物理设备，包括可能存在的软件实现。</summary>
    /// <param name="instance">已创建的 Vulkan 实例。</param>
    /// <param name="count">输入数组容量或返回设备数量。</param>
    /// <param name="devices">输出设备数组，为空时只查询数量。</param>
    /// <returns>零表示枚举完成，一表示数组不足，负值为错误码。</returns>
    [DllImport("libvulkan.so.1", CallingConvention = CallingConvention.Cdecl)]
    private static extern int vkEnumeratePhysicalDevices(nint instance, ref uint count, [Out] nint[]? devices);

    /// <summary>读取设备的队列簇属性。</summary>
    /// <param name="device">枚举得到的有效设备句柄。</param>
    /// <param name="count">输入数组容量或返回队列簇数量。</param>
    /// <param name="families">输出属性数组，为空时只查询数量。</param>
    [DllImport("libvulkan.so.1", CallingConvention = CallingConvention.Cdecl)]
    private static extern void vkGetPhysicalDeviceQueueFamilyProperties(nint device, ref uint count, [Out] QueueFamily[]? families);

    /// <summary>释放由本探针创建的 Vulkan 实例。</summary>
    /// <param name="instance">已创建且未销毁的实例句柄。</param>
    /// <param name="allocator">与创建时一致，为零表示默认分配器。</param>
    [DllImport("libvulkan.so.1", CallingConvention = CallingConvention.Cdecl)]
    private static extern void vkDestroyInstance(nint instance, nint allocator);

    /// <summary>输出设备枚举与计算队列信息，不运行模型或改变系统驱动配置。</summary>
    /// <param name="emit">接收诊断 JSON 记录的输出回调。</param>
    internal static void Run(Action<object> emit)
    {
        if (!OperatingSystem.IsLinux() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64
            || Marshal.SizeOf<InstanceInfo>() != 64 || Marshal.SizeOf<QueueFamily>() != 24)
        {
            throw new PlatformNotSupportedException("此 Vulkan ABI 探针要求 Linux ARM64。");
        }

        var info = new InstanceInfo { Type = 1 };
        int code = vkCreateInstance(ref info, 0, out var instance);
        emit(new
        {
            kind = "vulkan-create",
            code,
            icd = Environment.GetEnvironmentVariable("VK_ICD_FILENAMES"),
            productBackend = false
        });
        if (code != 0)
        {
            Environment.ExitCode = 1;
            return;
        }

        try
        {
            uint count = 0;
            code = vkEnumeratePhysicalDevices(instance, ref count, null);
            if (code != 0 || count > 64)
            {
                throw new InvalidOperationException($"枚举 Vulkan 设备失败：{code}，数量 {count}");
            }

            var devices = new nint[count];
            code = vkEnumeratePhysicalDevices(instance, ref count, devices);
            if (code != 0)
            {
                throw new InvalidOperationException($"读取 Vulkan 设备列表失败：{code}");
            }

            for (int index = 0; index < count; index++)
            {
                uint families = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(devices[index], ref families, null);
                if (families > 64)
                {
                    throw new InvalidOperationException("Vulkan 返回异常队列数量。");
                }

                var properties = new QueueFamily[families];
                vkGetPhysicalDeviceQueueFamilyProperties(devices[index], ref families, properties);
                emit(new
                {
                    kind = "vulkan-queues",
                    index,
                    families,
                    computeQueues = properties.Where(item => (item.Flags & 2) != 0).Sum(item => (long)item.Count),
                    hardwareIdentityVerified = false
                });
            }

            emit(new
            {
                kind = "vulkan-devices",
                code,
                count,
                hardwareIdentityVerified = false
            });
        }
        finally
        {
            vkDestroyInstance(instance, 0);
        }
    }
}
