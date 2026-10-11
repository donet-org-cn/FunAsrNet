using System.Diagnostics;

namespace FunAsrNet;

/// <summary>
/// 读取 Linux CPU 亲和性，并在可用时绑定当前计算线程。
/// </summary>
internal static class CpuAffinity
{
    /// <summary>
    /// 读取当前进程允许使用的 CPU 编号；不支持的平台返回空列表。
    /// </summary>
    /// <returns>按输出顺序排列的整数数组。</returns>
    internal static int[] AvailableCpus()
    {
        if (!OperatingSystem.IsLinux() || Environment.GetEnvironmentVariable("FUNASRNET_PIN_THREADS") != "1")
        {
            return [];
        }

        using var process = Process.GetCurrentProcess();
        ulong mask = (ulong)(nuint)process.ProcessorAffinity;
        return Enumerable.Range(0, IntPtr.Size * 8).Where(cpu => (mask & (1UL << cpu)) != 0).ToArray();
    }

    /// <summary>
    /// 尝试将当前 Linux 线程绑定到指定 CPU，并返回恢复原绑定的作用域。
    /// </summary>
    /// <param name="cpu">操作系统 CPU 编号。</param>
    /// <returns>用于恢复原亲和性的作用域；平台不支持时为空。</returns>
    internal static IDisposable? Pin(int cpu)
    {
        if (!OperatingSystem.IsLinux())
        {
            return null;
        }

        // Linux 通过该 procfs 链接暴露调用方线程编号，
        // 运行时的托管 Process API 可据此访问操作系统亲和性。
        var link = Directory.ResolveLinkTarget("/proc/thread-self", true) ?? throw new IOException("No Linux thread-self link");
        int tid = int.Parse(Path.GetFileName(link.FullName));
        using var thread = Process.GetProcessById(tid);
        nint original = thread.ProcessorAffinity;
        thread.ProcessorAffinity = (nint)(1UL << cpu);
        return new Restore(tid, original);
    }

    /// <summary>
    /// 保存临时 CPU 绑定之前的状态，在释放时恢复。
    /// </summary>
    /// <param name="tid">需要恢复亲和性的操作系统线程编号。</param>
    /// <param name="mask">需要恢复的 CPU 亲和性位掩码。</param>
    private sealed class Restore(int tid, nint mask) : IDisposable
    {
        /// <summary>
        /// 释放持有的资源或恢复进入作用域之前的状态；调用者不应再使用已释放资源。
        /// </summary>
        public void Dispose()
        {
            if (!OperatingSystem.IsLinux())
            {
                return;
            }

            using var thread = Process.GetProcessById(tid);
            thread.ProcessorAffinity = mask;
        }
    }
}
