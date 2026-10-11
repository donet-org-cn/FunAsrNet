using FunAsrNet;

/// <summary>检查共享线程组从休眠恢复时的通知握手及多个调用方的任务隔离。</summary>
internal static class KernelWorkerChecks
{
    /// <summary>交替空闲和连续任务，验证任务通知不会在休眠边界丢失。</summary>
    /// <param name="check">接收断言结果与检查名称的测试回调。</param>
    /// <returns>全部工作线程按时完成且各批结果一致后结束的任务。</returns>
    internal static async Task RunAsync(Action<bool, string> check)
    {
        var workers = CpuKernelWorkers.Get(4);
        var tasks = Enumerable.Range(0, 3).Select(_ => Task.Run(() =>
        {
            for (int iteration = 0; iteration < 80; iteration++)
            {
                // 超过五毫秒的休眠阈值，再立即发布两个批次以覆盖醒来和短暂轮询两条路径。
                Thread.Sleep(8);
                int[] values = new int[32];
                workers.For(values.Length, index => values[index] = index + iteration);
                workers.For(values.Length, index => values[index]++);
                if (values.Where((value, index) => value != index + iteration + 1).Any())
                {
                    return false;
                }
            }
            return true;
        })).ToArray();
        bool[] results = await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(20));
        check(results.All(result => result), "Shared kernel workers wake after sleeping and isolate concurrent callers");
    }
}
