using FunAsrNet;

/// <summary>
/// 验证调度上限、可变上下文隔离、取消时的借用内存所有权以及异常恢复。
/// </summary>
internal static class ConcurrencyChecks
{
    /// <summary>
    /// 执行异步工作并传播完成、异常或取消状态。
    /// </summary>
    /// <param name="check">记录断言结果的回调。</param>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    internal static async Task RunAsync(Action<bool, string> check)
    {
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await using (var pool = new BoundedInferencePool<object>([new object()], 1))
        {
            using var release = new ManualResetEventSlim();
            var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var first = pool.RunAsync(_ =>
                {
                    started.SetResult();
                    if (!release.Wait(10000))
                    {
                        throw new TimeoutException();
                    }

                    return 7;
                });
            try
            {
                await started.Task.WaitAsync(deadline.Token);
                using var cancelled = new CancellationTokenSource();
                bool executed = false;
                Task<int> queued = pool.RunAsync(_ =>
                    {
                        executed = true;
                        return 9;
                    }, cancelled.Token);
                bool full = false;
                try
                {
                    await pool.RunAsync(_ => 11);
                }
                catch (InferenceQueueFullException)
                {
                    full = true;
                }

                check(full, "Bounded queue rejects overload");
                cancelled.Cancel();
                bool canceled = false;
                try
                {
                    await queued;
                }
                catch (OperationCanceledException)
                {
                    canceled = true;
                }

                check(canceled && !executed, "Queued cancellation never reads caller input");
                release.Set();
                check(await first == 7, "Active work unaffected by queued cancellation");
            }
            finally
            {
                release.Set();
            }
        }

        await using (var pool = new BoundedInferencePool<object>([new object()], 4))
        {
            using var release = new ManualResetEventSlim();
            using var cancel = new CancellationTokenSource();
            var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            using var audio = new PooledAudioBuffer(16);
            audio.Memory.Span.Fill(0.25f);
            var running = pool.RunAsync(
                _ =>
                {
                    started.SetResult();
                    if (!release.Wait(10000))
                    {
                        throw new TimeoutException();
                    }

                    return audio.Memory.Span[0];
                },
                cancel.Token);
            try
            {
                await started.Task.WaitAsync(deadline.Token);
                cancel.Cancel();
                check(!running.IsCompleted, "Running cancellation retains borrowed audio until inference exits");
                release.Set();
                bool canceled = false;
                try
                {
                    await running;
                }
                catch (OperationCanceledException)
                {
                    canceled = true;
                }

                check(canceled, "Completed canceled work does not publish a result");
                try
                {
                    await pool.RunAsync<int>(_ => throw new ArithmeticException());
                }
                catch (ArithmeticException)
                {
                }

                check(await pool.RunAsync(_ => 42) == 42, "Worker survives a failing inference");
            }
            finally
            {
                release.Set();
            }
        }

        await using (var pool = new BoundedInferencePool<Counter>([new(), new()], 32))
        {
            using var entered = new CountdownEvent(2);
            using var release = new ManualResetEventSlim();
            var jobs = Enumerable.Range(0, 2).Select(_ => pool.RunAsync(context =>
                {
                    if (Interlocked.Increment(ref context.Active) != 1)
                    {
                        throw new InvalidOperationException("Context shared concurrently");
                    }

                    entered.Signal();
                    try
                    {
                        if (!release.Wait(10000))
                        {
                            throw new TimeoutException();
                        }

                        return 1;
                    }
                    finally
                    {
                        Interlocked.Decrement(ref context.Active);
                    }
                })).ToArray();
            try
            {
                check(await Task.Run(() => entered.Wait(10000)), "Distinct contexts execute concurrently");
                release.Set();
                await Task.WhenAll(jobs).WaitAsync(deadline.Token);
                check(pool.WorkerCount == 2, "Configured inference worker count");
            }
            finally
            {
                release.Set();
            }
        }

        var owner = new PooledAudioBuffer(3);
        check(owner.Memory.Length == 3, "Pooled audio never exposes excess rented capacity");
        owner.Dispose();
        owner.Dispose();
        bool disposed = false;
        try
        {
            _ = owner.Memory;
        }
        catch (ObjectDisposedException)
        {
            disposed = true;
        }

        check(disposed, "Pooled audio rejects access after return");
    }

    /// <summary>
    /// 并发测试上下文，用于发现同一上下文被重复使用。
    /// </summary>
    private sealed class Counter
    {
        /// <summary>
        /// 当前上下文正在执行的测试任务数。
        /// </summary>
        internal int Active;
    }
}
