using System.Collections.Concurrent;
using System.Diagnostics;

namespace FunAsrNet;
// 复用同步线程组；相邻算子间短暂轮询以减少唤醒开销，
// 空闲超过 5 ms 后休眠，避免无请求时持续占用 CPU。
/// <summary>
/// 复用同步算子线程组；短时间轮询后休眠，避免空闲服务持续占用 CPU。
/// </summary>
internal sealed class CpuKernelWorkers
{
    private static readonly ConcurrentDictionary<int, Lazy<CpuKernelWorkers>> Teams = new();
    [ThreadStatic]
    private static bool _inside;
    private readonly object _serialize = new();
    private readonly AutoResetEvent[] _wake;
    private readonly int[] _sleeping;
    private readonly int[] _cpus;
    private Action<int>? _body;
    private Exception? _error;
    private int _count, _sequence, _remaining;
    /// <summary>
    /// 复用同步算子线程组；短时间轮询后休眠，避免空闲服务持续占用 CPU。
    /// </summary>
    /// <param name="threads">算子使用的最大计算线程数。</param>
    private CpuKernelWorkers(int threads)
    {
        var cpus = CpuAffinity.AvailableCpus();
        _cpus = cpus.Length >= threads ? cpus.TakeLast(threads).ToArray() : [];
        _wake = new AutoResetEvent[threads - 1];
        _sleeping = new int[threads - 1];
        for (int i = 0; i < threads - 1; ++i)
        {
            int worker = i;
            _wake[i] = new(false);
            new Thread(() => Worker(worker))
            {
                IsBackground = true,
                Name = $"RK-Kernel-{threads}-{i}"

            }.Start();
        }
    }

    /// <summary>
    /// 获取指定线程数对应的可复用算子线程组。
    /// </summary>
    /// <param name="threads">算子使用的最大计算线程数。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public static CpuKernelWorkers Get(int threads) => Teams.GetOrAdd(threads, n => new(() => new(n))).Value;
    /// <summary>
    /// 向线程组发布一次同步算子任务，并等待所有参与者完成。
    /// </summary>
    /// <param name="worker">线程工作项或线程编号。</param>
    private void Run(int worker)
    {
        bool previous = _inside;
        _inside = true;
        try
        {
            int threads = _wake.Length + 1;
            for (int i = (int)((long)_count * worker / threads), end = (int)((long)_count * (worker + 1) / threads); i < end; ++i)
            {
                _body!(i);
            }
        }
        catch (Exception ex)
        {
            Interlocked.CompareExchange(ref _error, ex, null);
        }
        finally
        {
            _inside = previous;
        }
    }

    /// <summary>
    /// 消费当前发布的算子任务，空闲达到阈值后休眠。
    /// </summary>
    /// <param name="worker">线程工作项或线程编号。</param>
    private void Worker(int worker)
    {
        using var affinity = _cpus.Length == 0 ? null : CpuAffinity.Pin(_cpus[worker]);
        int seen = 0;
        while (true)
        {
            long idleUntil = Stopwatch.GetTimestamp() + Stopwatch.Frequency / 200;
            int sequence;
            while ((sequence = Volatile.Read(ref _sequence)) == seen)
            {
                if (Stopwatch.GetTimestamp() < idleUntil)
                {
                    Thread.SpinWait(16);
                    continue;
                }

                Volatile.Write(ref _sleeping[worker], 1);
                if (Volatile.Read(ref _sequence) == seen)
                {
                    _wake[worker].WaitOne();
                }

                Volatile.Write(ref _sleeping[worker], 0);
            }

            seen = sequence;
            Run(worker);
            Interlocked.Decrement(ref _remaining);
        }
    }

    /// <summary>
    /// 在配置允许时把当前调用线程绑定到算子线程组对应的 CPU。
    /// </summary>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    internal IDisposable? PinCaller() => _cpus.Length == 0 ? null : CpuAffinity.Pin(_cpus[^1]);
    /// <summary>
    /// 按线程分配索引区间，并等待该批索引上的操作全部完成。
    /// </summary>
    /// <param name="count">需要处理的元素或任务数量。</param>
    /// <param name="body">对分配到的索引执行的同步操作。</param>
    public void For(int count, Action<int> body)
    {
        if (_inside)
        {
            for (int i = 0; i < count; ++i)
            {
                body(i);
            }

            return;
        }

        lock (_serialize)
        {
            _body = body;
            _count = count;
            _error = null;
            _remaining = _wake.Length;
            Volatile.Write(ref _sequence, unchecked(_sequence + 1));
            for (int i = 0; i < _wake.Length; ++i)
            {
                if (Volatile.Read(ref _sleeping[i]) != 0)
                {
                    _wake[i].Set();
                }
            }

            Run(_wake.Length);
            while (Volatile.Read(ref _remaining) != 0)
            {
                Thread.SpinWait(16);
            }

            _body = null;
            if (_error is not null)
            {
                throw new AggregateException(_error);
            }
        }
    }
}
