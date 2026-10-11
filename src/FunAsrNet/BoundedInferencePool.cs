using System.Threading.Channels;

namespace FunAsrNet;

/// <summary>
/// 表示推理队列已满，调用方应稍后重试，而非无限缓存音频。
/// </summary>
public sealed class InferenceQueueFullException : InvalidOperationException
{
    /// <summary>
    /// 创建队列容量不足异常。
    /// </summary>
    public InferenceQueueFullException() : base("推理队列已满，请稍后重试。")
    {
    }
}

/// <summary>
/// 以有界队列调度独立的推理上下文；同一上下文不会被两个任务同时使用。
/// </summary>
/// <typeparam name="TContext">具有独立可变状态的推理上下文类型。</typeparam>
public sealed class BoundedInferencePool<TContext> : IAsyncDisposable where TContext : class
{
    private readonly Channel<IJob> _jobs;
    private readonly Task[] _workers;
    private int _queued;
    private int _active;
    private int _disposed;
    /// <summary>
    /// 创建推理工作池；上下文必须事先初始化，且每个元素都是不同实例。
    /// </summary>
    /// <param name="contexts">已经初始化且各不相同的推理上下文集合。</param>
    /// <param name="queueCapacity">等待队列的最大任务数，不含正在执行的任务。</param>
    public BoundedInferencePool(IReadOnlyList<TContext> contexts, int queueCapacity = 8)
    {
        ArgumentNullException.ThrowIfNull(contexts);
        if (contexts.Count == 0)
        {
            throw new ArgumentException("至少需要一个推理上下文。", nameof(contexts));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queueCapacity);
        if (contexts.Any(context => context is null) || contexts.Distinct(ReferenceEqualityComparer.Instance).Count() != contexts.Count)
        {
            throw new ArgumentException("推理上下文实例不能重复。", nameof(contexts));
        }

        _jobs = Channel.CreateBounded<IJob>(new BoundedChannelOptions(queueCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleWriter = false,
            SingleReader = contexts.Count == 1,
            AllowSynchronousContinuations = false
        });
        _workers = contexts.Select(context => Task.Run(() => WorkAsync(context))).ToArray();
    }

    /// <summary>
    /// 获取当前等待任务数；读取值是瞬时统计。
    /// </summary>
    public int QueuedCount => Volatile.Read(ref _queued);
    /// <summary>
    /// 获取当前执行任务数；不代表 CPU 线程数。
    /// </summary>
    public int ActiveCount => Volatile.Read(ref _active);
    /// <summary>
    /// 获取配置的独立推理上下文数量。
    /// </summary>
    public int WorkerCount => _workers.Length;

    /// <summary>
    /// 提交同步 CPU 任务；排队可取消，已执行的任务结束前不会提前交还输入缓冲区。
    /// </summary>
    /// <typeparam name="TResult">推理任务返回的结果类型。</typeparam>
    /// <param name="operation">在独立上下文中同步运行的推理函数；函数返回前仍持有输入缓冲区。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>任务结果；队列已满时以异常通知，不阻塞请求接收线程。</returns>
    public Task<TResult> RunAsync<TResult>(Func<TContext, TResult> operation, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var job = new Job<TResult>(operation, cancellationToken);
        Interlocked.Increment(ref _queued);
        if (!_jobs.Writer.TryWrite(job))
        {
            Interlocked.Decrement(ref _queued);
            job.Reject(new InferenceQueueFullException());
        }

        return job.Task;
    }

    /// <summary>
    /// 串行消费一个上下文的任务，单次推理失败不终止工作线程。
    /// </summary>
    /// <param name="context">由当前消费循环独占使用的推理上下文。</param>
    /// <returns>队列关闭并排空时完成的任务。</returns>
    private async Task WorkAsync(TContext context)
    {
        await foreach (IJob job in _jobs.Reader.ReadAllAsync())
        {
            Interlocked.Decrement(ref _queued);
            Interlocked.Increment(ref _active);
            try
            {
                job.Execute(context);
            }
            finally
            {
                Interlocked.Decrement(ref _active);
            }
        }
    }

    /// <summary>
    /// 停止接受任务并等待已接收任务完成，防止提前释放正在使用的模型。
    /// </summary>
    /// <returns>全部工作线程退出时完成的任务。</returns>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _jobs.Writer.TryComplete();
        }

        await Task.WhenAll(_workers);
    }

    /// <summary>
    /// 推理工作线程消费的任务契约。
    /// </summary>
    private interface IJob
    {
        /// <summary>
        /// 在独占的上下文上执行任务，结果由队列项自行通知提交者。
        /// </summary>
        /// <param name="context">由工作池独占分配给该任务的推理上下文。</param>
        void Execute(TContext context);
    }

    /// <summary>
    /// 携带任务结果与取消状态的队列项。
    /// </summary>
    /// <typeparam name="TResult">推理任务返回的结果类型。</typeparam>
    private sealed class Job<TResult> : IJob
    {
        private readonly Func<TContext, TResult> _operation;
        private readonly CancellationToken _token;
        private readonly TaskCompletionSource<TResult> _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly CancellationTokenRegistration _registration;
        // 0：排队；1：执行；2：完成或已取消。CAS 保证取消不会归还仍在计算的输入。
        private int _state;
        /// <summary>
        /// 携带任务结果与取消状态的队列项。
        /// </summary>
        /// <param name="operation">在独立上下文中同步运行的推理函数。</param>
        /// <param name="token">任务取消令牌；仅排队阶段可提前结束等待。</param>
        internal Job(Func<TContext, TResult> operation, CancellationToken token)
        {
            _operation = operation;
            _token = token;
            _registration = token.UnsafeRegister(static state => ((Job<TResult>)state!).CancelQueued(), this);
        }

        /// <summary>
        /// 供提交者等待的任务结果；正在计算时不会因取消而提前交还输入。
        /// </summary>
        internal Task<TResult> Task => _completion.Task;

        /// <summary>
        /// 只取消尚未开始的任务；运行中的任务保留输入所有权直到计算结束。
        /// </summary>
        private void CancelQueued()
        {
            if (Interlocked.CompareExchange(ref _state, 2, 0) == 0)
            {
                _completion.TrySetCanceled(_token);
            }
        }

        /// <summary>
        /// 使未入队任务以异常结束，并解除取消回调。
        /// </summary>
        /// <param name="exception">需要向调用方报告的任务异常。</param>
        internal void Reject(Exception exception)
        {
            if (Interlocked.CompareExchange(ref _state, 2, 0) == 0)
            {
                _completion.TrySetException(exception);
            }

            _registration.Dispose();
        }

        /// <summary>
        /// 原子取得执行权，传播结果或异常，最后解除取消回调。
        /// </summary>
        /// <param name="context">由工作池独占分配给该任务的推理上下文。</param>
        public void Execute(TContext context)
        {
            try
            {
                if (Interlocked.CompareExchange(ref _state, 1, 0) != 0)
                {
                    return;
                }

                TResult result = _operation(context);
                if (_token.IsCancellationRequested)
                {
                    _completion.TrySetCanceled(_token);
                }
                else
                {
                    _completion.TrySetResult(result);
                }
            }
            catch (Exception exception)
            {
                _completion.TrySetException(exception);
            }
            finally
            {
                Interlocked.Exchange(ref _state, 2);
                _registration.Dispose();
            }
        }
    }
}
