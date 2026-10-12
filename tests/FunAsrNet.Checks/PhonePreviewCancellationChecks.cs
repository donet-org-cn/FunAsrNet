using System.Collections.Concurrent;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证过期预览排队取消、运行中音频所有权和最终句事件顺序。</summary>
internal static class PhonePreviewCancellationChecks
{
    /// <summary>使用确定性微型 VAD 与受控推理任务，避免依赖正式模型或 CPU 基准。</summary>
    /// <param name="check">记录断言及其说明的回调。</param>
    /// <returns>全部调度检查完成后结束的任务。</returns>
    internal static async Task RunAsync(Action<bool, string> check)
    {
        var model = CreateSpeechVad();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await CheckQueuedAsync(model, check, deadline.Token);
        await CheckRunningAsync(model, check, deadline.Token);
        await CheckReplacementAsync(model, check, deadline.Token);
        await CheckStopAsync(model, check, deadline.Token);
    }

    /// <summary>占用全局池，确认最终句能取消排队预览而不执行多余模型调用。</summary>
    /// <param name="model">始终判定语音的确定性微型 VAD。</param>
    /// <param name="check">记录断言及其说明的回调。</param>
    /// <param name="deadline">检查超时令牌。</param>
    /// <returns>检查完成后结束的任务。</returns>
    private static async Task CheckQueuedAsync(Gguf model, Action<bool, string> check, CancellationToken deadline)
    {
        await using var pool = new BoundedInferencePool<object>([new object()], 4);
        using var release = new ManualResetEventSlim();
        var busy = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var previewQueued = new TaskCompletionSource<CancellationToken>(TaskCreationOptions.RunContinuationsAsynchronously);
        var finalQueued = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var events = new ConcurrentQueue<SpeechEvent>();
        int submitted = 0, executed = 0;
        var blocker = pool.RunAsync(_ =>
        {
            busy.SetResult();
            if (!release.Wait(10000))
            {
                throw new TimeoutException();
            }
            return 1;
        });
        await busy.Task.WaitAsync(deadline);
        await using var stream = new PhoneStream((pcm, token) =>
        {
            int call = Interlocked.Increment(ref submitted);
            var result = pool.RunAsync(_ =>
            {
                Interlocked.Increment(ref executed);
                return Result("final", pcm.Length);
            }, token);
            if (call == 1)
            {
                previewQueued.SetResult(token);
            }
            else
            {
                finalQueued.TrySetResult();
            }
            return result;
        }, model, message => { events.Enqueue(message); return Task.CompletedTask; }, true);
        try
        {
            FeedSeconds(stream, 2);
            CancellationToken previewToken = await previewQueued.Task.WaitAsync(deadline);
            Task complete = stream.CompleteAsync();
            await finalQueued.Task.WaitAsync(deadline);
            check(previewToken.IsCancellationRequested && executed == 0, "Final cancels queued preview before any obsolete inference executes");
            release.Set();
            await blocker.WaitAsync(deadline);
            await complete.WaitAsync(deadline);
            check(executed == 1 && submitted == 2, "Queued preview cancellation leaves exactly one final inference");
            check(events.Select(message => message.Event).SequenceEqual(new[] { "final", "done" }), "Queued preview cancellation publishes final then done without stale partial");
        }
        finally
        {
            release.Set();
        }
    }

    /// <summary>确认运行中取消仍持有音频，最终句必须等安全返回后开始。</summary>
    /// <param name="model">始终判定语音的确定性微型 VAD。</param>
    /// <param name="check">记录断言及其说明的回调。</param>
    /// <param name="deadline">检查超时令牌。</param>
    /// <returns>检查完成后结束的任务。</returns>
    private static async Task CheckRunningAsync(Gguf model, Action<bool, string> check, CancellationToken deadline)
    {
        await using var pool = new BoundedInferencePool<object>([new object()], 4);
        using var release = new ManualResetEventSlim();
        var previewStarted = new TaskCompletionSource<CancellationToken>(TaskCreationOptions.RunContinuationsAsynchronously);
        var events = new ConcurrentQueue<SpeechEvent>();
        int submitted = 0;
        bool audioPreserved = false;
        await using var stream = new PhoneStream((pcm, token) =>
        {
            int call = Interlocked.Increment(ref submitted);
            return pool.RunAsync(_ =>
            {
                if (call == 1)
                {
                    float expected = pcm.Span[0];
                    previewStarted.SetResult(token);
                    if (!release.Wait(10000))
                    {
                        throw new TimeoutException();
                    }
                    audioPreserved = pcm.Span[0] == expected && pcm.Span[^1] == expected;
                }
                return Result(call == 1 ? "obsolete" : "final", pcm.Length);
            }, token);
        }, model, message => { events.Enqueue(message); return Task.CompletedTask; }, true);
        try
        {
            FeedSeconds(stream, 2);
            CancellationToken previewToken = await previewStarted.Task.WaitAsync(deadline);
            Task complete = stream.CompleteAsync();
            check(previewToken.IsCancellationRequested && !complete.IsCompleted && submitted == 1, "Running preview cancellation retains borrowed audio until synchronous inference exits");
            release.Set();
            await complete.WaitAsync(deadline);
            check(audioPreserved && submitted == 2, "Running canceled preview releases safely before final inference");
            check(events.Select(message => message.Event).SequenceEqual(new[] { "final", "done" }), "Running canceled preview never publishes stale text");
        }
        finally
        {
            release.Set();
        }
    }

    /// <summary>确认新预览安全替换被全局队列阻塞的旧预览。</summary>
    /// <param name="model">始终判定语音的确定性微型 VAD。</param>
    /// <param name="check">记录断言及其说明的回调。</param>
    /// <param name="deadline">检查超时令牌。</param>
    /// <returns>检查完成后结束的任务。</returns>
    private static async Task CheckReplacementAsync(Gguf model, Action<bool, string> check, CancellationToken deadline)
    {
        var previewQueued = new TaskCompletionSource<CancellationToken>(TaskCreationOptions.RunContinuationsAsynchronously);
        var replacementQueued = new TaskCompletionSource<CancellationToken>(TaskCreationOptions.RunContinuationsAsynchronously);
        int submitted = 0;
        await using var stream = new PhoneStream(async (pcm, token) =>
        {
            int call = Interlocked.Increment(ref submitted);
            if (call == 1)
            {
                previewQueued.SetResult(token);
            }
            else if (call == 2)
            {
                replacementQueued.SetResult(token);
            }
            else
            {
                return Result("final", pcm.Length);
            }
            await Task.Delay(Timeout.Infinite, token);
            throw new InvalidOperationException("Canceled preview unexpectedly continued");
        }, model, _ => Task.CompletedTask, true);
        FeedSeconds(stream, 2);
        CancellationToken original = await previewQueued.Task.WaitAsync(deadline);
        FeedSeconds(stream, 2);
        CancellationToken replacement = await replacementQueued.Task.WaitAsync(deadline);
        check(original.IsCancellationRequested && !replacement.IsCancellationRequested, "New preview cancels obsolete queued preview while retaining latest preview");
        await stream.CompleteAsync().WaitAsync(deadline);
        check(replacement.IsCancellationRequested && submitted == 3, "Complete cancels latest preview and still recognizes final segment");
    }

    /// <summary>确认会话取消不会被当作预览局部取消吞掉或发送成功完成事件。</summary>
    /// <param name="model">始终判定语音的确定性微型 VAD。</param>
    /// <param name="check">记录断言及其说明的回调。</param>
    /// <param name="deadline">检查超时令牌。</param>
    /// <returns>检查完成后结束的任务。</returns>
    private static async Task CheckStopAsync(Gguf model, Action<bool, string> check, CancellationToken deadline)
    {
        using var stop = new CancellationTokenSource();
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var events = new ConcurrentQueue<SpeechEvent>();
        await using var stream = new PhoneStream(async (pcm, token) =>
        {
            started.SetResult();
            await Task.Delay(Timeout.Infinite, token);
            return Result("unexpected", pcm.Length);
        }, model, message => { events.Enqueue(message); return Task.CompletedTask; }, true, cancellationToken: stop.Token);
        FeedSeconds(stream, 2);
        await started.Task.WaitAsync(deadline);
        stop.Cancel();
        bool rejected = false;
        try
        {
            await stream.CompleteAsync().WaitAsync(deadline);
        }
        catch (InvalidOperationException exception) when (exception.InnerException is OperationCanceledException)
        {
            rejected = true;
        }
        check(rejected && events.IsEmpty, "Session cancellation is propagated without false final or done events");
    }

    /// <summary>构造始终为语音的微型 VAD，测试只覆盖调度而不依赖实际声学模型。</summary>
    /// <returns>具有最小线性层和确定性输出概率的独立模型。</returns>
    private static Gguf CreateSpeechVad()
    {
        string path = Path.GetTempFileName();
        Gguf model;
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747U);
                writer.Write(3U);
                writer.Write(0UL);
                writer.Write(0UL);
            }
            model = new Gguf(path);
        }
        finally
        {
            File.Delete(path);
        }
        model.Metadata.Add("vad.fsmn_layers", 0);
        model.Metadata.Add("vad.proj_dim", 1);
        model.Metadata.Add("vad.lorder", 1);
        model.Tensors.Add("cmvn.shift", new("cmvn.shift", [400]) { Values = new float[400] });
        model.Tensors.Add("cmvn.scale", new("cmvn.scale", [400]) { Values = new float[400] });
        AddLinear(model, "encoder.in_linear1.linear", 400, 1);
        AddLinear(model, "encoder.in_linear2.linear", 1, 1);
        AddLinear(model, "encoder.out_linear1.linear", 1, 1);
        AddLinear(model, "encoder.out_linear2.linear", 1, 2);
        model.Tensors["encoder.out_linear2.linear.bias"].Values = [-10f, 10f];
        return model;
    }

    /// <summary>添加输出为零的最小线性层。</summary>
    /// <param name="model">接收测试权重的模型。</param>
    /// <param name="name">线性层名称前缀。</param>
    /// <param name="width">输入宽度。</param>
    /// <param name="rows">输出宽度。</param>
    private static void AddLinear(Gguf model, string name, int width, int rows)
    {
        model.Tensors.Add(name + ".weight", new(name + ".weight", [width, rows]) { Values = new float[width * rows] });
        model.Tensors.Add(name + ".bias", new(name + ".bias", [rows]) { Values = new float[rows] });
    }

    /// <summary>输入指定秒数的有效音频，每次不超过服务允许的一秒。</summary>
    /// <param name="stream">接收音频的独立会话。</param>
    /// <param name="seconds">输入秒数。</param>
    private static void FeedSeconds(PhoneStream stream, int seconds)
    {
        float[] audio = Enumerable.Repeat(0.25f, 16000).ToArray();
        for (int second = 0; second < seconds; second++)
        {
            stream.Feed(audio);
        }
    }

    /// <summary>构造确定性的推理替身结果。</summary>
    /// <param name="text">应发送的测试文字。</param>
    /// <param name="samples">输入采样数。</param>
    /// <returns>不涉及声学或解码精度的识别结果。</returns>
    private static AsrResult Result(string text, int samples) => new(text, samples / 16000.0, 0, 0, 1);
}
