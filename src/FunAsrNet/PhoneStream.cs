using System.Diagnostics;
using System.Threading.Channels;

namespace FunAsrNet;

/// <summary>
/// 实时转写的状态或文本事件，计时单位为秒。
/// </summary>
/// <param name="Event">事件类型，例如 ready、partial、final 或 done。</param>
/// <param name="Text">识别得到的文本。</param>
/// <param name="ElapsedSeconds">自会话开始经过的秒数。</param>
/// <param name="AudioStartSeconds">当前句段在音频中的起始秒数。</param>
/// <param name="AudioEndSeconds">当前句段在音频中的结束秒数。</param>
/// <param name="InferenceSeconds">本句段的推理耗时，单位为秒。</param>
public sealed record SpeechEvent(
    string Event,
    string Text,
    double ElapsedSeconds,
    double? AudioStartSeconds = null,
    double? AudioEndSeconds = null,
    double? InferenceSeconds = null);
/// <summary>
/// 单生产者实时音频会话；句段异步识别，过期预览被更新的预览替换。
/// </summary>
public sealed class PhoneStream : IAsyncDisposable
{
    /// <summary>
    /// 携带任务结果与取消状态的队列项。
    /// </summary>
    /// <param name="Pcm">当前任务独占持有的音频池缓冲租约。</param>
    /// <param name="Start">句段的起始采样位置。</param>
    /// <param name="End">句段的结束采样位置，不包含该位置。</param>
    /// <param name="Final">是否为不可再替换的最终句段。</param>
    /// <param name="Epoch">预览所属的句段代数，用于丢弃过期预览结果。</param>
    /// <param name="PreviewCancellation">预览独有的取消源；最终句段为空并使用会话取消令牌。</param>
    private sealed record Job(PooledAudioBuffer Pcm, long Start, long End, bool Final, int Epoch, CancellationTokenSource? PreviewCancellation) : IDisposable
    {
        private readonly object _lifetime = new();
        private bool _disposed;
        /// <summary>安全取消过期预览，不归还仍由推理任务使用的音频。</summary>
        internal void CancelPreview()
        {
            lock (_lifetime)
            {
                if (!_disposed)
                {
                    PreviewCancellation?.Cancel();
                }
            }
        }
        /// <summary>在任务已结束或尚未提交时归还音频，并与并发取消安全协调。</summary>
        public void Dispose()
        {
            lock (_lifetime)
            {
                if (_disposed)
                {
                    return;
                }
                _disposed = true;
                PreviewCancellation?.Dispose();
                Pcm.Dispose();
            }
        }
    }
    private readonly Func<ReadOnlyMemory<float>, CancellationToken, Task<AsrResult>> _transcribe;
    private readonly CancellationTokenSource _stop;
    private readonly FsmnVad _vad;
    private readonly PhoneEndpoint _endpoint;
    private readonly Func<SpeechEvent, Task> _emit;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly float[] _ring = new float[16000 * 32];
    private readonly Channel<bool> _signal = Channel.CreateBounded<bool>(1);
    private readonly Queue<Job> _finals = new();
    private readonly object _sync = new();
    private readonly Task _worker;
    private readonly bool _partials;
    private Job? _preview;
    private Job? _activePreview;
    private volatile bool _finished;
    private int _epoch;
    private long _received, _lastPreview;
    private volatile Exception? _failure;
    /// <summary>
    /// 单生产者实时音频会话；句段异步识别，过期预览被更新的预览替换。
    /// </summary>
    /// <param name="asr">独占使用的识别上下文。</param>
    /// <param name="vadModel">只读的 FSMN VAD 模型；检测历史由各会话独立维护。</param>
    /// <param name="emit">异步发送会话事件的回调；接收方应保持事件顺序。</param>
    /// <param name="partials">是否生成可能被后续结果修订的预览文本。</param>
    /// <param name="silenceMilliseconds">判定句段结束所需的静音时长，单位为毫秒。</param>
    public PhoneStream(
        FunAsr asr,
        Gguf vadModel,
        Func<SpeechEvent, Task> emit,
        bool partials = false,
        int silenceMilliseconds = 1200) : this((pcm, _) => Task.FromResult(asr.Transcribe(pcm.Span)), vadModel, emit, partials, silenceMilliseconds)
    {
    }

    /// <summary>
    /// 单生产者实时音频会话；句段异步识别，过期预览被更新的预览替换。
    /// </summary>
    /// <param name="transcribe">借用音频视图并异步返回识别结果的调度回调。</param>
    /// <param name="vadModel">只读的 FSMN VAD 模型；检测历史由各会话独立维护。</param>
    /// <param name="emit">异步发送会话事件的回调；接收方应保持事件顺序。</param>
    /// <param name="partials">是否生成可能被后续结果修订的预览文本。</param>
    /// <param name="silenceMilliseconds">判定句段结束所需的静音时长，单位为毫秒。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    public PhoneStream(
        Func<ReadOnlyMemory<float>, CancellationToken, Task<AsrResult>> transcribe,
        Gguf vadModel,
        Func<SpeechEvent, Task> emit,
        bool partials = false,
        int silenceMilliseconds = 1200,
        CancellationToken cancellationToken = default)
    {
        _transcribe = transcribe;
        _stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _vad = new FsmnVad(vadModel);
        _endpoint = new PhoneEndpoint(silenceMilliseconds);
        _emit = emit;
        _partials = partials;
        _worker = Task.Run(Work);
    }

    /// <summary>
    /// 将当前帧同步写入会话缓冲并推进 VAD；异步任务使用另行持有的句段快照。
    /// </summary>
    /// <param name="pcm">单声道浮点音频采样。</param>
    public void Feed(ReadOnlySpan<float> pcm)
    {
        if (_finished)
        {
            throw new InvalidOperationException("Stream has ended");
        }

        if (_failure is not null)
        {
            throw new InvalidOperationException("Recognition worker failed", _failure);
        }

        if (pcm.Length > 16000)
        {
            throw new ArgumentException("Feed at most one second per call");
        }

        foreach (float x in pcm)
        {
            _ring[_received++ % _ring.Length] = x;
        }

        foreach (float score in _vad.Feed(pcm))
        {
            var segment = _endpoint.Step(score);
            if (segment.HasValue)
            {
                EnqueueFinal(segment.Value);
            }
        }

        if (_partials && _endpoint.ActiveStartSample >= 0 && _received - _lastPreview >= 24000 && _received - _endpoint.ActiveStartSample >= 12000)
        {
            var job = Snapshot(_endpoint.ActiveStartSample, _received, false);
            Job? obsolete, active;
            lock (_sync)
            {
                obsolete = _preview;
                _preview = job;
                active = _activePreview;
            }
            obsolete?.CancelPreview();
            obsolete?.Dispose();
            active?.CancelPreview();
            _lastPreview = _received;
            _signal.Writer.TryWrite(true);
        }
    }

    /// <summary>
    /// 刷新输入尾部并等待最终句段完成，然后发送完成事件。
    /// </summary>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    public async Task CompleteAsync()
    {
        if (!_finished)
        {
            foreach (float score in _vad.Feed([], true))
            {
                var segment = _endpoint.Step(score);
                if (segment.HasValue)
                {
                    EnqueueFinal(segment.Value);
                }
            }

            var trailing = _endpoint.Flush(_received);
            if (trailing.HasValue)
            {
                EnqueueFinal(trailing.Value);
            }

            Job? obsolete, active;
            lock (_sync)
            {
                _finished = true;
                obsolete = _preview;
                _preview = null;
                active = _activePreview;
            }
            obsolete?.CancelPreview();
            obsolete?.Dispose();
            active?.CancelPreview();
            _signal.Writer.TryWrite(true);
        }

        await _worker;
        if (_failure is not null)
        {
            throw new InvalidOperationException("Recognition worker failed", _failure);
        }
    }

    /// <summary>
    /// 从环形缓冲复制有效句段到池缓冲，保证排队推理期间的数据所有权。
    /// </summary>
    /// <param name="start">起始采样位置或输入偏移。</param>
    /// <param name="end">句段结束采样位置，不包含该位置。</param>
    /// <param name="final">是否为最后一批输入；为 true 时刷新内部尾部状态。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private Job Snapshot(long start, long end, bool final)
    {
        end = Math.Min(end, _received);
        start = Math.Max(0, start);
        if (start < _received - _ring.Length)
        {
            throw new InvalidOperationException("Audio ring exhausted; input exceeded recognition capacity");
        }

        var pcm = new PooledAudioBuffer(checked((int)(end - start)));
        int offset = (int)(start % _ring.Length), first = Math.Min(pcm.Memory.Length, _ring.Length - offset);
        _ring.AsSpan(offset, first).CopyTo(pcm.Memory.Span);
        _ring.AsSpan(0, pcm.Memory.Length - first).CopyTo(pcm.Memory.Span[first..]);
        return new(pcm, start, end, final, _epoch, final ? null : CancellationTokenSource.CreateLinkedTokenSource(_stop.Token));
    }

    /// <summary>
    /// 提交最终句段并淘汰过期预览；队列满时停止积累音频。
    /// </summary>
    /// <param name="segment">已经闭合的语音句段。</param>
    private void EnqueueFinal(SpeechSegment segment)
    {
        var job = Snapshot(segment.StartSample, segment.EndSample, true);
        if (job.Pcm.Memory.Length < 400)
        {
            job.Dispose();
            return;
        }
        Job? obsolete, active;
        lock (_sync)
        {
            if (_finals.Count >= 4)
            {
                job.Dispose();
                throw new InvalidOperationException("Recognition is overloaded (four queued segments)");
            }

            ++_epoch;
            obsolete = _preview;
            _preview = null;
            active = _activePreview;
            _finals.Enqueue(job);
        }
        obsolete?.CancelPreview();
        obsolete?.Dispose();
        active?.CancelPreview();
        _signal.Writer.TryWrite(true);
    }

    /// <summary>
    /// 依次处理该会话的句段与预览任务，并在任务结束后归还音频租约。
    /// </summary>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    private async Task Work()
    {
        try
        {
            while (true)
            {
                await _signal.Reader.ReadAsync();
                while (true)
                {
                    Job? job;
                    bool finished;
                    lock (_sync)
                    {
                        job = _finals.Count > 0 ? _finals.Dequeue() : _preview;
                        if (job == _preview)
                        {
                            _preview = null;
                        }
                        if (job is not null && !job.Final)
                        {
                            _activePreview = job;
                        }
                        finished = _finished;
                    }

                    if (job is null)
                    {
                        if (finished)
                        {
                            await _emit(new("done", "", _clock.Elapsed.TotalSeconds));
                            return;
                        }

                        break;
                    }

                    try
                    {
                        var result = await _transcribe(job.Pcm.Memory, job.PreviewCancellation?.Token ?? _stop.Token);
                        if (job.Final || (job.PreviewCancellation?.IsCancellationRequested != true && job.Epoch == Volatile.Read(ref _epoch)))
                        {
                            await _emit(new(job.Final ? "final" : "partial", result.Text, _clock.Elapsed.TotalSeconds, job.Start / 16000.0, job.End / 16000.0, result.TotalSeconds));
                        }
                    }
                    catch (OperationCanceledException exception) when (!job.Final && job.PreviewCancellation?.IsCancellationRequested == true
                        && exception.CancellationToken == job.PreviewCancellation.Token && !_stop.IsCancellationRequested)
                    {
                        // 只丢弃预览的局部取消；会话取消仍由外层报告并终止工作。
                    }
                    finally
                    {
                        if (!job.Final)
                        {
                            lock (_sync)
                            {
                                if (ReferenceEquals(_activePreview, job))
                                {
                                    _activePreview = null;
                                }
                            }
                        }
                        job.Dispose();
                    }
                }
            }
        }
        catch (Exception e)
        {
            _failure = e;
        }
    }

    /// <summary>
    /// 停止接收工作并异步释放所持资源，等待正在使用的缓冲区安全归还。
    /// </summary>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    public async ValueTask DisposeAsync()
    {
        // 连接断开会取消排队任务；已经开始的 CPU 推理须完成后才能安全归还缓冲区。
        _stop.Cancel();
        Job? obsolete, active;
        Job[] finals;
        lock (_sync)
        {
            _finished = true;
            obsolete = _preview;
            _preview = null;
            active = _activePreview;
            finals = _finals.ToArray();
            _finals.Clear();
        }
        obsolete?.CancelPreview();
        obsolete?.Dispose();
        active?.CancelPreview();
        foreach (var job in finals)
        {
            job.Dispose();
        }
        _signal.Writer.TryWrite(true);
        await _worker;
        _stop.Dispose();
    }
}

/// <summary>
/// 维护跨帧 PCM 字节与重采样历史的电话音频解码器，每条会话使用独立实例。
/// </summary>
public sealed class PhoneAudioDecoder
{
    private readonly AudioCodec _codec;
    private readonly PolyphaseResampler? _resampler;
    private readonly short[] _pcm = new short[8000];
    private readonly float[] _floats = new float[8000];
    private int _carry = -1;
    private bool _finished;
    /// <summary>
    /// 维护跨帧 PCM 字节与重采样历史的电话音频解码器，每条会话使用独立实例。
    /// </summary>
    /// <param name="codec">PCM16、A-law 或 μ-law 输入编码。</param>
    /// <param name="sampleRate">输入或输出采样率，单位为 Hz。</param>
    public PhoneAudioDecoder(AudioCodec codec, int sampleRate)
    {
        if (!Enum.IsDefined(codec)
            || sampleRate is not (8000 or 16000 or 22050 or 24000 or 32000 or 44100 or 48000 or 96000)
            || codec != AudioCodec.Pcm16 && sampleRate != 8000)
        {
            throw new ArgumentException("G.711 requires 8000 Hz; PCM supports 8000/16000/22050/24000/32000/44100/48000/96000 Hz mono");
        }

        _codec = codec;
        if (sampleRate != 16000)
        {
            _resampler = new(sampleRate, 16000);
        }
    }

    /// <summary>
    /// 解码当前电话音频帧并转换到 16 kHz，保留跨帧字节和滤波器历史。
    /// </summary>
    /// <param name="bytes">当前调用拥有或借用的字节数据。</param>
    /// <param name="final">是否为最后一批输入；为 true 时刷新内部尾部状态。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Decode(ReadOnlySpan<byte> bytes, bool final = false)
    {
        var output = new float[GetMaxOutput(bytes.Length)];
        int count = Decode(bytes, output, final);
        return count == output.Length ? output : output.AsSpan(0, count).ToArray();
    }

    /// <summary>
    /// 估算输入对应的输出容量上界，用于提前租用缓冲区。
    /// </summary>
    /// <param name="byteCount">本帧输入字节数。</param>
    /// <returns>足以容纳本次解码或重采样输出的采样数量上界。</returns>
    public int GetMaxOutput(int byteCount)
    {
        if (byteCount is < 0 or > 8000)
        {
            throw new ArgumentOutOfRangeException(nameof(byteCount));
        }

        int count = _codec == AudioCodec.Pcm16 ? (byteCount + 1) / 2 : byteCount;
        return _resampler?.GetMaxOutput(count) ?? count;
    }

    /// <summary>
    /// 解码当前电话音频帧并转换到 16 kHz，保留跨帧字节和滤波器历史。
    /// </summary>
    /// <param name="bytes">当前调用拥有或借用的字节数据。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    /// <param name="final">是否为最后一批输入；为 true 时刷新内部尾部状态。</param>
    /// <returns>计算得到的整数结果。</returns>
    public int Decode(ReadOnlySpan<byte> bytes, Span<float> destination, bool final = false)
    {
        if (_finished)
        {
            throw new InvalidOperationException("Audio already flushed");
        }

        if (bytes.Length > 8000)
        {
            throw new ArgumentException("Use short audio frames");
        }

        if (destination.Length < GetMaxOutput(bytes.Length))
        {
            throw new ArgumentException("Destination too small");
        }

        ReadOnlySpan<short> pcm;
        if (_codec != AudioCodec.Pcm16)
        {
            Span<short> values = _pcm.AsSpan(0, bytes.Length);
            G711.Decode(bytes, values, _codec == AudioCodec.Pcma);
            pcm = values;
        }
        else if (_carry < 0 && bytes.Length % 2 == 0 && BitConverter.IsLittleEndian)
        {
            // 常见的完整 PCM16 帧直接借用网络缓冲区，不再复制到 short[]。
            pcm = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, short>(bytes);
        }
        else
        {
            Span<short> values = _pcm.AsSpan(0, (bytes.Length + (_carry >= 0 ? 1 : 0)) / 2);
            int read = 0, written = 0;
            if (_carry >= 0 && bytes.Length > 0)
            {
                values[written++] = unchecked((short)(_carry | bytes[read++] << 8));
                _carry = -1;
            }

            for (; read + 1 < bytes.Length; read += 2)
            {
                values[written++] = unchecked((short)(bytes[read] | bytes[read + 1] << 8));
            }

            if (read < bytes.Length)
            {
                _carry = bytes[read];
            }

            if (final && _carry >= 0)
            {
                throw new InvalidDataException("Truncated PCM16 sample");
            }

            pcm = values;
        }

        _finished = final;
        if (_resampler is null)
        {
            AudioKernels.Pcm16ToFloat(pcm, destination);
            return pcm.Length;
        }

        Span<float> floats = _floats.AsSpan(0, pcm.Length);
        AudioKernels.Pcm16ToFloat(pcm, floats);
        return _resampler.Process(floats, destination, final);
    }
}
