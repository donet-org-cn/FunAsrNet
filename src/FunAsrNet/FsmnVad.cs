namespace FunAsrNet;

/// <summary>
/// 具有独立历史状态的因果 FSMN 检测器；特征拼接使用 20 ms 前视。
/// </summary>
public sealed class FsmnVad
{
    private readonly Gguf _model;
    private readonly TensorMath _math = new(1);
    private readonly FilterBank _bank = new();
    private readonly int _layers, _projection, _order;
    private readonly float[][] _memory;
    private readonly float[] _samples = new float[400];
    private readonly Queue<float[]> _features = new();
    private int _sampleCount, _memoryPosition;
    private bool _started, _finished;
    /// <summary>
    /// 具有独立历史状态的因果 FSMN 检测器；特征拼接使用 20 ms 前视。
    /// </summary>
    /// <param name="model">已加载的模型及其权重；发布给并发上下文后不得修改权重。</param>
    public FsmnVad(Gguf model)
    {
        _model = model;
        _layers = model.Int("vad.fsmn_layers", 4);
        _projection = model.Int("vad.proj_dim", 128);
        _order = model.Int("vad.lorder", 20);
        if (model.Int("vad.lfr_m", 5) != 5 || model.Int("vad.lfr_n", 1) != 1)
        {
            throw new NotSupportedException("VAD expects LFR 5/1");
        }

        _memory = Enumerable.Range(0, _layers).Select(_ => new float[_order * _projection]).ToArray();
    }

    /// <summary>
    /// 接收单声道采样并保留跨帧历史，返回新增帧的静音概率。
    /// </summary>
    /// <param name="samples">目标单声道采样缓冲区。</param>
    /// <param name="final">是否为最后一批输入；为 true 时刷新内部尾部状态。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public List<float> Feed(ReadOnlySpan<float> samples, bool final = false)
    {
        if (_finished)
        {
            throw new InvalidOperationException("VAD already flushed");
        }

        List<float> scores = [];
        foreach (float sample in samples)
        {
            _samples[_sampleCount++] = sample;
            if (_sampleCount != 400)
            {
                continue;
            }

            var feature = _bank.Compute80(_samples, out _);
            if (!_started)
            {
                _features.Enqueue(feature);
                _features.Enqueue(feature);
                _started = true;
            }

            _features.Enqueue(feature);
            Drain(scores);
            Array.Copy(_samples, 160, _samples, 0, 240);
            _sampleCount = 240;
        }

        if (final && _started && _features.Count > 0)
        {
            var last = _features.Last();
            _features.Enqueue(last);
            Drain(scores);
            _features.Enqueue(last);
            Drain(scores);
        }

        _finished = final;
        return scores;
    }

    /// <summary>
    /// 将已积累的特征推进 FSMN 网络，追加可输出的检测分数。
    /// </summary>
    /// <param name="scores">注意力分数或 VAD 静音概率的存储区。</param>
    private void Drain(List<float> scores)
    {
        while (_features.Count >= 5)
        {
            float[] features = new float[400];
            int offset = 0;
            foreach (var f in _features.Take(5))
            {
                f.CopyTo(features, offset);
                offset += 80;
            }

            _features.Dequeue();
            scores.Add(Score(features));
        }
    }

    /// <summary>
    /// 根据拼接后的特征计算当前帧的静音概率。
    /// </summary>
    /// <param name="features">按时间排列的输入特征。</param>
    /// <returns>计算得到的标量数值。</returns>
    public float Score(float[] features)
    {
        var shift = _model["cmvn.shift"].Values!;
        var scale = _model["cmvn.scale"].Values!;
        for (int i = 0; i < features.Length; ++i)
        {
            features[i] = (features[i] + shift[i]) * scale[i];
        }

        var h = Linear(features, "encoder.in_linear1.linear");
        h = Linear(h, "encoder.in_linear2.linear");
        Relu(h);
        for (int layer = 0; layer < _layers; ++layer)
        {
            string p = $"encoder.fsmn.{layer}.";
            var z = _math.Linear(h, 1, _model[p + "linear.linear.weight"]);
            var memory = _memory[layer];
            z.CopyTo(memory, _memoryPosition * _projection);
            var kernel = _model[p + "fsmn_block.conv_left.weight"].Values!;
            var acc = (float[])z.Clone();
            for (int k = 0; k < _order; ++k)
            {
                int at = (_memoryPosition + k + 1) % _order * _projection;
                for (int d = 0; d < _projection; ++d)
                {
                    acc[d] += memory[at + d] * kernel[k * _projection + d];
                }
            }

            h = Linear(acc, p + "affine.linear");
            Relu(h);
        }

        _memoryPosition = (_memoryPosition + 1) % _order;
        h = Linear(h, "encoder.out_linear1.linear");
        h = Linear(h, "encoder.out_linear2.linear");
        TensorMath.Softmax(h);
        return h[0];
    }

    /// <summary>
    /// 执行 VAD 模型指定层的线性变换。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="p">模型层的名称前缀。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    private float[] Linear(float[] x, string p) => _math.Linear(x, 1, _model[p + ".weight"], _model[p + ".bias"]);
    /// <summary>
    /// 原位将负数截断为零，实施 ReLU 激活。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    private static void Relu(float[] x)
    {
        for (int i = 0; i < x.Length; ++i)
        {
            x[i] = Math.Max(0, x[i]);
        }
    }
}

/// <summary>
/// 使用 16 kHz 采样位置表示的语音句段，结束位置不包含在区间内。
/// </summary>
/// <param name="StartSample">句段的起始采样位置。</param>
/// <param name="EndSample">句段的结束采样位置，不包含该采样。</param>
public readonly record struct SpeechSegment(long StartSample, long EndSample);
/// <summary>
/// 有界电话语音端点检测器，维护起止投票与静音计时。
/// </summary>
/// <param name="silenceMilliseconds">判定句段结束所需的静音时长，单位为毫秒。</param>
/// <param name="maxSegmentMilliseconds">单个语音句段的最大时长，单位为毫秒。</param>
public sealed class PhoneEndpoint(int silenceMilliseconds = 600, int maxSegmentMilliseconds = 12000)
{
    private readonly int[] _votes = new int[20];
    private int _voteAt, _sum, _silence, _speech;
    private long _frame, _previousEnd, _start = -1;
    /// <summary>
    /// 当前未闭合语音句段的起始采样位置；无活动句段时为 -1。
    /// </summary>
    public long ActiveStartSample => _start < 0 ? -1 : _start * 160;

    /// <summary>
    /// 推进一帧端点判定，并在满足结束条件时返回完整句段。
    /// </summary>
    /// <param name="silenceProbability">当前帧属于静音的概率。</param>
    /// <returns>达到端点时返回闭合句段，否则为空。</returns>
    public SpeechSegment? Step(float silenceProbability)
    {
        int vote = silenceProbability <= 0.25f ? 1 : 0;
        _sum -= _votes[_voteAt];
        _votes[_voteAt] = vote;
        _sum += vote;
        _voteAt = (_voteAt + 1) % 20;
        int change;
        if (_speech == 0 && _sum >= 15)
        {
            _speech = 1;
            change = 3;
        }
        else if (_speech == 1 && _sum <= 15)
        {
            _speech = 0;
            change = 1;
        }
        else
        {
            change = _speech == 0 ? 0 : 2;
        }

        if (change == 3 && _start < 0)
        {
            _start = Math.Max(_previousEnd, Math.Max(0, _frame - 40));
        }

        if (change != 0)
        {
            _silence = 0;
        }
        else
        {
            ++_silence;
        }

        SpeechSegment? result = null;
        if (_start >= 0)
        {
            int threshold = Math.Max(1, silenceMilliseconds / 10 - 15);
            if (_silence >= threshold)
            {
                result = Emit(Math.Max(_start + 1, _frame - Math.Max(0, threshold - 11)));
            }
            else if ((_frame - _start + 1) * 10 >= maxSegmentMilliseconds)
            {
                result = Emit(_frame);
            }
        }

        ++_frame;
        return result;
    }

    /// <summary>
    /// 在输入结束时刷新尚未闭合的语音句段。
    /// </summary>
    /// <param name="totalSamples">已经接收的 16 kHz 采样总数。</param>
    /// <returns>待闭合的末尾句段；没有语音时为空。</returns>
    public SpeechSegment? Flush(long totalSamples) => _start >= 0 ? Emit((totalSamples + 159) / 160) : null;
    /// <summary>
    /// 结束当前句段并更新端点检测状态。
    /// </summary>
    /// <param name="end">句段结束采样位置，不包含该位置。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private SpeechSegment Emit(long end)
    {
        var result = new SpeechSegment(_start * 160, end * 160);
        _previousEnd = end;
        _start = -1;
        _sum = 0;
        _silence = 0;
        _speech = 0;
        _voteAt = 0;
        Array.Clear(_votes);
        return result;
    }
}
