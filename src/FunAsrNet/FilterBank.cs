namespace FunAsrNet;

/// <summary>
/// 兼容 Fun-ASR 的 16 kHz、80 维对数滤波器组与 7/6 LFR 特征提取。
/// </summary>
public sealed partial class FilterBank
{
    private const int Window = 400, Shift = 160, FftSize = 512, Bins = 80;
    private readonly float[] _window = new float[Window];
    private readonly float[][] _filters = new float[Bins][];
    private readonly int[] _filterStarts = new int[Bins], _filterEnds = new int[Bins];
    /// <summary>
    /// 兼容 Fun-ASR 的 16 kHz、80 维对数滤波器组与 7/6 LFR 特征提取。
    /// </summary>
    public FilterBank()
    {
        for (int i = 0; i < Window; ++i)
        {
            _window[i] = 0.54f - 0.46f * MathF.Cos((float)(2.0f * Math.PI * i / (Window - 1)));
        }

        float low = Mel(20), high = Mel(8000), delta = (high - low) / (Bins + 1);
        for (int m = 0; m < Bins; ++m)
        {
            float left = low + m * delta, center = low + (m + 1) * delta, right = low + (m + 2) * delta;
            _filters[m] = new float[FftSize / 2 + 1];
            _filterStarts[m] = FftSize / 2 + 1;
            for (int k = 0; k <= FftSize / 2; ++k)
            {
                float f = Mel(16000f / FftSize * k);
                if (f > left && f < right)
                {
                    _filters[m][k] = f <= center ? (f - left) / (center - left) : (right - f) / (right - center);
                    _filterStarts[m] = Math.Min(_filterStarts[m], k);
                    _filterEnds[m] = k + 1;
                }
            }
        }
    }

    /// <summary>
    /// 将频率从 Hz 映射到 Mel 尺度。
    /// </summary>
    /// <param name="f">以 Hz 表示的频率。</param>
    /// <returns>计算得到的标量数值。</returns>
    private static float Mel(float f) => 1127f * MathF.Log(1f + f / 700f);
    /// <summary>
    /// 提取 80 维对数滤波器组特征，并报告输出帧数。
    /// </summary>
    /// <param name="pcm">单声道浮点音频采样。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Compute80(ReadOnlySpan<float> pcm, out int frames)
    {
        if (pcm.Length < Window)
        {
            throw new ArgumentException("At least 25 ms of 16 kHz audio required");
        }

        int count = (pcm.Length - Window) / Shift + 1;
        float[] features = new float[count * Bins];
        // 固定大小的约 6.6 KiB 工作区只属于本次调用，不产生堆分配。
        Span<float> re = stackalloc float[FftSize];
        Span<float> im = stackalloc float[FftSize];
        Span<float> frame = stackalloc float[Window];
        Span<float> power = stackalloc float[FftSize / 2 + 1];
        for (int t = 0; t < count; ++t)
        {
            double mean = 0;
            for (int i = 0; i < Window; ++i)
            {
                mean += pcm[t * Shift + i] * 32768f;
            }

            mean /= Window;
            for (int i = 0; i < Window; ++i)
            {
                frame[i] = pcm[t * Shift + i] * 32768f - (float)mean;
            }

            for (int i = Window - 1; i > 0; --i)
            {
                frame[i] -= 0.97f * frame[i - 1];
            }

            frame[0] -= 0.97f * frame[0];
            re.Slice(Window).Clear();
            im.Clear();
            for (int i = 0; i < Window; ++i)
            {
                re[i] = frame[i] * _window[i];
            }

            Fft(re, im);
            PowerSpectrum(re, im, power);
            for (int m = 0; m < Bins; ++m)
            {
                float energy = 0;
                // 非零系数连续；仍按原频点次序累加，避免改变特征舍入。
                for (int k = _filterStarts[m]; k < _filterEnds[m]; ++k)
                {
                    energy += _filters[m][k] * power[k];
                }

                features[t * Bins + m] = MathF.Log(Math.Max(energy, 1.1920929e-7f));
            }
        }

        frames = count;
        return features;
    }

    /// <summary>
    /// 提取滤波器组特征并完成低帧率拼接，生成编码器输入。
    /// </summary>
    /// <param name="pcm">单声道浮点音频采样。</param>
    /// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Compute(ReadOnlySpan<float> pcm, out int frames)
    {
        var features = Compute80(pcm, out int count);
        frames = (count + 5) / 6;
        float[] output = new float[frames * Bins * 7];
        for (int t = 0; t < frames; ++t)
        {
            for (int j = 0; j < 7; ++j)
            {
                features.AsSpan(Math.Clamp(t * 6 + j - 3, 0, count - 1) * Bins, Bins).CopyTo(output.AsSpan((t * 7 + j) * Bins));
            }
        }

        return output;
    }
}
