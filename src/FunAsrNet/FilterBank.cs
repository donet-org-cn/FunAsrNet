namespace FunAsrNet;

/// <summary>
/// 兼容 Fun-ASR 的 16 kHz、80 维对数滤波器组与 7/6 LFR 特征提取。
/// </summary>
public sealed class FilterBank
{
    private const int Window = 400, Shift = 160, FftSize = 512, Bins = 80;
    private readonly float[] _window = new float[Window];
    private readonly float[][] _filters = new float[Bins][];
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
            for (int k = 0; k <= FftSize / 2; ++k)
            {
                float f = Mel(16000f / FftSize * k);
                if (f > left && f < right)
                {
                    _filters[m][k] = f <= center ? (f - left) / (center - left) : (right - f) / (right - center);
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
        var re = new float[FftSize];
        var im = new float[FftSize];
        var frame = new float[Window];
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
            Array.Clear(re);
            Array.Clear(im);
            for (int i = 0; i < Window; ++i)
            {
                re[i] = frame[i] * _window[i];
            }

            Fft(re, im);
            for (int m = 0; m < Bins; ++m)
            {
                float energy = 0;
                for (int k = 0; k <= FftSize / 2; ++k)
                {
                    if (_filters[m][k] > 0)
                    {
                        energy += _filters[m][k] * (re[k] * re[k] + im[k] * im[k]);
                    }
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

    /// <summary>
    /// 原位计算窗内数据的复数快速傅里叶变换。
    /// </summary>
    /// <param name="re">FFT 实部数组，结果原位写回。</param>
    /// <param name="im">FFT 虚部数组，结果原位写回。</param>
    private static void Fft(float[] re, float[] im)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; ++i)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }

            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        for (int length = 2; length <= n; length <<= 1)
        {
            float angle = (float)(-2 * Math.PI / length), wr = MathF.Cos(angle), wi = MathF.Sin(angle);
            for (int i = 0; i < n; i += length)
            {
                float cr = 1, ci = 0;
                for (int k = 0; k < length / 2; ++k)
                {
                    int a = i + k, b = a + length / 2;
                    float ur = re[a], ui = im[a], vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
                    re[a] = ur + vr;
                    im[a] = ui + vi;
                    re[b] = ur - vr;
                    im[b] = ui - vi;
                    float next = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = next;
                }
            }
        }
    }
}
