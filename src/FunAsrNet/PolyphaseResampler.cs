namespace FunAsrNet;

/// <summary>
/// 采用 64 抽头 Blackman 窗 sinc 滤波器的有界、有状态重采样器。
/// </summary>
public sealed class PolyphaseResampler
{
    private const int Taps = 64, Half = Taps / 2, Capacity = 16384;
    private readonly float[] _ring = new float[2 * Capacity];
    private readonly float[][] _phases;
    private readonly int _inputRate, _outputRate, _gcd;
    private readonly bool _simd;
    private long _received, _generated;
    private bool _finished;
    /// <summary>
    /// 重采样滤波器需要的前视时长，单位为毫秒。
    /// </summary>
    public double LookaheadMilliseconds => 1000.0 * Half / _inputRate;

    /// <summary>
    /// 采用 64 抽头 Blackman 窗 sinc 滤波器的有界、有状态重采样器。
    /// </summary>
    /// <param name="inputRate">输入采样率，单位为 Hz。</param>
    /// <param name="outputRate">输出采样率，单位为 Hz。</param>
    /// <param name="simd">是否启用当前硬件支持的向量化路径。</param>
    public PolyphaseResampler(int inputRate, int outputRate, bool simd = true)
    {
        if (inputRate is < 8000 or > 192000 || outputRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(nameof(inputRate));
        }

        _inputRate = inputRate;
        _outputRate = outputRate;
        _simd = simd;
        _gcd = Gcd(inputRate, outputRate);
        int phases = outputRate / _gcd;
        if (phases > 4096)
        {
            throw new ArgumentException("Unsupported resampling ratio");
        }

        _phases = new float[phases][];
        double cutoff = 0.47 * Math.Min(1.0, (double)outputRate / inputRate);
        for (int p = 0; p < phases; ++p)
        {
            double fraction = (double)p / phases, total = 0;
            var coefficients = new float[Taps];
            for (int k = 0; k < Taps; ++k)
            {
                double offset = k - Half + 1 - fraction;
                double z = 2 * cutoff * offset;
                double sinc = Math.Abs(z) < 1e-12 ? 1 : Math.Sin(Math.PI * z) / (Math.PI * z);
                double window = Math.Abs(offset) >= Half ? 0 : 0.42 + 0.5 * Math.Cos(Math.PI * offset / Half) + 0.08 * Math.Cos(2 * Math.PI * offset / Half);
                coefficients[k] = (float)(2 * cutoff * sinc * window);
                total += coefficients[k];
            }

            for (int k = 0; k < Taps; ++k)
            {
                coefficients[k] /= (float)total;
            }

            _phases[p] = coefficients;
        }
    }

    /// <summary>
    /// 估算输入对应的输出容量上界，用于提前租用缓冲区。
    /// </summary>
    /// <param name="inputSamples">本批输入采样数量。</param>
    /// <returns>足以容纳本次解码或重采样输出的采样数量上界。</returns>
    public int GetMaxOutput(int inputSamples) => checked((int)(((long)inputSamples + Taps + 2) * _outputRate / _inputRate + 2));
    /// <summary>
    /// 计算本次输入后能够生成的精确采样数，包含末尾刷新规则。
    /// </summary>
    /// <param name="inputSamples">本批输入采样数量。</param>
    /// <param name="final">是否为最后一批输入；为 true 时刷新内部尾部状态。</param>
    /// <returns>本次处理需要的目标缓冲区长度。</returns>
    public int GetOutputCount(int inputSamples, bool final = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(inputSamples);
        long limit = Math.Max(0, _received + inputSamples - (final ? 0 : Half));
        return checked((int)Math.Max(0, (limit * _outputRate + _inputRate - 1) / _inputRate - _generated));
    }

    /// <summary>
    /// 推进有状态重采样器，将本次可输出采样写入调用方缓冲区。
    /// </summary>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <param name="output">由调用方提供的结果存储区。</param>
    /// <param name="final">是否为最后一批输入；为 true 时刷新内部尾部状态。</param>
    /// <returns>本次实际写入输出缓冲区的采样数量。</returns>
    public int Process(ReadOnlySpan<float> input, Span<float> output, bool final = false)
    {
        if (_finished)
        {
            throw new InvalidOperationException("Resampler already flushed");
        }

        if (input.Length > Capacity - 2 * Taps)
        {
            throw new ArgumentException("Split input into audio frames");
        }

        if (output.Length < GetOutputCount(input.Length, final))
        {
            throw new ArgumentException("Destination too small");
        }

        for (int i = 0; i < input.Length; ++i)
        {
            int at = (int)(_received++ & (Capacity - 1));
            _ring[at] = _ring[at + Capacity] = input[i];
        }

        Span<float> edge = stackalloc float[Taps];
        int count = 0;
        while (true)
        {
            long numerator = _generated * _inputRate;
            long center = numerator / _outputRate;
            if (final ? center >= _received : center + Half >= _received)
            {
                break;
            }

            int phase = (int)((numerator % _outputRate) / _gcd);
            long first = center - Half + 1;
            if (first >= 0 && first + Taps <= _received)
            {
                output[count++] = AudioKernels.Dot(_ring.AsSpan((int)(first & (Capacity - 1)), Taps), _phases[phase], _simd);
            }
            else
            {
                for (int k = 0; k < Taps; ++k)
                {
                    long index = first + k;
                    edge[k] = index < 0 || index >= _received ? 0 : _ring[(int)(index & (Capacity - 1))];
                }

                output[count++] = AudioKernels.Dot(edge, _phases[phase], _simd);
            }

            ++_generated;
        }

        if (final)
        {
            _finished = true;
        }

        return count;
    }

    /// <summary>
    /// 计算两个正整数的最大公约数以确定多相滤波相位。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <returns>计算得到的整数结果。</returns>
    private static int Gcd(int a, int b)
    {
        while (b != 0)
        {
            (a, b) = (b, a % b);
        }

        return a;
    }
}
