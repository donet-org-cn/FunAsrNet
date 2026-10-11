using System.Text;

namespace FunAsrNet;

/// <summary>
/// WAV 文件读取、采样率转换与 PCM16 文件写入工具。
/// </summary>
public static partial class AudioFile
{
    /// <summary>
    /// 读取有界 WAV 输入，校验格式并将多声道音频解码为单声道浮点采样。
    /// </summary>
    /// <param name="path">输入或输出文件的路径。</param>
    /// <returns>解码后的单声道浮点数组与原始采样率。</returns>
    public static (float[] Samples, int Rate) ReadWave(string path)
    {
        using var stream = File.OpenRead(path);
        return ReadWave(stream);
    }

    // 同步读取要求有界、可寻址输入；网络上传使用 ReadWaveAsync，流由调用方管理。
    /// <summary>
    /// 读取有界 WAV 输入，校验格式并将多声道音频解码为单声道浮点采样。
    /// </summary>
    /// <param name="stream">由调用方拥有的输入流；本方法不负责关闭它。</param>
    /// <param name="maxSeconds">允许的音频时长上限，单位为秒。</param>
    /// <returns>解码后的单声道浮点数组与原始采样率。</returns>
    public static (float[] Samples, int Rate) ReadWave(Stream stream, double maxSeconds = double.PositiveInfinity)
    {
        if (!stream.CanSeek || stream.Length - stream.Position < 12)
        {
            throw new InvalidDataException("Truncated WAV header");
        }

        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (r.ReadUInt32() != 0x46464952)
        {
            throw new InvalidDataException("Expected RIFF WAV");
        }

        long riffEnd = r.BaseStream.Position + 4L + r.ReadUInt32();
        if (riffEnd > stream.Length || riffEnd < stream.Position + 4)
        {
            throw new InvalidDataException("Invalid RIFF length");
        }

        if (r.ReadUInt32() != 0x45564157)
        {
            throw new InvalidDataException("Expected WAVE");
        }

        int format = 0, channels = 0, rate = 0, bits = 0, alignment = 0;
        long dataOffset = -1;
        int dataLength = 0;
        while (stream.Position + 8 <= riffEnd)
        {
            uint id = r.ReadUInt32();
            int length = checked((int)r.ReadUInt32());
            long end = checked(r.BaseStream.Position + length);
            if (end > riffEnd)
            {
                throw new InvalidDataException("Truncated WAV");
            }

            if (id == 0x20746d66)
            {
                if (length < 16 || format != 0)
                {
                    throw new InvalidDataException("Invalid WAV format chunk");
                }

                format = r.ReadUInt16();
                channels = r.ReadUInt16();
                rate = r.ReadInt32();
                r.ReadInt32();
                alignment = r.ReadUInt16();
                bits = r.ReadUInt16();
            }
            else if (id == 0x61746164)
            {
                if (dataOffset >= 0)
                {
                    throw new InvalidDataException("Multiple WAV data chunks are not supported");
                }

                dataOffset = stream.Position;
                dataLength = length;
            }

            r.BaseStream.Position = end + (length & 1);
        }

        if (dataOffset < 0
            || channels is < 1 or > 8
            || rate is < 8000 or > 96000
            || !(format == 1 && bits == 16
            || format == 3 && bits == 32
            || format is 6 or 7 && bits == 8))
        {
            throw new NotSupportedException("WAV must contain PCM16, float32, PCMA or PCMU");
        }

        int bytes = bits / 8;
        if (alignment != bytes * channels || dataLength == 0 || dataLength % alignment != 0)
        {
            throw new InvalidDataException("Invalid WAV sample alignment or empty audio");
        }

        int frames = dataLength / alignment;
        if (frames / (double)rate > maxSeconds)
        {
            throw new InvalidDataException($"WAV exceeds {maxSeconds} seconds");
        }

        stream.Position = dataOffset;
        byte[] data = r.ReadBytes(dataLength);
        var samples = new float[frames];
        for (int i = 0; i < samples.Length; ++i)
        {
            float value = 0;
            for (int c = 0; c < channels; ++c)
            {
                int offset = (i * channels + c) * bytes;
                value += format == 3 ? BitConverter.ToSingle(data, offset) : format == 1 ? BitConverter.ToInt16(data, offset) / 32768f : G711.DecodeSample(data[offset], format == 6) / 32768f;
            }

            if (!float.IsFinite(value))
            {
                throw new InvalidDataException("WAV contains non-finite samples");
            }

            samples[i] = Math.Clamp(value / channels, -1, 1);
        }

        return (samples, rate);
    }

    /// <summary>
    /// 转换采样率；目标相同时返回原数组，否则一次分配最终输出数组。
    /// </summary>
    /// <param name="pcm">单声道浮点音频采样。</param>
    /// <param name="from">输入采样率，单位为 Hz。</param>
    /// <param name="to">目标采样率，单位为 Hz。</param>
    /// <returns>目标采样率的浮点数组；采样率相同时返回原数组。</returns>
    public static float[] Resample(float[] pcm, int from, int to)
    {
        if (from == to)
        {
            return pcm;
        }

        var resampler = new PolyphaseResampler(from, to);
        var result = new float[resampler.GetOutputCount(pcm.Length, true)];
        int written = 0;
        for (int i = 0; i < pcm.Length; i += 4096)
        {
            int n = Math.Min(4096, pcm.Length - i);
            written += resampler.Process(pcm.AsSpan(i, n), result.AsSpan(written), i + n == pcm.Length);
        }

        if (written != result.Length)
        {
            throw new InvalidOperationException("Resampling output length mismatch");
        }

        return result;
    }

    /// <summary>
    /// 将浮点采样限幅后写为单声道 PCM16 WAV 文件。
    /// </summary>
    /// <param name="path">输入或输出文件的路径。</param>
    /// <param name="pcm">单声道浮点音频采样。</param>
    /// <param name="rate">采样率，单位为 Hz。</param>
    public static void WriteWave(string path, ReadOnlySpan<float> pcm, int rate)
    {
        using var w = new BinaryWriter(File.Create(path), Encoding.UTF8);
        w.Write(0x46464952);
        w.Write(36 + pcm.Length * 2);
        w.Write(0x45564157);
        w.Write(0x20746d66);
        w.Write(16);
        w.Write((short)1);
        w.Write((short)1);
        w.Write(rate);
        w.Write(rate * 2);
        w.Write((short)2);
        w.Write((short)16);
        w.Write(0x61746164);
        w.Write(pcm.Length * 2);
        foreach (float x in pcm)
        {
            w.Write((short)(Math.Clamp(x, -1f, 1f) * 32767));
        }
    }
}
