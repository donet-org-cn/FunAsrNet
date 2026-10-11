using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace FunAsrNet;

/// <summary>
/// WAV 文件读取、采样率转换与 PCM16 文件写入工具。
/// </summary>
public static partial class AudioFile
{
    /// <summary>
    /// 从不可寻址的上传流逐块读取 WAV；直接解码到最终浮点数组，不缓存整份原始文件。
    /// </summary>
    /// <param name="stream">由调用方拥有的输入流；本方法不负责关闭它。</param>
    /// <param name="maxSeconds">允许的音频时长上限，单位为秒。</param>
    /// <param name="maxBytes">允许读取的文件总字节数上限。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>调用方拥有的单声道浮点采样数组及原始采样率。</returns>
    public static async Task<(float[] Samples, int Rate)> ReadWaveAsync(
        Stream stream,
        double maxSeconds = 60,
        int maxBytes = 32 * 1024 * 1024,
        CancellationToken cancellationToken = default)
    {
        byte[] buffer = ArrayPool<byte>.Shared.Rent(65536);
        try
        {
            await stream.ReadExactlyAsync(buffer.AsMemory(0, 12), cancellationToken);
            if (BinaryPrimitives.ReadUInt32LittleEndian(buffer) != 0x46464952 || BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(8)) != 0x45564157)
            {
                throw new InvalidDataException("文件必须是 RIFF WAV。");
            }

            long remaining = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(4)) - 4L;
            if (remaining < 0 || remaining + 12 > maxBytes)
            {
                throw new InvalidDataException("WAV 文件长度超过限制或无效。");
            }

            int format = 0, channels = 0, rate = 0, bits = 0, alignment = 0;
            float[]? samples = null;
            while (remaining >= 8)
            {
                await stream.ReadExactlyAsync(buffer.AsMemory(0, 8), cancellationToken);
                remaining -= 8;
                uint id = BinaryPrimitives.ReadUInt32LittleEndian(buffer);
                long length = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(4));
                long padded = length + (length & 1);
                if (padded > remaining)
                {
                    throw new InvalidDataException("WAV 数据块被截断。");
                }

                if (id == 0x20746d66)
                {
                    if (length < 16 || format != 0)
                    {
                        throw new InvalidDataException("WAV 格式块长度无效或重复。");
                    }

                    await stream.ReadExactlyAsync(buffer.AsMemory(0, 16), cancellationToken);
                    format = BinaryPrimitives.ReadUInt16LittleEndian(buffer);
                    channels = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(2));
                    rate = BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(4));
                    alignment = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(12));
                    bits = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(14));
                    await SkipWaveBytesAsync(stream, padded - 16, buffer, cancellationToken);
                }
                else if (id == 0x61746164)
                {
                    if (samples is not null || format == 0)
                    {
                        throw new InvalidDataException("WAV 需要先提供 fmt 块，且只能有一个 data 块。");
                    }

                    if (channels is < 1 or > 8
                        || rate is < 8000 or > 96000
                        || !(format == 1 && bits == 16
                        || format == 3 && bits == 32
                        || format is 6 or 7 && bits == 8))
                    {
                        throw new NotSupportedException("支持 8–96 kHz、1–8 声道的 PCM16、Float32、PCMA 或 PCMU WAV。");
                    }

                    if (alignment != channels * (bits / 8) || length == 0 || length % alignment != 0 || length / (double)alignment / rate > maxSeconds)
                    {
                        throw new InvalidDataException("音频为空、采样未对齐或超过允许时长。");
                    }

                    samples = GC.AllocateUninitializedArray<float>(checked((int)(length / alignment)));
                    int framesPerBlock = buffer.Length / alignment, position = 0;
                    while (position < samples.Length)
                    {
                        int frames = Math.Min(framesPerBlock, samples.Length - position);
                        await stream.ReadExactlyAsync(buffer.AsMemory(0, frames * alignment), cancellationToken);
                        DecodeWaveSamples(buffer.AsSpan(0, frames * alignment), samples.AsSpan(position, frames), format, bits, channels);
                        position += frames;
                    }

                    await SkipWaveBytesAsync(stream, length & 1, buffer, cancellationToken);
                }
                else
                {
                    await SkipWaveBytesAsync(stream, padded, buffer, cancellationToken);
                }

                remaining -= padded;
            }

            if (remaining != 0 || samples is null)
            {
                throw new InvalidDataException("WAV 数据块不完整或缺失。");
            }

            return (samples, rate);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// 在固定大小缓冲区内跳过未知 WAV 块，避免按块长度分配内存。
    /// </summary>
    /// <param name="stream">由调用方拥有的输入流；本方法不负责关闭它。</param>
    /// <param name="count">需要处理的元素或任务数量。</param>
    /// <param name="buffer">可复用的读写缓冲区。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    private static async Task SkipWaveBytesAsync(Stream stream, long count, byte[] buffer, CancellationToken cancellationToken)
    {
        while (count > 0)
        {
            int block = (int)Math.Min(count, buffer.Length);
            await stream.ReadExactlyAsync(buffer.AsMemory(0, block), cancellationToken);
            count -= block;
        }
    }

    /// <summary>
    /// 将当前对齐的 WAV 数据块解码、混音到指定单声道采样区。
    /// </summary>
    /// <param name="data">连续存储的数据。</param>
    /// <param name="samples">目标单声道采样缓冲区。</param>
    /// <param name="format">WAV 格式标识或目标音频编码名称。</param>
    /// <param name="bits">每个声道采样的位数。</param>
    /// <param name="channels">交错排列的输入声道数量。</param>
    private static void DecodeWaveSamples(ReadOnlySpan<byte> data, Span<float> samples, int format, int bits, int channels)
    {
        if (format == 1 && channels == 1 && BitConverter.IsLittleEndian)
        {
            AudioKernels.Pcm16ToFloat(MemoryMarshal.Cast<byte, short>(data), samples);
            return;
        }

        int stride = bits / 8;
        for (int i = 0; i < samples.Length; i++)
        {
            float sum = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                ReadOnlySpan<byte> value = data[((i * channels + channel) * stride)..];
                sum += format == 3 ? BinaryPrimitives.ReadSingleLittleEndian(value) : format == 1 ? BinaryPrimitives.ReadInt16LittleEndian(value) / 32768f : G711.DecodeSample(value[0], format == 6) / 32768f;
            }

            if (!float.IsFinite(sum))
            {
                throw new InvalidDataException("WAV 含非有限浮点采样。");
            }

            samples[i] = Math.Clamp(sum / channels, -1, 1);
        }
    }
}
