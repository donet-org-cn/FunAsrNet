using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace FunAsrNet.Server;

/// <summary>
/// 直接向 HTTP 输出管道编码音频，避免整份 PCM/WAV 中间数组。
/// </summary>
internal static class AudioResponse
{
    /// <summary>
    /// 逐块将浮点音频编码到 HTTP 输出管道，避免完整中间 PCM 字节数组。
    /// </summary>
    /// <param name="response">当前 HTTP 响应对象。</param>
    /// <param name="audio">单声道浮点音频采样。</param>
    /// <param name="sampleRate">输入或输出采样率，单位为 Hz。</param>
    /// <param name="format">WAV 格式标识或目标音频编码名称。</param>
    /// <param name="volume">输出音量倍率，范围 0 到 2；放大后会限幅。</param>
    /// <param name="waveHeader">是否在首块音频前写入 WAV 文件头。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>表示操作完成、取消或失败的异步任务。</returns>
    internal static async Task WriteAsync(
        HttpResponse response,
        ReadOnlyMemory<float> audio,
        int sampleRate,
        string format,
        float volume,
        bool waveHeader,
        CancellationToken cancellationToken)
    {
        if (waveHeader)
        {
            WriteWaveHeader(response.BodyWriter.GetSpan(44), audio.Length, sampleRate);
            response.BodyWriter.Advance(44);
        }

        for (int position = 0; position < audio.Length; position += 2048)
        {
            int count = Math.Min(2048, audio.Length - position);
            int byteCount = count * (format is "pcma" or "pcmu" ? 1 : 2);
            Encode(audio.Span.Slice(position, count), response.BodyWriter.GetSpan(byteCount)[..byteCount], format, volume);
            response.BodyWriter.Advance(byteCount);
            var flush = await response.BodyWriter.FlushAsync(cancellationToken);
            if (flush.IsCanceled)
            {
                throw new OperationCanceledException(cancellationToken);
            }

            if (flush.IsCompleted)
            {
                throw new IOException("客户端已停止读取合成音频。");
            }
        }
    }

    /// <summary>
    /// 根据请求格式对当前采样块实施音量调整、限幅和字节编码。
    /// </summary>
    /// <param name="audio">单声道浮点音频采样。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    /// <param name="format">WAV 格式标识或目标音频编码名称。</param>
    /// <param name="volume">输出音量倍率，范围 0 到 2；放大后会限幅。</param>
    private static void Encode(ReadOnlySpan<float> audio, Span<byte> destination, string format, float volume)
    {
        if (format is "pcma" or "pcmu")
        {
            Span<short> pcm = stackalloc short[audio.Length];
            for (int i = 0; i < audio.Length; i++)
            {
                pcm[i] = (short)(Math.Clamp(audio[i] * volume, -1, 1) * 32767);
            }

            G711.Encode(pcm, destination, format == "pcma");
        }
        else if (volume == 1 && BitConverter.IsLittleEndian)
        {
            AudioKernels.FloatToPcm16(audio, MemoryMarshal.Cast<byte, short>(destination));
        }
        else
        {
            for (int i = 0; i < audio.Length; i++)
            {
                BinaryPrimitives.WriteInt16LittleEndian(destination[(i * 2)..], (short)(Math.Clamp(audio[i] * volume, -1, 1) * 32767));
            }
        }
    }

    /// <summary>
    /// 直接在输出视图中编码 44 字节的单声道 PCM16 WAV 文件头。
    /// </summary>
    /// <param name="header">是否写入 WAV 文件头。</param>
    /// <param name="samples">目标单声道采样缓冲区。</param>
    /// <param name="rate">采样率，单位为 Hz。</param>
    private static void WriteWaveHeader(Span<byte> header, int samples, int rate)
    {
        int bytes = checked(samples * 2);
        BinaryPrimitives.WriteUInt32LittleEndian(header, 0x46464952);
        BinaryPrimitives.WriteInt32LittleEndian(header[4..], checked(bytes + 36));
        BinaryPrimitives.WriteUInt32LittleEndian(header[8..], 0x45564157);
        BinaryPrimitives.WriteUInt32LittleEndian(header[12..], 0x20746d66);
        BinaryPrimitives.WriteInt32LittleEndian(header[16..], 16);
        BinaryPrimitives.WriteInt16LittleEndian(header[20..], 1);
        BinaryPrimitives.WriteInt16LittleEndian(header[22..], 1);
        BinaryPrimitives.WriteInt32LittleEndian(header[24..], rate);
        BinaryPrimitives.WriteInt32LittleEndian(header[28..], rate * 2);
        BinaryPrimitives.WriteInt16LittleEndian(header[32..], 2);
        BinaryPrimitives.WriteInt16LittleEndian(header[34..], 16);
        BinaryPrimitives.WriteUInt32LittleEndian(header[36..], 0x61746164);
        BinaryPrimitives.WriteInt32LittleEndian(header[40..], bytes);
    }
}
