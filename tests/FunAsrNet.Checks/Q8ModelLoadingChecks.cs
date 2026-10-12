using System.Buffers.Binary;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>穷举模型缩放位模式，验证解包不改变字节、NaN 载荷、负零及目标边界。</summary>
internal static class Q8ModelLoadingChecks
{
    /// <summary>独立按原格式生成块，并对短输入和错误目标进行预检。</summary>
    /// <param name="check">断言记录回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        const int blocks = 65536;
        var raw = new byte[blocks * 34 + 3];
        for (int block = 0; block < blocks; block++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(1 + block * 34, 2), (ushort)block);
            for (int i = 0; i < 32; i++)
            {
                raw[1 + block * 34 + 2 + i] = (byte)((block * 17 + i * 31) & 255);
            }
        }
        var q = new sbyte[blocks * 32 + 2];
        Array.Fill(q, (sbyte)-99);
        var scales = Enumerable.Repeat(-77f, blocks + 2).ToArray();
        TensorMath.UnpackQ8Model(raw.AsSpan(1, blocks * 34), q.AsSpan(1, blocks * 32), scales.AsSpan(1, blocks));
        for (int block = 0; block < blocks; block++)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf((ushort)block);
            check(BitConverter.SingleToInt32Bits(scales[block + 1]) == BitConverter.SingleToInt32Bits(expected), "Q8 load exact exhaustive F16 scale bits");
            for (int i = 0; i < 32; i++)
            {
                check(q[1 + block * 32 + i] == unchecked((sbyte)raw[1 + block * 34 + 2 + i]), "Q8 load preserves every signed model byte");
            }
        }
        check(q[0] == -99 && q[^1] == -99 && scales[0] == -77 && scales[^1] == -77, "Q8 load respects sliced target guards");
        bool rejected = false;
        try
        {
            TensorMath.UnpackQ8Model(raw.AsSpan(1, 33), q.AsSpan(0, 32), scales.AsSpan(0, 1));
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        check(rejected && q[0] == -99, "Q8 load rejects incomplete block before writing");
        TensorMath.UnpackQ8Model([], [], []);
        check(true, "Q8 load empty spans");
        CheckGgufChunks(check, raw.AsSpan(1, 9001 * 34).ToArray());
    }

    /// <summary>用跨越缓冲区边界的真实 GGUF 验证 F16、Q8 和截断文件处理。</summary>
    /// <param name="check">断言回调。</param>
    /// <param name="q8">独立生成的原始块。</param>
    private static void CheckGgufChunks(Action<bool, string> check, byte[] q8)
    {
        string path = Path.Combine(Path.GetTempPath(), "funasr-load-" + Guid.NewGuid().ToString("N") + ".gguf");
        const int halfCount = 65553;
        int q8Offset = (halfCount * 2 + 31) / 32 * 32;
        try
        {
            using (var writer = new BinaryWriter(File.Create(path)))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(2ul);
                writer.Write(0ul);
                WriteDescriptor(writer, "half", halfCount, 1, 0);
                WriteDescriptor(writer, "q8", q8.Length / 34 * 32, 8, (ulong)q8Offset);
                long start = (writer.BaseStream.Position + 31) / 32 * 32;
                writer.BaseStream.Position = start;
                for (int i = 0; i < halfCount; i++)
                {
                    writer.Write((ushort)i);
                }
                writer.BaseStream.Position = start + q8Offset;
                writer.Write(q8);
            }
            var model = new Gguf(path);
            bool halfEqual = true, qEqual = true, scaleEqual = true;
            for (int i = 0; i < halfCount; i++)
            {
                halfEqual &= BitConverter.SingleToInt32Bits(model["half"].Values![i]) == BitConverter.SingleToInt32Bits((float)BitConverter.UInt16BitsToHalf((ushort)i));
            }
            for (int b = 0; b < q8.Length / 34; b++)
            {
                scaleEqual &= BitConverter.SingleToInt32Bits(model["q8"].Scales![b]) == BitConverter.SingleToInt32Bits((float)BitConverter.UInt16BitsToHalf((ushort)b));
                for (int i = 0; i < 32; i++)
                {
                    qEqual &= model["q8"].Quantized![b * 32 + i] == unchecked((sbyte)q8[b * 34 + 2 + i]);
                }
            }
            check(halfEqual, "GGUF F16 exact across buffered read boundaries");
            check(qEqual && scaleEqual, "GGUF Q8 exact across buffered read boundaries");
            using (var stream = File.OpenWrite(path))
            {
                stream.SetLength(stream.Length - 3);
            }
            bool rejected = false;
            try { _ = new Gguf(path); }
            catch (EndOfStreamException) { rejected = true; }
            check(rejected, "GGUF truncated final Q8 block is rejected");
        }
        finally { File.Delete(path); }
    }

    /// <summary>独立按 GGUF v3 规范写入一维描述符。</summary>
    /// <param name="writer">目标文件写入器。</param>
    /// <param name="name">ASCII 张量名称。</param>
    /// <param name="count">元素总数。</param>
    /// <param name="type">GGML 类型。</param>
    /// <param name="offset">相对于数据区的偏移。</param>
    private static void WriteDescriptor(BinaryWriter writer, string name, int count, uint type, ulong offset)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(name);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
        writer.Write(1u);
        writer.Write((ulong)count);
        writer.Write(type);
        writer.Write(offset);
    }
}
