using System.Text;
using System.Runtime.InteropServices;

namespace FunAsrNet;

/// <summary>
/// 读取 GGUF v2/v3 元数据和张量；维度采用最内层优先的 GGML 顺序。
/// </summary>
public sealed class Gguf
{
    /// <summary>
    /// GGUF 元数据集合，模型发布后按只读方式使用。
    /// </summary>
    public Dictionary<string, object> Metadata { get; } = new(StringComparer.Ordinal);
    /// <summary>
    /// 按名称索引的模型权重，布局准备必须在发布给工作线程前完成。
    /// </summary>
    public Dictionary<string, Weight> Tensors { get; } = new(StringComparer.Ordinal);

    /// <summary>
    /// 读取整数元数据或节点属性，缺失时使用默认值。
    /// </summary>
    /// <param name="key">GGUF 元数据键或命令行选项名。</param>
    /// <param name="fallback">键或属性不存在时采用的默认值。</param>
    /// <returns>计算得到的整数结果。</returns>
    public int Int(string key, int fallback) => Metadata.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;
    /// <summary>
    /// 读取浮点元数据或节点属性，缺失时使用默认值。
    /// </summary>
    /// <param name="key">GGUF 元数据键或命令行选项名。</param>
    /// <param name="fallback">键或属性不存在时采用的默认值。</param>
    /// <returns>计算得到的标量数值。</returns>
    public float Float(string key, float fallback) => Metadata.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;
    /// <summary>按模型权重名称取得已加载的张量。</summary>
    /// <param name="key">GGUF 中的权重名称。</param>
    /// <returns>由模型拥有的权重；发布给并发上下文后不得修改。</returns>
    public Weight this[string key] => Tensors[key];
    /// <summary>
    /// 读取 GGUF v2/v3 元数据和张量；维度采用最内层优先的 GGML 顺序。
    /// </summary>
    /// <param name="path">输入或输出文件的路径。</param>
    public Gguf(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8);
        if (reader.ReadUInt32() != 0x46554747)
        {
            throw new InvalidDataException("Not a GGUF model");
        }

        uint version = reader.ReadUInt32();
        if (version is not (2 or 3))
        {
            throw new NotSupportedException($"GGUF version {version}");
        }

        int tensorCount = checked((int)reader.ReadUInt64()), metadataCount = checked((int)reader.ReadUInt64());
        for (int i = 0; i < metadataCount; ++i)
        {
            Metadata.Add(ReadString(reader), ReadValue(reader, reader.ReadUInt32()));
        }

        var descriptors = new List<(string Name, int[] Shape, uint Type, long Offset)>();
        for (int i = 0; i < tensorCount; ++i)
        {
            string name = ReadString(reader);
            int dimensions = checked((int)reader.ReadUInt32());
            if (dimensions is < 1 or > 4)
            {
                throw new InvalidDataException("Invalid tensor rank");
            }

            int[] shape = new int[dimensions];
            for (int j = 0; j < shape.Length; ++j)
            {
                shape[j] = checked((int)reader.ReadUInt64());
            }

            descriptors.Add((name, shape, reader.ReadUInt32(), checked((long)reader.ReadUInt64())));
        }

        int alignment = Int("general.alignment", 32);
        if (alignment <= 0)
        {
            throw new InvalidDataException("Invalid alignment");
        }

        long dataStart = (stream.Position + alignment - 1) / alignment * alignment;
        foreach (var d in descriptors)
        {
            int count = 1;
            foreach (int n in d.Shape)
            {
                count = checked(count * n);
            }

            stream.Position = checked(dataStart + d.Offset);
            var weight = new Weight(d.Name, d.Shape);
            if (d.Type == 0)
            {
                weight.Values = new float[count];
                stream.ReadExactly(MemoryMarshal.AsBytes(weight.Values.AsSpan()));
            }
            else if (d.Type == 1)
            {
                byte[] raw = new byte[checked(count * 2)];
                stream.ReadExactly(raw);
                var half = MemoryMarshal.Cast<byte, Half>(raw);
                weight.Values = GC.AllocateUninitializedArray<float>(count);
                for (int i = 0; i < count; ++i)
                {
                    weight.Values[i] = (float)half[i];
                }
            }
            else if (d.Type == 8)
            {
                if (d.Shape[0] % 32 != 0)
                {
                    throw new InvalidDataException("Q8_0 row not aligned");
                }

                int blocks = count / 32;
                byte[] raw = new byte[checked(blocks * 34)];
                stream.ReadExactly(raw);
                weight.Quantized = GC.AllocateUninitializedArray<sbyte>(count);
                weight.Scales = new float[blocks];
                for (int b = 0; b < blocks; ++b)
                {
                    weight.Scales[b] = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(raw, b * 34));
                    MemoryMarshal.Cast<byte, sbyte>(raw.AsSpan(b * 34 + 2, 32)).CopyTo(weight.Quantized.AsSpan(b * 32));
                }
            }
            else
            {
                throw new NotSupportedException($"Tensor {d.Name}: GGML type {d.Type}; use the preserved F16 encoder / Q8_0 decoder");
            }

            Tensors.Add(d.Name, weight);
        }
    }

    /// <summary>
    /// 读取 GGUF 长度前缀 UTF-8 字符串。
    /// </summary>
    /// <param name="reader">用于读取 GGUF 数据的二进制读取器。</param>
    /// <returns>完成处理后的文本。</returns>
    private static string ReadString(BinaryReader reader)
    {
        int n = checked((int)reader.ReadUInt64());
        if (n > 64 * 1024 * 1024)
        {
            throw new InvalidDataException("Oversized GGUF string");
        }

        byte[] bytes = reader.ReadBytes(n);
        if (bytes.Length != n)
        {
            throw new EndOfStreamException();
        }

        return Encoding.UTF8.GetString(bytes);
    }

    /// <summary>
    /// 根据 GGUF 类型标签读取一个元数据值。
    /// </summary>
    /// <param name="r">用于读取 GGUF 数据的二进制读取器。</param>
    /// <param name="type">GGUF 元数据的类型标签。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static object ReadValue(BinaryReader r, uint type) => type switch
    {
        0 => r.ReadByte(),
        1 => r.ReadSByte(),
        2 => r.ReadUInt16(),
        3 => r.ReadInt16(),
        4 => r.ReadUInt32(),
        5 => r.ReadInt32(),
        6 => r.ReadSingle(),
        7 => r.ReadBoolean(),
        8 => ReadString(r),
        9 => ReadArray(r),
        10 => r.ReadUInt64(),
        11 => r.ReadInt64(),
        12 => r.ReadDouble(),
        _ => throw new InvalidDataException($"Unknown GGUF metadata type {type}")
    };
    /// <summary>
    /// 根据元素类型和长度读取 GGUF 元数据数组。
    /// </summary>
    /// <param name="r">用于读取 GGUF 数据的二进制读取器。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static object[] ReadArray(BinaryReader r)
    {
        uint type = r.ReadUInt32();
        int n = checked((int)r.ReadUInt64());
        var result = new object[n];
        for (int i = 0; i < n; ++i)
        {
            result[i] = ReadValue(r, type);
        }

        return result;
    }
}

/// <summary>
/// 模型权重及其计算布局；加载完成后只允许在发布前准备布局。
/// </summary>
/// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
/// <param name="shape">按行主序排列的目标维度数组。</param>
public sealed partial class Weight(string name, int[] shape)
{
    /// <summary>
    /// 模型中的唯一权重名称。
    /// </summary>
    public string Name { get; } = name;
    /// <summary>
    /// 张量维度数组；视图共享时不得通过修改维度破坏其他使用者。
    /// </summary>
    public int[] Shape { get; } = shape;
    /// <summary>
    /// 浮点权重存储，F16 源权重按原值展开为 float。
    /// </summary>
    public float[]? Values
    {
        get; internal set;
    }
    /// <summary>
    /// Q8 权重整数存储；对应的缩放因子位于 Scales。
    /// </summary>
    public sbyte[]? Quantized
    {
        get; internal set;
    }
    /// <summary>
    /// Q8 权重每 32 个元素的缩放因子。
    /// </summary>
    public float[]? Scales
    {
        get; internal set;
    }
    /// <summary>
    /// 每个权重行的元素数量。
    /// </summary>
    public int Columns => Shape[0];
    /// <summary>
    /// 权重矩阵的输出通道数量。
    /// </summary>
    public int Rows => Shape.Length > 1 ? Shape[1] : 1;

    /// <summary>
    /// 读取指定权重行并还原为浮点数组。
    /// </summary>
    /// <param name="index">从零开始的行、元素或任务索引。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    public float[] Row(int index)
    {
        var row = new float[Columns];
        CopyRow(index, row);
        return row;
    }

    /// <summary>
    /// 将指定权重行解码到调用方缓冲区，避免返回数组分配。
    /// </summary>
    /// <param name="index">从零开始的行、元素或任务索引。</param>
    /// <param name="destination">由调用方提供且容量足够的输出视图。</param>
    public void CopyRow(int index, Span<float> destination)
    {
        if (FloatInterleaved || QuantizedInterleaved)
        {
            CopyInterleavedRow(index, destination);
            return;
        }

        if (Values is not null)
        {
            Values.AsSpan(index * Columns, Columns).CopyTo(destination);
            return;
        }

        int offset = index * Columns;
        for (int j = 0; j < Columns; ++j)
        {
            destination[j] = Quantized![offset + j] * Scales![(offset + j) / 32];
        }
    }
}
