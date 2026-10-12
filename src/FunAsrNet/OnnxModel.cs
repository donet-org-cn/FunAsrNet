using System.Buffers.Binary;
using System.Text;

namespace FunAsrNet;

/// <summary>
/// 连续行主序张量；整数标记用于模型中有界的形状与索引运算。
/// </summary>
/// <param name="data">连续存储的数据。</param>
/// <param name="shape">按行主序排列的目标维度数组。</param>
/// <param name="integer">是否将张量视为模型的整数形状或索引数据。</param>
public sealed class Tensor(float[] data, int[] shape, bool integer = false)
{
    /// <summary>
    /// 连续行主序元素存储；共享常量的数组不得修改。
    /// </summary>
    public float[] Data { get; } = data;
    /// <summary>
    /// 张量维度数组；视图共享时不得通过修改维度破坏其他使用者。
    /// </summary>
    public int[] Shape { get; } = shape;
    /// <summary>
    /// 该张量是否承载形状或索引等整数语义。
    /// </summary>
    public bool Integer { get; } = integer;
    /// <summary>
    /// 张量维度数量。
    /// </summary>
    public int Rank => Shape.Length;
    /// <summary>
    /// 张量连续数据区的元素数量。
    /// </summary>
    public int Count => Data.Length;

    /// <summary>
    /// 以受检整数乘法计算张量元素总数。
    /// </summary>
    /// <param name="shape">按行主序排列的目标维度数组。</param>
    /// <returns>计算得到的整数结果。</returns>
    public static int Size(IEnumerable<int> shape)
    {
        int size = 1;
        foreach (int n in shape)
        {
            size = checked(size * n);
        }

        return size;
    }

    /// <summary>
    /// 按形状创建零初始化的连续张量。
    /// </summary>
    /// <param name="shape">按行主序排列的目标维度数组。</param>
    /// <param name="integer">是否将张量视为模型的整数形状或索引数据。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Zeros(int[] shape, bool integer = false) => new(TensorWorkspace.Allocate(Size(shape), clear: true), shape, integer);

    /// <summary>为会完整写入结果的内部算子分配张量，避免先清零再覆盖；不得读取未写入元素。</summary>
    /// <param name="shape">每一元素都会被算子写入的结果形状。</param>
    /// <param name="integer">是否保留形状与索引张量的整数标记。</param>
    /// <returns>当前工作区持有的独立存储，算子必须在返回前完整填充。</returns>
    internal static Tensor ForOverwrite(int[] shape, bool integer = false) => new(TensorWorkspace.Allocate(Size(shape)), shape, integer);
    /// <summary>
    /// 将张量元素或节点整数属性转换为整数数组。
    /// </summary>
    /// <returns>按输出顺序排列的整数数组。</returns>
    public int[] Ints() => Data.Select(x => checked((int)x)).ToArray();
    /// <summary>
    /// 在不复制采样存储的情况下创建新形状视图，要求元素总数一致。
    /// </summary>
    /// <param name="shape">按行主序排列的目标维度数组。</param>
    /// <returns>复用相同数据数组且具有目标形状的张量视图。</returns>
    public Tensor View(int[] shape) => Size(shape) == Count ? new(Data, shape, Integer) : throw new ArgumentException("Reshape size mismatch");
    /// <summary>
    /// 返回张量形状的可读表示。
    /// </summary>
    /// <returns>完成处理后的文本。</returns>
    public override string ToString() => $"[{string.Join(',', Shape)}]";
}

/// <summary>
/// ONNX 图节点及其输入、输出和属性。
/// </summary>
/// <param name="Name">节点名称。</param>
/// <param name="Op">ONNX 算子名称。</param>
/// <param name="Inputs">输入张量名称数组。</param>
/// <param name="Outputs">输出张量名称数组。</param>
/// <param name="Attributes">节点属性集合，模型发布后作为只读数据使用。</param>
public sealed record OnnxNode(string Name, string Op, string[] Inputs, string[] Outputs, Dictionary<string, object> Attributes)
{
    /// <summary>
    /// 读取整数元数据或节点属性，缺失时使用默认值。
    /// </summary>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <param name="fallback">键或属性不存在时采用的默认值。</param>
    /// <returns>计算得到的整数结果。</returns>
    public int Int(string name, int fallback = 0) => Attributes.TryGetValue(name, out var v) ? Convert.ToInt32(v) : fallback;
    /// <summary>
    /// 读取浮点元数据或节点属性，缺失时使用默认值。
    /// </summary>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <param name="fallback">键或属性不存在时采用的默认值。</param>
    /// <returns>计算得到的标量数值。</returns>
    public float Float(string name, float fallback = 0) => Attributes.TryGetValue(name, out var v) ? Convert.ToSingle(v) : fallback;
    /// <summary>
    /// 将张量元素或节点整数属性转换为整数数组。
    /// </summary>
    /// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
    /// <param name="fallback">键或属性不存在时采用的默认值。</param>
    /// <returns>按输出顺序排列的整数数组。</returns>
    public int[] Ints(string name, int[]? fallback = null) => Attributes.TryGetValue(name, out var v) ? ((long[])v).Select(x => checked((int)x)).ToArray() : fallback ?? [];
}

/// <summary>
/// 无需原生运行时或 NuGet 的 ONNX 图与权重读取器，仅支持本项目模型使用的子集。
/// </summary>
public sealed class OnnxModel
{
    /// <summary>
    /// 模型常量张量，发布给多个执行上下文后不得修改。
    /// </summary>
    public Dictionary<string, Tensor> Constants { get; } = new(StringComparer.Ordinal);
    /// <summary>
    /// 按图中执行顺序排列的节点集合。
    /// </summary>
    public List<OnnxNode> Nodes { get; } = [];
    /// <summary>
    /// 模型对外输出的张量名称列表。
    /// </summary>
    public List<string> Outputs { get; } = [];
    /// <summary>
    /// 模型声明的输入名称，按图中顺序排列。
    /// </summary>
    public List<string> Inputs { get; } = [];
    /// <summary>
    /// 模型导出器写入的采样率、发音人数和文本编码约定。
    /// </summary>
    public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);

    /// <summary>
    /// 无需原生运行时或 NuGet 的 ONNX 图与权重读取器，仅支持本项目模型使用的子集。
    /// </summary>
    /// <param name="path">输入或输出文件的路径。</param>
    public OnnxModel(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var root = new Proto(bytes);
        ReadOnlyMemory<byte> graph = default;
        while (root.Next(out var f))
        {
            if (f.Number == 7)
            {
                graph = f.Bytes;
            }
            else if (f.Number == 14)
            {
                var entry = new Proto(f.Bytes);
                string key = "", value = "";
                while (entry.Next(out var field))
                {
                    if (field.Number == 1)
                    {
                        key = field.Text;
                    }
                    else if (field.Number == 2)
                    {
                        value = field.Text;
                    }
                }

                Metadata.Add(key, value);
            }
        }

        if (graph.IsEmpty)
        {
            throw new InvalidDataException("ONNX graph missing");
        }

        var p = new Proto(graph);
        while (p.Next(out var f))
        {
            if (f.Number == 1)
            {
                Nodes.Add(ReadNode(f.Bytes));
            }
            else if (f.Number == 5)
            {
                var (name, tensor) = ReadTensor(f.Bytes);
                Constants.Add(name, tensor);
            }
            else if (f.Number is 11 or 12)
            {
                var value = new Proto(f.Bytes);
                while (value.Next(out var field))
                {
                    if (field.Number == 1)
                    {
                        (f.Number == 11 ? Inputs : Outputs).Add(field.Text);
                    }
                }
            }
        }
    }

    /// <summary>
    /// 解析 ONNX 节点的算子名称、输入输出与属性。
    /// </summary>
    /// <param name="bytes">当前调用拥有或借用的字节数据。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static OnnxNode ReadNode(ReadOnlyMemory<byte> bytes)
    {
        var p = new Proto(bytes);
        string name = "", op = "";
        List<string> inputs = [], outputs = [];
        Dictionary<string, object> attributes = [];
        while (p.Next(out var f))
        {
            switch (f.Number)
            {
                case 1:
                    inputs.Add(f.Text);
                    break;
                case 2:
                    outputs.Add(f.Text);
                    break;
                case 3:
                    name = f.Text;
                    break;
                case 4:
                    op = f.Text;
                    break;
                case 5:
                    var a = ReadAttribute(f.Bytes);
                    attributes.Add(a.Name, a.Value);
                    break;
            }
        }

        return new(name, op, inputs.ToArray(), outputs.ToArray(), attributes);
    }

    /// <summary>
    /// 解析 ONNX 节点的标量、数组或张量属性。
    /// </summary>
    /// <param name="bytes">当前调用拥有或借用的字节数据。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static (string Name, object Value) ReadAttribute(ReadOnlyMemory<byte> bytes)
    {
        var p = new Proto(bytes);
        string name = "";
        object? value = null;
        List<long> ints = [];
        while (p.Next(out var f))
        {
            switch (f.Number)
            {
                case 1:
                    name = f.Text;
                    break;
                case 2:
                    value = BitConverter.Int32BitsToSingle(unchecked((int)f.Value));
                    break;
                case 3:
                    value = unchecked((long)f.Value);
                    break;
                case 4:
                    value = f.Text;
                    break;
                case 5:
                    value = ReadTensor(f.Bytes).Value;
                    break;
                case 8:
                    ReadInts(f, ints);
                    break;
            }
        }

        return (name, ints.Count != 0 ? ints.ToArray() : value ?? Array.Empty<long>());
    }

    /// <summary>
    /// 解析张量形状和原始数据，并转换为托管连续存储。
    /// </summary>
    /// <param name="bytes">当前调用拥有或借用的字节数据。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private static (string Name, Tensor Value) ReadTensor(ReadOnlyMemory<byte> bytes)
    {
        var p = new Proto(bytes);
        string name = "";
        int type = 0;
        List<long> dimensions = [], integers = [];
        List<float> floats = [];
        ReadOnlyMemory<byte> raw = default;
        while (p.Next(out var f))
        {
            switch (f.Number)
            {
                case 1:
                    ReadInts(f, dimensions);
                    break;
                case 2:
                    type = (int)f.Value;
                    break;
                case 8:
                    name = f.Text;
                    break;
                case 9:
                    raw = f.Bytes;
                    break;
                case 5:
                case 7:
                    ReadInts(f, integers);
                    break;
                case 4:
                    if (f.Wire == 5)
                    {
                        floats.Add(BitConverter.Int32BitsToSingle(unchecked((int)f.Value)));
                    }
                    else
                    {
                        for (int i = 0; i < f.Bytes.Length; i += 4)
                        {
                            floats.Add(BitConverter.ToSingle(f.Bytes.Span.Slice(i, 4)));
                        }
                    }

                    break;
            }
        }

        int[] shape = dimensions.Select(x => checked((int)x)).ToArray();
        int size = Tensor.Size(shape);
        var data = new float[size];
        if (!raw.IsEmpty)
        {
            int width = type switch
            {
                1 or 6 => 4,
                7 or 11 => 8,
                10 => 2,
                9 => 1,
                _ => throw new NotSupportedException($"ONNX tensor type {type}")
            };
            if (raw.Length != size * width)
            {
                throw new InvalidDataException("ONNX tensor byte size mismatch");
            }

            for (int i = 0; i < size; ++i)
            {
                var s = raw.Span.Slice(i * width, width);
                data[i] = type switch
                {
                    1 => BitConverter.ToSingle(s),
                    6 => BinaryPrimitives.ReadInt32LittleEndian(s),
                    7 => BinaryPrimitives.ReadInt64LittleEndian(s),
                    11 => (float)BitConverter.ToDouble(s),
                    10 => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(s)),
                    9 => s[0],
                    _ => 0
                };
            }
        }
        else if (floats.Count != 0)
        {
            floats.CopyTo(data);
        }
        else if (integers.Count != 0)
        {
            for (int i = 0; i < size; ++i)
            {
                data[i] = integers[i];
            }
        }

        return (name, new(data, shape, type is 6 or 7 or 9));
    }

    /// <summary>
    /// 解析单值或打包的 protobuf 整数序列。
    /// </summary>
    /// <param name="field">当前解析的 protobuf 字段。</param>
    /// <param name="list">用于追加整数元素的集合。</param>
    private static void ReadInts(Field field, List<long> list)
    {
        if (field.Wire == 0)
        {
            list.Add(unchecked((long)field.Value));
        }
        else
        {
            var p = new Proto(field.Bytes);
            while (!p.End)
            {
                list.Add(unchecked((long)p.Varint()));
            }
        }
    }

    /// <summary>
    /// Protocol Buffers 字段的标签、编码类型与有效载荷。
    /// </summary>
    /// <param name="Number">protobuf 字段编号。</param>
    /// <param name="Wire">protobuf 线格式编号。</param>
    /// <param name="Value">字段的整数或固定宽度数值。</param>
    /// <param name="Bytes">字段的有界字节载荷。</param>
    private readonly record struct Field(int Number, int Wire, ulong Value, ReadOnlyMemory<byte> Bytes)
    {
        /// <summary>
        /// 将字段字节内容解释为 UTF-8 文本。
        /// </summary>
        public string Text => Encoding.UTF8.GetString(Bytes.Span);
    }

    /// <summary>
    /// 在有界内存视图上逐字段解析 Protocol Buffers 数据。
    /// </summary>
    /// <param name="data">连续存储的数据。</param>
    private struct Proto(ReadOnlyMemory<byte> data)
    {
        private int _position;
        /// <summary>
        /// 输入游标是否已到达字节视图末尾。
        /// </summary>
        public bool End => _position >= data.Length;

        /// <summary>
        /// 读取 protobuf 可变长无符号整数，截断或超长编码会抛出异常。
        /// </summary>
        /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
        public ulong Varint()
        {
            ulong result = 0;
            for (int shift = 0; shift < 70; shift += 7)
            {
                if (End)
                {
                    throw new EndOfStreamException();
                }

                byte b = data.Span[_position++];
                result |= (ulong)(b & 127) << shift;
                if (b < 128)
                {
                    return result;
                }
            }

            throw new InvalidDataException("Oversized protobuf varint");
        }

        /// <summary>
        /// 读取下一个 protobuf 字段，并推进有界输入游标。
        /// </summary>
        /// <param name="field">当前解析的 protobuf 字段。</param>
        /// <returns>成功读取字段时为 true；输入已结束时为 false。</returns>
        public bool Next(out Field field)
        {
            if (End)
            {
                field = default;
                return false;
            }

            ulong tag = Varint();
            int wire = (int)(tag & 7);
            ulong value = 0;
            ReadOnlyMemory<byte> bytes = default;
            if (wire == 0)
            {
                value = Varint();
            }
            else if (wire == 2)
            {
                int n = checked((int)Varint());
                bytes = data.Slice(_position, n);
                _position += n;
            }
            else if (wire == 5)
            {
                value = BinaryPrimitives.ReadUInt32LittleEndian(data.Span.Slice(_position, 4));
                _position += 4;
            }
            else if (wire == 1)
            {
                value = BinaryPrimitives.ReadUInt64LittleEndian(data.Span.Slice(_position, 8));
                _position += 8;
            }
            else
            {
                throw new InvalidDataException($"Unsupported protobuf wire {wire}");
            }

            field = new(checked((int)(tag >> 3)), wire, value, bytes);
            return true;
        }
    }
}
