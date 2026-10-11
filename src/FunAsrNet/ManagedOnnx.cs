namespace FunAsrNet;

/// <summary>
/// 执行固定 Huayan 和 AISHELL3 VITS 模型所需的 ONNX 算子子集，不依赖原生推理库。
/// </summary>
public sealed class ManagedOnnx
{
    private readonly OnnxModel _model;
    private readonly TensorMath _math;
    private readonly Dictionary<string, int> _uses = [];
    private Random _random = new(1);
    /// <summary>
    /// 可选的中间张量观察回调，仅供验证使用；开启后会产生额外快照分配。
    /// </summary>
    public Action<int, OnnxNode, Tensor[]>? Trace
    {
        get; set;
    }

    /// <summary>
    /// 可选的节点性能回调，提供节点、输入形状及执行毫秒数；关闭时不读取时钟。
    /// </summary>
    public Action<OnnxNode, Tensor?[], double>? NodeMeasured
    {
        get; set;
    }

    /// <summary>可选计算后端，须在执行前配置；后端资源由调用方管理。</summary>
    public IFloatComputeBackend? FloatBackend
    {
        get => _math.FloatBackend;
        set => _math.FloatBackend = value;
    }

    /// <summary>
    /// 为共享只读模型创建独立执行状态，不依赖原生推理库。
    /// </summary>
    /// <param name="model">已加载的模型及其权重；发布给并发上下文后不得修改权重。</param>
    /// <param name="math">用于矩阵与向量运算的托管计算内核。</param>
    public ManagedOnnx(OnnxModel model, TensorMath math)
    {
        _model = model;
        _math = math;
        foreach (var node in model.Nodes)
        {
            foreach (string input in node.Inputs)
            {
                if (input.Length != 0)
                {
                    _uses[input] = _uses.GetValueOrDefault(input) + 1;
                }
            }
        }

        foreach (string output in model.Outputs)
        {
            _uses[output] = _uses.GetValueOrDefault(output) + 1;
        }
    }

    /// <summary>
    /// 使用指定随机种子执行模型图，返回命名输出张量；同一实例须串行运行。
    /// </summary>
    /// <param name="inputs">按模型输入名称索引的张量集合。</param>
    /// <param name="seed">随机种子；相同模型、文本、参数和运行环境可重复生成。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs, int seed = 1)
    {
        _random = new Random(seed);
        var values = new Dictionary<string, Tensor>(_model.Constants);
        var remaining = new Dictionary<string, int>(_uses);
        foreach (var item in inputs)
        {
            values[item.Key] = item.Value;
        }

        for (int i = 0; i < _model.Nodes.Count; ++i)
        {
            var node = _model.Nodes[i];
            Tensor?[] x = node.Inputs.Select(n => n.Length == 0 ? null : values[n]).ToArray();
            Tensor[] output;
            long started = NodeMeasured is null ? 0 : System.Diagnostics.Stopwatch.GetTimestamp();
            try
            {
                output = Execute(node, x);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException($"ONNX node {i} {node.Name} ({node.Op}) inputs={string.Join(';', x.Select(t => t?.ToString() ?? "none"))}", e);
            }

            NodeMeasured?.Invoke(node, x, System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);

            if (output.Length != node.Outputs.Length)
            {
                throw new InvalidDataException("Output count mismatch");
            }

            for (int j = 0; j < output.Length; ++j)
            {
                values[node.Outputs[j]] = output[j];
            }

            Trace?.Invoke(i, node, output);
            foreach (string input in node.Inputs)
            {
                if (input.Length != 0 && --remaining[input] == 0 && !_model.Constants.ContainsKey(input))
                {
                    values.Remove(input);
                }
            }
        }

        return _model.Outputs.ToDictionary(n => n, n => values[n]);
    }

    /// <summary>
    /// 执行一个 ONNX 节点，解析属性并分派到托管算子。
    /// </summary>
    /// <param name="n">要执行的 ONNX 图节点。</param>
    /// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    private Tensor[] Execute(OnnxNode n, Tensor?[] input)
    {
        // <summary>
        // 读取当前节点指定位置的必需输入张量。
        // </summary>
        // <param name="index">从零开始的行、元素或任务索引。</param>
        // <returns>具有输出形状的结果张量。</returns>
        Tensor X(int index) => input[index] ?? throw new ArgumentException("Required input missing");
        // <summary>
        // 读取可选输入；节点没有提供该输入时返回空值。
        // </summary>
        // <param name="index">从零开始的行、元素或任务索引。</param>
        // <returns>调用成功时返回的结果；失败通过异常报告。</returns>
        Tensor? Optional(int index) => index < input.Length ? input[index] : null;
        // <summary>
        // 读取算子输入或属性中定义的轴列表。
        // </summary>
        // <param name="index">从零开始的行、元素或任务索引。</param>
        // <returns>调用成功时返回的结果；失败通过异常报告。</returns>
        int[]? Axes(int index = 1) => Optional(index)?.Ints() ?? (n.Attributes.ContainsKey("axes") ? n.Ints("axes") : null);
        // <summary>
        // 解析节点输入并执行一元张量运算。
        // </summary>
        // <param name="operation">应用于输入元素的运算函数。</param>
        // <param name="integer">是否将张量视为模型的整数形状或索引数据。</param>
        // <returns>具有输出形状的结果张量。</returns>
        Tensor U(Func<float, float> operation, bool integer = false) => TensorOps.Unary(X(0), operation, integer);
        // <summary>
        // 解析两个节点输入并执行带广播的二元张量运算。
        // </summary>
        // <param name="operation">应用于输入元素的运算函数。</param>
        // <param name="integer">是否将张量视为模型的整数形状或索引数据。</param>
        // <returns>具有输出形状的结果张量。</returns>
        Tensor B(Func<float, float, float> operation, bool? integer = null) => TensorOps.Binary(X(0), X(1), operation, integer ?? (X(0).Integer && X(1).Integer));
        Tensor result;
        switch (n.Op)
        {
            case "Add":
            case "Sub":
            case "Mul":
            case "Div":
                result = TensorOps.Arithmetic(X(0), X(1), n.Op);
                break;
            case "Pow":
                result = B(MathF.Pow);
                break;
            case "Equal":
                result = B((a, b) => a == b ? 1 : 0, true);
                break;
            case "Less":
                result = B((a, b) => a < b ? 1 : 0, true);
                break;
            case "LessOrEqual":
                result = B((a, b) => a <= b ? 1 : 0, true);
                break;
            case "GreaterOrEqual":
                result = B((a, b) => a >= b ? 1 : 0, true);
                break;
            case "And":
                result = B((a, b) => a != 0 && b != 0 ? 1 : 0, true);
                break;
            case "Not":
                result = U(a => a == 0 ? 1 : 0, true);
                break;
            case "Neg":
                result = U(a => -a, X(0).Integer);
                break;
            case "Sqrt":
                result = U(MathF.Sqrt);
                break;
            case "Exp":
                result = U(MathF.Exp);
                break;
            case "Log":
                result = U(MathF.Log);
                break;
            case "Tanh":
                result = U(MathF.Tanh);
                break;
            case "Erf":
                result = U(Erf);
                break;
            case "Sigmoid":
                result = U(a => 1 / (1 + MathF.Exp(-a)));
                break;
            case "Softplus":
                result = U(a => a > 20 ? a : MathF.Log(1 + MathF.Exp(a)));
                break;
            case "Relu":
                result = U(a => Math.Max(0, a));
                break;
            case "LeakyRelu":
                float alpha = n.Float("alpha", 0.01f);
                result = TensorOps.LeakyRelu(X(0), alpha);
                break;
            case "Ceil":
                result = U(MathF.Ceiling);
                break;
            case "Clip":
                float min = Optional(1)?.Data[0] ?? float.MinValue, max = Optional(2)?.Data[0] ?? float.MaxValue;
                result = U(a => Math.Clamp(a, min, max));
                break;
            case "Cast":
                result = n.Int("to") == 7 ? U(MathF.Truncate, true) : new Tensor(X(0).Data, X(0).Shape);
                break;
            case "Identity":
                result = X(0);
                break;
            case "Constant":
                result = (Tensor)n.Attributes["value"];
                break;
            case "ConstantOfShape":
                result = Tensor.Zeros(X(0).Ints());
                Array.Fill(result.Data, n.Attributes.TryGetValue("value", out var value) ? ((Tensor)value).Data[0] : 0);
                break;
            case "Shape":
                int start = n.Int("start"), end = n.Int("end", X(0).Rank);
                start = Math.Clamp(TensorOps.Axis(start, X(0).Rank), 0, X(0).Rank);
                end = Math.Clamp(TensorOps.Axis(end, X(0).Rank), start, X(0).Rank);
                result = new(X(0).Shape[start..end].Select(i => (float)i).ToArray(), [end - start], true);
                break;
            case "Range":
                float first = X(0).Data[0], limit = X(1).Data[0], delta = X(2).Data[0];
                int count = Math.Max(0, checked((int)Math.Ceiling((limit - first) / delta)));
                if (count > 10000000)
                {
                    throw new ArgumentException("Range exceeds inference limit");
                }

                result = new(Enumerable.Range(0, count).Select(i => first + i * delta).ToArray(), [count], X(0).Integer);
                break;
            case "Reshape":
                result = TensorOps.Reshape(X(0), X(1).Ints(), n.Int("allowzero") != 0);
                break;
            case "Unsqueeze":
                result = TensorOps.Unsqueeze(X(0), Axes()!);
                break;
            case "Squeeze":
                result = TensorOps.Squeeze(X(0), Axes());
                break;
            case "Transpose":
                result = TensorOps.Transpose(X(0), n.Attributes.ContainsKey("perm") ? n.Ints("perm") : null);
                break;
            case "Gather":
                result = TensorOps.Gather(X(0), X(1), n.Int("axis"));
                break;
            case "GatherElements":
                result = TensorOps.GatherElements(X(0), X(1), n.Int("axis"));
                break;
            case "GatherND":
                if (n.Int("batch_dims") != 0)
                {
                    throw new NotSupportedException("GatherND batch_dims");
                }

                result = TensorOps.GatherND(X(0), X(1));
                break;
            case "ScatterND":
                result = TensorOps.ScatterND(X(0), X(1), X(2));
                break;
            case "Slice":
                result = TensorOps.Slice(X(0), X(1), X(2), Optional(3)?.Ints(), Optional(4)?.Ints());
                break;
            case "Concat":
                result = TensorOps.Concat(input.Select(x => x!).ToArray(), n.Int("axis"));
                break;
            case "Expand":
                result = TensorOps.Expand(X(0), X(1).Ints());
                break;
            case "Tile":
                result = TensorOps.Tile(X(0), X(1).Ints());
                break;
            case "Where":
                result = TensorOps.Where(X(0), X(1), X(2));
                break;
            case "Pad":
                result = TensorOps.Pad(X(0), X(1).Ints(), Optional(2)?.Data[0] ?? 0);
                break;
            case "ReduceMean":
            case "ReduceSum":
            case "ReduceMax":
                result = TensorOps.Reduce(X(0), Axes(), n.Int("keepdims", 1) != 0, n.Op);
                break;
            case "CumSum":
                result = TensorOps.CumSum(X(0), (int)X(1).Data[0], n.Int("exclusive") != 0, n.Int("reverse") != 0);
                break;
            case "NonZero":
                result = TensorOps.NonZero(X(0));
                break;
            case "Softmax":
                result = TensorOps.Softmax(X(0), n.Int("axis", -1));
                break;
            case "Split":
                int axis = TensorOps.Axis(n.Int("axis"), X(0).Rank), offset = 0;
                int[] lengths = Optional(1)?.Ints() ?? Enumerable.Repeat(X(0).Shape[axis] / n.Outputs.Length, n.Outputs.Length).ToArray();
                return lengths.Select(length =>
                {
                    var t = TensorOps.Slice(X(0), new([(float)offset], [1], true), new([(float)(offset + length)], [1], true), [axis], [1]);
                    offset += length;
                    return t;
                }).ToArray();
            case "MatMul":
                result = TensorOps.MatMul(X(0), X(1), _math);
                break;
            case "Conv":
                result = TensorOps.Conv1D(X(0), X(1), Optional(2), n, _math);
                break;
            case "ConvTranspose":
                result = TensorOps.ConvTranspose1D(X(0), X(1), Optional(2), n, _math);
                break;
            case "RandomNormalLike":
                result = Tensor.Zeros(X(0).Shape);
                for (int i = 0; i < result.Count; ++i)
                {
                    result.Data[i] = (float)(Math.Sqrt(-2 * Math.Log(1 - _random.NextDouble())) * Math.Cos(2 * Math.PI * _random.NextDouble())) * n.Float("scale", 1) + n.Float("mean");
                }

                break;
            default:
                throw new NotSupportedException($"ONNX operator {n.Op}");
        }

        return [result];
    }

    /// <summary>
    /// 计算误差函数的近似值，供激活算子使用。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <returns>计算得到的标量数值。</returns>
    private static float Erf(float x)
    {
        double a = Math.Abs(x), t = 1 / (1 + 0.3275911 * a);
        double p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        return (float)Math.CopySign(1 - p * Math.Exp(-a * a), x);
    }
}
