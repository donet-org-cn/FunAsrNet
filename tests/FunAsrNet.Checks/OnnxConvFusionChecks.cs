using FunAsrNet;

/// <summary>检查相邻卷积融合的准入条件、观察回退以及命名和数组别名生命期。</summary>
internal static class OnnxConvFusionChecks
{
    /// <summary>与关闭融合的解释器逐位比较，并检查保留输出、动态残差和外部输入不会被覆盖。</summary>
    /// <param name="check">接收断言结果及其检查说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        string? previous = Environment.GetEnvironmentVariable("FUNASRNET_FUSE_CONV");
        try
        {
            foreach (string operation in new[] { "Add", "LeakyRelu" })
            {
                foreach (int groups in new[] { 1, 2 })
                {
                    var model = CreateModel(operation, groups);
                    var inputs = CreateInputs(groups);
                    var snapshots = inputs.ToDictionary(item => item.Key, item => (float[])item.Value.Data.Clone());
                    var constants = model.Constants.ToDictionary(item => item.Key, item => (float[])item.Value.Data.Clone());
                    var reference = new ManagedOnnx(model, new TensorMath(4)) { ReuseBuffers = false };
                    var optimized = new ManagedOnnx(model, new TensorMath(4));
                    Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", "0");
                    var expected = reference.Run(inputs);
                    Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", "1");
                    var first = optimized.Run(inputs);
                    var saved = first.ToDictionary(item => item.Key, item => (float[])item.Value.Data.Clone());
                    check(optimized.ConvFusionsExecuted == 1, "ONNX eligible adjacent convolution performs one fusion");
                    Compare(first, expected, check, "ONNX fused convolution matches separate nodes exactly");
                    for (int iteration = 0; iteration < 4; iteration++)
                    {
                        Compare(optimized.Run(inputs), expected, check, "ONNX fused aliases match across repeated workspace reuse");
                        foreach (var item in first)
                        {
                            check(EqualBits(item.Value.Data, saved[item.Key]), "ONNX fused earlier output survives subsequent inference");
                        }
                    }
                    check(inputs.All(item => EqualBits(item.Value.Data, snapshots[item.Key])), "ONNX fused convolution leaves caller input and residual untouched");
                    check(model.Constants.All(item => EqualBits(item.Value.Data, constants[item.Key])), "ONNX fused convolution leaves shared constants untouched");

                    List<(Tensor Tensor, float[] Snapshot)> observed = [];
                    optimized.Trace = (_, _, output) => observed.AddRange(output.Select(t => (t, (float[])t.Data.Clone())));
                    Compare(optimized.Run(inputs), expected, check, "ONNX trace falls back to original convolution nodes");
                    check(optimized.ConvFusionsExecuted == 0 && observed.Count == model.Nodes.Count, "ONNX trace observes every original node with fusion disabled");
                    optimized.Trace = null;
                    optimized.Run(inputs);
                    check(observed.All(item => EqualBits(item.Tensor.Data, item.Snapshot)), "ONNX observed intermediates survive resumed fusion");

                    int measured = 0;
                    optimized.NodeMeasured = (_, _, _) => measured++;
                    Compare(optimized.Run(inputs), expected, check, "ONNX profiling falls back to separate convolution nodes");
                    check(optimized.ConvFusionsExecuted == 0 && measured == model.Nodes.Count, "ONNX profiling receives every original node");
                    optimized.NodeMeasured = null;

                    var backend = new DecliningBackend();
                    optimized.FloatBackend = backend;
                    Compare(optimized.Run(inputs), expected, check, "ONNX optional backend preserves original convolution dispatch");
                    check(optimized.ConvFusionsExecuted == 0 && backend.ConvolutionCalls == 1, "ONNX optional backend bypasses CPU graph fusion");
                    optimized.FloatBackend = null;

                    var invalid = new Dictionary<string, Tensor>(inputs)
                    {
                        ["input"] = new Tensor([1, 2, 3], [3])
                    };
                    bool failed = false;
                    try
                    {
                        optimized.Run(invalid);
                    }
                    catch (InvalidOperationException)
                    {
                        failed = true;
                    }
                    check(failed, "ONNX fused graph rejects invalid convolution input");
                    Compare(optimized.Run(inputs), expected, check, "ONNX fused workspace recovers after an exception");

                    if (operation == "Add")
                    {
                        var broadcast = new Dictionary<string, Tensor>(inputs)
                        {
                            ["residual"] = new Tensor(Enumerable.Range(0, 5 * groups).Select(i => i * 0.125f).ToArray(), [1, 5 * groups, 1])
                        };
                        Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", "0");
                        var broadcastExpected = reference.Run(broadcast);
                        Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", "1");
                        Compare(optimized.Run(broadcast), broadcastExpected, check, "ONNX broadcast residual preserves original Add behavior");
                        check(optimized.ConvFusionsExecuted == 0, "ONNX dynamic residual shape mismatch disables fusion");
                    }
                }
            }

            foreach (string rejection in new[] { "model-output", "other-consumer", "repeat-input", "reversed-add", "non-adjacent" })
            {
                var model = CreateModel("Add", 1);
                switch (rejection)
                {
                    case "model-output": model.Outputs.Add("convolution"); break;
                    case "other-consumer":
                        model.Nodes.Add(new("other-consumer", "Identity", ["convolution"], ["other"], []));
                        model.Outputs.Add("other");
                        break;
                    case "repeat-input": model.Nodes[3] = new("repeat-input", "Add", ["convolution", "convolution"], ["result"], []); break;
                    case "reversed-add": model.Nodes[3] = new("reversed-add", "Add", ["residual", "convolution"], ["result"], []); break;
                    case "non-adjacent": model.Nodes.Insert(3, new("gap", "Identity", ["alias"], ["gap"], [])); break;
                }
                var inputs = CreateInputs(1);
                Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", "0");
                var expected = new ManagedOnnx(model, new TensorMath(1)).Run(inputs);
                Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", "1");
                var optimized = new ManagedOnnx(model, new TensorMath(1));
                Compare(optimized.Run(inputs), expected, check, "ONNX ineligible fusion preserves original graph values");
                check(optimized.ConvFusionsExecuted == 0, "ONNX rejects convolution fusion when " + rejection);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("FUNASRNET_FUSE_CONV", previous);
        }
    }

    /// <summary>构造带保留输入别名和后续原地消费者的最小卷积图。</summary>
    /// <param name="operation">紧邻卷积的残差加法或激活名称。</param>
    /// <param name="groups">卷积分组数，覆盖完整输出块和组内尾部。</param>
    /// <returns>包含只读权重及两个对外输出的模型。</returns>
    private static OnnxModel CreateModel(string operation, int groups)
    {
        string path = Path.Combine(Path.GetTempPath(), $"funasr-conv-fusion-{Guid.NewGuid():N}.onnx");
        OnnxModel model;
        try
        {
            File.WriteAllBytes(path, [0x3a, 0x02, 0x12, 0x00]);
            model = new OnnxModel(path);
        }
        finally
        {
            File.Delete(path);
        }
        int channels = groups == 1 ? 4 : 6, outputs = 5 * groups;
        model.Constants["zero"] = new Tensor([0], []);
        model.Constants["weights"] = new Tensor(Enumerable.Range(0, outputs * (channels / groups) * 3).Select(i => (i % 19 - 9) * 0.03125f).ToArray(), [outputs, channels / groups, 3]);
        model.Constants["bias"] = new Tensor(Enumerable.Range(0, outputs).Select(i => (i - 2) * 0.0625f).ToArray(), [outputs]);
        model.Nodes.Add(new("owned", "Add", ["input", "zero"], ["owned"], []));
        model.Nodes.Add(new("alias", "Identity", ["owned"], ["alias"], []));
        model.Nodes.Add(new("convolution", "Conv", ["owned", "weights", "bias"], ["convolution"], new()
        {
            ["group"] = groups,
            ["dilations"] = new long[] { 3 },
            ["pads"] = new long[] { 3, 3 }
        }));
        model.Nodes.Add(operation == "Add"
            ? new("residual", "Add", ["convolution", "residual"], ["result"], [])
            : new("activation", "LeakyRelu", ["convolution"], ["result"], new() { ["alpha"] = 0.1f }));
        model.Nodes.Add(new("final", "Add", ["result", "zero"], ["final"], []));
        model.Outputs.AddRange(["alias", "final"]);
        return model;
    }

    /// <summary>创建有符号输入与完整残差，长度包含向量和时间任务的尾部。</summary>
    /// <param name="groups">需要的卷积分组数。</param>
    /// <returns>调用方持有的独立输入字典。</returns>
    private static Dictionary<string, Tensor> CreateInputs(int groups)
    {
        int channels = groups == 1 ? 4 : 6, length = 273, outputs = 5 * groups;
        return new()
        {
            ["input"] = new Tensor(Enumerable.Range(0, 2 * channels * length).Select(i => (i % 31 - 15) * 0.125f).ToArray(), [2, channels, length]),
            ["residual"] = new Tensor(Enumerable.Range(0, 2 * outputs * length).Select(i => (i % 11 - 5) * 0.0625f).ToArray(), [2, outputs, length])
        };
    }

    /// <summary>逐个比较命名输出的形状与浮点位模式。</summary>
    /// <param name="actual">被检查的运行结果。</param>
    /// <param name="expected">原节点运行产生的参考结果。</param>
    /// <param name="check">接收断言的回调。</param>
    /// <param name="label">此次比较的检查说明。</param>
    private static void Compare(Dictionary<string, Tensor> actual, Dictionary<string, Tensor> expected, Action<bool, string> check, string label)
    {
        foreach (var item in expected)
        {
            check(actual[item.Key].Shape.SequenceEqual(item.Value.Shape) && EqualBits(actual[item.Key].Data, item.Value.Data), label);
        }
    }

    /// <summary>保留负零和非数载荷的数组逐位相等比较。</summary>
    /// <param name="actual">实际浮点数组。</param>
    /// <param name="expected">参考浮点数组。</param>
    /// <returns>长度及所有位模式一致时返回真。</returns>
    private static bool EqualBits(float[] actual, float[] expected)
    {
        return actual.Length == expected.Length && actual.Zip(expected).All(pair => BitConverter.SingleToInt32Bits(pair.First) == BitConverter.SingleToInt32Bits(pair.Second));
    }

    /// <summary>拒绝处理计算的测试后端，确保解释器仍向它分派原卷积。</summary>
    private sealed class DecliningBackend : IFloatComputeBackend
    {
        /// <summary>后端的测试名称。</summary>
        public string Name => "declining-test";

        /// <summary>已经收到的原始卷积分派次数。</summary>
        internal int ConvolutionCalls { get; private set; }

        /// <summary>拒绝处理线性计算，并保持调用方输出原样。</summary>
        /// <param name="input">只读输入数组。</param>
        /// <param name="frames">有效时间帧数。</param>
        /// <param name="weight">只读模型权重。</param>
        /// <param name="bias">可选只读偏置。</param>
        /// <param name="output">不应修改的输出数组。</param>
        /// <returns>始终返回假。</returns>
        public bool TryLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output) => false;

        /// <summary>记录分派并拒绝卷积计算，由解释器回退 CPU。</summary>
        /// <param name="input">只读卷积输入。</param>
        /// <param name="weight">只读卷积权重。</param>
        /// <param name="bias">可选只读偏置。</param>
        /// <param name="node">原图卷积节点。</param>
        /// <param name="output">返回空值以表示未处理。</param>
        /// <returns>始终返回假。</returns>
        public bool TryConvolution(Tensor input, Tensor weight, Tensor? bias, OnnxNode node, out Tensor? output)
        {
            ConvolutionCalls++;
            output = null;
            return false;
        }
    }
}
