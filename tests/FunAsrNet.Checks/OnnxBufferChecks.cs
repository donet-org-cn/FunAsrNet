using FunAsrNet;

/// <summary>检查 ONNX 临时缓冲在别名、异常和跨调用输出情况下的生命期。</summary>
internal static class OnnxBufferChecks
{
    /// <summary>构造小图并与关闭复用的执行器比较，确保只回收不再被任何视图引用的数组。</summary>
    /// <param name="check">接收断言结果与检查名称的测试回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        string path = Path.Combine(Path.GetTempPath(), $"funasr-buffer-{Guid.NewGuid():N}.onnx");
        OnnxModel model;
        try
        {
            // 最小 protobuf ModelProto，GraphProto 只含名称；算子在下面显式构造。
            File.WriteAllBytes(path, [0x3a, 0x02, 0x12, 0x00]);
            model = new OnnxModel(path);
        }
        finally
        {
            File.Delete(path);
        }
        model.Constants["constant"] = new Tensor([2], []);
        model.Constants["shape"] = new Tensor([4, 4], [2], true);
        model.Nodes.Add(new("add", "Add", ["input", "constant"], ["a"], []));
        model.Nodes.Add(new("view", "Reshape", ["a", "shape"], ["v"], []));
        model.Nodes.Add(new("alias", "Identity", ["v"], ["alias"], []));
        model.Nodes.Add(new("branch", "Mul", ["a", "constant"], ["branch"], []));
        model.Nodes.Add(new("temporary", "Add", ["branch", "constant"], ["temporary"], []));
        model.Nodes.Add(new("join", "Add", ["alias", "temporary"], ["sum"], []));
        model.Nodes.Add(new("repeat-input", "Add", ["sum", "sum"], ["result"], []));
        model.Outputs.AddRange(["alias", "result"]);
        var pooled = new ManagedOnnx(model, new TensorMath(1));
        var reference = new ManagedOnnx(model, new TensorMath(1)) { ReuseBuffers = false };
        Dictionary<string, Tensor> inputs = new()
        {
            ["input"] = new Tensor(Enumerable.Range(0, 16).Select(value => (float)value).ToArray(), [4, 4])
        };
        float[] inputSnapshot = (float[])inputs["input"].Data.Clone();
        var first = pooled.Run(inputs);
        var saved = first.ToDictionary(item => item.Key, item => (float[])item.Value.Data.Clone());
        for (int iteration = 0; iteration < 8; iteration++)
        {
            var actual = pooled.Run(inputs);
            var expected = reference.Run(inputs);
            foreach (string output in model.Outputs)
            {
                check(actual[output].Data.SequenceEqual(expected[output].Data), "ONNX pooled aliases and repeated inputs match allocation reference");
                check(first[output].Data.SequenceEqual(saved[output]), "ONNX earlier outputs survive subsequent runs");
            }
        }
        check(inputs["input"].Data.SequenceEqual(inputSnapshot) && model.Constants["constant"].Data[0] == 2, "ONNX pooling leaves inputs and constants untouched");
        List<(Tensor Tensor, float[] Snapshot)> observed = [];
        pooled.Trace = (_, _, tensors) => observed.AddRange(tensors.Select(tensor => (tensor, (float[])tensor.Data.Clone())));
        pooled.Run(inputs);
        pooled.Trace = null;
        pooled.Run(inputs);
        check(observed.All(item => item.Tensor.Data.SequenceEqual(item.Snapshot)), "ONNX trace values remain stable after pooling resumes");
        var missing = new Dictionary<string, Tensor>(inputs);
        missing["input"] = new Tensor([1, 2, 3], [3]);
        bool failed = false;
        try
        {
            pooled.Run(missing);
        }
        catch (InvalidOperationException)
        {
            failed = true;
        }
        check(failed, "ONNX invalid input fails within the temporary-buffer scope");
        check(pooled.Run(inputs)["result"].Data.SequenceEqual(saved["result"]), "ONNX workspace recovers after an execution exception");
        Parallel.For(0, 8, _ =>
        {
            var independent = new ManagedOnnx(model, new TensorMath(1));
            if (!independent.Run(inputs)["result"].Data.SequenceEqual(saved["result"]))
            {
                throw new InvalidDataException("ONNX contexts shared temporary buffers");
            }
        });
        check(true, "ONNX concurrent contexts share constants without sharing temporary buffers");

        // 同形输入与标量广播都覆盖左右复用；减除能检出交换操作数的错误。
        model.Nodes.Clear();
        model.Outputs.Clear();
        model.Constants["constant"] = new Tensor([0.125f], []);
        var largeInput = new Tensor(Enumerable.Range(0, 131079).Select(i => (i % 101 - 50) * 0.25f).ToArray(), [131079]);
        var largeInputs = new Dictionary<string, Tensor> { ["input"] = largeInput };
        float[] largeSaved = (float[])largeInput.Data.Clone();
        foreach (string operation in new[] { "Add", "Sub", "Mul", "Div" })
        {
            foreach (bool rightSide in new[] { false, true })
            {
                model.Nodes.Clear();
                model.Nodes.Add(new("owned", "Add", ["input", "constant"], ["owned"], []));
                model.Nodes.Add(new("pointwise", operation, rightSide ? ["constant", "owned"] : ["owned", "constant"], ["pointwise"], []));
                model.Nodes.Add(new("activation", "LeakyRelu", ["pointwise"], ["activated"], new() { ["alpha"] = 0.1f }));
                model.Outputs.Clear();
                model.Outputs.Add("activated");
                var optimized = new ManagedOnnx(model, new TensorMath(4));
                var allocating = new ManagedOnnx(model, new TensorMath(4)) { ReuseBuffers = false };
                var actual = optimized.Run(largeInputs)["activated"];
                var expected = allocating.Run(largeInputs)["activated"];
                for (int index = 0; index < actual.Count; index++)
                {
                    check(BitConverter.SingleToInt32Bits(actual.Data[index]) == BitConverter.SingleToInt32Bits(expected.Data[index]), "ONNX in-place left/right arithmetic and activation preserve exact values across parallel tails");
                }
                check(largeInput.Data.SequenceEqual(largeSaved), "ONNX in-place operations never overwrite caller input");
                var savedOutput = (float[])actual.Data.Clone();
                optimized.Run(largeInputs);
                check(actual.Data.SequenceEqual(savedOutput), "ONNX in-place result survives next inference");
            }
        }
    }
}
