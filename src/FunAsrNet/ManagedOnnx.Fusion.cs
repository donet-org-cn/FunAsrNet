namespace FunAsrNet;

/// <summary>在没有中间观察者的纯 CPU 路径中融合相邻卷积与残差或激活。</summary>
public sealed partial class ManagedOnnx
{
    private readonly Dictionary<int, float?> _convFusions = [];

    /// <summary>本次运行实际完成的相邻卷积融合数量，供内部验证使用。</summary>
    internal int ConvFusionsExecuted { get; private set; }

    /// <summary>只规划未作为模型输出且唯一消费者紧邻的卷积，保留原加法操作数顺序。</summary>
    private void BuildConvFusionPlans()
    {
        for (int index = 0; index + 1 < _model.Nodes.Count; index++)
        {
            var convolution = _model.Nodes[index];
            var consumer = _model.Nodes[index + 1];
            if (convolution.Op != "Conv" || convolution.Inputs.Length is < 2 or > 3
                || convolution.Outputs.Length != 1 || consumer.Outputs.Length != 1
                || _model.Outputs.Contains(convolution.Outputs[0])
                || _uses.GetValueOrDefault(convolution.Outputs[0]) != 1)
            {
                continue;
            }
            string intermediate = convolution.Outputs[0];
            if (consumer.Op == "Add" && consumer.Inputs.Length == 2 && consumer.Inputs[0] == intermediate)
            {
                _convFusions.Add(index, null);
            }
            else if (consumer.Op == "LeakyRelu" && consumer.Inputs.Length == 1 && consumer.Inputs[0] == intermediate)
            {
                _convFusions.Add(index, consumer.Float("alpha", 0.01f));
            }
        }
    }

    /// <summary>运行时确认残差具有完整输出形状，随后让卷积写入独立的最终结果存储。</summary>
    /// <param name="index">当前卷积在原模型节点序列中的位置。</param>
    /// <param name="inputs">已解析且只读的卷积输入、权重和可选偏置。</param>
    /// <param name="values">当前仍然有效的命名张量，包括相邻残差输入。</param>
    /// <param name="result">成功时返回融合后的最终张量，失败时为空。</param>
    /// <returns>只有满足计划及运行时形状约束并完成融合时才返回真。</returns>
    private bool TryExecuteConvFusion(int index, Tensor?[] inputs, Dictionary<string, Tensor> values, out Tensor? result)
    {
        result = null;
        if (!_convFusions.TryGetValue(index, out float? alpha)
            || inputs[0] is not { Rank: 3, Integer: false } x
            || inputs[1] is not { Rank: 3, Integer: false } w)
        {
            return false;
        }
        var convolution = _model.Nodes[index];
        Tensor? residual = null;
        if (alpha is null)
        {
            if (!values.TryGetValue(_model.Nodes[index + 1].Inputs[1], out residual)
                || residual.Rank != 3 || residual.Integer)
            {
                return false;
            }
            int stride = convolution.Ints("strides", [1])[0];
            int dilation = convolution.Ints("dilations", [1])[0];
            int[] pads = convolution.Ints("pads", [0, 0]);
            if (stride <= 0 || dilation <= 0 || pads.Length != 2)
            {
                return false;
            }
            int length = (x.Shape[2] + pads[0] + pads[1] - dilation * (w.Shape[2] - 1) - 1) / stride + 1;
            if (residual.Shape[0] != x.Shape[0] || residual.Shape[1] != w.Shape[0] || residual.Shape[2] != length)
            {
                return false;
            }
        }
        result = TensorOps.Conv1DFused(x, w, inputs.Length == 3 ? inputs[2] : null, convolution, _math, residual, alpha);
        return true;
    }

    /// <summary>按原节点顺序消费命名引用，最后一个数组别名结束后才归还工作区。</summary>
    /// <param name="node">已经完成读取的原模型节点。</param>
    /// <param name="values">尚未结束生命期的命名张量集合。</param>
    /// <param name="remaining">包含模型输出保护引用的剩余读取次数。</param>
    /// <param name="references">每个数组仍持有的命名别名数量。</param>
    /// <param name="virtualInput">融合省略的中间名；只消费计数而不访问不存在的数组。</param>
    private void ReleaseNodeInputs(OnnxNode node, Dictionary<string, Tensor> values, Dictionary<string, int> remaining,
        Dictionary<float[], int> references, string? virtualInput = null)
    {
        foreach (string input in node.Inputs)
        {
            if (input.Length == 0 || --remaining[input] != 0 || _model.Constants.ContainsKey(input) || input == virtualInput)
            {
                continue;
            }
            float[] array = values[input].Data;
            values.Remove(input);
            if (--references[array] == 0)
            {
                references.Remove(array);
                TensorWorkspace.Release(array);
            }
        }
    }
}
