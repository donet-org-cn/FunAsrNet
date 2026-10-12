namespace FunAsrNet;

/// <summary>在命名引用和工作区所有权共同保证独占时复用逐元素算子的输入存储。</summary>
public sealed partial class ManagedOnnx
{
    /// <summary>仅选择最后一次读取的独占临时张量，观察回调、外部输入和别名均禁止覆盖。</summary>
    /// <param name="node">当前节点及其命名输入。</param>
    /// <param name="inputs">本次执行的输入张量。</param>
    /// <param name="remaining">每个命名张量尚未完成的读取次数，包含模型输出。</param>
    /// <param name="references">每个数组仍存在的命名引用数量，包含共享视图。</param>
    /// <returns>可以复用的输入；没有满足条件的输入时返回空。</returns>
    private Tensor? FindReusableInput(OnnxNode node, Tensor?[] inputs, Dictionary<string, int> remaining, Dictionary<float[], int> references)
    {
        if (!ReuseBuffers || Trace is not null || NodeMeasured is not null || Environment.GetEnvironmentVariable("FUNASRNET_INPLACE_TTS") == "0")
        {
            return null;
        }
        int candidates;
        int[] shape;
        if (node.Op == "LeakyRelu" && inputs.Length == 1 && inputs[0] is not null)
        {
            candidates = 1;
            shape = inputs[0]!.Shape;
        }
        else if (node.Op is "Add" or "Sub" or "Mul" or "Div" && inputs.Length == 2 && inputs[0] is { Integer: false } a && inputs[1] is { Integer: false } b
            && (a.Shape.SequenceEqual(b.Shape) || a.Count == 1 || b.Count == 1))
        {
            candidates = 2;
            shape = TensorOps.BroadcastShape(a.Shape, b.Shape);
        }
        else
        {
            return null;
        }
        for (int index = 0; index < candidates; index++)
        {
            var input = inputs[index]!;
            if (input.Shape.SequenceEqual(shape) && remaining[node.Inputs[index]] == 1 && references[input.Data] == 1 && TensorWorkspace.Owns(input.Data))
            {
                return input;
            }
        }
        return null;
    }
}
