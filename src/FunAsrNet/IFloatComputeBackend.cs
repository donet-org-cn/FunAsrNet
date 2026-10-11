namespace FunAsrNet;

/// <summary>可选的 FP32 计算后端；返回未处理时由现有 CPU 内核继续计算。</summary>
public interface IFloatComputeBackend
{
    /// <summary>后端的可读名称，用于诊断实际选中的设备与路径。</summary>
    string Name
    {
        get;
    }

    /// <summary>尝试计算浮点线性投影，成功时完整写入调用方输出。</summary>
    /// <param name="input">连续时间行输入，仅在本次同步调用内读取。</param>
    /// <param name="frames">输入时间行数。</param>
    /// <param name="weight">发布后只读的权重，后端不得改变其布局或精度。</param>
    /// <param name="bias">可选只读偏置。</param>
    /// <param name="output">容量与 frames 乘以输出通道数一致的独立结果数组。</param>
    /// <returns>已处理返回 true；不支持或规模过小时返回 false，不修改输出。</returns>
    bool TryLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output);

    /// <summary>尝试计算一维卷积，不支持的组或形状返回 CPU 路径。</summary>
    /// <param name="input">只读 NCL 输入张量。</param>
    /// <param name="weight">只读输出通道、输入通道、卷积核权重张量。</param>
    /// <param name="bias">可选的输出通道偏置。</param>
    /// <param name="node">提供步长、填充、膨胀及分组属性的节点。</param>
    /// <param name="output">成功时返回拥有独立存储的结果，否则为空。</param>
    /// <returns>成功完成计算时为 true。</returns>
    bool TryConvolution(Tensor input, Tensor weight, Tensor? bias, OnnxNode node, out Tensor? output);
}
