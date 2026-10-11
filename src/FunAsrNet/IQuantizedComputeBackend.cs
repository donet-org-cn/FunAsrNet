namespace FunAsrNet;

/// <summary>保持原 Q8_0 权重及每三十二元素激活量化规则的可选后端。</summary>
public interface IQuantizedComputeBackend
{
    /// <summary>尝试计算 Q8 线性投影，不使用更低精度或不同分组量化。</summary>
    /// <param name="input">FP32 输入，由后端按现有 Quantize 规则逐组转换。</param>
    /// <param name="frames">连续输入时间行数。</param>
    /// <param name="weight">只读的原始或交错 Q8_0 权重与缩放因子。</param>
    /// <param name="bias">可选的 FP32 偏置。</param>
    /// <param name="output">由调用方拥有的独立 FP32 结果数组。</param>
    /// <returns>成功完整处理时为 true；否则保留 CPU 路径。</returns>
    bool TryQuantizedLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output);
}
