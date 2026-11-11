namespace FunAsrNet;

/// <summary>可选的非因果 FP32 多头注意力后端，输出同步回到 CPU。</summary>
public interface IAttentionComputeBackend
{
    /// <summary>尝试完成分数、Softmax 和加权值，不支持时保持输出不变。</summary>
    /// <param name="query">按时间、头、通道排列的只读查询。</param>
    /// <param name="key">与查询形状相同的只读键。</param>
    /// <param name="value">与查询形状相同的只读值。</param>
    /// <param name="frames">有效时间帧数。</param>
    /// <param name="width">一帧的通道总数，必须能被头数整除。</param>
    /// <param name="heads">注意力头数。</param>
    /// <param name="output">与查询等长的独立输出，成功后可立即读取。</param>
    /// <returns>支持且计算完成时返回真。</returns>
    bool TryAttention(float[] query, float[] key, float[] value, int frames, int width, int heads, float[] output);
}
