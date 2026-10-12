namespace FunAsrNet;

/// <summary>可选的连续浮点投影后端，使 Linear、ReLU、Linear 的中间张量留在设备端。</summary>
public interface IReluProjectionBackend
{
    /// <summary>按原浮点次序执行两次投影，成功时仅同步最终输出。</summary>
    /// <param name="input">只读连续输入。</param>
    /// <param name="frames">输入行数。</param>
    /// <param name="first">第一层只读权重。</param>
    /// <param name="firstBias">第一层可选偏置。</param>
    /// <param name="second">第二层只读权重。</param>
    /// <param name="secondBias">第二层可选偏置。</param>
    /// <param name="output">独立的最终输出存储。</param>
    /// <returns>完成返回真；不支持时不修改输出并返回假。</returns>
    bool TryReluProjection(float[] input, int frames, Weight first, Weight? firstBias, Weight second, Weight? secondBias, float[] output);
}
