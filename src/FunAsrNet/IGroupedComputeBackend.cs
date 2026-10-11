namespace FunAsrNet;

/// <summary>合并共享输入的独立投影，以减少准备与设备同步次数。</summary>
public interface IGroupedComputeBackend
{
    /// <summary>尝试同时计算多个只读权重对应的线性投影。</summary>
    /// <param name="input">所有投影共享的连续输入，不得修改。</param>
    /// <param name="frames">连续输入行数。</param>
    /// <param name="weights">发布后只读、列数相同的稳定权重数组。</param>
    /// <param name="outputs">成功时按权重顺序返回独立 CPU 数组。</param>
    /// <returns>成功完整计算时为 true，否则输出为空并由 CPU 继续。</returns>
    bool TryLinearMany(float[] input, int frames, Weight[] weights, out float[][]? outputs);
}
