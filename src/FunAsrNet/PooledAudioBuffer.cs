using System.Buffers;

namespace FunAsrNet;

/// <summary>
/// 拥有一个音频池缓冲区；只公开有效采样范围，释放后禁止继续访问。
/// </summary>
public sealed class PooledAudioBuffer : IMemoryOwner<float>
{
    private float[]? _array;
    private readonly int _length;
    /// <summary>
    /// 租用指定有效长度的音频缓冲区，内容尚未初始化。
    /// </summary>
    /// <param name="length">需要的有效元素数量。</param>
    public PooledAudioBuffer(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        _length = length;
        _array = ArrayPool<float>.Shared.Rent(length);
    }

    /// <summary>
    /// 获取有效采样视图；视图不复制数据，其生命期不能超过缓冲区。
    /// </summary>
    public Memory<float> Memory => (_array ?? throw new ObjectDisposedException(nameof(PooledAudioBuffer))).AsMemory(0, _length);

    /// <summary>
    /// 将数组归还池；重复释放不会重复归还。
    /// </summary>
    public void Dispose()
    {
        float[]? array = Interlocked.Exchange(ref _array, null);
        if (array is not null)
        {
            ArrayPool<float>.Shared.Return(array);
        }
    }
}
