/// <summary>
/// 模拟网络分片：不可寻址，每次读取最多七字节。
/// </summary>
/// <param name="bytes">当前调用拥有或借用的字节数据。</param>
internal sealed class FragmentedReadStream(byte[] bytes) : Stream
{
    private readonly MemoryStream _inner = new(bytes);
    /// <summary>
    /// 测试流是否允许读取。
    /// </summary>
    public override bool CanRead => _inner.CanRead;
    /// <summary>
    /// 测试流不允许寻址，始终返回 false。
    /// </summary>
    public override bool CanSeek => false;
    /// <summary>
    /// 测试流不允许写入，始终返回 false。
    /// </summary>
    public override bool CanWrite => false;
    /// <summary>
    /// 底层测试数据的字节长度。
    /// </summary>
    public override long Length => throw new NotSupportedException();
    /// <summary>
    /// 当前测试读取位置；不允许通过该属性修改位置。
    /// </summary>
    public override long Position
    {
        get => throw new NotSupportedException(); set => throw new NotSupportedException();
    }

    /// <summary>
    /// 从测试输入读取不超过限定长度的数据。
    /// </summary>
    /// <param name="buffer">可复用的读写缓冲区。</param>
    /// <param name="offset">输入数据的起始元素偏移。</param>
    /// <param name="count">需要处理的元素或任务数量。</param>
    /// <returns>计算得到的整数结果。</returns>
    public override int Read(byte[] buffer, int offset, int count) => _inner.Read(buffer, offset, Math.Min(count, 7));
    /// <summary>
    /// 异步读取测试输入的小块数据，模拟网络短读。
    /// </summary>
    /// <param name="buffer">可复用的读写缓冲区。</param>
    /// <param name="cancellationToken">用于取消排队、读取或异步等待的令牌。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) => _inner.ReadAsync(buffer[..Math.Min(buffer.Length, 7)], cancellationToken);
    /// <summary>
    /// 拒绝寻址，以验证调用方不依赖可寻址流。
    /// </summary>
    /// <param name="offset">输入数据的起始元素偏移。</param>
    /// <param name="origin">寻址基准；此测试流不支持寻址。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    /// <summary>
    /// 拒绝修改只读测试流的长度。
    /// </summary>
    /// <param name="value">待编码、填充或处理的值。</param>
    public override void SetLength(long value) => throw new NotSupportedException();
    /// <summary>
    /// 拒绝写入只读测试流。
    /// </summary>
    /// <param name="buffer">可复用的读写缓冲区。</param>
    /// <param name="offset">输入数据的起始元素偏移。</param>
    /// <param name="count">需要处理的元素或任务数量。</param>
    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    /// <summary>
    /// 只读测试流不持有待写入数据。
    /// </summary>
    public override void Flush() => throw new NotSupportedException();
    /// <summary>
    /// 释放持有的资源或恢复进入作用域之前的状态；调用者不应再使用已释放资源。
    /// </summary>
    /// <param name="disposing">是否由显式释放触发；为 true 时释放托管资源。</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _inner.Dispose();
        }

        base.Dispose(disposing);
    }
}
