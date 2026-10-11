namespace FunAsrNet;
// 精确长度数组保持张量形状约束；只有同步推理作用域创建的临时张量
// 可以归还工作区，模型权重、KV 缓存和调用方输入始终不属于此池。
// 跟踪观察者收到独立快照，避免复用数组导致历史数据变化。
/// <summary>
/// 复用单次同步推理范围内的临时张量；不接管模型、KV 缓存或调用方输入。
/// </summary>
/// <param name="capacityBytes">允许工作区保留的缓冲区总字节数上限。</param>
internal sealed class TensorWorkspace(long capacityBytes = 64L * 1024 * 1024)
{
    [ThreadStatic]
    private static TensorWorkspace? _current;
    private readonly Dictionary<int, Stack<float[]>> _free = new();
    private readonly HashSet<float[]> _active = new(ReferenceEqualityComparer.Instance);
    private long _retained;
    /// <summary>
    /// 当前工作区保留用于复用的数组总字节数。
    /// </summary>
    internal long RetainedBytes => _retained;

    /// <summary>
    /// 进入一个同步临时张量作用域，嵌套结束时恢复外层工作区。
    /// </summary>
    /// <param name="enabled">是否启用临时张量复用。</param>
    /// <returns>用于结束当前工作区并恢复外层状态的作用域。</returns>
    internal IDisposable Enter(bool enabled = true)
    {
        if (_current is not null)
        {
            throw new InvalidOperationException("Nested tensor workspace");
        }

        if (!enabled)
        {
            return new Scope(null);
        }

        _current = this;
        return new Scope(this);
    }

    /// <summary>
    /// 获取精确长度的临时数组，并按需要清零。
    /// </summary>
    /// <param name="length">需要的有效元素数量。</param>
    /// <param name="clear">是否在返回数组前清零。</param>
    /// <returns>工作区拥有的精确长度浮点数组。</returns>
    internal static float[] Allocate(int length, bool clear = false)
    {
        var owner = _current;
        float[] array;
        if (owner is not null && owner._free.TryGetValue(length, out var bucket) && bucket.TryPop(out var found))
        {
            array = found;
            owner._retained -= 4L * length;
            if (bucket.Count == 0)
            {
                owner._free.Remove(length);
            }
        }
        else
        {
            array = GC.AllocateUninitializedArray<float>(length);
        }

        owner?._active.Add(array);
        if (clear)
        {
            Array.Clear(array);
        }

        return array;
    }

    /// <summary>
    /// 将输入复制到工作区拥有的数组，便于后续复用。
    /// </summary>
    /// <param name="source">在调用期间保持有效的只读输入视图。</param>
    /// <returns>按计算顺序排列的浮点结果数组。</returns>
    internal static float[] Copy(ReadOnlySpan<float> source)
    {
        var result = Allocate(source.Length);
        source.CopyTo(result);
        return result;
    }

    /// <summary>
    /// 将该作用域拥有且已不再使用的临时数组归还工作区。
    /// </summary>
    /// <param name="array">当前作用域拥有的临时数组。</param>
    internal static void Release(float[] array)
    {
        var owner = _current;
        if (owner is not null && owner._active.Remove(array))
        {
            owner.Keep(array);
        }
    }

    /// <summary>把返回给调用方的数组移出回收集合，后续推理不会覆盖该结果。</summary>
    /// <param name="array">当前作用域可能拥有的结果数组，不接管外部存储。</param>
    internal static void Detach(float[] array) => _current?._active.Remove(array);

    /// <summary>
    /// 将已结束生命期的数组保留在有界缓存中。
    /// </summary>
    /// <param name="array">当前作用域拥有的临时数组。</param>
    private void Keep(float[] array)
    {
        long bytes = 4L * array.Length;
        if (bytes > capacityBytes)
        {
            return;
        }

        while (_retained + bytes > capacityBytes && _free.Count != 0)
        {
            var oldest = _free.First();
            _retained -= 4L * oldest.Key * oldest.Value.Count;
            _free.Remove(oldest.Key);
        }

        if (!_free.TryGetValue(array.Length, out var bucket))
        {
            _free[array.Length] = bucket = new();
        }

        bucket.Push(array);
        _retained += bytes;
    }

    /// <summary>
    /// 临时张量作用域，释放时恢复外层工作区。
    /// </summary>
    /// <param name="owner">当前作用域所属的临时张量工作区。</param>
    private sealed class Scope(TensorWorkspace? owner) : IDisposable
    {
        private bool _disposed;
        /// <summary>
        /// 释放持有的资源或恢复进入作用域之前的状态；调用者不应再使用已释放资源。
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            if (owner is null)
            {
                return;
            }

            foreach (var array in owner._active)
            {
                owner.Keep(array);
            }

            owner._active.Clear();
            _current = null;
        }
    }
}
