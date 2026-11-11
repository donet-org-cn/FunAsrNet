using System.Runtime.Intrinsics;

namespace FunAsrNet;

/// <summary>单 token 的分组查询注意力，两个查询头共享同一组键值的加载。</summary>
public sealed partial class TensorMath
{
    /// <summary>对每个 KV 头同时计算两个查询头，保持原点积、Softmax 和逐帧累加顺序。</summary>
    /// <param name="query">一个 token 的连续查询向量。</param>
    /// <param name="keys">按时间、KV 头、通道排列的只读缓存。</param>
    /// <param name="values">与键布局一致的只读值缓存。</param>
    /// <param name="frames">缓存中的有效时间帧数。</param>
    /// <param name="kvHeads">KV 头数，查询头数必须为其两倍。</param>
    /// <param name="size">每个头的通道数。</param>
    /// <param name="output">一个 token 的独立注意力输出。</param>
    /// <returns>支持并完成计算时为 true，否则输出保持不变。</returns>
    internal bool TryDecoderAttentionPair(float[] query, float[] keys, float[] values, int frames, int kvHeads, int size, float[] output)
    {
        if (!Simd || !Vector256.IsHardwareAccelerated || frames < 1 || kvHeads < 1 || size < 1)
        {
            return false;
        }
        int stride = checked(kvHeads * size);
        if (query.Length != checked(2 * stride) || output.Length != query.Length || keys.Length < checked(frames * stride) || values.Length < checked(frames * stride))
        {
            throw new ArgumentException("成对查询注意力尺寸不一致。");
        }
        float divisor = MathF.Sqrt(size);
        For(kvHeads, head =>
        {
            Span<float> scores = stackalloc float[frames * 2];
            int qo = head * size * 2, ko = head * size, source = 0;
            for (; source + 3 < frames; source += 4)
            {
                AttentionScores2x4Avx(query, qo, keys, ko + source * stride, size, stride, size, scores.Slice(source), frames, divisor);
            }
            for (; source < frames; source++)
            {
                scores[source] = AudioKernels.Dot(query.AsSpan(qo, size), keys.AsSpan(ko + source * stride, size)) / divisor;
                scores[frames + source] = AudioKernels.Dot(query.AsSpan(qo + size, size), keys.AsSpan(ko + source * stride, size)) / divisor;
            }
            Softmax(scores.Slice(0, frames));
            Softmax(scores.Slice(frames, frames));
            int done = AttentionValues2Avx(scores, values, ko, stride, output, qo, size, size, frames);
            if (done < size)
            {
                AttentionValues(scores.Slice(0, frames), values, ko + done, stride, output.AsSpan(qo + done, size - done), true);
                AttentionValues(scores.Slice(frames, frames), values, ko + done, stride, output.AsSpan(qo + size + done, size - done), true);
            }
        });
        return true;
    }
}
