using System.Runtime.Intrinsics;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证成对 GQA 头与原独立查询头计算严格一致。</summary>
internal static class DecoderAttentionChecks
{
    /// <summary>覆盖时间尾部、通道尾部、不同 KV 头数和未使用的缓存容量。</summary>
    /// <param name="check">接收断言和说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        if (!Vector256.IsHardwareAccelerated)
        {
            return;
        }
        var random = new Random(7401);
        foreach (int frames in new[] { 1, 3, 4, 17, 73, 129 })
        {
            foreach (int size in new[] { 7, 32, 65, 128 })
            {
                foreach (int heads in new[] { 1, 3, 8 })
                {
                    int stride = heads * size;
                    var q = Enumerable.Range(0, stride * 2).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                    var k = Enumerable.Range(0, stride * (frames + 3)).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                    var v = Enumerable.Range(0, k.Length).Select(_ => (float)(random.NextDouble() - 0.5)).ToArray();
                    var expected = new float[q.Length];
                    for (int head = 0; head < heads * 2; head++)
                    {
                        var scores = new float[frames];
                        for (int time = 0; time < frames; time++)
                        {
                            scores[time] = AudioKernels.Dot(q.AsSpan(head * size, size), k.AsSpan(time * stride + head / 2 * size, size)) / MathF.Sqrt(size);
                        }
                        TensorMath.Softmax(scores);
                        TensorMath.AttentionValues(scores, v, head / 2 * size, stride, expected.AsSpan(head * size, size), true);
                    }
                    var actual = new float[q.Length];
                    check(new TensorMath(4).TryDecoderAttentionPair(q, k, v, frames, heads, size, actual), "Paired GQA uses supported AVX path");
                    for (int i = 0; i < actual.Length; i++)
                    {
                        check(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]), "Paired GQA exact independent-head result");
                    }
                    Array.Fill(actual, 123f);
                    check(!new TensorMath(1, false).TryDecoderAttentionPair(q, k, v, frames, heads, size, actual) && actual.All(value => value == 123f), "GQA fallback leaves output untouched");
                }
            }
        }
    }
}
