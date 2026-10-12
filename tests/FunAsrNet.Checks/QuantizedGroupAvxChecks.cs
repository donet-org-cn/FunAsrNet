using System.Runtime.Intrinsics.X86;
using FunAsrNet;

namespace FunAsrNet.Checks;

/// <summary>验证共享输入量化的行主序 Q8 多投影与独立标量运算逐位一致。</summary>
internal static class QuantizedGroupAvxChecks
{
    /// <summary>覆盖帧尾、通道尾、极值权重、输出生命周期、尺寸预检和后端回退。</summary>
    /// <param name="check">记录断言及其说明的回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        if (!Avx2.IsSupported)
        {
            return;
        }
        var random = new Random(512031);
        var math = new TensorMath(4);
        foreach (int frames in new[] { 0, 1, 2, 3, 4, 5, 17, 33 })
        {
            foreach (int width in new[] { 32, 96, 1024 })
            {
                var input = Enumerable.Range(0, frames * width).Select(index => index % 5 == 0 ? (index % 2 == 0 ? 1f : -1f) : (float)(random.NextDouble() * 2 - 1)).ToArray();
                var originalInput = input.ToArray();
                var weights = new[] { 0, 1, 3, 17, 32, 65 }.Select(rows => CreateWeight(width, rows, random)).ToArray();
                var originalWeights = weights.Select(w => w.Quantized!.ToArray()).ToArray();
                var originalScales = weights.Select(w => w.Scales!.ToArray()).ToArray();
                var q = new sbyte[input.Length];
                var scales = new float[input.Length / 32];
                QuantizeReference(input, q, scales);
                var grouped = math.LinearMany(input, frames, weights);
                for (int which = 0; which < weights.Length; which++)
                {
                    var w = weights[which];
                    check(grouped[which].Length == frames * w.Rows, "Grouped Q8 output dimensions");
                    var independent = math.Linear(input, frames, w);
                    for (int time = 0; time < frames; time++)
                    {
                        for (int row = 0; row < w.Rows; row++)
                        {
                            float sum = 0;
                            for (int start = 0; start < width; start += 32)
                            {
                                int dot = 0;
                                for (int k = 0; k < 32; k++)
                                {
                                    dot += w.Quantized![row * width + start + k] * q[time * width + start + k];
                                }
                                sum += dot * w.Scales![row * width / 32 + start / 32] * scales[time * width / 32 + start / 32];
                            }
                            int at = time * w.Rows + row;
                            check(BitConverter.SingleToInt32Bits(grouped[which][at]) == BitConverter.SingleToInt32Bits(0f + sum), "Grouped Q8 exact scalar oracle with -128 and signed scales");
                            check(BitConverter.SingleToInt32Bits(grouped[which][at]) == BitConverter.SingleToInt32Bits(independent[at]), "Grouped Q8 exact independent projection parity");
                        }
                    }
                    check(w.Quantized!.SequenceEqual(originalWeights[which]) && w.Scales!.SequenceEqual(originalScales[which]), "Grouped Q8 preserves model weights and scales");
                }
                check(input.SequenceEqual(originalInput), "Grouped Q8 preserves caller input");
                var snapshots = grouped.Select(output => output.ToArray()).ToArray();
                var next = math.LinearMany(input.Select(value => value * -0.75f).ToArray(), frames, weights);
                for (int which = 0; which < weights.Length; which++)
                {
                    check(grouped[which].SequenceEqual(snapshots[which]), "Grouped Q8 returned results survive pooled input reuse");
                    if (grouped[which].Length > 0)
                    {
                        check(!ReferenceEquals(grouped[which], next[which]), "Grouped Q8 calls produce distinct live outputs");
                    }
                }
            }
        }
        var valid = CreateWeight(32, 19, random);
        CheckRejected(math, new float[31], 1, [valid, valid], check, "Grouped Q8 rejects input shape mismatch");
        CheckRejected(math, new float[32], 1, [valid, CreateWeight(64, 3, random)], check, "Grouped Q8 rejects mixed input widths");
        CheckRejected(math, new float[32], 1, [valid, new Weight("truncated-bytes", [32, 3]) { Quantized = new sbyte[95], Scales = new float[3] }], check, "Grouped Q8 rejects truncated bytes before unsafe access");
        CheckRejected(math, new float[32], 1, [valid, new Weight("truncated-scales", [32, 3]) { Quantized = new sbyte[96], Scales = new float[2] }], check, "Grouped Q8 rejects truncated scales before unsafe access");
        var backend = new CountingQuantizedBackend();
        math.FloatBackend = backend;
        var backendOutputs = math.LinearMany(new float[32], 1, [valid, valid]);
        check(backend.Calls == 2 && backendOutputs.All(output => output.All(value => value == 123f)), "Grouped CPU optimization preserves individual backend fallback");
        math.FloatBackend = null;
        var mixed = new Weight("mixed-float", [32, 3]) { Values = Enumerable.Range(0, 96).Select(index => index * 0.002f).ToArray() };
        float[] mixedInput = Enumerable.Range(0, 32).Select(index => index * 0.01f).ToArray();
        var mixedOutputs = math.LinearMany(mixedInput, 1, [valid, mixed]);
        check(mixedOutputs[0].SequenceEqual(math.Linear(mixedInput, 1, valid)) && mixedOutputs[1].SequenceEqual(math.Linear(mixedInput, 1, mixed)), "Grouped Q8 preserves mixed precision fallback");
    }

    /// <summary>建立包含极值和零值的完整行主序 Q8 权重。</summary>
    /// <param name="width">量化块对齐的输入宽度。</param>
    /// <param name="rows">输出行数，可以为零。</param>
    /// <param name="random">可复现的随机源。</param>
    /// <returns>带正、负、零缩放因子的只读测试权重。</returns>
    private static Weight CreateWeight(int width, int rows, Random random) => new("grouped-avx", [width, rows])
    {
        Quantized = Enumerable.Range(0, width * rows).Select(index => index % 5 == 0 ? (sbyte)-128 : (sbyte)random.Next(-127, 128)).ToArray(),
        Scales = Enumerable.Range(0, width * rows / 32).Select(index => (index % 4) switch { 0 => 0f, 1 => -0.007f, 2 => 0.013f, _ => 0.125f }).ToArray()
    };

    /// <summary>独立执行原始逐块最大值与远离零舍入，避免通过被测量化器建立预期结果。</summary>
    /// <param name="input">只读输入数组。</param>
    /// <param name="q">存放量化字节的独立数组。</param>
    /// <param name="scales">存放逐块缩放因子的独立数组。</param>
    private static void QuantizeReference(float[] input, sbyte[] q, float[] scales)
    {
        for (int start = 0; start < input.Length; start += 32)
        {
            float max = 0;
            for (int k = 0; k < 32; k++)
            {
                max = Math.Max(max, Math.Abs(input[start + k]));
            }
            float scale = max / 127f, inverse = scale == 0 ? 0 : 1 / scale;
            scales[start / 32] = scale;
            for (int k = 0; k < 32; k++)
            {
                q[start + k] = (sbyte)Math.Clamp((int)MathF.Round(input[start + k] * inverse, MidpointRounding.AwayFromZero), -127, 127);
            }
        }
    }

    /// <summary>确认不完整尺寸在进入不安全 SIMD 读取前被拒绝。</summary>
    /// <param name="math">执行投影的计算实例。</param>
    /// <param name="input">不满足指定尺寸的输入或用于检查权重的有效输入。</param>
    /// <param name="frames">待检查的有效帧数。</param>
    /// <param name="weights">至少一项尺寸不正确的权重。</param>
    /// <param name="check">记录断言及其说明的回调。</param>
    /// <param name="name">失败时报告的检查名称。</param>
    private static void CheckRejected(TensorMath math, float[] input, int frames, Weight[] weights, Action<bool, string> check, string name)
    {
        bool rejected = false;
        try
        {
            math.LinearMany(input, frames, weights);
        }
        catch (ArgumentException)
        {
            rejected = true;
        }
        check(rejected, name);
    }

    /// <summary>记录单投影后端调用，确认 CPU 组路径不会绕过已配置后端。</summary>
    private sealed class CountingQuantizedBackend : IFloatComputeBackend, IQuantizedComputeBackend
    {
        /// <summary>已完成的单投影后端调用次数。</summary>
        internal int Calls
        {
            get; private set;
        }
        /// <summary>用于诊断的固定后端名称。</summary>
        public string Name => "grouped-fallback-check";
        /// <summary>不处理浮点投影，保留现有 CPU 回退。</summary>
        /// <param name="input">只读输入。</param>
        /// <param name="frames">有效帧数。</param>
        /// <param name="weight">只读权重。</param>
        /// <param name="bias">可选偏置。</param>
        /// <param name="output">独立输出。</param>
        /// <returns>固定返回假。</returns>
        public bool TryLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output) => false;
        /// <summary>完成 Q8 投影并写入可识别的结果。</summary>
        /// <param name="input">只读输入。</param>
        /// <param name="frames">有效帧数。</param>
        /// <param name="weight">只读权重。</param>
        /// <param name="bias">可选偏置。</param>
        /// <param name="output">独立输出。</param>
        /// <returns>完整处理时返回真。</returns>
        public bool TryQuantizedLinear(float[] input, int frames, Weight weight, Weight? bias, float[] output)
        {
            Calls++;
            Array.Fill(output, 123f);
            return true;
        }
        /// <summary>不处理卷积，保留现有 CPU 回退。</summary>
        /// <param name="input">只读输入。</param>
        /// <param name="weight">只读卷积权重。</param>
        /// <param name="bias">可选偏置。</param>
        /// <param name="node">卷积属性。</param>
        /// <param name="output">未处理时为空。</param>
        /// <returns>固定返回假。</returns>
        public bool TryConvolution(Tensor input, Tensor weight, Tensor? bias, OnnxNode node, out Tensor? output)
        {
            output = null;
            return false;
        }
    }
}
