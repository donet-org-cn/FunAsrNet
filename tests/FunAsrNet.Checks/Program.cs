using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using FunAsrNet;

int checks = 0;
// <summary>
// 断言测试条件成立，否则抛出包含检查说明的异常。
// </summary>
// <param name="condition">断言条件是否成立。</param>
// <param name="name">失败时报告的检查项目名称。</param>
void Check(bool condition, string name)
{
    ++checks;
    if (!condition)
    {
        throw new Exception(name);
    }
}

FunAsrNet.Checks.QuantizedAvxChecks.Run(Check);
FunAsrNet.Checks.ActivationChecks.Run(Check);
FunAsrNet.Checks.CpuTensorChecks.Run(Check);
FunAsrNet.Checks.CpuFollowupChecks.Run(Check);

// <summary>
// 按绝对误差阈值比较数值结果，用于验证 SIMD 与参考实现一致性。
// </summary>
// <param name="a">第一个输入操作数。</param>
// <param name="b">第二个输入操作数。</param>
// <param name="tolerance">相对绝对混合容差，阈值为 tolerance 乘以 max(1, abs(a))。</param>
// <param name="name">被比较结果的检查名称。</param>
void Close(float a, float b, float tolerance, string name) => Check(Math.Abs(a - b) <= tolerance * Math.Max(1, Math.Abs(b)), $"{name}: {a} vs {b}");
string data = Path.Combine(AppContext.BaseDirectory, "testdata");
foreach (bool alaw in new[]
    {
        true,
        false
    }
)
{
    string codec = alaw ? "alaw" : "ulaw";
    var input = Enumerable.Range(0, 65536).Select(n => unchecked((short)n)).ToArray();
    var encoded = new byte[input.Length];
    G711.Encode(input, encoded, alaw);
    Check(encoded.SequenceEqual(File.ReadAllBytes(Path.Combine(data, $"g711-encode-{codec}.bin"))), $"Exhaustive {codec} encoder vs CPython audioop");
    var source = Enumerable.Range(0, 256).Select(n => (byte)n).ToArray();
    var decoded = new short[256];
    var golden = MemoryMarshal.Cast<byte, short>(File.ReadAllBytes(Path.Combine(data, $"g711-decode-{codec}.bin"))).ToArray();
    G711.Decode(source, decoded, alaw);
    Check(decoded.SequenceEqual(golden), $"Exhaustive SIMD {codec} decoder");
    for (int length = 0; length <= 33; ++length)
    {
        G711.Decode(source.AsSpan(1, length), decoded, alaw);
        Check(decoded.AsSpan(0, length).SequenceEqual(golden.AsSpan(1, length)), $"{codec} unaligned tail {length}");
    }
}

var tiled = TensorOps.Tile(new Tensor([1, 2, 3, 4], [2, 2], true), [2, 3]);
Check(tiled.Shape.SequenceEqual(new[] { 4, 6 }) && tiled.Integer, "Tile shape and integer semantics");
Check(tiled.Data.SequenceEqual(new float[]
{
    1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4,
    1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4
}), "Tile repeats blocks along each axis");
Check(TensorOps.Tile(new Tensor([7], []), []).Data.SequenceEqual(new float[] { 7 }), "Tile scalar");
Check(TensorOps.Tile(new Tensor([1, 2], [2]), [0]).Count == 0, "Tile zero repeat");
var rng = new Random(713);
foreach (int length in new[]
    {
        1,
        7,
        8,
        31,
        32,
        160,
        319,
        320,
        1024
    }
)
{
    var pcm = Enumerable.Range(0, length).Select(_ => (short)rng.Next(short.MinValue, short.MaxValue + 1)).ToArray();
    var a = new float[length];
    var b = new float[length];
    AudioKernels.Pcm16ToFloat(pcm, a);
    AudioKernels.Pcm16ToFloat(pcm, b, false);
    Check(a.SequenceEqual(b), "PCM SIMD parity");
    var p = new short[length];
    var q = new short[length];
    AudioKernels.FloatToPcm16(a, p);
    AudioKernels.FloatToPcm16(b, q, false);
    Check(p.SequenceEqual(q), "PCM output SIMD parity");
    Close(AudioKernels.Dot(a, b), AudioKernels.Dot(a, b, false), 1e-5f, "Dot SIMD parity");
}

foreach (int width in new[]
    {
        32,
        96,
        1024
    }
)
{
    float[] w = Enumerable.Range(0, width * 19).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    float[] x = Enumerable.Range(0, width * 7).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    var weight = new Weight("test", [width, 19])
    {
        Values = w
    };
    var actual = new TensorMath(4).Linear(x, 7, weight);
    for (int t = 0; t < 7; ++t)
    {
        for (int row = 0; row < 19; ++row)
        {
            double expected = 0;
            for (int j = 0; j < width; ++j)
            {
                expected += (double)x[t * width + j] * w[row * width + j];
            }

            Close(actual[t * 19 + row], (float)expected, 2e-5f, "F32 GEMM vs double precision");
        }
    }

    var qw = new sbyte[w.Length];
    var sw = new float[w.Length / 32];
    TensorMath.Quantize(w, qw, sw);
    var qx = new sbyte[x.Length];
    var sx = new float[x.Length / 32];
    TensorMath.Quantize(x, qx, sx);
    for (int t = 0; t < 7; ++t)
    {
        for (int row = 0; row < 19; ++row)
        {
            double expected = 0;
            for (int j = 0; j < width; ++j)
            {
                expected += (double)qw[row * width + j] * qx[t * width + j] * sw[(row * width + j) / 32] * sx[(t * width + j) / 32];
            }

            Close(TensorMath.QuantizedDot(qw, sw, row * width, qx, sx, t * width, width), (float)expected, 1e-5f, "Q8 SDOT vs dequantized double");
        }
    }
}

// 验证打包矩阵小块、两个维度的尾块、奇数内积长度及偏置。
foreach (int frames in new[]
    {
        1,
        2,
        3,
        16,
        17,
        31,
        32,
        33
    }
)
{
    foreach (int width in new[]
        {
            31,
            32,
            35,
            96,
            128,
            131,
            512
        }
    )
    {
        foreach (int rows in new[]
            {
                3,
                4,
                19,
                33,
                129
            }
        )
        {
            float[] w = Enumerable.Range(0, width * rows).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            float[] x = Enumerable.Range(0, width * frames).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            float[] b = Enumerable.Range(0, rows).Select(_ => (float)rng.NextDouble()).ToArray();
            var weight = new Weight("tile-test", [width, rows])
            {
                Values = w
            };
            var bias = new Weight("bias", [rows])
            {
                Values = b
            };
            var optimized = new TensorMath(4).Linear(x, frames, weight, bias);
            float maxError = 0;
            for (int t = 0; t < frames; ++t)
            {
                for (int r = 0; r < rows; ++r)
                {
                    double expected = b[r];
                    for (int k = 0; k < width; ++k)
                    {
                        expected += (double)w[r * width + k] * x[t * width + k];
                    }

                    maxError = Math.Max(maxError, Math.Abs(optimized[t * rows + r] - (float)expected));
                }
            }

            Check(maxError < 3e-5f, $"Packed F32 {frames}x{width}x{rows}: {maxError}");
            weight.PrepareInterleaved();
            for (int r = 0; r < rows; ++r)
            {
                Check(weight.Row(r).AsSpan().SequenceEqual(w.AsSpan(r * width, width)), "Interleaved F32 preserves every weight");
            }

            var interleavedFloat = new TensorMath(4).Linear(x, frames, weight, bias);
            Check(interleavedFloat.Zip(optimized).All(p => Math.Abs(p.First - p.Second) < 8e-5f), $"Interleaved F32 {frames}x{width}x{rows}");
            if (width % 32 != 0)
            {
                continue;
            }

            var qw = new sbyte[w.Length];
            var sw = new float[w.Length / 32];
            TensorMath.Quantize(w, qw, sw);
            var quantized = new Weight("q8-tile-test", [width, rows])
            {
                Quantized = qw,
                Scales = sw

            };
            var got = new TensorMath(4).Linear(x, frames, quantized, bias);
            var reference = new TensorMath(1)
            {
                Optimized = false
            }.Linear(x, frames, quantized, bias);
            Check(got.Zip(reference).All(p => Math.Abs(p.First - p.Second) < 3e-5f), $"Packed Q8 {frames}x{width}x{rows}");
            quantized.PrepareInterleaved();
            var interleavedQuantized = new TensorMath(4).Linear(x, frames, quantized, bias);
            Check(interleavedQuantized.Zip(reference).All(p => Math.Abs(p.First - p.Second) < 8e-5f), $"Interleaved Q8 {frames}x{width}x{rows}");
            for (int r = 0; r < rows; ++r)
            {
                Check(quantized.Row(r).Select((v, k) => v == qw[r * width + k] * sw[(r * width + k) / 32]).All(v => v), "Interleaved Q8 preserves every weight");
            }
        }
    }
}

foreach (float amplitude in new[]
    {
        0f,
        1f,
        1000f,
        1e-30f,
        1e-40f
    }
)
{
    float[] input = Enumerable.Range(0, 64).Select(i => (i % 32 - 16) * amplitude).ToArray();
    sbyte[] q = new sbyte[64];
    float[] scales = new float[2];
    TensorMath.Quantize(input, q, scales);
    for (int b = 0; b < 2; ++b)
    {
        float max = input.AsSpan(b * 32, 32).ToArray().Max(v => Math.Abs(v)), d = max / 127f, inv = d == 0 ? 0 : 1 / d;
        Check(scales[b] == d, "Quantization scales match scalar");
        for (int i = 0; i < 32; ++i)
        {
            Check(
                q[b * 32 + i] == (sbyte)Math.Clamp((int)MathF.Round(input[b * 32 + i] * inv, MidpointRounding.AwayFromZero), -127, 127),
                "Quantization rounding matches scalar");
        }
    }
}

foreach (int width in new[]
    {
        1,
        3,
        4,
        7,
        8,
        17,
        128,
        512,
        1024
    }
)
{
    float[] x = Enumerable.Range(0, width * 3).Select(_ => (float)(rng.NextDouble() * 20 - 10)).ToArray();
    float[] scale = Enumerable.Range(0, width).Select(_ => (float)rng.NextDouble()).ToArray();
    float[] bias = Enumerable.Range(0, width).Select(_ => (float)rng.NextDouble()).ToArray();
    foreach (bool rms in new[]
        {
            false,
            true
        }
    )
    {
        var got = TensorMath.Norm(x, width, scale, rms ? null : bias, 1e-5f, rms);
        for (int row = 0; row < 3; ++row)
        {
            double mean = rms ? 0 : x.AsSpan(row * width, width).ToArray().Average(v => (double)v);
            double variance = x.AsSpan(row * width, width).ToArray().Average(v => Math.Pow(v - mean, 2));
            for (int j = 0; j < width; ++j)
            {
                Close(
                    got[row * width + j],
                    (float)((x[row * width + j] - mean) / Math.Sqrt(variance + 1e-5) * scale[j] + (rms ? 0 : bias[j])),
                    1e-5f,
                    "SIMD norm vs double");
            }
        }
    }

    foreach (float amplitude in new[]
        {
            0f,
            1f,
            50f
        }
    )
    {
        float[] scores = x.AsSpan(0, width).ToArray().Select(v => v * amplitude).ToArray();
        double max = scores.Max(), denominator = scores.Sum(v => Math.Exp(v - max));
        var got = (float[])scores.Clone();
        TensorMath.Softmax(got);
        for (int j = 0; j < width; ++j)
        {
            Close(got[j], (float)(Math.Exp(scores[j] - max) / denominator), 5e-7f, "SIMD softmax vs double");
        }

        Close(got.Sum(), 1, 1e-6f, "Softmax normalized");
    }

    var gate = (float[])x.Clone();
    var up = x.Select(v => v * .1f).ToArray();
    TensorMath.SiluMultiply(gate, up);
    for (int i = 0; i < x.Length; ++i)
    {
        Close(gate[i], (float)(x[i] / (1 + Math.Exp(-x[i])) * up[i]), 2e-6f, "SIMD SwiGLU vs double");
    }
}

// 穷举全部半精度编码，并测试每个舍入中点的两侧。
var halfInputs = new List<float>();
for (int i = 0; i <= ushort.MaxValue; ++i)
{
    float value = (float)BitConverter.UInt16BitsToHalf((ushort)i);
    halfInputs.Add(value);
    if (i < 0x7bff)
    {
        float next = (float)BitConverter.UInt16BitsToHalf((ushort)(i + 1));
        float middle = (value + next) * .5f;
        halfInputs.Add(middle);
        halfInputs.Add(float.BitIncrement(middle));
        halfInputs.Add(float.BitDecrement(middle));
        halfInputs.Add(-middle);
        halfInputs.Add(-float.BitIncrement(middle));
        halfInputs.Add(-float.BitDecrement(middle));
    }
}

for (int i = 0; i < 10000; ++i)
{
    halfInputs.Add(BitConverter.Int32BitsToSingle(rng.Next(int.MinValue, int.MaxValue)));
}

var halfSource = halfInputs.ToArray();
var halfOutput = new float[halfSource.Length];
TensorMath.RoundHalfCopy(halfSource, halfOutput);
for (int i = 0; i < halfSource.Length; ++i)
{
    Check(
        BitConverter.SingleToInt32Bits(halfOutput[i]) == BitConverter.SingleToInt32Bits((float)(Half)halfSource[i]),
        "SIMD Half rounding exact bits");
}

foreach (int count in new[]
    {
        1,
        17,
        100
    }
)
{
    foreach (int channels in new[]
        {
            1,
            15,
            16,
            33,
            128
        }
    )
    {
        int stride = channels * 4;
        var values = Enumerable.Range(0, count * stride).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
        var scores = Enumerable.Range(0, count).Select(_ => (float)rng.NextDouble()).ToArray();
        var expected = new float[channels];
        var actual = new float[channels];
        for (int t = 0; t < count; ++t)
        {
            TensorMath.AddScaled(expected, values.AsSpan(t * stride + channels, channels), scores[t]);
        }

        TensorMath.AttentionValues(scores, values, channels, stride, actual, true);
        Check(actual.SequenceEqual(expected), "Register attention exact parity");
    }
}

foreach (int frames in new[]
    {
        16,
        17,
        33,
        100
    }
)
{
    foreach (int headSize in new[]
        {
            7,
            32,
            128
        }
    )
    {
        const int heads = 4;
        int width = headSize * heads;
        var q = Enumerable.Range(0, frames * width).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
        var k = Enumerable.Range(0, q.Length).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
        var v = Enumerable.Range(0, q.Length).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
        var math = new TensorMath(4);
        Check(
            math.Attention(q, k, v, frames, width, heads).SequenceEqual(math.AttentionReference(q, k, v, frames, width, heads)),
            "Blocked attention exact parity");
    }
}

// 分组合并投影必须与独立计算一致，并覆盖尾块。
foreach (int frames in new[]
    {
        1,
        3,
        4,
        9,
        18
    }
)
{
    const int width = 128;
    var weights = new[]
    {
        19,
        128,
        257
    }.Select(rows =>
        {
            var values = Enumerable.Range(0, width * rows).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
            var w = new Weight("grouped-test", [width, rows])
            {
                Quantized = new sbyte[values.Length],
                Scales = new float[values.Length / 32]

            };
            TensorMath.Quantize(values, w.Quantized, w.Scales);
            w.PrepareInterleaved();
            return w;
        }).ToArray();
    var input = Enumerable.Range(0, width * frames).Select(_ => (float)(rng.NextDouble() - .5)).ToArray();
    var math = new TensorMath(4);
    var grouped = math.LinearMany(input, frames, weights);
    for (int i = 0; i < weights.Length; ++i)
    {
        Check(grouped[i].SequenceEqual(math.Linear(input, frames, weights[i])), "Grouped Q8 exact parity");
    }
}

var workspaceTest = new TensorWorkspace(4096);
float[] pooled;
using (workspaceTest.Enter())
{
    pooled = TensorWorkspace.Allocate(128);
    Array.Fill(pooled, 99f);
    TensorWorkspace.Release(pooled);
    var reused = TensorWorkspace.Allocate(128, clear: true);
    Check(ReferenceEquals(pooled, reused) && reused.All(v => v == 0), "Workspace reuses and clears");
    var other = TensorWorkspace.Allocate(128);
    Check(!ReferenceEquals(reused, other), "Live tensor buffers never alias");
    TensorWorkspace.Release(new float[128]);
    for (int i = 1; i < 200; ++i)
    {
        var buffer = TensorWorkspace.Allocate(i);
        TensorWorkspace.Release(buffer);
    }
}

Check(workspaceTest.RetainedBytes <= 4096, "Workspace has bounded retention across varying shapes");
using (workspaceTest.Enter(false))
{
    Check(!ReferenceEquals(pooled, TensorWorkspace.Allocate(128)), "Trace bypasses workspace");
}

if (args.Contains("--bench"))
{
    foreach (var (frames, width, rows, quantized) in new[]
        {
            (100, 512, 512, false),
            (100, 512, 2048, false),
            (100, 2048, 512, false),
            (1200, 32, 32, false),
            (37, 1024, 3072, true),
            (1, 1024, 3072, true),
            (1, 1024, 151936, true)
        }
    )
    {
        var w = new Weight("bench", [width, rows]);
        float[] values = new float[width * rows];
        for (int i = 0; i < values.Length; ++i)
        {
            values[i] = (float)(rng.NextDouble() - .5);
        }

        if (quantized)
        {
            w.Quantized = new sbyte[values.Length];
            w.Scales = new float[values.Length / 32];
            TensorMath.Quantize(values, w.Quantized, w.Scales);
        }
        else
        {
            w.Values = values;
        }

        var x = new float[width * frames];
        for (int i = 0; i < x.Length; ++i)
        {
            x[i] = (float)(rng.NextDouble() - .5);
        }

        var interleaved = new Weight("bench-interleaved", [width, rows])
        {
            Values = w.Values,
            Quantized = w.Quantized,
            Scales = w.Scales

        };
        interleaved.PrepareInterleaved();
        foreach (string kernel in new[]
            {
                "v2",
                "v3",
                "v2",
                "v3"
            }
        )
        {
            var math = new TensorMath(4);
            var selected = kernel == "v3" ? interleaved : w;
            for (int i = 0; i < 5; ++i)
            {
                math.Linear(x, frames, selected);
            }

            var timer2 = Stopwatch.StartNew();
            float checksum = 0;
            const int repeats = 15;
            for (int i = 0; i < repeats; ++i)
            {
                checksum += math.Linear(x, frames, selected)[i % rows];
            }

            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new
            {
                frames,
                width,
                rows,
                quantized,
                kernel,
                ms = timer2.Elapsed.TotalMilliseconds / repeats,
                checksum

            }));
        }
    }
}

var team = CpuKernelWorkers.Get(4);
Parallel.For(
    0,
    4,
    caller =>
    {
        for (int run = 0; run < 100; ++run)
        {
            int total = 0;
            team.For(37, i => team.For(3, j => Interlocked.Add(ref total, i + j)));
            if (total != 2109)
            {
                throw new Exception("Worker team publication or nesting failure");
            }
        }
    });
bool propagated = false;
try
{
    team.For(19, i =>
        {
            if (i == 1)
            {
                throw new InvalidOperationException();
            }
        });
}
catch (AggregateException)
{
    propagated = true;
}

Check(propagated, "Worker team exception propagation");
int afterFailure = 0;
team.For(17, _ => Interlocked.Increment(ref afterFailure));
Check(afterFailure == 17, "Worker team recovery");
// <summary>
// 转换采样率；目标相同时返回原数组，否则一次分配最终输出数组。
// </summary>
// <param name="input">当前操作的输入数据，在操作完成前不得由其他任务修改。</param>
// <param name="from">输入采样率，单位为 Hz。</param>
// <param name="to">目标采样率，单位为 Hz。</param>
// <param name="chunk">测试中每批输入的采样数量。</param>
// <returns>目标采样率的浮点数组；采样率相同时返回原数组。</returns>
float[] Resample(float[] input, int from, int to, int chunk)
{
    var r = new PolyphaseResampler(from, to);
    var output = new List<float>();
    var buffer = new float[r.GetMaxOutput(chunk)];
    for (int i = 0; i < input.Length; i += chunk)
    {
        int n = Math.Min(chunk, input.Length - i), m = r.Process(input.AsSpan(i, n), buffer, i + n == input.Length);
        output.AddRange(buffer.AsSpan(0, m));
    }

    return output.ToArray();
}

foreach (var (from, to) in new[]
    {
        (8000, 16000),
        (22050, 8000),
        (48000, 16000)
    }
)
{
    var input = Enumerable.Range(0, from * 2 + 17).Select(i => MathF.Sin(2 * MathF.PI * 1000 * i / from)).ToArray();
    var a = Resample(input, from, to, 160);
    var b = Resample(input, from, to, 4096);
    Check(a.SequenceEqual(b), "Chunk independent resampling across ring wraps");
    Check(a.Length == (int)Math.Ceiling((double)input.Length * to / from), "Resampling output length");
    double error = a.Skip(100).Take(a.Length - 200).Select((v, i) => Math.Pow(v - Math.Sin(2 * Math.PI * 1000 * (i + 100) / to), 2)).Average();
    Check(error < 1e-5, $"Resampling passband {error}");
}

var aliasInput = Enumerable.Range(0, 48000).Select(i => MathF.Sin(2 * MathF.PI * 12000 * i / 48000)).ToArray();
var alias = Resample(aliasInput, 48000, 16000, 960);
Check(alias.Skip(100).Take(alias.Length - 200).Select(v => v * v).Average() < 1e-6, "Anti-alias stopband rejection");
var originalPcm = Enumerable.Range(0, 5001).Select(i => (short)(i * 3 - 7000)).ToArray();
var pcmBytes = MemoryMarshal.AsBytes(originalPcm.AsSpan()).ToArray();
var decodedFrames = new List<float>();
var decoder = new PhoneAudioDecoder(AudioCodec.Pcm16, 16000);
for (int i = 0; i < pcmBytes.Length; i += 7)
{
    decodedFrames.AddRange(decoder.Decode(pcmBytes.AsSpan(i, Math.Min(7, pcmBytes.Length - i))));
}

decodedFrames.AddRange(decoder.Decode([], true));
Check(decodedFrames.SequenceEqual(originalPcm.Select(x => x / 32768f)), "PCM odd byte boundaries preserve every sample");
bool rejectedTruncated = false;
try
{
    new PhoneAudioDecoder(AudioCodec.Pcm16, 16000).Decode([1], true);
}
catch (InvalidDataException)
{
    rejectedTruncated = true;
}

Check(rejectedTruncated, "Reject incomplete PCM16 at EOF");
foreach (int rate in new[]
    {
        22050,
        44100,
        48000,
        96000
    }
)
{
    var browser = new PhoneAudioDecoder(AudioCodec.Pcm16, rate);
    var actual = new List<float>();
    for (int i = 0; i < pcmBytes.Length; i += 713)
    {
        actual.AddRange(browser.Decode(pcmBytes.AsSpan(i, Math.Min(713, pcmBytes.Length - i))));
    }

    actual.AddRange(browser.Decode([], true));
    Check(
        actual.SequenceEqual(AudioFile.Resample(originalPcm.Select(x => x / 32768f).ToArray(), rate, 16000)),
        $"Browser PCM {rate} Hz resampling with odd byte splits");
}

// <summary>
// 按测试指定格式构造 WAV 内容，用于检查解码与输入验证边界。
// </summary>
// <param name="format">WAV 格式标识或目标音频编码名称。</param>
// <param name="bits">每个声道采样的位数。</param>
// <param name="channels">交错排列的输入声道数量。</param>
// <param name="frames">输入或输出的时间帧数，具体布局由所在算子定义。</param>
// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
byte[] MakeWave(short format = 1, short bits = 16, short channels = 1, int frames = 1600)
{
    using var memory = new MemoryStream();
    using var writer = new BinaryWriter(memory);
    int size = frames * channels * bits / 8;
    writer.Write(0x46464952);
    writer.Write(36 + size);
    writer.Write(0x45564157);
    writer.Write(0x20746d66);
    writer.Write(16);
    writer.Write(format);
    writer.Write(channels);
    writer.Write(16000);
    writer.Write(16000 * channels * bits / 8);
    writer.Write((short)(channels * bits / 8));
    writer.Write(bits);
    writer.Write(0x61746164);
    writer.Write(size);
    writer.Write(new byte[size]);
    return memory.ToArray();
}

foreach (var (format, bits) in new[]
    {
        (1, 16),
        (3, 32),
        (6, 8),
        (7, 8)
    }
)
{
    using var wave = new MemoryStream(MakeWave((short)format, (short)bits, 2));
    var read = AudioFile.ReadWave(wave, 60);
    Check(read.Rate == 16000 && read.Samples.Length == 1600 && wave.CanRead, $"WAV format {format}, stereo mix, caller owns stream");
}

// <summary>
// 断言不合法的 WAV 会被读取器拒绝。
// </summary>
// <param name="bytes">当前调用拥有或借用的字节数据。</param>
// <param name="name">待访问的命名权重、属性或跟踪项目。</param>
// <param name="max">允许的音频时长上限，单位为秒。</param>
void BadWave(byte[] bytes, string name, double max = 60)
{
    bool rejected = false;
    try
    {
        using var wave = new MemoryStream(bytes);
        AudioFile.ReadWave(wave, max);
    }
    catch (Exception e) when (e is InvalidDataException or NotSupportedException or OverflowException or EndOfStreamException)
    {
        rejected = true;
    }

    Check(rejected, name);
}

BadWave([], "Reject empty WAV");
BadWave(MakeWave()[..^1], "Reject truncated WAV");
var shortFmt = MakeWave();
BitConverter.GetBytes(4).CopyTo(shortFmt, 16);
BadWave(shortFmt, "Reject undersized fmt chunk");
var hugeChunk = MakeWave();
BitConverter.GetBytes(uint.MaxValue).CopyTo(hugeChunk, 40);
BadWave(hugeChunk, "Reject overflowing chunk without allocation");
var misaligned = MakeWave();
BitConverter.GetBytes((short)3).CopyTo(misaligned, 32);
BadWave(misaligned, "Reject invalid block align");
var nanWave = MakeWave(3, 32);
BitConverter.GetBytes(float.NaN).CopyTo(nanWave, 44);
BadWave(nanWave, "Reject NaN float audio");
BadWave(MakeWave(), "Reject excessive duration before allocating samples", .05);
BadWave(MakeWave(frames: 0), "Reject zero-length data");
foreach (var (format, bits) in new[]
    {
        (1, 16),
        (3, 32),
        (6, 8),
        (7, 8)
    }
)
{
    byte[] bytes = MakeWave((short)format, (short)bits, 2, 18003);
    using var fragmented = new FragmentedReadStream(bytes);
    var streamed = await AudioFile.ReadWaveAsync(fragmented);
    using var complete = new MemoryStream(bytes);
    var expected = AudioFile.ReadWave(complete);
    Check(streamed.Rate == expected.Rate && streamed.Samples.SequenceEqual(expected.Samples), $"Non-seekable fragmented WAV format {format}");
    Check(fragmented.CanRead, "Async WAV leaves caller stream open");
}

foreach (byte[] malformed in new[]
    {
        Array.Empty<byte>(),
        MakeWave()[..^1],
        shortFmt,
        hugeChunk,
        misaligned,
        nanWave,
        MakeWave(frames: 0)
    }
)
{
    bool rejected = false;
    try
    {
        using var stream = new FragmentedReadStream(malformed);
        await AudioFile.ReadWaveAsync(stream);
    }
    catch (Exception exception) when (exception is InvalidDataException or EndOfStreamException or NotSupportedException or OverflowException)
    {
        rejected = true;
    }

    Check(rejected, "Streaming WAV rejects malformed file");
}

var reusableDecoder = new PhoneAudioDecoder(AudioCodec.Pcm16, 16000);
var audioFrame = new byte[640];
var decodedFrame = new float[320];
for (int i = 0; i < 100; i++)
{
    reusableDecoder.Decode(audioFrame, decodedFrame);
}

long beforeDecode = GC.GetAllocatedBytesForCurrentThread();
for (int i = 0; i < 10000; i++)
{
    reusableDecoder.Decode(audioFrame, decodedFrame);
}

long allocatedDecode = GC.GetAllocatedBytesForCurrentThread() - beforeDecode;
Check(allocatedDecode == 0, $"Caller-buffer PCM decoder: {allocatedDecode} allocated bytes in 10000 frames");
await ConcurrencyChecks.RunAsync(Check);
await KernelWorkerChecks.RunAsync(Check);
OnnxBufferChecks.Run(Check);
TensorOptimizationChecks.Run(Check);
var matrix = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
Check(TensorOps.Transpose(matrix).Data.SequenceEqual(new float[] { 1, 4, 2, 5, 3, 6 }), "Transpose known matrix");
Check(TensorOps.Gather(matrix, new Tensor([-1, 0], [2], true), 1).Data.SequenceEqual(new float[] { 3, 1, 6, 4 }), "Gather negative index");
Check(
    TensorOps.Slice(matrix, new Tensor([-1], [1]), new Tensor([int.MinValue], [1]), [1], [-1]).Data.SequenceEqual(new float[] { 3, 2, 1, 6, 5, 4 }),
    "Reverse slice with sentinel");
Check(
    TensorOps.Binary(matrix, new Tensor([10, 20], [2, 1]), (a, b) => a + b).Data.SequenceEqual(new float[] { 11, 12, 13, 24, 25, 26 }),
    "Broadcast rows");
Check(TensorOps.GatherND(matrix, new Tensor([0, 2, 1, 0], [2, 2], true)).Data.SequenceEqual(new float[] { 3, 4 }), "GatherND tuples");
Check(
    TensorOps.ScatterND(matrix, new Tensor([0, 2, 1, 0], [2, 2], true), new Tensor([7, 8], [2])).Data.SequenceEqual(new float[] { 1, 2, 7, 8, 5, 6 }),
    "ScatterND tuples");
var mm = TensorOps.MatMul(matrix, new Tensor([1, 2, 3, 4, 5, 6], [3, 2]), new TensorMath(1));
Check(mm.Data.SequenceEqual(new float[] { 22, 28, 49, 64 }), "ONNX MatMul known result");
var sharedMatrix = new Tensor([1, 2, 3, 4, 5, 6], [3, 2]);
var batchesMatrix = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]);
var sharedProduct = TensorOps.MatMul(batchesMatrix, sharedMatrix, new TensorMath(2));
Check(sharedProduct.Shape.SequenceEqual(new[] { 2, 2, 2 }), "Shared MatMul batch shape");
Check(sharedProduct.Data.SequenceEqual(new float[] { 22, 28, 49, 64, 76, 100, 103, 136 }), "Shared MatMul flattens batches without repeating weights");
Check(TensorOps.MatMul(new Tensor([], [0, 2, 3]), sharedMatrix, new TensorMath(1)).Count == 0, "Empty shared MatMul batch");
Check(TensorOps.MatMul(new Tensor([1, 2, 3], [3]), sharedMatrix, new TensorMath(1)).Shape.SequenceEqual(new[] { 2 }), "MatMul removes promoted vector dimension");
Check(TensorOps.MatMul(matrix, new Tensor([1, 2, 3], [3]), new TensorMath(1)).Data.SequenceEqual(new float[] { 14, 32 }), "MatMul right vector broadcast");
foreach (string operation in new[] { "Add", "Sub", "Mul", "Div" })
{
    Func<float, float, float> reference = operation switch
    {
        "Add" => (left, right) => left + right,
        "Sub" => (left, right) => left - right,
        "Mul" => (left, right) => left * right,
        _ => (left, right) => left / right
    };
    var full = new Tensor(Enumerable.Range(1, 30).Select(value => value / 7f).ToArray(), [2, 3, 5]);
    foreach (var other in new[] { full, new Tensor([0.75f], []), new Tensor([1.5f, 2.5f, 3.5f], [1, 3, 1]), new Tensor([1, 2, 3, 4, 5], [1, 1, 5]), new Tensor(Enumerable.Range(1, 10).Select(value => (float)value).ToArray(), [2, 1, 5]) })
    {
        foreach (bool reversed in new[] { false, true })
        {
            var left = reversed ? other : full;
            var right = reversed ? full : other;
            var actual = TensorOps.Arithmetic(left, right, operation);
            var expected = TensorOps.Binary(left, right, reference);
            Check(actual.Shape.SequenceEqual(expected.Shape), $"Arithmetic broadcast shape {operation}");
            Check(actual.Data.SequenceEqual(expected.Data), $"Arithmetic SIMD, channel broadcast and tails {operation}");
        }
    }
}

Check(TensorOps.Arithmetic(new Tensor([-7, 7], [2], true), new Tensor([2], [], true), "Div").Data.SequenceEqual(new float[] { -3, 3 }), "Integer division truncates toward zero");
float[] leakyInput = [-3, -0f, 0, 2, float.NaN, float.NegativeInfinity, float.PositiveInfinity];
var leaky = TensorOps.LeakyRelu(new Tensor(leakyInput, [7]), 0.1f);
for (int index = 0; index < leakyInput.Length; index++)
{
    float value = leakyInput[index], expected = value >= 0 ? value : value * 0.1f;
    Check(float.IsNaN(expected) ? float.IsNaN(leaky.Data[index]) : BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(leaky.Data[index]), "LeakyRelu signed zero, nonfinite and SIMD tail");
}

foreach (int length in new[] { 257, 519, 4097 })
{
    foreach (int kernel in new[] { 1, 3, 4, 7 })
    {
        foreach (int dilation in new[] { 1, 3 })
        {
            int channels = 3, padLeft = 2, padRight = 4;
            var input = new Tensor(Enumerable.Range(0, 2 * channels * length).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray(), [2, channels, length]);
            var weights = new Tensor(Enumerable.Range(0, channels * kernel).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray(), [1, channels, kernel]);
            var node = new OnnxNode("single-output-test", "Conv", [], [], new Dictionary<string, object> { ["pads"] = new long[] { padLeft, padRight }, ["dilations"] = new long[] { dilation } });
            var actual = TensorOps.Conv1D(input, weights, new Tensor([0.2f], [1]), node, new TensorMath(2));
            for (int batch = 0; batch < 2; batch++)
            {
                for (int time = 0; time < actual.Shape[2]; time++)
                {
                    double sum = 0.2;
                    for (int channel = 0; channel < channels; channel++)
                    {
                        for (int tap = 0; tap < kernel; tap++)
                        {
                            int at = time - padLeft + tap * dilation;
                            if ((uint)at < (uint)length)
                            {
                                sum += (double)input.Data[(batch * channels + channel) * length + at] * weights.Data[channel * kernel + tap];
                            }
                        }
                    }

                    Close(actual.Data[batch * actual.Shape[2] + time], (float)sum, 1e-5f, "Single-output convolution independent oracle: padding, dilation, tile and tail");
                }
            }
        }
    }
}

var endpoint = new PhoneEndpoint(600);
for (int i = 0; i < 100; ++i)
{
    Check(endpoint.Step(1) is null, "Silence emits no speech");
}

for (int i = 0; i < 100; ++i)
{
    endpoint.Step(0);
}

SpeechSegment? ended = null;
for (int i = 0; i < 80; ++i)
{
    ended ??= endpoint.Step(1);
}

Check(ended.HasValue && ended.Value.StartSample >= 0 && ended.Value.EndSample > ended.Value.StartSample, "Phone endpoint commits speech");
Console.WriteLine($"PASS {checks} checks; ARM NEON={AdvSimd.IsSupported}, SDOT={Dp.IsSupported}; runtime={Environment.Version}");
// 微基准复用相同缓冲，测量循环内不进行分配。
var pa = new short[160];
var ba = new byte[160];
var fa = new float[160];
var timer = Stopwatch.StartNew();
for (int i = 0; i < 100000; ++i)
{
    G711.Decode(ba, pa, true);
    AudioKernels.Pcm16ToFloat(pa, fa);
}

Console.WriteLine($"100000 PCMA 20ms frames + float conversion: {timer.Elapsed.TotalMilliseconds:F3} ms");
