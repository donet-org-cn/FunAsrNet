namespace FunAsrNet;

/// <summary>
/// Huayan 模型需要的张量形状、索引、逐元素与卷积算子。
/// </summary>
public static partial class TensorOps
{
    /// <summary>
    /// 对张量执行模型所需的矩阵乘法及批维广播。
    /// </summary>
    /// <param name="a">第一个输入操作数。</param>
    /// <param name="b">第二个输入操作数。</param>
    /// <param name="math">用于矩阵与向量运算的托管计算内核。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor MatMul(Tensor a, Tensor b, TensorMath math)
    {
        bool av = a.Rank == 1, bv = b.Rank == 1;
        if (av)
        {
            a = a.View([1, a.Count]);
        }

        if (bv)
        {
            b = b.View([b.Count, 1]);
        }

        int m = a.Shape[^2], k = a.Shape[^1], n = b.Shape[^1];
        if (k != b.Shape[^2])
        {
            throw new ArgumentException("MatMul inner dimension mismatch");
        }

        var batchShape = BroadcastShape(a.Shape[..^2], b.Shape[..^2]);
        int batches = Tensor.Size(batchShape);
        var outputShape = batchShape.Concat(new[] { m, n }).ToArray();
        var y = Tensor.Zeros(outputShape, a.Integer && b.Integer);
        var ab = new Tensor([], a.Shape[..^2]);
        var bb = new Tensor([], b.Shape[..^2]);
        var sa = BroadcastStrides(ab, batchShape.Length);
        var sb = BroadcastStrides(bb, batchShape.Length);
        int batchSteps = b.Rank == 2 && batches > 0 ? 1 : batches;
        for (int batch = 0; batch < batchSteps; ++batch)
        {
            int ao = Offset(batch, batchShape, sa) * m * k, bo = Offset(batch, batchShape, sb) * k * n;
            var weights = new float[k * n];
            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    weights[j * k + i] = b.Data[bo + i * n + j];
                }
            }

            // 右侧矩阵没有批维时，所有左侧行共享权重，一次调度并直接读取连续输入。
            int frames = b.Rank == 2 ? batches * m : m;
            float[] input = b.Rank == 2 ? a.Data : a.Data.AsSpan(ao, m * k).ToArray();
            var result = math.Linear(input, frames, new Weight("MatMul", [k, n]) { Values = weights });
            result.CopyTo(y.Data, batch * m * n);
            TensorWorkspace.Release(result);
        }

        if (bv)
        {
            outputShape = outputShape[..^1];
        }

        if (av)
        {
            outputShape = outputShape.Where((_, i) => i != batchShape.Length).ToArray();
        }

        return y.View(outputShape);
    }

    /// <summary>
    /// 执行一维卷积，按节点属性处理步长、填充和分组。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="node">当前 ONNX 节点及其属性。</param>
    /// <param name="math">用于矩阵与向量运算的托管计算内核。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor Conv1D(Tensor x, Tensor w, Tensor? bias, OnnxNode node, TensorMath math)
    {
        if (x.Rank != 3 || w.Rank != 3)
        {
            throw new NotSupportedException("Only 1D convolution is required by this voice");
        }

        int batches = x.Shape[0], channels = x.Shape[1], length = x.Shape[2], outputs = w.Shape[0], kernel = w.Shape[2];
        int groups = node.Int("group", 1), stride = node.Ints("strides", [1])[0], dilation = node.Ints("dilations", [1])[0];
        int[] pads = node.Ints("pads", [0, 0]);
        if (math.FloatBackend?.TryConvolution(x, w, bias, node, out var gpuOutput) == true)
        {
            return gpuOutput!;
        }

        int outLength = (length + pads[0] + pads[1] - dilation * (kernel - 1) - 1) / stride + 1;
        var y = Tensor.ForOverwrite([batches, outputs, outLength]);
        int inGroup = channels / groups, outGroup = outputs / groups;
        if (w.Shape[1] != inGroup || channels % groups != 0 || outputs % groups != 0)
        {
            throw new ArgumentException("Convolution group mismatch");
        }

        if (math.Simd && System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported
            && groups == 1 && stride == 1 && length >= 256 && channels <= 128 && outputs == 1
            && Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_CONV") != "1")
        {
            DirectConv1D(x, w, bias, y, dilation, pads[0], math);
            return y;
        }

        if (groups == channels && outputs == channels && stride == 1)
        {
            math.For(
                batches * channels,
                index =>
                {
                    int channel = index % channels;
                    var dest = y.Data.AsSpan(index * outLength, outLength);
                    dest.Fill(bias?.Data[channel] ?? 0);
                    for (int tap = 0; tap < kernel; ++tap)
                    {
                        int shift = tap * dilation - pads[0], first = Math.Max(0, -shift), last = Math.Min(outLength, length - shift);
                        if (last > first)
                        {
                            TensorMath.AddScaled(dest.Slice(first, last - first), x.Data.AsSpan(index * length + first + shift, last - first), w.Data[channel * kernel + tap]);
                        }
                    }
                });
            return y;
        }

        if (math.Simd && System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported
            && stride == 1 && outLength >= 128 && Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_CONV") != "1")
        {
            DirectConv1DAvx(x, w, bias, y, groups, dilation, pads[0], math);
            return y;
        }

        for (int batch = 0; batch < batches; ++batch)
        {
            for (int group = 0; group < groups; ++group)
            {
                int width = inGroup * kernel;
                var columns = TensorWorkspace.Allocate(outLength * width, clear: true);
                math.For(
                    (outLength + 31) / 32,
                    block =>
                    {
                        for (int t = block * 32; t < Math.Min(outLength, block * 32 + 32); ++t)
                        {
                            for (int c = 0; c < inGroup; ++c)
                            {
                                for (int tap = 0; tap < kernel; ++tap)
                                {
                                    int at = t * stride + tap * dilation - pads[0];
                                    if ((uint)at < length)
                                    {
                                        columns[t * width + c * kernel + tap] = x.Data[(batch * channels + group * inGroup + c) * length + at];
                                    }
                                }
                            }
                        }
                    });
                var weight = new Weight(node.Name, [width, outGroup])
                {
                    Values = groups == 1 ? w.Data : w.Data.AsSpan(group * outGroup * width, outGroup * width).ToArray()
                };
                var b = bias is null ? null : new Weight("bias", [outGroup])
                {
                    Values = groups == 1 ? bias.Data : bias.Data.AsSpan(group * outGroup, outGroup).ToArray()
                };
                var result = math.Linear(columns, outLength, weight, b);
                TensorWorkspace.Release(columns);
                for (int c = 0; c < outGroup; ++c)
                {
                    for (int t = 0; t < outLength; ++t)
                    {
                        y.Data[(batch * outputs + group * outGroup + c) * outLength + t] = result[t * outGroup + c];
                    }
                }
                TensorWorkspace.Release(result);
            }
        }

        return y;
    }

    /// <summary>
    /// 执行一维转置卷积，恢复声码器所需的时间分辨率。
    /// </summary>
    /// <param name="x">连续存储的输入张量或向量。</param>
    /// <param name="w">当前矩阵或卷积操作的权重。</param>
    /// <param name="bias">可选偏置；为空时不叠加偏置。</param>
    /// <param name="node">当前 ONNX 节点及其属性。</param>
    /// <param name="math">用于矩阵与向量运算的托管计算内核。</param>
    /// <returns>具有输出形状的结果张量。</returns>
    public static Tensor ConvTranspose1D(Tensor x, Tensor w, Tensor? bias, OnnxNode node, TensorMath math)
    {
        if (x.Rank != 3 || node.Int("group", 1) != 1)
        {
            throw new NotSupportedException("Transposed convolution: 1D group=1 required");
        }

        int batches = x.Shape[0], inputs = x.Shape[1], length = x.Shape[2], outputs = w.Shape[1], kernel = w.Shape[2];
        int stride = node.Ints("strides", [1])[0], dilation = node.Ints("dilations", [1])[0];
        int[] pads = node.Ints("pads", [0, 0]);
        int outLength = stride * (length - 1) + dilation * (kernel - 1) + 1 - pads[0] - pads[1] + node.Ints("output_padding", [0])[0];
        var y = Tensor.ForOverwrite([batches, outputs, outLength]);
        if (math.FloatBackend is null && math.Simd && System.Runtime.Intrinsics.X86.Avx2.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported
            && dilation == 1 && stride > 1 && kernel >= stride && length >= 16
            && Environment.GetEnvironmentVariable("FUNASRNET_LEGACY_TRANSPOSE_CONV") != "1")
        {
            PolyphaseConvTransposeAvx(x, w, bias, y, stride, pads[0], math);
            return y;
        }
        var packed = new float[inputs * outputs * kernel];
        for (int i = 0; i < inputs; ++i)
        {
            for (int o = 0; o < outputs; ++o)
            {
                for (int k = 0; k < kernel; ++k)
                {
                    packed[(o * kernel + k) * inputs + i] = w.Data[(i * outputs + o) * kernel + k];
                }
            }
        }

        var weight = new Weight(node.Name, [inputs, outputs * kernel])
        {
            Values = packed
        };
        for (int b = 0; b < batches; ++b)
        {
            var input = TensorWorkspace.Allocate(length * inputs);
            for (int t = 0; t < length; ++t)
            {
                for (int i = 0; i < inputs; ++i)
                {
                    input[t * inputs + i] = x.Data[(b * inputs + i) * length + t];
                }
            }

            var result = math.Linear(input, length, weight);
            TensorWorkspace.Release(input);
            math.For(
                outputs,
                channel =>
                {
                    var dest = y.Data.AsSpan((b * outputs + channel) * outLength, outLength);
                    dest.Fill(bias?.Data[channel] ?? 0);
                    for (int t = 0; t < length; ++t)
                    {
                        for (int k = 0; k < kernel; ++k)
                        {
                            int at = t * stride + k * dilation - pads[0];
                            if ((uint)at < outLength)
                            {
                                dest[at] += result[(t * outputs + channel) * kernel + k];
                            }
                        }
                    }
                });
            TensorWorkspace.Release(result);
        }

        return y;
    }
}
