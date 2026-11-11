using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FunAsrNet;

/// <summary>为 x64 推理中的指数提供固定的范围归约和双精度求值。</summary>
public sealed partial class TensorMath
{
    // .NET 11.0 在部分 x64 机器的预编译运行库上调用向量指数有显著开销。
    private static readonly bool UseLocalExp = Environment.Version.Major == 11 && Avx2.IsSupported && Fma.IsSupported;

    /// <summary>在实测有收益的运行时使用本地 AVX 实现，其他组合继续使用运行库。</summary>
    /// <param name="value">八个指数输入，允许非有限值。</param>
    /// <returns>逐通道计算的单精度指数。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> InferenceExp(Vector256<float> value) =>
        UseLocalExp ? ExpAvx(value) : Vector256.Exp(value);

    /// <summary>使用双精度范围归约计算八个单精度指数。</summary>
    /// <param name="value">输入向量，超出单精度指数范围的值饱和到零或正无穷。</param>
    /// <returns>逐通道指数，含 IEEE 溢出、下溢和非有限值。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<float> ExpAvx(Vector256<float> value)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
        {
            return Vector256.Exp(value);
        }

        // exp(±104) 窄化后已是零或无穷，限制范围可安全构造双精度二次幂。
        // AVX 的第二个操作数为 NaN 时保留该 NaN。
        value = Avx.Min(Vector256.Create(104f), Avx.Max(Vector256.Create(-104f), value));
        var (low, high) = Vector256.Widen(value);
        return Vector256.Narrow(ExpReduced(low), ExpReduced(high));
    }

    /// <summary>按 x = n·ln(2) + r 归约，在半个 ln(2) 的区间内用八阶泰勒多项式计算 exp(r)。</summary>
    /// <param name="x">绝对值不超过一百零四的输入，或传播为 NaN 的输入。</param>
    /// <returns>以双精度表示的指数，窄化之前保留单精度次正规数范围。</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> ExpReduced(Vector256<double> x)
    {
        var n = Avx.RoundToNearestInteger(x * Vector256.Create(1.4426950408889634074));
        var r = Fma.MultiplyAdd(n, Vector256.Create(-0.6931471805599453094), x);
        var square = r * r;
        var fourth = square * square;
        var p23 = Fma.MultiplyAdd(r, Vector256.Create(1d / 6), Vector256.Create(0.5));
        var p45 = Fma.MultiplyAdd(r, Vector256.Create(1d / 120), Vector256.Create(1d / 24));
        var p67 = Fma.MultiplyAdd(r, Vector256.Create(1d / 5040), Vector256.Create(1d / 720));
        var low = Fma.MultiplyAdd(square, p23, Vector256.Create(1d) + r);
        var high = Fma.MultiplyAdd(square, Fma.MultiplyAdd(square, Vector256.Create(1d / 40320), p67), p45);
        var exponential = Fma.MultiplyAdd(fourth, high, low);
        var biased = Avx.ConvertToVector128Int32WithTruncation(n) + Vector128.Create(1023);
        var scale = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int64(biased), 52).AsDouble();
        return exponential * scale;
    }
}
