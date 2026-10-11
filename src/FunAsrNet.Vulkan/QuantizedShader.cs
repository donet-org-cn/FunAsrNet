namespace FunAsrNet.Vulkan;

/// <summary>原始 Q8_0 字节与 FP32 分组缩放的 GPU 程序来源。</summary>
internal static partial class QuantizedShader
{
    /// <summary>按三十二个元素进行整数点积，不引入新的权重量化。</summary>
    internal const string Source = """
        #version 450
        layout(local_size_x=128) in;
        layout(set=0,binding=0) readonly buffer Input { uint xq[]; };
        layout(set=0,binding=1) readonly buffer Weight { uint wq[]; };
        layout(set=0,binding=2) readonly buffer InputScale { float xs[]; };
        layout(set=0,binding=3) readonly buffer WeightScale { float ws[]; };
        layout(set=0,binding=4) readonly buffer Bias { float bias[]; };
        layout(set=0,binding=5) writeonly buffer Output { float y[]; };
        layout(push_constant) uniform Parameters { uint M; uint N; uint K; uint hasBias; } p;
        int byteAt(uint value,uint at) { return int(value << (24-at*8)) >> 24; }
        void main() {
            uint row=gl_GlobalInvocationID.x, frame=gl_GlobalInvocationID.y;
            if(row>=p.N || frame>=p.M) return;
            precise float total=0.0;
            uint blocks=p.K/32;
            for(uint block=0;block<blocks;block++) {
                int sum=0;
                for(uint word=0;word<8;word++) {
                    uint a=xq[frame*(p.K/4)+block*8+word];
                    uint b=wq[row*(p.K/4)+block*8+word];
                    for(uint item=0;item<4;item++) sum+=byteAt(a,item)*byteAt(b,item);
                }
                precise float product;
                if((p.hasBias&2)!=0) {
                    precise float scale=ws[row*blocks+block]*xs[frame*blocks+block];
                    product=float(sum)*scale;
                } else {
                    precise float scaled=float(sum)*ws[row*blocks+block];
                    product=scaled*xs[frame*blocks+block];
                }
                total=total+product;
            }
            y[frame*p.N+row]=total+((p.hasBias&1)!=0 ? bias[row] : 0.0);
        }
        """;
}
