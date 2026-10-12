namespace FunAsrNet.Vulkan;

/// <summary>FP32 线性与隐式展开一维卷积的 GPU 程序来源；构建工具可生成 SPIR-V 字节码。</summary>
internal static partial class SpeechShader
{
    /// <summary>不启用 FP16 或 TF32 的 GLSL 来源，嵌入 C# 以保持来源可审阅。</summary>
    internal const string Source = """
        #version 450
        layout(local_size_x = 8, local_size_y = 8) in;
        layout(set=0,binding=0) readonly buffer Input { float x[]; };
        layout(set=0,binding=1) readonly buffer Weight { float w[]; };
        layout(set=0,binding=2) readonly buffer Bias { float bias[]; };
        layout(set=0,binding=3) writeonly buffer Output { float y[]; };
        layout(push_constant) uniform Parameters {
            uint M; uint N; uint K; uint mode;
            uint batches; uint channels; uint inputLength; uint outLength;
            uint kernel; uint stride; uint dilation; uint padLeft; uint hasBias;
        } p;
        shared float A[32][33];
        shared float B[32][33];
        float loadA(uint row, uint column) {
            if (row >= p.M || column >= p.K) return 0.0;
            if (p.mode == 0) return x[row * p.K + column];
            uint batch = row / p.outLength;
            uint time = row % p.outLength;
            uint channel = column / p.kernel;
            uint tap = column % p.kernel;
            int at = int(time * p.stride + tap * p.dilation) - int(p.padLeft);
            if (at < 0 || uint(at) >= p.inputLength) return 0.0;
            return x[(batch * p.channels + channel) * p.inputLength + uint(at)];
        }
        void main() {
            uint lx=gl_LocalInvocationID.x, ly=gl_LocalInvocationID.y;
            uint tid=ly*8+lx;
            uint m0=gl_WorkGroupID.x*32, n0=gl_WorkGroupID.y*32;
            float sum[16];
            for(uint i=0;i<16;i++) sum[i]=0.0;
            for(uint k0=0;k0<p.K;k0+=32) {
                for(uint item=tid;item<1024;item+=64) {
                    uint row=item/32, col=item%32;
                    A[col][row]=loadA(m0+row,k0+col);
                    B[col][row]=(n0+row<p.N && k0+col<p.K) ? w[(n0+row)*p.K+k0+col] : 0.0;
                }
                barrier();
                for(uint k=0;k<32;k++) {
                    for(uint mi=0;mi<4;mi++) {
                        float av=A[k][ly*4+mi];
                        for(uint ni=0;ni<4;ni++) {
                            uint index=mi*4+ni;
                            if((p.hasBias&4)!=0) {
                                precise float product=av*B[k][lx*4+ni];
                                precise float separate=sum[index]+product;
                                sum[index]=separate;
                            } else {
                                sum[index]=fma(av,B[k][lx*4+ni],sum[index]);
                            }
                        }
                    }
                }
                barrier();
            }
            for(uint mi=0;mi<4;mi++) {
                uint row=m0+ly*4+mi;
                for(uint ni=0;ni<4;ni++) {
                    uint channel=n0+lx*4+ni;
                    if(row<p.M && channel<p.N) {
                        uint at=p.mode==0 ? row*p.N+channel : (row/p.outLength*p.N+channel)*p.outLength+row%p.outLength;
                        y[at]=sum[mi*4+ni]+((p.hasBias&1)!=0 ? bias[channel] : 0.0);
                    }
                }
            }
        }
        """;
}
