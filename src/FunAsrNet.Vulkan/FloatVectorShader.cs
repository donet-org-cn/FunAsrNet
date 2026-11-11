namespace FunAsrNet.Vulkan;

/// <summary>原精度 FP32 矩阵向量内核，八或三十二通道共享一行，避免单帧浪费矩阵分块。</summary>
internal static partial class FloatVectorShader
{
    /// <summary>使用 FP32 FMA 和固定子组归约，不将激活或权重转换为低精度。</summary>
    internal const string Source = """
        #version 450
        #extension GL_KHR_shader_subgroup_basic : require
        #extension GL_KHR_shader_subgroup_shuffle : require
        #ifdef WIDE
        #define LANES 32
        #else
        #define LANES 8
        #endif
        layout(local_size_x=128) in;
        layout(set=0,binding=0) readonly buffer Input { float x[]; };
        layout(set=0,binding=1) readonly buffer Weight { float w[]; };
        layout(set=0,binding=2) readonly buffer Bias { float bias[]; };
        layout(set=0,binding=3) writeonly buffer Output { float y[]; };
        layout(push_constant) uniform Parameters {
            uint M; uint N; uint K; uint mode;
            uint batches; uint channels; uint inputLength; uint outLength;
            uint kernel; uint stride; uint dilation; uint padLeft; uint hasBias;
        } p;
        void main() {
            uint row=gl_WorkGroupID.x*(128/LANES)+gl_SubgroupID*(gl_SubgroupSize/LANES)+gl_SubgroupInvocationID/LANES;
            uint frame=gl_WorkGroupID.y, lane=gl_SubgroupInvocationID&(LANES-1);
            bool valid=row<p.N && frame<p.M;
            float sum=0.0;
            for(uint k=lane;k<p.K;k+=LANES) {
                float a=valid ? x[frame*p.K+k] : 0.0;
                float b=valid ? w[row*p.K+k] : 0.0;
                sum=fma(a,b,sum);
            }
            sum+=subgroupShuffleXor(sum,1);
            sum+=subgroupShuffleXor(sum,2);
            sum+=subgroupShuffleXor(sum,4);
            #ifdef WIDE
            sum+=subgroupShuffleXor(sum,8);
            sum+=subgroupShuffleXor(sum,16);
            #endif
            if(valid && lane==0) {
                float value=sum+((p.hasBias&1)!=0 ? bias[row] : 0.0);
                if((p.hasBias&8)!=0) value=value<0.0 ? 0.0 : value;
                y[frame*p.N+row]=value;
            }
        }
        """;
}
