namespace FunAsrNet.Vulkan;

/// <summary>八条 GPU 通道协同读取一个原始 Q8_0 块，整数求和及浮点累加顺序保持不变。</summary>
internal static partial class CooperativeQuantizedShader
{
    /// <summary>需要完整子组和 shuffle 支持；边界行仍参与交换，只有有效行写回。</summary>
    internal const string Source = """
        #version 450
        #extension GL_KHR_shader_subgroup_basic : require
        #extension GL_KHR_shader_subgroup_shuffle : require
        #ifdef INTEGER_DOT
        #extension GL_EXT_integer_dot_product : require
        #endif
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
            uint row=gl_WorkGroupID.x*16+gl_SubgroupID*(gl_SubgroupSize/8)+gl_SubgroupInvocationID/8;
            uint frame=gl_WorkGroupID.y, word=gl_SubgroupInvocationID&7;
            bool valid=row<p.N && frame<p.M;
            precise float total=0.0;
            uint blocks=p.K/32;
            for(uint block=0;block<blocks;block++) {
                uint a=valid ? xq[frame*(p.K/4)+block*8+word] : 0;
                uint b=valid ? wq[row*(p.K/4)+block*8+word] : 0;
                #ifdef INTEGER_DOT
                int sum=dotPacked4x8EXT(int(a),int(b));
                #else
                int sum=0;
                for(uint item=0;item<4;item++) sum+=byteAt(a,item)*byteAt(b,item);
                #endif
                sum+=subgroupShuffleXor(sum,1);
                sum+=subgroupShuffleXor(sum,2);
                sum+=subgroupShuffleXor(sum,4);
                if(valid && word==0) {
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
            }
            if(valid && word==0) y[frame*p.N+row]=total+((p.hasBias&1)!=0 ? bias[row] : 0.0);
        }
        """;
}
