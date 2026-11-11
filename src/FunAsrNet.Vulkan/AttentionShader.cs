namespace FunAsrNet.Vulkan;

/// <summary>一组线程完成一个查询头的 FP32 注意力，不在设备上生成完整平方分数矩阵。</summary>
internal static partial class AttentionShader
{
    /// <summary>八通道点积、稳定 Softmax 和顺序加权累加，全部张量保持 FP32。</summary>
    internal const string Source = """
        #version 450
        #extension GL_KHR_shader_subgroup_basic : require
        #extension GL_KHR_shader_subgroup_shuffle : require
        layout(local_size_x=128) in;
        layout(set=0,binding=0) readonly buffer Input { float data[]; };
        layout(set=0,binding=1) writeonly buffer Output { float outputValues[]; };
        layout(push_constant) uniform Parameters { uint frames; uint width; uint heads; float divisor; } p;
        shared float query[256];
        shared float scores[4096];
        shared float reduce[128];
        void main() {
            uint id=gl_LocalInvocationID.x;
            uint group=gl_SubgroupID*(gl_SubgroupSize/8)+gl_SubgroupInvocationID/8;
            uint lane=gl_SubgroupInvocationID&7;
            uint time=gl_WorkGroupID.x, head=gl_WorkGroupID.y, size=p.width/p.heads;
            uint base=time*p.width+head*size, count=p.frames*p.width;
            for(uint c=id;c<size;c+=128) query[c]=data[base+c];
            barrier();
            for(uint first=0;first<p.frames;first+=16) {
                uint source=first+group;
                precise float sum=0.0;
                for(uint c=lane;c<size;c+=8) {
                    float key=source<p.frames ? data[count+source*p.width+head*size+c] : 0.0;
                    sum=sum+query[c]*key;
                }
                sum=sum+subgroupShuffleXor(sum,4);
                sum=sum+subgroupShuffleXor(sum,1);
                sum=sum+subgroupShuffleXor(sum,2);
                if(lane==0 && source<p.frames) scores[source]=sum/p.divisor;
            }
            barrier();
            float maximum=-3.402823466e38;
            for(uint source=id;source<p.frames;source+=128) maximum=max(maximum,scores[source]);
            reduce[id]=maximum;
            barrier();
            for(uint step=64;step>0;step>>=1) {
                if(id<step) reduce[id]=max(reduce[id],reduce[id+step]);
                barrier();
            }
            maximum=reduce[0];
            barrier();
            precise float denominator=0.0;
            for(uint source=id;source<p.frames;source+=128) {
                float probability=exp(scores[source]-maximum);
                scores[source]=probability;
                denominator=denominator+probability;
            }
            reduce[id]=denominator;
            barrier();
            for(uint step=64;step>0;step>>=1) {
                if(id<step) reduce[id]=reduce[id]+reduce[id+step];
                barrier();
            }
            float inverse=1.0/reduce[0];
            for(uint source=id;source<p.frames;source+=128) scores[source]=scores[source]*inverse;
            barrier();
            for(uint c=id;c<size;c+=128) {
                precise float sum=0.0;
                for(uint source=0;source<p.frames;source++)
                    sum=sum+scores[source]*data[2*count+source*p.width+head*size+c];
                outputValues[base+c]=sum;
            }
        }
        """;
}
