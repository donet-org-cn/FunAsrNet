namespace FunAsrNet.Vulkan;

/// <summary>四个查询共享键和值的 FP32 注意力，减少 GPU 全局内存读取。</summary>
internal static partial class AttentionTiledShader
{
    /// <summary>仅用于至多一千零二十四帧，尾查询参与同步但不写回。</summary>
    internal const string Source = """
        #version 450
        #extension GL_KHR_shader_subgroup_basic : require
        #extension GL_KHR_shader_subgroup_shuffle : require
        layout(local_size_x=128) in;
        layout(set=0,binding=0) readonly buffer Input { float data[]; };
        layout(set=0,binding=1) writeonly buffer Output { float outputValues[]; };
        layout(push_constant) uniform Parameters { uint frames; uint width; uint heads; float divisor; } p;
        shared float queries[1024];
        shared float scores[4096];
        shared float reduce[512];
        void main() {
            uint id=gl_LocalInvocationID.x;
            uint group=gl_SubgroupID*(gl_SubgroupSize/8)+gl_SubgroupInvocationID/8;
            uint lane=gl_SubgroupInvocationID&7;
            uint time=gl_WorkGroupID.x*4, head=gl_WorkGroupID.y, size=p.width/p.heads, count=p.frames*p.width;
            for(uint j=0;j<4;j++)
                for(uint c=id;c<size;c+=128)
                    queries[j*256+c]=time+j<p.frames ? data[(time+j)*p.width+head*size+c] : 0.0;
            barrier();
            for(uint first=0;first<p.frames;first+=16) {
                uint source=first+group;
                precise float a=0.0,b=0.0,c=0.0,d=0.0;
                for(uint channel=lane;channel<size;channel+=8) {
                    float key=source<p.frames ? data[count+source*p.width+head*size+channel] : 0.0;
                    a=a+queries[channel]*key;
                    b=b+queries[256+channel]*key;
                    c=c+queries[512+channel]*key;
                    d=d+queries[768+channel]*key;
                }
                a=a+subgroupShuffleXor(a,4); a=a+subgroupShuffleXor(a,1); a=a+subgroupShuffleXor(a,2);
                b=b+subgroupShuffleXor(b,4); b=b+subgroupShuffleXor(b,1); b=b+subgroupShuffleXor(b,2);
                c=c+subgroupShuffleXor(c,4); c=c+subgroupShuffleXor(c,1); c=c+subgroupShuffleXor(c,2);
                d=d+subgroupShuffleXor(d,4); d=d+subgroupShuffleXor(d,1); d=d+subgroupShuffleXor(d,2);
                if(lane==0 && source<p.frames) {
                    scores[source]=a/p.divisor;
                    scores[1024+source]=b/p.divisor;
                    scores[2048+source]=c/p.divisor;
                    scores[3072+source]=d/p.divisor;
                }
            }
            barrier();
            for(uint j=0;j<4;j++) {
                float maximum=-3.402823466e38;
                for(uint source=id;source<p.frames;source+=128) maximum=max(maximum,scores[j*1024+source]);
                reduce[j*128+id]=maximum;
            }
            barrier();
            for(uint step=64;step>0;step>>=1) {
                if(id<step)
                    for(uint j=0;j<4;j++) reduce[j*128+id]=max(reduce[j*128+id],reduce[j*128+id+step]);
                barrier();
            }
            vec4 maxima=vec4(reduce[0],reduce[128],reduce[256],reduce[384]);
            barrier();
            for(uint j=0;j<4;j++) {
                precise float sum=0.0;
                for(uint source=id;source<p.frames;source+=128) {
                    float probability=exp(scores[j*1024+source]-maxima[j]);
                    scores[j*1024+source]=probability;
                    sum=sum+probability;
                }
                reduce[j*128+id]=sum;
            }
            barrier();
            for(uint step=64;step>0;step>>=1) {
                if(id<step)
                    for(uint j=0;j<4;j++) reduce[j*128+id]=reduce[j*128+id]+reduce[j*128+id+step];
                barrier();
            }
            for(uint j=0;j<4;j++) {
                float inverse=1.0/reduce[j*128];
                for(uint source=id;source<p.frames;source+=128) scores[j*1024+source]=scores[j*1024+source]*inverse;
            }
            barrier();
            for(uint channel=id;channel<size;channel+=128) {
                precise float a=0.0,b=0.0,c=0.0,d=0.0;
                for(uint source=0;source<p.frames;source++) {
                    float value=data[2*count+source*p.width+head*size+channel];
                    a=a+scores[source]*value;
                    b=b+scores[1024+source]*value;
                    c=c+scores[2048+source]*value;
                    d=d+scores[3072+source]*value;
                }
                outputValues[time*p.width+head*size+channel]=a;
                if(time+1<p.frames) outputValues[(time+1)*p.width+head*size+channel]=b;
                if(time+2<p.frames) outputValues[(time+2)*p.width+head*size+channel]=c;
                if(time+3<p.frames) outputValues[(time+3)*p.width+head*size+channel]=d;
            }
        }
        """;
}
