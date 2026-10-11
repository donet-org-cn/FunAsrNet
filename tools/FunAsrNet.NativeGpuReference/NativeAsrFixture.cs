namespace FunAsrNet.NativeGpuReference;

/// <summary>生成固定上游计算图的原生 GPU 对照适配器，产物只放在调用方指定的开发目录。</summary>
internal static class NativeAsrFixture
{
    /// <summary>保留上游算法，仅统一线程、选择 CUDA 设备并补充完整调用计时。</summary>
    /// <param name="outputDirectory">生成临时 C++ 与 CMake 文件的输出目录。</param>
    /// <param name="upstreamSource">固定 Fun-ASR 提交中的 funasr-cli.cpp 文件路径。</param>
    /// <param name="llamaDirectory">固定 llama.cpp 提交的源码根目录。</param>
    internal static void Prepare(string outputDirectory, string upstreamSource, string llamaDirectory)
    {
        string output = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(output);
        string source = File.ReadAllText(upstreamSource);
        (string Before, string After)[] replacements =
        [
            ("""
#include "ggml-cpu.h"
""", """
#include "ggml-cpu.h"
#include "ggml-cuda.h"
static ggml_backend_t reference_encoder_backend = nullptr;
static double reference_encoder_seconds = 0;
static int reference_encoder_nodes = 0;
"""),
            ("""
#if defined(__APPLE__)
""", """
#if defined(__APPLE__) || defined(FUNASR_REFERENCE_CUDA)
"""),
            ("""
cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;
""", """
cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;
    cp.n_threads=4; cp.n_threads_batch=4;
"""),
            ("""
ggml_backend_t be=ggml_backend_cpu_init();
""", """
int64_t reference_started=ggml_time_us();
    ggml_backend_t be=reference_encoder_backend ? reference_encoder_backend : ggml_backend_cpu_init();
"""),
            ("""
ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
""", """
if(reference_encoder_backend){
        reference_encoder_nodes=ggml_graph_n_nodes(gf);
        for(int i=0;i<reference_encoder_nodes;i++){
            auto node=ggml_graph_node(gf,i);
            if(!ggml_backend_supports_op(be,node)){fprintf(stderr,"Unsupported CUDA encoder op: %s\n",ggml_op_name(node->op));abort();}
        }
    }
    ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be)); ggml_gallocr_alloc_graph(ga,gf);
"""),
            ("""
ggml_backend_cpu_set_n_threads(be,8); ggml_backend_graph_compute(be,gf);
""", """
if(!reference_encoder_backend) ggml_backend_cpu_set_n_threads(be,4);
    if(ggml_backend_graph_compute(be,gf)!=GGML_STATUS_SUCCESS) abort();
"""),
            ("""
ggml_gallocr_free(ga); ggml_free(c); ggml_backend_free(be); return out;
""", """
ggml_gallocr_free(ga); ggml_free(c); if(!reference_encoder_backend) ggml_backend_free(be);
    reference_encoder_seconds=(ggml_time_us()-reference_started)/1e6; return out;
"""),
            ("""
std::vector<float> seg(pcm, pcm+n);
""", """
int64_t reference_feature_started=ggml_time_us();
    std::vector<float> seg(pcm, pcm+n);
"""),
            ("""
int D=0; auto adp=run_encoder(d.em,fbank,T,560,D);
""", """
int D=0; auto adp=run_encoder(d.em,fbank,T,560,D);
    reference_encoder_seconds=(ggml_time_us()-reference_feature_started)/1e6;
""")
        ];
        foreach (var replacement in replacements)
        {
            int first = source.IndexOf(replacement.Before, StringComparison.Ordinal);
            if (first < 0 || source.IndexOf(replacement.Before, first + replacement.Before.Length, StringComparison.Ordinal) >= 0)
            {
                throw new InvalidDataException("上游源码与固定版本不匹配，停止生成 GPU 适配器。");
            }
            source = source.Replace(replacement.Before, replacement.After, StringComparison.Ordinal);
        }
        File.WriteAllText(Path.Combine(output, "upstream-adapter.cpp"), source);
        File.WriteAllText(Path.Combine(output, "reference-main.cpp"), ReferenceMain);
        string llama = Path.GetFullPath(llamaDirectory).Replace('\\', '/');
        string common = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(upstreamSource)!, "../funasr-common")).Replace('\\', '/');
        string cmake = CMake.Replace("@LLAMA@", llama, StringComparison.Ordinal).Replace("@COMMON@", common, StringComparison.Ordinal);
        File.WriteAllText(Path.Combine(output, "CMakeLists.txt"), cmake);
        Console.WriteLine($"原生 GPU 对照文件已生成：{output}");
    }

    private const string ReferenceMain =
"""
#define _USE_MATH_DEFINES
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <fstream>
#include <sstream>
#define main upstream_cli_main
#include "upstream-adapter.cpp"
#undef main

static std::string json_string(const std::string & value) {
    std::string out;
    for(unsigned char c:value){
        if(c=='"' || c=='\\'){out+='\\';out+=c;}
        else if(c=='\n')out+="\\n";
        else if(c=='\r')out+="\\r";
        else if(c=='\t')out+="\\t";
        else if(c<32){char b[7];sprintf_s(b,"\\u%04x",c);out+=b;}
        else out+=c;
    }
    return out;
}

int main(int argc,char **argv){
    if(argc<5)return 2;
    ggml_time_init();
    SetConsoleOutputCP(CP_UTF8);
    SetProcessAffinityMask(GetCurrentProcess(),0x55);
    bool full=argc>5 && std::string(argv[5])=="--encoder-gpu";
    std::ifstream input(argv[3],std::ios::binary|std::ios::ate);
    auto size=input.tellg(); if(size<=0 || size%4!=0)return 3;
    std::vector<float> pcm((size_t)size/4);
    input.seekg(0); input.read((char*)pcm.data(),size);
    asr_decoder d;
    auto loaded=ggml_time_us();
    if(!load_decoder(argv[1],argv[2],1.0f,"",d))return 4;
    ggml_backend_buffer_t weights=nullptr;
    if(full){
        reference_encoder_backend=ggml_backend_cuda_init(0);
        if(!reference_encoder_backend)return 5;
        std::vector<std::pair<ggml_tensor*,void*>> original;
        for(auto entry:d.em.t){
            auto tensor=entry.second;
            original.push_back({tensor,tensor->data});tensor->data=nullptr;
        }
        ggml_set_no_alloc(d.em.ctx_w,true);
        weights=ggml_backend_alloc_ctx_tensors(d.em.ctx_w,reference_encoder_backend);
        if(!weights)return 6;
        for(auto item:original)ggml_backend_tensor_set(item.first,item.second,0,ggml_nbytes(item.first));
        ggml_backend_buffer_set_usage(weights,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    const char *engine=full?"cpp-cuda-encoder-decoder":"cpp-cpu-encoder-cuda-decoder";
    printf("{\"kind\":\"load\",\"engine\":\"%s\",\"seconds\":%.9f,\"threads\":4,\"gpuLayers\":99,\"encoderGpuWeightBytes\":%zu}\n",engine,(ggml_time_us()-loaded)/1e6,weights?ggml_backend_buffer_get_size(weights):0);
    int repeats=atoi(argv[4]);
    for(int i=0;i<=repeats;i++){
        auto started=ggml_time_us();
        auto text=clean_asr_text(asr_decode_window(d,pcm.data(),pcm.size(),128));
        double seconds=(ggml_time_us()-started)/1e6;
        PROCESS_MEMORY_COUNTERS memory{}; GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory));
        printf("{\"kind\":\"asr\",\"engine\":\"%s\",\"iteration\":%d,\"warmup\":%s,\"seconds\":%.9f,\"encoderSeconds\":%.9f,\"audioSeconds\":%.9f,\"encoderGpuNodes\":%d,\"maxTokens\":128,\"peakRssBytes\":%llu,\"Text\":\"%s\"}\n",engine,i,i==0?"true":"false",seconds,reference_encoder_seconds,pcm.size()/16000.0,reference_encoder_nodes,(unsigned long long)memory.PeakWorkingSetSize,json_string(text).c_str());
        fflush(stdout);
    }
    if(weights)ggml_backend_buffer_free(weights);
    if(reference_encoder_backend)ggml_backend_free(reference_encoder_backend);
    free_decoder(d);
    return 0;
}

""";

    private const string CMake =
"""
cmake_minimum_required(VERSION 3.18)
project(funasr_native_gpu_reference LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)
add_subdirectory("@LLAMA@" llama-build)
add_executable(funasr-native-gpu-reference reference-main.cpp)
target_include_directories(funasr-native-gpu-reference PRIVATE "@COMMON@")
target_compile_definitions(funasr-native-gpu-reference PRIVATE FUNASR_REFERENCE_CUDA _USE_MATH_DEFINES NOMINMAX)
target_compile_options(funasr-native-gpu-reference PRIVATE /utf-8)
target_link_libraries(funasr-native-gpu-reference PRIVATE llama ggml ggml-cuda psapi)
""";
}
