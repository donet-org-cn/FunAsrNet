namespace FunAsrNet.NativeReference;

/// <summary>生成仅使用原生 CPU 后端的 ASR 对照，统一线程和计时，不改变上游计算图。</summary>
internal static class NativeAsrCpuFixture
{
    /// <summary>从固定上游源码生成独立 CPU 基准，产物写入指定开发目录。</summary>
    /// <param name="outputDirectory">C++ 包装和 CMake 文件的输出目录。</param>
    /// <param name="upstreamSource">固定提交中的 funasr-cli.cpp 文件。</param>
    /// <param name="llamaDirectory">固定 llama.cpp 源码根目录。</param>
    internal static void Prepare(string outputDirectory, string upstreamSource, string llamaDirectory)
    {
        string output = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(output);
        string source = File.ReadAllText(upstreamSource);
        (string Before, string After)[] replacements =
        [
            ("#include \"ggml-cpu.h\"", "#include \"ggml-cpu.h\"\nstatic double reference_encoder_seconds = 0;"),
            ("mp.n_gpu_layers=99;", "mp.n_gpu_layers=0;"),
            ("cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;", "cp.n_ctx=2048; cp.n_batch=2048; cp.n_ubatch=2048;\n    cp.n_threads=4; cp.n_threads_batch=4;"),
            ("ggml_backend_cpu_set_n_threads(be,8);", "ggml_backend_cpu_set_n_threads(be,4);"),
            ("std::vector<float> seg(pcm, pcm+n);", "int64_t reference_feature_started=ggml_time_us();\n    std::vector<float> seg(pcm, pcm+n);"),
            ("int D=0; auto adp=run_encoder(d.em,fbank,T,560,D);", "int D=0; auto adp=run_encoder(d.em,fbank,T,560,D);\n    reference_encoder_seconds=(ggml_time_us()-reference_feature_started)/1e6;")
        ];
        foreach (var replacement in replacements)
        {
            int first = source.IndexOf(replacement.Before, StringComparison.Ordinal);
            if (first < 0 || source.IndexOf(replacement.Before, first + replacement.Before.Length, StringComparison.Ordinal) >= 0)
            {
                throw new InvalidDataException("上游源码与固定版本不匹配，停止生成 CPU 对照。");
            }
            source = source.Replace(replacement.Before, replacement.After, StringComparison.Ordinal);
        }
        File.WriteAllText(Path.Combine(output, "upstream-adapter.cpp"), source);
        File.WriteAllText(Path.Combine(output, "reference-main.cpp"), ReferenceMain);
        string llama = Path.GetFullPath(llamaDirectory).Replace('\\', '/');
        string common = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(upstreamSource)!, "../funasr-common")).Replace('\\', '/');
        File.WriteAllText(Path.Combine(output, "CMakeLists.txt"), CMake.Replace("@LLAMA@", llama, StringComparison.Ordinal).Replace("@COMMON@", common, StringComparison.Ordinal));
        Console.WriteLine($"原生 CPU 对照文件已生成：{output}");
    }

    private const string ReferenceMain =
"""
#define _USE_MATH_DEFINES
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <fstream>
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
    if(argc!=5)return 2;
    ggml_time_init();
    SetConsoleOutputCP(CP_UTF8);
    if(!SetProcessAffinityMask(GetCurrentProcess(),0x55))return 3;
    std::ifstream input(argv[3],std::ios::binary|std::ios::ate);
    auto size=input.tellg(); if(size<=0 || size%4!=0)return 4;
    std::vector<float> pcm((size_t)size/4);
    input.seekg(0); input.read((char*)pcm.data(),size);
    asr_decoder d;
    auto loaded=ggml_time_us();
    if(!load_decoder(argv[1],argv[2],1.0f,"",d))return 5;
    printf("{\"kind\":\"load\",\"engine\":\"cpp-cpu\",\"seconds\":%.9f,\"threads\":4,\"gpuLayers\":0,\"promptCache\":false}\n",(ggml_time_us()-loaded)/1e6);
    int repeats=atoi(argv[4]);
    for(int i=0;i<=repeats;i++){
        auto started=ggml_time_us();
        auto text=clean_asr_text(asr_decode_window(d,pcm.data(),pcm.size(),128));
        double seconds=(ggml_time_us()-started)/1e6;
        PROCESS_MEMORY_COUNTERS memory{}; GetProcessMemoryInfo(GetCurrentProcess(),&memory,sizeof(memory));
        printf("{\"kind\":\"asr\",\"engine\":\"cpp-cpu\",\"iteration\":%d,\"warmup\":%s,\"seconds\":%.9f,\"encoderSeconds\":%.9f,\"decoderSeconds\":%.9f,\"audioSeconds\":%.9f,\"maxTokens\":128,\"peakRssBytes\":%llu,\"Text\":\"%s\"}\n",i,i==0?"true":"false",seconds,reference_encoder_seconds,seconds-reference_encoder_seconds,pcm.size()/16000.0,(unsigned long long)memory.PeakWorkingSetSize,json_string(text).c_str());
        fflush(stdout);
    }
    free_decoder(d);
    return 0;
}

""";

    private const string CMake =
"""
cmake_minimum_required(VERSION 3.18)
project(funasr_native_cpu_reference LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)
add_subdirectory("@LLAMA@" llama-build)
add_executable(funasr-native-cpu-reference reference-main.cpp)
target_include_directories(funasr-native-cpu-reference PRIVATE "@COMMON@")
target_compile_definitions(funasr-native-cpu-reference PRIVATE _USE_MATH_DEFINES NOMINMAX)
target_compile_options(funasr-native-cpu-reference PRIVATE /utf-8)
target_link_libraries(funasr-native-cpu-reference PRIVATE llama ggml psapi)
""";
}
