#define _USE_MATH_DEFINES
#define NOMINMAX
#include <sys/resource.h>
#include <fstream>
#include <chrono>
#define main upstream_cli_main
#include "upstream-adapter.cpp"
#undef main

static double temperature() {
    std::ifstream input("/sys/class/thermal/thermal_zone0/temp");
    double value=0; input>>value; return value/1000;
}

static std::string json_string(const std::string & value) {
    std::string out;
    for(unsigned char c:value){
        if(c=='"' || c=='\\'){out+='\\';out+=c;}
        else if(c=='\n')out+="\\n";
        else if(c=='\r')out+="\\r";
        else if(c=='\t')out+="\\t";
        else if(c<32){char b[7];snprintf(b,sizeof(b),"\\u%04x",c);out+=b;}
        else out+=c;
    }
    return out;
}

int main(int argc,char **argv){
    if(argc!=5)return 2;
    ggml_time_init();
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
        auto cooling=ggml_time_us();
        while(temperature()>60){
            if(ggml_time_us()-cooling>300000000)return 6;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        auto started=ggml_time_us();
        auto text=clean_asr_text(asr_decode_window(d,pcm.data(),pcm.size(),128));
        double seconds=(ggml_time_us()-started)/1e6;
        struct rusage memory{}; getrusage(RUSAGE_SELF,&memory);
        printf("{\"kind\":\"asr\",\"engine\":\"cpp-cpu\",\"iteration\":%d,\"warmup\":%s,\"seconds\":%.9f,\"encoderSeconds\":%.9f,\"decoderSeconds\":%.9f,\"audioSeconds\":%.9f,\"maxTokens\":128,\"peakRssBytes\":%llu,\"Text\":\"%s\"}\n",i,i==0?"true":"false",seconds,reference_encoder_seconds,seconds-reference_encoder_seconds,pcm.size()/16000.0,(unsigned long long)(memory.ru_maxrss*1024),json_string(text).c_str());
        fflush(stdout);
    }
    free_decoder(d);
    return 0;
}
