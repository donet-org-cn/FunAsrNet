# FunAsrNet

[简体中文](README.md) · English · .NET 10+ · [Release notes](doc/releases/1.0.1.md) · [Apache-2.0](LICENSE)

**Mandarin speech recognition, Chinese speech synthesis, and telephony audio for .NET applications, with inference implemented in C# by default.**

FunAsrNet is an independent community implementation of the fixed [QwenAudio/Fun-ASR](https://github.com/QwenAudio/Fun-ASR) architecture. It provides CPU SIMD kernels, an optional Vulkan GPU backend, a C# API, a CLI, and ASP.NET Core examples. It is intended for developers who want to integrate validated speech models into .NET services and inspect, debug, or adapt the inference code.

The current scope covers specific Fun-ASR, FSMN VAD, and Chinese VITS models. It is not the official FunASR SDK, the entire FunASR toolkit, or a general ONNX runtime.

## Features

| Feature | Current support |
|---|---|
| Speech recognition | Mandarin WAV/PCM transcription; fixed F16/FP32 encoder and Qwen3 Q8_0/FP32 decoder |
| Live transcription UI | WebSocket, VAD segmentation, sliding-window previews, revisions, and cancellation of stale previews |
| Chinese speech synthesis | 1 Huayan voice and 174 AISHELL3 speakers, 175 in total after installing both models; speed, volume, sample rate, and format controls |
| Telephony audio | PCM16, G.711 PCMA/PCMU, and resampling for integration with an existing SIP/PJSIP media layer |
| CPU acceleration | x64 AVX2/FMA/AVX-VNNI, ARM64 NEON/SDOT, and fallback paths when the relevant instructions are unavailable |
| Optional GPU | Vulkan FP32 matrices, convolutions, and long-sequence attention; original Q8_0 projections; kernels selected by shape and device capabilities |
| Services and examples | HTTP/WebSocket APIs, browser microphone transcription, WAV upload, TTS playback, and downloads |
| Session management | Shared read-only weights, independent session workspaces, bounded queues, quotas, cancellation, and pooled buffers |

## Strengths and limitations

| Area | Strength | Current limitation |
|---|---|---|
| .NET integration | CPU model loading, features, operators, and decoding are implemented in C#; the core package has no native inference-engine dependency | Models and pronunciation assets must be prepared separately; Vulkan requires a system GPU driver |
| Maintainability | Inference paths, memory layouts, SIMD kernels, and shader sources are available for inspection and modification | Coverage is limited to fixed models and the required operator subset |
| Performance | Optimized matrices, features, decoding, weight loading, grouped projections, and buffer reuse | The latest matched GPU/TTS native comparisons still show a gap; first requests, long recordings, and concurrent workloads need separate measurements |
| Precision | Original BF16/FP32 weights can be expanded losslessly to FP32 GGUF; comparisons keep the same weight quantization | The existing F16 KV cache remains; GPU reductions are not guaranteed to match CPU results bit for bit; FP32 weights require more memory |
| Live and telephony integration | VAD, previews, queues, and audio codec APIs are provided | Live recognition uses segmentation and re-encoding; incremental frame encoding, SIP signaling, RTP jitter buffering, and AEC are not implemented |
| Platforms and languages | Hardware measurements exist for Windows x64 and Linux ARM64; model-free CI runs on Windows/Linux | Mandarin is the validated language scope; other languages, macOS, and other GPUs require separate validation; NPU execution is not integrated into production models |

TTS currently has no separate pitch, emotion, or voice-cloning API. Polyphonic characters, personal names, real-call accuracy, and sustained concurrency require validation on application-specific data. Call each model instance serially; adding sessions does not imply linear throughput scaling.

## Quick start

### Packages and models

Core package: [FunAsrNet](https://www.nuget.org/packages/FunAsrNet/1.0.1). Optional GPU package: [FunAsrNet.Vulkan](https://www.nuget.org/packages/FunAsrNet.Vulkan/1.0.1).

```shell
dotnet add package FunAsrNet --version 1.0.1
```

Requires .NET 10 or later; correctness regressions have also been run on .NET 11. **Model weights are not included in the NuGet packages.** Prepare weights and assets using the [model manifest and SHA-256 hashes](doc/models.md). See the [precision and export notes](doc/cpu-gpu-original-20261111.md) for original weights.

### Speech recognition

```csharp
using FunAsrNet;

var asr = new FunAsr(
    "models/funasr-encoder-f16.gguf",
    "models/qwen3-0.6b-q8_0.gguf",
    threads: 4);
var (samples, rate) = AudioFile.ReadWave("input.wav");
var result = asr.Transcribe(
    AudioFile.Resample(samples, rate, 16000),
    maxTokens: 128);
Console.WriteLine(result.Text);
```

### Enable the GPU backend

Install the extension package and attach the backend to the instance above before its first inference:

```shell
dotnet add package FunAsrNet.Vulkan --version 1.0.1
```

```csharp
using FunAsrNet.Vulkan;

using var gpu = new VulkanFloatBackend();
asr.FloatBackend = gpu;
```

Complete all inference using this backend before disposing it. The GPU handles selected operators; the remaining computation runs on the CPU. The current Web service defaults to CPU execution and does not expose a server GPU configuration switch.

### Chinese speech synthesis

```csharp
var tts = new MandarinTts(
    "models/zh_CN-huayan-medium.onnx",
    "assets/mandarin-lexicon.json",
    threads: 4);
var speech = tts.Synthesize("您好，欢迎致电。", speed: 1, seed: 1);
AudioFile.WriteWave("speech.wav", speech, tts.NativeSampleRate);
```

See the [TTS documentation](doc/tts.md) for AISHELL3 speaker selection, output formats, and parameters.

## Performance against native implementations

The tables below use matched runs on the same machine and report warmed-up median times in seconds. **Lower is faster.** Reports retain full conditions and slow samples. Do not combine timings from different dates into a speed ratio.

### Latest matched C++ CPU recheck: 2026-11-11

i7-12700, four threads, identical F16/Q8 weights, prompt caching disabled, and 6 measured calls per cell. ASR timing covers the complete transcription call:

| Input | FunAsrNet CPU | Original C++ CPU |
|---|---:|---:|
| 6 seconds | 0.749 | 0.857 |
| 24-second repeated sample | 2.920 | 3.022 |

This recheck predates the last kernel optimization round included in this release. Desktop load varied noticeably, and the inputs cover a small sample. These results do not establish a language advantage or recognition accuracy across a broad corpus. Source: [matched C++ recheck](doc/cpu-gpu-original-20261111.md).

### Latest full matched CPU/GPU native comparison: 2026-10-12 snapshot

i7-12700, RTX 3060, four threads, and 10 measured calls per cell. ASR covers full transcription. TTS covers model execution with identical phoneme input, excluding the text frontend, playback, and resampling:

| Workload | FunAsrNet CPU | Native CPU | FunAsrNet Vulkan | Native GPU |
|---|---:|---:|---:|---:|
| ASR, 6-second original PCM | 0.670 | 0.730 | 0.346 | 0.098 |
| ASR, 28-second repeated sample | 2.881 | 3.039 | 1.344 | 0.263 |
| TTS, Huayan speaker 0 | 0.141 | 0.108 | 0.154 | 0.028 |
| TTS, AISHELL3 speaker 50 | 0.067 | 0.046 | 0.087 | 0.026 |

The ASR references are fixed C++ CPU and original-graph CUDA adaptations. TTS uses the same models with ONNX Runtime CPU/CUDA. GPU timings include transfers and synchronization; internal ASR CUDA precision can differ from Vulkan FP32. **In this snapshot, Vulkan and TTS had not reached native-backend performance, and Vulkan TTS was also slower than its own CPU path.** See the [matched-run report](doc/gpu-round7-20261012.md) for initialization, first requests, precision boundaries, and complete data.

### Subsequent optimizations in this release

In the latest 24-second sample measurements on 2026-11-11, CPU F16/Q8 and original-weight paths took **2.406 / 6.460 seconds**. The expanded GPU recheck measured **0.865 / 0.980 seconds**, respectively. Against their own controls in that round, observed CPU time decreased by about 4.0% / 8.4% and GPU time by about 7.3% / 10.8%.

CPU cells contain 6 calls. GPU cells contain 16 interleaved calls with a 4 GiB weight cache; only that GPU benchmark process was temporarily set to AboveNormal priority. Ordinary desktop batches also showed regressions, and short Q8 inputs were essentially unchanged. Native CUDA and complete TTS models were not rerun in this round, so it does not establish the current native GPU gap. Text and token counts agreed for the same input across 320 measured model calls. See the [latest optimization report](doc/cpu-gpu-round2-20261111.md).

Historical measurements, loading and first requests, ARM64, Native AOT, and raw evidence are collected in the [performance index](doc/performance.md).

## Build and run from source

Source builds follow global.json and require .NET 10 SDK 10.0.300 or a newer SDK within the same major version accepted by its roll-forward policy:

```shell
dotnet build FunAsrNet.sln -c Release
dotnet run --project tests/FunAsrNet.Checks -c Release --no-build
dotnet run --project tools/FunAsrNet.CodeStyle -c Release --no-build -- .
```

After preparing the models, start the browser demo:

```shell
dotnet run --project src/FunAsrNet.Server -c Release -- --urls http://localhost:18080 --models models --lexicon assets/mandarin-lexicon.json --threads 4
```

Open http://localhost:18080. Remote microphone access requires HTTPS; see the [deployment guide](doc/deployment.md). The [CLI](src/FunAsrNet.Cli) also supports offline transcription and synthesis.

## Documentation and licensing

The detailed documents linked below are currently primarily in Chinese.

| Topic | Documentation |
|---|---|
| Models and precision | [Model manifest](doc/models.md) · [Original weights](doc/cpu-gpu-original-20261111.md) |
| Usage and deployment | [HTTP/WebSocket and PJSIP](doc/api.md) · [TTS](doc/tts.md) · [Deployment and HTTPS](doc/deployment.md) |
| Implementation and capacity | [Architecture](doc/architecture.md) · [Concurrency and memory](doc/concurrency-and-memory.md) |
| Performance and development | [Measurements and evidence](doc/performance.md) · [Development checks](doc/development.md) · [NuGet/Release workflow](doc/releases.md) |

Code is licensed under [Apache-2.0](LICENSE). Models and pronunciation assets retain their upstream terms: Huayan training-data licensing is listed as Unknown, and the AISHELL3 archive has no standalone model LICENSE. This project does not relicense those assets. See the [third-party notices](THIRD_PARTY_NOTICES.md) for provenance and the Vulkan binding license.
