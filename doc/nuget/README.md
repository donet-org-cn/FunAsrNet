# FunAsrNet

Independent C#/.NET 10 speech inference for fixed Fun-ASR speech recognition and Chinese VITS speech synthesis models. This is a community implementation, not the official FunASR SDK or a general ONNX runtime.

## Packages

| Package | Purpose |
|---|---|
| `FunAsrNet` | Managed CPU inference, SIMD kernels, VAD, WAV/resampling and G.711 utilities |
| `FunAsrNet.Vulkan` | Optional Vulkan GPU backend; depends on `FunAsrNet` |

```shell
dotnet add package FunAsrNet --version 1.0.0
dotnet add package FunAsrNet.Vulkan --version 1.0.0
```

Requires .NET 10 or later. The CPU package has no native inference dependency. The GPU package requires a compatible installed Vulkan driver; it does not bundle a driver, CUDA, ONNX Runtime or model weights.

## Speech recognition

Prepare the compatible encoder and decoder GGUF files described in the [model manifest](https://github.com/donet-org-cn/FunAsrNet/blob/main/doc/models.md).

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

To enable the optional GPU backend before inference:

```csharp
using FunAsrNet.Vulkan;

using var gpu = new VulkanFloatBackend(cacheBytes: 4L * 1024 * 1024 * 1024);
asr.FloatBackend = gpu;
var gpuResult = asr.Transcribe(
    AudioFile.Resample(samples, rate, 16000), maxTokens: 128);
```

The input to inference is mono float32 PCM at 16 kHz. Keep the backend alive until all inference using it has finished. Use each model instance serially; independent sessions can share read-only weights through the documented APIs.

## Model and precision scope

Supported configurations include the fixed F16/FP32 encoder with Q8_0/FP32 decoder, FSMN VAD, and the documented Huayan/AISHELL3 VITS models. Weights, lexicons and other model assets must be obtained separately under their own licenses. See [model preparation](https://github.com/donet-org-cn/FunAsrNet/blob/main/doc/models.md).

The original BF16/FP32 ASR weights can be expanded losslessly to FP32 GGUF. The existing F16 decoder KV cache remains in use. GPU FP32 reduction order can differ from CPU; accuracy tolerances and CPU/GPU measurements are documented in the [performance report](https://github.com/donet-org-cn/FunAsrNet/blob/main/doc/cpu-gpu-round2-20261111.md).

For TTS, real-time WebSocket examples, deployment, diagnostic switches and known limits, see the [repository documentation](https://github.com/donet-org-cn/FunAsrNet#readme).

Code is Apache-2.0. The package includes its license and third-party notices; the Vulkan package also includes the vendored binding license.
