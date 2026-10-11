# 固定模型文件

模型文件不提交 Git。准备以下权重及配置，放在仓库根目录 `models/`；不要替换为同名但不同量化精度的文件。ASR/VAD 的兼容 GGUF 暂未提供公开下载镜像和一键转换工具；需要自行准备与固定布局、散列相符的权重。只有代码时可以先运行无模型数值检查。

| 文件 | 字节数 | 用途 |
|---|---:|---|
| `funasr-encoder-f16.gguf` | 469331008 | Fun-ASR 编码器，F16 源权重 |
| `qwen3-0.6b-q8_0.gguf` | 804753280 | Qwen3 解码器，Q8_0 |
| `fsmn-vad.gguf` | 1720512 | FSMN 语音活动检测 |
| `zh_CN-huayan-medium.onnx` | 63201423 | 单发音人中文 VITS 合成 |
| `zh_CN-huayan-medium.onnx.json` | 4822 | 音素 ID 表及模型配置 |
| `MODEL_CARD` | 276 | 合成模型的来源说明 |

SHA-256：

```text
f92f91d01a24fbed6c863495b2ee8c6a6788144a02858b75743f0946668de8a2  funasr-encoder-f16.gguf
819f385dc0e035dccc3d9e7edaf6b7b044b8ba7ace63cbcbf84c7e397eecbf27  qwen3-0.6b-q8_0.gguf
1270f2559c495f4e7b6e739541151027d360761a3fda43fc147034f5719f5479  fsmn-vad.gguf
04234ec24907e9efa7e153669e07d4e0cf6d260adc005323aa929386dae53f86  zh_CN-huayan-medium.onnx
d521dc45504a8ccc99e325822b35946dd701840bfb07e3dbb31a40929ed6a82b  zh_CN-huayan-medium.onnx.json
25d7d8f7a03e9382e629c5c4f074176e7ca84e08be12a873022e91fcac98c2c7  MODEL_CARD
```

仓库内发音词典 `assets/mandarin-lexicon.json` 为 459108 字节，SHA-256 为 `646fbddd901e016d27a500b9cd1c11a9466a03a0d1871fd541d4ed9b0397002d`。它是发音数据，不是执行依赖；来源与局限见同目录的 `lexicon-provenance.json`。

Huayan ONNX、JSON 和模型卡来源为 [piper-voices 的中文 Huayan medium](https://huggingface.co/rhasspy/piper-voices/tree/main/zh/zh_CN/huayan/medium)。下载后核对上表散列，保存原模型卡；浮动 main 文件若变化不应当作本次固定版本。GGUF 架构来源为固定 Fun-ASR/llama.cpp，见第三方说明；这不是把普通 Hugging Face 原始权重改后缀即可使用。

核心架构来自 QwenAudio/Fun-ASR；GGUF/Qwen3 布局、ONNX 算子子集和发音资产来源见 [第三方说明](../THIRD_PARTY_NOTICES.md)。Huayan 原模型卡的训练数据许可字段为 Unknown；本仓库没有修改该声明或将模型许可重新解释为代码许可。

## 可选 AISHELL3 多发音人模型

以下四个文件放在 `models/aishell3/`。如果没有 `model.onnx`，服务只提供 Huayan 音色；模型存在但配套文件缺失会报告启动错误，避免展示不可用的发音人。

来源：[k2-fsa 官方模型说明](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-icefall-zh-aishell3-chinese-174-speakers)。[下载固定名称归档](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2)。归档解压时仅取 `model.onnx`、`lexicon.txt`、`tokens.txt`、`speakers.txt`，放在模型根目录的 `aishell3` 子目录；不需要 `.fst/.far` 或任何上游运行库。模型约 29 MiB，174 位发音人共享同一份权重。

SHA-256（本次实测版本）：

```text
5511d651b7840c0a93a6bbfd4afd070a2c7f39ca1ec3ff2ecd73191519bbb852  aishell3/model.onnx
ab2e61d357551e7b24ddd965d924aca784c20165ff58c150794e539c6b5e9e35  aishell3/lexicon.txt
50b45a7b7de1752fd3c7b4755661c285f1547f59186eca2281089a81307ad953  aishell3/tokens.txt
50a825fc9a89172eee04a4d68869d933d0eca0d847f3b6fcffff9514cc72442d  aishell3/speakers.txt
```

模型声明 `n_speakers=174`、`sample_rate=8000`、`language=Chinese`。输入为 `tokens`、`tokens_lens`、`noise_scale`、`alpha`、`noise_scale_dur`、`speaker`；其中 `alpha=1/speed`，`speaker` 为 0–173。不将 Huayan 的 IPA 词典用于此模型。部署时应同时核对四个文件，保证编号对应正确的音素和匿名发音人。

离线检查和生成示例：

```text
dotnet run --project src/FunAsrNet.Cli -c Release -- inspect-onnx models/aishell3/model.onnx
dotnet run --project src/FunAsrNet.Cli -c Release -- tts --model models/aishell3/model.onnx --lexicon models/aishell3/lexicon.txt --tokens models/aishell3/tokens.txt --speaker 50 --text 您好，欢迎致电。 --rate 8000 --out speech.wav
```
