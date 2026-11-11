#!/usr/bin/env python3
"""将固定 Fun-ASR-Nano checkpoint 的 BF16/FP32 原值无损展开为 FP32 GGUF。

沿用匹配 GGUF 模板的元数据、分词器和张量名，逐张量核对形状。
仅作为离线开发工具，需要 Python 3.11+、PyTorch 和 NumPy。
生产推理不依赖这些工具。
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
DECODER_NAMES = {
    "token_embd.weight": "llm.model.embed_tokens.weight",
    "output.weight": "llm.lm_head.weight",
    "output_norm.weight": "llm.model.norm.weight",
}
LAYER_NAMES = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def read_string(stream):
    """读取 GGUF 的长度前缀 UTF-8 字符串。"""
    length = struct.unpack("<Q", stream.read(8))[0]
    return stream.read(length).decode("utf-8")


def skip_value(stream, kind):
    """跳过元数据值，保留后续对原始字节的完整复制。"""
    if kind == 8:
        read_string(stream)
    elif kind == 9:
        element, count = struct.unpack("<IQ", stream.read(12))
        for _ in range(count):
            skip_value(stream, element)
    else:
        stream.seek(SCALAR_BYTES[kind], 1)


def encode_string(value):
    """编码 GGUF 的 UTF-8 字符串及八字节长度。"""
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def read_template(path):
    """读取匹配模板的元数据和形状，将文件精度标记改为 F32。"""
    with path.open("rb") as stream:
        magic, version, tensor_count, metadata_count = struct.unpack("<IIQQ", stream.read(24))
        if magic != 0x46554747 or version not in (2, 3):
            raise ValueError(f"不是支持的 GGUF 模板 {path}")
        metadata = []
        for _ in range(metadata_count):
            start = stream.tell()
            name = read_string(stream)
            kind = struct.unpack("<I", stream.read(4))[0]
            skip_value(stream, kind)
            end = stream.tell()
            stream.seek(start)
            raw = stream.read(end - start)
            if name == "general.alignment" and (kind != 4 or struct.unpack("<I", raw[-4:])[0] != 32):
                raise ValueError("此固定模型导出工具要求 32 字节 GGUF 对齐。")
            if name == "general.file_type":
                raw = encode_string(name) + struct.pack("<II", 4, 0)
            metadata.append(raw)
        tensors = []
        for _ in range(tensor_count):
            name = read_string(stream)
            dimensions = struct.unpack("<I", stream.read(4))[0]
            shape = struct.unpack("<" + "Q" * dimensions, stream.read(8 * dimensions))
            stream.read(12)  # 原张量类型及偏移，输出统一使用 F32 并重新计算偏移。
            tensors.append((name, shape))
    return version, metadata, tensors


def checkpoint_name(name, decoder):
    """按固定 Qwen3 架构映射权重名称，未知张量直接报错。"""
    if not decoder:
        return name
    if name in DECODER_NAMES:
        return DECODER_NAMES[name]
    prefix, layer, part, kind = name.split(".")
    if prefix != "blk":
        raise ValueError(f"不支持的解码器张量 {name}")
    return f"llm.model.layers.{layer}.{LAYER_NAMES[part]}.{kind}"


def export_model(state, template, destination, decoder):
    """先核对全部张量，再按原顺序无损展开写入，不覆盖已有文件。"""
    version, metadata, tensors = read_template(template)
    entries = []
    offset = 0
    for name, shape in tensors:
        key = checkpoint_name(name, decoder)
        tensor = state[key]
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"不能保证无损展开 {key} 的 {tensor.dtype}")
        expected = (shape[0], 1, shape[1]) if name.endswith("fsmn_block.weight") else tuple(reversed(shape))
        if tuple(tensor.shape) != expected:
            raise ValueError(f"模板与 checkpoint 形状不匹配 {name}: {tuple(tensor.shape)} / {expected}")
        entries.append((name, shape, key, offset))
        offset += (int(np.prod(shape)) * 4 + 31) // 32 * 32
    with destination.open("xb") as stream:
        stream.write(struct.pack("<IIQQ", 0x46554747, version, len(entries), len(metadata)))
        stream.write(b"".join(metadata))
        for name, shape, _, offset in entries:
            stream.write(encode_string(name))
            stream.write(struct.pack("<I", len(shape)))
            stream.write(struct.pack("<" + "Q" * len(shape), *shape))
            stream.write(struct.pack("<IQ", 0, offset))
        stream.write(bytes((-stream.tell()) % 32))
        data_start = stream.tell()
        for name, _, key, offset in entries:
            tensor = state[key].detach().float()
            if name.endswith("fsmn_block.weight"):
                tensor = tensor[:, 0, :].T
            array = tensor.contiguous().numpy()
            stream.seek(data_start + offset)
            stream.write(array.tobytes())
            del array, tensor
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(destination),
        "sha256": digest,
        "bytes": destination.stat().st_size,
        "tensors": len(entries),
        "source": "official BF16/FP32 checkpoint losslessly expanded to FP32",
    }


def main():
    """从命令行选择 checkpoint、匹配的 GGUF 模板及全新输出目录。"""
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True,
                        help="含匹配的 funasr-encoder-f16.gguf 与 qwen3-0.6b-q8_0.gguf 的目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = args.output / "funasr-encoder-original-f32.gguf"
    decoder = args.output / "qwen3-original-f32.gguf"
    if encoder.exists() or decoder.exists():
        parser.error("输出模型已经存在，请使用新目录。")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    results = [
        export_model(state, args.templates / "funasr-encoder-f16.gguf", encoder, False),
        export_model(state, args.templates / "qwen3-0.6b-q8_0.gguf", decoder, True),
    ]
    manifest = {
        "checkpoint": str(args.checkpoint),
        "models": results,
        "operation": "BF16/FP32 to FP32, no quantization; matching template metadata",
    }
    (args.output / "conversion.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
