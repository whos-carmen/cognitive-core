#!/usr/bin/env python3
"""Convert GGUF to HuggingFace safetensors.

Handles the tensor transpose GGUF uses vs HF convention, copies tokenizer files.

Usage:
    python scripts/gguf_to_hf.py <input.gguf> <output_dir> [--tokenizer-dir <dir>]

Requirements (install in a venv):
    pip install gguf safetensors numpy
"""
import argparse, json, os, sys, shutil
import numpy as np

GGUF_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "chat_template.jinja",
]


def map_name(name):
    if name in GGUF_TO_HF:
        return GGUF_TO_HF[name]
    if name.startswith("blk."):
        parts = name.split(".", 2)
        rest = parts[2]
        if rest in LAYER_MAP:
            return f"model.layers.{parts[1]}.{LAYER_MAP[rest]}"
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--tokenizer-dir", help="Dir with tokenizer files")
    parser.add_argument("--no-transpose", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    from gguf import GGUFReader
    import safetensors.numpy as st_np

    print(f"Reading {args.input}...")
    reader = GGUFReader(args.input)

    # Extract metadata
    metadata = {}
    for k, v in reader.fields.items():
        if isinstance(v, (str, int, float, bool)):
            metadata[k] = v
        elif hasattr(v, "parts"):
            try:
                val = v.parts[v.data]
                if len(val) > 0:
                    raw = val[0]
                    if hasattr(raw, "tolist"):
                        metadata[k] = raw.tolist()
                    elif hasattr(raw, "decode"):
                        metadata[k] = raw.decode("utf-8")
                    else:
                        metadata[k] = raw
            except (IndexError, AttributeError):
                pass

    arch = metadata.get("general.architecture", "llama")
    print(f"Architecture: {arch}")

    # Convert tensors
    tensors = {}
    print(f"Extracting {len(reader.tensors)} tensors...")
    for i, tensor in enumerate(reader.tensors):
        hf_name = map_name(tensor.name)
        shape = [int(s) for s in tensor.shape]

        if tensor.tensor_type == 1:
            arr = np.frombuffer(tensor.data, dtype=np.float16).reshape(shape)
        elif tensor.tensor_type == 0:
            arr = np.frombuffer(tensor.data, dtype=np.float32).reshape(shape).astype(np.float16)
        else:
            print(f"  Skipping {tensor.name} (type {tensor.tensor_type})")
            continue

        if len(shape) == 2 and not args.no_transpose:
            arr = arr.T

        tensors[hf_name] = arr
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(reader.tensors)}] {hf_name}: {arr.shape}")

    print(f"Total tensors: {len(tensors)}")

    # Validate
    critical = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    missing = [t for t in critical if t not in tensors]
    if missing:
        print(f"ERROR: Missing critical tensors: {missing}")
        sys.exit(1)

    num_layers = int(metadata.get(f"{arch}.block_count", 24))
    layers = {int(n.split(".")[2]) for n in tensors if n.startswith("model.layers.")}
    if set(range(num_layers)) - layers:
        print(f"WARNING: Missing layers: {sorted(set(range(num_layers)) - layers)}")
    else:
        print(f"All {num_layers} layers present")

    # Save
    os.makedirs(args.output, exist_ok=True)
    st_path = os.path.join(args.output, "model.safetensors")
    st_np.save_file(tensors, st_path)
    print(f"Saved: {st_path} ({os.path.getsize(st_path) / 1024**3:.2f} GB)")

    # config.json
    config = {
        "_name_or_path": "gguf-conversion",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": int(metadata.get(f"{arch}.bos_token_id", 0)),
        "eos_token_id": int(metadata.get(f"{arch}.eos_token_id", 1)),
        "hidden_act": "silu",
        "hidden_size": int(metadata.get(f"{arch}.embedding_length", 1536)),
        "initializer_range": 0.02,
        "intermediate_size": int(metadata.get(f"{arch}.feed_forward_length", 4608)),
        "max_position_embeddings": int(metadata.get(f"{arch}.context_length", 131072)),
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": int(metadata.get(f"{arch}.attention.head_count", 16)),
        "num_hidden_layers": num_layers,
        "num_key_value_heads": int(metadata.get(f"{arch}.attention.head_count_kv", 2)),
        "pad_token_id": int(metadata.get(f"{arch}.padding_token_id", 1)),
        "rms_norm_eps": float(metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_parameters": {
            "rope_theta": float(metadata.get(f"{arch}.rope.freq_base", 5000000.0)),
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
        "transformers_version": "4.45.0",
        "use_cache": True,
        "vocab_size": int(metadata.get(f"{arch}.vocab_size", 130560)),
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Tokenizer
    tok_dir = args.tokenizer_dir
    if not tok_dir:
        cand = os.path.join(os.path.dirname(args.input), "tokenizer")
        if os.path.isdir(cand):
            tok_dir = cand
    if tok_dir:
        n = 0
        for fname in TOKENIZER_FILES:
            src = os.path.join(tok_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.output, fname))
                n += 1
        print(f"Copied {n} tokenizer files from {tok_dir}")
    else:
        print("No tokenizer dir found — copy tokenizer files manually")

    print(f"Done! Model saved to {args.output}")


if __name__ == "__main__":
    main()
