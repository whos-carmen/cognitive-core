#!/usr/bin/env python3
"""Convert GGUF to HuggingFace safetensors using the gguf package.

Usage:
    python scripts/gguf_to_hf.py <input.gguf> <output_dir>
"""
import argparse, json, os, sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Convert GGUF to HuggingFace safetensors")
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    from gguf import GGUFReader
    import safetensors.numpy as st_np

    print(f"Reading {args.input}...")
    reader = GGUFReader(args.input)

    # Get metadata
    metadata = {}
    for k, v in reader.fields.items():
        if isinstance(v, (str, int, float, bool)):
            metadata[k] = v
        elif hasattr(v, 'parts'):
            # Could be a string or array field
            try:
                val = v.parts[0]
                if hasattr(val, 'tolist'):
                    metadata[k] = val.tolist()
                else:
                    metadata[k] = val
            except (IndexError, AttributeError):
                pass

    arch = metadata.get('general.architecture', 'llama')
    print(f"Architecture: {arch}")
    print(f"Metadata keys: {list(metadata.keys())[:10]}...")

    # Extract tensors with GGUF -> HF name mapping
    GGUF_TO_HF_NAMES = {
        'token_embd.weight': 'model.embed_tokens.weight',
        'output_norm.weight': 'model.norm.weight',
        'output.weight': 'lm_head.weight',
    }

    def map_name(name):
        if name in GGUF_TO_HF_NAMES:
            return GGUF_TO_HF_NAMES[name]
        if name.startswith('blk.'):
            parts = name.split('.', 2)
            layer_idx = parts[1]
            rest = parts[2]
            rest_mapped = {
                'attn_norm.weight': 'input_layernorm.weight',
                'attn_q.weight': 'self_attn.q_proj.weight',
                'attn_k.weight': 'self_attn.k_proj.weight',
                'attn_v.weight': 'self_attn.v_proj.weight',
                'attn_output.weight': 'self_attn.o_proj.weight',
                'ffn_norm.weight': 'post_attention_layernorm.weight',
                'ffn_gate.weight': 'mlp.gate_proj.weight',
                'ffn_up.weight': 'mlp.up_proj.weight',
                'ffn_down.weight': 'mlp.down_proj.weight',
            }
            if rest in rest_mapped:
                return f'model.layers.{layer_idx}.{rest_mapped[rest]}'
        return name

    tensors = {}
    unmapped_tensors = []
    print(f"\nExtracting {len(reader.tensors)} tensors...")
    for i, tensor in enumerate(reader.tensors):
        name = map_name(tensor.name)
        if name == tensor.name and not name.startswith(('model.', 'lm_head')):
            unmapped_tensors.append(tensor.name)
        # Convert to numpy array
        arr = tensor.data.astype(np.float16)
        tensors[name] = arr
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(reader.tensors)}] {name}: {arr.shape}")

    print(f"\nTotal tensors: {len(tensors)}")

    # Validation: warn about unmapped tensors
    if unmapped_tensors:
        print(f"\n⚠️  WARNING: {len(unmapped_tensors)} tensors could not be mapped to HF names:")
        for t in unmapped_tensors[:10]:
            print(f"    - {t}")
        if len(unmapped_tensors) > 10:
            print(f"    ... and {len(unmapped_tensors) - 10} more")
        print("  This may indicate an unsupported architecture. The converted model may be incomplete.")

    # Validation: check for critical tensors
    critical = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
    missing_critical = [t for t in critical if t not in tensors]
    if missing_critical:
        print(f"\n❌ ERROR: Missing critical tensors: {missing_critical}")
        print("  The GGUF file may use a non-Llama architecture.")
        print("  Check the GGUF metadata for 'general.architecture' and extend map_name() accordingly.")
        sys.exit(1)

    # Check layer completeness
    num_layers = int(metadata.get(f'{arch}.block_count', 24))
    layers_found = set()
    for name in tensors:
        if name.startswith('model.layers.'):
            layer_idx = int(name.split('.')[2])
            layers_found.add(layer_idx)
    expected_layers = set(range(num_layers))
    missing_layers = expected_layers - layers_found
    if missing_layers:
        print(f"\n⚠️  WARNING: Missing layers: {sorted(missing_layers)[:10]}...")
    else:
        print(f"✓ All {num_layers} layers found")

    # Save safetensors
    os.makedirs(args.output, exist_ok=True)
    safetensors_path = os.path.join(args.output, "model.safetensors")
    st_np.save_file(tensors, safetensors_path)
    print(f"Saved safetensors: {safetensors_path} ({os.path.getsize(safetensors_path) / 1024**3:.2f} GB)")

    # Build config.json
    config = {
        "_name_or_path": f"gguf-conversion",
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": int(metadata.get(f'{arch}.bos_token_id', 0)),
        "eos_token_id": int(metadata.get(f'{arch}.eos_token_id', 1)),
        "hidden_act": "silu",
        "hidden_size": int(metadata.get(f'{arch}.embedding_length', 1536)),
        "initializer_range": 0.02,
        "intermediate_size": int(metadata.get(f'{arch}.feed_forward_length', 4608)),
        "max_position_embeddings": int(metadata.get(f'{arch}.context_length', 131072)),
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": int(metadata.get(f'{arch}.attention.head_count', 16)),
        "num_hidden_layers": int(metadata.get(f'{arch}.block_count', 24)),
        "num_key_value_heads": int(metadata.get(f'{arch}.attention.head_count_kv', 2)),
        "pad_token_id": int(metadata.get(f'{arch}.padding_token_id', 1)),
        "rms_norm_eps": float(metadata.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6)),
        "rope_parameters": {
            "rope_theta": float(metadata.get(f'{arch}.rope.freq_base', 5000000.0)),
            "rope_type": "default"
        },
        "tie_word_embeddings": False,
        "transformers_version": "4.57.6",
        "use_cache": False,
        "vocab_size": int(metadata.get(f'{arch}.vocab_size', 130560)),
        "torch_dtype": "float16"
    }

    config_path = os.path.join(args.output, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    print(f"\nDone! Model saved to {args.output}")


if __name__ == "__main__":
    main()
