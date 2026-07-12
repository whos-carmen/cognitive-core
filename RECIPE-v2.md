# MiniCPM5-Claude-Toolusage — Training Recipe v2

Reproducible training pipeline from base model to final tool-calling agent.

## Architecture Decision

**Start from openbmb/MiniCPM5-1B** — the common base both parents fine-tuned from.
Gives a clean slate, no merge artifacts, full control over training.

## Tool Format Decision

**Use Luminia's XML tool format:**

```
<think>I need to check the weather in Tokyo.</think>
 name="get_weather"> name="location">Tokyo name="unit">celsius
```

Rationale:
- Already supported by inference code
- Simpler than JSON wrapping
- Both parents produce this format natively
- Fable 5 traces will be converted to this format

## Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| **Luminia SFT** | `Luminia/MiniCPM5-1B-Agent-GGUF` → `dataset/train_v4.jsonl` | 45,762 | Tool-calling base (XML format) |
| **Luminia DPO** | Same repo → `dataset/dpo_onpolicy_v4.jsonl` | 649 | Preference pairs (chosen vs rejected) |
| **Fable 5 traces** | `Glint-Research/Fable-5-traces` → `fable5_cot_merged.jsonl` | 3,799 tool rows | Claude reasoning + tool call patterns |

## Training Pipeline

```
openbmb/MiniCPM5-1B (1.09B params)
    │
    ├── Phase 1: SFT — Tool Calling (full fine-tune, 1 epoch)
    │   ├── Luminia 45,762 tool-calling examples
    │   ├── + Fable 5 traces (converted to XML format)
    │   └── Result: tool-proficient model
    │
    └── Phase 2: DPO — Preference (LoRA, 3 epochs)
        ├── Luminia 649 preference pairs
        └── Result: prefers good tool calls over bad ones
```

No reasoning SFT phase needed — Fable 5 traces already include `<think>` blocks.
No GnLOLot trace generation needed — Fable 5 IS the real Claude reasoning data.

## Data Preparation

### Step 1: Download base model
```bash
export PATH="$HOME/.local/bin:$PATH"
hf download openbmb/MiniCPM5-1B --local-dir models/base
```

### Step 2: Download Luminia training data
```bash
hf download Luminia/MiniCPM5-1B-Agent-GGUF \
    --include "dataset/*" --local-dir /tmp/luminia-data
cp /tmp/luminia-data/dataset/train_v4.jsonl dataset/
cp /tmp/luminia-data/dataset/dpo_onpolicy_v4.jsonl dataset/
```

### Step 3: Download and convert Fable 5 traces
```bash
# Download Fable 5 traces
hf download Glint-Research/Fable-5-traces \
    --include "fable5_cot_merged.jsonl" \
    --local-dir /tmp/fable5
```

### Step 4: Merge and convert datasets
```python
# Convert Fable 5 traces to Luminia XML format
# Input:  fable5_cot_merged.jsonl (context + cot + output)
# Output: train_v4_fable.jsonl (messages format)

import json
from transformers import AutoTokenizer

def convert_fable5_to_messages(row, tok):
    """Convert Fable 5 trace to {messages, tools} format."""
    context = row["context"]
    cot = row["cot"]
    output = row["output"]
    output_type = row["output_type"]

    if output_type != "tool_use":
        return None

    # Build system prompt from context
    system_prompt = "You are a helpful assistant with access to tools..."

    # Build user message
    user_msg = {"role": "user", "content": context}

    # Build assistant message with think + tool call
    tool_name = output.get("name", "")
    tool_args = output.get("input", {})
    args_str = " ".join(f'name="{k}">{v}' for k, v in tool_args.items())
    tool_call_xml = f' name="{tool_name}"> {args_str}'

    assistant_msg = {
        "role": "assistant",
        "content": f"<think>\n{cot}\n</think>\n\n{tool_call_xml}"
    }

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            user_msg,
            assistant_msg
        ],
        "tools": list(tool_args.keys())
    }
```

### Step 5: Combine datasets
```bash
# Merge Luminia + Fable 5 into one training file
cat dataset/train_v4.jsonl dataset/train_v4_fable.jsonl > dataset/train_v4_combined.jsonl
echo "Combined dataset: $(wc -l < dataset/train_v4_combined.jsonl) examples"
```

## Training: Phase 1 — SFT

### Hyperparameters (verified from SFT best practices research)

| Parameter | Value | Why |
|---|---|---|
| Batch size | 4 | Fits in 24GB VRAM |
| Gradient accumulation | 6 | Effective batch = 24 |
| Epochs | **1** | Loss curve shows diminishing returns after epoch 1 |
| Learning rate | 1e-5 | Conservative, research-backed |
| LR scheduler | cosine | Standard |
| Warmup ratio | 0.05 | Standard |
| Weight decay | 0.01 | Standard |
| Max grad norm | 1.0 | Upper bound of recommended range |
| Max length | 4096 | Fits in VRAM, covers most examples |
| Precision | BF16 | Memory efficient |
| Gradient checkpointing | Yes | Saves VRAM |
| Optimizer | AdamW 8-bit | Memory efficient |
| NEFTune | 5 | Anti-overfitting (helps at 1 epoch) |

### Launch SFT (g5.2xlarge A10G, ~4.5 hrs)
```bash
bash training/scripts/run_sft.sh --pretokenized /mnt/s3/cognitive-core/dataset/train_v4_tokenized
```

### Timing by Instance

| Instance | GPU | VRAM | Est. Time | Est. Cost |
|---|---|---|---|---|
| g5.2xlarge | A10G | 24GB | ~4.5 hrs | $3.30 |
| g6e.2xlarge | L40S | 44GB | ~3 hrs | $5.80 |
| g7e.2xlarge | L40S | 96GB | ~2.5 hrs | $3.80 |

## Training: Phase 2 — DPO

### Hyperparameters

| Parameter | Value |
|---|---|
| Method | LoRA (r=16, alpha=32) |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| Learning rate | 1e-6 |
| Beta | 0.1 |
| Max length | 4096 |

### Launch DPO (any GPU, ~20 min)
```bash
bash training/scripts/run_sft.sh dpo
```

## Full Timeline

```
Phase 1: Data preparation     ~30 min    (download + convert)
Phase 2: SFT (1 epoch)         ~4.5 hrs   (g5.2xlarge)
Phase 3: DPO (LoRA)            ~20 min    (any GPU)

Total:                         ~5.5 hrs   (not 14!)
```

## Expected Results

Based on the parent models' benchmarks and our loss curve:

| Metric | Luminia | GnLOLot | Expected (v2) |
|---|---|---|---|
| Tool Accuracy | 92% | 31% | **90%+** |
| Thinking/Reasoning | None | 100% relevance | **Both** |
| Params Accuracy | 85% | 31% | **85%+** |

## Setup Script

```bash
# One-command setup on any fresh EC2 instance
git clone https://github.com/whos-carmen/cognitive-core.git && cd cognitive-core
bash training/scripts/setup_instance.sh --skip-merge

# Then run the new training
bash training/scripts/pretokenize.sh     # tokenize Luminia data
bash training/scripts/prepare_data.sh    # download + convert Fable 5
bash training/scripts/run_sft.sh full --epochs 1  # SFT
bash training/scripts/run_sft.sh dpo              # DPO
```
