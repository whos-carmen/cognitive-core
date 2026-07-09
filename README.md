# Cognitive Core — Local 1B Agentic Router

A project to build Andrej Karpathy's "Cognitive Core" concept: a small, always-on local model that handles routing, tool calling, and delegation to larger cloud models for knowledge it lacks.

## Table of Contents

1. [Vision & Concept](#1-vision--concept)
2. [Model Research Summary](#2-model-research-summary)
3. [MiniCPM5-1B Technical Deep Dive](#3-minicpm5-1b-technical-deep-dive)
4. [Variant Comparison](#4-variant-comparison)
5. [Target Hardware](#5-target-hardware)
6. [Environment Setup](#6-environment-setup)
7. [Model Combination Pipeline](#7-model-combination-pipeline)
8. [Post-Combination: GGUF Deployment](#8-post-combination-gguf-deployment)
9. [Uncertainty Detection & Routing (Future Work)](#9-uncertainty-detection--routing-future-work)
10. [Appendix: Key Resources](#10-appendix-key-resources)

---

## 1. Vision & Concept

### Karpathy's "Cognitive Core" (June 2025)

Source: https://x.com/karpathy/status/1938626382248149433

> "The race for LLM 'cognitive core' — a few billion param model that maximally sacrifices encyclopedic knowledge for capability. It lives always-on and by default on every computer as the kernel of LLM personal computing."

Key features described:
- Natively multimodal (text/vision/audio at both input and output)
- Matryoshka-style architecture (dial capability up/down at test time)
- Reasoning with a dial (system 2 thinking)
- Aggressively tool-using
- On-device finetuning LoRA slots for personalization
- Delegates and double-checks with cloud oracles when internet is available

> "It doesn't know that William the Conqueror's reign ended in September 9 1087, but it vaguely recognizes the name and can look up the date. It can't recite the SHA-256 of empty string as e3b0c442..., but it can calculate it quickly should you really want it."

### Our Implementation Scope

For this project, we are focusing on:
- **Text-only** CLI/chat environment (no multimodal needed)
- **Tool calling & delegation** as the primary capability
- **Hybrid reasoning** (fast mode + deep `<think>` mode)
- **Combining two fine-tuned variants** for optimal behavior
- A routing framework that delegates knowledge queries to cloud oracles (to be built later)

---

## 2. Model Research Summary

### Why a Small Model?

A 1B model is not competing with GPT-4. It serves a fundamentally different purpose:

| Advantage | Detail |
|---|---|
| Zero ongoing cost | No API keys, subscriptions, or per-query fees |
| Full privacy | Data never leaves the machine |
| Zero latency | No network round-trip |
| Always-on | Runs locally 24/7 without cloud dependency |
| Fits anywhere | CPU with 4GB RAM, phones, edge devices |

### What Makes MiniCPM5-1B Special at 1B Scale

MiniCPM5-1B achieves 1B-class open-source SOTA (average 42.57 vs 35.61 for next-best) through a full-stack approach:

- **UltraData L0-L4 tiered data management**: LLMs themselves curate training data via quality scoring and content editing
- **400B tokens of SFT** (200B deep-thinking + 200B hybrid-thinking)
- **Domain-specialized RL teachers** for math, code, QA, writing
- **On-Policy Distillation (OPD)**: Dense token-level supervision via reverse KL divergence, 50-100x more compute-efficient than pure RL. Delivers +16 points average benchmark score improvement
- **Aggressive GQA (16:2)**: Extremely memory-efficient KV cache for 131K context on limited hardware

### Hybrid Reasoning Mechanism

A single checkpoint serves two modes:
- **Fast mode** (`enable_thinking=False`): Direct response, minimal latency
- **Deep mode** (`enable_thinking=True`): Internal `<think>...</think>` chain-of-thought before the answer

The mode switch is prompt-driven (special tokens in chat template), not a separate model.

---

## 3. MiniCPM5-1B Technical Deep Dive

### Architecture

| Parameter | Value |
|---|---|
| Architecture | `LlamaForCausalLM` |
| Total params | 1,080,632,832 |
| Non-embedding params | 679,552,512 |
| Layers | 24 |
| Hidden size | 1536 |
| Intermediate size | 4608 |
| Query heads (GQA) | 16 |
| KV heads | 2 |
| Head dim | 128 |
| RoPE theta | 5,000,000 |
| Max position embeddings | 131,072 |
| Vocab size | 130,560 |
| Activation | SiLU |
| Norm | RMSNorm (eps=1e-6) |

Key design choices:
- **16:2 GQA ratio** — 8 queries share each KV head. Dramatically reduces KV cache memory, allowing 131K context on limited hardware.
- **RoPE theta 5M** — 500x larger than LLaMA default (10K), provides frequency headroom for 131K context without exotic techniques.
- **Standard LlamaForCausalLM** — compatible with every major inference engine out of the box.

### Training Pipeline

```
1. Base Training
   └─ UltraData L0-L4 tiered corpus (Ultra-FineWeb, UltraData-Math)
   └─ Stable training + decay training

2. Mid-Training
   └─ Continue with tiered data to adapt to target distribution

3. SFT (400B tokens total)
   ├─ 200B tokens deep-thinking SFT (always produces <think> chains)
   └─ 200B tokens hybrid-thinking SFT (mixed think/no-think)

4. RL Teachers (domain-specialized)
   ├─ Math: DAPO-Math-17k, two-stage length schedule
   ├─ Code, QA (TriviaQA, NQ-Open)
   ├─ Writing (LongWriter-Zero-RLData)
   └─ General (pairwise RLHF)

5. On-Policy Distillation (OPD)
   └─ Distill all teachers into single student
   └─ Student generates own rollouts (on-policy)
   └─ Dense token-level signal via reverse KL divergence
   └─ 50-100x more compute-efficient than pure RL
```

### On-Policy Distillation (OPD) — The Key Innovation

Traditional distillation: Teacher generates data → Student learns off-policy
On-Policy Distillation: Student generates its own trajectories → Teacher scores them

At every token position:
1. Take top-k logits from both student and teacher
2. Compute reverse KL divergence on the union of token sets
3. This provides continuous token-by-token supervision

Results: +16 points average score, -29% reduction in overlong responses.

---

## 4. Variant Comparison

Three community variants of MiniCPM5-1B exist, each forked from the same base weights:

### Abiray/MiniCPM5-1B-GGUF
- **What**: Vanilla format conversion. Stock `openbmb/MiniCPM5-1B` → GGUF.
- **Training**: None — just format conversion.
- **Strength**: Balanced generalist, full OpenBMB benchmark scores.
- **GGUF options**: Q4_K_M (657MB), Q5_K_M (751MB), Q6_K (851MB), Q8_0 (1.1GB), F16 (2.1GB).

### Luminia/MiniCPM5-1B-Agent-GGUF
- **What**: Abliterated + fine-tuned for tool calling.
- **Training pipeline**:
  1. Abliterate base model (removes safety refusals / over-caution)
  2. SFT on 45,762 curated training rows from 26 source datasets
  3. DPO with 649 on-policy pairs (chosen = valid tool call, rejected = rambles in <think>)
- **Strength**: Acts more, stalls less. Best tool calling at 1B scale.
- **GGUF options**: Q8_0 (1.1GB), F16 (2.1GB).

### GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking
- **What**: Fine-tuned on leaked Claude Opus "Fable 5" reasoning traces.
- **Training**: Fine-tuned on Fable 5 data (real reasoning traces from Claude Opus with Chain-of-Thought).
- **Strength**: Claude-style structured reasoning, better code ability, better instruction following.
- **GGUF options**: Q4_K_M (657MB), Q5_K_M (751MB), Q8_0 (1.1GB), F16 (2.1GB).

### Tradeoff Triangle

```
        STOCK (Abiray)
           /\
          /  \
    Balanced  Balanced
    General   Quality
        /      \
       /        \
AGENT (Luminia)  CLAUDE DISTILL (GnLOLot)
   Tool calling    Reasoning depth
   Less stalling   Claude-style thinking
   Action-first    Better at code
```

### GGUF Quantization Reference

| Format | Size | Quality | Speed (token gen) | Use case |
|---|---|---|---|---|
| F16 | 2.1 GB | Baseline | Slowest (2.8 tok/s) | Fine-tuning base only |
| Q8_0 | 1.1 GB | Near-identical | Fastest (5.0 tok/s) | Recommended default |
| Q6_K | 851 MB | Near-identical | Very fast | Best quality-per-byte |
| Q5_K_M | 751 MB | Very good | Fast | Memory-constrained |
| Q4_K_M | 657 MB | Good | Fast | Mobile / minimal RAM |

At 1B scale, Q8_0 is both smaller AND faster than F16 (memory-bandwidth bound). Q6_K offers the best quality-per-byte.

---

## 5. Target Hardware

### Training Hardware (Lambda Labs)
- **GPU**: 1x NVIDIA H100 PCIe (80 GB VRAM)
- **CPU**: 26 vCPUs
- **RAM**: 200 GB
- **Storage**: 1 TB NVMe
- **Cost**: $3.29/hr
- **OS**: Ubuntu 24.04

### Local Dev Machine Specs
- **CPU**: AMD Ryzen 9 7900X (12 cores / 24 threads)
- **GPU**: AMD Radeon RX 7900 XTX (24GB VRAM, RDNA 3 / gfx1100)
- **OS**: Ubuntu (bare metal)

### VRAM Budget

| Task | VRAM Needed | H100 (80GB) | 7900 XTX (24GB) |
|---|---|---|---|
| GGUF→HF conversion | ~4 GB | ✅ Easy | ✅ Easy |
| TIES merge | ~8 GB | ✅ Easy | ✅ Easy |
| SFT (full, 1B model) | ~10-14 GB | ✅ Easy | ✅ Fits |
| SFT (batch 4, no grad-ckpt) | ~30 GB | ✅ Easy | ❌ Too big |
| DPO (full fine-tune) | ~18-22 GB | ✅ Easy | ✅ Tight but fits |
| GGUF conversion | ~4 GB | ✅ Easy | ✅ Easy |

---

## 6. Environment Setup

### Lambda Labs Cloud Setup (current training)

1. Create account at lambdalabs.com, add SSH key
2. Launch instance:
   ```bash
   # Via API
   curl -X POST -H "Authorization: Bearer $LAMBDA_AI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"region_name":"us-west-3","instance_type_name":"gpu_1x_h100_pcie",
          "ssh_key_names":["cognitive-core"],"quantity":1}' \
     https://cloud.lambdalabs.com/api/v1/instance-operations/launch
   ```
3. SSH in: `ssh -i ~/.ssh/lambda_ai ubuntu@<ip>`
4. Set up tools:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv ~/venv
   ln -sf ~/venv ~/.venv  # so uv run auto-discovers it
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   uv pip install transformers datasets accelerate bitsandbytes liger-kernel trl huggingface-hub mergekit gguf
   ```

### Local Development (ROCm, bare metal)

1. **Install ROCm**: `sudo apt install -y rocm-dev rocm-hip-sdk`
2. **Set GPU arch**: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
3. **Build Docker image**: `docker build -t cognitive-core:latest .`
4. **Run container**: `bash scripts/launch_container.sh`

---

## 7. Model Combination Pipeline

### Strategy

We combine **GnLOLot** (Claude-style reasoning) with **Luminia** (tool-calling reinforcement). The order matters:

```
Base (openbmb/MiniCPM5-1B)
  → GnLOLot adds: Claude-style structured reasoning
    → Luminia's training recipe adds: tool-calling SFT + DPO on top
```

This preserves Claude's thinking patterns while adding aggressive tool-calling behavior.

### Pipeline Overview

```
Stage 0: Download models (GnLOLot HF, Luminia GGUF)                          ✅ Done
Stage 1: Convert Luminia GGUF → HF safetensors                               ✅ Done
Stage 2: TIES merge (GnLOLot + Luminia HF) → merged checkpoint                ✅ Done
Stage 3: SFT — teach tool calling (3 epochs, H100-optimized)                  🟡 Running
Stage 4: DPO — reinforce acting over stalling                                 ⬜ Next
Stage 5: GGUF conversion for deployment                                       ⬜ Final
```

### Stage 0: Download Models and Training Data ✅

```bash
cd /workspace

# Models (already downloaded)
ls models/Luminia-MiniCPM5-1B-Agent-GGUF/   # GGUF files + training code + data
ls models/GnLOLot-MiniCPM5-1B-Claude-Opus-Fable5-Thinking/  # HF safetensors
ls models/openbmb-MiniCPM5-1B/               # Base model (HF)
```

### Stage 1: Convert Luminia GGUF → HF Safetensors ✅

```bash
python scripts/gguf_to_hf.py \
    models/Luminia-MiniCPM5-1B-Agent-GGUF/MiniCPM5-1B-Agent-v4-f16.gguf \
    models/Luminia-MiniCPM5-1B-Agent-HF
```

### Stage 2: TIES Merge (GnLOLot + Luminia HF)

```bash
pip install mergekit

cat > /workspace/merge_test.yaml << 'EOF'
models:
  - model: /workspace/models/GnLOLot-MiniCPM5-1B-Claude-Opus-Fable5-Thinking
    parameters:
      weight: 0.55
  - model: /workspace/models/Luminia-MiniCPM5-1B-Agent-HF
    parameters:
      weight: 0.45
merge_method: ties
base_model: /workspace/models/Luminia-MiniCPM5-1B-Agent-HF
parameters:
  normalize: true
  weight: 1.0
dtype: bfloat16
EOF

mergekit-yaml /workspace/merge_test.yaml /workspace/models/merged --cuda
```

**Why TIES over SLERP**: TIES identifies which parameters each model changed differently from the base and preserves only meaningful divergences. Better for models with distinct capabilities (reasoning vs tool calling) than simple interpolation.

**Why weight 0.55/0.45 favoring GnLOLot**: We want to preserve Claude's reasoning patterns as the foundation and layer tool-calling behavior on top. Luminia's base (abliterrated) also strips refusals, which we want but don't need to weight as heavily.

### Stage 3: SFT — Teach Tool Calling (H100 Optimized)

Training on Lambda Labs 1x H100 PCIe with 80GB VRAM:

```bash
# Smoke test (5 steps)
bash run_training.sh smoke-h100

# Full 3-epoch SFT (~2-3 hours)
bash run_training.sh full-h100
```

**H100-optimized settings** (`--bsz 4 --accum 6 --no-grad-ckpt --optim adamw`):
- Batch size 4 (vs 1 on 24GB cards) — 4x tokens per step
- Gradient checkpointing disabled — ~2-3x faster forward/backward
- Full AdamW (not 8-bit) — better convergence
- Effective batch: 24 (4 × 6)
- Expected: **~12-15 it/s**, 3 epochs in ~2-3 hours

**What this does**: Takes the merged checkpoint (Claude reasoning + Luminia tool-calling patterns) and trains it on Luminia's curated tool-calling data. The model learns to emit structured `<function>` tool calls while retaining its thinking capabilities.

### Stage 4: DPO — Reinforce Acting Over Stalling

Generate on-policy preference pairs from the SFT model, then train DPO:

```bash
# Build on-policy DPO pairs
# The SFT model generates its own responses to training prompts
# Chosen = valid <function> tool call (correct format)
# Rejected = rambles in <think> or answers in prose without tool call
python models/code/data/build_prefs_onpolicy_gpu.py \
    --model /workspace/train/outputs/sft_claude_agent \
    --src /workspace/models/Luminia-MiniCPM5-1B-Agent-GGUF/dataset/train_v4.jsonl \
    --out /workspace/models/dataset/dpo_onpolicy_claude.jsonl

# DPO training
python models/code/train/dpo.py \
    --model /workspace/train/outputs/sft_claude_agent \
    --data /workspace/models/dataset/dpo_onpolicy_claude.jsonl \
    --out /workspace/train/outputs/final-cognitive-core \
    --beta 0.1 \
    --lr 1e-6 \
    --epochs 3 \
    --accum 8
```

**What this does**: Shows the model its own tool-calling successes and failures, then rewards it for acting (emitting tool calls) rather than stalling (thinking about whether to act).

**Expected time**: 2-4 hours on 7900 XTX.

**VRAM**: ~18-22 GB (tight at 24GB). If OOM:
- Reduce `--accum` from 8 to 4
- Reduce `--epochs` from 3 to 2
- Or use 4-bit LoRA instead of full fine-tune

---

## 8. Post-Combination: GGUF Deployment

### Convert Final Model to GGUF

```bash
cd /workspace

# Clone llama.cpp for conversion tools
git clone https://github.com/ggerganov/llama.cpp

# Convert to F16 (the base for quantization)
python llama.cpp/convert_hf_to_gguf.py /workspace/train/outputs/final-cognitive-core \
    --outfile /workspace/train/outputs/final-cognitive-core-f16.gguf --outtype f16

# Quantize to all useful formats
cd llama.cpp
./llama-quantize ../train/outputs/final-cognitive-core-f16.gguf ../train/outputs/final-cognitive-core-Q8_0.gguf Q8_0
./llama-quantize ../train/outputs/final-cognitive-core-f16.gguf ../train/outputs/final-cognitive-core-Q6_K.gguf Q6_K
./llama-quantize ../train/outputs/final-cognitive-core-f16.gguf ../train/outputs/final-cognitive-core-Q4_K_M.gguf Q4_K_M
```

### Files produced

| File | Size | Use |
|---|---|---|
| `final-cognitive-core-Q8_0.gguf` | ~1.1 GB | Recommended — best quality + fastest |
| `final-cognitive-core-Q6_K.gguf` | ~851 MB | Best quality-per-byte |
| `final-cognitive-core-Q4_K_M.gguf` | ~657 MB | Minimal footprint |

### Deploy with Ollama

Files are accessible from the Ubuntu VM at the mounted workspace path (`/home/$USER/cognitive-core/`).

Create a `Modelfile`:
```
FROM ./final-cognitive-core-Q8_0.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.95
```

```bash
ollama create cognitive-core -f Modelfile
ollama run cognitive-core
```

### Deploy with llama.cpp (direct)

```bash
# Inside Docker or on the Ubuntu VM
./llama-server -m /workspace/final-cognitive-core-Q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -t 8 -c 131072
```

This exposes an OpenAI-compatible API at `http://<vm-ip>:8080/v1`.

---

## 9. Uncertainty Detection & Routing (Future Work)

This section outlines the approach for teaching the model to recognize uncertainty and delegate to cloud oracles. The actual routing framework will be built in a separate phase.

### Three-Layer Uncertainty Detection

**Layer 1: Prompt Engineering** (zero training)
- System prompt instructs the model to classify confidence before answering
- Model emits structured signals: `<confidence>confident|uncertain|unknown</confidence>`

**Layer 2: Logit-Based Monitoring** (code, no training)
- Monitor token entropy during generation
- High entropy on key tokens = model is uncertain
- Can trigger delegation mid-response

```python
# Pseudocode for logit-based uncertainty
import torch

def get_uncertainty(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    top1_conf = probs.max(dim=-1).values
    top2 = probs.topk(2, dim=-1).values
    gap = top2[:, 0] - top2[:, 1]
    return entropy, top1_conf, gap
```

**Layer 3: Fine-Tuning** (training)
- Create 500-1000 examples of uncertain vs confident responses
- Fine-tune to emit reliable uncertainty signals
- Can reuse Luminia's DPO pipeline with uncertainty-aware examples

### Planned Routing Architecture

```
User Input → MiniCPM5-1B (Cognitive Core)
  ├── Confident + No tool needed → Answer directly
  ├── Tool call needed → Execute locally (code, search, calculation)
  ├── Uncertain / Knowledge gap → Delegate to cloud oracle
  ├── Verification needed → Delegate, then integrate response
  └── Complex reasoning → Engage <think> mode
```

Oracle candidates:
- GPT-4o / Claude for complex knowledge
- Search APIs for current information
- Code execution for calculations

---

## 10. Appendix: Key Resources

### Models

| Model | URL |
|---|---|
| openbmb/MiniCPM5-1B (base) | https://huggingface.co/openbmb/MiniCPM5-1B |
| Abiray/MiniCPM5-1B-GGUF (stock quantized) | https://huggingface.co/Abiray/MiniCPM5-1B-GGUF |
| Luminia/MiniCPM5-1B-Agent (tool-calling tuned) | https://huggingface.co/Luminia/MiniCPM5-1B-Agent |
| GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking | https://huggingface.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking |

### Training Frameworks

| Tool | Purpose | URL |
|---|---|---|
| Unsloth (ROCm) | SFT/DLO training on AMD GPUs | https://hub.docker.com/r/goldengrapegentleman/unsloth-rocm |
| mergekit | Weight-space model merging | https://github.com/arcee-ai/mergekit |
| llama.cpp | GGUF conversion + quantization + inference | https://github.com/ggerganov/llama.cpp |
| Ollama | Easy local deployment | https://ollama.com |

### Papers & References

| Topic | Reference |
|---|---|
| MiniCPM4 Technical Report | https://arxiv.org/abs/2506.07900 |
| UltraData L0-L4 Framework | https://arxiv.org/abs/2602.09003 |
| On-Policy Distillation | https://arxiv.org/abs/2604.13016 |
| Logits-Induced Token Uncertainty | https://arxiv.org/abs/2502.00290 |
| Quantization Evaluation (llama.cpp) | https://arxiv.org/html/2601.14277v1 |
| Karpathy Cognitive Core Tweet | https://x.com/karpathy/status/1938626382248149433 |

### Fable 5 Datasets (Leaked Claude Reasoning Traces)

| Dataset | URL |
|---|---|
| Glint-Research/Fable-5-traces | https://huggingface.co/datasets/Glint-Research/Fable-5-traces |
| armand0e/claude-fable-5-claude-code | https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code |
| HelioAI/Fable-5-Distill-Reasoning-462x | https://huggingface.co/datasets/HelioAI/Fable-5-Distill-Reasoning-462x |

> **Note**: Fable 5 data consists of leaked Anthropic internal traces. Suitable for personal experimentation but may have licensing implications for distribution.

---

## License

The base model (MiniCPM5-1B) is Apache-2.0 licensed. All derived models in this project inherit this license.
