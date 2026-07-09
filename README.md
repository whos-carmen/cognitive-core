# Cognitive Core

A local 1B agentic router built on MiniCPM5-1B, combining Claude Opus reasoning traces with tool-calling reinforcement to create an always-on local model that knows what it doesn't know and delegates accordingly.

Inspired by Andrej Karpathy's "cognitive core" concept — a small model that sacrifices knowledge for capability, lives always-on on every computer, and aggressively uses tools and delegation to compensate.

> "It doesn't know that William the Conqueror's reign ended in September 9 1087, but it vaguely recognizes the name and can look up the date."
> — [Karpathy, June 2025](https://x.com/karpathy/status/1938626382248149433)

---

## What This Is

The goal is a single small model that can:

1. **Handle simple tasks locally** — reasoning, calculation, tool calls, code
2. **Recognize its own uncertainty** — know when it doesn't know something
3. **Delegate to cloud oracles** — query GPT-4, Claude, or search APIs for knowledge it lacks
4. **Run always-on** — on consumer hardware, no API costs, full privacy, zero latency

This repo contains the research, training pipeline, and deployment instructions. The actual routing/orchestration framework is a future phase.

---

## The Model: MiniCPM5-1B

MiniCPM5-1B is a 1B-parameter Transformer (standard LlamaForCausalLM architecture) by [OpenBMB](https://github.com/OpenBMB/MiniCPM), achieving 1B-class open-source SOTA (average 42.57 vs 35.61 for the next-best 1B model).

### Architecture

| Parameter | Value |
|---|---|
| Architecture | LlamaForCausalLM |
| Total params | 1,080,632,832 |
| Non-embedding params | 679,552,512 |
| Layers | 24 |
| Hidden size | 1536 |
| Intermediate size | 4608 |
| Query heads / KV heads (GQA) | 16 / 2 |
| Head dim | 128 |
| RoPE theta | 5,000,000 |
| Max context | 131,072 tokens |
| Vocab size | 130,560 |

### Why This Model

- **Hybrid reasoning** — same checkpoint serves fast mode and deep `<think>` reasoning mode, toggled via `enable_thinking`
- **16:2 GQA** — 8 queries share each KV head, dramatically reducing KV cache for 131K context on limited RAM
- **Tool calling** — native XML-style `<function>` format with SGLang parser support
- **1B size** — runs on CPU with 4GB RAM, or GPU with 2GB VRAM. Always-on capable.

### Training Pipeline (What Makes It SOTA at 1B)

```
Base Training (UltraData L0-L4 tiered corpus)
  → Mid-Training (adapt to target distribution)
    → SFT (200B deep-thinking + 200B hybrid-thinking tokens)
      → RL Teachers (math, code, QA, writing — each domain-specialized)
        → On-Policy Distillation (reverse KL divergence, 50-100x more efficient than RL)
```

The key innovation is OPD — the student generates its own rollouts, and domain-specialized RL teachers provide dense token-level supervision via reverse KL divergence. This delivers +16 points average benchmark score and -29% reduction in overlong responses.

### Quantization Tradeoffs

| Format | Size | Quality | Speed (7900 XTX) | Notes |
|---|---|---|---|---|
| F16 | 2.1 GB | Baseline | Slowest | Only for fine-tuning base |
| Q8_0 | 1.1 GB | Near-identical | **Fastest (2x F16)** | Memory-bandwidth bound — smaller is faster |
| Q6_K | 851 MB | Near-identical | Very fast | Best quality-per-byte |
| Q5_K_M | 751 MB | Very good | Fast | |
| Q4_K_M | 657 MB | Good | Fast | Minimal footprint |

At 1B scale, Q8_0 is both smaller AND faster than F16. The bottleneck is memory bandwidth, not compute — halving the weight size nearly doubles generation speed.

---

## The Combination: Why Three Models

Three community variants of MiniCPM5-1B exist, each forked from the same base weights:

### Variant 1: Abiray/MiniCPM5-1B-GGUF (Stock)
Vanilla format conversion. No training. Balanced generalist.
[HF Link](https://huggingface.co/Abiray/MiniCPM5-1B-GGUF)

### Variant 2: Luminia/MiniCPM5-1B-Agent (Tool-Calling Tuned)
Abliterated + SFT on 45K tool-calling rows + DPO with 649 on-policy pairs. Acts more, stalls less. Best tool calling at 1B scale. [Full training recipe published](https://huggingface.co/Luminia/MiniCPM5-1B-Agent-GGUF).
[HF Link](https://huggingface.co/Luminia/MiniCPM5-1B-Agent)

### Variant 3: GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking (Claude Distill)
Fine-tuned on leaked Claude Opus "Fable 5" reasoning traces — real Chain-of-Thought data from Anthropic's Claude with full reasoning chains. Claude-style structured reasoning, better code, better instruction following.
[HF Link](https://huggingface.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking)

### Why Combine Them

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

**GnLOLot** is better at *deciding* whether it knows something (Claude's reasoning calibration). **Luminia** is better at *acting* once it decides (tool-calling reinforcement). The goal is both.

### The Combination Approach

```
Base (openbmb/MiniCPM5-1B)
  → GnLOLot adds: Claude-style structured reasoning
    → Luminia's training recipe adds: tool-calling SFT + DPO on top
```

Preserves Claude's thinking patterns while adding aggressive tool-calling behavior. Three stages, each building on the last.

---

## Hardware

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 7900X (12 cores / 24 threads) |
| GPU | AMD Radeon RX 7900 XTX (24GB VRAM, RDNA 3) |
| RAM | 64 GB |
| Host | ESXi 8.0 (unlicensed — 8 vCPU per VM limit) |
| ML VM | Ubuntu 26.04 LTS, GPU passthrough |

### VM Allocation

| Consumer | CPU | RAM | Notes |
|---|---|---|---|
| ESXi host | reserve 2-4 threads | 4-8 GB | Hypervisor |
| Windows Server VM | 4-6 threads | 16-24 GB | Other workloads |
| **Ubuntu ML VM** | **8 vCPU (unlicensed limit)** | **40 GB (all reserved)** | Training |
| Buffer | — | ~4 GB | Headroom |

8 vCPU is sufficient because GPU training is not CPU-bound — the 7900 XTX does the compute while the CPU only feeds data and handles tokenization.

40GB RAM is comfortable — DPO peaks at ~23-28GB, leaving 12-17GB headroom for checkpoint saving spikes.

---

## Environment Setup

> Full step-by-step guide: [docs/ESXi-Ubuntu-Setup.md](docs/ESXi-Ubuntu-Setup.md)

### Quick Version

1. **BIOS**: Enable SVM + IOMMU
2. **ESXi**: Enable passthrough for 7900 XTX PCI devices (`1002:744c` + `1002:7444`), reboot host
3. **Create VM**: Ubuntu 26.04, 8 vCPU, 40GB RAM (all reserved), 200-300GB disk, GPU passed through
4. **ROCm**: `sudo apt install -y rocm-dev rocm-hip-sdk` (Ubuntu 26.04 ships native ROCm packages)
5. **GPU arch**: `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
6. **Docker**: Install Docker, pull `goldengrapegentleman/unsloth-rocm:2026.1.4-rocm7.1-gfx1100`
7. **Verify**: `python -c "import torch; print(torch.cuda.get_device_name(0))"` → AMD Radeon RX 7900 XTX

### Launch Training Container

```bash
docker run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --shm-size=16g \
    -v /home/$USER/cognitive-core:/workspace \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    goldengrapegentleman/unsloth-rocm:2026.1.4-rocm7.1-gfx1100 \
    bash
```

### Why ESXi + Ubuntu (Not WSL2)

| | WSL2 on Windows | ESXi VM |
|---|---|---|
| GPU access | Indirect (WSL2 + D3D12) | Direct passthrough |
| ROCm | Limited, unofficial | Full native packages |
| Stability | WSL2 quirks | Production-grade |
| Resources | Shares Windows | Dedicated |

---

## Training Pipeline

### Stage 0: Download Models and Data

```bash
cd /workspace

# Luminia's training recipe (code + data schemas)
git clone https://huggingface.co/Luminia/MiniCPM5-1B-Agent-GGUF
cd MiniCPM5-1B-Agent-GGUF

# GnLOLot checkpoint (HF weights, NOT GGUF)
git lfs install
git clone https://huggingface.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking
```

### Stage 1: Mergekit Test (5 min, no training)

Fast sanity check — blends weights via TIES to see if combining produces acceptable results before committing to training.

```bash
pip install mergekit

cat > /workspace/merge_test.yaml << 'EOF'
models:
  - model: /workspace/MiniCPM5-1B-Claude-Opus-Fable5-Thinking
    parameters:
      weight: 0.55
  - model: /workspace/MiniCPM5-1B-Agent
    parameters:
      weight: 0.45
merge_method: ties
base_model: /workspace/MiniCPM5-1B-Agent
parameters:
  normalize: true
  weight: 1.0
dtype: bfloat16
EOF

mergekit-yaml /workspace/merge_test.yaml /workspace/merged-test --cuda
```

Test the result with [test_prompts.txt](test_prompts.txt).

### Stage 2: SFT — Teach Tool Calling to the Claude Reasoning Model

```bash
cd /workspace/MiniCPM5-1B-Agent-GGUF

# Build v4 training data (45,762 curated rows from 26 sources)
python code/data/build_v4.py

# SFT — full fine-tune GnLOLot on tool-calling data
python code/train/sft.py \
    --model /workspace/MiniCPM5-1B-Claude-Opus-Fable5-Thinking \
    --train_file dataset/train_v4.jsonl \
    --out /workspace/sft_claude_agent \
    --epochs 1 \
    --bsz 1 \
    --accum 24 \
    --lr 1e-5 \
    --max_len 24576 \
    --train_cap 24576
```

Takes the GnLOLot checkpoint (Claude reasoning patterns) and trains it on Luminia's curated tool-calling data. The model learns to emit structured `<function>` calls while retaining thinking capabilities.

**Time**: ~1-2 hours. **VRAM**: ~10-14 GB.

### Stage 3: DPO — Reinforce Acting Over Stalling

```bash
# Generate on-policy preference pairs from the SFT model
python code/data/build_prefs_onpolicy_gpu.py \
    --model /workspace/sft_claude_agent \
    --src dataset/train_v4.jsonl \
    --out dataset/dpo_onpolicy_claude.jsonl

# DPO training
python code/train/dpo.py \
    --model /workspace/sft_claude_agent \
    --data dataset/dpo_onpolicy_claude.jsonl \
    --out /workspace/final-cognitive-core \
    --beta 0.1 \
    --lr 1e-6 \
    --epochs 3 \
    --accum 8
```

Shows the model its own tool-calling successes and failures. Rewards acting (emitting tool calls) over stalling (reasoning about whether to act).

**Time**: ~2-4 hours. **VRAM**: ~18-22 GB (tight — if OOM, reduce `--accum` to 4).

---

## GGUF Deployment

```bash
cd /workspace
git clone https://github.com/ggerganov/llama.cpp

# Convert to F16
python llama.cpp/convert_hf_to_gguf.py /workspace/final-cognitive-core \
    --outfile /workspace/final-cognitive-core-f16.gguf --outtype f16

# Quantize
cd llama.cpp
./llama-quantize ../final-cognitive-core-f16.gguf ../final-cognitive-core-Q8_0.gguf Q8_0
./llama-quantize ../final-cognitive-core-f16.gguf ../final-cognitive-core-Q6_K.gguf Q6_K
```

### Ollama

```bash
ollama create cognitive-core -f configs/Modelfile
ollama run cognitive-core
```

### llama.cpp Server (OpenAI-compatible API)

```bash
./llama-server -m /workspace/final-cognitive-core-Q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -t 8 -c 131072
```

API at `http://<vm-ip>:8080/v1`.

---

## Uncertainty Detection (Future Work)

The routing framework will use three layers of uncertainty detection:

1. **Prompt engineering** — system prompt instructs the model to classify confidence before answering
2. **Logit monitoring** — measure token entropy during generation; high entropy = uncertain, trigger delegation
3. **Fine-tuning** — train on examples of uncertain vs confident responses for reliable self-assessment

The actual orchestration framework (oracle delegation, tool execution, response integration) is a separate phase.

---

## Project Structure

```
cognitive-core/
├── README.md
├── Dockerfile
├── configs/
│   ├── merge_test.yaml          # Mergekit TIES config
│   └── Modelfile                # Ollama deployment
├── docs/
│   └── ESXi-Ubuntu-Setup.md     # Full ESXi + GPU passthrough guide
├── scripts/                     # Training & deployment automation
└── test_prompts.txt             # 10 evaluation prompts
```

---

## Key Resources

**Models**
- [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) — base model
- [Luminia/MiniCPM5-1B-Agent](https://huggingface.co/Luminia/MiniCPM5-1B-Agent) — tool-calling tuned
- [GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking](https://huggingface.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking) — Claude distill

**Tools**
- [mergekit](https://github.com/arcee-ai/mergekit) — weight-space model merging
- [Unsloth ROCm Docker](https://hub.docker.com/r/goldengrapegentleman/unsloth-rocm) — training on AMD GPUs
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF conversion + inference
- [Ollama](https://ollama.com) — local deployment

**Papers**
- [MiniCPM4 Technical Report](https://arxiv.org/abs/2506.07900)
- [UltraData L0-L4](https://arxiv.org/abs/2602.09003)
- [On-Policy Distillation](https://arxiv.org/abs/2604.13016)
- [Logits-Induced Token Uncertainty](https://arxiv.org/abs/2502.00290)

**Fable 5 Data**
- [Glint-Research/Fable-5-traces](https://huggingface.co/datasets/Glint-Research/Fable-5-traces)
- [armand0e/claude-fable-5-claude-code](https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code)

> Fable 5 data is leaked Anthropic internal traces. Personal experimentation OK, distribution may have licensing implications.

---

## License

Base model: Apache-2.0. All derived models inherit this license.
