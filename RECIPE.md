# Cognitive Core - Training Recipe

Reproducible setup for fine-tuning the cognitive-core model on AWS spot instances.

## Supported Instances

| Family | Types | VRAM | Notes |
|--------|-------|------|-------|
| g5 | xlarge, 2xlarge, 4xlarge | 24GB A10G | Good balance of cost/perf |
| g6 | xlarge, 2xlarge | 24GB L4 | Newer, slightly faster |
| g7e | xlarge, 2xlarge, 4xlarge | varies | High-memory options |

**Requirements:** Ubuntu 22.04+ AMI, 100GB+ EBS, 8+ vCPU G-class on-demand limit.

## Quick Start (Fresh Instance)

    ssh -i your-key.pem ubuntu@<instance-ip>
    git clone https://github.com/whos-carmen/cognitive-core.git && cd cognitive-core
    bash training/scripts/setup_instance.sh
    bash training/scripts/run_sft.sh smoke   # 5-step test
    bash training/scripts/run_sft.sh full    # Full SFT, 3 epochs
    bash training/scripts/run_sft.sh dpo     # DPO after SFT

## What setup_instance.sh Does

1. System packages - git, build tools, nvtop, tmux
2. NVIDIA drivers - installs if not present (reboots if needed)
3. Docker + NVIDIA Container Toolkit - GPU passthrough
4. uv - fast Python package manager
5. HuggingFace CLI - for model downloads (run hf auth login first)
6. Clone repo
7. Training container - pulls Unsloth + builds cognitive-core:latest
8. Models - downloads GnLOLot + Luminia, converts GGUF to HF, runs TIES merge

## Model Pipeline

    GnLOLot (Claude reasoning) --+
                                  +-- TIES merge -> merged -> SFT -> DPO -> GGUF
    Luminia (tool calling)     --+

- GnLOLot: GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking (HF, public)
- Luminia: Luminia/MiniCPM5-1B-Agent-GGUF (HF, GGUF to HF converted at setup)
- Merge: TIES method, 0.55/0.45 weights, density 0.6

## Spot Instance Workflow

Launch spot from saved AMI, SSH in, models are on EBS, run training.
No re-download needed on subsequent spots.

## Improving the Model

Add JSONL files to dataset/ with format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Concatenate with existing data and update run_sft.sh to point to new file.

## Directory Structure

    cognitive-core/
      RECIPE.md
      training/
        Dockerfile, requirements.txt
        configs/merge.yaml
        scripts/  (setup_instance.sh, prepare_models.sh, gguf_to_hf.py, run_sft.sh, ...)
        code/train/  (sft.py, dpo.py)
        code/data/   (build_v4.py, converters/)
        dataset/     (train_v4.jsonl, dpo_onpolicy_v4.jsonl)
      models/    (GnLOLot/, Luminia/, merged/)
      router/    (model serving)

## Troubleshooting

- SSH Permission denied: Use EC2 Instance Connect to push temp key
- Out of VRAM: Reduce --max_len or increase --accum in run_sft.sh
- Merge fails: Check both models have same architecture (LlamaForCausalLM)
- Container wont start: docker build -t cognitive-core:latest training/
