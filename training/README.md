# Training — Build the Model on AWS

End-to-end pipeline for producing the cognitive core model on a g7e.2xlarge:

1. **Merge** GnLOLot (Claude reasoning) + Luminia (tool calling) via TIES
2. **Baseline Eval** — measure merged model routing quality before SFT
3. **SFT** — 3 epochs on 45K tool-calling examples (cosine LR, gradient clipping)
4. **DPO** — reinforce acting over stalling
5. **Upload** to HuggingFace as Q8_0 GGUF

## Contents

| Path | Purpose |
|---|---|
| `Dockerfile` | CUDA + Unsloth training container |
| `requirements.txt` | Python training dependencies |
| `scripts/setup_instance.sh` | One-command EC2 setup (NVIDIA, Docker, uv, repo) |
| `scripts/launch_container.sh` | Launch the training container (persistent volumes) |
| `scripts/run_sft.sh` | SFT / DPO training launcher |
| `scripts/dashboard.py` | Training progress dashboard (port 8765, auth supported) |
| `scripts/gguf_to_hf.py` | GGUF to HuggingFace safetensors converter (with validation) |
| `configs/merge.yaml` | Mergekit TIES config |
| `eval/run_eval.py` | Eval runner (OpenAI-compatible API, hallucination rate) |
| `eval/dataset.jsonl` | 200 test cases across 5 routing categories |
| `test_prompts.txt` | Quick sanity-check prompts |

## Quick Start

```bash
# Launch g7e.2xlarge, SSH in
bash scripts/setup_instance.sh

# Inside container: merge → baseline eval → SFT → DPO → GGUF → upload
```

See [docs/phase-plan.md](docs/phase-plan.md) for full step-by-step.
