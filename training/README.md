# Training — Build the Model on AWS

End-to-end pipeline for producing the cognitive core model on a g7e.2xlarge:

1. **Merge** GnLOLot (Claude reasoning) + Luminia (tool calling) via TIES
2. **SFT** — 3 epochs on 45K tool-calling examples
3. **DPO** — reinforce acting over stalling
4. **Upload** to HuggingFace as Q8_0 GGUF

## Contents

| Path | Purpose |
|---|---|
| `Dockerfile` | CUDA + Unsloth training container |
| `scripts/setup_instance.sh` | One-command EC2 setup (NVIDIA, Docker, uv, repo) |
| `scripts/launch_container.sh` | Launch the training container |
| `scripts/run_sft.sh` | SFT / DPO training launcher |
| `scripts/dashboard.py` | Training progress dashboard (port 8765) |
| `scripts/gguf_to_hf.py` | GGUF to HuggingFace safetensors converter |
| `configs/merge.yaml` | Mergekit TIES config |
| `configs/Modelfile` | Ollama deployment template |
| `docs/phase-plan.md` | Step-by-step work plan with exact commands |
| `docs/sft-runner.html` | Visual training guide |
| `eval/dataset.jsonl` | 200 test cases for routing evaluation |
| `eval/run_eval.py` | Eval runner (Ollama / OpenAI-compatible API) |
| `test_prompts.txt` | Quick sanity-check prompts |

## Quick Start

```bash
# Launch g7e.2xlarge, SSH in
bash scripts/setup_instance.sh

# Inside container: merge → SFT → DPO → GGUF → upload
```

See [docs/phase-plan.md](docs/phase-plan.md) for full step-by-step.
