# Cognitive Core — Phase-by-Phase Work Plan

> Each phase lists exact commands in order. Estimated times are for g7e.2xlarge.

---

## Phase 0: Instance Setup

**Goal**: EC2 ready with GPU, Docker, repo, and training container.

### Steps

1. Launch g7e.2xlarge with Ubuntu 26.04 AMI
   ```
   aws ec2 run-instances \
       --instance-type g7e.2xlarge \
       --image-id ami-xxx \
       --key-name your-key \
       --security-group-ids sg-xxx \
       --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]'
   ```

2. SSH in
   ```
   ssh -i your-key.pem ubuntu@<public-ip>
   ```

3. Run setup script
   ```
   bash scripts/setup_instance.sh
   ```

4. Reboot if NVIDIA driver was installed, then re-SSH

5. Verify everything
   ```
   nvidia-smi
   docker --version
   uv --version
   docker images | grep cognitive-core
   ```

6. Clone repo (if not already cloned)
   ```
   git clone https://github.com/whos-carmen/cognitive-core.git && cd cognitive-core
   ```

****Time**: ~30 min.

---

## Phase 1: Merge

**Goal**: Combine GnLOLot (Claude reasoning) with Luminia (tool calling) via TIES merge.

### Steps

1. Launch training container
   ```
   bash scripts/launch_container.sh
   ```

2. Inside container, download models
   ```
   cd /workspace
   git lfs install
   git clone https://huggingface.co/GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking /workspace/models/GnLOLot
   git clone https://huggingface.co/Luminia/MiniCPM5-1B-Agent /workspace/models/Luminia
   ```

3. Run merge
   ```
   pip install mergekit
   mergekit-yaml /workspace/configs/merge.yaml /workspace/models/merged --cuda
   ```

4. Verify merge output exists
   ```
   ls /workspace/models/merged/
   # Should see config.json, model-*.safetensors, etc.
   ```

**Time**: ~5 min.

---

## Phase 2: SFT Training

**Goal**: Fine-tune the merged model on 45K tool-calling examples.

### Steps

1. In the container, clone Luminia's training recipe
   ```
   git clone https://huggingface.co/Luminia/MiniCPM5-1B-Agent-GGUF /workspace/code
   cd /workspace/code
   ```

2. Build the training dataset
   ```
   python code/data/build_v4.py
   ```

3. Run SFT
   ```
   python code/train/sft.py \
       --model /workspace/models/merged \
       --train_file dataset/train_v4.jsonl \
       --out /workspace/train/outputs/sft_claude_agent \
       --epochs 1 \
       --bsz 1 \
       --accum 24 \
       --lr 1e-5 \
       --max_len 24576 \
       --train_cap 24576
   ```

4. On host (separate terminal), launch dashboard
   ```
   python3 scripts/dashboard.py --port 8765 --host 0.0.0.0
   # Open http://<instance-ip>:8765
   ```

5. Monitor training loss curve and GPU utilization

6. After SFT completes, test it
   ```
   python -c "
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained('/workspace/train/outputs/sft_claude_agent')
   tokenizer = AutoTokenizer.from_pretrained('/workspace/train/outputs/sft_claude_agent')
   inputs = tokenizer('What is the capital of France?', return_tensors='pt').to('cuda')
   outputs = model.generate(**inputs, max_new_tokens=50)
   print(tokenizer.decode(outputs[0]))
   "
   ```

**Time**: ~3-6 hr.

---

## Phase 3: DPO Training

**Goal**: Reinforce tool-calling behavior over stalling/reasoning-in-circles.

### Steps

1. Generate on-policy preference pairs
   ```
   python code/data/build_prefs_onpolicy_gpu.py \
       --model /workspace/train/outputs/sft_claude_agent \
       --src dataset/train_v4.jsonl \
       --out dataset/dpo_onpolicy_claude.jsonl
   ```

2. Run DPO
   ```
   python code/train/dpo.py \
       --model /workspace/train/outputs/sft_claude_agent \
       --data dataset/dpo_onpolicy_claude.jsonl \
       --out /workspace/train/outputs/final-cognitive-core \
       --beta 0.1 \
       --lr 1e-6 \
       --epochs 3 \
       --accum 8
   ```

3. Monitor dashboard for DPO metrics (loss, accuracy, reward)

4. After DPO, clean up intermediate checkpoints to free disk
   ```
   rm -rf /workspace/train/outputs/sft_claude_agent
   ```

**Time**: ~2-4 hr.

---

## Phase 4: Upload to HuggingFace

**Goal**: Convert to Q8_0 GGUF, push to private HF repo.

### Steps

1. Exit container (or in a new terminal)

2. Build llama.cpp tools
   ```
   cd /workspace
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make -j8
   ```

3. Convert to F16
   ```
   python convert_hf_to_gguf.py /workspace/train/outputs/final-cognitive-core \
       --outfile /workspace/train/outputs/final-cognitive-core-f16.gguf \
       --outtype f16
   ```

4. Quantize to Q8_0
   ```
   ./llama-quantize /workspace/train/outputs/final-cognitive-core-f16.gguf \
       /workspace/train/outputs/final-cognitive-core-Q8_0.gguf Q8_0
   ```

5. Delete the F16 intermediate
   ```
   rm /workspace/train/outputs/final-cognitive-core-f16.gguf
   ```

6. Install huggingface-hub
   ```
   pip install huggingface_hub
   ```

7. Login and upload
   ```
   huggingface-cli login --token hf_your_token_here
   huggingface-cli repo create your-org/cognitive-core-v1 --type model --private
   huggingface-cli upload your-org/cognitive-core-v1 \
       /workspace/train/outputs/final-cognitive-core-Q8_0.gguf \
       /cognitive-core-v1-Q8_0.gguf --repo-type model
   ```

**Time**: ~15 min.

---

## Phase 5: Evaluation

**Goal**: Measure Routing Precision across 200 test cases (5 categories).

### Steps

1. Run the evaluation suite
   ```
   python eval/run_eval.py --model /workspace/train/outputs/final-cognitive-core-Q8_0.gguf \
       --base-url http://localhost:8080/v1
   ```

2. Review per-category accuracy
   ```
   cat eval/results/summary_*.json | python -m json.tool
   ```

3. Check specific failures
   ```
   python eval/run_eval.py --report-only --results eval/results/eval_*.jsonl
   ```

4. Target scores:
   - Routing Precision: >85%
   - Category A (Answer): >85%
   - Category B (Tool Call): >90%
   - Category C (Delegate): >80%
   - Category D (Reasoning): >80%
   - Category E (Abstain): >85%
   - Hallucination Rate: <5%

**Time**: ~30 min.

---

## Phase 6: Iterate

**Goal**: Extend training data, retrain, improve weak categories.

### Steps

1. Identify weak categories from Phase 5 results

2. Source or create additional training data for those categories
   ```
   # Format: JSONL with { messages: [...] }
   # Example:
   {"messages": [
       {"role": "user", "content": "What is the current price of Ethereum?"},
       {"role": "assistant", "content": "<tool_call>{\"name\": \"crypto_price\", \"parameters\": {\"coin\": \"ethereum\"}}</tool_call>"}
   ]}
   ```

3. Append to training data
   ```
   cat code/dataset/train_v4.jsonl my_new_data.jsonl > code/dataset/train_v4_extended.jsonl
   ```

4. Re-run SFT (1 epoch on extended data, starting from previous SFT checkpoint)
   ```
   python code/train/sft.py \
       --model /workspace/models/merged \
       --train_file code/dataset/train_v4_extended.jsonl \
       --out /workspace/train/outputs/sft_v2 \
       --epochs 1 --bsz 1 --accum 24 --lr 1e-5
   ```

5. Re-run DPO
   ```
   python code/data/build_prefs_onpolicy_gpu.py \
       --model /workspace/train/outputs/sft_v2 \
       --src code/dataset/train_v4_extended.jsonl \
       --out code/dataset/dpo_v2.jsonl
   python code/train/dpo.py \
       --model /workspace/train/outputs/sft_v2 \
       --data code/dataset/dpo_v2.jsonl \
       --out /workspace/train/outputs/cognitive-core-v2 \
       --beta 0.1 --lr 1e-6 --epochs 3 --accum 8
   ```

6. Re-upload to HF
   ```
   # Repeat Phase 4 convert + upload steps with the v2 output
   ```

7. Re-run eval
   ```
   python eval/run_eval.py ...
   ```

8. Repeat until all categories meet target scores

**Time**: ~5-10 hr per iteration.

---

## Phase 7: Cost Optimization

**Goal**: Minimize AWS spend for training runs.

### Steps

1. Create an AMI snapshot of the fully-configured instance
   ```
   # From EC2 console: Instance → Actions → Image and templates → Create image
   # Name: cognitive-core-ubuntu-26.04
   ```

2. Switch to spot instances for training
   ```
   aws ec2 request-spot-instances \
       --instance-count 1 \
       --type one-time \
       --launch-specification '{
           "ImageId": "ami-your-snapshot",
           "InstanceType": "g7e.2xlarge",
           "KeyName": "your-key"
       }'
   ```

3. Provision EBS-backed storage for checkpoints, use S3 for final models
   ```
   aws s3 mb s3://cognitive-core-checkpoints
   aws s3 sync /workspace/train/outputs s3://cognitive-core-checkpoints/runs/run-001
   ```

4. Automate the full pipeline
   ```
   # scripts/run_training.sh already exists — extend to:
   # 1. Launch instance from AMI
   # 2. SSH in
   # 3. Run merge → SFT → DPO → convert → upload
   # 4. Terminate instance
   ```

**Time**: ~1 hr to set up, savings of 60-70% per training run.

---

## Summary Timeline

```
Phase 0: Setup        30 min     (once)
Phase 1: Merge         5 min     (once)
Phase 2: SFT         3-6 hr     (per iteration)
Phase 3: DPO         2-4 hr     (per iteration)
Phase 4: Upload       15 min     (per iteration)
Phase 5: Eval         30 min     (per iteration)
Phase 6: Iterate    5-10 hr     (per iteration, optional)
Phase 7: Cost Opt     1 hr       (one-time automation)
```

First full run (Phases 0-5): ~7-11 hr.
Each iteration (Phases 2-6 on spot): ~5-10 hr at ~$1-2/hr.
