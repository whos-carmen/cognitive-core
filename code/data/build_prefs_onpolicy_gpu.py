"""On-policy preference capture via transformers on GPU (llama-server is CPU-only -> too slow).

Samples K completions per prompt from the bf16 sft_v2_ablit (the EXACT model DPO trains -> truly
on-policy) and harvests v2's REAL failures to emit a valid tool call:
  VALID  -> parses to a correct <function name><param> XML call
  BAD    -> WRONG (markdown/Claude/JSON/broken-XML attempt) OR NOCALL (punted, no call) = the eval `calls=0` failure
Pairs:  chosen  = a VALID sample (the model's own correct format) else the GOLD tool-call from the SFT row
        rejected= a BAD sample (the model's real mistake)
-> chosen/rejected target exactly "emit a valid tool call vs not", the failure that costs eval points.

Outputs: dpo_format_onpolicy.jsonl, kto_format_onpolicy.jsonl  + the real BAD rate (decision signal).
  python data/build_prefs_onpolicy_gpu.py [--prompts N] [--k 4] [--temp 0.8] [--maxprompt 13312] [--stride 3]
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PROJ, "backend"))
import schema
from build_prefs_onpolicy import classify
import torch
# Blackwell sm_120 SDPA: force O(L) mem-efficient (math = O(L^2) -> OOM at long ctx); repeat_kv over GQA.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.set_float32_matmul_precision("high")
import transformers.integrations.sdpa_attention as _sdpa_attn
_sdpa_attn.use_gqa_in_sdpa = lambda *a, **k: False
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.join(PROJ, "train", "outputs", "sft_v2_ablit")
SRC = os.path.join(HERE, "built", "dataset_golden.jsonl")
TOK = AutoTokenizer.from_pretrained(os.path.join(PROJ, "model", "final"), trust_remote_code=True)
ASSIST = "<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=800)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--maxprompt", type=int, default=13312)
    ap.add_argument("--npred", type=int, default=320)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--model", default=MODEL_DIR)   # point at the DPO base (e.g. sft_v3/checkpoint-5900)
    ap.add_argument("--out", default=os.path.join(HERE, "built", "dpo_format_onpolicy.jsonl"))
    ap.add_argument("--src", default=SRC)   # prompt source (dataset_golden was cleaned -> pass train_v4.jsonl)
    a = ap.parse_args()
    if TOK.pad_token_id is None:
        TOK.pad_token = TOK.eos_token
    print(f"[gpu] loading {a.model} bf16 (mem-efficient SDPA) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 trust_remote_code=True, attn_implementation="sdpa").to("cuda").eval()

    pool = []; seen = toolong = 0
    for line in open(a.src, encoding="utf-8"):
        row = json.loads(line); msgs, tools = row.get("messages", []), row.get("tools")
        ti = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")), None)
        if ti is None:
            continue
        seen += 1
        if seen % a.stride:
            continue
        try:
            prompt = schema.render(msgs[:ti], tools, TOK, enable_thinking=True, add_generation_prompt=True)
            full = schema.render(msgs[:ti + 1], tools, TOK, enable_thinking=True, add_generation_prompt=False)
        except Exception:
            continue
        sp = full.rfind(ASSIST)
        gold = full[sp + len(ASSIST):] if sp >= 0 else ""
        if "<function name=" not in gold:
            continue
        if len(TOK(prompt, add_special_tokens=False)["input_ids"]) > a.maxprompt:
            toolong += 1
            continue
        pool.append((prompt, gold))
        if len(pool) >= a.prompts:
            break
    print(f"[gpu] {len(pool)} prompts (skipped {toolong} >{a.maxprompt} tok); k={a.k} temp={a.temp}", flush=True)

    dpo_f = open(a.out, "w", encoding="utf-8")
    kto_f = open(a.out.replace("dpo_", "kto_"), "w", encoding="utf-8")
    nv = nb = ns = ndpo = 0
    for pi, (prompt, gold) in enumerate(pool):
        ids = TOK(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        try:
            with torch.no_grad():
                out = model.generate(**ids, do_sample=True, temperature=a.temp, top_p=0.95,
                                     num_return_sequences=a.k, max_new_tokens=a.npred, pad_token_id=TOK.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        gens = [g.split("<|im_end|>")[0] for g in TOK.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=False)]
        valids, bads = [], []
        for g in gens:
            ok = classify(g) == "VALID"; ns += 1
            (valids if ok else bads).append(g)
            kto_f.write(json.dumps({"prompt": prompt, "completion": g, "label": ok}, ensure_ascii=False) + "\n")
            nv += ok; nb += (not ok)
        kto_f.write(json.dumps({"prompt": prompt, "completion": gold, "label": True}, ensure_ascii=False) + "\n")  # gold = known-good
        if bads:
            chosen = valids[0] if valids else gold
            dpo_f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": bads[0]}, ensure_ascii=False) + "\n"); ndpo += 1
        if (pi + 1) % 50 == 0:
            print(f"  {pi+1}/{len(pool)} valid={nv} bad={nb} dpo={ndpo}", flush=True)
    dpo_f.close(); kto_f.close()
    print(f"\n=== ON-POLICY (GPU) REPORT ===")
    print(f"samples={ns} VALID={nv} ({100*nv/max(1,ns):.1f}%) BAD={nb} ({100*nb/max(1,ns):.1f}%)  DPO pairs={ndpo}")
    print(f"--> real failure rate {100*nb/max(1,ns):.1f}%  (chosen=valid-call / rejected=model's real miss)")


if __name__ == "__main__":
    main()
