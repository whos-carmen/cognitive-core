"""Full-FT DPO from sft_v2_ablit — CUSTOM loop (TRL DPOTrainer blocked by a mergekit dep cascade;
TRL KTO needs bsz>1 -> OOM at 13k). Memory fits on 32GB via the same tricks as sft.py PLUS the key one:

  DPO logprobs only need logits at the COMPLETION positions (~300 tok), NOT the full 13k sequence.
  So we slice the base-model hidden states to the completion span and apply lm_head to ONLY those
  -> the [L,130560] logit tensor is never materialized (only [comp_len,130560], ~80MB).

Blackwell mem-efficient SDPA (flash/math/cudnn off, repeat_kv over GQA) — identical to sft.py. bsz1.
Frozen reference = the initial sft_v2_ablit (bf16, no_grad). Prompt span is masked (loss only on completion).

Usage: python train/dpo.py [--data data/built/dpo_train.jsonl] [--beta 0.1] [--lr 5e-7] [--epochs 3] [--max_steps N]
"""
import os, sys, json, gc, argparse, datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJ, ".hfcache"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"
sys.path.insert(0, os.path.join(PROJ, "data"))
LOG = os.path.join(PROJ, "logs", "dpo.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(m):
    s = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}"
    print(s, flush=True); open(LOG, "a", encoding="utf-8").write(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJ, "data", "built", "dpo_train.jsonl"))
    ap.add_argument("--model", default=os.path.join(PROJ, "train", "outputs", "sft_v2_ablit"))
    ap.add_argument("--out", default=os.path.join(PROJ, "train", "outputs", "dpo_v3"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=13824)   # prompt(~12.6k)+completion; drop longer
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")
    import transformers.integrations.sdpa_attention as _sdpa_attn
    _sdpa_attn.use_gqa_in_sdpa = lambda *a, **k: False
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from datasets import Dataset, Features, Sequence, Value

    log(f"=== DPO(custom) start {vars(args)} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ===")
    tok = AutoTokenizer.from_pretrained(os.path.join(PROJ, "model", "final"), trust_remote_code=True)  # canonical tokenizer (checkpoint dirs lack tokenizer files)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else 1

    def _gen(path):
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                ex = json.loads(ln)
            except Exception:
                continue
            p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(ex["chosen"], add_special_tokens=False)["input_ids"]
            r = tok(ex["rejected"], add_special_tokens=False)["input_ids"]
            if not c or not r or len(p) + max(len(c), len(r)) > args.max_len:
                continue
            yield {"chosen_ids": p + c, "rejected_ids": p + r, "plen": len(p)}

    feats = Features({"chosen_ids": Sequence(Value("int32")), "rejected_ids": Sequence(Value("int32")),
                      "plen": Value("int32")})
    cache = os.path.join(PROJ, ".hfcache", "dpo_arrow_" + os.path.splitext(os.path.basename(args.data))[0])
    ds = Dataset.from_generator(_gen, gen_kwargs={"path": args.data}, features=feats, cache_dir=cache)
    log(f"DPO pairs tokenized: {len(ds)}")

    class Collator:
        def __call__(self, feats):  # bsz1
            f = feats[0]
            return {"chosen_ids": torch.tensor([f["chosen_ids"]]),
                    "rejected_ids": torch.tensor([f["rejected_ids"]]),
                    "plen": int(f["plen"])}

    policy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                  trust_remote_code=True, attn_implementation="sdpa")
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               trust_remote_code=True, attn_implementation="sdpa")
    ref.config.use_cache = False
    ref.eval()
    for pp in ref.parameters():
        pp.requires_grad_(False)
    ref.to("cuda")

    def comp_logp(model, input_ids, plen):
        """Sum log-prob of the completion tokens (positions >= plen). lm_head applied ONLY to the
        completion span -> no [L,vocab] logits. input_ids: [1,L]."""
        hidden = model.model(input_ids=input_ids, attention_mask=None, use_cache=False)[0]  # [1,L,H]
        # token at position t is predicted by hidden[t-1]; completion tokens are [plen:L]
        ch = hidden[:, plen - 1:-1, :]                      # [1, comp_len, H]
        tgt = input_ids[:, plen:]                            # [1, comp_len]
        logits = model.lm_head(ch).float()                  # [1, comp_len, vocab] (comp_len small)
        lp = torch.log_softmax(logits, dim=-1)
        return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # [1]

    class DPOTrainer(Trainer):
        _diag = False
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            cids = inputs["chosen_ids"].to(model.device)
            rids = inputs["rejected_ids"].to(model.device)
            plen = inputs["plen"]
            lp_c = comp_logp(model, cids, plen)
            lp_r = comp_logp(model, rids, plen)
            with torch.no_grad():
                rlp_c = comp_logp(ref, cids, plen)
                rlp_r = comp_logp(ref, rids, plen)
            logits = args.beta * ((lp_c - lp_r) - (rlp_c - rlp_r))
            loss = -F.logsigmoid(logits).mean()
            if not DPOTrainer._diag:
                DPOTrainer._diag = True
                print(f"DIAG L_c={cids.shape[1]} L_r={rids.shape[1]} plen={plen} "
                      f"margin={(lp_c-lp_r).item():.3f} mem={torch.cuda.memory_allocated()/2**30:.1f}GiB", flush=True)
            # acc = chosen preferred over rejected (reward = beta*(lp - rlp))
            with torch.no_grad():
                acc = ((args.beta * (lp_c - rlp_c)) > (args.beta * (lp_r - rlp_r))).float().mean()
            self._acc = acc.item()
            return (loss, {"logits": logits}) if return_outputs else loss

    ta = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=1, gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
        optim="adamw_8bit", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, weight_decay=0.0, logging_steps=5, save_steps=100, save_total_limit=2,
        dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
        report_to="none", seed=3407,
    )
    metrics_path = os.path.join(PROJ, "logs", "dpo_metrics.jsonl")

    class MetricCB(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                rec = {"step": state.global_step, "acc": round(getattr(trainer, "_acc", 0.0), 3)}
                rec.update({k: round(v, 5) for k, v in logs.items() if isinstance(v, (int, float))})
                open(metrics_path, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
                print("METRIC " + json.dumps(rec), flush=True)

    class MemCleanCB(TrainerCallback):
        def on_step_end(self, a, state, control, **kw):
            if state.global_step % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()

    trainer = DPOTrainer(model=policy, args=ta, train_dataset=ds, data_collator=Collator(),
                         callbacks=[MetricCB(), MemCleanCB()])
    log("trainer ready; starting custom DPO train()")
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "DPO_DONE.json"), "w") as f:
        json.dump({"done": True, "args": vars(args), "ts": datetime.datetime.now().isoformat()}, f, indent=2)
    log("=== DPO DONE ===")


if __name__ == "__main__":
    main()
