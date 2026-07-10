"""Full fine-tune SFT of MiniCPM5-1B (FINAL) on the built mix.
Pre-tokenizes data/built/{train,eval}.jsonl via schema.encode_example (assistant-span mask),
memory-mapped Arrow cache on D:; plain transformers.Trainer; adamw_8bit (GPU-resident, not paged);
batch=1 x grad-accum (no pad waste at 24k); grad-ckpt. Logs to logs/sft.log, marker SFT_DONE.json.

Usage: python train/sft.py [--max_len 24576] [--epochs 2] [--accum 24] [--lr 1e-5] [--max_steps N(smoke)]
"""
import os, sys, json, gc, argparse, datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJ, ".hfcache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(PROJ, ".hfcache", "datasets"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Windows build has NO expandable_segments -> variable-len seqs fragment the CUDA caching allocator,
# whose reserved pool is mirrored into host commit (private bytes grew ~0.1GB/min -> RAM exhaustion).
# Bound it: GC reserved blocks at 80% + cap split size to reduce fragmentation (+ periodic empty_cache below).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8"  # NO max_split_size_mb (it bloated reserved VRAM 15->26GB); periodic empty_cache (below) bounds host commit
sys.path.insert(0, os.path.join(PROJ, "data"))
import schema

LOG = os.path.join(PROJ, "logs", "sft.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(m):
    s = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}"
    print(s, flush=True)
    open(LOG, "a", encoding="utf-8").write(s + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=24576)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--accum", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=-1)  # >0 = smoke test
    ap.add_argument("--train_cap", type=int, default=12288)  # drop examples longer than this (VRAM: logits=L*vocab)
    ap.add_argument("--model", default=os.path.join(PROJ, "model", "final"))
    ap.add_argument("--out", default=os.path.join(PROJ, "train", "outputs", "sft"))
    ap.add_argument("--train_file", default=os.path.join(PROJ, "data", "built", "train.jsonl"))  # override for SFT-v2 (e.g. retail-dropped mix)
    ap.add_argument("--neftune", type=float, default=0.0)  # NEFTune noise alpha (e.g. 5); 0 = off. Anti-overfit for multi-epoch runs.
    args = ap.parse_args()

    import torch
    # Force O(L) attention GLOBALLY. On this Blackwell sm_120 / torch2.11+cu128 win build there is NO
    # flash kernel, and the SDPA auto-dispatcher PREFERS the math backend for causal head_dim=128 bf16,
    # materializing a [B,H,L,L] score matrix -> OOM at L>=16k. Forbidding math forces the mem-efficient /
    # cudnn kernel (O(L)); probe-verified full 131k native ctx fits at 23GiB. Global (not a context mgr) so
    # it also covers grad-checkpoint recompute in backward.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN SDPA caches a workspace per input SHAPE -> with variable seq-lens
                                                  # this LEAKS ~100MB/step into host commit. Force mem-efficient (O(L), no per-shape cache).
    torch.backends.cuda.enable_math_sdp(False)    # math = O(L^2) score matrix -> OOM; forbid it
    torch.set_float32_matmul_precision("high")
    # mem-efficient does NOT support SDPA enable_gqa; make the model use repeat_kv (standard MHA, mathematically identical)
    # by forcing use_gqa_in_sdpa->False, so it dispatches to mem-efficient instead of cuDNN.
    import transformers.integrations.sdpa_attention as _sdpa_attn
    _sdpa_attn.use_gqa_in_sdpa = lambda *a, **k: False
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from datasets import Dataset, Features, Sequence, Value
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

    log(f"=== SFT start {vars(args)} | cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''} ===")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else 1
    ML = args.max_len

    # Pre-tokenize via from_generator: yields UNIFORM flat int-lists (Arrow handles them; the raw
    # nested canonical rows break load_dataset's schema inference). Memory-mapped on disk -> RAM-safe.
    def _gen(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ex = json.loads(ln)
                except Exception:
                    continue
                enc = schema.encode_example({"messages": ex["messages"], "tools": ex.get("tools")}, tok, max_len=ML)
                if enc:
                    yield {"input_ids": enc["input_ids"], "labels": enc["labels"],
                           "attention_mask": enc["attention_mask"]}

    feats = Features({"input_ids": Sequence(Value("int32")), "labels": Sequence(Value("int32")),
                      "attention_mask": Sequence(Value("int8"))})
    built = os.path.join(PROJ, "data", "built")
    train_path = args.train_file
    # cache keyed by train-file name so a different mix (e.g. train_v2) NEVER reuses stale tokenization
    cache = os.path.join(PROJ, ".hfcache", "sft_arrow_" + os.path.splitext(os.path.basename(train_path))[0])
    log(f"train_file={train_path}  cache={cache}")
    train_ds = Dataset.from_generator(_gen, gen_kwargs={"path": train_path},
                                      features=feats, cache_dir=cache)
    eval_ds = None
    ep = os.path.join(built, "eval.jsonl")
    if os.path.exists(ep):
        eval_ds = Dataset.from_generator(_gen, gen_kwargs={"path": ep}, features=feats, cache_dir=cache)
    log(f"tokenized: train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")
    _cap = args.train_cap
    train_ds = train_ds.filter(lambda b: [len(x) <= _cap for x in b["input_ids"]], batched=True, batch_size=2000)
    if eval_ds is not None:
        eval_ds = eval_ds.filter(lambda b: [len(x) <= _cap for x in b["input_ids"]], batched=True, batch_size=2000)
    log(f"after train_cap={args.train_cap}: train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")

    class Collator:
        def __call__(self, feats):
            mx = max(len(f["input_ids"]) for f in feats)
            ii, ll, aa = [], [], []
            for f in feats:
                p = mx - len(f["input_ids"])
                ii.append(f["input_ids"] + [PAD] * p)
                ll.append(f["labels"] + [-100] * p)
                aa.append(f["attention_mask"] + [0] * p)
            return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
                    "attention_mask": torch.tensor(aa)}

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # NOTE: liger's class/instance monkeypatch does NOT fuse linear-CE here (it leaves a [B,L,vocab] logit
    # tensor whose 10.86GiB gradient OOMs in backward). We instead call the fused kernel DIRECTLY in
    # compute_loss below (model/version-agnostic), so logits are never materialized.

    ta = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz, gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=1, prediction_loss_only=True,  # bsz1 => no pad => is_causal O(L) path; loss only (no logits)
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        optim="adamw_8bit", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, weight_decay=0.0, logging_steps=10, save_steps=100,
        save_total_limit=2, eval_strategy=("steps" if eval_ds is not None else "no"), eval_steps=200,
        dataloader_num_workers=0, dataloader_pin_memory=False,  # pinned host buffers were a leak source
        ignore_data_skip=True,  # on resume, don't re-iterate skipped batches (slow) — start fresh shuffle at resume step;
                                # enables fast periodic resume-RESETS to clear the ~120MB/step cuDNN/allocator host leak
        report_to="none", seed=3407, logging_dir=os.path.join(args.out, "tb"),
        neftune_noise_alpha=(args.neftune if args.neftune and args.neftune > 0 else None),  # noisy embeddings -> regularize / anti-overfit over the extra epochs
    )
    metrics_path = os.path.join(PROJ, "logs", "sft_metrics.jsonl")

    class MetricCB(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs and any(k in logs for k in ("loss", "eval_loss")):
                rec = {"step": state.global_step}
                rec.update({k: round(v, 5) for k, v in logs.items() if isinstance(v, (int, float))})
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print("METRIC " + json.dumps(rec), flush=True)

    class MemCleanCB(TrainerCallback):
        """Release fragmented CUDA reserved blocks (mirrored into host commit on this win build) every N steps,
        which otherwise grow ~0.1GB/min with variable-len sequences and exhaust RAM."""
        def on_step_end(self, a, state, control, **kw):
            if state.global_step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    class LceTrainer(Trainer):
        """compute_loss via liger fused-linear-CE: runs base transformer -> hidden, then the fused kernel
        on lm_head.weight directly, so the [B,L,vocab] logits are NEVER materialized. For unpadded bsz=1
        microbatches we pass attention_mask=None so SDPA takes the is_causal (O(L)) path."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._lce_sum = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="sum")
            self._lce_mean = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")

        _diag = False
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            base = self.accelerator.unwrap_model(model)
            if not LceTrainer._diag:
                LceTrainer._diag = True
                gc = getattr(base.model, "gradient_checkpointing", "n/a")
                print(f"DIAG grad_ckpt={gc} training={base.training} "
                      f"L={inputs['input_ids'].shape} memalloc={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)
            labels = inputs["labels"]
            am = inputs.get("attention_mask")
            pass_mask = am if (am is not None and (am == 0).any()) else None  # None => is_causal fast path
            out = base.model(input_ids=inputs["input_ids"], attention_mask=pass_mask, use_cache=False)
            hidden = out[0]
            Hd = hidden.size(-1)
            sh = hidden[..., :-1, :].contiguous().view(-1, Hd)
            sl = labels[..., 1:].contiguous().view(-1).to(sh.device)
            head = base.lm_head
            bias = getattr(head, "bias", None)
            if num_items_in_batch is not None:
                loss = self._lce_sum(head.weight, sh, sl, bias) / num_items_in_batch
            else:
                loss = self._lce_mean(head.weight, sh, sl, bias)
            return (loss, out) if return_outputs else loss

    trainer = LceTrainer(model=model, args=ta, train_dataset=train_ds,
                         eval_dataset=eval_ds, data_collator=Collator(), callbacks=[MetricCB(), MemCleanCB()])
    from transformers.trainer_utils import get_last_checkpoint
    ckpt = get_last_checkpoint(args.out) if os.path.isdir(args.out) else None
    log(f"trainer ready; starting train() resume_from={ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "SFT_DONE.json"), "w") as f:
        json.dump({"done": True, "args": vars(args), "ts": datetime.datetime.now().isoformat()}, f, indent=2)
    log("=== SFT DONE ===")


if __name__ == "__main__":
    main()
