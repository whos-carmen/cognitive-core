"""On-policy preference capture: sample v2 itself and harvest its REAL format mistakes.

NOT synthetic. For each prompt we sample v2 K times (temp>0); we classify each completion by whether
its tool-call parses as correct MiniCPM <function name><param> XML:
  - VALID   -> a usable `chosen` (the model's own correct format)
  - WRONG   -> a real `rejected` (markdown fence / Claude <invoke> / JSON / broken XML the model actually emits)
  - NOCALL  -> plain answer, no tool attempt (excluded from format pairs)

Outputs:
  data/built/dpo_format_onpolicy.jsonl   DPO pairs: prompts that produced BOTH a VALID and a WRONG sample
                                         (chosen = a VALID sample, rejected = a WRONG sample — pure on-policy)
  data/built/kto_format_onpolicy.jsonl   KTO rows: {prompt, completion, label} for every VALID/WRONG sample
  + prints the real per-sample format-error rate (the key signal: is format even worth a DPO run?)

Prompts where ALL K samples are WRONG (model never finds the format) are logged to all_wrong.jsonl for a
sub-agent to write a correct `chosen` later.

  python data/build_prefs_onpolicy.py [--prompts N] [--k 6] [--temp 0.8] [--gguf <path>]
"""
import os, sys, re, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(PROJ, "backend"))
import schema, agent
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained(os.path.join(PROJ, "model", "final"), trust_remote_code=True)
SRC = os.path.join(HERE, "built", "dataset_golden.jsonl")
FENCE = chr(96) * 3
WRONG_MARKERS = re.compile(r"<function_calls>|<invoke |<tool_call>|<parameter |```|\"arguments\"\s*:", re.I)
GOOD_CALL = re.compile(r"<function name=\"[^\"]+\">.*?</function>", re.DOTALL)


def classify(text):
    """VALID (parses to correct XML call) / WRONG (a tool-call attempt in a bad format) / NOCALL."""
    if GOOD_CALL.search(text):
        try:
            if agent.parse_assistant(text).get("tool_calls"):
                return "VALID"
        except Exception:
            pass
    # broken <function ...> without a proper close, OR another call syntax => a wrong-format attempt
    if "<function" in text or WRONG_MARKERS.search(text):
        return "WRONG"
    return "NOCALL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--npred", type=int, default=320)
    ap.add_argument("--stride", type=int, default=1, help="take every Nth eligible row for prompt diversity")
    ap.add_argument("--gguf", default=os.path.join(PROJ, "gguf", "sft_v2_ablit-Q8_0.gguf"))
    a = ap.parse_args()

    budget = a.ctx - a.npred - 64                       # max prompt tokens that still leave room to generate
    # prompt pool: rows whose first assistant turn makes a tool call (a tool call is the natural next action)
    prompts = []; n_seen = n_toolong = 0
    for line in open(SRC, encoding="utf-8"):
        row = json.loads(line); msgs, tools = row.get("messages", []), row.get("tools")
        ti = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")), None)
        if ti is None:
            continue
        n_seen += 1
        if n_seen % a.stride:                            # stride for diversity
            continue
        try:
            p = schema.render(msgs[:ti], tools, TOK, enable_thinking=True, add_generation_prompt=True)
        except Exception:
            continue
        if len(TOK(p, add_special_tokens=False)["input_ids"]) > budget:   # won't fit ctx -> skip
            n_toolong += 1
            continue
        prompts.append(p)
        if len(prompts) >= a.prompts:
            break
    print(f"[onpolicy] {len(prompts)} prompts (skipped {n_toolong} too-long > {budget} tok); "
          f"k={a.k} temp={a.temp} ctx={a.ctx} on {os.path.basename(a.gguf)}", flush=True)

    dpo_f = open(os.path.join(HERE, "built", "dpo_format_onpolicy.jsonl"), "w", encoding="utf-8")
    kto_f = open(os.path.join(HERE, "built", "kto_format_onpolicy.jsonl"), "w", encoding="utf-8")
    allwrong_f = open(os.path.join(HERE, "built", "all_wrong.jsonl"), "w", encoding="utf-8")
    n_valid = n_wrong = n_nocall = n_samples = 0
    n_dpo = n_allwrong = 0
    with agent.LlamaServer(a.gguf, ctx=a.ctx, ngl=99) as srv:
        for pi, prompt in enumerate(prompts):
            ids = TOK(prompt, add_special_tokens=False)["input_ids"]
            valids, wrongs = [], []
            for _ in range(a.k):
                out = srv.complete(ids, n_predict=a.npred, temperature=a.temp, top_p=0.95)
                gen = TOK.decode(out.get("tokens") or [], skip_special_tokens=False) if out.get("tokens") else out.get("content", "")
                gen = gen.split("<|im_end|>")[0]
                c = classify(gen); n_samples += 1
                if c == "VALID":
                    n_valid += 1; valids.append(gen)
                    kto_f.write(json.dumps({"prompt": prompt, "completion": gen, "label": True}, ensure_ascii=False) + "\n")
                elif c == "WRONG":
                    n_wrong += 1; wrongs.append(gen)
                    kto_f.write(json.dumps({"prompt": prompt, "completion": gen, "label": False}, ensure_ascii=False) + "\n")
                else:
                    n_nocall += 1
            if valids and wrongs:               # pure on-policy DPO pair
                dpo_f.write(json.dumps({"prompt": prompt, "chosen": valids[0], "rejected": wrongs[0]}, ensure_ascii=False) + "\n")
                n_dpo += 1
            elif wrongs and not valids:         # model never got format right -> sub-agent should write chosen
                allwrong_f.write(json.dumps({"prompt": prompt, "rejected_samples": wrongs}, ensure_ascii=False) + "\n")
                n_allwrong += 1
            if (pi + 1) % 50 == 0:
                print(f"  {pi+1}/{len(prompts)}  valid={n_valid} wrong={n_wrong} nocall={n_nocall} dpo_pairs={n_dpo}", flush=True)
    for f in (dpo_f, kto_f, allwrong_f):
        f.close()
    print(f"\n=== ON-POLICY FORMAT REPORT ===")
    print(f"samples={n_samples}  VALID={n_valid} ({100*n_valid/max(1,n_samples):.1f}%)  "
          f"WRONG={n_wrong} ({100*n_wrong/max(1,n_samples):.1f}%)  NOCALL={n_nocall} ({100*n_nocall/max(1,n_samples):.1f}%)")
    print(f"on-policy DPO pairs (had both valid+wrong)={n_dpo}  all-wrong prompts (need sub-agent chosen)={n_allwrong}")
    print(f"--> format-error rate {100*n_wrong/max(1,n_samples):.1f}% : if tiny, format-DPO won't move the needle.")


if __name__ == "__main__":
    main()
