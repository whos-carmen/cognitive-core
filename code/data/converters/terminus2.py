"""Convert terminus-2 terminal-agent traces -> canonical schema.

Covers: open-thoughts/AgentTrove, nvidia/Nemotron-Terminal-Corpus.
Source: row["conversations"] = ShareGPT list (role/content or from/value). Protocol:
  - first non-system user turn = the task instruction (role user)
  - assistant turns = a JSON string {"analysis","plan","commands":[{"keystrokes","duration"}],"task_complete"}
        -> analysis+plan => reasoning_content ; each command.keystrokes => a bash tool_call ; task_complete => final
  - subsequent user turns = raw terminal output  => role "tool"
Implicit single tool = bash. (Filter to strong-teacher rows upstream; drop gpt-5-nano slices.)
"""
import os, sys, json, ast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../data
import schema

_BASH_TOOL = [{"type": "function", "function": {
    "name": "bash", "description": "Run shell command(s) in the terminal.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]


def _as_list(convs):
    if isinstance(convs, str):
        for parse in (json.loads, ast.literal_eval):  # JSON, then Python-repr (single-quoted)
            try:
                v = parse(convs)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
        return None
    return convs if isinstance(convs, list) else None


def convert_row(row):
    convs = _as_list(row.get("conversations") or row.get("messages"))
    if not convs:
        return None
    msgs = []
    seen_user = False
    for turn in convs:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from")
        val = turn.get("content")
        if val is None:
            val = turn.get("value", "")
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)
        if role == "system":
            msgs.append({"role": "system", "content": val})
        elif role in ("user", "human"):
            if not seen_user:
                msgs.append({"role": "user", "content": val})
                seen_user = True
            else:
                msgs.append({"role": "tool", "content": val})  # terminal output
        elif role in ("assistant", "gpt"):
            m = {"role": "assistant"}
            try:
                act = json.loads(val)
            except Exception:
                act = None
            if isinstance(act, dict) and ("commands" in act or "analysis" in act or "task_complete" in act):
                reason = " ".join(x for x in [act.get("analysis"), act.get("plan")] if isinstance(x, str)).strip()
                if reason:
                    m["reasoning_content"] = reason
                tcs = []
                for c in act.get("commands", []) or []:
                    ks = c.get("keystrokes") if isinstance(c, dict) else (c if isinstance(c, str) else None)
                    if ks:
                        tcs.append({"type": "function", "function": {"name": "bash", "arguments": {"cmd": ks}}})
                if tcs:
                    m["tool_calls"] = tcs
                m["content"] = "Task complete." if act.get("task_complete") and not tcs else ""
            else:
                m["content"] = val  # non-JSON assistant -> plain content
            msgs.append(m)
    if not msgs:
        return None
    ex = {"messages": msgs, "tools": _BASH_TOOL}
    ok, _ = schema.validate(ex)
    return ex if ok else None


if __name__ == "__main__":
    BASE = r"datasets-analayse"
    MODEL = r"model\final"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    for safe in ["nvidia__Nemotron-Terminal-Corpus", "open-thoughts__AgentTrove"]:
        p = os.path.join(BASE, safe, "sample.jsonl")
        rows = []
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
        ok = enc = 0
        lens = []
        prev = None
        for r in rows[:60]:
            ex = convert_row(r)
            if not ex:
                continue
            ok += 1
            e = schema.encode_example(ex, tok, max_len=24576)
            if e:
                enc += 1
                lens.append(len(e["input_ids"]))
                if prev is None:
                    prev = schema.render(schema.cap_tool_outputs(ex["messages"]), ex.get("tools"), tok)
        lens.sort()
        med = lens[len(lens)//2] if lens else 0
        print(f"{safe}: rows={len(rows)} convert_ok={ok} encode_ok(<=24k)={enc} len[min/med/max]={lens[0] if lens else 0}/{med}/{lens[-1] if lens else 0}")
        if prev and safe.startswith("nvidia"):
            print("----- terminus-2 rendered preview (first ~1100 chars) -----")
            print(prev[:1100])
