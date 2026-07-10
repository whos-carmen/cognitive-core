"""Convert Hermes-lineage agent traces -> canonical schema.

Source rows: {"conversations":[{"from","value"}], "tools": <json string>, ...}
 - from: system|human|gpt|tool   value: text (gpt has inline <think>..</think> + <tool_call>{json}</tool_call>;
                                          tool has <tool_response>{json}</tool_response>)
Covers: lambda/hermes-agent-reasoning-traces, DJLougen/hermes-agent-traces-filtered,
        sroecker/hermes-agent-traces-chatml (ChatML variant uses same {from,value} or {role,content}).
"""
import os, sys, json, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../data
import schema

_ROLE = {"system": "system", "human": "user", "user": "user",
         "gpt": "assistant", "assistant": "assistant", "tool": "tool", "observation": "tool"}
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TC = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TR = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def convert_row(row):
    convs = row.get("conversations") or row.get("messages") or []
    tools = schema.normalize_tools(row.get("tools"))
    msgs = []
    for turn in convs:
        role = _ROLE.get(turn.get("from") or turn.get("role"))
        val = turn.get("value")
        if val is None:
            val = turn.get("content") or ""
        if not isinstance(val, str):
            val = json.dumps(val, ensure_ascii=False)
        if role is None:
            continue
        if role == "assistant":
            m = {"role": "assistant"}
            tm = _THINK.search(val)
            if tm:
                m["reasoning_content"] = tm.group(1).strip()
            tcs = []
            for tcjson in _TC.findall(val):
                try:
                    d = json.loads(tcjson)
                except Exception:
                    continue
                name = d.get("name")
                args = d.get("arguments", d.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                if name:
                    tcs.append({"type": "function", "function": {"name": name, "arguments": args}})
            if tcs:
                m["tool_calls"] = tcs
            m["content"] = _TC.sub("", _THINK.sub("", val)).strip()
            msgs.append(m)
        elif role == "tool":
            tr = _TR.search(val)
            msgs.append({"role": "tool", "content": (tr.group(1).strip() if tr else val.strip())})
        else:
            msgs.append({"role": role, "content": val})
    if not msgs:
        return None
    ex = {"messages": msgs}
    if tools:
        ex["tools"] = tools
    ok, _ = schema.validate(ex)
    return ex if ok else None


if __name__ == "__main__":
    # End-to-end test on the local sample: convert real rows -> canonical -> render+mask.
    SAMP = r"datasets-analayse\lambda__hermes-agent-reasoning-traces\sample.jsonl"
    MODEL = r"model\final"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = []
    for ln in open(SAMP, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass  # skip truncated sample lines
    print(f"valid sample rows: {len(rows)}")
    n = min(len(rows), 80)
    ok = 0
    lens = []
    fit16 = fit24 = fit32 = 0
    sup_ratio = []
    for r in rows[:n]:
        ex = convert_row(r)
        if not ex:
            continue
        ok += 1
        capped = schema.cap_tool_outputs(ex["messages"], 2000)
        text = schema.render(capped, ex.get("tools"), tok)
        L = len(tok(text, add_special_tokens=False)["input_ids"])
        lens.append(L)
        fit16 += L <= 16384; fit24 += L <= 24576; fit32 += L <= 32768
        enc = schema.encode_example(ex, tok, max_len=32768)
        if enc:
            sup_ratio.append(sum(1 for l in enc["labels"] if l != -100) / len(enc["input_ids"]))
    lens.sort()
    med = lens[len(lens)//2] if lens else 0
    print(f"converted ok: {ok}/{n}")
    print(f"token len (capped tool-out): min={lens[0] if lens else 0} median={med} max={lens[-1] if lens else 0}")
    print(f"fit<=16k: {fit16}/{ok}  <=24k: {fit24}/{ok}  <=32k: {fit32}/{ok}")
    if sup_ratio:
        print(f"supervised ratio: mean={sum(sup_ratio)/len(sup_ratio):.3f} (n={len(sup_ratio)})")
