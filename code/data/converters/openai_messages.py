"""Convert OpenAI-style `messages` agent datasets -> canonical schema.

Covers: nvidia/Nemotron-SFT-OpenCode-v1, nvidia/Nemotron-SFT-SWE-v2, nvidia/Nemotron-Agentic-v1,
        nvidia/Nemotron-SFT-Agentic-v2, and any {messages:[{role,content,tool_calls?}], tools} set.
Handles: content as str OR list-of-blocks; tool_calls (OpenAI {id,type,function:{name,arguments}});
         arguments as dict OR json-string; reasoning in `reasoning`/`reasoning_content` or inline <think>;
         role "developer" -> system; tool results -> role "tool".
"""
import os, sys, json, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../data
import schema

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def flatten_content(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or b.get("value") or "")
        return "".join(parts)
    if isinstance(c, dict):
        return c.get("text") or json.dumps(c, ensure_ascii=False)
    return str(c)


def _norm_tool_calls(tcs):
    out = []
    for tc in tcs or []:
        fn = tc.get("function") if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if name:
            out.append({"type": "function", "function": {"name": name, "arguments": args or {}}})
    return out


def convert_row(row):
    msgs_in = row.get("messages") or row.get("conversations") or []
    if isinstance(msgs_in, str):
        try:
            msgs_in = json.loads(msgs_in)
        except Exception:
            return None
    if not isinstance(msgs_in, list):
        return None
    tools = schema.normalize_tools(row.get("tools"))
    msgs = []
    for m in msgs_in:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        if role in ("human",):
            role = "user"
        if role in ("assistant", "gpt"):
            content = flatten_content(m.get("content", m.get("value", "")))
            mm = {"role": "assistant"}
            rc = m.get("reasoning_content") or m.get("reasoning")
            if rc:
                rc = flatten_content(rc) if not isinstance(rc, str) else rc
            else:
                tm = _THINK.search(content)
                if tm:
                    rc = tm.group(1).strip()
                    content = _THINK.sub("", content).strip()
            if rc:
                mm["reasoning_content"] = rc
            tcs = _norm_tool_calls(m.get("tool_calls"))
            if tcs:
                mm["tool_calls"] = tcs
            mm["content"] = content
            msgs.append(mm)
        elif role in ("tool", "tool_result", "observation", "function"):
            msgs.append({"role": "tool", "name": m.get("name"),
                         "content": flatten_content(m.get("content", m.get("value", "")))})
        elif role in ("system", "developer", "user"):
            r = "user" if role == "user" else "system"
            msgs.append({"role": r, "content": flatten_content(m.get("content", m.get("value", "")))})
    if not msgs:
        return None
    ex = {"messages": msgs}
    if tools:
        ex["tools"] = tools
    ok, _ = schema.validate(ex)
    return ex if ok else None


if __name__ == "__main__":
    BASE = r"datasets-analayse"
    MODEL = r"model\final"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    for safe in ["nvidia__Nemotron-SFT-OpenCode-v1", "nvidia__Nemotron-SFT-SWE-v2", "nvidia__Nemotron-Agentic-v1"]:
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
        for r in rows[:60]:
            ex = convert_row(r)
            if not ex:
                continue
            ok += 1
            e = schema.encode_example(ex, tok, max_len=24576)
            if e:
                enc += 1
                lens.append(len(e["input_ids"]))
        lens.sort()
        med = lens[len(lens)//2] if lens else 0
        print(f"{safe}: rows={len(rows)} convert_ok={ok} encode_ok(<=24k)={enc} "
              f"len[min/med/max]={lens[0] if lens else 0}/{med}/{lens[-1] if lens else 0}")
