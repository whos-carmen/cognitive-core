"""Canonical SFT schema + render + mask for MiniCPM5-1B agentic coding.

Canonical example = {"messages": [...], "tools": [...]} where:
  - {"role":"system","content": str}
  - {"role":"user","content": str}
  - {"role":"assistant", "reasoning_content"?: str, "content"?: str,
       "tool_calls"?: [{"type":"function","function":{"name": str, "arguments": dict}}]}
  - {"role":"tool", "name"?: str, "content": str}        # tool RESULT (rendered as <tool_response>)
  - tools = OpenAI function-def list: [{"type":"function","function":{"name","description","parameters"}}]

VERIFIED against model/final/chat_template.jinja (probe, 2026-06-01):
  * `<think>` renders at EVERY assistant turn iff tool results use role "tool" (not user).
  * tool_calls render once each as XML <function name=..><param name=..>val</param></function>
    (param value CDATA-wrapped automatically when it contains <, & or newline).
  * template has NO {% generation %} tags -> mask assistant spans by regex on the rendered text.
"""
import json
import re

# supervise everything between "<|im_start|>assistant\n" and the closing "<|im_end|>" (inclusive,
# so the model learns to STOP). Mask system / user / <tool_response>.
_ASSIST_SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)


def normalize_tools(tools):
    """Coerce a source's tool list (varied shapes) into OpenAI function-def list. Returns None if empty."""
    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            return None
    if isinstance(tools, dict):
        tools = [tools]
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
        elif "function" in t and isinstance(t["function"], dict):
            fn = t["function"]
        elif "inputSchema" in t:  # OpenCode style {id/name, description, inputSchema}
            fn = {"name": t.get("name") or t.get("id"),
                  "description": t.get("description", ""),
                  "parameters": t.get("inputSchema", {"type": "object", "properties": {}})}
        elif "parameters" in t or "name" in t:  # {name, description, parameters}
            fn = {"name": t.get("name"), "description": t.get("description", ""),
                  "parameters": t.get("parameters", {"type": "object", "properties": {}})}
        else:
            continue
        if not fn.get("name"):
            continue
        fn.setdefault("description", "")
        fn.setdefault("parameters", {"type": "object", "properties": {}})
        out.append({"type": "function", "function": fn})
    return out or None


def validate(example):
    """Lightweight structural check. Returns (ok: bool, reason: str)."""
    msgs = example.get("messages")
    if not msgs or not isinstance(msgs, list):
        return False, "no messages"
    roles = {m.get("role") for m in msgs}
    if "assistant" not in roles:
        return False, "no assistant turn"
    has_signal = any(
        m.get("role") == "assistant" and (m.get("tool_calls") or m.get("reasoning_content") or m.get("content"))
        for m in msgs)
    if not has_signal:
        return False, "no assistant content to train on"
    return True, "ok"


def render(messages, tools, tokenizer, enable_thinking=True, add_generation_prompt=False):
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False,
        add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)


def cap_tool_outputs(messages, max_chars=8000, head=5000, tail=3000):
    """Cap long tool-RESULT contents HEAD+TAIL (keep what ran AND the error/result tail).
    Generous by default so debugging traces (compiler/test/stack) survive; the Space sandbox MUST use
    the SAME cap for train<->serve parity. Long-context training (~24k) accommodates these."""
    out = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > max_chars:
            mm = dict(m)
            cut = len(c) - head - tail
            mm["content"] = c[:head] + ("\n...[%d chars truncated]...\n" % cut) + c[-tail:]
            out.append(mm)
        else:
            out.append(m)
    return out


def encode_example(example, tokenizer, max_len=24576, max_tool_chars=8000):
    """Render + tokenize + assistant-only label mask.
    Returns {"input_ids","attention_mask","labels"} or None (oversized / nothing to supervise)."""
    msgs = cap_tool_outputs(example["messages"], max_tool_chars)
    text = render(msgs, example.get("tools"), tokenizer)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None
    spans = [(m.start(1), m.end(1)) for m in _ASSIST_SPAN.finditer(text)]
    if not spans:
        return None
    # pointer walk (spans are ordered, non-overlapping) -> O(n)
    labels, si = [], 0
    for tid, (a, b) in zip(ids, offs):
        while si < len(spans) and offs and a >= spans[si][1]:
            si += 1
        sup = si < len(spans) and a >= spans[si][0] and b <= spans[si][1]
        labels.append(tid if sup else -100)
    if not any(l != -100 for l in labels):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
