"""nvidia/SWE-Zero-openhands-trajectories (and SWE-Hero, same shape): real-repo
issue -> edit -> unified-diff patch. `trajectory` is ALREADY an OpenAI messages list
(content/role/tool_calls). OpenHands stores reasoning in a `think` tool call (whose result
is 'Your thought has been logged') -> we lift that thought into reasoning_content and drop
both the think-call and its logged result, keeping the REAL tool_calls (execute_bash,
str_replace_editor, finish). Tool schemas are synthesized from the observed argument keys.
"""
import json
import schema

_TOOL_DESC = {
    "execute_bash": "Execute a bash command in the repository and return its output.",
    "str_replace_editor": "View, create, or edit a file (str-replace / insert / view).",
    "finish": "Signal that the task is complete.",
}


def _text(c):
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {"_raw": a}


def convert_row(row):
    tr = row.get("trajectory")
    if isinstance(tr, str):
        try:
            tr = json.loads(tr)
        except Exception:
            return None
    if not isinstance(tr, list):
        return None
    out = []
    toolkeys = {}
    last_call = None
    for m in tr:
        role = m.get("role")
        content = _text(m.get("content"))
        if role == "system":
            out.append({"role": "system", "content": content})
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            reasoning, calls = "", []
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {}) or {}
                nm = fn.get("name")
                if not nm:
                    continue
                args = _args(fn.get("arguments"))
                if "think" in nm.lower():
                    if isinstance(args, dict):
                        reasoning += (args.get("thought") or args.get("thinking") or "")
                    continue
                calls.append({"type": "function", "function": {"name": nm, "arguments": args}})
                if isinstance(args, dict):
                    toolkeys.setdefault(nm, set()).update(args.keys())
                else:
                    toolkeys.setdefault(nm, set())
                last_call = nm
            a = {"role": "assistant", "content": content}
            if reasoning.strip():
                a["reasoning_content"] = reasoning.strip()
            if calls:
                a["tool_calls"] = calls
            if a["content"] or a.get("tool_calls") or a.get("reasoning_content"):
                out.append(a)
        elif role in ("tool", "function", "observation"):
            if content.strip().lower().startswith("your thought has been logged"):
                continue  # the think-tool's logged result -> dropped with the call
            out.append({"role": "tool", "name": m.get("name") or last_call, "content": content})
    tools = []
    for nm, keys in toolkeys.items():
        tools.append({"type": "function", "function": {
            "name": nm, "description": _TOOL_DESC.get(nm, nm),
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string"} for k in sorted(keys)},
                           "required": []}}})
    ex = {"messages": out}
    if tools:
        ex["tools"] = tools
    ok, _ = schema.validate(ex)
    return ex if ok else None
