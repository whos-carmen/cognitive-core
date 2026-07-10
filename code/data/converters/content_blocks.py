"""Unified content-block -> canonical {messages, tools} converter for agent SESSION logs whose turns use
content-blocks. Handles every variant seen across the recent-teacher pi-trace / Claude-Code dumps:
  block types : text | thinking/reasoning | toolCall/tool_use/tool_call/function_call | tool_result
  roles       : system | developer(->system) | user | assistant | toolResult/tool(->our tool role)
  tool names  : bash/read/edit/write (pi-harness, already ours) | Bash/Edit/Read/Glob/Grep/WebSearch
                (Claude-Code) | functions.Write (Kimi)  -> all mapped to our served vocabulary.
Thinking blocks -> reasoning_content; toolCall -> structured tool_calls; toolResult/tool_result -> role:tool.
Tools are synthesized from the tool names actually called (arg-key union), since pi-harness has no tools list.

`session_to_example(raw_msgs)` takes an ORDERED list of raw message dicts (each {role, content, [toolName]})
and returns a validated {messages, tools} (or None). Session grouping lives in build_keepadds3.py.
"""
import json
import schema

TOOLMAP = {"bash": "bash", "read": "read", "edit": "edit", "write": "write", "glob": "glob", "grep": "grep",
           "Bash": "bash", "Edit": "edit", "MultiEdit": "edit", "Read": "read", "Glob": "glob", "Grep": "grep",
           "Write": "write", "WebSearch": "web_search", "WebFetch": "web_fetch", "NotebookEdit": "edit"}


def _norm_tool(name):
    if not name:
        return None
    name = str(name)
    if "." in name:                      # Kimi: "functions.Write" -> "Write"
        name = name.split(".")[-1]
    name = name.split(":")[0]            # "functions.Write:0" style -> "Write"
    return TOOLMAP.get(name, name)


def _extract(content):
    """-> (text, reasoning, calls[(name,args,id)], tool_results[(id,content)])"""
    if isinstance(content, str):
        return content, "", [], []
    text, reasoning, calls, tres = [], [], [], []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                text.append(b.get("text", ""))
            elif t in ("thinking", "reasoning"):
                reasoning.append(b.get("thinking") or b.get("reasoning") or b.get("text") or "")
            elif t in ("toolCall", "tool_use", "tool_call", "function_call"):
                fn = b.get("function", {}) if isinstance(b.get("function"), dict) else {}
                name = b.get("name") or b.get("toolName") or fn.get("name")
                args = b.get("arguments")
                if args is None:
                    args = b.get("input")
                if args is None:
                    args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                calls.append((name, args if isinstance(args, dict) else {"_raw": args}, b.get("id")))
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(bb.get("text", "") for bb in c if isinstance(bb, dict))
                tres.append((b.get("tool_use_id"), c or ""))
    return "\n".join(t for t in text if t), "\n".join(r for r in reasoning if r), calls, tres


def session_to_example(raw_msgs):
    out = []
    toolkeys = {}
    id2name = {}
    last_call = None
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text, reasoning, calls, tres = _extract(m.get("content"))
        if role in ("system", "developer"):
            out.append({"role": "system", "content": text})
        elif role == "user":
            if tres:                       # Claude-Code: tool results ride in a user turn
                for tid, c in tres:
                    out.append({"role": "tool", "name": id2name.get(tid, last_call or "tool"), "content": c})
                if text.strip():
                    out.append({"role": "user", "content": text})
            else:
                out.append({"role": "user", "content": text})
        elif role == "assistant":
            tcs = []
            for name, args, cid in calls:
                nm = _norm_tool(name)
                if not nm:
                    continue
                tcs.append({"type": "function", "function": {"name": nm, "arguments": args}})
                toolkeys.setdefault(nm, set()).update(args.keys() if isinstance(args, dict) else [])
                if cid:
                    id2name[cid] = nm
                last_call = nm
            a = {"role": "assistant", "content": text}
            if reasoning.strip():
                a["reasoning_content"] = reasoning.strip()
            if tcs:
                a["tool_calls"] = tcs
            if a["content"] or a.get("tool_calls") or a.get("reasoning_content"):
                out.append(a)
        elif role in ("toolResult", "tool", "toolresult", "tool_result"):
            nm = _norm_tool(m.get("toolName")) or last_call or "tool"
            out.append({"role": "tool", "name": nm, "content": text or ""})
    tools = []
    for nm, keys in toolkeys.items():
        tools.append({"type": "function", "function": {
            "name": nm, "description": nm,
            "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in sorted(keys)}, "required": []}}})
    ex = {"messages": out}
    if tools:
        ex["tools"] = tools
    ok, _ = schema.validate(ex)
    return ex if ok else None
