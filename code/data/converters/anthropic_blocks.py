"""nlile/misc-merged-claude-code-traces-v1 (+ any Claude-Code dump) stores REAL Claude-Code tool use in
ANTHROPIC content-block format: assistant.content = [{type:text}, {type:tool_use, name, input}],
user.content = [{type:tool_result, tool_use_id, content}]. The `oai` converter misses these (0 tool_calls).

This converter lifts tool_use -> structured assistant tool_calls and tool_result -> role:"tool", AND maps
Claude-Code tool names onto OUR served vocabulary (Bash->bash, Edit/MultiEdit->edit, Read->read,
Glob->glob, Grep->grep, Write->write, WebSearch->web_search, WebFetch->web_fetch) so these real traces
match the exact tools the Space serves (train<->serve parity). Other Claude-Code tools (TodoWrite/Task/...)
are kept as-is (generalized tool-use). Rows are read from messages_json (+ tools_json). Single-user
snippet rows (no assistant tool use) are dropped by schema validation.
"""
import json
import schema

TOOLMAP = {"Bash": "bash", "Edit": "edit", "MultiEdit": "edit", "Read": "read", "Glob": "glob",
           "Grep": "grep", "Write": "write", "WebSearch": "web_search", "WebFetch": "web_fetch",
           "NotebookEdit": "edit", "BashOutput": "bash"}


def _map(name):
    return TOOLMAP.get(name, name)


def _load(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


def _blocktext(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def convert_row(row):
    msgs = _load(row.get("messages_json")) or _load(row.get("messages"))
    if not isinstance(msgs, list) or not msgs:
        return None
    out = []
    id2name = {}
    for x in msgs:
        if not isinstance(x, dict):
            continue
        role = x.get("role")
        content = x.get("content")
        if role == "system":
            out.append({"role": "system", "content": _blocktext(content)})
        elif role == "assistant":
            if isinstance(content, list):
                calls = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_call"):
                        nm = _map(b.get("name") or (b.get("function", {}) or {}).get("name"))
                        args = b.get("input")
                        if args is None:
                            args = (b.get("function", {}) or {}).get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {"_raw": args}
                        calls.append({"type": "function", "function": {"name": nm, "arguments": args if isinstance(args, dict) else {"_raw": args}}})
                        if b.get("id"):
                            id2name[b["id"]] = nm
                a = {"role": "assistant", "content": _blocktext(content)}
                if calls:
                    a["tool_calls"] = calls
                out.append(a)
            else:
                out.append({"role": "assistant", "content": content or ""})
        elif role == "user":
            if isinstance(content, list):
                results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if results:
                    for tr in results:
                        c = tr.get("content")
                        if isinstance(c, list):
                            c = "\n".join(bb.get("text", "") for bb in c if isinstance(bb, dict))
                        out.append({"role": "tool", "name": id2name.get(tr.get("tool_use_id"), "tool"), "content": c or ""})
                    txt = _blocktext(content)
                    if txt.strip():
                        out.append({"role": "user", "content": txt})
                else:
                    out.append({"role": "user", "content": _blocktext(content)})
            else:
                out.append({"role": "user", "content": content or ""})
    # tools: map names, dedup
    tools_in = _load(row.get("tools_json")) or _load(row.get("tools"))
    tools, seen = [], set()
    for t in (tools_in or []):
        fn = t.get("function", t) if isinstance(t, dict) else {}
        nm = _map(fn.get("name"))
        if not nm or nm in seen:
            continue
        seen.add(nm)
        params = fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        tools.append({"type": "function", "function": {"name": nm, "description": fn.get("description", nm), "parameters": params}})
    ex = {"messages": out}
    if tools:
        ex["tools"] = tools
    ok, _ = schema.validate(ex)
    return ex if ok else None
