"""Emperorizzis/ASTRA-SFT-1k: MCP tool-use, reward-filtered. `mcp_info` (JSON str) carries
base_info.tool_list = [{name, description, parameters}] -> our tools list directly. `trajectory`
(JSON str) is a messages list where each tool use is split into TWO assistant turns -
assistant(content + reasoning_content) then assistant(function_call={name, arguments-str}) -
followed by a role:"function" result. We MERGE the talk+call into one assistant turn (our schema:
content + reasoning_content + tool_calls) and map function -> role:"tool". MCP tool names are kept
as-is (generalized tool-use); re-rendering through our template gives train<->serve XML parity.
"""
import json
import schema


def _text(c):
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _tools_from_mcp(mcp_info):
    try:
        mcp = json.loads(mcp_info) if isinstance(mcp_info, str) else mcp_info
    except Exception:
        return []
    tl = (((mcp or {}).get("base_info") or {}).get("tool_list")) or []
    tools = []
    for t in tl:
        if isinstance(t, dict) and t.get("name"):
            tools.append({"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", t["name"]),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
    return tools


def convert_row(row):
    tr = row.get("trajectory")
    if isinstance(tr, str):
        try:
            tr = json.loads(tr)
        except Exception:
            return None
    if not isinstance(tr, list):
        return None
    out, pend = [], None

    def flush():
        nonlocal pend
        if pend is not None:
            out.append(pend)
            pend = None

    for m in tr:
        role = m.get("role")
        content = _text(m.get("content"))
        if role == "system":
            flush(); out.append({"role": "system", "content": content})
        elif role == "user":
            flush(); out.append({"role": "user", "content": content})
        elif role == "assistant":
            fc = m.get("function_call")
            if fc and fc.get("name"):
                if pend is None:
                    pend = {"role": "assistant", "content": ""}
                args = fc.get("arguments", "{}")
                try:
                    args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    args = {"_raw": args}
                pend.setdefault("tool_calls", []).append(
                    {"type": "function", "function": {"name": fc["name"], "arguments": args}})
            else:
                if pend is not None and pend.get("tool_calls"):
                    flush()
                if pend is None:
                    pend = {"role": "assistant", "content": ""}
                if content:
                    pend["content"] = (pend["content"] + ("\n" if pend["content"] else "") + content)
                rc = m.get("reasoning_content")
                if rc:
                    pend["reasoning_content"] = pend.get("reasoning_content", "") + rc
        elif role in ("function", "tool", "observation"):
            flush()
            out.append({"role": "tool", "name": m.get("name"), "content": content})
    flush()
    ex = {"messages": out}
    tools = _tools_from_mcp(row.get("mcp_info"))
    if tools:
        ex["tools"] = tools
    ok, _ = schema.validate(ex)
    return ex if ok else None
