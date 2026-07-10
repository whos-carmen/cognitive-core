"""nlile/misc-merged-claude-code-traces-v1 and thoughtworks/agentic-coding-trajectories store the chat
as a JSON STRING in `messages_json` (+ optional `tools_json`). nlile (merged from many source tables) is
heterogeneous: some rows only have a user turn in messages_json with the reply in `assistant_response`.
We parse messages_json (+tools_json), fall back to system_prompt/user_prompt/assistant_response when the
parsed messages lack an assistant turn, then hand the {messages, tools} to the proven `oai` normalizer
(which enforces our schema, structured tool_calls, reasoning_content). Rows without a real assistant turn
are dropped by oai.convert_row -> None.
"""
import json
import openai_messages as Coai


def _load(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


def convert_row(row):
    msgs = _load(row.get("messages_json")) or _load(row.get("messages"))
    if not isinstance(msgs, list):
        msgs = []
    # nlile fallback: rebuild from the split fields if messages_json has no assistant turn
    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in msgs):
        sp, up, ar = row.get("system_prompt"), row.get("user_prompt"), row.get("assistant_response")
        rebuilt = []
        if not msgs:
            if sp:
                rebuilt.append({"role": "system", "content": sp})
            if up:
                rebuilt.append({"role": "user", "content": up})
            msgs = rebuilt or msgs
        if ar:
            msgs = list(msgs) + [{"role": "assistant", "content": ar}]
    if not msgs:
        return None
    oai_row = {"messages": msgs}
    tools = _load(row.get("tools_json")) or _load(row.get("tools"))
    if tools:
        oai_row["tools"] = tools
    return Coai.convert_row(oai_row)
