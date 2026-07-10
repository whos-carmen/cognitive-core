"""ricdomolm/mini-coder-trajs-400k: agentic repo-repair. The assistant writes
'THOUGHT: <reasoning>\n```bash\n<cmd>\n```' as PLAIN TEXT and the shell output comes
back as a `user` turn ('<returncode>..</returncode><output>..</output>').

Converting verbatim would teach bash-in-markdown (NOT our XML tool protocol) -> parity loss.
So we map: THOUGHT -> reasoning_content, the ```bash block -> a structured bash tool_call,
and each env-feedback user turn -> a role:"tool" result. Final answers (no bash block) stay
as assistant content. Only the 'verified' (reward-passing) trajectories are kept.
"""
import re, json
import schema

BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": "Executes a bash command in the working directory and returns its stdout+stderr.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The command to run."}}, "required": ["command"]}}}

_THOUGHT = re.compile(r"THOUGHT:\s*(.*?)(?=```(?:bash|sh)|\Z)", re.DOTALL | re.IGNORECASE)
_BASH = re.compile(r"```(?:bash|sh)\s*\n?(.*?)```", re.DOTALL)


def _text(c):
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def convert_row(row):
    if row.get("verified") is False:          # keep only reward-passing trajectories
        return None
    msgs = row.get("messages")
    if not msgs:
        return None
    out = []
    saw_assistant = False
    for m in msgs:
        role = m.get("role")
        content = _text(m.get("content"))
        if role == "system":
            out.append({"role": "system", "content": content})
        elif role == "user":
            # env feedback after a turn -> tool result; the real task is the first user turn
            if saw_assistant and ("<returncode>" in content or "<output>" in content):
                out.append({"role": "tool", "name": "bash", "content": content})
            else:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            saw_assistant = True
            mth = _THOUGHT.search(content)
            mb = _BASH.search(content)
            a = {"role": "assistant", "content": ""}
            if mth and mth.group(1).strip():
                a["reasoning_content"] = mth.group(1).strip()
            if mb and mb.group(1).strip():
                a["tool_calls"] = [{"type": "function", "function": {
                    "name": "bash", "arguments": {"command": mb.group(1).strip()}}}]
            else:
                # no command => final answer; drop the THOUGHT: prefix if it's the whole thing
                a["content"] = content.strip()
            if a["content"] or a.get("tool_calls") or a.get("reasoning_content"):
                out.append(a)
    ex = {"messages": out, "tools": [BASH_TOOL]}
    ok, _ = schema.validate(ex)
    return ex if ok else None
