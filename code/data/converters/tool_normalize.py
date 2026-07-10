"""Normalize the bash/file/edit/search tool SYNONYMS in the training data to our SINGLE served vocab
(bash/read/write/edit/glob/grep), the same parity move as web_normalize but for the SWE/OpenHands/Claude-Code
tools. Operates on the STRUCTURED canonical {messages, tools} (renames tool_calls[].function.name + remaps
arg keys, rewrites tool declarations + role:tool result names) - NOT regex on text, so a tool name appearing
as a plain word in content is never touched (only real structured calls are).

Mappings (served arg schema in parens):
  execute_bash / run_bash / shell / terminal      -> bash(command)
  list_directory(dir_path)                         -> bash(command="ls -la <dir_path>")
  read_file(file_path|path)                        -> read(file_path)
  write_file(file_path|path, content)              -> write(file_path, content)
  edit_file(file_path, old_text, new_text)         -> edit(file_path, old_string, new_string)
  search_files(pattern, ...)                       -> grep(pattern)
  str_replace_editor/str_replace_based_edit_tool   -> ROUTE by command:
      view->read(file_path=path); create->write(file_path=path, content=file_text);
      str_replace/insert->edit(file_path=path, old_string=old_str, new_string=new_str);
      undo_edit (and unknown commands) -> LEFT AS-IS (rare, no clean target).
Genuinely-distinct tools the user accepted as left-out (todowrite/skill/question/task/browser_*/patch/finish)
are NOT touched.

  python data/converters/tool_normalize.py <in.jsonl> [--inplace | --check | --sample N]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import agent
import schema

SERVED = {t["function"]["name"]: t for t in agent.TOOLS}          # bash/read/write/edit/glob/grep canonical defs
BASH_SYN = {"execute_bash", "run_bash", "shell", "terminal", "bash_command"}
SRE = {"str_replace_editor", "str_replace_based_edit_tool"}


def _s(args, keys):
    if isinstance(args, dict):
        for k in keys:
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def remap_call(name, args):
    """-> (served_name, new_args) for a synonym, or None to leave the call unchanged."""
    a = args if isinstance(args, dict) else {}
    n = name
    if n in BASH_SYN:
        return "bash", {"command": _s(a, ["command", "cmd"])}
    if n == "list_directory":
        d = _s(a, ["dir_path", "path", "directory"])
        return "bash", {"command": ("ls -la " + d).strip()}
    if n == "read_file":
        return "read", {"file_path": _s(a, ["file_path", "path"])}
    if n == "write_file":
        c = a.get("content")
        return "write", {"file_path": _s(a, ["file_path", "path"]), "content": c if isinstance(c, str) else (json.dumps(c) if c is not None else "")}
    if n == "edit_file":
        return "edit", {"file_path": _s(a, ["file_path", "path"]), "old_string": _s(a, ["old_text", "old_string", "old_str"]), "new_string": _s(a, ["new_text", "new_string", "new_str"])}
    if n == "search_files":
        # search_files is a MULTIPLEXED search: content search -> grep, filename/glob search -> glob.
        # (verified: most calls are globs like **/*.json, *.py; only target/output_mode=content are grep.)
        tgt = str(a.get("target") or "").lower(); om = str(a.get("output_mode") or "").lower()
        if tgt == "content" or "content" in om:
            return "grep", {"pattern": _s(a, ["pattern", "query"])}
        return "glob", {"pattern": _s(a, ["glob", "file_glob", "pattern", "query"])}
    if n in SRE:
        cmd = a.get("command")
        path = _s(a, ["path", "file_path"])
        if cmd == "view":
            return "read", {"file_path": path}
        if cmd == "create":
            return "write", {"file_path": path, "content": (a.get("file_text") or "")}
        if cmd in ("str_replace", "insert"):
            return "edit", {"file_path": path, "old_string": (a.get("old_str") or ""), "new_string": (a.get("new_str") or "")}
        return None                                                  # undo_edit / unknown -> leave
    return None


def _decl_targets(name):
    """served tool name(s) a synonym's DECLARATION maps to (str_replace_editor -> read+write+edit)."""
    if name in SRE:
        return ["read", "write", "edit"]
    r = remap_call(name, {"command": "view"} if name in SRE else {})
    if r:
        return [r[0]]
    # bash-syn / file-syn with empty args still resolve by name:
    for fake in ({"command": "x"},):
        r = remap_call(name, fake)
        if r:
            return [r[0]]
    return None


def normalize(ex, stats=None):
    used = set()
    for m in ex.get("messages", []):
        pending = []
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            r = remap_call(fn.get("name"), fn.get("arguments", {}))
            if r:
                if stats is not None:
                    stats[fn.get("name")] = stats.get(fn.get("name"), 0) + 1
                fn["name"], fn["arguments"] = r
                used.add(r[0])
            pending.append(fn.get("name"))
        m["_pending"] = pending
    # second pass: rename role:tool result names to follow the preceding assistant's mapped calls
    queue = []
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            queue = list(m.pop("_pending", []) or [])
        else:
            m.pop("_pending", None)
        if m.get("role") == "tool":
            tn = m.get("name")
            mapped = queue.pop(0) if queue else None
            if mapped:
                m["name"] = mapped
            else:
                r = remap_call(tn, {})
                if r:
                    m["name"] = r[0]
    # declarations: synonym defs -> served defs, deduped
    tools = ex.get("tools")
    if tools:
        new, seen = [], set()
        for t in tools:
            nm = (t.get("function", t)).get("name")
            tgts = _decl_targets(nm)
            if tgts:
                for tg in tgts:
                    if tg in SERVED and tg not in seen:
                        new.append(SERVED[tg]); seen.add(tg)
            else:
                if nm not in seen:
                    new.append(t); seen.add(nm)
        ex["tools"] = new
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sample", type=int, default=0, help="write N context samples per synonym to logs/tool_norm_sample.txt")
    args = ap.parse_args()
    from collections import Counter
    if args.check or args.sample:
        SYN = BASH_SYN | SRE | {"list_directory", "read_file", "write_file", "edit_file", "search_files"}
        before = Counter(); shapes = Counter(); samples = {}
        n = bad_shape = 0
        for line in open(args.src, encoding="utf-8"):
            n += 1
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            for i, m in enumerate(msgs):
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", tc)
                    nm = fn.get("name"); a = fn.get("arguments")
                    shapes[("dict" if isinstance(a, dict) else type(a).__name__)] += 1
                    if not (isinstance(tc, dict) and isinstance(fn, dict) and "name" in fn):
                        bad_shape += 1
                    if nm in SYN:
                        before[nm] += 1
                        if args.sample and len(samples.get(nm, [])) < args.sample:
                            ctx = {"user": next((mm.get("content", "")[:200] for mm in msgs[max(0, i-2):i] if mm.get("role") == "user"), ""),
                                   "assistant_reasoning": (m.get("reasoning_content") or "")[:160],
                                   "CALL": {"name": nm, "arguments": a},
                                   "remapped_to": remap_call(nm, a),
                                   "tool_result_next": next((mm.get("content", "")[:160] for mm in msgs[i+1:i+3] if mm.get("role") == "tool"), "")}
                            samples.setdefault(nm, []).append(ctx)
        served = {"bash", "read", "write", "edit", "glob", "grep"}
        all_calls = Counter()
        for line in open(args.src, encoding="utf-8"):
            for m in json.loads(line).get("messages", []):
                for tc in (m.get("tool_calls") or []):
                    all_calls[(tc.get("function", tc)).get("name")] += 1
        tot = sum(all_calls.values()); srv = sum(v for k, v in all_calls.items() if k in served)
        print(f"rows={n} tool_call arg-shapes={dict(shapes)} non-conforming={bad_shape}")
        print(f"served-now={srv}/{tot} ({100*srv//tot}%); synonyms to normalize: {dict(before)}")
        proj = srv + sum(before.values())  # str_replace_editor undo_edit (~9) won't map, negligible
        print(f"projected served-after ~= {proj}/{tot} ({100*proj//tot}%)")
        if args.sample:
            out = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "tool_norm_sample.txt")
            with open(out, "w", encoding="utf-8") as w:
                for nm, lst in samples.items():
                    w.write(f"\n===== {nm} ({before[nm]} total calls) =====\n")
                    for c in lst:
                        w.write(json.dumps(c, ensure_ascii=False)[:1400] + "\n")
            print("wrote samples ->", out)
        return
    out = args.src if args.inplace else args.src + ".norm"
    n = changed = 0
    stats = {}
    tmp = out + ".tmp"
    with open(args.src, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            ex = json.loads(line)
            b = json.dumps([[c.get("function", c).get("name") for c in (m.get("tool_calls") or [])] for m in ex.get("messages", [])])
            normalize(ex, stats)
            a = json.dumps([[c.get("function", c).get("name") for c in (m.get("tool_calls") or [])] for m in ex.get("messages", [])])
            if b != a:
                changed += 1
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(f"normalized {n} rows -> {out} | rows_changed={changed} | renames {stats}")


if __name__ == "__main__":
    main()
