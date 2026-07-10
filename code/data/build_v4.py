"""Build SFT-v4 = clean v2 backbone + a CURATED cull of the four v3-added shards.

v3 regressed (34->23) because the added shards (realdata, keepadds, keepadds2, keepadds3) teach
over-exploration, foreign/unbindable tool names, and non-termination to a 1B. This applies the
APPROVED 10-step cull to ONLY the added shards, normalizes dashes/emoji everywhere, then writes
  data/built/train_v4.jsonl = train_v2.jsonl (all) + curated added rows.

Served tool vocab (gate target) = {bash,read,write,edit,glob,grep,web_search,web_fetch}.
Reuses data/converters/tool_normalize.remap_call for the structured synonym remap, plus a few extra
text-name synonyms the cull lists (apply_patch/replace/str_replace/edit_file/read_file/write_file/
search_code/list_directory/webfetch/websearch/run_command...).

  python data/build_v4.py
"""
import os, sys, json, re, hashlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "backend"))
sys.path.insert(0, os.path.join(HERE, "converters"))
import schema
import agent
import tool_normalize as tn

BUILT = os.path.join(HERE, "built")
V2 = os.path.join(BUILT, "train_v2.jsonl")
ADDED = ["realdata", "keepadds", "keepadds2", "keepadds3"]
OUT = os.path.join(BUILT, "train_v4.jsonl")

SERVED = {"bash", "read", "write", "edit", "glob", "grep", "web_search", "web_fetch"}
# served declaration objects, keyed by name (6 base from agent.TOOLS + 2 web from agent.WEB_TOOLS)
SERVED_DECL = {t["function"]["name"]: t for t in (agent.TOOLS + agent.WEB_TOOLS)}

# ---- STEP 0: extra synonym map (case-insensitive) on TOP of tool_normalize.remap_call ----
# These are name-only remaps (args mostly already match served keys, or are best-effort passthrough).
EXTRA_SYN = {
    "run_shell_command": "bash", "execute_bash": "bash", "run_command": "bash",
    "run_bash": "bash", "shell": "bash", "terminal": "bash", "bash_command": "bash",
    "list_directory": "bash",
    "write_file": "write",
    "str_replace_editor": "edit", "str_replace": "edit", "apply_patch": "edit",
    "replace": "edit", "edit_file": "edit", "str_replace_based_edit_tool": "edit",
    "read_file": "read",
    "search_code": "grep", "search_files": "grep", "grep_search": "grep",
    "webfetch": "web_fetch", "web_fetch": "web_fetch",
    "websearch": "web_search", "web_search": "web_search",
}


def _argmap_name_only(served, args):
    """Best-effort arg coercion when we remap by NAME only (extra synonyms not handled by remap_call)."""
    a = args if isinstance(args, dict) else {}
    if served == "bash":
        cmd = a.get("command") or a.get("cmd") or a.get("dir_path") or a.get("path") or ""
        if served == "bash" and ("dir_path" in a or (not (a.get("command") or a.get("cmd")) and a.get("path"))):
            cmd = ("ls -la " + str(cmd)).strip()
        return {"command": str(cmd)}
    if served == "read":
        return {"file_path": str(a.get("file_path") or a.get("path") or "")}
    if served == "write":
        c = a.get("content")
        return {"file_path": str(a.get("file_path") or a.get("path") or ""),
                "content": c if isinstance(c, str) else (json.dumps(c) if c is not None else "")}
    if served == "edit":
        return {"file_path": str(a.get("file_path") or a.get("path") or ""),
                "old_string": str(a.get("old_string") or a.get("old_str") or a.get("old_text") or ""),
                "new_string": str(a.get("new_string") or a.get("new_str") or a.get("new_text") or "")}
    if served == "glob":
        return {"pattern": str(a.get("pattern") or a.get("glob") or a.get("query") or "")}
    if served == "grep":
        return {"pattern": str(a.get("pattern") or a.get("query") or "")}
    if served == "web_search":
        return {"query": str(a.get("query") or a.get("q") or "")}
    if served == "web_fetch":
        return {"url": str(a.get("url") or a.get("link") or "")}
    return a


def vocab_gate(ex):
    """STEP 0. Remap synonyms (tool_normalize first, then EXTRA_SYN by name), remap role:tool names,
    rewrite tools[] to served schema. Return True to KEEP, False to DROP (any name outside SERVED)."""
    # 1) tool_normalize structured remap (handles execute_bash/str_replace_editor/read_file/... with arg routing)
    tn.normalize(ex)
    # 2) extra name-only remaps + collect which served names each assistant turn ends up calling
    for m in ex.get("messages", []):
        pend = []
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            nm = fn.get("name")
            low = nm.lower() if isinstance(nm, str) else nm
            if low in EXTRA_SYN:
                served = EXTRA_SYN[low]
                fn["name"] = served
                fn["arguments"] = _argmap_name_only(served, fn.get("arguments", {}))
                nm = served
            pend.append(nm)
        m["_pend"] = pend
    # 3) role:tool result names follow the preceding assistant's calls (or direct synonym)
    queue = []
    for m in ex.get("messages", []):
        if m.get("role") == "assistant":
            queue = list(m.pop("_pend", []) or [])
        else:
            m.pop("_pend", None)
        if m.get("role") == "tool":
            tnm = m.get("name")
            mapped = queue.pop(0) if queue else None
            if mapped:
                m["name"] = mapped
            elif isinstance(tnm, str) and tnm.lower() in EXTRA_SYN:
                m["name"] = EXTRA_SYN[tnm.lower()]
    # 4) GATE: any tool_call name outside SERVED -> drop
    used = set()
    for m in ex.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("function", tc).get("name")
            used.add(nm)
            if nm not in SERVED:
                return False
        if m.get("role") == "tool":
            n = m.get("name")
            if n is not None and n not in SERVED:
                # an unmapped tool RESULT name implies a foreign call somewhere -> drop
                return False
    # 5) rewrite tools[] to served schema (only the served tools actually used, deduped, stable order)
    order = ["bash", "read", "write", "edit", "glob", "grep", "web_search", "web_fetch"]
    ex["tools"] = [SERVED_DECL[n] for n in order if n in used] or [SERVED_DECL[n] for n in order[:6]]
    return True


# ---------- helpers ----------
def call_names(ex):
    return [tc.get("function", tc).get("name") for m in ex.get("messages", [])
            for tc in (m.get("tool_calls") or [])]


def n_calls(ex):
    return sum(len(m.get("tool_calls") or []) for m in ex.get("messages", []))


def first_user(ex):
    for m in ex.get("messages", []):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def row_text(ex):
    parts = []
    for m in ex.get("messages", []):
        for fld in ("content", "reasoning_content"):
            v = m.get(fld)
            if isinstance(v, str):
                parts.append(v)
        for tc in (m.get("tool_calls") or []):
            a = tc.get("function", tc).get("arguments")
            if isinstance(a, dict):
                parts.append(json.dumps(a, ensure_ascii=False))
    return "\n".join(parts)


# ---------- STEP 1..7 predicates (True = DROP) ----------
def step1_last_tool(ex):
    m = ex.get("messages", [])
    return bool(m) and m[-1].get("role") == "tool"


def step2_explore_only(ex):
    names = call_names(ex)
    if not names:
        return False
    return all(n in {"glob", "grep", "read"} for n in names)


_HYPER = re.compile(r"juspay__hyperswitch|trace_generation/repos", re.I)
def step3_hyperswitch(ex):
    t = row_text(ex)
    if _HYPER.search(t):
        return True
    return len(re.findall(r"hyperswitch", t, re.I)) >= 2


_ERRPAT = re.compile(r"InputValidationError|tool_use_error|Sibling tool call errored")
def step4_broken(ex):
    msgs = ex.get("messages", [])
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            a = tc.get("function", tc).get("arguments")
            if isinstance(a, dict) and "_raw" in a:
                return True
            if isinstance(a, str):
                try:
                    json.loads(a)
                except Exception:
                    return True
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and _ERRPAT.search(m["content"]):
            return True
    # error result immediately followed by a same-name retry
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and _ERRPAT.search(m["content"]):
            tnm = m.get("name")
            for j in range(i + 1, len(msgs)):
                mj = msgs[j]
                if mj.get("role") == "assistant" and mj.get("tool_calls"):
                    if any(tc.get("function", tc).get("name") == tnm for tc in mj["tool_calls"]):
                        return True
                    break
    return False


def step5_overlong(ex):
    return n_calls(ex) >= 15


_GPU = re.compile(r"rocprof|tflops|\bvgpr\b|wmma|hip_force|bank_conflict|gfx115|occupancy|\bsimd\b", re.I)
def step6_gpu(ex):
    return bool(_GPU.search(row_text(ex)))


_META = re.compile(r"Your task is to create a detailed summary|^# /loop|already running inside the megaplan|<local-command-caveat>")
_BARE = {"go on", "yes", "yes please", "continue", "ok", "proceed"}
def step7_meta(ex):
    fu = first_user(ex)
    if _META.search(fu):
        return True
    s = fu.strip().lower()
    if len(s) <= 14 and s in _BARE:
        return True
    if n_calls(ex) == 0:
        ac = "".join(m.get("content") or "" for m in ex.get("messages", []) if m.get("role") == "assistant")
        if len(ac) < 80:
            return True
    return False


# ---------- STEP 8 normalize ----------
_DASH = re.compile("[—–‑‒―]")
_EMOJI = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F0FF"
    "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F"
    "\U0001F1E6-\U0001F1FF" "♀♂⚕⚖✈❤" "]", flags=re.UNICODE)

# ====================== STEP 2: context-aware em/en-dash handling (PROSE ONLY) ======================
# Replaces U+2014/2013 (and the rarer U+2011/2012/2015) by CONTEXT, never inside code. Code is masked
# out first: fenced ```...``` blocks and inline `...` spans are protected, so a dash inside code is
# left exactly as-is. Operates ONLY on reasoning_content + assistant text content (callers guarantee
# this); tool_call arguments and tool RESULTS are never passed in.
_EMDASH_CHARS = "—–‑‒―"                          # U+2014 U+2013 U+2011 U+2012 U+2015
_DASH_ANY = re.compile("[" + _EMDASH_CHARS + "]")
# split a string into (is_code, text) segments: fenced blocks first, then inline-code within prose.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`\n]*`")


def _segments(s):
    """Yield (is_code, chunk). Fenced blocks and inline-code spans are is_code=True (left untouched)."""
    pos = 0
    for fm in _FENCE.finditer(s):
        # prose before the fence -> further split by inline code
        for seg in _split_inline(s[pos:fm.start()]):
            yield seg
        yield (True, s[fm.start():fm.end()])
        pos = fm.end()
    for seg in _split_inline(s[pos:]):
        yield seg


def _split_inline(s):
    pos = 0
    for im in _INLINE.finditer(s):
        if im.start() > pos:
            yield (False, s[pos:im.start()])
        yield (True, s[im.start():im.end()])
        pos = im.end()
    if pos < len(s):
        yield (False, s[pos:])


def _classify(prose, i):
    """Classify the dash at index i within a (non-code) prose chunk. Returns a group key.
      'range'  : intra-word / numeric compound or range  (replace -> '-')
      'aside'  : spaced clause-join or parenthetical aside (replace -> ', ')
      'default': anything else                            (replace -> '-')
    """
    prev = prose[i - 1] if i > 0 else ""
    nxt = prose[i + 1] if i + 1 < len(prose) else ""
    # range / compound: tight (no surrounding spaces) between word chars or digits  e.g. 3-5, X-Y, well-known
    if prev and nxt and not prev.isspace() and not nxt.isspace():
        if (prev.isalnum() and nxt.isalnum()):
            return "range"
        return "default"
    # spaced on at least one side -> clause-joining dash or parenthetical aside
    if prev.isspace() or nxt.isspace() or prev == "" or nxt == "":
        return "aside"
    return "default"


_GROUP_REPL = {"range": "-", "aside": ", ", "default": "-"}


def _ctx_label(prose, i):
    """Human-readable surrounding-context bucket for the ANALYSIS pass (2-3 word window)."""
    a = prose[max(0, i - 18):i]
    b = prose[i + 1:i + 19]
    wa = a.split()[-2:] if a.strip() else []
    wb = b.split()[:2] if b.strip() else []
    prev = prose[i - 1] if i > 0 else "^"
    nxt = prose[i + 1] if i + 1 < len(prose) else "$"
    spaced = prev.isspace() or prev == "^", nxt.isspace() or nxt == "$"
    if (prev.isalnum() and nxt.isalnum()):
        return ("range/compound  e.g. '%s-%s'" % (wa[-1] if wa else prev, wb[0] if wb else nxt), _classify(prose, i))
    if spaced[0] and spaced[1]:
        return ("spaced clause/aside  ' - %s'" % (" ".join(wb) if wb else "<end>"), _classify(prose, i))
    if spaced[0] or spaced[1]:
        return ("half-spaced  '%s-%s'" % (" ".join(wa) or prev, " ".join(wb) or nxt), _classify(prose, i))
    return ("other  '%s[%s]%s'" % (prev, "dash", nxt), _classify(prose, i))


def replace_dashes_prose(s, counter=None):
    """Context-aware dash replacement over PROSE ONLY (code masked). Returns new string."""
    if not isinstance(s, str) or not _DASH_ANY.search(s):
        return s
    out = []
    for is_code, chunk in _segments(s):
        if is_code or not _DASH_ANY.search(chunk):
            out.append(chunk)
            continue
        buf = []
        for i, ch in enumerate(chunk):
            if ch in _EMDASH_CHARS:
                g = _classify(chunk, i)
                if counter is not None:
                    counter[g] += 1
                rep = _GROUP_REPL[g]
                # collapse " , " -> ", " when the original was "word - word" (space already before dash)
                if rep == ", " and buf and buf[-1] == " ":
                    buf.pop()
                buf.append(rep)
                # if aside replacement and the next char is a space, avoid ",  " double space
                if rep == ", " and i + 1 < len(chunk) and chunk[i + 1] == " ":
                    # mark to skip the following space by inserting a sentinel handled below
                    buf.append("\x00")
            else:
                if buf and buf[-1] == "\x00":
                    buf.pop()  # drop sentinel; skip this (space) char
                    if ch == " ":
                        continue
                buf.append(ch)
        out.append("".join(c for c in buf if c != "\x00"))
    return "".join(out)


def analyze_dashes_prose(s, ctx_counter, group_counter):
    """Tally surrounding-context buckets for the analysis report (prose only)."""
    if not isinstance(s, str) or not _DASH_ANY.search(s):
        return
    for is_code, chunk in _segments(s):
        if is_code:
            continue
        for i, ch in enumerate(chunk):
            if ch in _EMDASH_CHARS:
                label, group = _ctx_label(chunk, i)
                ctx_counter[label] += 1
                group_counter[group] += 1


def _strip_emoji(s):
    """Emoji-only strip for PROSE fields. Dashes are handled separately by the context-aware pass
    (STEP 2 refinement), prose-only, so we no longer blind-replace dashes here and never touch args."""
    if not isinstance(s, str):
        return s
    return _EMOJI.sub("", s)


def step8_normalize(ex):
    """Strip emoji from prose; trim any single reasoning_content >2000c. Return False if incoherent.
    (Dash handling moved to the unified context-aware prose pass; tool_call args are NOT touched.)"""
    total_rc = 0
    for m in ex.get("messages", []):
        for fld in ("content", "reasoning_content"):
            if isinstance(m.get(fld), str):
                m[fld] = _strip_emoji(m[fld])
        rc = m.get("reasoning_content")
        if isinstance(rc, str):
            if len(rc) > 2000:
                # keep head (setup) + tail (the decision); cut the rumination in the middle
                m["reasoning_content"] = rc[:1200].rstrip() + "\n...\n" + rc[-700:].lstrip()
            total_rc += len(m["reasoning_content"])
    if total_rc > 4000:
        # leave if the row still has a usable terminal assistant answer or real tool work; else drop
        last_asst = next((m for m in reversed(ex.get("messages", [])) if m.get("role") == "assistant"), None)
        ok = bool(last_asst and (last_asst.get("content") or last_asst.get("tool_calls")))
        if not ok:
            return False
    return True


def fu_hash(ex):
    return hashlib.md5(first_user(ex)[:200].encode("utf-8", "ignore")).hexdigest()


def main():
    stats = {}
    kept_rows = []          # list of (shard, ex)
    DROP_STEPS = [
        ("step1", step1_last_tool), ("step2", step2_explore_only), ("step3", step3_hyperswitch),
        ("step4", step4_broken), ("step5", step5_overlong), ("step6", step6_gpu), ("step7", step7_meta),
    ]
    global_seen = set()     # cross-shard first-user dedup (STEP 9 part a)

    for shard in ADDED:
        path = os.path.join(BUILT, shard + ".jsonl")
        c = Counter()
        survivors = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c["in"] += 1
                try:
                    ex = json.loads(line)
                except Exception:
                    c["badjson"] += 1
                    continue
                # STEP 0 vocab gate (mutates ex)
                if not vocab_gate(ex):
                    c["step0"] += 1
                    continue
                # STEP 1..7
                dropped = False
                for name, pred in DROP_STEPS:
                    if pred(ex):
                        c[name] += 1
                        dropped = True
                        break
                if dropped:
                    continue
                # STEP 8 normalize
                if not step8_normalize(ex):
                    c["step8"] += 1
                    continue
                survivors.append(ex)

        # ---- STEP 9: dedup (<=1 per first-user[:200]) + keepadds3 shape caps + drop trivial-unverified ----
        deduped = []
        for ex in survivors:
            h = fu_hash(ex)
            if h in global_seen:
                c["step9_dup"] += 1
                continue
            global_seen.add(h)
            deduped.append(ex)
        survivors = deduped

        if shard == "keepadds3":
            # cap dominant shapes + index.html-writer rows to <=150 each
            CAP = 150
            shape_count = Counter()
            cap_shapes = {("bash", "write"), ("write",), ("bash", "write", "bash")}
            tmp = []
            idx_html = 0
            for ex in survivors:
                seq = tuple(call_names(ex))
                # drop trivial unverified: ends on 'write', <=2 calls, no bash/read after the write
                names = list(seq)
                if names and names[-1] == "write" and len(names) <= 2 and not any(n in ("bash", "read") for n in names):
                    c["step9_trivial"] += 1
                    continue
                is_idx = any(str(tc.get("function", tc).get("arguments", {}).get("file_path", "")).endswith("index.html")
                             for m in ex.get("messages", []) for tc in (m.get("tool_calls") or []))
                if seq in cap_shapes:
                    if shape_count[seq] >= CAP:
                        c["step9_shapecap"] += 1
                        continue
                    shape_count[seq] += 1
                if is_idx:
                    if idx_html >= CAP:
                        c["step9_idxcap"] += 1
                        continue
                    idx_html += 1
                tmp.append(ex)
            survivors = tmp

        c["after_cull"] = len(survivors)
        stats[shard] = c
        for ex in survivors:
            kept_rows.append((shard, ex))

    # ---- STEP 10: rebalance so ADDS together contribute <= ~20% of total tool-call mass (v2 dominant) ----
    # v2 is KEPT WHOLE (per approved refinement): it scored 38/65 trained WITH its todowrite/skill/
    # question/browser_* rows and the model provably SUPPRESSES those at inference, so they are harmless
    # (unlike the adds' str_replace_editor/execute_bash, which a 1B imitates). So the vocab-gate is OFF
    # for v2 and NO v2 rows are dropped. The "0 foreign tool names" invariant now applies to the ADDED
    # rows only. v2 is written verbatim here; the em-dash pass (STEP 2) runs later over the whole file's
    # PROSE only (think + assistant content), never code/args/tool-results - so we do NOT touch v2 here.
    print("writing v2 backbone WHOLE (gate OFF, untouched) ...", flush=True)
    v2_mass = 0
    v2_rows = 0
    v2_tmp = OUT + ".v2norm.tmp"
    with open(V2, encoding="utf-8") as f, open(v2_tmp, "w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            v2_mass += n_calls(ex)
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            v2_rows += 1
    print("  v2: kept ALL %d rows (mass=%d)" % (v2_rows, v2_mass))

    # target: adds_mass <= 0.20 * total  => adds_mass <= 0.25 * v2_mass
    TARGET_ADDS = int(0.25 * v2_mass)
    # current adds mass per shard
    per_shard = defaultdict(list)
    for shard, ex in kept_rows:
        per_shard[shard].append(ex)
    cur = {s: sum(n_calls(e) for e in rows) for s, rows in per_shard.items()}
    cur_total = sum(cur.values())

    final_added = []           # final kept added rows
    cap_log = {}
    if cur_total <= TARGET_ADDS:
        for s in ADDED:
            final_added.extend(per_shard.get(s, []))
            cap_log[s] = (cur.get(s, 0), cur.get(s, 0), len(per_shard.get(s, [])))
    else:
        # Cap keepadds hardest: allocate the budget by shrinking each shard proportionally, but
        # give keepadds the smallest multiplier. Use ordered priority weights.
        # priority weight = relative share we WANT to preserve (realdata/keepadds3 high, keepadds lowest).
        W = {"realdata": 1.0, "keepadds3": 1.0, "keepadds2": 0.6, "keepadds": 0.35}
        wsum = sum(W[s] * cur.get(s, 0) for s in ADDED) or 1
        for s in ADDED:
            rows = per_shard.get(s, [])
            if not rows:
                cap_log[s] = (0, 0, 0)
                continue
            budget = TARGET_ADDS * (W[s] * cur.get(s, 0)) / wsum   # tool-call budget for this shard
            # keep whole rows (smallest-call first to maximize row diversity per call) until budget hit
            rows_sorted = sorted(rows, key=lambda e: n_calls(e))
            acc = 0
            keep = []
            for e in rows_sorted:
                nc = n_calls(e)
                if acc + nc > budget and keep:
                    break
                acc += nc
                keep.append(e)
            final_added.extend(keep)
            cap_log[s] = (cur.get(s, 0), acc, len(keep))

    # ---- WRITE OUTPUT: v2 (normalized) + curated added, all validated ----
    print("validating + writing train_v4 ...", flush=True)
    n_out = 0
    n_badval = 0
    added_mass = 0
    added_rows_out = 0
    with open(OUT, "w", encoding="utf-8") as w:
        # v2 first
        with open(v2_tmp, encoding="utf-8") as vf:
            for line in vf:
                ok, _ = schema.validate(json.loads(line))
                if not ok:
                    n_badval += 1
                    continue
                w.write(line if line.endswith("\n") else line + "\n")
                n_out += 1
        # added
        for ex in final_added:
            ok, reason = schema.validate(ex)
            if not ok:
                n_badval += 1
                continue
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_out += 1
            added_rows_out += 1
            added_mass += n_calls(ex)
    os.remove(v2_tmp)

    # ===================== STEP 2: context-aware em/en-dash pass over PROSE ONLY =====================
    # (a) ANALYZE: scan the full train_v4 prose (think + assistant content), tally top context buckets.
    print("\nanalyzing em/en-dash contexts in prose (think + assistant content only) ...", flush=True)
    ctx_counter = Counter()
    group_counter = Counter()
    n_dash_rows_before = 0
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            row_has = False
            for m in ex.get("messages", []):
                if m.get("role") == "assistant":
                    for fld in ("reasoning_content", "content"):
                        v = m.get(fld)
                        if isinstance(v, str) and _DASH_ANY.search(v):
                            analyze_dashes_prose(v, ctx_counter, group_counter)
                            row_has = True
            if row_has:
                n_dash_rows_before += 1
    print("DASH ANALYSIS: %d prose rows contain em/en dashes; %d total dash occurrences." % (
        n_dash_rows_before, sum(ctx_counter.values())))
    print("Top ~10 surrounding-context patterns (context -> group it maps to -> replacement):")
    for label, cnt in ctx_counter.most_common(10):
        grp = label  # label already encodes the bucket; recover group from the most common mapping
        # find the group this label was classified into (stored alongside in _ctx_label via group_counter overall)
        print("  %6d  %-44s" % (cnt, label[:44]))
    print("Group totals -> replacement:")
    for g in ("range", "aside", "default"):
        print("  %-8s x%-7d -> '%s'  (%s)" % (
            g, group_counter.get(g, 0), _GROUP_REPL[g],
            {"range": "compound/numeric range, NO surrounding spaces",
             "aside": "spaced clause-join / parenthetical aside",
             "default": "everything else (single-sided, punctuation-adjacent)"}[g]))

    # (b) REPLACE in place over prose only; rewrite the file.
    print("applying context-aware replacement (code blocks / inline code / args / tool-results untouched) ...", flush=True)
    repl_counter = Counter()
    tmp2 = OUT + ".dash.tmp"
    with open(OUT, encoding="utf-8") as f, open(tmp2, "w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for m in ex.get("messages", []):
                if m.get("role") == "assistant":
                    for fld in ("reasoning_content", "content"):
                        v = m.get(fld)
                        if isinstance(v, str):
                            m[fld] = replace_dashes_prose(v, repl_counter)
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp2, OUT)
    print("replaced %d dashes in prose: %s" % (sum(repl_counter.values()), dict(repl_counter)))

    total_mass = v2_mass + added_mass
    # ---- REPORT ----
    print("\n================= SFT-v4 BUILD REPORT =================")
    print("%-12s %8s %10s %10s" % ("shard", "in", "after_cull", "final"))
    final_counts = {s: len(per_shard.get(s, [])) for s in ADDED}
    # recompute final per-shard after rebalance
    fc = Counter()
    # map back: we need per-shard final counts; recompute from cap_log row counts
    for s in ADDED:
        cl = cap_log.get(s, (0, 0, 0))
        fc[s] = cl[2]
    for s in ADDED:
        c = stats[s]
        print("%-12s %8d %10d %10d   (drops: step0=%d s1=%d s2=%d s3=%d s4=%d s5=%d s6=%d s7=%d s8=%d dup=%d trivial=%d shapecap=%d idxcap=%d)" % (
            s, c["in"], c["after_cull"], fc[s],
            c["step0"], c["step1"], c["step2"], c["step3"], c["step4"], c["step5"], c["step6"],
            c["step7"], c["step8"], c["step9_dup"], c["step9_trivial"], c["step9_shapecap"], c["step9_idxcap"]))
    print("-" * 54)
    print("v2 rows=%d (mass=%d)  added rows=%d (mass=%d)  total rows=%d" % (
        v2_rows, v2_mass, added_rows_out, added_mass, n_out))
    print("STEP10 cap (shard: mass_before -> mass_after, rows): %s" % {s: cap_log[s] for s in ADDED})
    print("added share of total tool-call mass = %.2f%% (target <= ~20%%)" % (100.0 * added_mass / max(1, total_mass)))
    print("schema.validate drops = %d" % n_badval)
    print("OUTPUT: %s" % OUT)
    print("====================================================\n")


if __name__ == "__main__":
    main()
