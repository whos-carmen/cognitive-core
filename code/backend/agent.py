"""Agentic backend / eval harness for MiniCPM5-1B-Agent (CPU, llama-server GGUF).

The model is trained (schema.py) to: think concisely in <think>, then call tools via XML
  <function name="NAME"><param name="P">value</param></function>
(CDATA-wrapped when value has <, & or newline), read the <tool_response>, and loop
write->run->read->debug->patch->verify until it answers with no tool call.

Train<->serve parity: we build the prompt with the SAME tokenizer.apply_chat_template as training
(via data/schema.render), send token-ids to llama-server /completion, parse the XML the model emits,
execute tools in a sandbox, append role:"tool" results (capped with the SAME cap_tool_outputs), repeat.

This module is BOTH the Space backend and the eval harness (eval/run_eval.py drives it).
"""
import os, sys, re, json, time, tempfile, shutil, subprocess, urllib.request

# Paths are env-overridable so the SAME module runs locally (Windows defaults below) AND inside the
# deployed Docker Space (set CODEAGENT_PROJ=/app, CODEAGENT_LLAMA_BIN=llama-server). Defaults preserve
# local behavior exactly - no env vars needed for dev/eval.
PROJ = os.environ.get("CODEAGENT_PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LLAMA_BIN = os.environ.get("CODEAGENT_LLAMA_BIN", "llama-server")
sys.path.insert(0, os.path.join(PROJ, "data"))
import schema  # render + cap_tool_outputs (parity with training)

# ---- optional HTML self-correction: render the agent's .html in headless Chrome (ISOLATED .venv-browser)
# and feed JS/runtime errors back so the model FIXES them. Best-effort + swappable: LOCALLY this uses
# Selenium (html_check.py); in the deployed Space the FRONTEND captures window.onerror and supplies the
# same signal (BROWSER_CHECK_ENABLED auto-False there since the venv is absent -> no server-side Chrome).
_LT = os.path.normpath(os.path.join(PROJ, "..", "lora-train"))
_BROWSER_PY = os.path.join(_LT, ".venv-browser", "Scripts", "python.exe")
_HTML_CHECK = os.path.join(PROJ, "backend", "html_check.py")
BROWSER_CHECK_ENABLED = os.path.exists(_BROWSER_PY)


def browser_check(abs_path, timeout=45):
    """Render an .html file headless -> {ok,errors,console,title,body_size} or None if unavailable."""
    if not BROWSER_CHECK_ENABLED:
        return None
    try:
        r = subprocess.run([_BROWSER_PY, _HTML_CHECK, abs_path], capture_output=True, text=True, timeout=timeout)
        lines = [l for l in (r.stdout or "").strip().splitlines() if l.strip().startswith("{")]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def _html_feedback(rel, abs_path):
    """If rel is HTML, render it + return a one-line browser verdict to APPEND to the tool result
    (drives the build->render->see-error->fix loop). Empty string if not HTML or no local browser."""
    if not str(rel).lower().endswith((".html", ".htm")):
        return ""
    r = browser_check(abs_path)
    if r is None:
        return ""  # no local browser (e.g. the Space) -> the UI widget supplies window.onerror feedback instead
    if r.get("ok"):
        return f"\n[browser check] OK - renders with no JS errors (title={r.get('title')!r})."
    errs = "; ".join(r.get("errors", [])[:4]) or "unknown render error"
    return f"\n[browser check] FAILED - fix the HTML/JS and rewrite. JS errors: {errs}"

# ---------------------------------------------------------------- tools (implemented set) ----
# The model trained on PER-EXAMPLE tools (generalized tool use); at serve time we declare the set
# the sandbox actually implements. Keep names/params simple + agentic-coding focused.
# Tool names + params ALIGNED to the dominant trained vocabulary (Claude-Code suite, ~12k examples:
# bash/read/write/edit/glob/grep with command/file_path/old_string/new_string). The sandbox also accepts
# the SWE-style aliases (read_file/write_file/path/cmd/old_str) so it's robust to whatever the model emits.
TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Executes a bash command in the working directory and returns its stdout+stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command to run."}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Reads a file from the workspace and returns its content.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Path to the file (relative to the workspace)."}}, "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Writes (creates or overwrites) a file with the given content.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Exact string replacement in a file: replaces old_string with new_string.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}},
            "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "Fast file pattern matching; returns workspace paths matching a glob like '**/*.py'.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Searches file contents in the workspace with a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}, "required": ["pattern"]}}},
]

# Optional WEB tools - gated by CODEAGENT_ENABLE_WEB (OFF by default, so the Off-the-Grid/local demo + the frozen
# eval are unaffected). When ON, the model can search/read the web (it has latent web/browser tool-use from
# training). Validated on the HF Space datacenter IP: ddgs (search) + trafilatura (fetch/extract) work for general
# web; Reddit/JS-SPAs are refused and need a JS-rendering browser tier (camoufox - TODO). Using web forfeits the
# Off-the-Grid badge for that run (logged), so it's a deliberate, opt-in capability-vs-locality trade-off.
WEB_ENABLED = False     # turned on by enable_web(): via env at import, or AUTO-DETECTED by the app at startup
WEB_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web for current or factual information you don't already know. Returns the top results (title, url, snippet).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch", "description": "Fetch a web page by URL and return its main text as markdown. Use it on a URL from web_search to read the page.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "The page URL to read."}}, "required": ["url"]}}},
]

SYSTEM_PROMPT = (
    "You are a coding agent working in a fresh, empty working directory. Think briefly in <think>, then ACT by "
    "emitting tool calls. "
    "You MUST use the tools to do the work. NEVER put a file's contents in your reply - not in a markdown ``` block, "
    "and not as raw text (e.g. do not paste an <!DOCTYPE html> page into your answer). The ONLY way to deliver a file "
    "(a script, an HTML page, anything) is to CALL the write tool with file_path + content; then your final answer is "
    "just one short sentence. Never just describe what you would do. The directory "
    "starts EMPTY, so your FIRST action is normally a write - do NOT glob/read/grep for a file you have not created "
    "yet. (If a task gives you an existing file to fix, read it first.) After writing, run it with the bash tool "
    "(e.g. command='python add.py'). "
    "The sandbox runs Python with numpy, pandas, matplotlib (use the 'Agg' backend - no display) and Pillow already "
    "installed; if a task needs any other package, install it first with bash (command='pip install <package>'). "
    "DO NOT ask the user clarifying questions - make reasonable assumptions and PROCEED immediately. "
    "Use RELATIVE paths only (e.g. 'add.py', never '/workspace/add.py'). Write a small file in one write call; if a "
    "file would be long, write a short skeleton first and then use edit to fill it in (one giant write can corrupt "
    "the tool call). If the task produces a chart, plot, or image, SAVE it to a file with code (e.g. matplotlib "
    "savefig to a .png) so it can be shown to the user - never rely on an interactive display window. "
    "ALWAYS run the code with bash to verify it works before finishing; if it errors, read the "
    "output, fix it, and rerun. Only when it is verified working, give a short final answer with no further tool calls."
)
_WEB_HINT = (
    " You also have web access: call web_search(query) to find current/external information you don't know, then "
    "web_fetch(url) to read a result page (works for docs, news, Reddit threads, etc.). Use them ONLY when the task "
    "asks for a real-world FACT you don't have (a current price, a date, an API's docs). For a self-contained task - "
    "writing a script, or a static web page with made-up content - do NOT search; just write the file directly. When "
    "you do use the web, cite the source URL in your answer.")


def enable_web():
    """Idempotently turn ON the web tools (declare web_search/web_fetch + add the web hint to the system prompt).
    Safe to call AFTER import because run_agent reads TOOLS/SYSTEM_PROMPT at CALL time (tools=None/system=None)."""
    global WEB_ENABLED, TOOLS, SYSTEM_PROMPT
    if WEB_ENABLED:
        return
    WEB_ENABLED = True
    if not any((t.get("function") or {}).get("name") == "web_search" for t in TOOLS):
        TOOLS = TOOLS + WEB_TOOLS
    SYSTEM_PROMPT = SYSTEM_PROMPT + _WEB_HINT
    print("[agent] web tools ENABLED (web_search + web_fetch)", flush=True)


def web_available(timeout=5):
    """Quick reachability probe so the app can AUTO-enable web at startup (no manual on/off flag needed)."""
    for u in ("https://duckduckgo.com/", "https://en.wikipedia.org/"):
        try:
            urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (CodeAgent)"}), timeout=timeout)
            return True
        except Exception:
            continue
    return False


# explicit env override at import (CODEAGENT_ENABLE_WEB=1); otherwise the app auto-detects at startup via web_available()
if os.environ.get("CODEAGENT_ENABLE_WEB", "").lower() in ("1", "true", "yes", "on"):
    enable_web()

# Deterministic small-model steering knobs (all CPU-free; only fire when the 1B is demonstrably stuck, so a
# clean trajectory is unaffected - see backend notes / smallcode mapping). Tunable in one place.
# NOTE on <think>: in our agentic loop (tool results are role:"tool", single user task) the chat template keeps
# EVERY past assistant turn's <think> in context, and training SUPERVISED full multi-step think - so we store it
# VERBATIM (parity). It is trimmed ONLY under context-budget pressure, inside fit_context, and only on OLD turns.
# Live "burns the whole turn thinking" runaway is a SEPARATE concern bounded by n_predict, NOT by any history cap.
OLD_THINK_KEEP = 1000     # chars of reasoning_content kept on OLD assistant turns when fit_context must compact
READONLY_TOOLS = {"read", "read_file", "view", "cat", "glob", "grep", "search_files"}
EDIT_TOOLS = {"edit", "edit_file", "str_replace"}
_PUNT_RE = re.compile(
    r"\b(how (can|may) i (help|assist)|what (would|do) you (like|want|need)|let me know (if|what|how)|"
    r"happy to help|please (provide|clarify|specify|let me know)|could you (clarify|provide|specify)|"
    r"i'?m (ready|here) to (help|assist)|is there anything|feel free to)\b", re.I)


def _looks_like_punt(text):
    """True if the model produced no real work - empty, or a 'how can I help?' style greeting/clarification
    punt. A 1B sometimes regresses to this mid-task; we re-inject the task instead of accepting it as final."""
    t = (text or "").strip()
    return (not t) or bool(_PUNT_RE.search(t))


# ---------------------------------------------------------------- sandbox ----
def fuzzy_replace(text, old, new):
    """Replace old->new tolerantly (a 1B often gets whitespace/indentation slightly wrong, which breaks
    exact-match edits - the #1 small-model agent failure). Cascade: exact -> line-trimmed-block ->
    whitespace-collapsed. Returns (new_text|None, status) where status in ok|empty|multi|notfound."""
    if not old:
        return None, "empty"
    c = text.count(old)
    if c == 1:
        return text.replace(old, new, 1), "ok"
    if c > 1:
        return None, "multi"
    # line-trimmed block match (ignore per-line leading/trailing whitespace)
    tl = text.split("\n"); ol = old.split("\n"); n = len(ol)
    onorm = [x.strip() for x in ol]
    hits = [i for i in range(len(tl) - n + 1) if [x.strip() for x in tl[i:i + n]] == onorm]
    if len(hits) == 1:
        i = hits[0]
        return "\n".join(tl[:i] + new.split("\n") + tl[i + n:]), "ok"
    if len(hits) > 1:
        return None, "multi"
    # whitespace-collapsed single-substring match
    ws = lambda s: re.sub(r"\s+", " ", s).strip()
    ow = ws(old)
    if ow and "".join(text.split()).find("".join(old.split())) != -1:
        # locate by collapsing on a sliding window of the original lines
        for i in range(len(tl)):
            for j in range(i + 1, len(tl) + 1):
                if ws("\n".join(tl[i:j])) == ow:
                    return "\n".join(tl[:i] + new.split("\n") + tl[j:]), "ok"
    return None, "notfound"


def web_search(query, max_results=5):
    """No-key web search: ddgs (aggregates Google/Bing/Brave/...) with retry, then a Wikipedia-API fallback.
    Never raises; logs which backend it used to stdout (Space container logs). Validated on the HF datacenter IP."""
    query = (query or "").strip()
    if not query:
        return "[error] empty query"
    try:
        from ddgs import DDGS
        for attempt in range(3):
            try:
                rows = list(DDGS().text(query, max_results=max_results))
            except Exception as e:
                print(f"[web_search] ddgs attempt {attempt} error: {type(e).__name__}: {e}", flush=True)
                rows = []
            if rows:
                out = [f"[web_search via ddgs] top {len(rows)} results for {query!r}:"]
                for i, r in enumerate(rows, 1):
                    out.append(f"{i}. {r.get('title','')}\n   {r.get('href') or r.get('url','')}\n   {(r.get('body') or '')[:200]}")
                return "\n".join(out)[:2200]
            time.sleep(1.0 * (attempt + 1))
        print("[web_search] ddgs empty after retries -> Wikipedia fallback", flush=True)
    except Exception as e:
        print(f"[web_search] ddgs unavailable ({type(e).__name__}: {e}) -> Wikipedia fallback", flush=True)
    try:
        import urllib.parse
        u = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=5&srsearch="
             + urllib.parse.quote(query))
        req = urllib.request.Request(u, headers={"User-Agent": "MiniCPM5-Agent/1.0 (HF Space; hackathon)"})
        with urllib.request.urlopen(req, timeout=10) as r:
            hits = json.loads(r.read()).get("query", {}).get("search", [])
        if hits:
            out = [f"[web_search via Wikipedia - general web blocked/empty] top {len(hits)} for {query!r}:"]
            for i, h in enumerate(hits, 1):
                title = h.get("title", ""); snip = re.sub("<[^>]+>", "", h.get("snippet", ""))
                link = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                out.append(f"{i}. {title}\n   {link}\n   {snip[:200]}")
            return "\n".join(out)[:2200]
    except Exception as e:
        print(f"[web_search] wikipedia fallback failed: {type(e).__name__}: {e}", flush=True)
        return f"NO_RESULTS: search blocked/empty for {query!r} ({type(e).__name__})"
    return f"NO_RESULTS: nothing found for {query!r}"


_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/124.0.0.0 Safari/537.36")


def _camoufox_render(url, timeout_ms=35000):
    """Tier-2: render a JS / anti-bot page with camoufox (anti-detect Firefox) -> HTML, or None if camoufox is
    not installed or fails. Heavy (~10s/page) so it's used ONLY as a fallback when plain-HTTP extraction is empty.
    Auto-disabled where camoufox isn't installed (e.g. a Space without it) -> graceful degradation."""
    try:
        from camoufox.sync_api import Camoufox
    except Exception:
        return None
    cf = html = None
    try:
        cf = Camoufox(headless=True)
        browser = cf.__enter__()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(2500)                   # let the SPA hydrate
        html = page.content()                         # capture BEFORE close (close can crash on Win/2-vCPU)
    except Exception as e:
        print(f"[web_fetch] camoufox render failed for {url}: {type(e).__name__}: {e}", flush=True)
    finally:
        if cf is not None:
            try:
                cf.__exit__(None, None, None)
            except Exception:
                pass
    return html or None


# Shared Reddit-POST URL matcher: group(1) is the post id. Used both to ROUTE a fetch to the keyless
# arctic_shift archive (web_fetch) and to PULL the post id from that same URL (_reddit_via_arctic).
_REDDIT_POST_RE = re.compile(r"reddit\.com/(?:r/[^/]+/)?comments/([a-z0-9]+)", re.I)


def _reddit_via_arctic(url, max_comments=15):
    """Reddit blocks datacenter IPs + walls content behind JS, so direct fetch fails from a Space. The
    arctic_shift community ARCHIVE (photon-reddit) serves a Reddit post + its comments via a KEYLESS API and
    is NOT Reddit (so the datacenter-IP block doesn't apply). Returns formatted markdown, or None on miss."""
    m = _REDDIT_POST_RE.search(url)
    if not m:
        return None
    pid = m.group(1)
    base = "https://arctic-shift.photon-reddit.com/api"

    def _get(u):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": _BROWSER_UA})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"[web_fetch] arctic_shift {u} failed: {type(e).__name__}: {e}", flush=True)
            return None

    data = (_get(f"{base}/posts/ids?ids={pid}") or {}).get("data") or []
    if not data:
        return None
    p = data[0]
    out = [f"# {p.get('title','')}", f"r/{p.get('subreddit','')} · u/{p.get('author','')} · score {p.get('score','?')}", ""]
    if (p.get("selftext") or "").strip():
        out.append(p["selftext"].strip())
    cm = (_get(f"{base}/comments/search?link_id={pid}&limit={max_comments}&sort=desc") or {}).get("data") or []
    if cm:
        out.append("\n## Top comments")
        for c in cm:
            b = (c.get("body") or "").strip()
            if b:
                out.append(f"- u/{c.get('author','?')} ({c.get('score','?')}): {b}")
    return "\n".join(out)


def web_fetch(url, max_chars=6000):
    """Two-tier fetch+extract -> markdown.
    Tier 1 (fast; static / server-rendered): a real browser User-Agent (many sites - incl. Reddit - 403 the
      default lib UA) + auto reddit.com->old.reddit.com (server-rendered HTML, no JS), then trafilatura.extract.
    Tier 2 (only if tier 1 is empty/blocked AND camoufox is installed): render the JS page with camoufox + extract.
    Returns clear text on failure so the model can try another source."""
    url = (url or "").strip()
    if not url:
        return "[error] empty url"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    # Reddit POSTS: pull from the keyless arctic_shift archive (Reddit blocks datacenter IPs + JS-walls content,
    # so a direct fetch fails from a Space; the archive is a normal API server and isn't IP-blocked).
    if _REDDIT_POST_RE.search(url):
        arctic = _reddit_via_arctic(url)
        if arctic and arctic.strip():
            return (f"# {url}  (via arctic_shift Reddit archive)\n\n{arctic[:max_chars]}"
                    + ("\n...[truncated]" if len(arctic) > max_chars else ""))
    fetch_url = url
    if "reddit.com" in fetch_url and "old.reddit.com" not in fetch_url:    # non-post reddit URLs -> old.reddit
        fetch_url = re.sub(r"https?://(www\.|np\.|new\.)?reddit\.com", "https://old.reddit.com", fetch_url)
    # tier 1: browser-UA plain HTTP + trafilatura extract
    html_text = None
    try:
        req = urllib.request.Request(fetch_url, headers={"User-Agent": _BROWSER_UA, "Accept": "text/html,*/*"})
        with urllib.request.urlopen(req, timeout=12) as r:
            html_text = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"[web_fetch] tier1 browser-UA fetch failed for {fetch_url}: {type(e).__name__}: {e}", flush=True)
    txt, via = "", fetch_url
    try:
        import trafilatura
        if not html_text:                             # trafilatura's own fetcher as a secondary tier-1 attempt
            html_text = trafilatura.fetch_url(fetch_url)
        if html_text:
            txt = trafilatura.extract(html_text, output_format="markdown", include_links=False) or ""
    except Exception as e:
        print(f"[web_fetch] tier1 extract error {url}: {type(e).__name__}: {e}", flush=True)
    # tier 2: camoufox render of the ORIGINAL (JS) url if tier 1 produced nothing
    if not txt.strip():
        rendered = _camoufox_render(url)
        if rendered:
            try:
                import trafilatura
                txt = trafilatura.extract(rendered, output_format="markdown", include_links=False) or ""
                via = url + " [camoufox]"
            except Exception as e:
                print(f"[web_fetch] tier2 extract error {url}: {type(e).__name__}: {e}", flush=True)
    if not txt.strip():
        return (f"BLOCKED/EMPTY: couldn't get readable content from {url} (datacenter-IP blocked, or JS-only with "
                f"no browser available here). Try a different source.")
    note = f"  (read via {via})" if via != url else ""
    return f"# {url}{note}\n\n{txt[:max_chars]}" + ("\n...[truncated]" if len(txt) > max_chars else "")


def _run_bash_idle(cmd, cwd, env, idle_timeout, hard_cap, max_bytes):
    """Run a shell command, streaming output, with an INACTIVITY timeout: kill only if it goes SILENT for
    idle_timeout s (stuck), or after hard_cap s (absolute backstop), or if output exceeds max_bytes (runaway/
    spam). Long-but-progressing jobs (pip builds, training) keep running. Returns (output, returncode, note)."""
    import threading
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    buf, total, last, over, note = [], [0], [time.time()], [False], ""

    def _reader():
        try:
            for line in proc.stdout:
                buf.append(line); total[0] += len(line); last[0] = time.time()
                if total[0] > max_bytes:
                    over[0] = True
                    break
        except Exception:
            pass

    th = threading.Thread(target=_reader, daemon=True); th.start()
    start = time.time()
    while proc.poll() is None:
        time.sleep(0.4)
        now = time.time()
        if over[0]:
            note = f"\n[killed: output exceeded {max_bytes // 1024}KB - looks like a runaway/error-spam loop]"
        elif now - last[0] > idle_timeout:
            note = f"\n[killed: no new output for {idle_timeout}s - the process looks stuck/hung]"
        elif now - start > hard_cap:
            note = f"\n[killed: exceeded the {hard_cap}s hard limit]"
        else:
            continue
        try:
            proc.kill()
        except Exception:
            pass
        break
    try:
        proc.wait(timeout=5)
    except Exception:
        try: proc.kill()
        except Exception: pass
    th.join(timeout=2)
    return "".join(buf), proc.returncode, note


_PY3_SHIM_DIR = None


def _python3_shim_dir():
    """Windows-local-dev parity: the model (Linux-trained) often runs `python3 x.py`, which doesn't exist on
    Windows but DOES on the Linux Space. Provide a python3.bat -> python shim so local runs match the Space.
    Created once, OUTSIDE any workspace (so it never shows in the agent's file listing). Inert on Linux."""
    global _PY3_SHIM_DIR
    if os.name != "nt":
        return None
    if _PY3_SHIM_DIR is None:
        d = tempfile.mkdtemp(prefix="codeagent_bin_")
        with open(os.path.join(d, "python3.bat"), "w", encoding="utf-8") as f:
            f.write('@echo off\r\n"%s" %%*\r\n' % sys.executable)
        _PY3_SHIM_DIR = d
    return _PY3_SHIM_DIR


class Sandbox:
    """A temp working dir; tools operate only within it. bash runs with an inactivity timeout, cwd=workspace."""
    def __init__(self, bash_timeout=None):
        self.dir = tempfile.mkdtemp(prefix="agent_ws_")
        # bash uses an INACTIVITY timeout, NOT a hard wall-clock one: a long-but-progressing job (pip building a
        # wheel, training a small classifier, a slow download) keeps running as long as it emits output; we only
        # kill it if it goes SILENT for bash_idle seconds (stuck/hung). Plus a generous hard backstop and an
        # output-size cap (kills runaway/error-spam loops). All env-overridable.
        self.bash_idle = bash_timeout if bash_timeout is not None else int(os.environ.get("CODEAGENT_BASH_TIMEOUT", "150"))
        self.bash_hardcap = int(os.environ.get("CODEAGENT_BASH_HARDCAP", "1800"))           # absolute max seconds
        self.bash_maxbytes = int(os.environ.get("CODEAGENT_BASH_MAXBYTES", str(256 * 1024)))  # spam/runaway guard
        # ensure `python` is on PATH for the agent's run/verify steps (venv python dir prepended;
        # harmless on the Linux Space where python3 is already native)
        # SECURITY: the bash tool runs model-emitted shell with this env, so SCRUB any secret-looking var
        # (token/secret/api key) before it reaches the sandbox - a task must not be able to `echo $HF_TOKEN`.
        # (The app also pops HF_TOKEN after the model download; this is defense-in-depth for any future secret.)
        self.env = {k: v for k, v in os.environ.items()
                    if not any(s in k.upper() for s in ("TOKEN", "SECRET", "_KEY", "PASSWORD", "HUGGINGFACE"))}
        self.env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + self.env.get("PATH", "")
        _shim = _python3_shim_dir()                 # Windows: make `python3` resolve (Linux Space already has it)
        if _shim:
            self.env["PATH"] = _shim + os.pathsep + self.env["PATH"]
        # Force matplotlib headless: never open a GUI window (would BLOCK the bash call on a machine with a
        # display, and locally pops a figure window). With Agg, plt.show() is a harmless no-op and savefig
        # still writes the PNG - which the UI then shows inline (gr.Image bubble). Parity with the headless Space.
        self.env["MPLBACKEND"] = "Agg"
        # Force UTF-8 I/O so code that prints Unicode (arrows, sigma, box-drawing) does not crash on Windows
        # cp1252 with UnicodeEncodeError. The Linux Space is already UTF-8, so this is train/serve PARITY for
        # the local eval (an otherwise-correct program shouldn't FAIL a case purely on the local console codec).
        self.env["PYTHONUTF8"] = "1"
        self.env["PYTHONIOENCODING"] = "utf-8"

    def _resolve(self, path):
        # tolerate absolute-ish paths the model may emit (/workspace/x, /x) -> treat as workspace-relative
        path = str(path).strip().lstrip("/\\")
        for pre in ("workspace/", "workspace\\"):
            if path.startswith(pre):
                path = path[len(pre):]
        p = os.path.normpath(os.path.join(self.dir, path))
        if not (p == self.dir or p.startswith(self.dir + os.sep)):
            raise ValueError(f"path escapes workspace: {path}")
        return p

    def execute(self, name, args):
        gp = lambda *keys: next((args[k] for k in keys if isinstance(args, dict) and args.get(k) is not None), None)
        try:
            if name in ("bash", "shell", "terminal", "run", "process"):
                cmd = gp("command", "cmd") or ""
                # The model (Linux / Claude-Code habit) often uses ABSOLUTE /workspace/... paths. bash runs with
                # cwd = the sandbox, so rewrite /workspace/ -> relative. (write/read already strip it via _resolve,
                # but raw bash did NOT - this is exactly what made `python /workspace/chart.py` fail to find the file.)
                cmd = cmd.replace("/workspace/", "").replace("\\workspace\\", "").replace("/workspace", ".")
                out, rc, note = _run_bash_idle(cmd, self.dir, self.env, self.bash_idle,
                                               self.bash_hardcap, self.bash_maxbytes)
                out = out + note
                return out if out.strip() else f"[exit {rc}, no output]"
            if name in ("write", "write_file"):
                rel = gp("file_path", "path", "filename") or ""
                p = self._resolve(rel); os.makedirs(os.path.dirname(p) or self.dir, exist_ok=True)
                c = gp("content", "text", "new_str") or ""
                open(p, "w", encoding="utf-8").write(c)
                return f"Wrote {len(c)} chars to {rel}" + _html_feedback(rel, p)
            if name in ("read", "read_file", "view", "cat"):
                rel = gp("file_path", "path", "filename") or ""
                p = self._resolve(rel)
                # Reading a binary/image file as text returns garbage the 1B then loops on -> report it exists.
                _BIN = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".pdf", ".zip", ".gz",
                        ".pyc", ".so", ".dll", ".exe", ".bin", ".o", ".mp4", ".wav")
                def _read_file(fp, shown):
                    if os.path.splitext(fp)[1].lower() in _BIN:
                        return (f"[{shown}: binary file, {os.path.getsize(fp)} bytes - it exists and was "
                                f"created successfully (binary, not shown as text).]")
                    return open(fp, encoding="utf-8", errors="replace").read()
                if not os.path.exists(p):
                    # A 1B IGNORES "did you mean" text and re-reads the wrong name forever (it confuses its
                    # script stem with the output - e.g. reads bar_chart.png when it saved chart.png). So do NOT
                    # just suggest: DETERMINISTICALLY resolve - if there is a strong same-extension close match,
                    # read THAT (with a note) so the model gets what it wanted and stops looping.
                    import difflib
                    d = os.path.dirname(p) or self.dir
                    sib = sorted(os.listdir(d)) if os.path.isdir(d) else []
                    base = os.path.basename(p); ext = os.path.splitext(base)[1].lower()
                    pool = [s for s in sib if os.path.splitext(s)[1].lower() == ext] if ext else sib
                    near = difflib.get_close_matches(base, pool or sib, n=1, cutoff=0.6)
                    if near:
                        return (f"[note] '{rel}' does not exist; the closest match is '{near[0]}', reading it instead.\n"
                                + _read_file(os.path.join(d, near[0]), near[0]))
                    listing = "\n".join(sib[:30]) if sib else "(empty)"
                    return f"[error] file not found: {rel}. Files here:\n{listing}"
                if os.path.isdir(p):                        # reading a directory -> list it (the model often reads /workspace/)
                    items = sorted(os.listdir(p))
                    return f"[directory {rel or '.'}] contains:\n" + ("\n".join(items) if items else "(empty)")
                return _read_file(p, rel)
            if name in ("edit", "edit_file", "str_replace"):
                rel = gp("file_path", "path", "filename") or ""
                p = self._resolve(rel)
                old = gp("old_string", "old_str", "old") or ""
                new = gp("new_string", "new_str", "new") or ""
                if not os.path.exists(p):
                    return f"[error] file not found: {rel}. Read it first to get the exact path/content."
                txt = open(p, encoding="utf-8", errors="replace").read()
                out, st = fuzzy_replace(txt, old, new)
                if st == "ok":
                    open(p, "w", encoding="utf-8").write(out)
                    return f"Edited {rel}" + _html_feedback(rel, p)
                if st == "multi":
                    return f"[error] old_string matches multiple places in {rel} - add more surrounding context to make it unique."
                if st == "empty":
                    return f"[error] old_string is empty - provide the exact text to replace."
                return f"[error] old_string not found in {rel} (even allowing for whitespace). Re-read the file and copy the exact lines to change."
            if name == "glob":
                import glob as _g
                pat = gp("pattern", "glob", "path") or "**/*"
                ms = [os.path.relpath(m, self.dir) for m in _g.glob(os.path.join(self.dir, pat), recursive=True)]
                return "\n".join(sorted(ms)) if ms else "[no matches]"
            if name in ("grep", "search_files"):
                import re as _re
                pat = gp("pattern", "query", "regex") or ""
                try:
                    rx = _re.compile(pat)
                except Exception as e:
                    return f"[error] bad regex: {e}"
                hits = []
                for root, _, files in os.walk(self.dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            for i, ln in enumerate(open(fp, encoding="utf-8", errors="replace"), 1):
                                if rx.search(ln):
                                    hits.append(f"{os.path.relpath(fp, self.dir)}:{i}: {ln.strip()[:200]}")
                                    if len(hits) >= 50: break
                        except Exception:
                            pass
                    if len(hits) >= 50: break
                return "\n".join(hits) if hits else "[no matches]"
            # Web tools: the model only ever sees these when enable_web() declared them, so route the calls to
            # the module web_search/web_fetch implementations. (Without this branch they fell through to the
            # "unknown tool" error below, so every web call the model made failed.)
            if name in ("web_search", "websearch"):
                if not WEB_ENABLED:
                    return "[error] web tools are not enabled in this environment"
                return web_search(gp("query", "q") or "")
            if name in ("web_fetch", "webfetch", "fetch"):
                if not WEB_ENABLED:
                    return "[error] web tools are not enabled in this environment"
                return web_fetch(gp("url", "link") or "")
            return f"[error] unknown tool: {name}"
        except subprocess.TimeoutExpired:
            return f"[error] command timed out after {self.bash_idle}s of inactivity"
        except Exception as e:
            return f"[error] {type(e).__name__}: {e}"

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------- XML tool-call parsing ----
_FUNC_RE = re.compile(r'<function\s+name="([^"]+)"\s*>(.*?)</function>', re.DOTALL)
_PARAM_RE = re.compile(r'<param\s+name="([^"]+)"\s*>(.*?)</param>', re.DOTALL)
_CDATA_RE = re.compile(r'^\s*<!\[CDATA\[(.*?)\]\]>\s*$', re.DOTALL)
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def _unwrap(v):
    m = _CDATA_RE.match(v)
    return m.group(1) if m else v


def parse_assistant(text):
    """Return {reasoning, tool_calls:[{name,arguments}], final}. final is the answer text iff no tool calls."""
    think = _THINK_RE.search(text)
    reasoning = think.group(1).strip() if think else ""
    calls = []
    for fm in _FUNC_RE.finditer(text):
        name, body = fm.group(1), fm.group(2)
        args = {pn: _unwrap(pv) for pn, pv in _PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    final = ""
    if not calls:
        # strip the <think> block; whatever remains is the answer
        final = _THINK_RE.sub("", text).strip()
    return {"reasoning": reasoning, "tool_calls": calls, "final": final}


# ---------------------------------------------------------------- llama-server client ----
class LlamaServer:
    def __init__(self, gguf, port=8099, ctx=8192, threads=6, ngl=0):
        self.gguf, self.port, self.ctx, self.threads, self.ngl = gguf, port, ctx, threads, ngl
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [LLAMA_BIN, "-m", self.gguf, "--host", "127.0.0.1", "--port", str(self.port),
             "-c", str(self.ctx), "-t", str(self.threads), "-ngl", str(self.ngl), "--jinja"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{self.port}"
        for _ in range(120):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()

    def complete(self, token_ids, n_predict=1024, temperature=0.3, top_p=0.9, stop=None, grammar=None,
                 repeat_penalty=1.0, repeat_last_n=64):
        body = {"prompt": token_ids, "n_predict": n_predict, "temperature": temperature,
                "top_p": top_p, "cache_prompt": True, "stop": stop or ["<|im_end|>"],
                "repeat_penalty": repeat_penalty, "repeat_last_n": repeat_last_n,  # break small-model degenerate repetition loops (gentle: code legitimately repeats tokens)
                "return_tokens": True, "timings_per_token": True, "special": True}  # need raw tokens: <function>/<param> are special tokens stripped from `content`
        if grammar:
            body["grammar"] = grammar
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        # Time the call with a monotonic clock as a FALLBACK: llama-server's /completion response normally
        # carries a `timings` object (prompt_n/prompt_ms/prompt_per_second/predicted_n/predicted_ms/...), but if
        # it's absent we synthesize one from wall time + the generated token count so callers always get t/s.
        t0 = time.monotonic()
        # CPU generation is slow (~8 tok/s on a free 2-vCPU tier): a long turn can exceed a 600s HTTP timeout,
        # which raised and ended the run as "iters=1, tool-calls=0, empty". 1800s covers a full capped turn so
        # the generation COMPLETES instead of erroring out. (Eval runs at n_predict=1024 = far under this.)
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        wall_ms = (time.monotonic() - t0) * 1000.0
        if not isinstance(out.get("timings"), dict) or not out["timings"]:
            n_pred = len(out.get("tokens") or []) or out.get("tokens_predicted") or 0
            out["timings"] = {"prompt_n": 0, "prompt_ms": 0.0, "predicted_n": n_pred, "predicted_ms": wall_ms}
        return out  # full /completion JSON incl. tokens, content, and timings (real or synthesized)


# ---------------------------------------------------------------- agent loop ----
def _ntok(messages, tokenizer, tools, max_tool_chars):
    text = schema.render(schema.cap_tool_outputs(messages, max_tool_chars), tools, tokenizer,
                         enable_thinking=True, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_context(messages, tokenizer, tools, budget, max_tool_chars=8000):
    """Keep the live agent context within `budget` tokens (multi-turn sessions accumulate past the served ctx).
    Budget-GATED escalation (does NOTHING when already under budget → train/serve parity preserved in the common
    case): (1) elide older tool OUTPUTS (the bulk), keeping the 2 most recent verbatim; (2) if still over, trim
    reasoning_content on OLD assistant turns (all but the 2 most recent) to OLD_THINK_KEEP chars - keeps each
    turn's DECISION (tool_calls/content) + recent <think> full; (3) if still over, drop oldest post-task turns.
    The model was trained to consume its own full prior-turn <think>, so we touch it LAST and only under pressure,
    never unconditionally. Returns a compacted copy."""
    msgs = [dict(m) for m in messages]
    over = lambda: _ntok(msgs, tokenizer, tools, max_tool_chars) > budget
    if not over():
        return msgs
    # tier 1: elide all but the 2 newest tool RESULTS (they're the bulk)
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    for i in tool_idx[:-2]:
        msgs[i] = {"role": "tool", "name": msgs[i].get("name"), "content": "[earlier tool output elided to fit context]"}
    # tier 2: trim OLD assistant reasoning (all but the 2 most recent), keeping the decision intact
    if over():
        asst_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("reasoning_content")]
        for i in asst_idx[:-2]:
            r = msgs[i]["reasoning_content"]
            if len(r) > OLD_THINK_KEEP:
                mm = dict(msgs[i]); mm["reasoning_content"] = r[:OLD_THINK_KEEP] + " …[earlier reasoning trimmed]"
                msgs[i] = mm
    # tier 3: drop oldest post-task turns, preserving system[0] + task[1] + recent
    while over() and len(msgs) > 5:
        del msgs[2]  # drop oldest post-task turn
    return msgs


def run_agent(server, tokenizer, task, tools=None, system=None, max_iters=8,
              n_predict=1024, temperature=0.3, max_tool_chars=8000, verbose=False, keep_workspace=False,
              seed_files=None, sandbox=None, history=None, repeat_penalty=None, repeat_last_n=None):
    """Run the write->run->verify loop. Returns {messages, final, iters, tool_calls_made, workspace, sandbox}.
    tools/system default to the module globals AT CALL TIME (so a post-import enable_web() takes effect).
    MULTI-TURN: pass `sandbox` (a prior Sandbox) + `history` (prior messages) to CONTINUE the session in the SAME
    workspace - the new `task` is appended to the history and files from earlier turns persist (iterate without
    restarting). If keep_workspace, the caller cleans up result['sandbox'] later. seed_files pre-populates a NEW
    workspace before the agent acts (e.g. a broken repo to debug) -> real, ungameable tasks."""
    tools = tools if tools is not None else TOOLS
    system = system if system is not None else SYSTEM_PROMPT
    if repeat_penalty is None:   # eval/Space set CODEAGENT_REPEAT_PENALTY to break degenerate looping without a retrain
        repeat_penalty = float(os.environ.get("CODEAGENT_REPEAT_PENALTY", "1.0"))
    if repeat_last_n is None:
        repeat_last_n = int(os.environ.get("CODEAGENT_REPEAT_LAST_N", "64"))
    own_sb = sandbox is None
    sb = sandbox if sandbox is not None else Sandbox()
    if own_sb:
        for _rel, _content in (seed_files or {}).items():  # seed broken-repo / discovery files the agent must work with
            _p = sb._resolve(_rel); os.makedirs(os.path.dirname(_p) or sb.dir, exist_ok=True)
            open(_p, "w", encoding="utf-8").write(_content)
    if history:
        messages = list(history) + [{"role": "user", "content": task}]   # continue the same conversation/workspace
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    final, made, tool_counts, prev_sig, repeat = "", 0, {}, None, 0
    reinjects = readonly_streak = edit_fail_streak = web_search_streak = notfound_streak = 0  # deterministic stuck-detection (small-model steering)
    # token-speed accounting (aggregated across every complete() call in this turn): TG = total generated
    # tokens / total generation seconds; PP = total prompt (prefill) tokens / total prefill seconds.
    tg_tokens = tg_ms = pp_tokens = pp_ms = 0.0
    try:
        budget = max(2048, getattr(server, "ctx", 24576) - n_predict - 512)  # leave room for the response
        for it in range(max_iters):
            fitted = fit_context(messages, tokenizer, tools, budget, max_tool_chars)  # compact long multi-turn sessions
            capped = schema.cap_tool_outputs(fitted, max_tool_chars)
            text = schema.render(capped, tools, tokenizer, enable_thinking=True, add_generation_prompt=True)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            try:
                think_cap = int(os.environ.get("CODEAGENT_THINK_CAP", "1024"))
                if think_cap > 0:
                    # TWO-PHASE generation = the over-thinking fix. The 1B otherwise rambles in <think> for
                    # minutes (the demo-killer: never acts / acts wrong). PHASE A generates ONLY the <think>,
                    # capped at think_cap tokens, with a gentle rep-penalty (breaks degenerate think-loops),
                    # stopping at </think>. We then FORCE-CLOSE the think and PHASE B generates the ACTION with
                    # NO rep-penalty (penalizing repeats garbles code). Combined token shape == a single call, so
                    # parse_assistant is unchanged. CODEAGENT_THINK_CAP=0 restores the old single-call behavior.
                    oa = server.complete(ids, n_predict=think_cap, temperature=temperature,
                                         repeat_penalty=1.15, repeat_last_n=256, stop=["</think>"])
                    think_toks = list(oa.get("tokens") or [])
                    # phase A MAY already include the </think> stop string; ensure EXACTLY one close so the
                    # combined output is "[think]</think>\n[action]" - the exact shape parse_assistant expects
                    # (a double </think> leaks the think text into the final answer).
                    _adec = tokenizer.decode(think_toks, skip_special_tokens=False) if think_toks else ""
                    close_toks = [] if "</think>" in _adec else tokenizer("</think>\n", add_special_tokens=False)["input_ids"]
                    ob = server.complete(ids + think_toks + close_toks, n_predict=n_predict,
                                         temperature=temperature, repeat_penalty=repeat_penalty,
                                         repeat_last_n=repeat_last_n)
                    act_toks = list(ob.get("tokens") or [])
                    ta, tb = oa.get("timings") or {}, ob.get("timings") or {}
                    out = {"tokens": think_toks + close_toks + act_toks, "content": ob.get("content"),
                           "timings": {k: float(ta.get(k, 0) or 0) + float(tb.get(k, 0) or 0)
                                       for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}}
                else:
                    out = server.complete(ids, n_predict=n_predict, temperature=temperature,
                                          repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n)
            except Exception as e:
                # llama-server returns HTTP 400 when the prompt overflows ctx (a 1B over-iterating on a hard
                # task). Stop the loop GRACEFULLY with whatever we produced, instead of crashing the whole run.
                final = final or f"[stopped: ran past the {getattr(server, 'ctx', '?')}-token context limit on this task]"
                if verbose:
                    print(f"[stopped] complete() failed at iter {it}: {type(e).__name__}: {e}", flush=True)
                break
            tm = out.get("timings") or {}     # accumulate prefill (PP) + generation (TG) for the turn's t/s readout
            tg_tokens += float(tm.get("predicted_n") or 0); tg_ms += float(tm.get("predicted_ms") or 0)
            pp_tokens += float(tm.get("prompt_n") or 0); pp_ms += float(tm.get("prompt_ms") or 0)
            # <function>/<param> are special tokens stripped from `content`; decode raw tokens with the
            # HF tokenizer (skip_special_tokens=False) for exact train-format parity.
            toks = out.get("tokens")
            gen = tokenizer.decode(toks, skip_special_tokens=False) if toks else out.get("content", "")
            parsed = parse_assistant(gen)
            if verbose:
                print(f"--- iter {it} ---\n{gen[:800]}\n", flush=True)
            # record the assistant turn in canonical form
            amsg = {"role": "assistant"}
            if parsed["reasoning"]:
                amsg["reasoning_content"] = parsed["reasoning"]  # VERBATIM (train/serve parity); trimmed only under budget pressure in fit_context
            if parsed["tool_calls"]:
                amsg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                      for c in parsed["tool_calls"]]
            amsg["content"] = parsed["final"]
            messages.append(amsg)
            if not parsed["tool_calls"]:
                # no-action / no-ANSWER guard. Re-inject (up to 2x) when the model stops WITHOUT delivering:
                #  (a) made==0  -> it did nothing (greeting / bare plan / code pasted in markdown) -> force it to ACT.
                #  (b) used tools but the final is EMPTY -> it gathered data then quit with no answer (the
                #      "here's the tool log, I'm done" failure) -> force it to SYNTHESIZE a user-facing answer.
                final_txt = (parsed["final"] or "").strip()
                if (made == 0 or not final_txt) and reinjects < 2:
                    reinjects += 1
                    if made > 0:
                        nudge = ("You gathered information with the tools but did not actually answer the user. "
                                 "Now write the FINAL answer to their request using what you found: do the "
                                 "arithmetic / draw the conclusion and state it clearly in plain prose. Do NOT "
                                 "call any more tools.")
                    else:
                        nudge = ("You replied without using any tools, so nothing was created or run. You MUST use "
                                 "the tools: call `write` to create the file, then `bash` to run it. Emit a tool "
                                 "call now - do not answer in plain text or markdown.")
                    messages.append({"role": "user", "content": nudge})
                    if verbose: print(f"[steer] no-{'answer' if made else 'action'} -> re-injected ({reinjects})", flush=True)
                    continue
                final = parsed["final"]
                break
            # doom-loop breaker: identical tool call(s) repeated -> the 1B is stuck, stop wasting iters
            sig = json.dumps([(c["name"], c["arguments"]) for c in parsed["tool_calls"]], sort_keys=True)
            repeat = repeat + 1 if sig == prev_sig else 0
            prev_sig = sig
            if repeat >= 2:
                final = parsed["final"] or "[stopped: repeated identical tool call]"
                break
            iter_edit_failed = iter_notfound = False
            for c in parsed["tool_calls"]:
                made += 1
                tool_counts[c["name"]] = tool_counts.get(c["name"], 0) + 1  # per-tool usage -> empirical prune
                result = sb.execute(c["name"], c["arguments"])
                if c["name"] in EDIT_TOOLS and result.startswith("[error]"):
                    iter_edit_failed = True
                if c["name"] in ("read", "read_file", "view", "cat") and result.startswith("[error] file not found"):
                    iter_notfound = True
                messages.append({"role": "tool", "name": c["name"], "content": result})
            # deterministic stuck-steering: a 1B loops on failed edits, re-reads a wrong filename, reads forever, or web_searches forever.
            edit_fail_streak = edit_fail_streak + 1 if iter_edit_failed else 0
            notfound_streak = notfound_streak + 1 if iter_notfound else 0
            readonly_streak = readonly_streak + 1 if all(c["name"] in READONLY_TOOLS for c in parsed["tool_calls"]) else 0
            web_search_streak = web_search_streak + 1 if all(c["name"] == "web_search" for c in parsed["tool_calls"]) else 0
            nudge = None
            if notfound_streak >= 2:   # re-reading a non-existent name; priority over readonly (rewriting the file does NOT help)
                nudge = ("You keep reading a file that does not exist. STOP guessing the name - look at the "
                         "'Files here:' / 'Did you mean' list in the error above and read that EXACT filename. "
                         "Your output was likely saved under a different name than your script.")
            elif edit_fail_streak >= 2:
                nudge = ("The edit keeps failing to match. Stop editing - use the write tool to rewrite the whole "
                         "file with the full corrected content, then run it.")
            elif web_search_streak >= 2:
                nudge = ("You already have web_search results above - STOP searching. Read the relevant figure/fact "
                         "from those snippets (or web_fetch ONE result URL once), then give your final answer using "
                         "it. Do NOT call web_search again.")
            elif readonly_streak >= 3:
                nudge = ("You've been reading/searching without writing. Write the code now with the write tool, "
                         "then run it to verify.")
            if nudge:
                messages.append({"role": "user", "content": nudge})
                edit_fail_streak = readonly_streak = web_search_streak = notfound_streak = 0  # reset so we steer, not spam
                if verbose: print(f"[steer] {nudge[:48]}...", flush=True)
        tps = {"tg": (tg_tokens / (tg_ms / 1000.0)) if tg_ms > 0 else 0.0,
               "pp": (pp_tokens / (pp_ms / 1000.0)) if pp_ms > 0 else 0.0,
               "gen_tokens": int(tg_tokens)}
        return {"messages": messages, "final": final, "iters": it + 1, "tool_calls_made": made,
                "tool_counts": tool_counts, "workspace": sb.dir, "sandbox": sb, "tps": tps}
    finally:
        if own_sb and not keep_workspace:   # don't clean a sandbox the caller owns (multi-turn session)
            sb.cleanup()


if __name__ == "__main__":
    # quick self-test against a GGUF passed as argv[1] (defaults to stock Q8)
    from transformers import AutoTokenizer
    gguf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "gguf", "stock-Q8_0.gguf")
    tok = AutoTokenizer.from_pretrained(os.path.join(PROJ, "model", "final"), trust_remote_code=True)
    task = ("Create add.py with a function add(a,b) that returns a+b, then run a quick test that "
            "prints add(2,3) and confirm it outputs 5.")
    with LlamaServer(gguf, ctx=8192) as srv:
        res = run_agent(srv, tok, task, verbose=True)
    print("\n==== FINAL ====\n", res["final"])
    print(f"iters={res['iters']} tool_calls={res['tool_calls_made']}")
