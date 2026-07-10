"""Normalize every web-SEARCH / web-FETCH synonym across the training data to the SINGLE canonical
tool the Space serves: web_search({query}) / web_fetch({url}) (our MCP-shaped schema). A small model
fragments if it sees the same web action under many names; this collapses them so train==serve.

CONSERVATIVE allowlist only — does NOT touch domain searches (search_transactions/medications/code),
tool/grep search (toolsearch/grep_search), or the interactive browser_* automation toolset (a different
paradigm we don't serve). Rewrites: assistant tool_calls (name + arg-key remap), tool DECLARATIONS
(replace synonym defs with the canonical web_search/web_fetch def, deduped), and role:tool result names.

  python data/converters/web_normalize.py <in.jsonl> [--inplace | --out OUT]
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
try:
    import agent
    _WS = next(t for t in agent.WEB_TOOLS if t["function"]["name"] == "web_search")
    _WF = next(t for t in agent.WEB_TOOLS if t["function"]["name"] == "web_fetch")
except Exception:                                   # fallback canonical defs (kept identical to agent.WEB_TOOLS)
    _WS = {"type": "function", "function": {"name": "web_search", "description": "Search the web for current or factual information you don't already know. Returns the top results (title, url, snippet).", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}}
    _WF = {"type": "function", "function": {"name": "web_fetch", "description": "Fetch a web page by URL and return its main text as markdown. Use it on a URL from web_search to read the page.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The page URL to read."}}, "required": ["url"]}}}

SEARCH_SYN = {"websearch", "web_search", "google_search", "googlesearch", "google_web_search",
              "bing_search", "duckduckgo_search", "ddg_search", "internet_search", "online_search",
              "web_query", "search_web", "search_internet"}
FETCH_SYN = {"webfetch", "web_fetch", "visit_page", "open_url", "openurl", "fetch_url",
             "read_url", "browse_url", "visit_url", "fetch_page", "open_page", "read_page", "get_webpage"}
_QUERY_KEYS = ("query", "input", "q", "search_query", "text", "keyword", "term", "search")
_URL_KEYS = ("url", "input", "link", "href", "page", "uri", "address")


def _first_str(args, keys):
    if isinstance(args, dict):
        for k in keys:
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for v in args.values():                     # last resort: first string value
            if isinstance(v, str) and v.strip():
                return v
    elif isinstance(args, str):
        return args
    return ""


def normalize_web_tools(ex, stats=None):
    """Rewrite synonyms -> canonical web_search/web_fetch in one {messages, tools} example. Returns ex."""
    def bump(k):
        if stats is not None:
            stats[k] = stats.get(k, 0) + 1

    has_ws = has_wf = False
    for m in ex.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", tc)
            nm = str(fn.get("name", "")).lower()
            if nm in SEARCH_SYN:
                if nm != "web_search":
                    fn["name"] = "web_search"; fn["arguments"] = {"query": _first_str(fn.get("arguments"), _QUERY_KEYS)}; bump("calls_search")
                has_ws = True
            elif nm in FETCH_SYN:
                if nm != "web_fetch":
                    fn["name"] = "web_fetch"; fn["arguments"] = {"url": _first_str(fn.get("arguments"), _URL_KEYS)}; bump("calls_fetch")
                has_wf = True
        if m.get("role") == "tool":
            tn = str(m.get("name", "")).lower()
            if tn in SEARCH_SYN and tn != "web_search":
                m["name"] = "web_search"; bump("results")
            elif tn in FETCH_SYN and tn != "web_fetch":
                m["name"] = "web_fetch"; bump("results")
    # declarations: drop synonym defs, ensure ONE canonical def for each used
    tools = ex.get("tools")
    if tools:
        new, seen = [], set()
        for t in tools:
            nm = str((t.get("function", t)).get("name", "")).lower()
            if nm in SEARCH_SYN:
                if "web_search" not in seen:
                    new.append(_WS); seen.add("web_search")
                if nm != "web_search":
                    bump("decls_search")
            elif nm in FETCH_SYN:
                if "web_fetch" not in seen:
                    new.append(_WF); seen.add("web_fetch")
                if nm != "web_fetch":
                    bump("decls_fetch")
            else:
                key = (t.get("function", t)).get("name")
                if key not in seen:
                    new.append(t); seen.add(key)
        ex["tools"] = new
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = args.src if args.inplace else (args.out or args.src + ".norm")
    stats = {}; n = changed = 0
    tmp = out + ".tmp"
    with open(args.src, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as w:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            ex = json.loads(line)
            before = json.dumps(ex.get("tools"), ensure_ascii=False) + json.dumps(
                [[(c.get("function", c)).get("name") for c in (m.get("tool_calls") or [])] for m in ex.get("messages", [])])
            normalize_web_tools(ex, stats)
            after = json.dumps(ex.get("tools"), ensure_ascii=False) + json.dumps(
                [[(c.get("function", c)).get("name") for c in (m.get("tool_calls") or [])] for m in ex.get("messages", [])])
            if before != after:
                changed += 1
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(f"normalized {n} rows -> {out} | rows_changed={changed} | renames {stats}")


if __name__ == "__main__":
    main()
