#!/usr/bin/env python3
"""Test the Cognitive Core router.

Usage:
    python test_prompt.py                           # Default: direct answer
    python test_prompt.py --tool-test                # Test tool-call format
    python test_prompt.py "Your question here"       # Custom prompt
    python test_prompt.py --parse                    # Show parsed tool calls
"""

import sys, json, os
from datetime import datetime
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="not-needed")
prompt_path = os.path.join(os.path.dirname(__file__), "configs", "system-prompt.md")
with open(prompt_path) as f:
    system_prompt = f.read()

# Import tool parser (optional — only if available)
tool_parser = None
tool_parser_path = os.path.join(os.path.dirname(__file__), "eval", "tool_parser.py")
if os.path.exists(tool_parser_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("tool_parser", tool_parser_path)
    tool_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_parser)
    tool_parser = tool_parser.ToolCallParser()


def chat(user_msg: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    t0 = datetime.now()
    r = client.chat.completions.create(
        model="minicpm5",
        messages=messages,
        max_tokens=500,
        stream=False,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    msg = r.choices[0].message
    content = msg.content or ""
    reasoning = msg.reasoning_content or ""
    usage = r.usage

    # Parse tool calls
    tool_calls = []
    if tool_parser:
        tool_calls = tool_parser.parse(content + reasoning)

    return {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "tokens": usage.completion_tokens + usage.prompt_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "elapsed_s": elapsed,
        "tps": round(usage.completion_tokens / elapsed) if elapsed > 0 else 0,
    }


if __name__ == "__main__":
    show_parse = "--parse" in sys.argv or "-p" in sys.argv

    if "--tool-test" in sys.argv:
        prompts = ["Search the web for the latest AMD RX 7900 XTX specs"]
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        prompts = [" ".join(a for a in sys.argv if not a.startswith("-"))]
    else:
        prompts = ["What is 2+2?", "Search the web for the latest AI news"]

    for prompt in prompts:
        print(f"\n> {prompt}")
        print("-" * 60)
        result = chat(prompt)

        if result["reasoning"]:
            print(f"[Reasoning]\n  {result['reasoning'][:200]}")

        print(f"[Response]\n  {result['content']}")

        if result["tool_calls"]:
            print(f"[Tool Calls]")
            for c in result["tool_calls"]:
                print(f"  -> {c['name']}({json.dumps(c['parameters'])})")

        print(f"[Stats] {result['completion_tokens']} tok in {result['elapsed_s']:.1f}s = {result['tps']} tok/s")
