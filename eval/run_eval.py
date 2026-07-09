#!/usr/bin/env python3
"""Cognitive Core — Routing Evaluation Runner

Evaluates a model's ability to route between answering, tool calling,
delegating to oracles, reasoning, and abstaining.

Usage:
    # Ollama (default)
    python eval/run_eval.py --model cognitive-core

    # llama.cpp server (OpenAI-compatible)
    python eval/run_eval.py --model cognitive-core --base-url http://localhost:8080/v1

    # With specific quantization
    python eval/run_eval.py --model cognitive-core --quant Q8_0

    # Generate report only (from saved results)
    python eval/run_eval.py --report-only --results eval/results.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

try:
    import httpx
    HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    HTTPX = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(PROJ, "eval", "dataset.jsonl")
RESULTS_DIR = os.path.join(PROJ, "eval", "results")

SYSTEM_PROMPT = """You are a cognitive core — a routing assistant that handles tasks locally when possible and delegates when it doesn't know.

Rules:
- If you can answer from reasoning or logic → answer directly
- If you need to calculate or run code → emit a tool call
- If you need current/real-time data you don't have → say you need to look it up or delegate
- If the question requires opinion or judgment → reason through it step by step
- If you're not confident or the question is unanswerable → say so honestly

Never guess on facts. Never fabricate data. It's better to delegate or abstain than to hallucinate.

When you need to use a tool, emit:
<tool_call>{"name": "tool_name", "parameters": {"key": "value"}}</tool_call>

When you need to delegate, say something like: "I'd need to look that up" or "Let me search for that"

When you're uncertain, acknowledge it: "I'm not sure about this, but..." or "This varies, but roughly..."
"""

# Category expected actions
EXPECTED = {
    "A": "answer",
    "B": "tool_call",
    "C": "delegate",
    "D": "answer",  # with reasoning
    "E": "abstain",
}


def call_ollama(model: str, prompt: str, system: str = "", base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    if HTTPX:
        r = httpx.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    else:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["message"]["content"]


def call_openai_compatible(model: str, prompt: str, system: str, base_url: str) -> str:
    """Call any OpenAI-compatible API (llama.cpp server, vLLM, etc)."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    if HTTPX:
        r = httpx.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    else:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]


def classify_response(response: str, expected_action: str) -> dict:
    """Classify model response into an action category."""
    resp_lower = response.lower()

    # Tool call detection
    has_tool_call = bool(re.search(r'<tool_call>|function_call|"name"\s*:', response))

    # Delegation detection
    delegation_phrases = [
        "i need to look", "let me search", "i'd need to look",
        "i should look up", "let me find out", "i'll search",
        "i don't have access", "i can't look up", "i would need to check",
        "let me check", "i don't have real-time", "i don't have current",
        "this requires looking up", "i'd need to check",
    ]
    has_delegation = any(phrase in resp_lower for phrase in delegation_phrases)

    # Abstention / uncertainty detection
    abstention_phrases = [
        "i'm not sure", "i'm uncertain", "i don't know",
        "i can't provide", "i cannot provide", "i shouldn't guess",
        "this is unknowable", "i can't know", "i'm unable to determine",
        "i don't have access to", "i can't access",
        "i can't determine", "i cannot determine",
        "it's not possible to know", "this information isn't",
        "i shouldn't fabricate", "i won't guess",
    ]
    has_abstention = any(phrase in resp_lower for phrase in abstention_phrases)

    # Reasoning detection
    has_reasoning = bool(re.search(r'<think>|let me think|let me reason|step by step|on the other hand', resp_lower))

    # Classify
    if has_tool_call:
        detected = "tool_call"
    elif has_abstention:
        detected = "abstain"
    elif has_delegation:
        detected = "delegate"
    else:
        detected = "answer"

    correct = detected == expected_action

    return {
        "detected": detected,
        "correct": correct,
        "has_tool_call": has_tool_call,
        "has_delegation": has_delegation,
        "has_abstention": has_abstention,
        "has_reasoning": has_reasoning,
    }


def load_dataset():
    """Load evaluation dataset."""
    entries = []
    with open(DATASET) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_evaluation(args):
    """Run full evaluation."""
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} test cases")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"eval_{timestamp}.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Stats
    stats = {
        "total": 0,
        "correct": 0,
        "by_category": {},
        "by_difficulty": {},
    }
    for cat in ["A", "B", "C", "D", "E"]:
        stats["by_category"][cat] = {"total": 0, "correct": 0}

    results = []

    for i, entry in enumerate(dataset):
        cat = entry["category"]
        expected = entry["expected_action"]
        prompt = entry["prompt"]

        print(f"[{i+1}/{len(dataset)}] {entry['id']} ({cat}) ", end="", flush=True)

        try:
            if args.base_url:
                response = call_openai_compatible(
                    args.model, prompt, SYSTEM_PROMPT, args.base_url
                )
            else:
                response = call_ollama(args.model, prompt, SYSTEM_PROMPT)
        except Exception as e:
            print(f"ERROR: {e}")
            response = ""
            time.sleep(2)

        classification = classify_response(response, expected)

        result = {
            **entry,
            "response": response[:500],  # truncate for storage
            "classification": classification,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)

        # Update stats
        stats["total"] += 1
        stats["by_category"][cat]["total"] += 1
        if classification["correct"]:
            stats["correct"] += 1
            stats["by_category"][cat]["correct"] += 1

        status = "✓" if classification["correct"] else f"✗ ({classification['detected']} instead of {expected})"
        print(status)

        # Save incrementally
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Rate limit
        time.sleep(0.5)

    # Final report
    print("\n" + "=" * 60)
    print("  COGNITIVE CORE EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Total: {stats['total']} test cases")
    print(f"  Overall Accuracy: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.1f}%)\n")

    print("  By Category:")
    for cat in ["A", "B", "C", "D", "E"]:
        s = stats["by_category"][cat]
        label = {
            "A": "Answer Directly",
            "B": "Tool Calling",
            "C": "Delegate to Oracle",
            "D": "Reasoning (Uncertain)",
            "E": "Abstain / Hallucination Trap",
        }[cat]
        if s["total"] > 0:
            acc = s["correct"] / s["total"] * 100
            print(f"    {cat}: {label:<35} {s['correct']}/{s['total']} ({acc:.1f}%)")

    # Save summary
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "base_url": args.base_url,
        "stats": stats,
        "accuracy": round(stats["correct"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0,
    }
    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results: {results_file}")
    print(f"  Summary: {summary_file}")
    print("=" * 60)

    return summary


def generate_report(results_file: str):
    """Generate report from saved results."""
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))

    stats = {"total": 0, "correct": 0, "by_category": {}}
    for cat in ["A", "B", "C", "D", "E"]:
        stats["by_category"][cat] = {"total": 0, "correct": 0}

    for r in results:
        cat = r["category"]
        stats["total"] += 1
        stats["by_category"][cat]["total"] += 1
        if r["classification"]["correct"]:
            stats["correct"] += 1
            stats["by_category"][cat]["correct"] += 1

    print(f"  Total: {stats['total']}")
    print(f"  Overall: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.1f}%)")
    for cat in ["A", "B", "C", "D", "E"]:
        s = stats["by_category"][cat]
        if s["total"] > 0:
            print(f"  {cat}: {s['correct']}/{s['total']} ({s['correct']/s['total']*100:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cognitive Core Routing Evaluation")
    ap.add_argument("--model", default="cognitive-core", help="Model name (Ollama or HF)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible API URL (llama.cpp server, etc)")
    ap.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    ap.add_argument("--results", default=None, help="Results file for --report-only")
    args = ap.parse_args()

    if args.report_only:
        generate_report(args.results)
    else:
        run_evaluation(args)
