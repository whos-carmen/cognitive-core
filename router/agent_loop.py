#!/usr/bin/env python3
"""Cognitive Core — Agent Loop with MCP tool support

Connects to MCP servers (Tavily, etc.), runs the router model,
executes tool calls, and feeds results back for final answers.

Usage:
    export TAVILY_API_KEY="your-key"
    python agent_loop.py                  # Interactive CLI
    python agent_loop.py "your question"   # Single shot
"""

import asyncio
import json
import os
import sys
import re
from datetime import datetime
from openai import OpenAI

# ── Paths ──
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tools_config.json")
TRACES_PATH = "/var/log/cognitive-core/traces.jsonl"
ROUTER_URL = "http://localhost:8081/v1"
RAG_URL = "http://localhost:8082/v1"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Suppress torchao noise
os.environ["TORCHCODEC_DISABLE"] = "1"


# ═══════════════════════════════════════════
#  Tool Call Parser
# ═══════════════════════════════════════════

def parse_tool_calls(text: str) -> list[dict]:
    """Parse <function> and <tool_call> XML from model output."""
    calls = []

    # <function name="X"><param name="Y">value</param></function>
    for m in re.finditer(
        r'<function\s+name\s*=\s*"([^"]*)"\s*>(.*?)</function>',
        text, re.DOTALL
    ):
        name = m.group(1)
        inner = m.group(2)
        params = {}
        for pm in re.finditer(r'<param\s+name\s*=\s*"([^"]*)">(.*?)</param>', inner, re.DOTALL):
            val = pm.group(2).strip()
            if val.startswith("<![CDATA[") and val.endswith("]]>"):
                val = val[9:-3]
            params[pm.group(1)] = val
        calls.append({"name": name, "parameters": params})

    # <tool_call>{"name":"X","parameters":{...}}</tool_call>
    for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj:
                p = obj.get("parameters") or obj.get("arguments") or {}
                calls.append({"name": obj["name"], "parameters": p})
        except json.JSONDecodeError:
            pass

    return calls


# ═══════════════════════════════════════════
#  MCP Client
# ═══════════════════════════════════════════

class MCPManager:
    """Manages connections to MCP servers and dispatches tool calls."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self._sessions: dict[str, tuple] = {}  # name → (session, read, write)

    async def connect_all(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        servers = self.config.get("mcp_servers", {})
        if not servers:
            print("  No MCP servers configured.")
            return

        for name, cfg in servers.items():
            try:
                # Resolve env vars like ${TAVILY_API_KEY}
                env = {}
                for k, v in cfg.get("env", {}).items():
                    if v.startswith("${") and v.endswith("}"):
                        env[k] = os.environ.get(v[2:-1], "")
                    else:
                        env[k] = v

                params = StdioServerParameters(
                    command=cfg["command"],
                    args=cfg["args"],
                    env=env if env else None,
                )
                ctx_mgr = stdio_client(params)
                read, write = await ctx_mgr.__aenter__()
                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()

                # List tools to confirm connection
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                self._sessions[name] = (session, ctx_mgr)
                print(f"  ✓ {name}: {len(tools.tools)} tools ({', '.join(tool_names[:3])}...)")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    async def disconnect_all(self):
        for name, (session, ctx_mgr) in self._sessions.items():
            try:
                await session.__aexit__(None, None, None)
                await ctx_mgr.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()

    def get_tool_mappings(self) -> dict:
        return self.config.get("tool_mappings", {})

    async def call_tool(self, mcp_server: str, mcp_tool: str, params: dict) -> str:
        if mcp_server not in self._sessions:
            return f"Error: MCP server '{mcp_server}' not connected"
        session = self._sessions[mcp_server][0]
        try:
            result = await session.call_tool(mcp_tool, params)
            # Format result as text
            if hasattr(result, "content"):
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"Error calling {mcp_tool}: {e}"


# ═══════════════════════════════════════════
#  Agent Loop
# ═══════════════════════════════════════════

class Agent:
    def __init__(self, config_path: str = CONFIG_PATH):
        self.client = OpenAI(base_url=ROUTER_URL, api_key="not-needed")
        self.mcp = MCPManager(config_path)
        self.tool_mappings = {}
        self._session_history = []
        self._reranker = None  # lazy-loaded

    async def start(self):
        print("Connecting to MCP servers...")
        await self.mcp.connect_all()
        self.tool_mappings = self.mcp.get_tool_mappings()
        print(f"  {len(self.tool_mappings)} tool mappings loaded")

    async def stop(self):
        await self.mcp.disconnect_all()

    async def run(self, prompt: str, system_prompt: str = None, max_turns: int = 5) -> str:
        """Run one prompt through the agent loop."""
        self._t_start = datetime.now()
        if system_prompt is None:
            sp_path = os.path.join(os.path.dirname(__file__), "configs", "system-prompt.md")
            try:
                with open(sp_path) as f:
                    system_prompt = f.read()
            except FileNotFoundError:
                system_prompt = "You are a cognitive core. Use tools when needed."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        for turn in range(max_turns):
            # ── Get model response ──
            response = self.client.chat.completions.create(
                model="minicpm5",
                messages=messages,
                max_tokens=500,
                stream=False,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = msg.reasoning_content or ""
            full_text = content + reasoning

            # ── Check for tool calls ──
            calls = parse_tool_calls(full_text)

            if not calls:
                # Check if the model refused or speculated (didn't use tools when it should)
                refusal_keywords = [
                    "i don't have access", "i cannot", "i apologize",
                    "i'm unable", "i am unable", "can't answer", "don't have",
                    "not have access", "no access", "cannot answer",
                ]
                is_refusal = any(kw in reasoning.lower() or kw in content.lower()
                               for kw in refusal_keywords)

                if is_refusal and turn == 0:
                    # Try RAG (Chroma + Granite) first
                    rag_answer = self._query_rag(prompt)
                    if rag_answer and "not have enough information" not in rag_answer.lower():
                        self._write_trace(prompt, "needs_knowledge", rag_answer)
                        return rag_answer

                    # RAG didn't have it → force web search
                    messages.append({
                        "role": "user",
                        "content": f"Please use the web_search tool to find information about this question. Search the web for: {prompt}"
                    })
                    continue

                # No tool calls and not a refusal → this is the final answer
                answer = content if content.strip() else reasoning.strip() or "(no response)"
                self._write_trace(prompt, "answer_directly", answer, reasoning)
                return answer

            # ── Execute tool calls ──
            for call in calls:
                tool_name = call["name"]
                tool_params = call["parameters"]

                mapping = self.tool_mappings.get(tool_name)
                if not mapping:
                    result = f"Unknown tool: {tool_name}"
                else:
                    # Map params from model format to MCP format
                    mcp_params = {}
                    param_map = mapping.get("param_mapping", {})
                    for model_key, mcp_key in param_map.items():
                        if model_key in tool_params:
                            val = tool_params[model_key]
                            # Convert string "true"/"false" to bool if needed
                            if val in ("true", "false"):
                                val = val == "true"
                            mcp_params[mcp_key] = val
                    # Add default params
                    for k, v in mapping.get("default_params", {}).items():
                        if k not in mcp_params:
                            mcp_params[k] = v

                    result = await self.mcp.call_tool(
                        mapping["mcp_server"],
                        mapping["mcp_tool"],
                        mcp_params,
                    )

                self._write_trace(prompt, f"tool_call:{tool_name}", result[:200])

                # For web search tools: synthesize results with Granite instead of 1B router
                if tool_name in ("web_search", "web_fetch", "tavily_search", "tavily_research"):
                    synthesis = self._synthesize(prompt, result)
                    return synthesis

                # For other tools: feed result back to router model
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "tool",
                    "content": str(result)[:2000],
                    "name": tool_name,
                })

        # Max turns reached — return last response
        return "Max tool call turns reached."

    def _get_reranker(self):
        """Lazy-load the LlamaNemotron reranker on first use."""
        if self._reranker is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            model_id = "nvidia/llama-nemotron-rerank-1b-v2"
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="cuda:0",
                trust_remote_code=True,
            )
            model.eval()
            self._reranker = (model, tok)
        return self._reranker

    def _rerank(self, query: str, docs: list[str]) -> list[int]:
        """Return indices of docs sorted by relevance (highest first)."""
        model, tokenizer = self._get_reranker()
        import torch
        pairs = [[query, d[:500]] for d in docs]
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1).cpu().float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def _query_rag(self, question: str, n_results: int = 5) -> str | None:
        """Query Chroma + reranker + Granite RAG model. Returns answer or None."""
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb

            embed_model = SentenceTransformer("ibm-granite/granite-embedding-english-r2")
            db = chromadb.PersistentClient(path=CHROMA_PATH)
            collection = db.get_or_create_collection("knowledge")

            total = collection.count()
            if total == 0:
                return None

            # Retrieve more chunks (15) for reranking
            q_emb = embed_model.encode([question], normalize_embeddings=True).tolist()[0]
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=min(15, total),
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            if not docs:
                return None

            # Rerank with LlamaNemotron, keep top n_results
            indices = self._rerank(question, docs)[:n_results]
            docs = [docs[i] for i in indices]
            metas = [metas[i] if metas else None for i in indices]

            context = "\n\n---\n\n".join(
                f"[Source: {m.get('source','?') if m else '?'}]\n{d}"
                for d, m in zip(docs, metas)
            )

            rag_system = "You are a knowledge assistant. Answer based ONLY on the context below. If the context doesn't contain the answer, say 'I don't have enough information to answer that.'"

            client = OpenAI(base_url=RAG_URL, api_key="not-needed")
            response = client.chat.completions.create(
                model="granite",
                messages=[
                    {"role": "system", "content": rag_system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                max_tokens=400,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  RAG error: {e}")
            return None

    def _synthesize(self, question: str, search_results: str) -> str:
        """Use Granite 4.1-8B to synthesize clean answer from search results."""
        try:
            client = OpenAI(base_url=RAG_URL, api_key="not-needed")
            response = client.chat.completions.create(
                model="granite",
                messages=[
                    {"role": "system", "content": "You are a research assistant. Given web search results and a question, produce a concise, accurate answer. Cite specific numbers and facts. If the results don't contain the answer, say so."},
                    {"role": "user", "content": f"Search results:\n{search_results[:3000]}\n\nQuestion: {question}"},
                ],
                max_tokens=500,
            )
            answer = response.choices[0].message.content or "(no response)"
            self._write_trace(question, "answer_directly", answer)
            return answer
        except Exception as e:
            return f"[Granite synthesis error: {e}]"

    def _write_trace(self, prompt, decision, content=None, reasoning=None):
        latency = 0
        if hasattr(self, '_t_start') and self._t_start:
            latency = round((datetime.now() - self._t_start).total_seconds() * 1000)
        trace = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "user": str(prompt)[:120],
            "latency_ms": latency,
            "reasoning_snippet": (reasoning or str(content or ""))[:200],
        }
        try:
            os.makedirs(os.path.dirname(TRACES_PATH), exist_ok=True)
            with open(TRACES_PATH, "a") as f:
                f.write(json.dumps(trace) + "\n")
        except (IOError, PermissionError):
            pass


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

async def main():
    agent = Agent()
    try:
        await agent.start()

        if len(sys.argv) > 1:
            # Single shot
            prompt = " ".join(sys.argv[1:])
            print(f"\n> {prompt}")
            result = await agent.run(prompt)
            print(f"\n{result}")
        else:
            # Interactive
            print("\nCognitive Core Agent Loop — type 'exit' to quit")
            print("=" * 50)
            while True:
                try:
                    prompt = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not prompt:
                    continue
                if prompt.lower() in ("exit", "quit"):
                    break
                result = await agent.run(prompt)
                print(f"\n{result}")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
