#!/usr/bin/env python3
"""Unified tool call parser — handles both MiniCPM5 and Luminia agent XML formats.

Usage:
    parser = ToolCallParser()
    calls = parser.parse(response_text)
    for call in calls:
        name = call["name"]
        params = call["parameters"]
        print(f"Tool: {name}, Params: {params}")
"""

import re
import json
import sys

class ToolCallParser:
    """Parser for LLM tool call XML tags.

    Handles these formats out of the box:
      <tool_call>{"name": "...", "parameters": {...}}</tool_call>
      <tool_call>{"name": "...", "arguments": {...}}</tool_call>
      <function name="..." parameters='{"key": "value"}' />
      <function name="...">{"key": "value"}</function>
      <function name="..."><param name="...">value</param></function>  (native MiniCPM5 format)
    """

    # Match MiniCPM5 format: <tool_call>JSON_BLOB</tool_call>
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL
    )

    # Match Luminia agent format: <function name="..." parameters='JSON' />
    # Also matches the native MiniCPM5 XML-param format:
    #   <function name="..."><param name="key">value</param>...</function>
    # Group 1: name
    # Group 2: parameters='...' (single-quoted)
    # Group 3: parameters="..." (double-quoted)
    # Group 4: text content between > and </function> (JSON or XML params)
    FUNCTION_PATTERN = re.compile(
        r'<function\s+'
        r'name\s*=\s*"([^"]*)"\s*'
        r'(?:parameters\s*=\s*\'([^\']*)\'\s*)?'
        r'(?:parameters\s*=\s*"([^"]*)"\s*)?'
        r'(?:>(.*?)</function>|/>)'
    )

    def parse(self, text: str) -> list[dict]:
        """Parse tool calls from model response text.

        Returns list of dicts with keys: name, parameters (dict)
        """
        calls = []

        # 1. MiniCPM5 <tool_call> format — JSON has {name, parameters}
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            parsed = self._parse_tool_call_json(match.group(1).strip())
            if parsed:
                calls.append(parsed)

        # 2. Luminia <function> format — name in attribute, JSON is just params
        for match in self.FUNCTION_PATTERN.finditer(text):
            name = match.group(1)
            params = None

            # Try parameters from single-quoted attribute (group 2)
            if match.group(2):
                params = self._parse_plain_json(match.group(2))
            # Try parameters from double-quoted attribute (group 3)
            if params is None and match.group(3):
                params = self._parse_plain_json(match.group(3))
            # Try content between tags (group 4)
            if params is None and match.group(4):
                inner = match.group(4).strip()
                # First try parsing as JSON
                params = self._parse_plain_json(inner)
                # If not JSON, try parsing as <param name="...">value</param> XML
                if params is None:
                    params = self._parse_param_xml(inner)

            if params is None:
                params = {}

            calls.append({"name": name, "parameters": params})

        # Deduplicate
        seen = set()
        unique = []
        for c in calls:
            key = (c["name"], json.dumps(c["parameters"], sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _parse_tool_call_json(self, text: str) -> dict | None:
        """Parse a tool_call JSON blob that has {name, parameters}."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "name" in obj:
                params = obj.get("parameters") or obj.get("arguments") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {"value": params}
                return {"name": obj["name"], "parameters": params}
        except json.JSONDecodeError:
            pass
        return None

    def _parse_plain_json(self, text: str) -> dict | None:
        """Parse a plain JSON dict (no name wrapper)."""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        return None

    PARAM_XML_PATTERN = re.compile(
        r'<param\s+name\s*=\s*"([^"]*)">(.*?)</param>', re.DOTALL
    )

    def _parse_param_xml(self, text: str) -> dict | None:
        """Parse <param name="key">value</param> XML into a dict."""
        matches = self.PARAM_XML_PATTERN.findall(text)
        if matches:
            params = {}
            for name, value in matches:
                value = value.strip()
                value = value.strip()
                # Strip CDATA wrapper
                if value.startswith("<![CDATA[") and value.endswith("]]>"):
                    value = value[9:-3]
                # Try parsing as JSON first
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
                params[name] = value
            return params
        return None


def main():
    text = sys.stdin.read()
    parser = ToolCallParser()
    calls = parser.parse(text)
    print(json.dumps(calls, indent=2))


if __name__ == "__main__":
    test_cases = [
        # MiniCPM5 format
        '<tool_call>{"name": "web_search", "parameters": {"query": "weather"}}</tool_call>',
        # Luminia format
        '<function name="web_search" parameters=\'{"query": "weather"}\' />',
        # Both in one response
        'Let me search.\n<tool_call>{"name": "web_search", "parameters": {"query": "Tokyo"}}</tool_call>\n\nNow calculate.\n<function name="calculator" parameters=\'{"expr": "2+2"}\' />',
        # Function with content instead of attribute
        '<function name="code_runner">{"language": "python", "code": "print(1)"}</function>',
        # Empty — no tool calls
        "The capital of France is Paris.",
        # Arguments instead of parameters
        '<tool_call>{"name": "web_search", "arguments": {"query": "news"}}</tool_call>',
    ]

    parser = ToolCallParser()
    print(f"{'Test':<10} {'Result':<60}")
    print("-" * 70)
    for tc in test_cases:
        result = parser.parse(tc)
        label = tc[:45].replace("\n", " ")
        print(f"{'OK' if result else '--':<10} {label}...")
        for c in result:
            print(f"           → {c['name']}({c['parameters']})")
