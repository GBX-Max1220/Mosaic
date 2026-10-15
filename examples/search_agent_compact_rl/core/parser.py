from __future__ import annotations

import re
from enum import IntEnum
from typing import Any

from examples.search_agent_compact_rl.core.prompts import build_tool_schemas
from examples.search_agent_compact_rl.core.tool_parser import extract_tool_calls


class GenerationFlag(IntEnum):
    ERROR = 2
    CALL = 1
    END = 0


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STANDALONE_ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL)


def _sanitize_content(content: str) -> str:
    if "<think>" in content and "</think>" not in content:
        content = content + "\n</think>"
    if "<think>" not in content and "</think>" in content:
        content = "<think>\n" + content
    return content


def parse_response(content: str, tools: list[dict] | None = None) -> tuple[GenerationFlag, str, Any]:
    """Parse assistant output into (flag, thinking, payload)."""
    if "<tool_response>" in content:
        return GenerationFlag.ERROR, "", "Assistant output must not contain <tool_response>."

    content = _sanitize_content(content)

    think_match = _THINK_RE.search(content)
    if not think_match:
        standalone_answer_match = _STANDALONE_ANSWER_RE.fullmatch(content)
        if standalone_answer_match is None:
            return GenerationFlag.ERROR, "", "Missing <think></think>"
        answer = standalone_answer_match.group(1).strip()
        if not answer:
            return GenerationFlag.ERROR, "", "Answer is empty"
        lowered_answer = answer.lower()
        forbidden_tags = (
            "<answer",
            "</answer>",
            "<tool_call",
            "</tool_call>",
            "<tool_response",
            "</tool_response>",
        )
        if len(_ANSWER_RE.findall(content)) != 1 or any(
            tag in lowered_answer for tag in forbidden_tags
        ):
            return GenerationFlag.ERROR, "", "Ambiguous or incomplete output"
        return GenerationFlag.END, "", answer

    thinking = think_match.group(1).strip()
    parse_target = content[think_match.end() :]

    answer_matches = _ANSWER_RE.findall(parse_target)
    tool_call_attempts = parse_target.count("<tool_call>")

    if len(answer_matches) > 1:
        return GenerationFlag.ERROR, thinking, "Multiple final answers are not allowed"
    if tool_call_attempts > 1:
        return GenerationFlag.ERROR, thinking, "Multiple tool calls are not allowed"

    answer_match = answer_matches[0] if answer_matches else None

    if answer_match is not None and tool_call_attempts == 0:
        answer = answer_match.strip()
        if not answer:
            return GenerationFlag.ERROR, thinking, "Answer is empty"
        return GenerationFlag.END, thinking, answer

    if tool_call_attempts == 1 and answer_match is None:
        if tools is None:
            tools = build_tool_schemas()
        calls = extract_tool_calls(parse_target, tools)
        if len(calls) > 1:
            return GenerationFlag.ERROR, thinking, "Multiple tool calls are not allowed"
        if len(calls) == 1:
            schema_error = _validate_tool_schema(calls[0]["name"], calls[0]["arguments"])
            if schema_error is not None:
                return GenerationFlag.ERROR, thinking, schema_error
            return GenerationFlag.CALL, thinking, calls[0]

    return GenerationFlag.ERROR, thinking, "Ambiguous or incomplete output"


def _validate_tool_schema(name: Any, arguments: Any) -> str | None:
    if name not in {"search", "visit"}:
        return "Unknown tool. Available tools: search, visit"
    if not isinstance(arguments, dict):
        return "Tool call arguments must be a JSON object"
    if name == "search":
        unexpected = sorted(set(arguments) - {"query"})
        if unexpected:
            return f"Unexpected search.arguments fields: {', '.join(unexpected)}"
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "search.arguments.query must be a non-empty string"
        return None
    unexpected = sorted(set(arguments) - {"url", "goal"})
    if unexpected:
        return f"Unexpected visit.arguments fields: {', '.join(unexpected)}"
    url = arguments.get("url")
    goal = arguments.get("goal")
    if not isinstance(url, str) or not url.strip():
        return "visit.arguments.url must be a non-empty string"
    if not isinstance(goal, str) or not goal.strip():
        return "visit.arguments.goal must be a non-empty string"
    return None
