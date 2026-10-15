from __future__ import annotations

import re
from enum import IntEnum
from typing import Any

from examples.search_agent_subagent_rl.core.prompts import build_main_tool_schemas
from examples.search_agent_subagent_rl.core.tool_parser import extract_tool_calls


class GenerationFlag(IntEnum):
    ANSWER = 0
    TOOL_CALL = 1
    MALFORMED = 2


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STANDALONE_ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>|<tool_call>.*\Z", re.DOTALL)


def _sanitize_content(content: str) -> str:
    if "<think>" in content and "</think>" not in content:
        content = content + "\n</think>"
    if "<think>" not in content and "</think>" in content:
        content = "<think>\n" + content
    return content


def parse_response(
    content: str, tool_schemas: list[dict] | None = None
) -> tuple[GenerationFlag, str | None, Any]:
    """Parse assistant output into (flag, thinking, payload)."""
    if "<tool_response>" in content:
        return GenerationFlag.MALFORMED, None, "Assistant output must not contain <tool_response>."

    content = _sanitize_content(content)

    think_match = _THINK_RE.search(content)
    if not think_match:
        if content.count("<tool_call>") == 1:
            return _parse_response_after_thinking(content, tool_schemas, None)
        standalone_answer_match = _STANDALONE_ANSWER_RE.fullmatch(content)
        if standalone_answer_match is None:
            return GenerationFlag.MALFORMED, None, "Missing <think></think>"
        answer = standalone_answer_match.group(1).strip()
        if not answer:
            return GenerationFlag.MALFORMED, None, "Answer is empty"
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
            return GenerationFlag.MALFORMED, None, "Ambiguous or incomplete output"
        return GenerationFlag.ANSWER, None, answer

    return _parse_response_after_thinking(
        content[think_match.end() :],
        tool_schemas,
        think_match.group(1).strip(),
    )


def _parse_response_after_thinking(
    parse_target: str,
    tool_schemas: list[dict] | None,
    thinking: str | None,
) -> tuple[GenerationFlag, str | None, Any]:
    content_outside_tool_calls = _TOOL_CALL_BLOCK_RE.sub("", parse_target)
    answer_matches = _ANSWER_RE.findall(content_outside_tool_calls)
    tool_call_attempts = parse_target.count("<tool_call>")

    if len(answer_matches) > 1:
        return GenerationFlag.MALFORMED, thinking, "Multiple final answers are not allowed"
    if tool_call_attempts > 1:
        return GenerationFlag.MALFORMED, thinking, "Multiple tool calls are not allowed"

    answer_match = answer_matches[0] if answer_matches else None

    if answer_match is not None and tool_call_attempts == 0:
        answer = answer_match.strip()
        if not answer:
            return GenerationFlag.MALFORMED, thinking, "Answer is empty"
        return GenerationFlag.ANSWER, thinking, answer

    if tool_call_attempts == 1 and answer_match is None:
        if tool_schemas is None:
            tool_schemas = build_main_tool_schemas()
        calls = extract_tool_calls(parse_target, tool_schemas)
        if len(calls) > 1:
            return GenerationFlag.MALFORMED, thinking, "Multiple tool calls are not allowed"
        if len(calls) == 1:
            try:
                _validate_tool_schema(calls[0]["name"], calls[0]["arguments"], tool_schemas)
            except ValueError as exc:
                return GenerationFlag.MALFORMED, thinking, str(exc)
            return GenerationFlag.TOOL_CALL, thinking, calls[0]

    return GenerationFlag.MALFORMED, thinking, "Ambiguous or incomplete output"


def _available_tool_names(tool_schemas: list[dict]) -> set[str]:
    names: set[str] = set()
    for schema in tool_schemas:
        function = schema.get("function", schema)
        name = function.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _validate_tool_schema(name: Any, arguments: Any, tool_schemas: list[dict]) -> None:
    available_names = _available_tool_names(tool_schemas)
    if name not in available_names:
        available = ", ".join(sorted(available_names))
        raise ValueError(f"Unknown tool. Available tools: {available}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be a JSON object")
    if name == "search":
        unexpected = sorted(set(arguments) - {"query"})
        if unexpected:
            raise ValueError(f"Unexpected search.arguments fields: {', '.join(unexpected)}")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search.arguments.query must be a non-empty string")
        return
    if name == "visit":
        unexpected = sorted(set(arguments) - {"url", "goal"})
        if unexpected:
            raise ValueError(f"Unexpected visit.arguments fields: {', '.join(unexpected)}")
        url = arguments.get("url")
        goal = arguments.get("goal")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("visit.arguments.url must be a non-empty string")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("visit.arguments.goal must be a non-empty string")
        return
    if name == "fork":
        subagents = arguments.get("subagents")
        if not isinstance(subagents, list) or not 1 <= len(subagents) <= 3:
            raise ValueError("fork arguments.subagents must contain 1 to 3 items")
        for item in subagents:
            if not isinstance(item, dict) or set(item) != {"task"}:
                raise ValueError("each fork subagent must contain exactly one task")
            if not isinstance(item["task"], str) or not item["task"].strip():
                raise ValueError("fork subagent task must be a non-empty string")
