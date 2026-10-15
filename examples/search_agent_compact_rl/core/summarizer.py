from __future__ import annotations

import html
import json
import logging
import re
from typing import Any

import json5
from openai import AsyncOpenAI

from examples.search_agent_compact_rl.core.config import SummarizerConfig
from examples.search_agent_compact_rl.core.prompts import SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_TEMPLATE


logger = logging.getLogger(__name__)


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_SUMMARY_RE = re.compile(r"(?<!`)<summary>(?!`)(.*?)(?<!`)</summary>(?!`)", re.DOTALL)


def _escape_history_text(text: str) -> str:
    return html.escape(text.strip(), quote=False)


def _strip_blocks(text: str, patterns: list[re.Pattern[str]]) -> str:
    for pattern in patterns:
        text = pattern.sub("", text)
    return text


def _extract_thinking(content: str) -> str:
    match = _THINK_RE.search(content)
    if match:
        return _escape_history_text(match.group(1))
    return _escape_history_text(_strip_blocks(content, [_TOOL_CALL_RE, _ANSWER_RE]))


def _extract_answer(content: str) -> str:
    match = _ANSWER_RE.search(content)
    if not match:
        return ""
    return _escape_history_text(match.group(1))


def _extract_tool_response(content: str) -> str:
    match = _TOOL_RESPONSE_RE.search(content)
    if not match:
        return _escape_history_text(content)
    return _escape_history_text(match.group(1))


def _extract_tool_call_text(content: str) -> str:
    match = _TOOL_CALL_RE.search(content)
    if match:
        return match.group(1).strip()
    if "<tool_call>" not in content:
        return ""
    return content.split("<tool_call>", 1)[1].strip()


def _parse_tool_call(content: str) -> tuple[str, dict[str, Any], str] | None:
    raw_tool_call = _extract_tool_call_text(content)
    if not raw_tool_call:
        return None
    try:
        data = json.loads(raw_tool_call)
    except json.JSONDecodeError:
        try:
            data = json5.loads(raw_tool_call)
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    arguments = data.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name, arguments, raw_tool_call


def _format_tool_call(content: str) -> list[str]:
    parsed = _parse_tool_call(content)
    if parsed is None:
        raw_tool_call = _extract_tool_call_text(content)
        if raw_tool_call:
            return ["Tool Name: malformed tool call", f"Tool Call Text:\n{_escape_history_text(raw_tool_call)}"]
        answer = _extract_answer(content)
        if answer:
            return [f"Final Answer:\n{answer}"]
        residual = _escape_history_text(_strip_blocks(content, [_THINK_RE]))
        if residual:
            return [f"Assistant Output:\n{residual}"]
        return ["Tool Name: none"]

    name, arguments, _raw_tool_call = parsed
    lines = [f"Tool Name: {_escape_history_text(name)}"]
    if name == "search":
        lines.append(f"Search Query: {_escape_history_text(str(arguments.get('query', '')))}")
    elif name == "visit":
        lines.append(f"Visit URL: {_escape_history_text(str(arguments.get('url', '')))}")
        goal = arguments.get("goal")
        if goal is not None:
            lines.append(f"Visit Goal: {_escape_history_text(str(goal))}")
    elif arguments:
        lines.append(f"Tool Arguments:\n{_escape_history_text(json.dumps(arguments, ensure_ascii=False))}")
    return lines


def format_history_for_summary(messages: list[dict[str, str]]) -> str:
    """Render prior search-agent messages as inert text for compaction."""
    lines: list[str] = []
    query_index: int | None = None
    for idx, message in enumerate(messages):
        if message.get("role") == "user" and not message.get("content", "").lstrip().startswith("<tool_response>"):
            query_index = idx
            break

    if query_index is not None:
        lines.append(f"Current session query:\n{_escape_history_text(messages[query_index].get('content', ''))}")
        start_index = query_index + 1
    else:
        start_index = 1 if messages and messages[0].get("role") == "system" else 0

    lines.append("This session's tool interaction history:")
    turn_number = 1
    idx = start_index
    while idx < len(messages):
        message = messages[idx]
        if message.get("role") != "assistant":
            idx += 1
            continue

        content = message.get("content", "")
        turn_lines = [f"Turn #{turn_number}", f"Thinking Process:\n{_extract_thinking(content) or '(none)'}"]
        turn_lines.extend(_format_tool_call(content))

        next_idx = idx + 1
        if next_idx < len(messages) and messages[next_idx].get("role") == "user":
            turn_lines.append(f"Tool Response:\n{_extract_tool_response(messages[next_idx].get('content', ''))}")
            idx = next_idx + 1
        else:
            idx += 1

        lines.append("\n".join(turn_lines))
        turn_number += 1

    return "\n\n".join(lines)


class Summarizer:
    def __init__(self, config: SummarizerConfig, tokenizer: Any, on_policy_base_url: str):
        self.config = config
        self.tokenizer = tokenizer
        self.on_policy_base_url = on_policy_base_url
        self._off_policy_client: AsyncOpenAI | None = None
        if config.mode == "off_policy":
            base_url = config.off_policy_base_url
            if "xiaoma.best" in base_url and not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            self._off_policy_client = AsyncOpenAI(
                base_url=base_url,
                api_key=config.off_policy_api_key,
            )

    def _build_messages(self, history_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARIZER_USER_TEMPLATE.format(history=history_text)},
        ]

    def extract_summary(self, text: str) -> str:
        matches = list(_SUMMARY_RE.finditer(text))
        if not matches:
            return ""
        return matches[-1].group(1).strip()

    async def compact(self, messages: list[dict[str, str]]) -> str:
        history_text = format_history_for_summary(messages)
        summary_text = await self._call_summarizer(history_text)
        summary = self.extract_summary(summary_text)
        if not summary:
            raise ValueError("Summarizer did not produce a <summary> block")
        return summary

    async def _call_summarizer(self, history_text: str) -> str:
        messages = self._build_messages(history_text)
        if self.config.mode == "off_policy":
            client = self._off_policy_client
            model_name = self.config.off_policy_model_name
        else:
            client = AsyncOpenAI(base_url=self.on_policy_base_url, api_key="dummy")
            model_name = "policy"
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=self.config.off_policy_max_tokens,
            temperature=self.config.temperature,
        )
        if not hasattr(response, "choices") or not response.choices:
            logger.warning(
                "Summarizer returned a non-standard response (type=%s): %.200r",
                type(response).__name__,
                response,
            )
            return ""
        return response.choices[0].message.content or ""
