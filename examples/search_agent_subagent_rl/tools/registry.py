from __future__ import annotations

import json
import json5
import os
from typing import Any

from examples.search_agent_subagent_rl.tools import web


def _single_nonempty_str(value: Any, field_name: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, f"{field_name} must be a non-empty string"
    return value, None


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Convert model-provided tool arguments into a dict.

    The model sometimes emits the ``arguments`` field as a JSON-encoded string
    (or even a list) instead of an object. Parse it when necessary so the rest
    of the tool code can safely call ``.get()``.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, list):
        arguments = arguments[0] if arguments else ""
    if isinstance(arguments, str):
        arguments = arguments.strip()
        if not arguments:
            return {}
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = json5.loads(arguments)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            return parsed
        # The model may have given a bare string for the primary argument.
        return {"query": arguments}
    return {}


def _unexpected_argument_error(name: str, arguments: dict[str, Any]) -> str | None:
    allowed = {"query"} if name == "search" else {"url", "goal"}
    unexpected = sorted(set(arguments) - allowed)
    if unexpected:
        return f"Unexpected {name}.arguments fields: {', '.join(unexpected)}"
    return None


async def execute_tool(
    name: str,
    arguments: Any,
) -> str:
    arguments = _coerce_arguments(arguments)
    if name == "search":
        unexpected_error = _unexpected_argument_error(name, arguments)
        if unexpected_error is not None:
            return f"Tool execution error: {unexpected_error}"
        query, error = _single_nonempty_str(arguments.get("query"), "search.arguments.query")
        if error is not None:
            return f"Tool execution error: {error}"
    elif name == "visit":
        unexpected_error = _unexpected_argument_error(name, arguments)
        if unexpected_error is not None:
            return f"Tool execution error: {unexpected_error}"
        url, url_error = _single_nonempty_str(arguments.get("url"), "visit.arguments.url")
        if url_error is not None:
            return f"Tool execution error: {url_error}"
        goal, goal_error = _single_nonempty_str(arguments.get("goal"), "visit.arguments.goal")
        if goal_error is not None:
            return f"Tool execution error: {goal_error}"
    else:
        return f"Error: Unknown tool '{name}'. Available tools: search, visit."

    backend = os.environ.get("SA_TOOL_BACKEND", "web")
    if backend == "web":
        if name == "search":
            return await web.search(
                query,
                api_key="",
                topk=10,
                timeout=30.0,
            )
        if name == "visit":
            return await web.visit(
                url,
                goal=goal,
                api_key="",
                timeout=30.0,
                max_length=150000,
            )
    else:
        return f"Tool execution error: unknown SA_TOOL_BACKEND {backend!r}"

    return f"Error: Unknown tool '{name}'. Available tools: search, visit."
