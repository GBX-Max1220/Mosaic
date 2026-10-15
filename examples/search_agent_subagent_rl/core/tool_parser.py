"""Adapted from https://github.com/RolandXMR/ToolAgent"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

_JSON_SCALARS = (str, int, float, bool)


def _to_jsonable(value: Any) -> Any:
    """Narrow an ``ast.literal_eval`` result to something JSON accepts."""
    if value is None or isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted((_to_jsonable(v) for v in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    raise TypeError(f"{type(value).__name__} is not JSON representable")


class Qwen3ToolParser:
    """JSON tool-call structure support for Qwen3."""

    def __init__(self) -> None:
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def extract_tool_calls(
        self,
        text: str,
        tools: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract valid Qwen3 JSON tool calls from ``text``."""
        del tools
        calls: list[dict[str, Any]] = []
        for raw_call in self.tool_call_regex.findall(text):
            try:
                tool_call = json.loads(raw_call)
                name = tool_call["name"]
                arguments = tool_call["arguments"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            calls.append({"name": name, "arguments": arguments})
        return calls


class Qwen35ToolParser:
    """XML tool call structure support for Qwen3-Coder and Qwen3.5."""

    def __init__(self) -> None:
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.function_regex = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
        self.parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL
        )

    @staticmethod
    def _param_types(tools: list[dict] | None) -> dict[str, dict[str, str]]:
        """Map ``{function_name: {param_name: json_schema_type}}`` from tool schemas."""
        types: dict[str, dict[str, str]] = {}
        for tool in tools or []:
            fn = tool.get("function", tool)
            name = fn.get("name")
            props = (fn.get("parameters") or {}).get("properties") or {}
            if name:
                types[name] = {k: v.get("type") for k, v in props.items()}
        return types

    @staticmethod
    def _strip_newlines(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    @staticmethod
    def _coerce(value: str, schema_type: Any) -> Any:
        """Convert a raw string value to its schema type."""
        if value.lower() == "null":
            return None
        if isinstance(schema_type, (list, tuple)):
            members = [s for s in schema_type if str(s).lower() != "null"]
            for member in members:
                coerced = Qwen35ToolParser._coerce(value, member)
                is_string_member = str(member or "").strip().lower() in (
                    "string",
                    "str",
                    "text",
                    "varchar",
                    "char",
                    "enum",
                )
                if is_string_member or not isinstance(coerced, str):
                    return coerced
            return value
        t = str(schema_type or "").strip().lower()
        if t in ("string", "str", "text", "varchar", "char", "enum"):
            return value
        if t.startswith(("int", "uint", "long", "short", "unsigned")):
            try:
                return int(value)
            except Exception:
                return value
        if t.startswith("num") or t.startswith("float"):
            try:
                f = float(value)
                return int(f) if f.is_integer() else f
            except Exception:
                return value
        if t in ("boolean", "bool", "binary"):
            return value.strip().lower() == "true"
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return _to_jsonable(ast.literal_eval(value))
        except Exception:
            return value

    def _parse_function(
        self, func_str: str, param_types: dict[str, dict[str, str]]
    ) -> tuple[str, dict[str, Any]] | None:
        """Parse one ``name>...<parameter=...>`` chunk into (name, arguments dict)."""
        if ">" not in func_str:
            return None
        name = func_str[: func_str.index(">")].strip()
        params_body = func_str[func_str.index(">") + 1 :]
        schema = param_types.get(name, {})
        arguments: dict[str, Any] = {}
        for match_text in self.parameter_regex.findall(params_body):
            if ">" not in match_text:
                continue
            idx = match_text.index(">")
            key = match_text[:idx].strip()
            value = self._strip_newlines(match_text[idx + 1 :])
            arguments[key] = self._coerce(value, schema.get(key))
        return name, arguments

    def extract_tool_calls(self, text: str, tools: list[dict] | None = None) -> list[dict[str, Any]]:
        """Extract Qwen3.5-style XML tool calls from ``text``."""
        param_types = self._param_types(tools)
        contents = [g1 or g2 for g1, g2 in self.tool_call_regex.findall(text)]
        calls: list[dict[str, Any]] = []
        for content in contents:
            for g1, g2 in self.function_regex.findall(content.strip()):
                parsed = self._parse_function(g1 or g2, param_types)
                if parsed is None:
                    continue
                name, arguments = parsed
                if name:
                    calls.append({"name": name, "arguments": arguments})
        return calls


_QWEN35_TOOL_PARSER = Qwen35ToolParser()
_QWEN3_TOOL_PARSER = Qwen3ToolParser()


def extract_tool_calls(
    text: str,
    tools: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Extract Qwen3.5 XML calls first, then fall back to Qwen3 JSON."""
    calls = _QWEN35_TOOL_PARSER.extract_tool_calls(text, tools)
    if calls:
        return calls
    return _QWEN3_TOOL_PARSER.extract_tool_calls(text, tools)
