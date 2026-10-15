from __future__ import annotations

import asyncio
import json
import os
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
from filelock import FileLock, Timeout


def _search_tracking_path() -> str | None:
    return os.environ.get("SA_SEARCH_TRACKING_FILE") or "/tmp/search_agent_search_calls.jsonl"


def _visit_tracking_path() -> str | None:
    return os.environ.get("SA_VISIT_TRACKING_FILE") or "/tmp/search_agent_visit_calls.jsonl"


class _ToolUsageLogger:
    def __init__(self, path_fn):
        self._path_fn = path_fn

    @property
    def path(self) -> str | None:
        return self._path_fn()

    async def log(self, record: dict[str, Any]) -> None:
        path = self.path
        if path is None:
            return
        line = json.dumps(record, ensure_ascii=False) + "\n"
        lock_path = f"{path}.lock"
        lock = FileLock(lock_path, timeout=5.0)

        def _write() -> None:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with lock:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(line)
            except Timeout:
                warnings.warn(
                    f"Tool tracking logger could not acquire lock for {path} within 5s; dropping log line.",
                    stacklevel=2,
                )
            except OSError as exc:
                warnings.warn(
                    f"Tool tracking logger could not write to {path}: {exc!r}; dropping log line.",
                    stacklevel=2,
                )

        await asyncio.to_thread(_write)


_search_logger = _ToolUsageLogger(_search_tracking_path)
_visit_logger = _ToolUsageLogger(_visit_tracking_path)


async def _track_tool_call(name: str, arguments: dict[str, Any]) -> None:
    record = {"tool_name": name, "arguments": arguments}
    if name == "search":
        await _search_logger.log(record)
    elif name == "visit":
        await _visit_logger.log(record)


def _contains_chinese(text: str) -> bool:
    return any("一" <= char <= "鿿" for char in text)


def _coerce_str(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value) if value is not None else ""


async def search(query: str, api_key: str = "", topk: int = 10, timeout: float = 30.0) -> str:
    query = _coerce_str(query)
    base_url = (os.environ.get("SERPER_BASE_URL", "https://google.serper.dev") or "https://google.serper.dev").rstrip("/")
    key = api_key or os.environ.get("SERPER_API_KEY", "")
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": key,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {
        "q": query,
        "num": topk,
        "gl": "us",
        "hl": "en",
    }
    if _contains_chinese(query):
        payload.update({"location": "China", "gl": "cn", "hl": "zh-cn"})
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/search",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                resp.raise_for_status()
                await _track_tool_call("search", {"query": query, "topk": topk})
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return f"Tool execution error for search: {repr(e)}"
    results = data.get("organic", [])[:topk]
    lines = []
    for i, result in enumerate(results, 1):
        lines.append(f"[{i}] {result.get('title', '')}\n{result.get('link', '')}\n{result.get('snippet', '')}")
    return "\n\n".join(lines) if lines else "No search results found."


async def visit(url: str, goal: str = "", api_key: str = "", timeout: float = 30.0, max_length: int = 150000) -> str:
    url = _coerce_str(url)
    goal = _coerce_str(goal)
    base_url = (os.environ.get("JINA_BASE_URL", "https://r.jina.ai") or "https://r.jina.ai").rstrip("/")
    key = api_key or os.environ.get("JINA_API_KEY", "")
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        parsed = urlparse(url)
    except ValueError as e:
        return f"Tool execution error for visit: invalid URL '{url}': {repr(e)}"
    if not parsed.scheme:
        url = f"http://{url}"
    jina_url = f"{base_url}/{url}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                resp.raise_for_status()
                await _track_tool_call("visit", {"url": url, "goal": goal})
                text = await resp.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return f"Tool execution error for visit: {repr(e)}"
    if len(text) > max_length:
        suffix = "\n[Content truncated]"
        text = text[: max_length - len(suffix)] + suffix
    return text
