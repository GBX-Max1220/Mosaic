from __future__ import annotations

from typing import Any


class ContextManager:
    """Tracks token usage and decides when to compact or stop."""

    def __init__(
        self,
        max_context_tokens: int,
        reserve_tokens: int,
        tokenizer: Any,
        tools: list[dict] | None = None,
    ):
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens
        self.tokenizer = tokenizer
        self.tools = tools
        self.current_prefix_tokens: list[int] = []

    def set_prefix(self, messages: list[dict[str, str]]) -> None:
        self.current_prefix_tokens = self._encode_messages(messages, add_generation_prompt=False)

    def record_turn(self, assistant_text: str, tool_text: str) -> None:
        self.current_prefix_tokens += self.tokenizer.encode(assistant_text, add_special_tokens=False)
        self.current_prefix_tokens += self.tokenizer.encode(tool_text, add_special_tokens=False)

    def _encode_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def token_length(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> int:
        return len(self._encode_messages(messages, add_generation_prompt=add_generation_prompt))

    def over_budget(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> bool:
        projected = self.token_length(messages, add_generation_prompt=add_generation_prompt)
        return projected > self.max_context_tokens - self.reserve_tokens

    def would_overflow(self, assistant_content: str, tool_content: str) -> bool:
        extra = self.tokenizer.encode(assistant_content, add_special_tokens=False)
        extra += self.tokenizer.encode(tool_content, add_special_tokens=False)
        projected = len(self.current_prefix_tokens) + len(extra)
        return projected > self.max_context_tokens - self.reserve_tokens

    def current_length(self) -> int:
        return len(self.current_prefix_tokens)
