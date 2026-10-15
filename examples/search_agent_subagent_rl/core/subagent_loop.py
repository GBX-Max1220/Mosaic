from __future__ import annotations

from typing import Any

from slime.rollout.sglang_rollout import GenerateState, get_model_url
from slime.utils.http_utils import post

from examples.search_agent_subagent_rl.core.config import SubagentSearchConfig
from examples.search_agent_subagent_rl.core.parser import GenerationFlag, parse_response
from examples.search_agent_subagent_rl.core.prompts import (
    FORCED_ANSWER_PROMPT,
    GENERATION_ERROR_FEEDBACK,
    SYSTEM_PROMPT,
    build_malformed_tool_feedback,
    build_subagent_tool_schemas,
    build_subagent_user_message,
)
from examples.search_agent_subagent_rl.core.session import SessionSegment, SubagentResult, TurnTrace
from examples.search_agent_subagent_rl.tools.registry import execute_tool


def _text_for_tokenizer(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


class SubagentLoop:
    """Single-session ReAct loop for forked search subagents."""

    def __init__(
        self,
        config: SubagentSearchConfig,
        state: GenerateState,
        task: str,
        agent_id: str,
        parent_agent_id: str,
        subagent_index: int,
        fork_turn_index: int,
    ):
        self.config = config
        self.state = state
        self.task = task
        self.agent_id = agent_id
        self.parent_agent_id = parent_agent_id
        self.subagent_index = subagent_index
        self.fork_turn_index = fork_turn_index
        self.tokenizer = state.tokenizer
        self.sampling_params = dict(state.sampling_params)
        if config.context.max_new_tokens > 0:
            self.sampling_params["max_new_tokens"] = config.context.max_new_tokens
        self.tool_schemas = build_subagent_tool_schemas()
        self.consecutive_malformed_tool_call_count = 0
        self.forced_answer_count = 0
        self.final_answer: str | None = None

        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_subagent_user_message(task)},
        ]
        self.generation_prompt_delta = self._derive_generation_prompt_delta(self.messages)
        self.session = self._new_session(self.messages)

    def _new_session(self, messages: list[dict[str, str]]) -> SessionSegment:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return SessionSegment(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_ids,
            agent_role="subagent",
            agent_id=self.agent_id,
            parent_agent_id=self.parent_agent_id,
            fork_turn_index=self.fork_turn_index,
            subagent_index=self.subagent_index,
        )

    def _derive_generation_prompt_delta(self, messages: list[dict[str, str]]) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            tokenize=False,
            add_generation_prompt=False,
        )
        target_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not target_text.startswith(prompt_text):
            raise ValueError("Generation chat template is not an extension of the prompt template.")
        return target_text[len(prompt_text) :]

    def _generation_hard_context_limit(self) -> int:
        return int(
            getattr(self.state.args, "rollout_max_context_len", 0)
            or self.config.context.max_context_tokens
        )

    def _current_generation_prompt_ids(self) -> list[int]:
        return (
            self.session.prompt_token_ids
            + self.session.response_token_ids
            + self.tokenizer.encode(self.generation_prompt_delta, add_special_tokens=False)
        )

    async def _generate_once(
        self,
        messages: list[dict[str, str]] | None = None,
        max_new_tokens: int | None = None,
        sampling_param_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if messages is None:
                prompt_ids = self._current_generation_prompt_ids()
            else:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            sampling_params = dict(self.sampling_params)
            if max_new_tokens is not None:
                sampling_params["max_new_tokens"] = max_new_tokens
            if sampling_param_overrides:
                sampling_params.update(sampling_param_overrides)
            requested_new_tokens = int(sampling_params.get("max_new_tokens", 0) or 0)
            hard_context_limit = self._generation_hard_context_limit()
            if len(prompt_ids) + requested_new_tokens > hard_context_limit:
                return {
                    "text": "",
                    "tokens": [],
                    "log_probs": [],
                    "error": (
                        "context_overflow_preflight("
                        f"prompt_tokens={len(prompt_ids)}, "
                        f"max_new_tokens={requested_new_tokens}, "
                        f"max_context_tokens={hard_context_limit})"
                    ),
                }

            url = get_model_url(self.state.args, model_name="policy", endpoint="/generate")
            output = await post(
                url,
                {
                    "input_ids": prompt_ids,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                },
            )
            text = output["text"]
            meta = output.get("meta_info", {})
            tokens = [item[1] for item in meta.get("output_token_logprobs", [])]
            log_probs = [item[0] for item in meta.get("output_token_logprobs", [])]
            return {"text": text, "tokens": tokens, "log_probs": log_probs, "error": None}
        except Exception as exc:
            return {"text": "", "tokens": [], "log_probs": [], "error": repr(exc)}

    async def run(self) -> SubagentResult:
        failure: str | None = None
        while True:
            gen = await self._generate_once()
            if gen["error"] is not None:
                feedback = f"<tool_response>\n{GENERATION_ERROR_FEEDBACK}\n</tool_response>"
                if self._would_exceed_budget(
                    gen["text"],
                    feedback,
                    continue_generation=True,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    await self._force_answer_at_context_boundary()
                    break
                self._append_turn_with_trace(gen["text"], gen, turn_type="generation_error")
                self._append_user_message(feedback)
                self._finalize_last_turn_context_after(needs_generation_prompt=True)
                continue

            assistant_text = gen["text"]
            flag, _thinking, payload = parse_response(assistant_text, self.tool_schemas)

            if flag == GenerationFlag.ANSWER:
                if self._would_exceed_budget(
                    assistant_text,
                    "",
                    continue_generation=False,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    await self._force_answer_at_context_boundary()
                    break
                self._append_turn_with_trace(assistant_text, gen, turn_type="final_answer")
                self._finalize_last_turn_context_after(needs_generation_prompt=False)
                self._record_final_answer(payload)
                break

            if flag == GenerationFlag.TOOL_CALL:
                self.consecutive_malformed_tool_call_count = 0
                tool_name = payload["name"]
                tool_result = await execute_tool(
                    tool_name,
                    payload.get("arguments", {}),
                )
                tool_result_text = _text_for_tokenizer(tool_result)
                tool_text = f"<tool_response>\n{tool_result_text}\n</tool_response>"
                if self._would_exceed_budget(
                    assistant_text,
                    tool_text,
                    continue_generation=True,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    await self._force_answer_at_context_boundary()
                    break
                self._append_turn_with_trace(
                    assistant_text,
                    gen,
                    turn_type="tool_call",
                    tool_name=tool_name,
                )
                self._append_user_message(tool_text)
                self._finalize_last_turn_context_after(needs_generation_prompt=True)
                continue

            self.consecutive_malformed_tool_call_count += 1
            feedback = f"<tool_response>\n{build_malformed_tool_feedback(payload)}\n</tool_response>"
            if self._would_exceed_budget(
                assistant_text,
                feedback,
                continue_generation=True,
                assistant_tokens=gen["tokens"],
                assistant_log_probs=gen["log_probs"],
            ):
                await self._force_answer_at_context_boundary()
                break
            self._append_turn_with_trace(assistant_text, gen, turn_type="malformed")
            if (
                self.consecutive_malformed_tool_call_count
                >= self.config.agent.max_consecutive_malformed_tool_calls
            ):
                self._finalize_last_turn_context_after(needs_generation_prompt=False)
                failure = "No valid <answer> before malformed tool-call limit."
                break
            self._append_user_message(feedback)
            self._finalize_last_turn_context_after(needs_generation_prompt=True)

        if self.final_answer is not None:
            return SubagentResult(
                agent_id=self.agent_id,
                subagent_index=self.subagent_index,
                task=self.task,
                success=True,
                answer=self.final_answer,
                failure=None,
                session=self.session,
            )
        return SubagentResult(
            agent_id=self.agent_id,
            subagent_index=self.subagent_index,
            task=self.task,
            success=False,
            answer=None,
            failure=failure or "No valid <answer> found in subagent response.",
            session=self.session,
        )

    def _would_exceed_budget(
        self,
        assistant_text: str,
        tool_text: str,
        *,
        continue_generation: bool,
        assistant_tokens: list[int] | None = None,
        assistant_log_probs: list[float] | None = None,
    ) -> bool:
        prepared = self._prepare_assistant_turn_append(
            assistant_text,
            {
                "tokens": (
                    assistant_tokens
                    if assistant_tokens is not None
                    else self.tokenizer.encode(assistant_text, add_special_tokens=False)
                ),
                "log_probs": assistant_log_probs or [],
            },
        )
        projected_len = (
            len(self.session.prompt_token_ids)
            + len(self.session.response_token_ids)
            + len(self._encode_zero_loss_text(prepared["template_delta"]))
            + len(prepared["tokens"])
            + len(self._encode_zero_loss_text(prepared["suffix_text"]))
        )
        if tool_text:
            projected_len += len(self._encode_zero_loss_text(self._user_message_delta(tool_text)))
        if continue_generation:
            projected_len += len(self._encode_zero_loss_text(self.generation_prompt_delta))
        budget = self.config.context.max_context_tokens - self.config.context.reserve_tokens
        return projected_len > budget

    async def _force_answer_at_context_boundary(self) -> None:
        self.forced_answer_count += 1
        self._append_user_message(f"<tool_response>\n{FORCED_ANSWER_PROMPT}\n</tool_response>")
        forced = await self._generate_once()
        self._append_turn_with_trace(forced["text"], forced, turn_type="forced_answer")
        self._finalize_last_turn_context_after(needs_generation_prompt=False)
        flag, _thinking, payload = parse_response(forced["text"], self.tool_schemas)
        if flag == GenerationFlag.ANSWER:
            self._record_final_answer(payload)

    def _record_final_answer(self, payload: Any) -> None:
        if isinstance(payload, str) and payload.strip():
            self.final_answer = payload

    def _append_turn_with_trace(
        self,
        text: str,
        gen: dict[str, Any],
        *,
        turn_type: str,
        tool_name: str | None = None,
    ) -> TurnTrace:
        response_start = len(self.session.response_token_ids)
        template_len = len(
            self.tokenizer.encode(self.generation_prompt_delta, add_special_tokens=False)
        )
        prepared = self._prepare_assistant_turn_append(text, gen)
        token_len = len(prepared["tokens"])
        self._append_turn(text, gen)
        trace = TurnTrace(
            turn_index=self.session.turn_count - 1,
            turn_type=turn_type,
            tool_name=tool_name,
            response_start=response_start,
            response_end=len(self.session.response_token_ids),
            policy_token_start=response_start + template_len,
            policy_token_end=response_start + template_len + token_len,
            context_before_response_end=response_start + template_len,
            context_after_response_end=None,
            context_after_needs_generation_prompt=False,
        )
        self.session.turn_traces.append(trace)
        return trace

    def _finalize_last_turn_context_after(self, *, needs_generation_prompt: bool) -> None:
        if not self.session.turn_traces:
            return
        trace = self.session.turn_traces[-1]
        trace.context_after_response_end = len(self.session.response_token_ids)
        trace.context_after_needs_generation_prompt = needs_generation_prompt

    def _append_turn(self, text: str, gen: dict[str, Any]) -> None:
        prepared = self._prepare_assistant_turn_append(text, gen)
        self._append_zero_loss_text(prepared["template_delta"])
        tokens = prepared["tokens"]
        log_probs = prepared["log_probs"]
        self.messages.append(
            {
                "role": "assistant",
                "content": prepared["assistant_content"],
            }
        )
        self.session.response_text += prepared["session_text"]
        self.session.response_token_ids += tokens
        self.session.loss_mask += [1] * len(tokens)
        self.session.rollout_log_probs += log_probs
        self._append_zero_loss_text(prepared["suffix_text"])
        self.session.turn_count += 1

    def _prepare_assistant_turn_append(self, text: str, gen: dict[str, Any]) -> dict[str, Any]:
        template_delta = self.generation_prompt_delta
        tokens = list(gen["tokens"])
        log_probs = list(gen["log_probs"])
        stripped_text = self._strip_trailing_chat_end_text(text)
        terminal_text = text[len(stripped_text) :]
        session_text = stripped_text + terminal_text
        assistant_content = self._assistant_content_prefix(template_delta) + stripped_text
        unclosed_thinking = "<think>" in template_delta and "</think>" not in assistant_content

        if unclosed_thinking:
            assistant_content += "\n</think>"
            terminal_tokens = self.tokenizer.encode(terminal_text, add_special_tokens=False)
            if terminal_tokens and tokens[-len(terminal_tokens) :] == terminal_tokens:
                tokens = tokens[: -len(terminal_tokens)]
                log_probs = log_probs[: -len(terminal_tokens)]
            session_text = stripped_text
            suffix_text = self._unclosed_thinking_suffix()
        else:
            suffix_text = self._closed_assistant_suffix_for_text(session_text)
        return {
            "template_delta": template_delta,
            "tokens": tokens,
            "log_probs": log_probs,
            "session_text": session_text,
            "assistant_content": assistant_content,
            "suffix_text": suffix_text,
        }

    def _strip_trailing_chat_end_text(self, text: str) -> str:
        stop_texts = []
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token:
            stop_texts.append(eos_token)
        stop_texts.append("<|im_end|>")

        for stop_text in dict.fromkeys(stop_texts):
            if stop_text and text.endswith(stop_text):
                return text[: -len(stop_text)]
        return text

    @staticmethod
    def _assistant_content_prefix(template_delta: str) -> str:
        marker = "<|im_start|>assistant\n"
        if marker in template_delta:
            return template_delta.split(marker, 1)[1]
        return ""

    def _closed_assistant_suffix_for_text(self, session_text: str) -> str:
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            return ""
        if (
            self.generation_prompt_delta.startswith("<assistant>")
            and not (
                self.session.full_text + self.generation_prompt_delta + session_text
            ).endswith("</assistant>")
        ):
            return "</assistant>"
        return ""

    def _append_user_message(self, text: str) -> None:
        text = _text_for_tokenizer(text)
        self.messages.append({"role": "user", "content": text})
        self._append_zero_loss_text(self._user_message_delta(text))

    def _append_zero_loss_text(self, text: str) -> None:
        text = _text_for_tokenizer(text)
        if not text:
            return
        tokens = self._encode_zero_loss_text(text)
        self.session.response_text += text
        self.session.response_token_ids += tokens
        self.session.loss_mask += [0] * len(tokens)
        self.session.rollout_log_probs += [0.0] * len(tokens)

    def _encode_zero_loss_text(self, text: Any) -> list[int]:
        return self.tokenizer.encode(_text_for_tokenizer(text), add_special_tokens=False)

    def _user_message_delta(self, text: str) -> str:
        text = _text_for_tokenizer(text)
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            prefix = "" if self.session.full_text.endswith("\n") else "\n"
            return f"{prefix}<|im_start|>user\n{text}<|im_end|>\n"
        return f"<user>{text}</user>"

    def _unclosed_thinking_suffix(self) -> str:
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            return "\n</think>\n\n<|im_end|>\n"
        if self.generation_prompt_delta.endswith("<think>\n"):
            return "\n</think>\n\n</assistant>"
        return ""
