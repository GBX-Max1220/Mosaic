from __future__ import annotations

import copy
import os
import uuid
from types import SimpleNamespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from examples.search_agent_compact_rl.core.config import SearchAgentConfig, session_ig_enabled
from examples.search_agent_compact_rl.core.context_manager import ContextManager
from examples.search_agent_compact_rl.core.parser import GenerationFlag, parse_response
from examples.search_agent_compact_rl.core.prompts import (
    FORCED_ANSWER_PROMPT,
    GENERATION_ERROR_FEEDBACK,
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_USER_TEMPLATE,
    SYSTEM_PROMPT,
    build_malformed_tool_feedback,
    build_summary_user_message,
    build_tool_schemas,
)
from examples.search_agent_compact_rl.runtime.reward import _extract_answer, browsecomp_judge_with_metrics
from examples.search_agent_compact_rl.core.session import SessionSegment, TurnTrace
from examples.search_agent_compact_rl.core.summarizer import Summarizer, format_history_for_summary
from examples.search_agent_compact_rl.tools.registry import execute_tool


DEFAULT_IG_GT_ANSWER_PREFIX = "Now there's enough information to answer\n</think>\n<answer>\n"
DEFAULT_IG_GT_ANSWER_SUFFIX = "\n</answer><|im_end|>"


def _format_debug_messages_path(path: str, rollout_id: int) -> str:
    return path.format(rollout_id=rollout_id) if "{rollout_id}" in path else path


def _env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _text_for_tokenizer(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


class AgentLoop:
    """DR-Venus aligned multi-turn ReAct loop with optional compaction."""

    def __init__(
        self,
        config: SearchAgentConfig,
        state: GenerateState,
        original_question: str,
        label: str | None,
        group_index: int,
        input_metadata: dict[str, Any] | None = None,
    ):
        self.config = config
        self.state = state
        self.original_question = original_question
        self.label = label
        self.group_index = group_index
        self.input_metadata = dict(input_metadata or {})
        self.rollout_id = uuid.uuid4().int
        self.tokenizer = state.tokenizer
        self.sampling_params = state.sampling_params
        self.total_reward: float = 0.0
        self.judge_metrics: dict[str, float] = {}
        self.malformed_tool_call_count = 0
        self.consecutive_malformed_tool_call_count = 0
        self.max_consecutive_malformed_tool_call_count = 0
        self.stopped_by_consecutive_malformed = False
        self.generation_error_count = 0
        self.tool_call_count = 0
        self.search_call_count = 0
        self.visit_call_count = 0
        self.attempted_tool_call_count = 0
        self.attempted_search_call_count = 0
        self.attempted_visit_call_count = 0
        self.forced_answer_count = 0
        self.has_final_answer = False
        self.compaction_count = 0
        self.summary_attempt_count = 0
        self.summary_success_count = 0
        self.summary_failure_count = 0
        self.fallback_force_answer_after_compact_fail = 0
        self.hit_max_compactions = False
        self.rollout_session_count = 1

        self.tool_schemas = build_tool_schemas()
        self.context_manager = ContextManager(
            max_context_tokens=config.compact.max_context_tokens,
            reserve_tokens=config.compact.reserve_tokens,
            tokenizer=self.tokenizer,
            tools=self.tool_schemas,
        )
        self.summarizer = Summarizer(
            config.summarizer,
            self.tokenizer,
            on_policy_base_url=f"http://{state.args.sglang_router_ip}:{state.args.sglang_router_port}/v1",
        )

        initial_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": original_question},
        ]
        self.messages: list[dict[str, str]] = copy.deepcopy(initial_messages)
        self.context_manager.set_prefix(initial_messages)
        self.generation_prompt_delta = self._derive_generation_prompt_delta(initial_messages)
        self.max_full_context_tokens = self.context_manager.token_length(
            self.messages,
            add_generation_prompt=False,
        )

        self.sessions: list[SessionSegment] = []
        self.current_session = self._new_session(initial_messages)
        self.current_session.ig_generation_prompt_delta_token_ids = self.tokenizer.encode(
            self.generation_prompt_delta,
            add_special_tokens=False,
        )
        self._set_session_ig_answer_tokens(self.current_session)
        self.sessions.append(self.current_session)

    def _new_session(self, messages: list[dict[str, str]]) -> SessionSegment:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        session = SessionSegment(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_ids,
            session_type="normal",
            ig_before_token_ids=list(prompt_ids) if session_ig_enabled() else [],
        )
        self._set_session_ig_answer_tokens(session)
        return session

    def _set_session_ig_answer_tokens(self, session: SessionSegment) -> None:
        prefix = _env_or_default("SA_IG_GT_ANSWER_PREFIX", DEFAULT_IG_GT_ANSWER_PREFIX)
        suffix = _env_or_default("SA_IG_GT_ANSWER_SUFFIX", DEFAULT_IG_GT_ANSWER_SUFFIX)
        label = self.label or ""
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        label_ids = self.tokenizer.encode(label, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        session.ig_answer_token_ids = prefix_ids + label_ids + suffix_ids
        session.ig_answer_start = len(prefix_ids)
        session.ig_answer_end = len(prefix_ids) + len(label_ids)

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
        return target_text[len(prompt_text):]

    def _generation_hard_context_limit(self) -> int:
        return int(
            getattr(self.state.args, "rollout_max_context_len", 0)
            or self.config.compact.max_context_tokens
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
            payload = {
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            output = await post(url, payload)

            text = output["text"]
            meta = output.get("meta_info", {})
            tokens = [item[1] for item in meta.get("output_token_logprobs", [])]
            log_probs = [item[0] for item in meta.get("output_token_logprobs", [])]
            return {"text": text, "tokens": tokens, "log_probs": log_probs, "error": None}
        except Exception as e:
            return {"text": "", "tokens": [], "log_probs": [], "error": repr(e)}

    def _current_generation_prompt_ids(self) -> list[int]:
        return (
            self.current_session.prompt_token_ids
            + self.current_session.response_token_ids
            + self.tokenizer.encode(self.generation_prompt_delta, add_special_tokens=False)
        )

    def _record_attempted_tool_call(self, tool_name: str) -> None:
        self.attempted_tool_call_count += 1
        if tool_name == "search":
            self.attempted_search_call_count += 1
        elif tool_name == "visit":
            self.attempted_visit_call_count += 1

    def _record_accepted_tool_call(self, tool_name: str) -> None:
        self.tool_call_count += 1
        if tool_name == "search":
            self.search_call_count += 1
        elif tool_name == "visit":
            self.visit_call_count += 1

    def _record_malformed_tool_call(self) -> None:
        self.malformed_tool_call_count += 1
        self.consecutive_malformed_tool_call_count += 1
        self.max_consecutive_malformed_tool_call_count = max(
            self.max_consecutive_malformed_tool_call_count,
            self.consecutive_malformed_tool_call_count,
        )

    def _reset_malformed_streak(self) -> None:
        self.consecutive_malformed_tool_call_count = 0

    def _hit_malformed_stop_limit(self) -> bool:
        return (
            self.consecutive_malformed_tool_call_count
            >= self.config.agent.max_consecutive_malformed_tool_calls
        )

    async def run(self) -> list[Any]:
        step = 0
        while True:
            gen = await self._generate_once()
            if gen["error"] is not None:
                self.generation_error_count += 1
                error_msg = f"<tool_response>\n{GENERATION_ERROR_FEEDBACK}\n</tool_response>"
                if self._would_exceed_budget(
                    gen["text"],
                    error_msg,
                    continue_generation=True,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    if await self._handle_budget_boundary():
                        break
                    continue
                self._append_turn_with_trace(gen["text"], gen, turn_type="generation_error")
                self._append_user_message(error_msg)
                self._finalize_last_turn_context_after(needs_generation_prompt=True)
                step += 1
                continue

            assistant_text = gen["text"]
            flag, thinking, payload = parse_response(assistant_text, self.tool_schemas)

            print(
                f"[AGENT_TURN] rollout_id={self.rollout_id} turn={step} "
                f"flag={flag.name} len={len(assistant_text)} "
                f"tool={payload.get('name') if isinstance(payload, dict) else None} "
                f"text_start={assistant_text[:120]!r} "
                f"text_end={assistant_text[-120:]!r}",
                flush=True,
            )

            if flag == GenerationFlag.ERROR:
                self._record_malformed_tool_call()
                error_msg = (
                    f"<tool_response>\n{build_malformed_tool_feedback(payload)}\n</tool_response>"
                )
                if self._would_exceed_budget(
                    assistant_text,
                    error_msg,
                    continue_generation=True,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    if await self._handle_budget_boundary():
                        break
                    continue
                self._append_turn_with_trace(assistant_text, gen, turn_type="malformed")
                step += 1
                if self._hit_malformed_stop_limit():
                    self.stopped_by_consecutive_malformed = True
                    self._finalize_last_turn_context_after(needs_generation_prompt=False)
                    break
                self._append_user_message(error_msg)
                self._finalize_last_turn_context_after(needs_generation_prompt=True)
                continue

            if flag == GenerationFlag.END:
                self._reset_malformed_streak()
                if self._would_exceed_budget(
                    assistant_text,
                    "",
                    continue_generation=False,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    if await self._handle_budget_boundary():
                        break
                    continue
                self.has_final_answer = True
                self._append_turn_with_trace(assistant_text, gen, turn_type="final_answer")
                self._finalize_last_turn_context_after(needs_generation_prompt=False)
                break

            if flag == GenerationFlag.CALL:
                self._reset_malformed_streak()
                tool_name = payload["name"]
                self._record_attempted_tool_call(tool_name)

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
                    if await self._handle_budget_boundary():
                        break
                    continue

                self._record_accepted_tool_call(tool_name)
                self._append_turn_with_trace(
                    assistant_text,
                    gen,
                    turn_type="tool_call",
                    tool_name=tool_name,
                )
                self._append_user_message(tool_text)
                self._finalize_last_turn_context_after(needs_generation_prompt=True)
                step += 1

        await self._compute_and_broadcast_reward()

        debug_msgs_path = os.environ.get("SA_DEBUG_MESSAGES_PATH")
        if debug_msgs_path:
            import json

            debug_msgs_path = _format_debug_messages_path(debug_msgs_path, self.rollout_id)
            os.makedirs(os.path.dirname(debug_msgs_path), exist_ok=True)
            with open(debug_msgs_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "original_question": self.original_question,
                        "label": self.label,
                        "num_sessions": len(self.sessions),
                        "messages": self.messages,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        return self._finalize()

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
            len(self.current_session.prompt_token_ids)
            + len(self.current_session.response_token_ids)
            + len(self._encode_zero_loss_text(prepared["template_delta"]))
            + len(prepared["tokens"])
            + len(self._encode_zero_loss_text(prepared["suffix_text"]))
        )
        if tool_text:
            projected_len += len(self._encode_zero_loss_text(self._user_message_delta(tool_text)))
        if continue_generation:
            projected_len += len(self._encode_zero_loss_text(self.generation_prompt_delta))
        return projected_len > self.config.compact.max_context_tokens - self.config.compact.reserve_tokens

    async def _handle_budget_boundary(self) -> bool:
        """Resolve a context-budget hit.

        Returns True when the caller should stop the loop, False when it should
        continue with the updated session state.
        """
        if self.config.compact.enabled and self.compaction_count < self.config.compact.max_compactions:
            ok = await self._compact()
            if ok:
                return False
            self.fallback_force_answer_after_compact_fail += 1
        elif self.config.compact.enabled:
            self.hit_max_compactions = True

        forced = await self._generate_forced_answer()
        self.forced_answer_count += 1
        forced_flag, _, _ = parse_response(forced["text"], self.tool_schemas)
        self.has_final_answer = forced_flag == GenerationFlag.END
        if self.has_final_answer:
            self._reset_malformed_streak()
        self._append_turn_with_trace(forced["text"], forced, turn_type="forced_answer")
        self._finalize_last_turn_context_after(needs_generation_prompt=False)
        return True

    def _append_turn_with_trace(
        self,
        text: str,
        gen: dict[str, Any],
        *,
        turn_type: str,
        tool_name: str | None = None,
    ) -> TurnTrace:
        response_start = len(self.current_session.response_token_ids)
        template_len = len(
            self.tokenizer.encode(self.generation_prompt_delta, add_special_tokens=False)
        )
        prepared = self._prepare_assistant_turn_append(text, gen)
        token_len = len(prepared["tokens"])
        self._append_turn(text, gen)
        trace = TurnTrace(
            turn_index=self.current_session.turn_count - 1,
            turn_type=turn_type,
            tool_name=tool_name,
            response_start=response_start,
            response_end=len(self.current_session.response_token_ids),
            policy_token_start=response_start + template_len,
            policy_token_end=response_start + template_len + token_len,
            context_before_response_end=response_start + template_len,
            context_after_response_end=None,
            context_after_needs_generation_prompt=False,
        )
        self.current_session.turn_traces.append(trace)
        return trace

    def _finalize_last_turn_context_after(self, *, needs_generation_prompt: bool) -> None:
        if not self.current_session.turn_traces:
            return
        trace = self.current_session.turn_traces[-1]
        trace.context_after_response_end = len(self.current_session.response_token_ids)
        trace.context_after_needs_generation_prompt = needs_generation_prompt

    def _append_turn(self, text: str, gen: dict[str, Any]) -> None:
        prepared = self._prepare_assistant_turn_append(text, gen)
        self._append_zero_loss_text(prepared["template_delta"])
        tokens = prepared["tokens"]
        log_probs = prepared["log_probs"]
        loss_mask = [1] * len(tokens)
        self.messages.append(
            {
                "role": "assistant",
                "content": prepared["assistant_content"],
            }
        )
        self.current_session.response_text += prepared["session_text"]
        self.current_session.response_token_ids += tokens
        self.current_session.loss_mask += loss_mask
        self.current_session.rollout_log_probs += log_probs
        self._append_zero_loss_text(prepared["suffix_text"])
        self.current_session.turn_count += 1
        self.context_manager.record_turn(text, "")
        self._record_full_context_tokens()

    def _prepare_assistant_turn_append(self, text: str, gen: dict[str, Any]) -> dict[str, Any]:
        template_delta = self.generation_prompt_delta
        tokens = list(gen["tokens"])
        log_probs = list(gen["log_probs"])
        stripped_text = self._strip_trailing_chat_end_text(text)
        terminal_text = text[len(self._strip_trailing_chat_end_text(text)) :]
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

    def _closed_assistant_suffix_for_text(self, session_text: str) -> str:
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            return ""
        if (
            self.generation_prompt_delta.startswith("<assistant>")
            and not (self.current_session.full_text + self.generation_prompt_delta + session_text).endswith("</assistant>")
        ):
            return "</assistant>"
        return ""

    def _strip_trailing_chat_end_text(
        self,
        text: str,
    ) -> str:
        stop_texts = []
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token:
            stop_texts.append(eos_token)
        stop_texts.append("<|im_end|>")

        for stop_text in dict.fromkeys(stop_texts):
            if not stop_text or not text.endswith(stop_text):
                continue
            return text[: -len(stop_text)]
        return text

    def _append_user_message(self, text: str) -> None:
        text = _text_for_tokenizer(text)
        self.messages.append({"role": "user", "content": text})
        self._append_zero_loss_text(self._user_message_delta(text))
        self.context_manager.record_turn("", text)
        self._record_full_context_tokens()

    def _append_generation_prompt_delta(self) -> str:
        self._append_zero_loss_text(self.generation_prompt_delta)
        return self.generation_prompt_delta

    def _append_zero_loss_text(self, text: str) -> None:
        text = _text_for_tokenizer(text)
        if not text:
            return
        tokens = self._encode_zero_loss_text(text)
        self.current_session.response_text += text
        self.current_session.response_token_ids += tokens
        self.current_session.loss_mask += [0] * len(tokens)
        self.current_session.rollout_log_probs += [0.0] * len(tokens)

    def _encode_zero_loss_text(self, text: Any) -> list[int]:
        return self.tokenizer.encode(_text_for_tokenizer(text), add_special_tokens=False)

    def _user_message_delta(self, text: str) -> str:
        text = _text_for_tokenizer(text)
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            prefix = "" if self.current_session.full_text.endswith("\n") else "\n"
            return f"{prefix}<|im_start|>user\n{text}<|im_end|>\n"
        return f"<user>{text}</user>"

    def _unclosed_thinking_suffix(self) -> str:
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            return "\n</think>\n\n<|im_end|>\n"
        if self.generation_prompt_delta.endswith("<think>\n"):
            return "\n</think>\n\n</assistant>"
        return ""

    def _closed_assistant_suffix(self) -> str:
        if "<|im_start|>assistant" in self.generation_prompt_delta:
            return ""
        if self.generation_prompt_delta.startswith("<assistant>") and not self.current_session.full_text.endswith("</assistant>"):
            return "</assistant>"
        return ""

    @staticmethod
    def _assistant_content_prefix(template_delta: str) -> str:
        marker = "<|im_start|>assistant\n"
        if marker in template_delta:
            return template_delta.split(marker, 1)[1]
        return ""

    def _record_full_context_tokens(self) -> int:
        full_context_tokens = (
            len(self.current_session.prompt_token_ids)
            + len(self.current_session.response_token_ids)
        )
        self.max_full_context_tokens = max(self.max_full_context_tokens, full_context_tokens)
        return full_context_tokens

    async def _compact(self) -> bool:
        """Compact current prefix. Returns True on success."""
        self.summary_attempt_count += 1
        pre_compact_history_token_ids = (
            list(self.current_session.prompt_token_ids + self.current_session.response_token_ids)
            if session_ig_enabled()
            else []
        )
        history_text = format_history_for_summary(self.messages)
        summarizer_messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARIZER_USER_TEMPLATE.format(history=history_text)},
        ]
        summary_debug_text = ""

        summary_segment: SessionSegment | None = None
        if self.config.summarizer.mode == "on_policy":
            summary, summary_segment = await self._generate_summary_on_policy(summarizer_messages)
            summary_debug_text = summary_segment.response_text
            if summary and self.config.summarizer.train_summary_session:
                self.sessions.append(summary_segment)
        else:
            try:
                summary = await self.summarizer.compact(self.messages)
            except Exception as exc:
                summary_debug_text = repr(exc)
                summary = ""

        if not summary:
            self.summary_failure_count += 1
            print(
                f"[AGENT_COMPACT] rollout_id={self.rollout_id} success=false "
                f"mode={self.config.summarizer.mode} "
                f"summary_start={summary_debug_text[:120]!r} "
                f"summary_end={summary_debug_text[-120:]!r}",
                flush=True,
            )
            return False

        self.summary_success_count += 1
        new_user_msg = build_summary_user_message(self.original_question, summary)
        new_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": new_user_msg},
        ]
        self.messages = new_messages
        self.context_manager.set_prefix(new_messages)
        self.current_session = self._new_session(new_messages)
        if summary_segment is not None and self.config.summarizer.train_summary_session:
            summary_segment.session_type = "compact"
            if session_ig_enabled():
                summary_segment.ig_before_token_ids = pre_compact_history_token_ids
                summary_segment.ig_after_token_ids = list(self.current_session.prompt_token_ids)
        self.sessions.append(self.current_session)
        self.compaction_count += 1
        self.rollout_session_count += 2 if self.config.summarizer.mode == "on_policy" else 1
        print(
            f"[AGENT_COMPACT] rollout_id={self.rollout_id} success=true "
            f"mode={self.config.summarizer.mode} compactions={self.compaction_count} "
            f"rollout_sessions={self.rollout_session_count} train_sessions={len(self.sessions)}",
            flush=True,
        )
        return True

    async def _generate_summary_on_policy(
        self, summarizer_messages: list[dict[str, str]]
    ) -> tuple[str, SessionSegment]:
        prompt_text = self.tokenizer.apply_chat_template(
            summarizer_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        gen = await self._generate_once(
            messages=summarizer_messages,
            max_new_tokens=self.config.summarizer.off_policy_max_tokens,
            sampling_param_overrides={
                "temperature": self.config.summarizer.temperature,
            },
        )
        summary_text = self.summarizer.extract_summary(gen["text"])
        segment = SessionSegment(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_ids,
            response_text=gen["text"],
            response_token_ids=gen["tokens"],
            loss_mask=[1] * len(gen["tokens"]),
            rollout_log_probs=gen["log_probs"],
            turn_count=1,
        )
        self._set_session_ig_answer_tokens(segment)
        return summary_text, segment

    async def _generate_forced_answer(self) -> dict[str, Any]:
        self._append_user_message(f"<tool_response>\n{FORCED_ANSWER_PROMPT}\n</tool_response>")
        return await self._generate_once()

    async def _compute_and_broadcast_reward(self) -> None:
        """Score the full trajectory once and split the reward across segments.

        The reward must be computed on the *whole* rollout, not per session
        segment: only the final segment contains ``<answer>...</answer>``, so a
        per-segment judge would score every intermediate / summary segment 0.

        We reconstruct the full response by concatenating every session's
        response text and judge it once. The terminal reward is stored sparsely:
        only the last session carries the trajectory-level reward, and earlier
        sessions carry ``0.0``. Setting ``session.reward`` to a non-``None``
        value makes slime skip its per-sample reward model
        (``slime/rollout/sglang_rollout.py``: "Some custom generate paths may
        have already filled the reward"). The custom GRPO post-process recovers
        the trajectory reward and broadcasts the resulting advantage to every
        sibling session.
        """
        full_response = "".join(s.response_text for s in self.sessions)
        if not self.has_final_answer or _extract_answer(full_response) is None:
            self.total_reward = 0.0
            self.judge_metrics = {
                "attempts": 0.0,
                "retries": 0.0,
                "latency_sec": 0.0,
                "success": 0.0,
                "fallback": 0.0,
                "timeout": 0.0,
                "status_524": 0.0,
                "retryable_error": 0.0,
                "unparseable": 0.0,
                "skipped_no_answer": 1.0,
            }
            for session in self.sessions:
                session.reward = 0.0
            return

        temp_sample = SimpleNamespace(
            prompt=self.original_question,
            response=full_response,
            label=self.label,
            metadata={"original_question": self.original_question},
        )
        judge_result = await browsecomp_judge_with_metrics(self.state.args, temp_sample)
        self.total_reward = float(judge_result.score)
        self.judge_metrics = dict(judge_result.metrics)

        for i, session in enumerate(self.sessions):
            session.reward = self.total_reward if i == len(self.sessions) - 1 else 0.0

    def _finalize(self) -> list[Any]:
        from examples.search_agent_compact_rl.core.session import sessions_to_samples

        capture_session_ig = session_ig_enabled()
        for idx, session in enumerate(self.sessions):
            session.session_index = idx
            session.is_terminal_session = idx == len(self.sessions) - 1
            if capture_session_ig and not session.is_terminal_session and not session.ig_after_token_ids:
                session.ig_after_token_ids = list(
                    session.prompt_token_ids + session.response_token_ids
                )
            if capture_session_ig and session.is_terminal_session:
                session.ig_after_token_ids = []

        metadata = {
            **self.input_metadata,
            "num_sessions": len(self.sessions),
            "original_question": self.original_question,
            "trajectory_raw_reward": self.total_reward,
            "judge_metrics": self.judge_metrics,
            "search_call_count": self.search_call_count,
            "visit_call_count": self.visit_call_count,
            "tool_call_count": self.tool_call_count,
            "attempted_tool_call_count": self.attempted_tool_call_count,
            "attempted_search_call_count": self.attempted_search_call_count,
            "attempted_visit_call_count": self.attempted_visit_call_count,
            "consecutive_malformed_tool_call_count": self.consecutive_malformed_tool_call_count,
            "max_consecutive_malformed_tool_call_count": self.max_consecutive_malformed_tool_call_count,
            "max_consecutive_malformed_tool_call_limit": self.config.agent.max_consecutive_malformed_tool_calls,
            "stopped_by_consecutive_malformed": 1 if self.stopped_by_consecutive_malformed else 0,
        }

        max_context_tokens = self.config.compact.max_context_tokens
        full_context_budget_tokens = max(
            1,
            max_context_tokens - self.config.compact.reserve_tokens,
        )
        final_full_context_tokens = self._record_full_context_tokens()
        session_lengths = [
            len(s.prompt_token_ids) + len(s.response_token_ids) for s in self.sessions
        ]
        total_session_tokens = sum(session_lengths)
        max_session_tokens = max(session_lengths) if session_lengths else 0
        avg_session_tokens = total_session_tokens / max(1, len(session_lengths))
        policy_tokens_total = sum(sum(s.loss_mask) for s in self.sessions)
        summary_attempt_count = self.summary_success_count + self.summary_failure_count
        summary_success_rate = self.summary_success_count / summary_attempt_count if summary_attempt_count else 0.0
        dropped_tool_call_count = self.attempted_tool_call_count - self.tool_call_count
        dropped_search_call_count = self.attempted_search_call_count - self.search_call_count
        dropped_visit_call_count = self.attempted_visit_call_count - self.visit_call_count
        train_num_sessions = len(self.sessions)
        compact_metrics = {
            "num_sessions": train_num_sessions,
            "train_num_sessions": train_num_sessions,
            "rollout_num_sessions": self.rollout_session_count,
            "num_compactions": self.compaction_count,
            "max_compactions": self.config.compact.max_compactions,
            "hit_max_compactions": 1 if self.hit_max_compactions else 0,
            "total_turns": sum(s.turn_count for s in self.sessions),
            "did_compact": 1 if self.compaction_count > 0 else 0,
            "summary_attempt_count": summary_attempt_count,
            "summary_success_count": self.summary_success_count,
            "summary_failure_count": self.summary_failure_count,
            "summary_success_rate": summary_success_rate,
            "fallback_force_answer_after_compact_fail": self.fallback_force_answer_after_compact_fail,
            "was_aborted": 0,
            "max_context_util": self.max_full_context_tokens / full_context_budget_tokens,
            "policy_tokens_total": policy_tokens_total,
            "total_session_tokens": total_session_tokens,
            "avg_session_length": avg_session_tokens,
            "max_session_length": max_session_tokens,
            "total_session_budget_util": total_session_tokens / full_context_budget_tokens,
            "avg_session_budget_util": avg_session_tokens / full_context_budget_tokens,
            "max_session_budget_util": max_session_tokens / full_context_budget_tokens,
            "final_full_context_tokens": final_full_context_tokens,
            "max_full_context_tokens": self.max_full_context_tokens,
            "full_context_budget_tokens": full_context_budget_tokens,
            "final_full_context_budget_util": final_full_context_tokens / full_context_budget_tokens,
            "max_full_context_budget_util": self.max_full_context_tokens / full_context_budget_tokens,
            "malformed_tool_call_count": self.malformed_tool_call_count,
            "consecutive_malformed_tool_call_count": self.consecutive_malformed_tool_call_count,
            "max_consecutive_malformed_tool_call_count": self.max_consecutive_malformed_tool_call_count,
            "max_consecutive_malformed_tool_call_limit": self.config.agent.max_consecutive_malformed_tool_calls,
            "stopped_by_consecutive_malformed": 1 if self.stopped_by_consecutive_malformed else 0,
            "generation_error_count": self.generation_error_count,
            "tool_call_count": self.tool_call_count,
            "executed_tool_call_count": self.tool_call_count,
            "search_call_count": self.search_call_count,
            "visit_call_count": self.visit_call_count,
            "attempted_tool_call_count": self.attempted_tool_call_count,
            "attempted_search_call_count": self.attempted_search_call_count,
            "attempted_visit_call_count": self.attempted_visit_call_count,
            "dropped_tool_call_count": dropped_tool_call_count,
            "dropped_search_call_count": dropped_search_call_count,
            "dropped_visit_call_count": dropped_visit_call_count,
            "forced_answer_count": self.forced_answer_count,
            "trajectory_raw_reward": float(self.total_reward),
        }
        return sessions_to_samples(
            self.sessions,
            rollout_id=self.rollout_id,
            group_index=self.group_index,
            label=self.label,
            metadata=metadata,
            compact_metrics=compact_metrics,
        )
