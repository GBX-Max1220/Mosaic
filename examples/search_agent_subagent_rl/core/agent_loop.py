from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from slime.rollout.sglang_rollout import GenerateState, get_model_url
from slime.utils.http_utils import post

from examples.search_agent_subagent_rl.runtime.reward import (
    _extract_answer,
    browsecomp_judge_with_metrics,
)
from examples.search_agent_subagent_rl.core.config import SubagentSearchConfig, session_ig_enabled
from examples.search_agent_subagent_rl.core.parser import GenerationFlag, parse_response
from examples.search_agent_subagent_rl.core.prompts import (
    FORCED_ANSWER_PROMPT,
    GENERATION_ERROR_FEEDBACK,
    MAIN_SYSTEM_PROMPT,
    build_fork_limit_feedback,
    build_main_tool_schemas,
    build_malformed_tool_feedback,
)
from examples.search_agent_subagent_rl.core.session import (
    SessionSegment,
    SubagentResult,
    TurnTrace,
    sessions_to_samples,
)
from examples.search_agent_subagent_rl.core.subagent_loop import SubagentLoop
from examples.search_agent_subagent_rl.tools.registry import execute_tool


def _text_for_tokenizer(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def format_subagent_results(results: list[SubagentResult]) -> str:
    blocks = ["<subagent_results>"]
    for result in results:
        status = "success" if result.success else "failed"
        blocks.append(f"Subagent #{result.subagent_index + 1}: {status}")
        blocks.append("Task:")
        blocks.append(result.task)
        if result.success:
            blocks.append("Result:")
            blocks.append(result.answer or "")
        else:
            blocks.append("Failure:")
            blocks.append(result.failure or "Subagent failed.")
        blocks.append("")
    if blocks[-1] == "":
        blocks.pop()
    blocks.append("</subagent_results>")
    return "\n".join(blocks)


@dataclass
class PendingFork:
    tool_result_text: str
    results: list[SubagentResult] = field(default_factory=list)
    is_successful: bool = False
    limit_reached: bool = False


def build_shared_metrics(
    *,
    trajectory_raw_reward: float,
    main_turn_count: int,
    search_call_count: int,
    visit_call_count: int,
    attempted_fork_call_count: int,
    successful_fork_call_count: int,
    fork_limit_reached_count: int,
    subagent_success_count: int,
    subagent_failure_count: int,
    max_full_context_tokens: int,
) -> dict[str, float | int | str]:
    subagent_count = subagent_success_count + subagent_failure_count
    rollout_num_sessions = 1 + subagent_count
    return {
        "metric_schema": "search_agent_subagent_rl",
        "trajectory_raw_reward": trajectory_raw_reward,
        "rollout_num_sessions": rollout_num_sessions,
        "train_num_sessions": rollout_num_sessions,
        "main_count": 1,
        "subagent_count": subagent_count,
        "subagent_success_count": subagent_success_count,
        "subagent_failure_count": subagent_failure_count,
        "subagent_success_rate": subagent_success_count / subagent_count if subagent_count else 0.0,
        "main_turn_count": main_turn_count,
        "search_call_count": search_call_count,
        "visit_call_count": visit_call_count,
        "attempted_fork_call_count": attempted_fork_call_count,
        "fork_call_count": successful_fork_call_count,
        "fork_limit_reached_count": fork_limit_reached_count,
        "max_full_context_tokens": max_full_context_tokens,
    }


class AgentLoop:
    """Single-session main ReAct loop with forked search subagents."""

    def __init__(
        self,
        config: SubagentSearchConfig,
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
        self.agent_id = "main"
        self.tokenizer = state.tokenizer
        self.sampling_params = dict(state.sampling_params)
        if config.context.max_new_tokens > 0:
            self.sampling_params["max_new_tokens"] = config.context.max_new_tokens

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
        self.successful_fork_call_count = 0
        self.attempted_fork_call_count = 0
        self.fork_limit_reached_count = 0
        self.subagent_results: list[SubagentResult] = []
        self.subagent_sessions: list[SessionSegment] = []
        self.subagent_success_count = 0
        self.subagent_failure_count = 0

        self.tool_schemas = build_main_tool_schemas()
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": MAIN_SYSTEM_PROMPT},
            {"role": "user", "content": original_question},
        ]
        self.generation_prompt_delta = self._derive_generation_prompt_delta(self.messages)
        self.current_session = self._new_session(self.messages)

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
            agent_role="main",
            agent_id=self.agent_id,
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
            self.current_session.prompt_token_ids
            + self.current_session.response_token_ids
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
        turn_index = 0
        while True:
            gen = await self._generate_once()
            if gen["error"] is not None:
                self.generation_error_count += 1
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
                turn_index += 1
                continue

            assistant_text = gen["text"]
            flag, _thinking, payload = parse_response(assistant_text, self.tool_schemas)

            if flag == GenerationFlag.ANSWER:
                self._reset_malformed_streak()
                if self._would_exceed_budget(
                    assistant_text,
                    "",
                    continue_generation=False,
                    assistant_tokens=gen["tokens"],
                    assistant_log_probs=gen["log_probs"],
                ):
                    await self._force_answer_at_context_boundary()
                    break
                self.has_final_answer = True
                self._append_turn_with_trace(assistant_text, gen, turn_type="final_answer")
                self._finalize_last_turn_context_after(needs_generation_prompt=False)
                break

            if flag == GenerationFlag.TOOL_CALL:
                self._reset_malformed_streak()
                tool_name = payload["name"]
                pending_fork: PendingFork | None = None
                fork_before_ids: list[int] = []

                if tool_name == "fork":
                    if session_ig_enabled():
                        fork_before_ids = self._current_generation_prompt_ids()
                    pending_fork = await self._stage_fork(
                        payload["arguments"],
                        turn_index,
                    )
                    tool_result_text = pending_fork.tool_result_text
                else:
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
                    await self._force_answer_at_context_boundary()
                    break

                if pending_fork is not None:
                    self._record_attempted_tool_call(tool_name)
                    self._commit_fork(pending_fork)
                self._record_accepted_tool_call(tool_name)
                self._append_turn_with_trace(
                    assistant_text,
                    gen,
                    turn_type="tool_call",
                    tool_name=tool_name,
                )
                if pending_fork is not None and session_ig_enabled():
                    self._capture_fork_session_ig(pending_fork, fork_before_ids)
                self._append_user_message(tool_text)
                self._finalize_last_turn_context_after(needs_generation_prompt=True)
                turn_index += 1
                continue

            self._record_malformed_tool_call()
            feedback = (
                f"<tool_response>\n{build_malformed_tool_feedback(payload)}\n</tool_response>"
            )
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
            turn_index += 1
            if self._hit_malformed_stop_limit():
                self.stopped_by_consecutive_malformed = True
                self._finalize_last_turn_context_after(needs_generation_prompt=False)
                break
            self._append_user_message(feedback)
            self._finalize_last_turn_context_after(needs_generation_prompt=True)

        await self._compute_and_broadcast_reward()
        return self._finalize()

    async def _stage_fork(self, arguments: dict, fork_turn_index: int) -> PendingFork:
        if self.successful_fork_call_count >= self.config.fork.max_fork_calls_per_rollout:
            return PendingFork(
                tool_result_text=build_fork_limit_feedback(),
                limit_reached=True,
            )

        raw_subagents = arguments.get("subagents", []) if isinstance(arguments, dict) else []
        if not isinstance(raw_subagents, list):
            return PendingFork(
                tool_result_text=build_malformed_tool_feedback(
                    "fork arguments.subagents must contain task strings"
                )
            )
        if len(raw_subagents) > self.config.fork.max_subagents:
            return PendingFork(
                tool_result_text=build_malformed_tool_feedback(
                    "fork arguments.subagents exceeds SA_FORK_MAX_SUBAGENTS"
                )
            )
        tasks = [
            str(item["task"]).strip()
            for item in raw_subagents
            if isinstance(item, dict) and isinstance(item.get("task"), str) and item["task"].strip()
        ]
        if not tasks:
            return PendingFork(
                tool_result_text=build_malformed_tool_feedback(
                    "fork arguments.subagents must contain task strings"
                )
            )

        async def run_subagent(task: str, subagent_index: int) -> SubagentResult:
            agent_id = f"{self.agent_id}-sub-{len(self.subagent_results) + subagent_index + 1}"
            loop = SubagentLoop(
                config=self.config,
                state=self.state,
                task=task,
                agent_id=agent_id,
                parent_agent_id=self.agent_id,
                subagent_index=subagent_index,
                fork_turn_index=fork_turn_index,
            )
            return await loop.run()

        gathered = await asyncio.gather(
            *(run_subagent(task, idx) for idx, task in enumerate(tasks)),
            return_exceptions=True,
        )
        results = []
        for idx, item in enumerate(gathered):
            if isinstance(item, SubagentResult):
                results.append(item)
            else:
                results.append(
                    SubagentResult(
                        agent_id=f"{self.agent_id}-sub-{len(self.subagent_results) + idx + 1}",
                        subagent_index=idx,
                        task=tasks[idx],
                        success=False,
                        answer=None,
                        failure=repr(item),
                        session=None,
                    )
                )
        return PendingFork(
            tool_result_text=format_subagent_results(results),
            results=results,
            is_successful=True,
        )

    def _capture_fork_session_ig(self, pending_fork: PendingFork, before_ids: list[int]) -> None:
        if not pending_fork.is_successful:
            return
        prefix = os.environ.get("SA_IG_GT_ANSWER_PREFIX") or "Now there's enough information to answer\n</think>\n<answer>\n"
        suffix = os.environ.get("SA_IG_GT_ANSWER_SUFFIX") or "\n</answer><|im_end|>"
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        label_ids = self.tokenizer.encode(self.label or "", add_special_tokens=False)
        answer_ids = prefix_ids + label_ids + self.tokenizer.encode(suffix, add_special_tokens=False)
        # Keep the sampled assistant history exactly as seen by the actor.
        # Reapplying Qwen's template can alter earlier reasoning whitespace.
        history_ids = self.current_session.prompt_token_ids + self.current_session.response_token_ids
        generation_ids = self.tokenizer.encode(self.generation_prompt_delta, add_special_tokens=False)
        for result in pending_fork.results:
            session = result.session
            if session is None:
                continue
            observation = f"<tool_response>\n{format_subagent_results([result])}\n</tool_response>"
            observation_ids = self.tokenizer.encode(
                self._user_message_delta(observation), add_special_tokens=False,
            )
            session.ig_before_token_ids = list(before_ids)
            session.ig_after_token_ids = history_ids + observation_ids + generation_ids
            session.ig_answer_token_ids = list(answer_ids)
            session.ig_answer_start = len(prefix_ids)
            session.ig_answer_end = len(prefix_ids) + len(label_ids)

    def _commit_fork(self, pending_fork: PendingFork) -> None:
        self.attempted_fork_call_count += 1
        if pending_fork.limit_reached:
            self.fork_limit_reached_count += 1
            return
        if not pending_fork.is_successful:
            return
        for result in pending_fork.results:
            self.subagent_results.append(result)
            if result.session is not None:
                self.subagent_sessions.append(result.session)
            if result.success:
                self.subagent_success_count += 1
            else:
                self.subagent_failure_count += 1
        self.successful_fork_call_count += 1

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
        budget = self.config.context.max_context_tokens - self.config.context.reserve_tokens
        return projected_len > budget

    async def _force_answer_at_context_boundary(self) -> None:
        self.forced_answer_count += 1
        self._append_user_message(f"<tool_response>\n{FORCED_ANSWER_PROMPT}\n</tool_response>")
        forced = await self._generate_once()
        self._append_turn_with_trace(forced["text"], forced, turn_type="forced_answer")
        self._finalize_last_turn_context_after(needs_generation_prompt=False)
        flag, _thinking, _payload = parse_response(forced["text"], self.tool_schemas)
        if flag == GenerationFlag.ANSWER:
            self.has_final_answer = True
            self._reset_malformed_streak()

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
        self.messages.append(
            {
                "role": "assistant",
                "content": prepared["assistant_content"],
            }
        )
        self.current_session.response_text += prepared["session_text"]
        self.current_session.response_token_ids += tokens
        self.current_session.loss_mask += [1] * len(tokens)
        self.current_session.rollout_log_probs += log_probs
        self._append_zero_loss_text(prepared["suffix_text"])
        self.current_session.turn_count += 1

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
                self.current_session.full_text + self.generation_prompt_delta + session_text
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

    async def _compute_and_broadcast_reward(self) -> None:
        if not self.has_final_answer or _extract_answer(self.current_session.response_text) is None:
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
        else:
            temp_sample = SimpleNamespace(
                prompt=self.original_question,
                response=self.current_session.response_text,
                label=self.label,
                metadata={"original_question": self.original_question},
            )
            judge_result = await browsecomp_judge_with_metrics(self.state.args, temp_sample)
            self.total_reward = float(judge_result.score)
            self.judge_metrics = dict(judge_result.metrics)

        for session in [self.current_session] + self.subagent_sessions:
            session.reward = self.total_reward

    def _finalize(self) -> list[Any]:
        sessions = [self.current_session] + self.subagent_sessions
        session_lengths = [
            len(session.prompt_token_ids) + len(session.response_token_ids) for session in sessions
        ]
        total_session_tokens = sum(session_lengths)
        max_session_tokens = max(session_lengths) if session_lengths else 0
        avg_session_tokens = total_session_tokens / max(1, len(sessions))
        policy_tokens_total = sum(sum(session.loss_mask) for session in sessions)
        full_context_budget_tokens = max(
            1,
            self.config.context.max_context_tokens - self.config.context.reserve_tokens,
        )
        final_full_context_tokens = (
            len(self.current_session.prompt_token_ids)
            + len(self.current_session.response_token_ids)
        )
        max_full_context_tokens = max(final_full_context_tokens, max_session_tokens)
        metadata = {
            **self.input_metadata,
            "num_sessions": len(sessions),
            "original_question": self.original_question,
            "trajectory_raw_reward": self.total_reward,
            "judge_metrics": self.judge_metrics,
            "search_call_count": self.search_call_count,
            "visit_call_count": self.visit_call_count,
            "tool_call_count": self.tool_call_count,
            "attempted_tool_call_count": self.attempted_tool_call_count,
            "attempted_search_call_count": self.attempted_search_call_count,
            "attempted_visit_call_count": self.attempted_visit_call_count,
            "attempted_fork_call_count": self.attempted_fork_call_count,
            "successful_fork_call_count": self.successful_fork_call_count,
            "fork_limit_reached_count": self.fork_limit_reached_count,
            "subagent_success_count": self.subagent_success_count,
            "subagent_failure_count": self.subagent_failure_count,
            "consecutive_malformed_tool_call_count": self.consecutive_malformed_tool_call_count,
            "max_consecutive_malformed_tool_call_count": (
                self.max_consecutive_malformed_tool_call_count
            ),
            "max_consecutive_malformed_tool_call_limit": (
                self.config.agent.max_consecutive_malformed_tool_calls
            ),
            "stopped_by_consecutive_malformed": 1 if self.stopped_by_consecutive_malformed else 0,
        }
        shared_metrics = build_shared_metrics(
            trajectory_raw_reward=float(self.total_reward),
            main_turn_count=self.current_session.turn_count,
            search_call_count=self.search_call_count,
            visit_call_count=self.visit_call_count,
            attempted_fork_call_count=self.attempted_fork_call_count,
            successful_fork_call_count=self.successful_fork_call_count,
            fork_limit_reached_count=self.fork_limit_reached_count,
            subagent_success_count=self.subagent_success_count,
            subagent_failure_count=self.subagent_failure_count,
            max_full_context_tokens=max_full_context_tokens,
        )
        shared_metrics.update(
            {
                "num_sessions": len(sessions),
                "total_turns": sum(session.turn_count for session in sessions),
                "policy_tokens_total": policy_tokens_total,
                "total_session_tokens": total_session_tokens,
                "avg_session_length": avg_session_tokens,
                "max_session_length": max_session_tokens,
                "total_session_budget_util": total_session_tokens / full_context_budget_tokens,
                "avg_session_budget_util": avg_session_tokens / full_context_budget_tokens,
                "max_session_budget_util": max_session_tokens / full_context_budget_tokens,
                "final_full_context_tokens": final_full_context_tokens,
                "max_full_context_budget_util": max_full_context_tokens / full_context_budget_tokens,
                "full_context_budget_tokens": full_context_budget_tokens,
                "final_full_context_budget_util": (
                    final_full_context_tokens / full_context_budget_tokens
                ),
                "malformed_tool_call_count": self.malformed_tool_call_count,
                "consecutive_malformed_tool_call_count": self.consecutive_malformed_tool_call_count,
                "max_consecutive_malformed_tool_call_count": (
                    self.max_consecutive_malformed_tool_call_count
                ),
                "max_consecutive_malformed_tool_call_limit": (
                    self.config.agent.max_consecutive_malformed_tool_calls
                ),
                "stopped_by_consecutive_malformed": (
                    1 if self.stopped_by_consecutive_malformed else 0
                ),
                "generation_error_count": self.generation_error_count,
                "tool_call_count": self.tool_call_count,
                "executed_tool_call_count": self.tool_call_count,
                "search_call_count": self.search_call_count,
                "visit_call_count": self.visit_call_count,
                "attempted_tool_call_count": self.attempted_tool_call_count,
                "attempted_search_call_count": self.attempted_search_call_count,
                "attempted_visit_call_count": self.attempted_visit_call_count,
                "attempted_fork_call_count": self.attempted_fork_call_count,
                "successful_fork_call_count": self.successful_fork_call_count,
                "fork_limit_reached_count": self.fork_limit_reached_count,
                "subagent_success_count": self.subagent_success_count,
                "subagent_failure_count": self.subagent_failure_count,
                "forced_answer_count": self.forced_answer_count,
                "trajectory_raw_reward": float(self.total_reward),
            }
        )
        return sessions_to_samples(
            sessions,
            rollout_id=self.rollout_id,
            group_index=self.group_index,
            label=self.label,
            metadata=metadata,
            shared_metrics=shared_metrics,
        )
