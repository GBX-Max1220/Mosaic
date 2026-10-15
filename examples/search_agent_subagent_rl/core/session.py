from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from examples.search_agent_subagent_rl.core.config import session_ig_enabled

@dataclass
class TurnTrace:
    turn_index: int
    turn_type: str
    tool_name: str | None
    response_start: int
    response_end: int
    policy_token_start: int
    policy_token_end: int
    context_before_response_end: int
    context_after_response_end: int | None = None
    context_after_needs_generation_prompt: bool = False


@dataclass
class SessionSegment:
    prompt_text: str = ""
    prompt_token_ids: list[int] = field(default_factory=list)
    response_text: str = ""
    response_token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    reward: float | None = None
    session_index: int = 0
    turn_count: int = 0
    turn_traces: list[TurnTrace] = field(default_factory=list)
    agent_role: Literal["main", "subagent"] = "main"
    agent_id: str = ""
    parent_agent_id: str | None = None
    fork_turn_index: int | None = None
    subagent_index: int | None = None
    ig_before_token_ids: list[int] = field(default_factory=list)
    ig_after_token_ids: list[int] = field(default_factory=list)
    ig_answer_token_ids: list[int] = field(default_factory=list)
    ig_answer_start: int = 0
    ig_answer_end: int = 0

    @property
    def full_text(self) -> str:
        return self.prompt_text + self.response_text


@dataclass
class SubagentResult:
    agent_id: str
    subagent_index: int
    task: str
    success: bool
    answer: str | None
    failure: str | None
    session: SessionSegment | None


def sessions_to_samples(
    sessions: list[SessionSegment],
    rollout_id: int,
    group_index: int,
    label: str | None,
    metadata: dict[str, Any],
    shared_metrics: dict[str, Any],
) -> list[Any]:
    # list[Any] avoids importing Sample at module load time in rollout workers.
    from slime.utils.types import Sample

    policy_tokens_total = sum(sum(session.loss_mask) for session in sessions)
    samples = []
    for idx, session in enumerate(sessions):
        if not (
            len(session.response_token_ids)
            == len(session.loss_mask)
            == len(session.rollout_log_probs)
        ):
            raise ValueError(
                "response_token_ids, loss_mask, and rollout_log_probs "
                f"must all have the same length for session {idx}"
            )

        train_metadata = {
            **shared_metrics,
            "metric_schema": "search_agent_subagent_rl",
            "session_idx": idx,
            "session_length": len(session.prompt_token_ids) + len(session.response_token_ids),
            "session_policy_tokens": sum(session.loss_mask),
            "session_policy_token_share": (
                sum(session.loss_mask) / policy_tokens_total if policy_tokens_total else 0.0
            ),
            "turn_count": session.turn_count,
            "turns": [asdict(turn) for turn in session.turn_traces],
            "agent_role": session.agent_role,
            "agent_id": session.agent_id,
            "parent_agent_id": session.parent_agent_id,
            "fork_turn_index": session.fork_turn_index,
            "subagent_index": session.subagent_index,
        }
        if session_ig_enabled():
            train_metadata.update({
                "rollout_id": rollout_id,
                "group_index": group_index,
                "session_index": idx,
                "session_type": session.agent_role,
                # The main session ends in the final answer, regardless of its
                # position in the flattened main/child sample list.
                "is_terminal_session": session.agent_role == "main",
                "ig_before_token_ids": list(session.ig_before_token_ids),
                "ig_after_token_ids": list(session.ig_after_token_ids),
                "ig_answer_token_ids": list(session.ig_answer_token_ids),
                "ig_answer_start": session.ig_answer_start,
                "ig_answer_end": session.ig_answer_end,
            })

        sample = Sample(
            prompt=session.prompt_text,
            response=session.response_text,
            tokens=session.prompt_token_ids + session.response_token_ids,
            response_length=len(session.response_token_ids),
            loss_mask=list(session.loss_mask),
            rollout_log_probs=list(session.rollout_log_probs),
            reward=session.reward,
            rollout_id=rollout_id,
            group_index=group_index,
            label=label,
            metadata={**metadata, "session_index": idx},
            status=Sample.Status.COMPLETED,
            train_metadata=train_metadata,
        )
        samples.append(sample)

    return samples
