from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from examples.search_agent_compact_rl.core.config import session_ig_enabled


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
    session_type: str = "normal"
    is_terminal_session: bool = False
    ig_before_token_ids: list[int] = field(default_factory=list)
    ig_after_token_ids: list[int] = field(default_factory=list)
    turn_count: int = 0
    turn_traces: list[TurnTrace] = field(default_factory=list)
    ig_generation_prompt_delta_token_ids: list[int] = field(default_factory=list)
    ig_answer_token_ids: list[int] = field(default_factory=list)
    ig_answer_start: int = 0
    ig_answer_end: int = 0

    @property
    def full_text(self) -> str:
        return self.prompt_text + self.response_text


def sessions_to_samples(
    sessions: list[SessionSegment],
    rollout_id: int,
    group_index: int,
    label: str | None,
    metadata: dict[str, Any],
    compact_metrics: dict[str, Any] | None = None,
) -> list[Any]:
    # list[Any] avoids a circular import at module load time because the Sample
    # class is imported locally inside this function.
    """Convert SessionSegments to slime Sample objects."""
    from slime.utils.types import Sample

    samples = []
    shared_metrics = compact_metrics or {}
    num_segments = len(sessions)
    policy_tokens_total = sum(sum(session.loss_mask) for session in sessions)
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
            "metric_schema": "search_agent_compact_rl",
            **shared_metrics,
            "policy_tokens_total": policy_tokens_total,
            "rollout_id": rollout_id,
            "group_index": group_index,
            "session_index": idx,
            "session_idx": idx,
            "num_segments": num_segments,
            "reward_anchor_segment_index": num_segments - 1,
            "session_length": len(session.prompt_token_ids) + len(session.response_token_ids),
            "session_policy_tokens": sum(session.loss_mask),
            "session_policy_token_share": (
                sum(session.loss_mask) / policy_tokens_total if policy_tokens_total else 0.0
            ),
            "session_type": session.session_type,
            "is_terminal_session": bool(session.is_terminal_session),
            "turn_count": session.turn_count,
            "ig_turns": [asdict(turn) for turn in session.turn_traces],
            "ig_generation_prompt_delta_token_ids": list(
                session.ig_generation_prompt_delta_token_ids
            ),
            "ig_answer_token_ids": list(session.ig_answer_token_ids),
            "ig_answer_start": session.ig_answer_start,
            "ig_answer_end": session.ig_answer_end,
        }
        if session_ig_enabled():
            train_metadata.update(
                {
                    "session_start_token_ids": list(session.prompt_token_ids),
                    "session_end_token_ids": list(
                        session.prompt_token_ids + session.response_token_ids
                    ),
                    "ig_before_token_ids": list(session.ig_before_token_ids),
                    "ig_after_token_ids": list(session.ig_after_token_ids),
                }
            )

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
