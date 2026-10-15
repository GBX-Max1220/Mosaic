"""Broadcast trajectory advantage plus gated subagent information gain."""

from typing import Any

from examples.search_agent_compact_rl.baselines import session_level_ig_adv_dressage_broadcast as session_ig
from examples.search_agent_subagent_rl.baselines.dressage_prompt_equal_broadcast import (
    convert_samples_to_train_data,
)


def build_session_ig_score_requests(rollout_data: dict) -> list[dict[str, Any]]:
    return session_ig.build_session_ig_score_requests(rollout_data, session_types=("subagent",))


def compute_old_actor_session_ig_deltas(args: Any, trainer: Any, rollout_data: dict) -> None:
    session_ig.compute_old_actor_session_ig_deltas(args, trainer, rollout_data, session_types=("subagent",))


def compute_advantages_and_returns(args: Any, rollout_data: dict) -> None:
    advantages, returns, metrics = session_ig.compute_session_level_advantages(
        args, rollout_data, session_types=("subagent",)
    )
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    rollout_data["session_ig_adv_metrics"] = metrics
