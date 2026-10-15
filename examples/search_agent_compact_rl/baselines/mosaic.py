"""Mosaic: session-aware trajectory credit with gated session-level IG."""

from examples.search_agent_compact_rl.baselines.session_aware_traj_level_dressage_broadcast import (
    convert_samples_to_train_data,
)
from examples.search_agent_compact_rl.baselines.session_level_ig_adv_dressage_broadcast import (
    compute_advantages_and_returns,
    compute_old_actor_session_ig_deltas,
)


__all__ = [
    "compute_advantages_and_returns",
    "compute_old_actor_session_ig_deltas",
    "convert_samples_to_train_data",
]
