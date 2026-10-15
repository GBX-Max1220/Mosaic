from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def _metadata_trajectory_reward(args: Any, sample: Any) -> float | None:
    train_metadata = getattr(sample, "train_metadata", None)
    if isinstance(train_metadata, dict) and "trajectory_raw_reward" in train_metadata:
        value = train_metadata["trajectory_raw_reward"]
    else:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        if "trajectory_raw_reward" in metadata:
            value = metadata["trajectory_raw_reward"]
        elif "raw_reward" in metadata:
            value = metadata["raw_reward"]
        else:
            return None
    if value is None:
        return None
    reward_key = getattr(args, "reward_key", None)
    if reward_key:
        if not isinstance(value, dict) or reward_key not in value:
            return None
        value = value[reward_key]
    return float(value) if isinstance(value, int | float) else None


def grpo_normalize_by_group_index(args, samples: list[Any]) -> tuple[list[float], list[float]]:
    """GRPO reward post-process that groups samples by ``group_index``.

    When one rollout fans out into multiple session samples (context
    compaction), the per-prompt sample count becomes uneven. slime's default
    ``_post_process_rewards`` reshapes the flat reward tensor by
    ``n_samples_per_prompt`` and silently falls back to a single giant group
    when the counts don't match — which mean-centers across the whole batch
    instead of per-prompt.

    Grouping by ``Sample.group_index`` (the data-source per-prompt counter,
    preserved through ``deepcopy`` during fan-out) restores correct per-prompt
    normalization. Within each prompt, compact fan-out siblings share
    ``rollout_id``; normalize once per rollout and broadcast the resulting
    advantage back to each sibling so session count does not change the GRPO
    baseline. If present, ``train_metadata["trajectory_raw_reward"]`` is used
    as the rollout-level reward; ``metadata["trajectory_raw_reward"]`` and
    ``metadata["raw_reward"]`` are still accepted for backwards compatibility.
    Reward is read through ``get_reward_value`` so ``--reward-key`` dict
    rewards keep working.

    Returns ``(raw_rewards, normalized_rewards)`` in the input ``samples``
    order, matching the default's return contract.
    """
    raw_rewards = [s.get_reward_value(args) for s in samples]

    groups: dict[int | None, dict[Any, list[tuple[int, float, Any]]]] = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(samples):
        rollout_id = getattr(s, "rollout_id", None)
        rollout_key = rollout_id if rollout_id is not None else ("sample", i)
        groups[s.group_index][rollout_key].append((i, raw_rewards[i], s))

    normalized = [0.0] * len(samples)
    use_std = getattr(args, "grpo_std_normalization", True)
    zero_std_groups = 0

    for rollouts in groups.values():
        rollout_items = list(rollouts.values())
        rollout_rewards = []
        for indexed in rollout_items:
            raw_reward = next(
                (
                    metadata_reward
                    for _, _, sample in indexed
                    if (metadata_reward := _metadata_trajectory_reward(args, sample)) is not None
                ),
                None,
            )
            if raw_reward is None:
                raw_reward = sum(reward for _, reward, _ in indexed) if len(indexed) > 1 else indexed[0][1]
            rollout_rewards.append(raw_reward)

        rewards = torch.tensor(rollout_rewards, dtype=torch.float)
        centered = rewards - rewards.mean()
        if use_std:
            std = rewards.std(unbiased=False)
            if std.item() == 0.0:
                zero_std_groups += 1
            centered = centered / (std + 1e-6)
        elif len(rollout_rewards) > 1 and all(r == rollout_rewards[0] for r in rollout_rewards):
            zero_std_groups += 1

        for indexed, rollout_advantage in zip(rollout_items, centered.tolist(), strict=True):
            for pos, _, _ in indexed:
                normalized[pos] = rollout_advantage

    # Compute some variance diagnostics for monitoring.
    raw_tensor = torch.tensor(raw_rewards, dtype=torch.float)
    norm_tensor = torch.tensor(normalized, dtype=torch.float)
    print(
        f"[GRPO_STATS] "
        f"raw_reward_mean={raw_tensor.mean().item():.6f} "
        f"raw_reward_var={raw_tensor.var(unbiased=False).item():.6f} "
        f"norm_reward_var={norm_tensor.var(unbiased=False).item():.6f} "
        f"num_groups={len(groups)} "
        f"zero_std_groups={zero_std_groups}",
        flush=True,
    )

    return raw_rewards, normalized
