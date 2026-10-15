"""Session-aware trajectory-level Dressage broadcast baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from os import environ
from typing import Any

import torch

from slime.utils.types import Sample

_PROMPT_EQUAL_ESTIMATORS = ("grpo", "reinforce_plus_plus_baseline")


@dataclass(frozen=True)
class SessionAwareConfig:
    rho: float
    beta_0: float
    beta_1: float
    session_count_key: str


@dataclass
class RolloutRecord:
    group_index: int | None
    rollout_key: Any
    sample_indices: list[int]
    raw_reward: float
    binary_reward: float
    session_count: float
    outcome_advantage: float = 0.0
    session_signal: float = 0.0
    scale: float | None = None
    traj_advantage: float = 0.0


def _float_env(name: str, default: float) -> float:
    raw = environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _read_session_aware_config() -> SessionAwareConfig:
    config = SessionAwareConfig(
        rho=_float_env("SA_SESSION_AWARE_RHO", 0.2),
        beta_0=_float_env("SA_SESSION_AWARE_BETA_0", 0.1),
        beta_1=_float_env("SA_SESSION_AWARE_BETA_1", 0.2),
        session_count_key=environ.get("SA_SESSION_AWARE_SESSION_COUNT_KEY", "rollout_num_sessions"),
    )
    if not (0.0 <= config.rho < 1.0):
        raise ValueError(f"SA_SESSION_AWARE_RHO must satisfy 0 <= rho < 1, got {config.rho}")
    if not (0.0 <= config.beta_0 < config.beta_1):
        raise ValueError(
            f"SA_SESSION_AWARE_BETA_0/BETA_1 must satisfy 0 <= beta_0 < beta_1, "
            f"got beta_0={config.beta_0}, beta_1={config.beta_1}"
        )
    if not config.session_count_key:
        raise ValueError("SA_SESSION_AWARE_SESSION_COUNT_KEY must be non-empty")
    return config


def _assign_missing_rollout_ids(samples: list[Sample]) -> list[int]:
    rollout_ids = [sample.rollout_id for sample in samples]
    existed_rollout_id_values = set(rid for rid in rollout_ids if rid is not None)
    tmp_id = 0
    for i, rollout_id in enumerate(rollout_ids):
        if rollout_id is not None:
            continue
        while tmp_id in existed_rollout_id_values:
            tmp_id += 1
        rollout_ids[i] = tmp_id
        existed_rollout_id_values.add(tmp_id)
    return rollout_ids


def _loss_masks(samples: list[Sample]) -> list[list[int]]:
    loss_masks = []
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert len(sample.loss_mask) == sample.response_length, (
            f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        )
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    return loss_masks


def _prompt_equal_rollout_mask_sums(args: Any, samples: list[Sample], loss_masks: list[list[int]]) -> list[float]:
    prompt_token_counts: dict[int, int] = {}
    live_prompt_ids: set[int] = set()
    for sample, loss_mask in zip(samples, loss_masks, strict=True):
        if sample.remove_sample:
            continue
        assert sample.group_index is not None, (
            f"live sample at index {getattr(sample, 'index', '?')} has group_index=None; "
            "session_aware_traj_level_dressage_broadcast requires prompt grouping."
        )
        live_prompt_ids.add(sample.group_index)
        prompt_token_counts[sample.group_index] = prompt_token_counts.get(sample.group_index, 0) + sum(loss_mask)
    prompt_count = len(live_prompt_ids)
    global_batch_size = int(getattr(args, "global_batch_size", 0))
    scale = prompt_count / global_batch_size if prompt_count > 0 and global_batch_size > 0 else 0.0
    return [
        0.0
        if sample.remove_sample or sample.group_index is None
        else float(prompt_token_counts.get(sample.group_index, 0)) * scale
        for sample in samples
    ]


def _per_rollout_mask_sums(rollout_ids: list[int], loss_masks: list[list[int]]) -> list[int]:
    mask_sums_per_sample = [sum(mask) for mask in loss_masks]
    rollout_total_mask: dict[int, int] = {}
    for rollout_id, mask_sum in zip(rollout_ids, mask_sums_per_sample, strict=True):
        rollout_total_mask[rollout_id] = rollout_total_mask.get(rollout_id, 0) + mask_sum
    return [rollout_total_mask[rollout_id] for rollout_id in rollout_ids]


def _metadata_trajectory_reward(args: Any, sample: Sample) -> float | None:
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


def _binary_reward(raw_reward: float) -> float:
    rounded = float(round(raw_reward))
    if rounded not in (0.0, 1.0):
        raise ValueError(f"rounded session-aware reward must be 0 or 1, got raw={raw_reward}, rounded={rounded}")
    return rounded


def _session_count(sample: Sample, key: str) -> float:
    metadata = getattr(sample, "train_metadata", None)
    if not isinstance(metadata, dict) or key not in metadata:
        raise ValueError(f"sample rollout_id={sample.rollout_id} missing train_metadata[{key!r}]")
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"train_metadata[{key!r}] must be numeric, got {value!r}")
    if value < 1:
        raise ValueError(f"train_metadata[{key!r}] must be >= 1, got {value}")
    return float(value)


def _centered_rank_signal(values: list[float]) -> tuple[list[float], bool]:
    if len(values) <= 1 or len(set(values)) <= 1:
        return [0.0] * len(values), True
    sorted_values = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(sorted_values):
        end = pos + 1
        while end < len(sorted_values) and sorted_values[end][0] == sorted_values[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for _, idx in sorted_values[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    denom = len(values) - 1
    return [2.0 * ((rank - 1.0) / denom) - 1.0 for rank in ranks], False


def _stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float)
    return {
        f"{prefix}/mean": tensor.mean().item(),
        f"{prefix}/var": tensor.var(unbiased=False).item(),
        f"{prefix}/abs_mean": tensor.abs().mean().item(),
        f"{prefix}/min": tensor.min().item(),
        f"{prefix}/max": tensor.max().item(),
    }


def _broadcast_consistency_error_count(
    records_by_rollout: dict[Any, RolloutRecord],
    assigned_rewards: list[float],
    *,
    tolerance: float = 1e-6,
) -> float:
    error_count = 0
    for record in records_by_rollout.values():
        if not all(
            isclose(assigned_rewards[pos], record.traj_advantage, rel_tol=tolerance, abs_tol=tolerance)
            for pos in record.sample_indices
        ):
            error_count += 1
    return float(error_count)


def _build_rollout_records(args: Any, samples: list[Sample]) -> tuple[list[float], dict[Any, RolloutRecord]]:
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    records: dict[Any, RolloutRecord] = {}
    config = _read_session_aware_config()
    for pos, sample in enumerate(samples):
        rollout_key = sample.rollout_id if sample.rollout_id is not None else ("sample", pos)
        if rollout_key in records:
            records[rollout_key].sample_indices.append(pos)
            continue
        raw_reward = _metadata_trajectory_reward(args, sample)
        if raw_reward is None:
            raw_reward = float(raw_rewards[pos])
        records[rollout_key] = RolloutRecord(
            group_index=sample.group_index,
            rollout_key=rollout_key,
            sample_indices=[pos],
            raw_reward=raw_reward,
            binary_reward=_binary_reward(raw_reward),
            session_count=_session_count(sample, config.session_count_key),
        )
    return raw_rewards, records


def _compute_session_aware_advantages(args: Any, samples: list[Sample]) -> tuple[list[float], list[float], dict[str, float]]:
    config = _read_session_aware_config()
    raw_rewards, records_by_rollout = _build_rollout_records(args, samples)
    groups: dict[int | None, list[RolloutRecord]] = {}
    for record in records_by_rollout.values():
        groups.setdefault(record.group_index, []).append(record)

    metrics = {
        "group/mixed_group_count": 0.0,
        "group/all_success_group_count": 0.0,
        "group/all_failure_group_count": 0.0,
        "group/zero_outcome_std_group_count": 0.0,
        "group/session_subgroup_fallback_count": 0.0,
        "session_signal/rho": config.rho,
        "session_signal/beta_0": config.beta_0,
        "session_signal/beta_1": config.beta_1,
        "advantage/sign_flip_count": 0.0,
        "advantage/broadcast_consistency_error_count": 0.0,
    }

    for records in groups.values():
        rewards = torch.tensor([record.binary_reward for record in records], dtype=torch.float)
        reward_sum = rewards.sum().item()
        mixed = 0.0 < reward_sum < len(records)
        if mixed:
            metrics["group/mixed_group_count"] += 1.0
            centered = rewards - rewards.mean()
            std = rewards.std(unbiased=False)
            if std.item() == 0.0:
                metrics["group/zero_outcome_std_group_count"] += 1.0
            outcome_advantages = (centered / (std + 1e-6)).tolist() if getattr(args, "grpo_std_normalization", True) else centered.tolist()
            for record, advantage in zip(records, outcome_advantages, strict=True):
                record.outcome_advantage = advantage
        elif reward_sum == len(records):
            metrics["group/all_success_group_count"] += 1.0
            metrics["group/zero_outcome_std_group_count"] += 1.0
        else:
            metrics["group/all_failure_group_count"] += 1.0
            metrics["group/zero_outcome_std_group_count"] += 1.0

        for reward_value in (0.0, 1.0):
            subgroup = [record for record in records if record.binary_reward == reward_value]
            if not subgroup:
                continue
            signals, fallback = _centered_rank_signal([record.session_count for record in subgroup])
            if fallback:
                metrics["group/session_subgroup_fallback_count"] += 1.0
            for record, signal in zip(subgroup, signals, strict=True):
                record.session_signal = signal

        for record in records:
            if mixed:
                record.scale = 1.0 - config.rho * record.session_signal
                record.traj_advantage = record.scale * record.outcome_advantage
                if record.outcome_advantage != 0.0 and record.traj_advantage * record.outcome_advantage <= 0.0:
                    metrics["advantage/sign_flip_count"] += 1.0
            elif reward_sum == len(records):
                record.traj_advantage = -config.beta_1 * record.session_signal
            else:
                record.traj_advantage = config.beta_0 * min(record.session_signal, 0.0)

    rewards = [0.0] * len(samples)
    for record in records_by_rollout.values():
        for pos in record.sample_indices:
            rewards[pos] = record.traj_advantage
    metrics["advantage/broadcast_consistency_error_count"] = _broadcast_consistency_error_count(
        records_by_rollout, rewards
    )

    metrics |= _stats("session_signal/u", [record.session_signal for record in records_by_rollout.values()])
    metrics |= _stats("session_signal/scale", [record.scale for record in records_by_rollout.values() if record.scale is not None])
    metrics |= _stats("advantage/outcome_adv", [record.outcome_advantage for record in records_by_rollout.values()])
    metrics |= _stats("advantage/traj_adv", [record.traj_advantage for record in records_by_rollout.values()])
    return raw_rewards, rewards, metrics


def convert_samples_to_train_data(args: Any, samples: list[Sample]) -> dict:
    raw_rewards, rewards, session_aware_metrics = _compute_session_aware_advantages(args, samples)
    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    rollout_ids = _assign_missing_rollout_ids(samples)
    loss_masks = _loss_masks(samples)
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": rollout_ids,
        "loss_masks": loss_masks,
        "session_aware_metrics": session_aware_metrics,
    }
    if getattr(args, "advantage_estimator", None) in _PROMPT_EQUAL_ESTIMATORS:
        train_data["rollout_mask_sums"] = _prompt_equal_rollout_mask_sums(args, samples, loss_masks)
    else:
        train_data["rollout_mask_sums"] = _per_rollout_mask_sums(rollout_ids, loss_masks)
    if samples and samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    if samples and samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
    if samples and samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]
    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]
    if samples and samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]
    return train_data
