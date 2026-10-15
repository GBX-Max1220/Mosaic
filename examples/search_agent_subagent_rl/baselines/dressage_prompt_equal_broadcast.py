"""Dressage-style prompt-equal denominator for broadcast GRPO.

Registered via ``--custom-convert-samples-to-train-data-path``.

This intentionally preserves the search-agent broadcast baseline's reward /
advantage behavior and changes only ``rollout_mask_sums`` for GRPO-style
estimators. The denominator is prompt-equal: every prompt group contributes
equal total loss scale regardless of how many rollout/session samples it
produced.
"""

from __future__ import annotations

from typing import Any

from slime.utils.misc import load_function
from slime.utils.types import Sample

from examples.search_agent_subagent_rl.runtime.grpo_hook import grpo_normalize_by_group_index

_PROMPT_EQUAL_ESTIMATORS = ("grpo", "reinforce_plus_plus_baseline")


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

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
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
            "dressage_prompt_equal_broadcast requires prompt grouping."
        )
        live_prompt_ids.add(sample.group_index)
        prompt_token_counts[sample.group_index] = prompt_token_counts.get(sample.group_index, 0) + sum(loss_mask)

    prompt_count = len(live_prompt_ids)
    global_batch_size = int(getattr(args, "global_batch_size", 0))
    scale = prompt_count / global_batch_size if prompt_count > 0 and global_batch_size > 0 else 0.0

    result = []
    for sample in samples:
        if sample.remove_sample or sample.group_index is None:
            result.append(0.0)
        else:
            result.append(float(prompt_token_counts.get(sample.group_index, 0)) * scale)
    return result


def _per_rollout_mask_sums(rollout_ids: list[int], loss_masks: list[list[int]]) -> list[int]:
    mask_sums_per_sample = [sum(mask) for mask in loss_masks]
    rollout_total_mask: dict[int, int] = {}
    for rollout_id, mask_sum in zip(rollout_ids, mask_sums_per_sample, strict=True):
        rollout_total_mask[rollout_id] = rollout_total_mask.get(rollout_id, 0) + mask_sum
    return [rollout_total_mask[rollout_id] for rollout_id in rollout_ids]


def convert_samples_to_train_data(args: Any, samples: list[Sample]) -> dict:
    """Convert samples with Dressage-style prompt-equal GRPO denominators."""
    if getattr(args, "custom_reward_post_process_path", None) is not None:
        raw_rewards, rewards = load_function(args.custom_reward_post_process_path)(args, samples)
    elif getattr(args, "advantage_estimator", None) in _PROMPT_EQUAL_ESTIMATORS:
        raw_rewards, rewards = grpo_normalize_by_group_index(args, samples)
    else:
        raw_rewards = [sample.get_reward_value(args) for sample in samples]
        rewards = raw_rewards

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    rollout_ids = _assign_missing_rollout_ids(samples)
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": rollout_ids,
    }

    loss_masks = _loss_masks(samples)
    train_data["loss_masks"] = loss_masks

    if getattr(args, "advantage_estimator", None) in _PROMPT_EQUAL_ESTIMATORS:
        train_data["rollout_mask_sums"] = _prompt_equal_rollout_mask_sums(args, samples, loss_masks)
    else:
        train_data["rollout_mask_sums"] = _per_rollout_mask_sums(rollout_ids, loss_masks)

    if samples and samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

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
