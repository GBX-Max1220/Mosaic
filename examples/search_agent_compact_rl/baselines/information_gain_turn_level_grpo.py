from __future__ import annotations

from collections import defaultdict
from math import isfinite
from os import environ
from typing import Any

import torch
from examples.search_agent_compact_rl.baselines.ig_scoring import (
    attach_pairwise_score_results,
    build_answer_continuation_from_metadata,
)
from examples.search_agent_compact_rl.runtime.grpo_hook import grpo_normalize_by_group_index

from slime.utils.misc import load_function
from slime.utils.types import Sample


def _float_env(name: str, default: float) -> float:
    raw = environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ig_tool_filter() -> set[str]:
    raw = environ.get("SA_IG_TOOL_FILTER", "search,visit")
    return {name.strip().lower() for name in raw.split(",") if name.strip()}


def _loss_masks(samples: list[Sample]) -> list[list[int]]:
    masks = []
    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert len(sample.loss_mask) == sample.response_length
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        masks.append(sample.loss_mask)
    return masks


def _assign_missing_rollout_ids(samples: list[Sample]) -> list[int]:
    rollout_ids = [sample.rollout_id for sample in samples]
    used = {rollout_id for rollout_id in rollout_ids if rollout_id is not None}
    next_id = 0
    for idx, rollout_id in enumerate(rollout_ids):
        if rollout_id is not None:
            continue
        while next_id in used:
            next_id += 1
        rollout_ids[idx] = next_id
        used.add(next_id)
    return rollout_ids


def _per_rollout_mask_sums(rollout_ids: list[int], loss_masks: list[list[int]]) -> list[int]:
    totals: dict[int, int] = {}
    for rollout_id, mask in zip(rollout_ids, loss_masks, strict=True):
        totals[rollout_id] = totals.get(rollout_id, 0) + sum(mask)
    return [totals[rollout_id] for rollout_id in rollout_ids]


def convert_samples_to_train_data(args: Any, samples: list[Sample]) -> dict:
    if getattr(args, "custom_reward_post_process_path", None) is not None:
        raw_rewards, rewards = load_function(args.custom_reward_post_process_path)(args, samples)
    else:
        raw_rewards, rewards = grpo_normalize_by_group_index(args, samples)

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
        "group_indices": [sample.group_index for sample in samples],
        "loss_masks": loss_masks,
        "rollout_mask_sums": _per_rollout_mask_sums(rollout_ids, loss_masks),
    }
    if samples and samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    if samples and samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]
    return train_data


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return list(value)


def sum_ig_answer_logprobs_in_local_spans(
    logprobs: torch.Tensor,
    answer_start: int,
    answer_end: int,
    response_spans: list[tuple[int, int]],
) -> tuple[torch.Tensor, int]:
    total = logprobs.new_zeros(())
    count = 0
    local_start = 0
    for span_start, span_end in response_spans:
        span_length = span_end - span_start
        if span_length < 0:
            raise ValueError(f"Invalid response span: {(span_start, span_end)}")
        overlap_start = max(answer_start, span_start)
        overlap_end = min(answer_end, span_end)
        if overlap_end > overlap_start:
            start = local_start + overlap_start - span_start
            end = local_start + overlap_end - span_start
            total += logprobs[start:end].sum()
            count += end - start
        local_start += span_length
    if local_start != logprobs.numel():
        raise ValueError(f"Response spans cover {local_start} tokens, but received {logprobs.numel()} logprobs")
    return total, count


def build_ig_score_requests(rollout_data: dict) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    tool_filter = _ig_tool_filter()
    metadata = rollout_data.get("metadata") or []
    for sample_index, meta in enumerate(metadata):
        if not isinstance(meta, dict):
            continue
        continuation = build_answer_continuation_from_metadata(meta)
        if continuation is None:
            continue
        answer_ids = continuation["answer_token_ids"]
        answer_start = continuation["answer_start"]
        answer_end = continuation["answer_end"]
        response_length = int(rollout_data["response_lengths"][sample_index])
        tokens = _as_list(rollout_data["tokens"][sample_index])
        prompt_len = len(tokens) - response_length
        gen_delta_ids = list(meta.get("ig_generation_prompt_delta_token_ids") or [])
        for turn in meta.get("ig_turns", []):
            tool_name = str(turn.get("tool_name") or "").lower()
            if turn.get("turn_type") != "tool_call" or tool_name not in tool_filter:
                continue
            before_end = int(turn.get("context_before_response_end", -1))
            after_end = turn.get("context_after_response_end")
            if before_end < 0 or after_end is None:
                continue
            after_end = int(after_end)
            before_context = tokens[: prompt_len + before_end]
            after_context = tokens[: prompt_len + after_end]
            if turn.get("context_after_needs_generation_prompt", False):
                after_context = after_context + gen_delta_ids
            for kind, context in (("before", before_context), ("after", after_context)):
                requests.append(
                    {
                        "sample_index": sample_index,
                        "turn_index": int(turn["turn_index"]),
                        "item_key": str(int(turn["turn_index"])),
                        "kind": kind,
                        "tokens": context + answer_ids,
                        "response_length": len(answer_ids),
                        "answer_start": answer_start,
                        "answer_end": answer_end,
                    }
                )
    return requests


def compute_old_actor_gt_logprob_deltas(args: Any, trainer: Any, rollout_data: dict) -> None:
    requests = build_ig_score_requests(rollout_data)
    for meta in rollout_data.get("metadata") or []:
        if isinstance(meta, dict):
            meta.setdefault("ig_score_results", {})
    if not requests:
        return
    scores = trainer.compute_ig_request_logprobs(requests)
    attach_pairwise_score_results(
        rollout_data["metadata"],
        requests,
        [float(score) for score in scores],
        "ig_score_results",
    )


def _normalize(values: list[float], eps: float) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, dtype=torch.float32)
    return ((tensor - tensor.mean()) / (tensor.std(unbiased=False) + eps)).tolist()


def _turn_process_reward(
    metadata: dict[str, Any], turn: dict[str, Any], fmt_penalty: float, tool_filter: set[str]
) -> float | None:
    turn_type = turn.get("turn_type")
    if turn_type == "malformed":
        return -fmt_penalty
    tool_name = str(turn.get("tool_name") or "").lower()
    if turn_type != "tool_call" or tool_name not in tool_filter:
        return None
    results = metadata.get("ig_score_results", {})
    key = str(turn.get("turn_index"))
    result = results.get(key)
    if not isinstance(result, dict) or not result.get("success", False):
        return None
    delta = result.get("delta")
    return float(delta) if isinstance(delta, int | float) and isfinite(float(delta)) else None


def _slice_response_tensor_to_local_cp(
    args: Any, rollout_data: dict, sample_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    try:
        from megatron.core import mpu
    except ImportError:
        return tensor

    if mpu.get_context_parallel_world_size() == 1:
        return tensor
    if "total_lengths" not in rollout_data:
        return tensor

    from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

    max_seq_lens = rollout_data.get("max_seq_lens")
    max_seq_len = max_seq_lens[sample_idx] if max_seq_lens is not None else None
    return slice_log_prob_with_cp(
        tensor,
        int(rollout_data["total_lengths"][sample_idx]),
        int(rollout_data["response_lengths"][sample_idx]),
        getattr(args, "qkv_format", "thd"),
        max_seq_len,
    )


def compute_turn_level_advantages(
    args: Any, rollout_data: dict
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    eps = _float_env("SA_IG_EPS", 1e-6)
    gamma = _float_env("SA_IG_GAMMA", 0.95)
    fmt_penalty = _float_env("SA_IG_FORMAT_PENALTY", 1.0)
    lambda_ig = _float_env("SA_IG_LAMBDA", 1.0)
    weighted_fusion_enabled = _bool_env("SA_IG_WEIGHTED_FUSION_ENABLED", False)
    rewards = [float(x) for x in rollout_data["rewards"]]
    metadata = rollout_data.get("metadata") or [{} for _ in rewards]
    response_lengths = rollout_data["response_lengths"]
    loss_masks = rollout_data["loss_masks"]

    groups: dict[int | None, list[int]] = defaultdict(list)
    for idx, meta in enumerate(metadata):
        groups[meta.get("group_index")].append(idx)
    if all("group_index" not in meta for meta in metadata):
        sample_group_indices = rollout_data.get("group_indices")
        if sample_group_indices:
            groups.clear()
            for idx, group_index in enumerate(sample_group_indices):
                groups[group_index].append(idx)

    outcome_adv = [0.0] * len(rewards)
    process_adv_by_sample_turn: dict[tuple[int, int], float] = {}
    metrics = {
        "outcome_adv/mean": 0.0,
        "outcome_adv/var": 0.0,
        "process/raw_ig/mean": 0.0,
        "process/raw_ig/var": 0.0,
        "process/format_penalty_count": 0.0,
        "process/scored_tool_turn_count": 0.0,
        "process/malformed_turn_count": 0.0,
        "process/group_no_process_reward_count": 0.0,
        "gt_logprob/scoring_failure_count": 0.0,
        "process/outcome_abs_mean": 0.0,
        "process/normalized_abs_mean_before_scale": 0.0,
        "process/normalized_abs_mean_after_scale": 0.0,
        "process/ig_scale": 1.0,
        "fusion/weighted_enabled": 1.0 if weighted_fusion_enabled else 0.0,
        "fusion/lambda_ig": lambda_ig,
    }

    raw_ig_values: list[float] = []
    normalized_process_adv_values: list[float] = []
    tool_filter = _ig_tool_filter()
    for sample_indices in groups.values():
        norm_out = _normalize([rewards[idx] for idx in sample_indices], eps)
        for idx, value in zip(sample_indices, norm_out, strict=True):
            outcome_adv[idx] = value

        proc_items: list[tuple[int, int, float]] = []
        for idx in sample_indices:
            meta = metadata[idx] if isinstance(metadata[idx], dict) else {}
            for turn in meta.get("ig_turns", []):
                reward = _turn_process_reward(meta, turn, fmt_penalty, tool_filter)
                if turn.get("turn_type") == "malformed":
                    metrics["process/malformed_turn_count"] += 1.0
                if reward is None:
                    tool_name = str(turn.get("tool_name") or "").lower()
                    if turn.get("turn_type") == "tool_call" and tool_name in tool_filter:
                        metrics["gt_logprob/scoring_failure_count"] += 1.0
                    continue
                if turn.get("turn_type") == "malformed":
                    metrics["process/format_penalty_count"] += 1.0
                else:
                    metrics["process/scored_tool_turn_count"] += 1.0
                    raw_ig_values.append(reward)
                proc_items.append((idx, int(turn["turn_index"]), reward))
        if not proc_items:
            metrics["process/group_no_process_reward_count"] += 1.0
            continue
        normalized_process = _normalize([item[2] for item in proc_items], eps)
        normalized_process_adv_values.extend(float(value) for value in normalized_process if isfinite(float(value)))
        for (idx, turn_index, _), adv in zip(proc_items, normalized_process, strict=True):
            process_adv_by_sample_turn[(idx, turn_index)] = adv

    finite_outcome_adv = [value for value in outcome_adv if isfinite(value)]
    if finite_outcome_adv:
        outcome_adv_tensor = torch.tensor(finite_outcome_adv, dtype=torch.float32)
        metrics["outcome_adv/mean"] = outcome_adv_tensor.mean().item()
        metrics["outcome_adv/var"] = outcome_adv_tensor.var(unbiased=False).item()

    if finite_outcome_adv:
        outcome_abs_mean = torch.tensor(finite_outcome_adv, dtype=torch.float32).abs().mean()
    else:
        outcome_abs_mean = torch.tensor(1.0, dtype=torch.float32)
    outcome_abs_mean = torch.clamp(outcome_abs_mean, min=0.3)
    if normalized_process_adv_values:
        process_abs_mean = torch.tensor(normalized_process_adv_values, dtype=torch.float32).abs().mean()
    else:
        process_abs_mean = torch.tensor(1.0, dtype=torch.float32)
    ig_scale = torch.clamp(outcome_abs_mean / (process_abs_mean + eps), max=10.0).item()
    if process_adv_by_sample_turn:
        process_adv_by_sample_turn = {
            key: float(value) * ig_scale for key, value in process_adv_by_sample_turn.items()
        }
    metrics["process/outcome_abs_mean"] = outcome_abs_mean.item()
    metrics["process/normalized_abs_mean_before_scale"] = process_abs_mean.item()
    metrics["process/normalized_abs_mean_after_scale"] = process_abs_mean.item() * ig_scale
    metrics["process/ig_scale"] = ig_scale

    if raw_ig_values:
        raw_ig_tensor = torch.tensor(raw_ig_values, dtype=torch.float32)
        metrics["process/raw_ig/mean"] = raw_ig_tensor.mean().item()
        metrics["process/raw_ig/var"] = raw_ig_tensor.var(unbiased=False).item()

    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    all_turn_adv_values: list[float] = []
    for idx, response_length in enumerate(response_lengths):
        mask = (
            loss_masks[idx].to(dtype=torch.float32)
            if hasattr(loss_masks[idx], "to")
            else torch.tensor(loss_masks[idx], dtype=torch.float32)
        )
        adv = torch.zeros(response_length, dtype=torch.float32, device=mask.device)
        turns = list((metadata[idx] or {}).get("ig_turns", []))
        if not turns:
            adv += float(outcome_adv[idx])
        else:
            turn_rewards = [0.0] * len(turns)
            for pos, turn in enumerate(turns):
                turn_rewards[pos] += process_adv_by_sample_turn.get((idx, int(turn.get("turn_index", pos))), 0.0)
            running = 0.0
            turn_advs = [0.0] * len(turns)
            if weighted_fusion_enabled:
                for pos in range(len(turns) - 1, -1, -1):
                    running = turn_rewards[pos] + gamma * running
                    turn_advs[pos] = float(outcome_adv[idx]) + lambda_ig * running
            else:
                turn_rewards[-1] += outcome_adv[idx]
                for pos in range(len(turns) - 1, -1, -1):
                    running = turn_rewards[pos] + gamma * running
                    turn_advs[pos] = running
            for turn, value in zip(turns, turn_advs, strict=True):
                start = max(0, int(turn.get("policy_token_start", 0)))
                end = min(response_length, int(turn.get("policy_token_end", response_length)))
                if end > start:
                    adv[start:end] = value
                    all_turn_adv_values.append(float(value))
        adv = _slice_response_tensor_to_local_cp(args, rollout_data, idx, adv)
        if mask.numel() != adv.numel():
            mask = _slice_response_tensor_to_local_cp(args, rollout_data, idx, mask)
        adv = adv * mask
        advantages.append(adv)
        returns.append(adv.clone())

    if all_turn_adv_values:
        turn_adv_tensor = torch.tensor(all_turn_adv_values, dtype=torch.float32)
        metrics["turn_adv/mean"] = turn_adv_tensor.mean().item()
        metrics["turn_adv/var"] = turn_adv_tensor.var(unbiased=False).item()
        metrics["turn_adv/min"] = turn_adv_tensor.min().item()
        metrics["turn_adv/max"] = turn_adv_tensor.max().item()
    return advantages, returns, metrics


def compute_advantages_and_returns(args: Any, rollout_data: dict) -> None:
    advantages, returns, metrics = compute_turn_level_advantages(args, rollout_data)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    rollout_data["ig_turn_level_metrics"] = metrics
