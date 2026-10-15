"""Session-level information-gain score requests for setting 2."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from os import environ
from typing import Any

import torch

from slime.utils.misc import load_function
from slime.utils.types import Sample

from examples.search_agent_compact_rl.baselines.dressage_prompt_equal_broadcast import (
    _PROMPT_EQUAL_ESTIMATORS,
    _assign_missing_rollout_ids,
    _loss_masks,
    _per_rollout_mask_sums,
    _prompt_equal_rollout_mask_sums,
)
from examples.search_agent_compact_rl.baselines.ig_scoring import (
    attach_pairwise_score_results,
    build_answer_continuation_from_metadata,
)
from examples.search_agent_compact_rl.runtime.grpo_hook import _metadata_trajectory_reward, grpo_normalize_by_group_index


_SESSION_TYPES = ("normal", "compact")


def _float_env(name: str, default: float) -> float:
    raw = environ.get(name)
    return default if raw is None or raw == "" else float(raw)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_session_ig_score_requests(
    rollout_data: dict, *, session_types: tuple[str, ...] = _SESSION_TYPES
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for sample_index, meta in enumerate(rollout_data.get("metadata") or []):
        if not isinstance(meta, dict):
            continue
        if bool(meta.get("is_terminal_session", False)):
            continue
        session_type = str(meta.get("session_type") or "normal")
        if session_type not in session_types:
            continue
        continuation = build_answer_continuation_from_metadata(meta)
        if continuation is None:
            continue
        before_ids = list(meta.get("ig_before_token_ids") or [])
        after_ids = list(meta.get("ig_after_token_ids") or [])
        if not before_ids or not after_ids:
            continue
        item_key = str(sample_index)
        for kind, context in (("before", before_ids), ("after", after_ids)):
            requests.append(
                {
                    "sample_index": sample_index,
                    "item_key": item_key,
                    "kind": kind,
                    "session_type": session_type,
                    "tokens": context + continuation["answer_token_ids"],
                    "response_length": continuation["response_length"],
                    "answer_start": continuation["answer_start"],
                    "answer_end": continuation["answer_end"],
                }
            )
    return requests


def compute_old_actor_session_ig_deltas(
    args: Any, trainer: Any, rollout_data: dict, *, session_types: tuple[str, ...] = _SESSION_TYPES
) -> None:
    metadata = rollout_data.get("metadata") or []
    for meta in metadata:
        if isinstance(meta, dict):
            meta.setdefault("session_ig_score_results", {})
    requests = build_session_ig_score_requests(rollout_data, session_types=session_types)
    if not requests:
        return
    scores = trainer.compute_ig_request_logprobs(requests)
    attach_pairwise_score_results(metadata, requests, [float(score) for score in scores], "session_ig_score_results")


def _normalize(values: list[float], eps: float) -> tuple[list[float], float, bool]:
    if len(values) <= 1:
        return [0.0 for _ in values], 0.0, True
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False).item()
    if std == 0.0:
        return [0.0 for _ in values], std, True
    return ((tensor - tensor.mean()) / (std + eps)).tolist(), std, False


def _session_lambda(session_type: str) -> float:
    if session_type == "subagent":
        return _float_env("SA_SESSION_IG_LAMBDA_SUBAGENT", 0.1)
    if session_type == "compact":
        return _float_env("SA_SESSION_IG_LAMBDA_COMPACT", 0.1)
    return _float_env("SA_SESSION_IG_LAMBDA_NORMAL", 0.1)


def _slice_response_tensor_to_local_cp(
    args: Any, rollout_data: dict, sample_idx: int, tensor: torch.Tensor
) -> torch.Tensor:
    try:
        from megatron.core import mpu
    except ImportError:
        return tensor

    if mpu.get_context_parallel_world_size() == 1 or "total_lengths" not in rollout_data:
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


def _finite_mean(values: list[float]) -> float:
    return torch.tensor(values, dtype=torch.float32).mean().item() if values else 0.0


def _finite_std(values: list[float]) -> float:
    return torch.tensor(values, dtype=torch.float32).std(unbiased=False).item() if values else 0.0


def _finite_quantile(values: list[float], q: float) -> float:
    return torch.tensor(values, dtype=torch.float32).quantile(q).item() if values else 0.0


def _add_distribution_metrics(metrics: dict[str, float], prefix: str, values: list[float]) -> None:
    metrics[f"{prefix}/count"] = float(len(values))
    metrics[f"{prefix}/mean"] = _finite_mean(values)
    metrics[f"{prefix}/std"] = _finite_std(values)
    metrics[f"{prefix}/p25"] = _finite_quantile(values, 0.25)
    metrics[f"{prefix}/p50"] = _finite_quantile(values, 0.50)
    metrics[f"{prefix}/p75"] = _finite_quantile(values, 0.75)
    metrics[f"{prefix}/abs_mean"] = _finite_mean([abs(value) for value in values])


def _add_bucket_size_metrics(metrics: dict[str, float], prefix: str, sizes: list[int]) -> None:
    metrics[f"{prefix}/count"] = float(len(sizes))
    metrics[f"{prefix}/mean"] = _finite_mean([float(size) for size in sizes])
    metrics[f"{prefix}/min"] = float(min(sizes)) if sizes else 0.0
    metrics[f"{prefix}/max"] = float(max(sizes)) if sizes else 0.0


def _relative_depth_bin(meta: dict[str, Any]) -> str | None:
    session_index = meta.get("session_index")
    num_sessions = meta.get("num_sessions")
    if not isinstance(session_index, int) or not isinstance(num_sessions, int) or num_sessions <= 1:
        return None
    rel_depth = session_index / max(num_sessions - 1, 1)
    if rel_depth < 1.0 / 3.0:
        return "early"
    if rel_depth < 2.0 / 3.0:
        return "mid"
    return "late"


def _is_failed_zero_adv_group(
    idx: int,
    base_adv: float,
    metadata: list[Any],
    raw_rewards: list[float | None],
    eps: float,
) -> bool:
    if abs(base_adv) > eps:
        return False
    raw_reward = raw_rewards[idx] if idx < len(raw_rewards) else None
    if raw_reward is None or raw_reward > eps:
        return False
    meta = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
    group_index = meta.get("group_index")
    group_values = [
        value
        for sample_idx, value in enumerate(raw_rewards)
        if value is not None
        and sample_idx < len(metadata)
        and isinstance(metadata[sample_idx], dict)
        and metadata[sample_idx].get("group_index") == group_index
    ]
    return bool(group_values) and all(value <= eps for value in group_values)


def compute_session_level_advantages(
    args: Any, rollout_data: dict, *, session_types: tuple[str, ...] = _SESSION_TYPES
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    eps = _float_env("SA_SESSION_IG_EPS", 1e-6)
    clip_value = _float_env("SA_SESSION_IG_CLIP", 2.0)
    gating_enabled = _bool_env("SA_SESSION_IG_GATING_ENABLED", False)
    base_adv = [float(value) for value in rollout_data["rewards"]]
    aligned_raw_rewards = rollout_data.get(
        "sample_raw_rewards", rollout_data.get("raw_reward", [None] * len(base_adv))
    )
    raw_rewards = [
        float(value) if isinstance(value, int | float) and isfinite(float(value)) else None
        for value in aligned_raw_rewards
    ]
    metadata = rollout_data.get("metadata") or [{} for _ in base_adv]
    response_lengths = rollout_data["response_lengths"]
    loss_masks = rollout_data["loss_masks"]

    metrics = {
        "scoring/before_phi_mean": 0.0,
        "scoring/after_phi_mean": 0.0,
        "scoring/raw_ig/mean": 0.0,
        "scoring/raw_ig/std": 0.0,
        "scoring/scored_session_count": 0.0,
        "scoring/skipped_session_count": 0.0,
        "normalization/fallback_count": 0.0,
        "fusion/ig_adv/mean": 0.0,
        "fusion/ig_adv/abs_mean": 0.0,
        "fusion/ig_correction/mean": 0.0,
        "fusion/ig_correction/abs_mean": 0.0,
        "fusion/session_adv/mean": 0.0,
        "fusion/session_adv/std": 0.0,
        "fusion/sign_flip_count": 0.0,
        "gating/all_failed_positive/count": 0.0,
        "gating/all_failed_positive/ratio": 0.0,
        "gating/opposite_large/count": 0.0,
        "gating/opposite_large/ratio": 0.0,
        "gating/total/count": 0.0,
        "gating/total/ratio": 0.0,
        "hparams/gating_enabled": 1.0 if gating_enabled else 0.0,
        "hparams/clip": clip_value,
    }
    for session_type in session_types:
        metrics[f"hparams/lambda_{session_type}"] = _session_lambda(session_type)

    buckets: dict[tuple[int | None, str], list[tuple[int, float]]] = defaultdict(list)
    before_scores: list[float] = []
    after_scores: list[float] = []
    raw_deltas: list[float] = []
    raw_by_type: dict[str, list[float]] = {kind: [] for kind in session_types}
    raw_by_rel_depth: dict[str, dict[str, list[float]]] = {
        kind: {"early": [], "mid": [], "late": []} for kind in session_types
    }
    depth_by_index: dict[int, str] = {}
    for idx in range(len(base_adv)):
        meta = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
        if bool(meta.get("is_terminal_session", False)):
            continue
        session_type = str(meta.get("session_type") or "normal")
        if session_type not in session_types:
            continue
        result = (meta.get("session_ig_score_results") or {}).get(str(idx))
        if not isinstance(result, dict) or not result.get("success", False):
            metrics["scoring/skipped_session_count"] += 1.0
            continue
        delta = result.get("delta")
        if not isinstance(delta, int | float) or not isfinite(float(delta)):
            metrics["scoring/skipped_session_count"] += 1.0
            continue
        buckets[(meta.get("group_index"), session_type)].append((idx, float(delta)))
        raw_deltas.append(float(delta))
        raw_by_type[session_type].append(float(delta))
        if (depth_bin := _relative_depth_bin(meta)) is not None:
            depth_by_index[idx] = depth_bin
            raw_by_rel_depth[session_type][depth_bin].append(float(delta))
        if isinstance(result.get("before"), int | float) and isfinite(float(result["before"])):
            before_scores.append(float(result["before"]))
        if isinstance(result.get("after"), int | float) and isfinite(float(result["after"])):
            after_scores.append(float(result["after"]))

    metrics["scoring/before_phi_mean"] = _finite_mean(before_scores)
    metrics["scoring/after_phi_mean"] = _finite_mean(after_scores)
    metrics["scoring/raw_ig/mean"] = _finite_mean(raw_deltas)
    metrics["scoring/raw_ig/std"] = _finite_std(raw_deltas)
    metrics["scoring/scored_session_count"] = float(len(raw_deltas))

    correction_by_index = [0.0] * len(base_adv)
    ig_adv_values: list[float] = []
    norm_by_type: dict[str, list[float]] = {kind: [] for kind in session_types}
    norm_by_rel_depth: dict[str, dict[str, list[float]]] = {
        kind: {"early": [], "mid": [], "late": []} for kind in session_types
    }
    correction_by_type: dict[str, list[float]] = {kind: [] for kind in session_types}
    session_adv_by_type: dict[str, list[float]] = {kind: [] for kind in session_types}
    normalization_stds: dict[str, list[float]] = {kind: [] for kind in session_types}
    bucket_sizes: dict[str, list[int]] = {kind: [] for kind in session_types}
    fallback_counts_by_type: dict[str, float] = {kind: 0.0 for kind in session_types}
    gating_counts_by_type: dict[str, dict[str, float]] = {
        session_type: {"all_failed_positive": 0.0, "opposite_large": 0.0, "total": 0.0}
        for session_type in session_types
    }
    gated_indices: set[int] = set()
    for (_, session_type), items in buckets.items():
        normalized, std, fallback = _normalize([delta for _, delta in items], eps)
        normalization_stds[session_type].append(std)
        bucket_sizes[session_type].append(len(items))
        if fallback:
            metrics["normalization/fallback_count"] += 1.0
            fallback_counts_by_type[session_type] += 1.0
        for (idx, _), normalized_delta in zip(items, normalized, strict=True):
            ig_adv = max(-clip_value, min(clip_value, normalized_delta))
            ig_adv_values.append(ig_adv)
            norm_by_type[session_type].append(ig_adv)
            if (depth_bin := depth_by_index.get(idx)) is not None:
                norm_by_rel_depth[session_type][depth_bin].append(ig_adv)
            correction = _session_lambda(session_type) * ig_adv
            all_failed_positive_gated = (
                gating_enabled
                and correction > 0.0
                and _is_failed_zero_adv_group(idx, base_adv[idx], metadata, raw_rewards, eps)
            )
            opposite_large_gated = (
                gating_enabled
                and base_adv[idx] * correction < 0.0
                and abs(correction) > abs(base_adv[idx])
            )
            if all_failed_positive_gated:
                metrics["gating/all_failed_positive/count"] += 1.0
                gating_counts_by_type[session_type]["all_failed_positive"] += 1.0
            if opposite_large_gated:
                metrics["gating/opposite_large/count"] += 1.0
                gating_counts_by_type[session_type]["opposite_large"] += 1.0
            if all_failed_positive_gated or opposite_large_gated:
                gated_indices.add(idx)
                gating_counts_by_type[session_type]["total"] += 1.0
                correction = 0.0
            correction_by_index[idx] = correction
            correction_by_type[session_type].append(correction)

    for session_type in sorted(session_types):
        metrics[f"normalization/{session_type}/std"] = _finite_mean(normalization_stds[session_type])
        metrics[f"normalization/{session_type}/bucket_count"] = float(len(bucket_sizes[session_type]))
        metrics[f"normalization/{session_type}/fallback_count"] = fallback_counts_by_type[session_type]
        metrics[f"normalization/{session_type}/fallback_ratio"] = (
            fallback_counts_by_type[session_type] / len(bucket_sizes[session_type])
            if bucket_sizes[session_type]
            else 0.0
        )
        _add_bucket_size_metrics(metrics, f"normalization/{session_type}/bucket_size", bucket_sizes[session_type])
        _add_distribution_metrics(metrics, f"by_type/{session_type}/raw_ig", raw_by_type[session_type])
        _add_distribution_metrics(metrics, f"by_type/{session_type}/norm_ig", norm_by_type[session_type])
        for depth_bin in ("early", "mid", "late"):
            _add_distribution_metrics(
                metrics,
                f"by_rel_depth/{session_type}/{depth_bin}/raw_ig",
                raw_by_rel_depth[session_type][depth_bin],
            )
            _add_distribution_metrics(
                metrics,
                f"by_rel_depth/{session_type}/{depth_bin}/norm_ig",
                norm_by_rel_depth[session_type][depth_bin],
            )
    metrics["fusion/ig_adv/mean"] = _finite_mean(ig_adv_values)
    metrics["fusion/ig_adv/abs_mean"] = _finite_mean([abs(value) for value in ig_adv_values])
    metrics["fusion/ig_correction/mean"] = _finite_mean(correction_by_index)
    metrics["fusion/ig_correction/abs_mean"] = _finite_mean([abs(value) for value in correction_by_index])
    if ig_adv_values:
        metrics["gating/all_failed_positive/ratio"] = metrics["gating/all_failed_positive/count"] / len(ig_adv_values)
        metrics["gating/opposite_large/ratio"] = metrics["gating/opposite_large/count"] / len(ig_adv_values)
        metrics["gating/total/count"] = float(len(gated_indices))
        metrics["gating/total/ratio"] = metrics["gating/total/count"] / len(ig_adv_values)
    for session_type in sorted(session_types):
        type_count = len(norm_by_type[session_type])
        for gate_name in ("all_failed_positive", "opposite_large", "total"):
            count = gating_counts_by_type[session_type][gate_name]
            metrics[f"gating/by_type/{session_type}/{gate_name}/count"] = count
            metrics[f"gating/by_type/{session_type}/{gate_name}/ratio"] = count / type_count if type_count else 0.0

    session_advantages = [base + correction for base, correction in zip(base_adv, correction_by_index, strict=True)]
    for idx, session_adv in enumerate(session_advantages):
        meta = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
        session_type = str(meta.get("session_type") or "normal")
        if session_type in session_types and not bool(meta.get("is_terminal_session", False)):
            session_adv_by_type[session_type].append(session_adv)
    metrics["fusion/session_adv/mean"] = _finite_mean(session_advantages)
    metrics["fusion/session_adv/std"] = _finite_std(session_advantages)
    for session_type in sorted(session_types):
        _add_distribution_metrics(metrics, f"fusion/by_type/{session_type}/ig_correction", correction_by_type[session_type])
        _add_distribution_metrics(metrics, f"fusion/by_type/{session_type}/session_adv", session_adv_by_type[session_type])
    metrics["fusion/sign_flip_count"] = float(
        sum(base * session_adv < 0.0 for base, session_adv in zip(base_adv, session_advantages, strict=True))
    )

    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for idx, (response_length, session_adv) in enumerate(zip(response_lengths, session_advantages, strict=True)):
        loss_mask = loss_masks[idx]
        mask = (
            loss_mask.to(dtype=torch.float32)
            if hasattr(loss_mask, "to")
            else torch.tensor(loss_mask, dtype=torch.float32)
        )
        adv = torch.ones(response_length, dtype=torch.float32, device=mask.device) * session_adv
        adv = _slice_response_tensor_to_local_cp(args, rollout_data, idx, adv)
        if mask.numel() != adv.numel():
            mask = _slice_response_tensor_to_local_cp(args, rollout_data, idx, mask)
        adv = adv * mask
        advantages.append(adv)
        returns.append(adv.clone())

    return advantages, returns, metrics


def convert_samples_to_train_data(args: Any, samples: list[Sample]) -> dict:
    """Convert samples with Dressage prompt-equal broadcast semantics."""
    if getattr(args, "custom_reward_post_process_path", None) is not None:
        raw_rewards, rewards = load_function(args.custom_reward_post_process_path)(args, samples)
    elif getattr(args, "advantage_estimator", None) in _PROMPT_EQUAL_ESTIMATORS:
        raw_rewards, rewards = grpo_normalize_by_group_index(args, samples)
        trajectory_raw_rewards = [_metadata_trajectory_reward(args, sample) for sample in samples]
        raw_rewards = [
            trajectory_raw_reward if trajectory_raw_reward is not None else raw_reward
            for trajectory_raw_reward, raw_reward in zip(trajectory_raw_rewards, raw_rewards, strict=True)
        ]
    else:
        raw_rewards = [sample.get_reward_value(args) for sample in samples]
        rewards = raw_rewards

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
        "group_indices": [sample.group_index for sample in samples],
        "loss_masks": loss_masks,
    }
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


def compute_advantages_and_returns(args: Any, rollout_data: dict) -> None:
    advantages, returns, metrics = compute_session_level_advantages(args, rollout_data)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
    rollout_data["session_ig_adv_metrics"] = metrics
