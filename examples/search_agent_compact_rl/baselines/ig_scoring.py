from __future__ import annotations

from math import isfinite
from typing import Any

import torch


def build_answer_continuation_from_metadata(meta: dict[str, Any]) -> dict[str, Any] | None:
    answer_token_ids = list(meta.get("ig_answer_token_ids") or [])
    answer_start = int(meta.get("ig_answer_start", 0))
    answer_end = int(meta.get("ig_answer_end", 0))
    if not answer_token_ids or answer_end <= answer_start:
        return None
    if answer_start < 0 or answer_end > len(answer_token_ids):
        return None
    return {
        "answer_token_ids": answer_token_ids,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "response_length": len(answer_token_ids),
    }


def average_answer_logprob(total_logprob: torch.Tensor | float, count: int) -> float:
    if count <= 0:
        raise ValueError("answer logprob count must be positive")
    value = float(total_logprob.item() if hasattr(total_logprob, "item") else total_logprob)
    return value / count


def attach_pairwise_score_results(
    metadata: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    scores: list[float],
    result_key: str,
) -> None:
    pending: dict[tuple[int, str], dict[str, float]] = {}
    for request, score in zip(requests, scores, strict=True):
        sample_index = int(request["sample_index"])
        item_key = str(request["item_key"])
        kind = str(request["kind"])
        pending.setdefault((sample_index, item_key), {})[kind] = float(score)

    for (sample_index, item_key), values in pending.items():
        meta = metadata[sample_index]
        before = values.get("before")
        after = values.get("after")
        if before is None or after is None or not isfinite(before) or not isfinite(after):
            result = {"success": False}
        else:
            result = {"before": before, "after": after, "delta": after - before, "success": True}
        meta.setdefault(result_key, {})[item_key] = result
