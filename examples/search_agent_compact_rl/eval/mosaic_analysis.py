"""Raw analysis records for Mosaic mechanism figures."""

from __future__ import annotations

import json
import os
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

from examples.search_agent_compact_rl.baselines.session_aware_traj_level_dressage_broadcast import RolloutRecord


def analysis_output_dir() -> Path | None:
    raw_path = os.environ.get("SA_MOSAIC_ANALYSIS_OUTPUT_DIR", "").strip()
    return Path(raw_path) if raw_path else None


def append_jsonl_records(output_dir: Path, filename: str, records: Iterable[dict[str, Any]]) -> None:
    records = list(records)
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / filename).open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_trajectory_ordering_records(records: Iterable[RolloutRecord]) -> list[dict[str, Any]]:
    """Return one serializable ordering record for each rollout."""
    return [
        {
            "group_index": record.group_index,
            "rollout_id": record.rollout_key,
            "sample_indices": list(record.sample_indices),
            "raw_reward": record.raw_reward,
            "binary_reward": record.binary_reward,
            "session_count": record.session_count,
            "outcome_advantage": record.outcome_advantage,
            "session_signal": record.session_signal,
            "scale": record.scale,
            "trajectory_advantage": record.traj_advantage,
            "group_type": record.group_type,
            "session_signal_fallback": record.session_signal_fallback,
        }
        for record in records
    ]


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    value = float(value)
    return value if isfinite(value) else None


def build_session_ig_records(rollout_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one raw score record per training session.

    Terminal and failed-scoring sessions are retained with an explicit status so
    offline analyses can report coverage instead of silently dropping them.
    """
    records: list[dict[str, Any]] = []
    for sample_index, meta in enumerate(rollout_data.get("metadata") or []):
        if not isinstance(meta, dict):
            continue
        num_sessions = meta.get("num_sessions")
        session_index = meta.get("session_index")
        relative_depth = None
        if isinstance(num_sessions, int) and isinstance(session_index, int) and num_sessions > 1:
            relative_depth = session_index / (num_sessions - 1)

        record: dict[str, Any] = {
            "group_index": meta.get("group_index"),
            "rollout_id": meta.get("rollout_id"),
            "session_index": session_index,
            "num_sessions": num_sessions,
            "session_type": str(meta.get("session_type") or "normal"),
            "relative_depth": relative_depth,
            "is_terminal_session": bool(meta.get("is_terminal_session", False)),
            "score_status": "missing_score",
            "phi_before": None,
            "phi_after": None,
            "raw_ig": None,
        }
        if record["is_terminal_session"]:
            record["score_status"] = "terminal"
        else:
            result = (meta.get("session_ig_score_results") or {}).get(str(sample_index))
            if isinstance(result, dict) and result.get("success", False):
                before = _finite_float(result.get("before"))
                after = _finite_float(result.get("after"))
                delta = _finite_float(result.get("delta"))
                if before is not None and after is not None and delta is not None:
                    record.update(
                        score_status="scored",
                        phi_before=before,
                        phi_after=after,
                        raw_ig=delta,
                    )
                else:
                    record["score_status"] = "invalid_score"
            elif isinstance(result, dict):
                record["score_status"] = "score_failed"
        records.append(record)
    return records
