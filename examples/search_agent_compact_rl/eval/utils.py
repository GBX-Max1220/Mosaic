from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def setting_env(setting: str | int) -> dict[str, str]:
    setting = str(setting)
    shared = {
        "SA_COMPACT_MAX_CONTEXT_TOKENS": "131072",
        "SA_COMPACT_RESERVE_TOKENS": "16384",
        "SA_COMPACT_MAX_COMPACTIONS": "10",
        "SA_SUMMARIZER_OFF_POLICY_MAX_TOKENS": "16384",
        "SA_SUMMARIZER_TEMPERATURE": "0.1",
    }
    if setting == "1":
        return {
            **shared,
            "SA_COMPACT_ENABLED": "false",
            "SA_SUMMARIZER_MODE": "on_policy",
            "SA_SUMMARIZER_TRAIN_SUMMARY_SESSION": "false",
        }
    if setting == "2":
        return {
            **shared,
            "SA_COMPACT_ENABLED": "true",
            "SA_SUMMARIZER_MODE": "on_policy",
            "SA_SUMMARIZER_TRAIN_SUMMARY_SESSION": "true",
        }
    raise ValueError(f"Unsupported setting: {setting!r}. Expected one of 1, 2.")


def extract_answer_text(response: str) -> str:
    matches = _ANSWER_RE.findall(response or "")
    return matches[-1].strip() if matches else ""


def extract_eval_metadata(sample: Any) -> dict[str, Any]:
    metadata = _sample_value(sample, "metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def extract_eval_train_metadata(sample: Any) -> dict[str, Any]:
    train_metadata = _sample_value(sample, "train_metadata")
    return dict(train_metadata) if isinstance(train_metadata, dict) else {}


def _serialize_status(status: Any) -> Any:
    return getattr(status, "value", status)


def _sample_value(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def build_compact_eval_record(sample: Any) -> dict[str, Any]:
    metadata = extract_eval_metadata(sample)
    train_metadata = extract_eval_train_metadata(sample)
    response = _sample_value(sample, "response", "") or ""
    record = {
        "index": _sample_value(sample, "index"),
        "prompt": _sample_value(sample, "prompt"),
        "label": _sample_value(sample, "label"),
        "response": response,
        "extracted_answer": extract_answer_text(response),
        "reward": _sample_value(sample, "reward"),
        "status": _serialize_status(_sample_value(sample, "status", "")),
        "metadata": metadata,
    }
    rollout_id = _sample_value(sample, "rollout_id")
    if rollout_id is not None:
        record["rollout_id"] = rollout_id
    if train_metadata:
        record["train_metadata"] = train_metadata
    return record
