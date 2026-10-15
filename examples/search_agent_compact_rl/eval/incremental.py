from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock

from examples.search_agent_compact_rl.eval.postprocess import write_compact_jsonl
from examples.search_agent_compact_rl.eval.utils import build_compact_eval_record
from slime.utils.types import Sample

_FORMAT_VERSION = 1
_RUN_ARG_NAMES = (
    "hf_checkpoint",
    "custom_generate_function_path",
    "apply_chat_template",
    "eval_temperature",
    "eval_top_p",
    "eval_top_k",
    "eval_max_response_len",
    "rollout_temperature",
    "rollout_top_p",
    "rollout_top_k",
    "rollout_max_response_len",
    "rollout_max_context_len",
    "rollout_stop",
    "rollout_stop_token_ids",
    "rollout_skip_special_tokens",
    "rollout_seed",
)
_RUN_ENV_NAMES = (
    "SA_EVAL_DATASET_NAME",
    "SA_EVAL_DATASET_PATH",
    "SA_EVAL_SETTING",
    "SA_TOOL_BACKEND",
    "SA_COMPACT_ENABLED",
    "SA_COMPACT_MAX_CONTEXT_TOKENS",
    "SA_COMPACT_RESERVE_TOKENS",
    "SA_COMPACT_MAX_COMPACTIONS",
    "SA_MAX_CONSECUTIVE_MALFORMED_TOOL_CALLS",
    "SA_SUMMARIZER_MODE",
    "SA_SUMMARIZER_TRAIN_SUMMARY_SESSION",
    "SA_SUMMARIZER_OFF_POLICY_MODEL_NAME",
    "SA_SUMMARIZER_OFF_POLICY_BASE_URL",
    "SA_SUMMARIZER_OFF_POLICY_MAX_TOKENS",
    "SA_SUMMARIZER_TEMPERATURE",
    "JUDGE_BASE_URL",
    "JUDGE_MODEL",
    "JUDGE_TIMEOUT_SECONDS",
    "JUDGE_MAX_RETRIES",
    "SERPER_BASE_URL",
    "JINA_BASE_URL",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _result_dir() -> Path | None:
    value = os.environ.get("SA_EVAL_INCREMENTAL_DIR", "").strip()
    return Path(value) if value else None


@lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parents[1]
    source_paths = []
    for directory_name in ("core", "runtime", "tools"):
        source_paths.extend((package_root / directory_name).rglob("*.py"))
    source_paths.extend(
        (
            Path(__file__).resolve(),
            repository_root / "slime/rollout/sglang_rollout.py",
            repository_root / "slime/utils/types.py",
        )
    )

    digest = hashlib.sha256()
    for source_path in sorted(set(source_paths)):
        relative_path = source_path.relative_to(repository_root)
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_identity(args: Any) -> dict[str, Any]:
    arg_values = {name: getattr(args, name, None) for name in _RUN_ARG_NAMES}
    checkpoint = arg_values.get("hf_checkpoint")
    if checkpoint:
        arg_values["hf_checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    return {
        "format_version": _FORMAT_VERSION,
        "code_fingerprint": _code_fingerprint(),
        "args": arg_values,
        "env": {name: os.environ.get(name, "") for name in _RUN_ENV_NAMES},
    }


def _sample_identity(sample: Sample) -> dict[str, Any]:
    identity = {
        "index": sample.index,
        "group_index": sample.group_index,
        "prompt": sample.prompt,
        "label": sample.label,
        "metadata": sample.metadata,
    }
    return {"digest": _digest(identity), **identity}


def incremental_run_identity(args: Any) -> dict[str, Any]:
    """Return the validated identity used by the incremental rollout cache."""

    return _run_identity(args)


def incremental_sample_identity(sample: Sample) -> dict[str, Any]:
    """Return the validated identity used for one cached source sample."""

    return _sample_identity(sample)


def _sample_path(result_dir: Path, sample: Sample) -> Path:
    if not isinstance(sample.index, int) or sample.index < 0:
        raise ValueError("Incremental eval requires each source sample to have a non-negative integer index")
    return result_dir / "samples" / f"{sample.index:08d}.pt"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, indent=2, ensure_ascii=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_manifest(result_dir: Path, args: Any, *, create: bool) -> dict[str, Any]:
    manifest_path = result_dir / "manifest.json"
    expected_identity = _run_identity(args)
    with FileLock(str(result_dir / ".manifest.lock")):
        if not manifest_path.exists():
            if any((result_dir / "samples").glob("*.pt")):
                raise ValueError(
                    "Incremental eval manifest is missing while cached sample files exist; "
                    "use a new SA_EVAL_INCREMENTAL_DIR or restore the manifest"
                )
            if not create:
                return expected_identity
            _atomic_write_json(manifest_path, expected_identity)
            return expected_identity
        actual_identity = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_identity != expected_identity:
            raise ValueError(
                "Incremental eval run identity does not match the existing manifest; "
                "use a new SA_EVAL_INCREMENTAL_DIR for this evaluation configuration"
            )
    return expected_identity


def _is_cacheable(samples: list[Sample]) -> bool:
    valid_statuses = {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}
    return bool(samples) and all(sample.status in valid_statuses and sample.reward is not None for sample in samples)


def save_incremental_eval_result(args: Any, source_sample: Sample, samples: list[Sample]) -> Path | None:
    result_dir = _result_dir()
    if result_dir is None or not _is_cacheable(samples):
        return None

    result_dir.mkdir(parents=True, exist_ok=True)
    run_identity = _validate_manifest(result_dir, args, create=True)
    sample_path = _sample_path(result_dir, source_sample)
    sample_identity = _sample_identity(source_sample)
    payload = {
        "format_version": _FORMAT_VERSION,
        "run_identity_digest": _digest(run_identity),
        "sample_identity": sample_identity,
        "samples": [sample.to_dict() for sample in samples],
    }

    with FileLock(f"{sample_path}.lock"):
        if sample_path.exists():
            existing = torch.load(sample_path, map_location="cpu", weights_only=False)
            if existing.get("run_identity_digest") != _digest(run_identity):
                raise ValueError("Incremental eval run identity does not match the cached sample")
            if existing.get("sample_identity") != sample_identity:
                raise ValueError("Incremental eval sample identity does not match the cached sample")
            return sample_path
        _atomic_torch_save(sample_path, payload)
    return sample_path


def load_incremental_eval_result(args: Any, source_sample: Sample) -> list[Sample] | None:
    result_dir = _result_dir()
    if result_dir is None or not result_dir.exists():
        return None

    run_identity = _validate_manifest(result_dir, args, create=False)
    sample_path = _sample_path(result_dir, source_sample)
    if not sample_path.exists():
        return None

    with FileLock(f"{sample_path}.lock"):
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError(f"Unsupported incremental eval format: {payload.get('format_version')!r}")
    if payload.get("run_identity_digest") != _digest(run_identity):
        raise ValueError("Incremental eval run identity does not match the cached sample")
    if payload.get("sample_identity") != _sample_identity(source_sample):
        raise ValueError("Incremental eval sample identity does not match the cached sample")

    samples_data = payload.get("samples")
    if not isinstance(samples_data, list):
        raise ValueError(f"Invalid incremental eval sample payload: {sample_path}")
    samples = [Sample.from_dict(value) for value in samples_data]
    if not _is_cacheable(samples):
        raise ValueError(f"Incremental eval cache contains an incomplete trajectory: {sample_path}")
    return samples


def export_incremental_results(input_dir: Path, output_jsonl: Path) -> int:
    sample_paths = sorted((input_dir / "samples").glob("*.pt"))
    records = []
    for sample_path in sample_paths:
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        samples_data = payload.get("samples")
        if not isinstance(samples_data, list):
            raise ValueError(f"Invalid incremental eval sample payload: {sample_path}")
        records.extend(build_compact_eval_record(Sample.from_dict(value)) for value in samples_data)
    write_compact_jsonl(records, output_jsonl)
    return len(sample_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage incremental search-agent evaluation results")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Export completed trajectories to JSONL")
    export_parser.add_argument("--input-dir", type=Path, required=True)
    export_parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        count = export_incremental_results(args.input_dir, args.output_jsonl)
        print(f"Exported {count} completed trajectories to {args.output_jsonl}")


if __name__ == "__main__":
    main()
