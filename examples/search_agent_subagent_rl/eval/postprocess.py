from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.search_agent_compact_rl.eval.utils import (
    build_compact_eval_record,
    extract_answer_text,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _record_trajectory_key(record: dict[str, Any], record_index: int) -> tuple[str, Any]:
    rollout_id = record.get("rollout_id")
    if rollout_id is not None:
        return "rollout", rollout_id
    metadata = _mapping(record.get("metadata"))
    query_id = metadata.get("query_id")
    if query_id not in (None, ""):
        return "query", str(query_id)
    return "record", record_index


def _trajectory_reward(records: list[dict[str, Any]]) -> float | None:
    for record in records:
        for metadata_key in ("metadata", "train_metadata"):
            value = _numeric_value(_mapping(record.get(metadata_key)).get("trajectory_raw_reward"))
            if value is not None:
                return value
    if len(records) == 1:
        return _numeric_value(records[0].get("reward"))
    return None


def _agent_role(record: dict[str, Any]) -> str | None:
    for metadata_key in ("train_metadata", "metadata"):
        role = _mapping(record.get(metadata_key)).get("agent_role")
        if isinstance(role, str):
            return role
    return None


def _session_index(record: dict[str, Any]) -> int:
    value = _mapping(record.get("metadata")).get("session_index")
    return value if isinstance(value, int) else -1


def _main_session_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    main_records = [record for record in records if _agent_role(record) == "main"]
    candidates = main_records or records
    return max(enumerate(candidates), key=lambda indexed_record: (_session_index(indexed_record[1]), indexed_record[0]))[1]


def collapse_trajectory_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_records: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for record_index, record in enumerate(records):
        grouped_records.setdefault(_record_trajectory_key(record, record_index), []).append(record)

    collapsed_records = []
    for trajectory_records in grouped_records.values():
        main_record = dict(_main_session_record(trajectory_records))
        metadata = _mapping(main_record.get("metadata"))
        train_metadata = _mapping(main_record.get("train_metadata"))
        original_question = next(
            (
                _mapping(record.get("metadata")).get("original_question")
                for record in trajectory_records
                if _mapping(record.get("metadata")).get("original_question")
            ),
            None,
        )
        reward = _trajectory_reward(trajectory_records)
        if reward is not None:
            main_record["reward"] = reward
        elif len(trajectory_records) > 1:
            main_record["reward"] = None
        if original_question:
            main_record["prompt"] = original_question
        if "extracted_answer" in main_record:
            main_record["extracted_answer"] = extract_answer_text(main_record.get("response", "") or "")
        main_record["metadata"] = metadata
        if train_metadata:
            main_record["train_metadata"] = train_metadata
        collapsed_records.append(main_record)
    return collapsed_records


def write_subagent_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in collapse_trajectory_records(records):
            output_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_records(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix == ".jsonl":
        return [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if input_path.suffix == ".pt":
        import torch

        loaded = torch.load(input_path, map_location="cpu", weights_only=False)
        samples = loaded if isinstance(loaded, list) else loaded.get("samples", [loaded]) if isinstance(loaded, Mapping) else None
        if not isinstance(samples, list):
            raise ValueError("Expected .pt input to contain a sample list or a mapping of samples")
        return [build_compact_eval_record(sample) for sample in samples]
    raise ValueError(f"Unsupported input format: {input_path.suffix!r}; expected .pt or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert subagent eval records to BrowseComp JSONL")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    write_subagent_jsonl(_load_records(args.input), args.output_jsonl)


if __name__ == "__main__":
    main()
