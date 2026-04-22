"""JSON and JSONL helpers for MEIO experiment logging artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable


def jsonable(value: object) -> object:
    """Convert typed values into JSON-safe primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    return str(value)


def canonical_json(value: object) -> str:
    """Return a stable JSON string for hashing and persistence."""

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def hash_jsonable(value: object) -> str:
    """Return a stable SHA-256 hash for a JSON-safe object."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    """Return the current UTC timestamp in a stable log-friendly format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_output_dir(base_dir: str | Path, run_group_id: str) -> Path:
    """Create and return the run output directory."""

    output_dir = Path(base_dir) / run_group_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: str | Path, record: object) -> Path:
    """Write one JSON artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def write_jsonl(path: str | Path, records: Iterable[object]) -> Path:
    """Write a JSONL artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(jsonable(record), sort_keys=True) for record in records]
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    target.write_text(payload, encoding="utf-8")
    return target


def write_experiment_metadata_json(output_dir: str | Path, metadata: object) -> Path:
    return write_json(Path(output_dir) / "experiment_metadata.json", metadata)


def write_episode_summaries_jsonl(output_dir: str | Path, records: Iterable[object]) -> Path:
    return write_jsonl(Path(output_dir) / "episode_summaries.jsonl", records)


def write_step_traces_jsonl(output_dir: str | Path, records: Iterable[object]) -> Path:
    return write_jsonl(Path(output_dir) / "step_traces.jsonl", records)


def write_llm_call_traces_jsonl(output_dir: str | Path, records: Iterable[object]) -> Path:
    return write_jsonl(Path(output_dir) / "llm_call_traces.jsonl", records)


def write_tool_call_traces_jsonl(output_dir: str | Path, records: Iterable[object]) -> Path:
    return write_jsonl(Path(output_dir) / "tool_call_traces.jsonl", records)


def write_run_manifest_json(output_dir: str | Path, record: object) -> Path:
    return write_json(Path(output_dir) / "run_manifest.json", record)


__all__ = [
    "canonical_json",
    "ensure_output_dir",
    "hash_jsonable",
    "jsonable",
    "utc_timestamp",
    "write_episode_summaries_jsonl",
    "write_experiment_metadata_json",
    "write_json",
    "write_jsonl",
    "write_llm_call_traces_jsonl",
    "write_run_manifest_json",
    "write_step_traces_jsonl",
    "write_tool_call_traces_jsonl",
]
