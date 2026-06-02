"""Utilities for loading agent SFT data from parquet or trajectory JSON files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def clean_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Strip empty messages and keep only role/content fields."""
    if not isinstance(raw_messages, list):
        return []

    cleaned = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if not role or not isinstance(content, str):
            continue
        if not content.strip():
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def extract_trajectories(payload: Any) -> list[dict[str, Any]]:
    """Extract a trajectory list from common offline-RS JSON payload shapes."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object or list, got {type(payload).__name__}")

    for key in ("selected_trajectories", "trajectories", "accept_traj"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    raise ValueError(
        "Could not find trajectories in JSON payload. Expected one of "
        "'selected_trajectories', 'trajectories', or 'accept_traj'."
    )


def load_trajectories_json(path: str | Path) -> list[dict[str, Any]]:
    """Load trajectories from a JSON file or newline-delimited JSON file."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        trajectories = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))
        return trajectories

    with path.open() as f:
        return extract_trajectories(json.load(f))


def build_records(
    trajectories: list[dict[str, Any]],
    reward_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], Counter]:
    """Turn raw trajectory dicts into SFT records, filtering unusable ones."""
    records = []
    skipped = Counter()
    for traj in trajectories:
        if not isinstance(traj, dict):
            skipped["not_a_dict"] += 1
            continue
        if traj.get("reward", 0.0) < reward_threshold:
            skipped["below_reward_threshold"] += 1
            continue

        raw_messages = traj.get("messages")
        if raw_messages is None:
            raw_messages = traj.get("trajectory", [])
        messages = clean_messages(raw_messages)
        if not any(m["role"] == "assistant" for m in messages):
            skipped["no_assistant_message"] += 1
            continue
        if len(messages) < 2:
            skipped["too_few_messages"] += 1
            continue

        records.append(
            {
                "messages": messages,
                "data_source": traj.get("data_source", "unknown"),
                "reward": float(traj.get("reward", 0.0)),
                "uuid": traj.get("uuid") or traj.get("uid", ""),
            }
        )
    return records, skipped


def records_from_json(path: str | Path, reward_threshold: float = 0.0) -> tuple[list[dict[str, Any]], Counter]:
    """Load a trajectory JSON path and convert it into SFT records."""
    return build_records(load_trajectories_json(path), reward_threshold=reward_threshold)
