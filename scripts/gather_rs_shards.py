#!/usr/bin/env python3
"""Aggregate per-step rejection-sampling shards into one unified JSON file.

Each shard (e.g. ``global_steps_<N>.json``) is produced by
``_dump_offline_rs_batch_results`` and contains:
    - summary counters (num_questions, num_trials, source_trials, ...)
    - a ``selected_trajectories`` list of rollouts that already passed the
      reward threshold for that step.

This script walks a shard directory, sorts the shards by their step number,
concatenates ``selected_trajectories`` (annotating each with the source
shard / step), and re-aggregates the scalar counters so the output can be
fed straight into a downstream rejection-sampling / SFT pipeline.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys


SHARD_PATTERN = re.compile(r"global_steps_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing global_steps_*.json shards " "(e.g. .../offline-rs-*/batch_results).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to the unified output JSON file.",
    )
    p.add_argument(
        "--pattern",
        default="global_steps_*.json",
        help="Glob pattern for shards inside --input-dir.",
    )
    p.add_argument(
        "--dedup",
        choices=["none", "uuid", "uuid_rank"],
        default="none",
        help="Optional dedup of selected trajectories. " "'uuid' keeps the highest-reward trajectory per uuid; " "'uuid_rank' keeps the first occurrence of each (uuid, accepted_rank).",
    )
    p.add_argument(
        "--no-shard-tag",
        action="store_true",
        help="Do not annotate trajectories with their source shard / step.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indent for the output JSON (default: compact).",
    )
    return p.parse_args()


def shard_step(path: str) -> int:
    m = SHARD_PATTERN.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def reward_value(traj: dict) -> float:
    r = traj.get("reward")
    if isinstance(r, (int, float)):
        return float(r)
    return float("-inf")


def load_shard(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _add_counter(dst: collections.Counter, src) -> None:
    if isinstance(src, dict):
        dst.update(src)


def main() -> int:
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[gather-rs] input dir not found: {args.input_dir}", file=sys.stderr)
        return 2

    shard_paths = sorted(
        glob.glob(os.path.join(args.input_dir, args.pattern)),
        key=shard_step,
    )
    if not shard_paths:
        print(f"[gather-rs] no shards matched in {args.input_dir}", file=sys.stderr)
        return 2

    merged_trajs: list = []
    shard_summaries: list = []

    total_questions = 0
    total_trials = 0
    total_usable_questions = 0
    total_selected = 0
    source_trials: collections.Counter = collections.Counter()
    source_selected: collections.Counter = collections.Counter()
    rewards_min: list = []
    rewards_max: list = []

    sample_n = reward_threshold = max_per_problem = min_sample_trial = None
    skipped_shards: list = []

    for path in shard_paths:
        try:
            shard = load_shard(path)
        except Exception as e:
            print(f"[gather-rs] skip {path}: {e}", file=sys.stderr)
            skipped_shards.append(os.path.basename(path))
            continue

        step = shard_step(path)
        stem = shard.get("batch_file") or os.path.splitext(os.path.basename(path))[0]

        if sample_n is None:
            sample_n = shard.get("sample_n")
            reward_threshold = shard.get("reward_threshold")
            max_per_problem = shard.get("max_trajectory_per_problem")
            min_sample_trial = shard.get("min_sample_trial")

        total_questions += int(shard.get("num_questions", 0) or 0)
        total_trials += int(shard.get("num_trials", 0) or 0)
        total_usable_questions += int(shard.get("num_usable_questions", 0) or 0)
        total_selected += int(shard.get("num_selected_trajectories", 0) or 0)
        _add_counter(source_trials, shard.get("source_trials"))
        _add_counter(source_selected, shard.get("source_selected"))

        mn = shard.get("min_selected_reward")
        mx = shard.get("max_selected_reward")
        if isinstance(mn, (int, float)):
            rewards_min.append(float(mn))
        if isinstance(mx, (int, float)):
            rewards_max.append(float(mx))

        trajs = shard.get("selected_trajectories") or []
        if not args.no_shard_tag:
            for t in trajs:
                t.setdefault("source_shard", stem)
                t.setdefault("source_step", step if step >= 0 else None)
        merged_trajs.extend(trajs)

        shard_summaries.append(
            {
                "shard": stem,
                "step": step if step >= 0 else None,
                "num_questions": shard.get("num_questions"),
                "num_trials": shard.get("num_trials"),
                "num_selected_trajectories": shard.get("num_selected_trajectories"),
                "min_selected_reward": mn,
                "max_selected_reward": mx,
            }
        )

    if args.dedup == "uuid":
        best: dict = {}
        for t in merged_trajs:
            uid = t.get("uuid") or t.get("prompt")
            if uid is None:
                continue
            if uid not in best or reward_value(t) > reward_value(best[uid]):
                best[uid] = t
        merged_trajs = list(best.values())
    elif args.dedup == "uuid_rank":
        seen: set = set()
        unique: list = []
        for t in merged_trajs:
            key = (t.get("uuid") or t.get("prompt"), t.get("accepted_rank"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        merged_trajs = unique

    unified = {
        "input_dir": os.path.abspath(args.input_dir),
        "num_shards": len(shard_summaries),
        "skipped_shards": skipped_shards,
        "sample_n": sample_n,
        "reward_threshold": reward_threshold,
        "max_trajectory_per_problem": max_per_problem,
        "min_sample_trial": min_sample_trial,
        "dedup": args.dedup,
        "totals": {
            "num_questions": total_questions,
            "num_trials": total_trials,
            "num_usable_questions": total_usable_questions,
            "num_selected_trajectories_raw": total_selected,
            "num_selected_trajectories_unified": len(merged_trajs),
            "min_selected_reward": min(rewards_min) if rewards_min else None,
            "max_selected_reward": max(rewards_max) if rewards_max else None,
            "source_trials": dict(source_trials),
            "source_selected": dict(source_selected),
        },
        "shards": shard_summaries,
        "selected_trajectories": merged_trajs,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{args.output}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(unified, f, ensure_ascii=False, indent=args.indent)
    os.replace(tmp_path, args.output)

    print("[gather-rs] " f"shards={len(shard_summaries)} " f"trajs_raw={total_selected} trajs_unified={len(merged_trajs)} " f"questions={total_questions} trials={total_trials} " f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
