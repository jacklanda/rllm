"""Convert offline rejection-sampling trajectories into SFT parquet files.

Reads the ``selected_trajectories`` from a rejection-sampling ``trajectories.json`` and
writes ``train.parquet`` / ``val.parquet`` with a ``messages`` column (list of
``{"role", "content"}`` dicts) consumable by ``AgentSFTDataset`` / verl's multi-turn SFT
dataset.

The validation set is a stratified hold-out by ``data_source`` so each source
(mcp / search / search_agent) is represented in both splits.

Usage:
    python prepare_agent_sft_data.py \
        --input experiments/rejection_sampling/offline-rs-fused-20260527101144/trajectories.json \
        --output-dir ./data
"""

import argparse
import os
from collections import Counter, defaultdict

import pandas as pd

try:
    from agent_sft_data_utils import build_records, load_trajectories_json
except ImportError:  # pragma: no cover - supports package-style imports in tests
    from examples.sft_agent.agent_sft_data_utils import build_records, load_trajectories_json

DEFAULT_INPUT = "experiments/rejection_sampling/offline-rs-fused-20260527101144/trajectories.json"


def stratified_split(records, val_frac, seed):
    """Hold out ~val_frac of each data_source for validation."""
    import random

    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec["data_source"]].append(rec)

    train, val = [], []
    for source, recs in sorted(by_source.items()):
        recs = recs[:]
        rng.shuffle(recs)
        n_val = max(1, round(len(recs) * val_frac)) if len(recs) > 1 else 0
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def print_length_histogram(records, model_path):
    """Print a token-length histogram of the full rendered trajectories (best-effort)."""
    try:
        import numpy as np
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover - visibility only
        print(f"[hist] skipped (import failed: {e})")
        return

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lens = []
    for rec in records:
        full = "".join(m["content"] for m in rec["messages"])
        lens.append(len(tok.encode(full, add_special_tokens=False)))
    lens = np.array(lens)
    print("[hist] approx full-trajectory token lengths (content only):")
    for p in (50, 75, 90, 95, 99, 100):
        print(f"[hist]   p{p}: {int(np.percentile(lens, p))}")
    for thr in (8192, 16384, 24576, 32768):
        print(f"[hist]   frac <= {thr}: {(lens <= thr).mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to trajectories.json")
    parser.add_argument("--output-dir", default="./data", help="Output dir for parquet files")
    parser.add_argument("--val-frac", type=float, default=0.03, help="Validation fraction per source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.0,
        help="Drop trajectories below this reward (data is usually already filtered)",
    )
    parser.add_argument(
        "--model-path",
        default="/share/nlp/share/plm/Qwen3-4B-Thinking-2507",
        help="Tokenizer used only for the length histogram",
    )
    parser.add_argument("--no-histogram", action="store_true", help="Skip the token-length histogram")
    args = parser.parse_args()

    trajectories = load_trajectories_json(args.input)
    print(f"Loaded {len(trajectories)} trajectories from {args.input}")

    records, skipped = build_records(trajectories, args.reward_threshold)
    print(f"Kept {len(records)} usable trajectories; skipped: {dict(skipped)}")
    print(f"Source distribution: {dict(Counter(r['data_source'] for r in records))}")

    if not args.no_histogram:
        print_length_histogram(records, args.model_path)

    train, val = stratified_split(records, args.val_frac, args.seed)
    print(f"Split -> train={len(train)} val={len(val)}")
    print(f"  train sources: {dict(Counter(r['data_source'] for r in train))}")
    print(f"  val sources:   {dict(Counter(r['data_source'] for r in val))}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    pd.DataFrame(train).to_parquet(train_path, index=False)
    pd.DataFrame(val).to_parquet(val_path, index=False)
    print(f"Wrote {train_path} and {val_path}")


if __name__ == "__main__":
    main()
