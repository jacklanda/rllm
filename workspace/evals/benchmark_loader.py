"""Loaders for agentic benchmarks under experiments/artifacts/benchmarks/.

Each benchmark directory has its own layout (data.json, sub-folders with
parquet, csv, or heterogeneous JSON). This module normalizes all of them into
a single :class:`AgenticTask` shape compatible with
:func:`rllm.rewards.reward_fn.search_reward_fn` and
:class:`rllm.environments.tools.tool_env.ToolEnvironment`.

The public entry point is :func:`load_benchmark`. Unknown benchmarks raise
``KeyError`` so missing coverage is loud rather than silent.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ARTIFACTS_BENCHMARKS_DIR = Path(
    os.environ.get(
        "AGENTIC_ARTIFACTS_DIR",
        "/share/nlp/liuyang/workspace/gem/rllm/experiments/artifacts/benchmarks",
    )
)


_QA_HINTS = (
    "Please answer the above question",
    "Answer the above question",
    "When ready, output the final answer",
    "When ready, please output",
    "Output the final answer enclosed in",
    "Put your final answer",
)


def _strip_qa_suffix(q: str) -> str:
    """Remove common instruction suffixes appended to benchmark questions."""
    if not isinstance(q, str):
        return str(q)
    out = q
    for hint in _QA_HINTS:
        idx = out.find(hint)
        if idx > 0:
            out = out[:idx]
    return out.strip()


def _as_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v) != ""]
    if isinstance(value, (tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


@dataclass
class AgenticTask:
    """Normalized representation of one benchmark example."""

    task_id: str
    question: str
    ground_truth: list[str]
    data_source: str
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_env_task(self) -> dict[str, Any]:
        """Build the task dict consumed by ToolEnvironment + search_reward_fn.

        rllm expects ``ground_truth`` (str or list) and ``data_source``.
        """
        gt: Any = self.ground_truth
        if len(gt) == 1:
            gt = gt[0]
        return {
            "question": self.question,
            "ground_truth": gt,
            "data_source": self.data_source,
            "extra_info": self.extra_info,
        }


# -------- Per-benchmark loaders -----------------------------------------------


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_parquet(path: Path) -> list[dict]:
    import pandas as pd  # local to avoid hard dep when unused

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def _load_data_json(
    bench_dir: Path,
    data_source: str,
    question_keys: tuple[str, ...] = ("question", "query", "input", "problem"),
    answer_keys: tuple[str, ...] = (
        "answer",
        "gt_answer",
        "ground_truth_answer",
        "target",
    ),
    strip_suffix: bool = True,
) -> list[AgenticTask]:
    for rel in ("data/data.json", "data.json"):
        p = bench_dir / rel
        if p.is_file():
            samples = _load_json(p)
            break
    else:
        samples = _load_records_from_prior_output(data_source)
        if samples is None:
            raise FileNotFoundError(f"No data.json under {bench_dir}")

    tasks: list[AgenticTask] = []
    for i, s in enumerate(samples):
        q = None
        for k in question_keys:
            if s.get(k):
                q = s[k]
                break
        if q is None:
            # Look into extra_info for nested question
            ei = s.get("extra_info") or {}
            for k in question_keys:
                if isinstance(ei, dict) and ei.get(k):
                    q = ei[k]
                    break
        if q is None:
            continue

        ans = None
        for k in answer_keys:
            if s.get(k) is not None:
                ans = s[k]
                break
        if ans is None:
            ei = s.get("extra_info") or {}
            for k in answer_keys:
                if isinstance(ei, dict) and ei.get(k) is not None:
                    ans = ei[k]
                    break

        task_id = str(s.get("task_id") or s.get("id") or f"{data_source}-{i}")
        tasks.append(
            AgenticTask(
                task_id=task_id,
                question=_strip_qa_suffix(q) if strip_suffix else str(q),
                ground_truth=_as_answer_list(ans),
                data_source=data_source,
                extra_info={
                    k: v
                    for k, v in s.items()
                    if k not in question_keys and k not in answer_keys
                },
            )
        )
    return tasks


def _load_records_from_prior_output(data_source: str) -> list[dict] | None:
    """Recover benchmark inputs from prior eval outputs when artifacts are absent."""

    evals_dir = Path(__file__).resolve().parent
    for path in sorted((evals_dir / "output").glob(f"*/{data_source}/results.json")):
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list) or not records:
            continue
        samples: list[dict] = []
        for i, record in enumerate(records):
            if not isinstance(record, dict) or not record.get("question"):
                continue
            samples.append(
                {
                    "task_id": record.get("task_id") or f"{data_source}-{i}",
                    "question": record.get("question"),
                    "answer": record.get("ground_truth"),
                    "extra_info": record.get("extra_info", {}),
                }
            )
        if samples:
            return samples
    return None


def load_bamboogle(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="bamboogle")


def load_hotpotqa(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="hotpotqa")


def load_2wiki(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir, data_source="2wiki", question_keys=("query", "question", "input")
    )


def load_musique(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir, data_source="musique", question_keys=("query", "question", "input")
    )


def load_gaia(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="gaia")


def load_medqa(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(bench_dir, data_source="medqa")


def load_browse_comp(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir,
        data_source="browsecomp",
        question_keys=("input", "question", "query"),
        answer_keys=("ground_truth_answer", "answer", "gt_answer"),
        strip_suffix=False,
    )
def load_browsecomp_plus(bench_dir: Path) -> list[AgenticTask]:
    return _load_data_json(
        bench_dir,
        data_source="browsecomp_plus"
    )

#def load_browsecomp_plus(bench_dir: Path) -> list[AgenticTask]:
#    return _load_data_json(
#        bench_dir,
#        data_source="browsecomp_plus",
#        question_keys=("input", "question", "query"),
#        answer_keys=("ground_truth_answer", "answer", "gt_answer"),
#        strip_suffix=False,
#    )


def load_simpleqa_verified(bench_dir: Path) -> list[AgenticTask]:
    # CSV in the folder root
    for fname in ("simpleqa_verified.csv", "data.csv"):
        p = bench_dir / fname
        if p.is_file():
            rows = _load_csv(p)
            break
    else:
        # fall back to data.json if present
        return _load_data_json(
            bench_dir, data_source="simpleqa_verified", strip_suffix=False
        )

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or r.get("input")
        ans = r.get("answer") or r.get("ground_truth_answer")
        if not q:
            continue
        tasks.append(
            AgenticTask(
                task_id=str(r.get("task_id") or f"simpleqa_verified-{i}"),
                question=_strip_qa_suffix(q),
                ground_truth=_as_answer_list(ans),
                data_source="simpleqa_verified",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k not in ("problem", "question", "input", "answer")
                },
            )
        )
    return tasks


def _xbench_xor_decrypt(b64_text: str, key: str) -> str:
    """Decode xbench-style (base64 + per-row XOR) ciphertext to UTF-8.

    The xbench ScienceQA shard (``ScienceQA.csv``) stores ``prompt`` and
    ``answer`` as base64(XOR(plaintext, canary)) to keep benchmark data out
    of scraped web indices. Without decryption the model is given pure
    ciphertext, which crashes pass@1 to ~0. Mirrors
    ``experiments/artifacts/benchmarks/scienceqa/xbench_evals.py``.
    """
    if not b64_text or not key:
        return b64_text
    raw = base64.b64decode(b64_text)
    kb = key.encode("utf-8")
    kl = len(kb) or 1
    return bytes(raw[i] ^ kb[i % kl] for i in range(len(raw))).decode("utf-8")


def load_scienceqa(bench_dir: Path) -> list[AgenticTask]:
    for fname in ("ScienceQA.csv", "scienceqa.csv", "data.csv"):
        p = bench_dir / fname
        if p.is_file():
            rows = _load_csv(p)
            break
    else:
        return _load_data_json(bench_dir, data_source="scienceqa")

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        canary = r.get("canary") or ""
        q_raw = (
            r.get("prompt") or r.get("problem") or r.get("question") or r.get("input")
        )
        ans_raw = r.get("answer") or r.get("ground_truth_answer")
        if not q_raw:
            continue
        # xbench ships ScienceQA with base64+XOR ciphertext keyed on
        # ``canary``. Decode both sides; plain-text rows (no canary or
        # decode failure) fall through verbatim.
        q = q_raw
        ans = ans_raw
        if canary:
            try:
                q = _xbench_xor_decrypt(q_raw, canary)
            except Exception:
                q = q_raw
            if ans_raw:
                try:
                    ans = _xbench_xor_decrypt(ans_raw, canary)
                except Exception:
                    ans = ans_raw
        tasks.append(
            AgenticTask(
                task_id=str(r.get("id") or r.get("task_id") or f"scienceqa-{i}"),
                question=_strip_qa_suffix(q),
                ground_truth=_as_answer_list(ans),
                data_source="scienceqa",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k
                    not in (
                        "problem",
                        "question",
                        "prompt",
                        "input",
                        "answer",
                        "canary",
                    )
                },
            )
        )
    return tasks


def load_hle(bench_dir: Path) -> list[AgenticTask]:
    # Real content lives in the parquet shard, not data.json.
    for rel in (
        "data/test-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
        "data.parquet",
    ):
        p = bench_dir / rel
        if p.is_file():
            rows = _load_parquet(p)
            break
    else:
        return _load_data_json(bench_dir, data_source="hle", strip_suffix=False)

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("problem") or r.get("input")
        ans = r.get("answer") or r.get("ground_truth_answer")
        if not q:
            continue
        tasks.append(
            AgenticTask(
                task_id=str(r.get("id") or r.get("task_id") or f"hle-{i}"),
                question=str(q),
                ground_truth=_as_answer_list(ans),
                data_source="hle",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k in ("category", "answer_type", "subject")
                },
            )
        )
    return tasks


def load_deepsearchqa(bench_dir: Path) -> list[AgenticTask]:
    for fname in ("DSQA-full.csv", "data.csv", "deepsearchqa.csv"):
        p = bench_dir / fname
        if p.is_file():
            rows = _load_csv(p)
            break
    else:
        return _load_data_json(bench_dir, data_source="deepsearchqa")

    tasks: list[AgenticTask] = []
    for i, r in enumerate(rows):
        q = r.get("problem") or r.get("question") or r.get("query") or r.get("input")
        ans = r.get("answer") or r.get("ground_truth_answer")
        if not q:
            continue
        tasks.append(
            AgenticTask(
                task_id=str(r.get("task_id") or f"deepsearchqa-{i}"),
                question=_strip_qa_suffix(q),
                ground_truth=_as_answer_list(ans),
                data_source="deepsearchqa",
                extra_info={
                    k: v
                    for k, v in r.items()
                    if k not in ("problem", "question", "query", "input", "answer")
                },
            )
        )
    return tasks


def load_gpqa_diamond(bench_dir: Path) -> list[AgenticTask]:
    # data.json under experiments/artifacts/benchmarks/gpqa_diamond/ exposes
    # `input` (multiple-choice prompt) and `ground_truth_answer` / `target`
    # (one of A/B/C/D). Strip QA suffixes is unnecessary here; the prompt is
    # already a clean MCQ block.
    return _load_data_json(
        bench_dir,
        data_source="gpqa_diamond",
        question_keys=("input", "question", "problem"),
        answer_keys=("ground_truth_answer", "target", "gt_answer", "answer"),
        strip_suffix=False,
    )


BENCHMARK_REGISTRY: dict[str, Callable[[Path], list[AgenticTask]]] = {
    "bamboogle": load_bamboogle,
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
    "gaia": load_gaia,
    "medqa": load_medqa,
    #"browse_comp": load_browse_comp,
    "browsecomp_plus": load_browsecomp_plus,
    "simpleqa_verified": load_simpleqa_verified,
    "scienceqa": load_scienceqa,
    #"hle": load_hle,
    #"deepsearchqa": load_deepsearchqa,
    "gpqa_diamond": load_gpqa_diamond,
}


def load_benchmark(
    name: str,
    artifacts_dir: Path | str | None = None,
    max_problems: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 0,
) -> list[AgenticTask]:
    """Load a benchmark by name and return a list of normalized tasks.

    When ``shuffle`` is true (the default), tasks are deterministically
    permuted with ``random.Random(shuffle_seed)`` *before* ``max_problems``
    truncation, so capped runs sample across the whole benchmark rather
    than taking a head slice. Pass ``shuffle=False`` to preserve original
    file order.

    Raises :class:`KeyError` if ``name`` is not registered.
    """
    root = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_BENCHMARKS_DIR
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown agentic benchmark '{name}'. Registered: {sorted(BENCHMARK_REGISTRY)}"
        )
    bench_dir = root / name
    if not bench_dir.is_dir():
        recovered = _load_records_from_prior_output(name)
        if recovered is None:
            raise FileNotFoundError(f"Benchmark directory not found: {bench_dir}")
        tasks = [
            AgenticTask(
                task_id=str(s.get("task_id") or f"{name}-{i}"),
                question=_strip_qa_suffix(str(s.get("question") or "")),
                ground_truth=_as_answer_list(s.get("answer")),
                data_source=name,
                extra_info=s.get("extra_info") if isinstance(s.get("extra_info"), dict) else {},
            )
            for i, s in enumerate(recovered)
            if s.get("question")
        ]
    else:
        tasks = BENCHMARK_REGISTRY[name](bench_dir)
    if shuffle:
        random.Random(shuffle_seed).shuffle(tasks)
    if max_problems is not None:
        tasks = tasks[:max_problems]
    return tasks
