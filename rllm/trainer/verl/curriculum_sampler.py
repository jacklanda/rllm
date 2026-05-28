"""Curriculum sampler: easy-to-hard difficulty-aware batch sampling for CLI agent RL training.

Replaces the default shuffle-based DataLoader sampling with a step-aware
weighted sampler that progressively introduces harder tasks as training
progresses. This ensures the model receives positive reward signal early
(from easy tasks) before facing harder challenges.

Schedule (configurable):
  Phase 1 [1, warmup_steps]:           easy only
  Phase 2 (warmup_steps, transition]:  easy + medium (linear ramp)
  Phase 3 (transition, full_mix]:      easy + medium + hard (linear ramp)
  Phase 4 (full_mix, ...):             uniform over all difficulties
"""

from __future__ import annotations

import json
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler


_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


class CurriculumSampler(Sampler[int]):
    """Step-aware weighted sampler with easy-to-hard curriculum.

    Compatible with ``torch.utils.data.DataLoader(sampler=...)``.
    Samples WITH replacement so that minority classes (easy=25%) can
    fill entire batches during early phases.
    """

    def __init__(
        self,
        difficulties: list[str],
        rank_scores: list[int],
        batch_size: int,
        *,
        warmup_steps: int = 20,
        transition_steps: int = 50,
        full_mix_steps: int = 100,
        num_batches_per_epoch: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            difficulties: per-sample difficulty label ("easy"/"medium"/"hard")
            rank_scores: per-sample rank_score for intra-difficulty ordering
            batch_size: number of samples per batch
            warmup_steps: training steps using only easy tasks
            transition_steps: step at which medium tasks reach full weight
            full_mix_steps: step at which hard tasks reach full weight
            num_batches_per_epoch: how many batches constitute one epoch
            seed: random seed for reproducibility
        """
        super().__init__()
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.transition_steps = transition_steps
        self.full_mix_steps = full_mix_steps
        self.seed = seed
        self._step = 1
        self._rng = np.random.default_rng(seed)

        n = len(difficulties)
        self._n = n
        self._num_batches = num_batches_per_epoch or max(1, n // batch_size)

        self._difficulty_ids = np.array([_DIFFICULTY_ORDER.get(d, 1) for d in difficulties], dtype=np.int32)
        self._rank_scores = np.array(rank_scores, dtype=np.float64)

        rs_min = self._rank_scores.min()
        rs_max = self._rank_scores.max()
        if rs_max > rs_min:
            self._rank_scores_norm = (self._rank_scores - rs_min) / (rs_max - rs_min)
        else:
            self._rank_scores_norm = np.zeros(n, dtype=np.float64)

        self._easy_mask = self._difficulty_ids == 0
        self._medium_mask = self._difficulty_ids == 1
        self._hard_mask = self._difficulty_ids == 2

        self._n_easy = int(self._easy_mask.sum())
        self._n_medium = int(self._medium_mask.sum())
        self._n_hard = int(self._hard_mask.sum())

    @property
    def current_phase(self) -> str:
        if self._step <= self.warmup_steps:
            return "easy_only"
        elif self._step <= self.transition_steps:
            return "easy_medium"
        elif self._step <= self.full_mix_steps:
            return "easy_medium_hard"
        else:
            return "full_mix"

    def set_step(self, step: int) -> None:
        self._step = max(1, step)

    def get_current_weights(self) -> dict[str, float]:
        w_easy, w_medium, w_hard = self._compute_tier_weights(self._step)
        return {"easy": w_easy, "medium": w_medium, "hard": w_hard}

    def _compute_tier_weights(self, step: int) -> tuple[float, float, float]:
        if step <= self.warmup_steps:
            return (1.0, 0.0, 0.0)

        if step <= self.transition_steps:
            t = (step - self.warmup_steps) / max(1, self.transition_steps - self.warmup_steps)
            w_medium = t * 0.5
            w_easy = 1.0 - w_medium
            return (w_easy, w_medium, 0.0)

        if step <= self.full_mix_steps:
            t = (step - self.transition_steps) / max(1, self.full_mix_steps - self.transition_steps)
            w_hard = t * (1.0 / 3.0)
            remaining = 1.0 - w_hard
            w_easy = remaining * 0.5
            w_medium = remaining * 0.5
            return (w_easy, w_medium, w_hard)

        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    def _compute_sample_weights(self) -> np.ndarray:
        w_easy, w_medium, w_hard = self._compute_tier_weights(self._step)
        weights = np.zeros(self._n, dtype=np.float64)

        if self._n_easy > 0 and w_easy > 0:
            per_sample_easy = w_easy / self._n_easy
            easy_bonus = 1.0 - self._rank_scores_norm[self._easy_mask]
            easy_bonus = easy_bonus / (easy_bonus.sum() + 1e-8) * self._n_easy
            weights[self._easy_mask] = per_sample_easy * easy_bonus

        if self._n_medium > 0 and w_medium > 0:
            per_sample_medium = w_medium / self._n_medium
            medium_bonus = 1.0 - self._rank_scores_norm[self._medium_mask]
            medium_bonus = medium_bonus / (medium_bonus.sum() + 1e-8) * self._n_medium
            weights[self._medium_mask] = per_sample_medium * medium_bonus

        if self._n_hard > 0 and w_hard > 0:
            per_sample_hard = w_hard / self._n_hard
            hard_bonus = 1.0 - self._rank_scores_norm[self._hard_mask]
            hard_bonus = hard_bonus / (hard_bonus.sum() + 1e-8) * self._n_hard
            weights[self._hard_mask] = per_sample_hard * hard_bonus

        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(self._n, dtype=np.float64) / self._n

        return weights

    def __iter__(self) -> Iterator[int]:
        for _ in range(self._num_batches):
            weights = self._compute_sample_weights()
            indices = self._rng.choice(self._n, size=self.batch_size, replace=True, p=weights)
            yield from indices.tolist()

    def __len__(self) -> int:
        return self._num_batches * self.batch_size


def extract_difficulty_info(
    dataset,
    difficulty_key: str = "difficulty",
    rank_score_key: str = "rank_score",
) -> tuple[list[str], list[int]]:
    """Extract difficulty and rank_score from a verl RLHFDataset's dataframe.

    Supports both HuggingFace datasets.Dataset and pandas DataFrame as the
    underlying dataframe type. Returns (difficulties, rank_scores) lists
    aligned with dataset indices.
    """
    difficulties = []
    rank_scores = []

    dataframe = dataset.dataframe
    n = len(dataframe)

    # Batch-access the extra_info column if possible (HuggingFace Dataset)
    if hasattr(dataframe, "column_names") and "extra_info" in dataframe.column_names:
        extra_infos = dataframe["extra_info"]
    elif hasattr(dataframe, "columns") and "extra_info" in dataframe.columns:
        extra_infos = dataframe["extra_info"].tolist()
    else:
        extra_infos = [None] * n

    for ei in extra_infos:
        if isinstance(ei, str):
            try:
                ei = json.loads(ei)
            except (ValueError, TypeError):
                ei = {}
        if not isinstance(ei, dict):
            ei = {}

        diff = ei.get(difficulty_key, "medium")
        if diff not in _DIFFICULTY_ORDER:
            diff = "medium"
        difficulties.append(diff)

        rs = ei.get(rank_score_key, 5000)
        try:
            rs = int(rs)
        except (ValueError, TypeError):
            rs = 5000
        rank_scores.append(rs)

    return difficulties, rank_scores
