"""Curriculum filter: quarantine CLI/SWE tasks that are effectively unsolvable.

Two layers:

1. **Runtime quarantine** (this module) — during training, group a step's
   rollouts by ``(docker_image, commit_hash)`` and watch their
   ``omnigril_exit_code``. When every rollout in a group returns the *same*
   non-zero exit code for ``consecutive_steps`` in a row, the task is added
   to an on-disk blocklist. All currently-in-flight rollouts of that task
   have their reward set to ``None`` so the trainer masks them from loss,
   and the blocklist is consulted on subsequent steps to skip (mask) any
   future rollouts of the same task. This is the cheap proxy described in
   the RL env stability guideline.

2. **Offline pre-filter** (not this module) — at dataset build time, invoke
   ``experiments/artifacts/cli_data_20260429/run_eval_in_container.py --mode
   before`` for each entry (zero agent edits). Drop entries whose rc != 0
   *and* whose log contains an infra-class signature. This avoids burning
   compute during training.

The runtime layer is gated by ``config.rllm.curriculum_filter.enable``.
"""
from __future__ import annotations

import json
import os
import threading
from collections import Counter, defaultdict
from typing import Any, Optional

_INFRA_ERROR_SIGS = (
    "__NUMPY_SETUP__",
    "BadStatusLine",
    "BuildBackendException",
    "Could not find a version that satisfies",
    "setuptools.build_meta",
    "docker: Error",
    "No such image",
    "Cannot connect to the Docker daemon",
    # Container-integrity failures that are not the agent's fault.
    "Segmentation fault",
    "SIGSEGV",
    "core dumped",
    "Fatal Python error",
    "double free or corruption",
    "Bus error",
    "symbol lookup error",
    "undefined symbol",
    "ImportError while loading conftest",
    "error while loading shared libraries",
    # test_patch vs source drift — eval_script itself can't apply.
    "error: patch failed",
    "error: cannot apply",
    "patch does not apply",
    "malformed patch",
    "fatal: corrupt patch",
)


def _task_key(extra_info: Any) -> Optional[str]:
    """Stable key for a task. Preferred: docker_image@commit_hash."""
    if not isinstance(extra_info, dict):
        return None
    img = extra_info.get("docker_image")
    sha = extra_info.get("commit_hash")
    if img and sha:
        return f"{img}@{sha}"
    if img:
        return str(img)
    tid = extra_info.get("task_id")
    return str(tid) if tid else None


def is_infra_failure(log_text: str) -> bool:
    """Heuristic: does this eval log look like a broken Docker env rather
    than a policy mistake? Used by the offline dry-run filter."""
    if not log_text:
        return False
    return any(sig in log_text for sig in _INFRA_ERROR_SIGS)


class CurriculumFilter:
    """Per-trainer-instance quarantine tracker.

    Thread-safe. Persists its blocklist to ``blocklist_path`` so that
    restarts don't re-visit dead tasks.
    """

    def __init__(
        self,
        *,
        enable: bool,
        consecutive_steps: int = 3,
        apply_to_sources: Optional[list[str]] = None,
        blocklist_path: Optional[str] = None,
        infra_min_rollouts: int = 2,
    ) -> None:
        self.enable = bool(enable)
        self.consecutive_steps = int(max(1, consecutive_steps))
        self.apply_to_sources = set(apply_to_sources or ["cli"])
        self.blocklist_path = blocklist_path
        # Minimum number of rollouts that must exhibit an infra-class signature
        # (segfault, patch-failed, missing image, ...) for same-step quarantine
        # to fire. Two is enough: a single flake shouldn't kill a task, but if
        # two independent rollouts hit the same structural failure, the task
        # is almost certainly unsolvable in the shipped container.
        self.infra_min_rollouts = int(max(1, infra_min_rollouts))
        self._lock = threading.Lock()
        # task_key -> #consecutive steps with identical nonzero exit_code
        self._streak: dict[str, int] = defaultdict(int)
        # task_key -> last seen exit code (int or None)
        self._last_code: dict[str, int] = {}
        # Loaded blocklist: task_key -> {"reason": str, "exit_code": int, "count": int}
        self._blocked: dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.blocklist_path or not os.path.exists(self.blocklist_path):
            return
        try:
            with open(self.blocklist_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._blocked.update(data)
        except Exception as exc:  # noqa: BLE001
            print(f"[curriculum_filter] failed to load blocklist: {exc}")

    def _persist(self) -> None:
        if not self.blocklist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.blocklist_path), exist_ok=True)
            tmp = self.blocklist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._blocked, f, indent=2, sort_keys=True)
            os.replace(tmp, self.blocklist_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[curriculum_filter] failed to persist blocklist: {exc}")

    # ------------------------------------------------------------------
    def is_blocked(self, extra_info: Any) -> bool:
        if not self.enable:
            return False
        key = _task_key(extra_info)
        return bool(key and key in self._blocked)

    def update_from_dump(self, traj_dump: list[dict], batch) -> dict[str, int]:
        """Update streak counters from this step's accepted trajectories.

        ``traj_dump`` is the list built in ``_transform_agent_trajectories``.
        ``batch`` carries ``non_tensor_batch['extra_info']`` for task keying.

        Returns a small metrics dict for logging.
        """
        metrics = {
            "curriculum_filter/newly_blocked": 0,
            "curriculum_filter/total_blocked": 0,
            "curriculum_filter/newly_blocked_infra": 0,
            "curriculum_filter/newly_blocked_unapplicable": 0,
        }
        if not self.enable:
            return metrics

        # Group by task_key. Each entry collects the fields we need for both
        # the legacy "identical-exit-code streak" trigger and the new
        # "same-step infra signature" fast path.
        #   code    : int omnigril_exit_code or None
        #   src     : data_source
        #   idx     : row index
        #   reward  : float or None (None when trainer-masked upstream)
        #   verr    : verifier_error string (e.g. "unapplicable_patch")
        #   log     : log_tail used for infra-signature matching
        groups: dict[str, list[dict]] = defaultdict(list)
        try:
            extras = batch.non_tensor_batch.get("extra_info") if batch is not None else None
        except Exception:
            extras = None
        if extras is None:
            return metrics

        for row in traj_dump:
            idx = row.get("_idx")
            if idx is None:
                continue  # filled in by caller; see trainer hook
            try:
                ei = extras[idx]
                if hasattr(ei, "item"):
                    ei = ei.item()
            except Exception:
                continue
            key = _task_key(ei)
            if not key:
                continue
            src = row.get("data_source", "unknown")
            if self.apply_to_sources and src not in self.apply_to_sources:
                continue
            verf = (row.get("debug") or {}).get("verification") or {}
            raw_code = verf.get("omnigril_exit_code")
            try:
                code = int(raw_code) if raw_code not in (None, "", "None") else None
            except Exception:
                code = None
            groups[key].append({
                "code": code,
                "src": src,
                "idx": idx,
                "reward": row.get("reward"),
                "verr": str(verf.get("verifier_error") or ""),
                "log": str(verf.get("log_tail") or verf.get("log") or ""),
            })

        # Decide which tasks to quarantine this step.
        newly_blocked = 0
        newly_blocked_infra = 0
        newly_blocked_unapplicable = 0
        with self._lock:
            for key, entries in groups.items():
                if key in self._blocked:
                    continue

                # --- Fast path: same-step structural failure. ---------------
                # If every rollout of this task earned reward=0 (or None/mask),
                # and a threshold of them exhibit an infra-class log signature
                # or an "unapplicable_patch" verifier error, we don't need to
                # wait consecutive_steps — the task is structurally broken.
                rewards = [e.get("reward") for e in entries]
                all_zero_reward = len(entries) >= self.infra_min_rollouts and all(
                    (r is None) or (isinstance(r, (int, float)) and r <= 0.0)
                    for r in rewards
                )
                if all_zero_reward:
                    infra_hits = sum(1 for e in entries if is_infra_failure(e.get("log", "")))
                    unapp_hits = sum(
                        1 for e in entries
                        if e.get("verr") == "unapplicable_patch"
                        or "unapplicable_patch" in e.get("verr", "")
                    )
                    if infra_hits >= self.infra_min_rollouts:
                        self._blocked[key] = {
                            "reason": "infra_signature",
                            "infra_hits": int(infra_hits),
                            "num_rollouts": len(entries),
                        }
                        newly_blocked += 1
                        newly_blocked_infra += 1
                        self._streak.pop(key, None)
                        self._last_code.pop(key, None)
                        continue
                    if unapp_hits >= self.infra_min_rollouts and unapp_hits == len(entries):
                        # Every rollout produced an unapplicable patch — the
                        # test_patch references a symbol that no longer exists
                        # or the source is pre-broken. Quarantine same-step.
                        self._blocked[key] = {
                            "reason": "unapplicable_patch_all_rollouts",
                            "num_rollouts": len(entries),
                        }
                        newly_blocked += 1
                        newly_blocked_unapplicable += 1
                        self._streak.pop(key, None)
                        self._last_code.pop(key, None)
                        continue

                # --- Legacy path: N consecutive identical non-zero exits. ---
                codes = [e["code"] for e in entries if e["code"] is not None]
                if not codes:
                    self._streak[key] = 0
                    continue
                c = Counter(codes)
                top_code, top_count = c.most_common(1)[0]
                all_identical_nonzero = (
                    top_count == len(entries) and top_code != 0 and len(entries) >= 2
                )
                if all_identical_nonzero and self._last_code.get(key) == top_code:
                    self._streak[key] += 1
                else:
                    self._streak[key] = 1 if all_identical_nonzero else 0
                self._last_code[key] = top_code if all_identical_nonzero else 0

                if self._streak[key] >= self.consecutive_steps:
                    self._blocked[key] = {
                        "reason": "all_rollouts_identical_exit",
                        "exit_code": int(top_code),
                        "consecutive_steps": int(self._streak[key]),
                    }
                    newly_blocked += 1

            if newly_blocked:
                self._persist()

            metrics["curriculum_filter/newly_blocked"] = newly_blocked
            metrics["curriculum_filter/newly_blocked_infra"] = newly_blocked_infra
            metrics["curriculum_filter/newly_blocked_unapplicable"] = newly_blocked_unapplicable
            metrics["curriculum_filter/total_blocked"] = len(self._blocked)
        return metrics

    # ------------------------------------------------------------------
    def mask_rewards_in_place(self, traj_dump: list[dict], batch) -> int:
        """Set reward to None on any trajectory whose task is blocked.

        The trainer's loss-mask path treats reward=None as 'mask this
        trajectory from the policy gradient', matching env-error handling.
        Returns the number of trajectories masked.
        """
        if not self.enable or not self._blocked:
            return 0
        try:
            extras = batch.non_tensor_batch.get("extra_info") if batch is not None else None
        except Exception:
            extras = None
        if extras is None:
            return 0
        masked = 0
        for row in traj_dump:
            idx = row.get("_idx")
            if idx is None:
                continue
            try:
                ei = extras[idx]
                if hasattr(ei, "item"):
                    ei = ei.item()
            except Exception:
                continue
            key = _task_key(ei)
            if key and key in self._blocked:
                row["_curriculum_blocked"] = True
                # Only mask if the sample is from a watched source.
                if row.get("data_source", "unknown") in self.apply_to_sources:
                    row["reward"] = None
                    masked += 1
        return masked
