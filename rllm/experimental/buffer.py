"""TrajectoryGroupBuffer for async training.

Accumulates episodes, processes into ready-to-train trajectory groups,
with optional NVMe offloading for memory management.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tqdm import tqdm

from rllm.experimental.common import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.experimental.common.transform import transform_episodes_to_trajectory_groups
from rllm.experimental.metrics import MetricsAggregator
from rllm.experimental.sync_coordinator import SyncCoordinator
from rllm.types import Episode, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason, infer_task_source

logger = logging.getLogger(__name__)


@dataclass
class TaskBatch:
    """All trajectory groups produced from one task's episodes, plus stripped episodes for UI logging."""

    groups: list[TrajectoryGroup]
    episodes: list[Episode] = field(default_factory=list)


class TrajectoryGroupBuffer:
    """Accumulates episodes, processes into trajectory groups, yields to training.

    When all rollouts for a task arrive:
    1. Record episode-level metrics to aggregator (before any filtering)
    2. Transform episodes -> trajectory groups
    3. Compact filtering + drop groups with < min_trajs_per_group
    4. Compute advantages
    5. If rejection sampling enabled: drop groups with all-zero advantage
    6. Queue the task batch for training

    Filtered groups are reported directly to the coordinator (which tracks
    throttle slots and filter counts). Only non-empty task batches are queued.
    All metrics flow through the shared MetricsAggregator.

    Optionally offloads pending episodes and/or queued task batches to
    disk to reduce memory pressure (disabled by default).
    """

    def __init__(
        self,
        group_size: int,
        coordinator: SyncCoordinator,
        aggregator: MetricsAggregator,
        algorithm_config: AlgorithmConfig,
        transform_config: TransformConfig,
        cf_config: CompactFilteringConfig,
        rs_config: RejectionSamplingConfig,
        episode_offload_dir: str | None = None,
        trajectory_group_offload_dir: str | None = None,
        pbar: tqdm | None = None,
    ):
        self._group_size = group_size
        self._coordinator = coordinator
        self._aggregator = aggregator
        self._algorithm_config = algorithm_config
        self._transform_config = transform_config
        self._cf_config = cf_config
        self._rs_config = rs_config
        self._pbar = pbar

        # Episode offloading: pending episodes serialized to disk
        self._episode_offload_dir = episode_offload_dir
        if episode_offload_dir:
            os.makedirs(episode_offload_dir, exist_ok=True)
        self._pending: dict[str, list[Episode | str]] = {}  # str = offloaded file path

        # Trajectory group offloading: queued task batches serialized to disk
        self._tg_offload_dir = trajectory_group_offload_dir
        if trajectory_group_offload_dir:
            os.makedirs(trajectory_group_offload_dir, exist_ok=True)
        self._queue: asyncio.Queue[TaskBatch | str | None] = asyncio.Queue()
        self._training_queue_size = 0
        self._filtered_count = 0
        self._consumed_count = 0
        self._training_step = 0
        self._last_reason = "init"
        self._accepted_by_source: Counter[str] = Counter()
        self._filtered_by_source: Counter[str] = Counter()
        self._queue_update_event = asyncio.Event()
        self._generation_complete = False

    def set_training_step(self, step: int) -> None:
        self._training_step = step
        self._refresh_pbar_counters()

    def _refresh_pbar_counters(self) -> None:
        if self._pbar is not None:
            self._pbar.set_postfix(
                queued=self._training_queue_size,
                filtered=self._filtered_count,
                consumed=self._consumed_count,
                refresh=False,
            )

    def _record_classified_prompt_group(self) -> None:
        self._refresh_pbar_counters()
        if self._pbar is not None:
            self._pbar.update(1)

    async def _offload_episode(self, task_id: str, episode: Episode) -> str:
        """Serialize episode to disk, return file path."""
        idx = len(self._pending.get(task_id, []))
        path = os.path.join(self._episode_offload_dir, f"{task_id}_{idx}.pkl")
        await asyncio.to_thread(self._pickle_dump, path, episode)
        return path

    async def _load_pending_episodes(self, task_id: str) -> list[Episode]:
        """Load all pending episodes for a task, deserializing offloaded ones."""
        episodes = []
        for item in self._pending.pop(task_id, []):
            if isinstance(item, str):
                ep = await asyncio.to_thread(self._pickle_load, item)
                episodes.append(ep)
            else:
                episodes.append(item)
        return episodes

    async def _offload_task_batch(self, batch: TaskBatch) -> str:
        """Serialize task batch to disk, return file path."""
        fd, path = tempfile.mkstemp(dir=self._tg_offload_dir, suffix=".pkl")
        os.close(fd)
        await asyncio.to_thread(self._pickle_dump, path, batch)
        return path

    async def _load_task_batch(self, item: TaskBatch | str) -> TaskBatch:
        """Load task batch, deserializing if offloaded."""
        if isinstance(item, str):
            return await asyncio.to_thread(self._pickle_load, item)
        return item

    @staticmethod
    def _pickle_dump(path: str, obj) -> None:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _pickle_load(path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        os.remove(path)
        return obj

    async def add_episode(self, task_id: str, episode: Episode) -> bool:
        """Add episode. When group completes, process and queue task batch."""
        if self._generation_complete:
            logger.warning("Ignoring episode for task %s after generation was marked complete", task_id)
            return False

        self._log_episode_debug("received", task_id, [episode])

        # Offload episode to disk if enabled
        if self._episode_offload_dir:
            path = await self._offload_episode(task_id, episode)
            self._pending.setdefault(task_id, []).append(path)
        else:
            self._pending.setdefault(task_id, []).append(episode)

        if len(self._pending[task_id]) < self._group_size:
            return False

        # Load all episodes
        if self._episode_offload_dir:
            episodes = await self._load_pending_episodes(task_id)
        else:
            episodes = self._pending.pop(task_id, [])

        self._log_episode_debug("group_ready", task_id, episodes)

        weight_version = self._min_weight_version(episodes)

        # 1. Record episode-level metrics (includes filtered tasks)
        self._record_episode_metrics(episodes)

        # 2. Transform episodes -> trajectory groups
        traj_groups, transform_metrics = transform_episodes_to_trajectory_groups(
            episodes,
            self._transform_config,
            self._cf_config,
        )
        self._aggregator.record_dict(transform_metrics)

        # 3. Drop groups with too few trajectories
        before_min_traj = len(traj_groups)
        traj_groups = [g for g in traj_groups if len(g.trajectories) >= self._rs_config.min_trajs_per_group]
        self._aggregator.record("groups/dropped_min_trajs", before_min_traj - len(traj_groups))

        # Drop groups that cannot produce any trainable backend rows. Some
        # agent terminations still carry trajectory shells, but all steps may
        # lack model_output/prompt ids; letting those through crashes the
        # verl tensor batcher with an empty sequence list.
        before_trainable = len(traj_groups)
        traj_groups = [g for g in traj_groups if self._group_has_trainable_step(g)]
        dropped_untrainable = before_trainable - len(traj_groups)
        self._aggregator.record("groups/dropped_no_trainable_steps", dropped_untrainable)

        if not traj_groups:
            if dropped_untrainable > 0:
                self._log_untrainable_debug(task_id, episodes)
            if before_min_traj > 0:
                filter_reason = "no_trainable_steps" if dropped_untrainable > 0 else "min_trajs"
            elif self._all_episodes_compact_filtered(episodes):
                filter_reason = "compact_filtering"
            else:
                filter_reason = "no_trajectory_groups"
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="filtered",
                reason=filter_reason,
                groups_after_transform=before_min_traj,
                groups_after_min_trajs=0,
                groups_after_reward_filter=0,
            )
            self._filtered_by_source[self._episodes_source(episodes)] += 1
            self._coordinator.on_group_filtered()
            self._filtered_count += 1
            self._record_classified_prompt_group()
            return True

        # 4. Compute advantages
        adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
            traj_groups,
            self._algorithm_config,
        )
        self._aggregator.record_dict(adv_metrics)

        # 5. Rejection sampling: drop groups with all-zero advantage
        filtered_zero_adv = 0
        if self._rs_config.filter_uniform_groups:
            before_adv = len(traj_groups)
            traj_groups = [g for g in traj_groups if any(abs(step.advantage) > 1e-8 for traj in g.trajectories for step in traj.steps if step.advantage is not None)]
            filtered_zero_adv = before_adv - len(traj_groups)
        self._aggregator.record("groups/dropped_zero_adv", filtered_zero_adv)

        if not traj_groups:
            self._log_prompt_group_finished(
                task_id=task_id,
                episodes=episodes,
                status="filtered",
                reason="uniform_reward",
                groups_after_transform=before_min_traj,
                groups_after_min_trajs=before_adv,
                groups_after_reward_filter=0,
            )
            self._filtered_by_source[self._episodes_source(episodes)] += 1
            self._coordinator.on_group_filtered()
            self._filtered_count += 1
            self._record_classified_prompt_group()
            return True

        # 6. Set weight version and queue
        for g in traj_groups:
            g.weight_version = weight_version

        batch = TaskBatch(groups=traj_groups, episodes=episodes)
        if self._tg_offload_dir:
            await self._queue.put(await self._offload_task_batch(batch))
        else:
            await self._queue.put(batch)
        self._training_queue_size += 1
        self._queue_update_event.set()
        self._accepted_by_source[self._episodes_source(episodes)] += 1
        self._record_classified_prompt_group()

        self._log_prompt_group_finished(
            task_id=task_id,
            episodes=episodes,
            status="queued",
            reason="accepted",
            groups_after_transform=before_min_traj,
            groups_after_min_trajs=len(traj_groups) + filtered_zero_adv,
            groups_after_reward_filter=len(traj_groups),
        )

        return True

    async def get(self) -> TaskBatch | None:
        """Get next task batch. Returns None when generation is done and buffer is drained."""
        item = await self._queue.get()
        if item is None:
            return None
        self._training_queue_size = max(0, self._training_queue_size - 1)
        self._consumed_count += 1
        self._refresh_pbar_counters()
        return await self._load_task_batch(item)

    async def get_many(self, count: int) -> list[TaskBatch] | None:
        """Get a full forward/backward chunk, or None if generation ended first."""
        while self._training_queue_size < count:
            if self._generation_complete:
                return None
            self._queue_update_event.clear()
            if self._training_queue_size >= count or self._generation_complete:
                continue
            await self._queue_update_event.wait()

        items = []
        for _ in range(count):
            item = await self._queue.get()
            if item is None:
                return None
            items.append(await self._load_task_batch(item))

        self._training_queue_size = max(0, self._training_queue_size - count)
        self._consumed_count += count
        self._refresh_pbar_counters()
        return items

    def mark_generation_complete(self) -> None:
        """Signal that generation is finished. Flushes incomplete groups and enqueues a sentinel."""
        if self._generation_complete:
            return
        self._generation_complete = True
        for task_id in list(self._pending.keys()):
            items = self._pending.pop(task_id, [])
            for item in items:
                if isinstance(item, str):
                    try:
                        os.remove(item)
                    except OSError:
                        pass
            self._coordinator.on_group_filtered()
            self._filtered_count += 1
            self._record_classified_prompt_group()
        self._queue.put_nowait(None)
        self._queue_update_event.set()

    def stats(self) -> dict:
        return {
            "async/buffer_qsize": self._training_queue_size,
            "async/buffer_pending": len(self._pending),
            "async/buffer_filtered": self._filtered_count,
            "async/buffer_consumed": self._consumed_count,
            **{f"async/buffer_accepted_source/{source}": count for source, count in self._accepted_by_source.items()},
            **{f"async/buffer_filtered_source/{source}": count for source, count in self._filtered_by_source.items()},
        }

    def _record_episode_metrics(self, episodes: list[Episode]) -> None:
        """Record episode-level metrics to aggregator (all episodes, including filtered)."""
        for ep in episodes:
            reason = ep.termination_reason or TerminationReason.UNKNOWN
            for r in TerminationReason:
                self._aggregator.record(
                    f"episode/termination_reason/{r.value}",
                    1.0 if reason == r else 0.0,
                )
            for k, v in ep.metrics.items():
                try:
                    self._aggregator.record(f"episode/{k}", float(v))
                except (TypeError, ValueError):
                    continue

            # Episode-level totals across all trajectories
            total_turns = sum(len(traj.steps) for traj in ep.trajectories)
            total_prompt_tokens = sum(len(s.prompt_ids) for traj in ep.trajectories for s in traj.steps)
            total_response_tokens = sum(len(s.response_ids) for traj in ep.trajectories for s in traj.steps)
            self._aggregator.record("episode/num_turns", total_turns)
            self._aggregator.record("episode/prompt_tokens", total_prompt_tokens)
            self._aggregator.record("episode/response_tokens", total_response_tokens)
            self._aggregator.record("episode/correct", 1.0 if ep.is_correct else 0.0)

    def _all_episodes_compact_filtered(self, episodes: list[Episode]) -> bool:
        return all(self._cf_config.should_mask(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes)

    @staticmethod
    def _episodes_source(episodes: list[Episode]) -> str:
        sources = Counter()
        for ep in episodes:
            task = getattr(ep, "task", None)
            if task is not None:
                sources[infer_task_source(task)] += 1
                continue
            metadata = getattr(ep, "metadata", None)
            if isinstance(metadata, dict) and metadata.get("data_source") is not None:
                sources[str(metadata["data_source"])] += 1
                continue
            info = getattr(ep, "info", None)
            if isinstance(info, dict) and info.get("data_source") is not None:
                sources[str(info["data_source"])] += 1
        if not sources:
            return "unknown"
        return sources.most_common(1)[0][0]

    @staticmethod
    def _group_has_trainable_step(group: TrajectoryGroup) -> bool:
        for trajectory in group.trajectories:
            for step in trajectory.steps:
                model_output = getattr(step, "model_output", None)
                if model_output is None:
                    continue
                if getattr(model_output, "prompt_ids", None) is None:
                    continue
                return True
        return False

    @staticmethod
    def _log_untrainable_debug(task_id: str, episodes: list[Episode]) -> None:
        TrajectoryGroupBuffer._log_episode_debug("no_trainable_steps", task_id, episodes)

    @staticmethod
    def _log_episode_debug(stage: str, task_id: str, episodes: list[Episode]) -> None:
        if os.environ.get("RLLM_ASYNC_BUFFER_DEBUG", "0").lower() not in {"1", "true", "yes"}:
            return
        lines = [f"[rLLM async buffer debug] stage={stage} task_id={task_id}"]
        for ep_idx, ep in enumerate(episodes[:2]):
            lines.append(
                f"  ep={ep_idx} id={getattr(ep, 'id', None)} "
                f"term={getattr(ep, 'termination_reason', None)} trajs={len(ep.trajectories)}"
            )
            for traj_idx, traj in enumerate(ep.trajectories[:2]):
                lines.append(f"    traj={traj_idx} name={getattr(traj, 'name', None)} steps={len(traj.steps)} reward={getattr(traj, 'reward', None)}")
                for step_idx, step in enumerate(traj.steps[:3]):
                    model_output = getattr(step, "model_output", None)
                    lines.append(
                        "      "
                        f"step={step_idx} type={type(step).__name__} "
                        f"has_model_output={model_output is not None} "
                        f"prompt_len={len(getattr(step, 'prompt_ids', []) or [])} "
                        f"response_len={len(getattr(step, 'response_ids', []) or [])} "
                        f"model_prompt_len={len(getattr(model_output, 'prompt_ids', []) or []) if model_output is not None else 0} "
                        f"model_completion_len={len(getattr(model_output, 'completion_ids', []) or []) if model_output is not None else 0} "
                        f"reward={getattr(step, 'reward', None)} done={getattr(step, 'done', None)}"
                    )
        print("\n".join(lines), flush=True)

    @staticmethod
    def _termination_value(reason: TerminationReason | str) -> str:
        return str(getattr(reason, "value", reason))

    def _log_prompt_group_finished(
        self,
        *,
        task_id: str,
        episodes: list[Episode],
        status: str,
        reason: str,
        groups_after_transform: int,
        groups_after_min_trajs: int,
        groups_after_reward_filter: int,
    ) -> None:
        self._last_reason = reason
        self._refresh_pbar_counters()
        termination_counts = Counter(self._termination_value(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes)
        compact_masked = Counter(
            self._termination_value(ep.termination_reason or TerminationReason.UNKNOWN) for ep in episodes if self._cf_config.should_mask(ep.termination_reason or TerminationReason.UNKNOWN)
        )
        data_sources = Counter()
        rewards = []
        for ep in episodes:
            data_source = getattr(ep, "metadata", {}).get("data_source") if hasattr(ep, "metadata") else None
            if data_source is None:
                data_source = getattr(ep, "info", {}).get("data_source") if hasattr(ep, "info") else None
            if data_source is not None:
                data_sources[str(data_source)] += 1
            reward = None
            for traj in ep.trajectories:
                if traj.reward is not None:
                    reward = traj.reward
                elif traj.steps:
                    reward = traj.steps[-1].reward
            rewards.append(reward)

        logger.info(
            "Prompt group finished task_id=%s status=%s reason=%s data_sources=%s episodes=%d rewards=%s "
            "terminations=%s compact_masked=%s groups_after_transform=%d "
            "groups_after_min_trajs=%d groups_after_reward_filter=%d",
            task_id,
            status,
            reason,
            dict(data_sources),
            len(episodes),
            rewards,
            dict(termination_counts),
            dict(compact_masked),
            groups_after_transform,
            groups_after_min_trajs,
            groups_after_reward_filter,
        )

    @staticmethod
    def _min_weight_version(episodes: list[Episode]) -> int:
        min_v = float("inf")
        for ep in episodes:
            for traj in ep.trajectories:
                for step in traj.steps:
                    if step.weight_version is not None:
                        min_v = min(min_v, step.weight_version)
        return int(min_v) if min_v != float("inf") else 0
