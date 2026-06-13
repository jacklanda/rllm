"""Episode JSON Logger for saving detailed episode information."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from rllm.types import Episode
from rllm.utils.think_tags import (
    format_assistant_content_for_dump,
    format_think_block,
    sanitize_messages_for_dump,
)


class EpisodeLogger:
    """Logger to save each rollout batch to a single JSON file."""

    def __init__(self, base_dir: str, subdirectory: str = "episodes"):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Subdirectory within base_dir for episodes (default: "episodes")
                         Final path will be: {base_dir}/{subdirectory}/
        """
        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_task_hash(task: Any, length: int = 8) -> str:
        """Compute a hash from the task data.

        Args:
            task: The task dictionary or data
            length: Length of the hash to use (default 8 chars)

        Returns:
            Hash string
        """
        # Convert task to a stable string representation
        task_str = json.dumps(task, sort_keys=True, default=str)
        # Compute SHA256 hash
        hash_obj = hashlib.sha256(task_str.encode("utf-8"))
        # Return first `length` characters of hex digest
        return hash_obj.hexdigest()[:length]

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the legacy directory path for a specific training or validation step.

        The current logger writes batch files directly under ``self.log_dir`` via
        :meth:`get_batch_file`; this helper is kept for callers that still need
        to reason about the old ``{mode}_step_{step}_epoch_{epoch}`` layout.
        """
        return self.log_dir / f"{mode}_step_{step}_epoch_{epoch}"

    def get_batch_file(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the merged JSON file path for a rollout batch."""
        if mode == "train":
            filename = f"global_steps_{step}.json"
        else:
            filename = f"{mode}_global_steps_{step}_epoch_{epoch}.json"
        return self.log_dir / filename

    def _cleanup_legacy_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> None:
        """Remove legacy per-step directory output if it exists."""
        legacy_step_dir = self.get_step_dir(step, mode, epoch)
        if legacy_step_dir.exists():
            shutil.rmtree(legacy_step_dir)

    @staticmethod
    def _format_thought_for_dump(thought: Any) -> str:
        """Return a complete think block for the dumped thought field."""
        return format_think_block(thought)

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate legacy filename for an episode.

        Format: episode_hash{task_hash}_id{episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        task_hash = self.compute_task_hash(episode.task)
        # Clean episode_id to make it filesystem-safe
        episode_id_safe = str(episode.id).replace(":", "_").replace("/", "_")

        filename = f"episode_hash{task_hash}_id{episode_id_safe}.json"
        return filename

    def _episode_to_dict(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0) -> dict:
        episode_data = {
            "training_step": step,
            "epoch": epoch,
            "mode": mode,
            "episode_id": episode.id,
            "session_id": episode.session_id,
            "task": episode.task,
            "task_hash": self.compute_task_hash(episode.task),
            "is_correct": episode.is_correct,
            "termination_reason": (episode.termination_reason.value if episode.termination_reason else None),
            "metrics": episode.metrics,
            "metadata": episode.metadata,
            "timing": episode.info.get("timing", {}),
            "trajectories": [],
        }

        for traj in episode.trajectories:
            traj_data = {
                "name": traj.name,
                "uid": traj.uid,
                "reward": traj.reward,
                "num_steps": len(traj.steps),
                "timing": traj.info.get("timing", {}),
                "steps": [
                    {
                        "observation": step.observation,
                        "thought": self._format_thought_for_dump(step.thought),
                        "action": step.action,
                        "reward": step.reward,
                        "done": step.done,
                        "model_response": format_assistant_content_for_dump(step.model_response),
                        "chat_completions": sanitize_messages_for_dump(step.chat_completions),
                        "timing": step.info.get("timing", {}),
                    }
                    for step in traj.steps
                ],
            }
            episode_data["trajectories"].append(traj_data)

        return episode_data

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to the merged batch JSON file.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        self.log_episodes([episode], step, mode, epoch)

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes to one step-level JSON file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        print(f"[EpisodeLogger] Logging {len(episodes)} episodes for step={step}, mode={mode}, epoch={epoch}")
        batch_data = {
            "training_step": step,
            "epoch": epoch,
            "mode": mode,
            "num_episodes": len(episodes),
            "trajectories": [self._episode_to_dict(episode, step, mode, epoch) for episode in episodes],
        }

        filepath = self.get_batch_file(step, mode, epoch)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, indent=4, ensure_ascii=False, default=str)
                f.write("\n")
                f.flush()  # Ensure data is written to disk
            self._cleanup_legacy_step_dir(step, mode, epoch)
            print(f"[EpisodeLogger] Successfully logged {len(episodes)} episodes to {filepath}")
        except Exception as e:
            print(f"Error writing episodes to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes to one merged JSON file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Kept for API compatibility. Batch summaries are no
                longer written because the merged JSON file is the only output.
        """
        self.log_episodes(episodes, step, mode, epoch)
