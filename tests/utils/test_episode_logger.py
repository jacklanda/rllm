import json

from rllm.types import Episode, Step, Trajectory
from rllm.utils.episode_logger import EpisodeLogger


def test_episode_logger_writes_single_global_steps_file(tmp_path):
    logger = EpisodeLogger(base_dir=str(tmp_path), subdirectory="episodes")
    legacy_dir = tmp_path / "episodes" / "train_step_1_epoch_0"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "stale.json").write_text("{}", encoding="utf-8")
    episodes = [
        Episode(
            id="task:0",
            session_id="session-0",
            task={"question": "你好"},
            is_correct=True,
            metrics={"score": 1.0},
            metadata={"timing": {"total": 2.0}},
            trajectories=[
                Trajectory(
                    uid="traj-0",
                    name="solver",
                    reward=1.0,
                    metadata={"timing": {"rollout": 1.5}},
                    steps=[
                        Step(
                            observation="obs",
                            thought="想法",
                            action={"tool": "search"},
                            reward=1.0,
                            done=True,
                            model_response="答案",
                            chat_completions=[{"role": "assistant", "content": "答案"}],
                            metadata={"timing": {"step": 0.5}},
                        )
                    ],
                )
            ],
        ),
        Episode(
            id="task:1",
            task={"question": "second"},
            trajectories=[Trajectory(uid="traj-1", name="solver", reward=0.0, steps=[])],
        ),
    ]

    logger.log_episodes_batch(episodes, step=1, mode="train", epoch=0)

    out_path = tmp_path / "episodes" / "global_steps_1.json"
    assert out_path.exists()
    assert not legacy_dir.exists()
    assert not (tmp_path / "episodes" / "batch_summary.json").exists()

    raw = out_path.read_text(encoding="utf-8")
    assert "你好" in raw
    payload = json.loads(raw)

    assert payload["training_step"] == 1
    assert payload["epoch"] == 0
    assert payload["mode"] == "train"
    assert payload["num_episodes"] == 2
    assert len(payload["trajectories"]) == 2
    assert payload["trajectories"][0]["episode_id"] == "task:0"
    assert payload["trajectories"][0]["trajectories"][0]["steps"][0]["thought"] == (
        "<think>想法</think>"
    )
    assert payload["trajectories"][1]["episode_id"] == "task:1"


def test_episode_logger_dumps_complete_think_blocks():
    assert EpisodeLogger._format_thought_for_dump("reason") == "<think>reason</think>"
    assert EpisodeLogger._format_thought_for_dump("reason</think>") == "<think>reason</think>"
    assert EpisodeLogger._format_thought_for_dump("<think>reason") == "<think>reason</think>"
    assert EpisodeLogger._format_thought_for_dump("<think>reason</think>") == "<think>reason</think>"
    assert EpisodeLogger._format_thought_for_dump("") == "<think></think>"
    assert EpisodeLogger._format_thought_for_dump(None) == "<think></think>"


def test_episode_logger_completes_assistant_think_tags_in_dumps(tmp_path):
    logger = EpisodeLogger(base_dir=str(tmp_path), subdirectory="episodes")
    episode = Episode(
        id="task:0",
        task={"question": "q"},
        trajectories=[
            Trajectory(
                uid="traj-0",
                name="solver",
                steps=[
                    Step(
                        thought="reason",
                        model_response="reason</think>\n\nanswer",
                        chat_completions=[
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": "reason</think>\n\nanswer"},
                        ],
                    )
                ],
            )
        ],
    )

    logger.log_episodes_batch([episode], step=2, mode="train", epoch=0)

    payload = json.loads((tmp_path / "episodes" / "global_steps_2.json").read_text(encoding="utf-8"))
    step = payload["trajectories"][0]["trajectories"][0]["steps"][0]
    assert step["model_response"].startswith("<think>reason</think>")
    assert step["chat_completions"][-1]["content"].startswith("<think>reason</think>")
