import json

from experiments.fused.merge_eval_json import _make_search_reward_config, merge_eval_json


def test_merge_eval_json_combines_shards_and_eval_log(tmp_path):
    episode_dir = tmp_path / "work" / "episodes"
    episode_dir.mkdir(parents=True)
    (episode_dir / "val_global_steps_0_epoch_0.json").write_text(
        json.dumps(
            {
                "training_step": 0,
                "epoch": 0,
                "mode": "val",
                "num_episodes": 1,
                "trajectories": [{"episode_id": "a", "task": {"question": "你好"}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (episode_dir / "val_global_steps_1_epoch_0.json").write_text(
        json.dumps(
            {
                "training_step": 1,
                "epoch": 0,
                "mode": "val",
                "num_episodes": 1,
                "trajectories": [{"episode_id": "b", "task": {"question": "second"}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    eval_log = tmp_path / "work" / "eval.log"
    eval_log.write_text("Initial validation metrics: {'acc': 1.0}\n中文日志\n", encoding="utf-8")
    output_json = tmp_path / "cot_evals_20260609124339.json"

    payload = merge_eval_json(
        episode_log_dir=tmp_path / "work",
        eval_log=eval_log,
        output_json=output_json,
        harness="cot",
        experiment_name="mcp-dr-grpo-4b-evals",
        timestamp="20260609124339",
        eval_status=0,
    )

    raw = output_json.read_text(encoding="utf-8")
    written = json.loads(raw)

    assert payload["harness"] == "cot"
    assert written["num_episode_shards"] == 2
    assert written["num_episodes"] == 2
    assert [traj["episode_id"] for traj in written["trajectories"]] == ["a", "b"]
    assert "Initial validation metrics" in written["eval_log"]
    assert "中文日志" in written["eval_log"]
    assert "\\u4f60\\u597d" not in raw
    assert raw.startswith("{\n    ")


def test_search_reward_config_supports_older_constructor_signature():
    class OldRewardConfig:
        def __init__(self, toolcall_bonus=0.5, enable_step_bonus=True):
            self.toolcall_bonus = toolcall_bonus
            self.enable_step_bonus = enable_step_bonus

    config = _make_search_reward_config(OldRewardConfig)

    assert config.toolcall_bonus == 0.0
    assert config.apply_repetition_penalty is True
    assert config.repetition_penalty_weight == 0.2
    assert config.apply_length_penalty is True
    assert config.length_penalty_weight == 0.15
    assert config.enable_step_bonus is False


def test_merge_eval_json_rescores_false_negatives_and_false_positives(tmp_path):
    episode_dir = tmp_path / "work" / "episodes"
    episode_dir.mkdir(parents=True)
    question = "Pick one.\nA. alpha\nB. beta\nC. gamma\nD. delta"
    (episode_dir / "val_global_steps_0_epoch_0.json").write_text(
        json.dumps(
            {
                "training_step": 0,
                "epoch": 0,
                "mode": "val",
                "num_episodes": 2,
                "trajectories": [
                    {
                        "episode_id": "false-negative",
                        "termination_reason": "env_done",
                        "task": {"question": question, "answer": "B", "data_source": "gpqa_diamond"},
                        "is_correct": False,
                        "metrics": {"default_traj_name_acc": 0.0},
                        "metadata": {"reward_debug": {"extracted_answer": ""}},
                        "trajectories": [{"reward": 0.0, "steps": [{"action": r"Reasoning. \boxed{B}", "reward": 0.0}]}],
                    },
                    {
                        "episode_id": "false-positive",
                        "termination_reason": "env_done",
                        "task": {"question": question, "answer": "B", "data_source": "gpqa_diamond"},
                        "is_correct": True,
                        "metrics": {"default_traj_name_acc": 1.0},
                        "metadata": {"reward_debug": {"extracted_answer": "B"}},
                        "trajectories": [{"reward": 1.0, "steps": [{"action": r"Reasoning. \boxed{A}", "reward": 1.0}]}],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    eval_log = tmp_path / "work" / "eval.log"
    eval_log.write_text("", encoding="utf-8")
    output_json = tmp_path / "cot_evals.json"

    payload = merge_eval_json(
        episode_log_dir=tmp_path / "work",
        eval_log=eval_log,
        output_json=output_json,
        harness="cot",
        experiment_name="evals",
        timestamp="20260611120000",
        eval_status=0,
    )

    summary = payload["rescore_summary"]
    assert summary["checked"] == 2
    assert summary["rescued"] == 1
    assert summary["downgraded"] == 1
    assert payload["rescored_metrics"]["correct"] == 1
    assert payload["trajectories"][0]["is_correct"] is True
    assert payload["trajectories"][1]["is_correct"] is False
    assert payload["trajectories"][1]["trajectories"][0]["reward"] == 0.0


def test_merge_eval_json_rescues_committed_then_truncated_turn(tmp_path):
    """A length-exceeded turn that committed a clean marker before the cap is
    rescued; a degenerate truncation with no commitment stays penalized."""
    episode_dir = tmp_path / "work" / "episodes"
    episode_dir.mkdir(parents=True)
    question = "Pick one.\nA. alpha\nB. beta\nC. gamma\nD. delta"
    (episode_dir / "val_global_steps_0_epoch_0.json").write_text(
        json.dumps(
            {
                "training_step": 0,
                "epoch": 0,
                "mode": "val",
                "num_episodes": 2,
                "trajectories": [
                    {
                        # committed <answer>D</answer> then ran past the cap with
                        # degenerate repetition -> workflow aborted before env.step.
                        "episode_id": "committed-then-truncated",
                        "termination_reason": "max_response_length_exceeded",
                        "task": {"question": question, "answer": "D", "data_source": "gpqa_diamond"},
                        "is_correct": False,
                        "metrics": {"default_traj_name_acc": 0.0},
                        "trajectories": [
                            {
                                "reward": -1.0,
                                "steps": [
                                    {
                                        "model_response": "Reasoning. <answer>D</answer>\nWait. Wait. Wait. Wait.",
                                        "reward": -1.0,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        # pure degenerate truncation, never committed -> stays penalized.
                        "episode_id": "degenerate-truncation",
                        "termination_reason": "max_response_length_exceeded",
                        "task": {"question": question, "answer": "D", "data_source": "gpqa_diamond"},
                        "is_correct": False,
                        "metrics": {"default_traj_name_acc": 0.0},
                        "trajectories": [
                            {
                                "reward": -1.0,
                                "steps": [{"model_response": "Wait. Wait. Wait. Wait. Wait.", "reward": -1.0}],
                            }
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    eval_log = tmp_path / "work" / "eval.log"
    eval_log.write_text("", encoding="utf-8")
    output_json = tmp_path / "cot_evals.json"

    payload = merge_eval_json(
        episode_log_dir=tmp_path / "work",
        eval_log=eval_log,
        output_json=output_json,
        harness="cot",
        experiment_name="evals",
        timestamp="20260611120001",
        eval_status=0,
    )

    summary = payload["rescore_summary"]
    assert summary["checked"] == 1  # only the marker-bearing length-exceeded turn
    assert summary["rescued"] == 1
    assert summary["skipped_non_env_done"] == 1  # the degenerate one
    assert payload["trajectories"][0]["is_correct"] is True
    assert payload["trajectories"][0]["trajectories"][0]["reward"] == 1.0
    # degenerate truncation untouched: still penalized, not counted correct.
    assert payload["trajectories"][1]["trajectories"][0]["reward"] == -1.0
