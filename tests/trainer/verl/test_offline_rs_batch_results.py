import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verl import DataProto

from rllm.agents.system_prompts import FUSED_SEARCH_SYSTEM_PROMPT, FUSED_SEARCH_USER_PROMPT
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


def test_offline_rs_batch_results_schema_and_prompts(tmp_path: Path):
    trainer = object.__new__(AgentPPOTrainer)
    trainer.config = SimpleNamespace(
        rllm={
            "batch_results_dir": str(tmp_path),
            "offline_rs_sample_n": 2,
            "offline_rs_reward_threshold": 0.6,
            "offline_rs_max_trajectory_per_problem": 1,
            "offline_rs_min_sample_trial": 1,
        },
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2)),
    )

    question = "Who founded Example Corp?"
    trajectory = [
        {"role": "system", "content": FUSED_SEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": FUSED_SEARCH_USER_PROMPT.replace("{problem_statement}", question)},
        {"role": "assistant", "content": "reason</think>\n\n<tool_call>{}</tool_call>"},
    ]
    trainer._dump_offline_rs_batch_results(
        {
            "accept_traj": [
                {
                    "uuid": "task-1",
                    "prompt": question,
                    "data_source": "web_search",
                    "reward": 1.0,
                    "trajectory": trajectory,
                    "debug": {"metrics": {"task_label": "web search"}},
                }
            ],
            "reject_traj": [],
        },
        "global_steps_9",
    )

    out_path = tmp_path / "global_steps_9.json"
    assert out_path.exists()
    assert not (tmp_path / "global_steps_9.usable_trajectories.json").exists()

    payload = json.loads(out_path.read_text())
    row = payload["selected_trajectories"][0]
    assert row["debug"]["metrics"]["task"] == "web search"
    assert "task_label" not in row["debug"]["metrics"]
    assert row["trajectory"][0]["content"] == FUSED_SEARCH_SYSTEM_PROMPT
    assert row["trajectory"][1]["content"] == FUSED_SEARCH_USER_PROMPT.replace("{problem_statement}", question)
    assert row["trajectory"][2]["content"].startswith("<think>reason</think>")


def test_offline_rs_raw_dump_matches_batch_result_schema(tmp_path: Path):
    trainer = object.__new__(AgentPPOTrainer)
    trainer.config = SimpleNamespace(
        rllm={
            "batch_results_dir": str(tmp_path),
            "offline_rs_sample_n": 2,
            "offline_rs_reward_threshold": 0.6,
            "offline_rs_max_trajectory_per_problem": 1,
            "offline_rs_min_sample_trial": 1,
        },
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2)),
    )

    batch = DataProto.from_dict(
        tensors={},
        non_tensors={
            "uid": np.array(["task-raw"], dtype=object),
            "data_source": np.array(["web_search"], dtype=object),
            "extra_info": np.array([{"ground_truth": "Example Founder"}], dtype=object),
        },
    )
    merged_data = trainer._raw_trajectories_to_dump(
        [
            {
                "idx": 0,
                "trajectory_reward": 1.0,
                "reward_debug": {"exact_match": True},
                "reward_metadata": {"exact_match": True},
                "termination_reason": "ENV_DONE",
                "exception": "",
                "chat_completions": [
                    {"role": "user", "content": "Who founded Example Corp?"},
                    {"role": "assistant", "content": "Example Founder"},
                ],
                "metrics": {"steps": 1, "task_label": "web search"},
            }
        ],
        dropped_dump=[],
        batch=batch,
    )
    trainer._dump_offline_rs_batch_results(merged_data, "global_steps_10")

    payload = json.loads((tmp_path / "global_steps_10.json").read_text())
    row = payload["selected_trajectories"][0]
    assert row["uuid"] == "task-raw"
    assert row["reward"] == 1.0
    assert row["sample_trial"] == 1
    assert row["debug"]["ground_truth"] == "Example Founder"
    assert row["debug"]["reward_metadata"]["exact_match"] is True


def test_offline_rs_pass_rate_is_threshold_pass_rate_over_sample_n(tmp_path: Path):
    trainer = object.__new__(AgentPPOTrainer)
    trainer.config = SimpleNamespace(
        rllm={
            "batch_results_dir": str(tmp_path),
            "offline_rs_sample_n": 4,
            "offline_rs_reward_threshold": 0.6,
            "offline_rs_max_trajectory_per_problem": 2,
            "offline_rs_min_sample_trial": 1,
        },
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=4)),
    )

    merged_data = {
        "accept_traj": [
            {"uuid": "task-full", "prompt": "q1", "data_source": "web_search", "reward": 1.0, "trajectory": []},
            {"uuid": "task-full", "prompt": "q1", "data_source": "web_search", "reward": 0.7, "trajectory": []},
            {"uuid": "task-full", "prompt": "q1", "data_source": "web_search", "reward": 0.2, "trajectory": []},
            {"uuid": "task-short", "prompt": "q2", "data_source": "web_search", "reward": 1.0, "trajectory": []},
        ],
        "reject_traj": [
            {"uuid": "task-full", "prompt": "q1", "data_source": "web_search", "reward": None, "trajectory": []},
            {"uuid": "task-short", "prompt": "q2", "data_source": "web_search", "reward": 0.0, "trajectory": []},
        ],
    }

    trainer._dump_offline_rs_batch_results(merged_data, "global_steps_11")

    payload = json.loads((tmp_path / "global_steps_11.json").read_text())
    selected_by_uid = {row["uuid"]: row for row in payload["selected_trajectories"]}

    assert selected_by_uid["task-full"]["pass_rate"] == 0.5
    assert selected_by_uid["task-full"]["sample_trial"] == 4
    assert selected_by_uid["task-short"]["pass_rate"] == 0.25
    assert selected_by_uid["task-short"]["sample_trial"] == 2
    assert {row["pass_rate"] for row in merged_data["accept_traj"] if row["uuid"] == "task-full"} == {0.5}
    assert {row["pass_rate"] for row in merged_data["reject_traj"] if row["uuid"] == "task-short"} == {0.25}
