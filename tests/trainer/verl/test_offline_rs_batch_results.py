import json
from pathlib import Path
from types import SimpleNamespace

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
        {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
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
