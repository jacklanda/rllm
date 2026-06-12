"""Tests for step / tool-call-turn / per-source channel metrics emitted by
:meth:`rllm.workflows.workflow.Workflow.collect_metrics` and the shared
task-source inference helpers.
"""

from rllm.types import Episode, Step, Trajectory
from rllm.workflows.workflow import (
    Workflow,
    count_tool_call_turns,
    infer_task_source,
)


def _make_workflow(task=None):
    """Construct a minimal concrete Workflow without its heavy __init__.

    collect_metrics only depends on ``self.task`` (as a fallback) and the
    passed-in episode, so a bare instance is sufficient.
    """

    class _StubWorkflow(Workflow):
        async def run(self, task, uid, **kwargs):  # pragma: no cover - never called
            raise NotImplementedError

    wf = _StubWorkflow.__new__(_StubWorkflow)
    wf.task = task
    wf.strict_eval_accuracy = False
    return wf


def _step_with_tool_call(qwen_style=True):
    if qwen_style:
        return Step(
            model_response='thinking...<tool_call>{"name": "search"}</tool_call>',
            chat_completions=[{"role": "assistant", "content": "<tool_call>x</tool_call>"}],
        )
    return Step(chat_completions=[{"role": "assistant", "tool_calls": [{"id": "1"}], "content": ""}])


def _step_without_tool_call():
    return Step(
        model_response="final answer is 42",
        chat_completions=[{"role": "assistant", "content": "final answer is 42"}],
    )


def _episode(steps, reward=1.0, task=None):
    traj = Trajectory(name="agent", steps=steps, reward=reward)
    return Episode(trajectories=[traj], task=task)


def _strict_workflow(task=None):
    wf = _make_workflow(task=task)
    wf.strict_eval_accuracy = True
    return wf


# --------------------------------------------------------------------------- #
# infer_task_source
# --------------------------------------------------------------------------- #


def test_infer_task_source_explicit_task_type():
    assert infer_task_source({"task_type": "mcp"}) == "mcp"
    assert infer_task_source({"task_type": "web_search"}) == "web search"
    assert infer_task_source({"task_type": "swe"}) == "cli"


def test_infer_task_source_from_fields():
    assert infer_task_source({"tools_py": "print('x')"}) == "mcp"
    assert infer_task_source({"docker_image": "img:latest"}) == "cli"
    assert infer_task_source({"data_source": "hotpotqa"}) == "web search"


def test_infer_task_source_json_string_and_unknown():
    assert infer_task_source('{"task_type": "mcp"}') == "mcp"
    assert infer_task_source("not json") == "unknown"
    assert infer_task_source({}) == "unknown"


# --------------------------------------------------------------------------- #
# count_tool_call_turns
# --------------------------------------------------------------------------- #


def test_count_tool_call_turns_mixed():
    traj = Trajectory(
        steps=[
            _step_with_tool_call(qwen_style=True),
            _step_without_tool_call(),
            _step_with_tool_call(qwen_style=False),
        ]
    )
    assert count_tool_call_turns(traj) == 2


def test_count_tool_call_turns_info_count():
    traj = Trajectory(steps=[Step(metadata={"tool_calls": 3}), Step(metadata={"tool_calls": 0})])
    assert count_tool_call_turns(traj) == 1


# --------------------------------------------------------------------------- #
# collect_metrics
# --------------------------------------------------------------------------- #


def test_collect_metrics_global_steps_and_tool_turns():
    steps = [_step_with_tool_call(), _step_without_tool_call(), _step_with_tool_call()]
    episode = _episode(steps, task={"task_type": "mcp"})

    _make_workflow().collect_metrics(episode)

    assert episode.metrics["traj/steps"] == 3.0
    assert episode.metrics["turn/tool_call_turn"] == 2.0
    assert episode.metrics["agent_acc"] == 1.0


def test_collect_metrics_mcp_channel():
    steps = [_step_with_tool_call(), _step_with_tool_call()]
    episode = _episode(steps, task={"task_type": "mcp"})

    _make_workflow().collect_metrics(episode)

    assert episode.metrics["traj/steps/mcp"] == 2.0
    assert episode.metrics["turn/tool_call_turn/mcp"] == 2.0
    # Other source channels must not be populated for an mcp episode.
    assert "traj/steps/cli" not in episode.metrics
    assert "traj/steps/search" not in episode.metrics


def test_collect_metrics_cli_and_search_channels():
    cli_episode = _episode([_step_without_tool_call()], task={"docker_image": "img"})
    _make_workflow().collect_metrics(cli_episode)
    assert cli_episode.metrics["traj/steps/cli"] == 1.0
    assert "traj/steps/mcp" not in cli_episode.metrics

    search_episode = _episode([_step_with_tool_call(), _step_without_tool_call()], task={"data_source": "hotpotqa"})
    _make_workflow().collect_metrics(search_episode)
    assert search_episode.metrics["traj/steps/search"] == 2.0
    assert search_episode.metrics["turn/tool_call_turn/search"] == 1.0


def test_collect_metrics_unknown_source_has_no_channel():
    episode = _episode([_step_without_tool_call()], task={})
    _make_workflow().collect_metrics(episode)

    assert episode.metrics["traj/steps"] == 1.0
    assert not any(k.startswith("traj/steps/") for k in episode.metrics)


def test_collect_metrics_falls_back_to_workflow_task():
    # episode.task is None -> falls back to self.task on the workflow.
    episode = _episode([_step_with_tool_call()], task=None)
    _make_workflow(task={"task_type": "mcp"}).collect_metrics(episode)

    assert episode.metrics["traj/steps/mcp"] == 1.0


def test_strict_eval_accuracy_counts_search_exact_match_from_episode_metadata():
    episode = _episode([_step_without_tool_call()], reward=0.5, task={"data_source": "2wiki"})
    episode.info["reward_debug"] = {
        "type": "web search",
        "reward_source": "search_reward_fn",
        "reward_mode": "f1",
        "resolved": False,
        "exact_match": True,
        "f1_score": 1.0,
    }

    wf = _strict_workflow()
    wf.assign_episode_correctness(episode, is_validation=True)
    wf.collect_metrics(episode, is_validation=True)

    assert episode.is_correct is True
    assert episode.metrics["agent_acc"] == 1.0


def test_strict_eval_accuracy_rejects_search_partial_f1():
    episode = _episode([_step_without_tool_call()], reward=0.4, task={"data_source": "2wiki"})
    episode.info["reward_debug"] = {
        "type": "web search",
        "reward_source": "search_reward_fn",
        "reward_mode": "f1",
        "is_correct": True,
        "resolved": False,
        "exact_match": False,
        "f1_score": 0.8,
    }

    wf = _strict_workflow()
    wf.assign_episode_correctness(episode, is_validation=True)
    wf.collect_metrics(episode, is_validation=True)

    assert episode.is_correct is False
    assert episode.metrics["agent_acc"] == 0.0


def test_strict_eval_accuracy_rejects_search_f1_one_without_exact_match():
    episode = _episode([_step_without_tool_call()], reward=1.0, task={"data_source": "2wiki"})
    episode.info["reward_debug"] = {
        "type": "web search",
        "reward_source": "search_reward_fn",
        "reward_mode": "f1",
        "resolved": True,
        "exact_match": False,
        "f1_score": 1.0,
    }

    wf = _strict_workflow()
    wf.assign_episode_correctness(episode, is_validation=True)
    wf.collect_metrics(episode, is_validation=True)

    assert episode.is_correct is False
    assert episode.metrics["agent_acc"] == 0.0


def test_strict_eval_accuracy_keeps_resolved_signal_for_non_search_tasks():
    episode = _episode([Step(metadata={"reward_debug": {"resolved": True}})], reward=0.2, task={"task_type": "mcp"})

    wf = _strict_workflow()
    wf.assign_episode_correctness(episode, is_validation=True)
    wf.collect_metrics(episode, is_validation=True)

    assert episode.is_correct is True
    assert episode.metrics["agent_acc"] == 1.0
