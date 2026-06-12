import importlib.util
import asyncio
import sys
import types
from pathlib import Path

import pytest


def _load_workflow_engine_module(monkeypatch):
    # Stub verl + torch bits imported inside the function.
    verl = types.ModuleType("verl")

    class FakeDataProto:
        def __init__(self, meta_info):
            self.meta_info = meta_info

        @classmethod
        def from_dict(cls, tensors=None, non_tensors=None, meta_info=None):  # noqa: ARG002
            return cls(meta_info=meta_info)

    verl.DataProto = FakeDataProto

    verl_utils = types.ModuleType("verl.utils")
    verl_tf = types.ModuleType("verl.utils.torch_functional")
    verl_tf.pad_sequence_to_length = lambda x, *a, **k: x

    sys.modules["verl"] = verl
    sys.modules["verl.utils"] = verl_utils
    sys.modules["verl.utils.torch_functional"] = verl_tf

    # Stub torch so type annotations don't crash import in minimal env.
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.long = "long"
    torch.float32 = "float32"
    torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(rnn=types.SimpleNamespace(pad_sequence=lambda *a, **k: [])))
    torch.zeros_like = lambda *a, **k: []
    torch.as_tensor = lambda *a, **k: []
    torch.arange = lambda *a, **k: []
    torch.cumsum = lambda *a, **k: []
    torch.concat = lambda *a, **k: []
    torch.empty = lambda *a, **k: []
    sys.modules["torch"] = torch

    module_path = Path(__file__).resolve().parents[2] / "rllm" / "engine" / "agent_workflow_engine.py"
    spec = importlib.util.spec_from_file_location("rllm_agent_workflow_engine_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_transform_results_includes_dropped_episodes_meta(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    # Minimal episode stubs
    class Ep:
        def __init__(self, id, termination_reason=None, trajectories=None):
            self.id = id
            self.termination_reason = termination_reason
            self.trajectories = trajectories or []
            self.metrics = {}
            self.is_correct = False

    class Term:
        def __init__(self, value):
            self.value = value

    episodes = [
        None,
        Ep("t1:0", termination_reason=Term("max_prompt_length_exceeded"), trajectories=[type("T", (), {"steps": []})()]),
    ]

    # Instantiate engine without running __init__
    engine = object.__new__(mod.AgentWorkflowEngine)
    engine.config = type("Cfg", (), {"rllm": type("R", (), {"stepwise_advantage": type("S", (), {"enable": False})()})()})()
    engine.rollout_engine = type("RE", (), {"chat_parser": type("CP", (), {})()})()

    out = mod.AgentWorkflowEngine.transform_results_for_verl(engine, episodes, ["t0", "t1"])

    assert "dropped_episodes" in out.meta_info
    assert len(out.meta_info["dropped_episodes"]) == 2


def test_extract_task_type_for_logging(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    assert mod._extract_task_type_for_logging({"task_type": "mcp"}) == "mcp"
    assert mod._extract_task_type_for_logging({"task_type": "web_search"}) == "web search"
    assert mod._extract_task_type_for_logging({"data_source": "web_search"}) == "web search"
    assert mod._extract_task_type_for_logging({"tools_py": "tools.py"}) == "mcp"
    assert mod._extract_task_type_for_logging({"docker_image": "python:3.11"}) == "cli"
    assert mod._extract_task_type_for_logging({"data_source": "simpleqa"}) == "web search"
    assert mod._extract_task_type_for_logging('{"task_type": "cli"}') == "cli"


def test_progress_safe_print_delegates_to_colorful_print(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    calls = []
    monkeypatch.setattr(mod, "colorful_print", lambda *args, **kwargs: calls.append((args, kwargs)))

    engine = object.__new__(mod.AgentWorkflowEngine)

    mod.AgentWorkflowEngine._progress_safe_print(engine, "rollout done", fg="green")

    assert calls == [(("rollout done",), {"fg": "green"})]


def test_execute_tasks_progress_log_preserves_small_positive_reward(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    class Traj:
        reward = 0.0123

    class Ep:
        id = "task:0"
        trajectories = [Traj()]
        is_correct = True
        termination_reason = mod.TerminationReason.ENV_DONE

    engine = object.__new__(mod.AgentWorkflowEngine)
    engine.workflow_queue = asyncio.Queue()
    engine.episode_logger = None
    engine.executor = None
    calls = []
    engine._progress_safe_print = lambda *args, **kwargs: calls.append((args, kwargs))

    async def process_task_with_retry(task, task_id, rollout_idx, **kwargs):  # noqa: ARG001
        return task_id, rollout_idx, Ep()

    engine.process_task_with_retry = process_task_with_retry

    episodes = asyncio.run(engine.execute_tasks([{"task_type": "web_search"}], task_ids=["task"]))

    assert len(episodes) == 1
    assert "Reward: 0.0123." in calls[0][0][0]
    assert "Reward: 0.0." not in calls[0][0][0]
    assert calls[0][1] == {"fg": "green"}
    engine.shutdown()


def test_workflow_engine_uses_agent_timeout_and_replaces_timed_out_slot(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    class SlowWorkflow(mod.Workflow):
        instances = []
        run_calls = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            SlowWorkflow.instances.append(self)

        async def run(self, task, uid, **kwargs):
            SlowWorkflow.run_calls += 1
            self.reset(task=task, uid=uid)
            await asyncio.sleep(0.05)
            return None

        def close(self):
            self.close_calls += 1

    config = types.SimpleNamespace(
        rllm=types.SimpleNamespace(
            agent=types.SimpleNamespace(
                trajectory_timeout=0.01,
                eval_trajectory_timeout=None,
            )
        )
    )
    rollout_engine = types.SimpleNamespace(validate=False)
    engine = mod.AgentWorkflowEngine(
        workflow_cls=SlowWorkflow,
        workflow_args={},
        rollout_engine=rollout_engine,
        config=config,
        n_parallel_tasks=1,
        retry_limit=5,
    )
    engine._progress_safe_print = lambda *args, **kwargs: None

    async def run_case():
        await engine.initialize_pool()
        first_workflow = SlowWorkflow.instances[0]
        _task_id, _rollout_idx, episode = await engine.process_task_with_retry({"task_type": "mcp"}, "task", 0)
        replacement = await engine.workflow_queue.get()
        try:
            assert episode.termination_reason == mod.TerminationReason.TIMEOUT
            assert SlowWorkflow.run_calls == 1
            assert first_workflow.close_calls == 1
            assert replacement is not first_workflow
            assert replacement.close_calls == 0
        finally:
            await engine.workflow_queue.put(replacement)
            engine.shutdown()

    asyncio.run(run_case())


class _ListLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _FakeBatch:
    def __init__(self, meta_info=None):
        self.meta_info = meta_info or {}
        self.non_tensor_batch = {
            "extra_info": _ListLike([{"task_type": "mcp"}]),
            "task_ids": _ListLike(["task-0"]),
        }


class _FakeRolloutEngine:
    def __init__(self):
        self.validate = False
        self.wake_calls = 0
        self.sleep_calls = 0

    async def wake_up(self):
        self.wake_calls += 1

    async def sleep(self):
        self.sleep_calls += 1


def _make_execute_tasks_verl_engine(mod, config):
    engine = object.__new__(mod.AgentWorkflowEngine)
    engine.config = config
    engine.rollout_engine = _FakeRolloutEngine()
    engine.current_mode = "train"
    engine.execute_task_calls = []

    async def execute_tasks(tasks, task_ids, **kwargs):
        engine.execute_task_calls.append((tasks, task_ids, kwargs, engine.current_mode, engine.rollout_engine.validate))
        return ["episode"]

    engine.execute_tasks = execute_tasks
    engine.transform_results_for_verl = lambda results, task_ids: (results, task_ids)
    return engine


def test_execute_tasks_verl_skips_wake_sleep_when_sleep_mode_disabled(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)
    config = {"rllm": {"rollout_enable_sleep_mode": False}}
    engine = _make_execute_tasks_verl_engine(mod, config)

    result = asyncio.run(engine.execute_tasks_verl(_FakeBatch()))

    assert result == (["episode"], ["task-0"])
    assert engine.rollout_engine.wake_calls == 0
    assert engine.rollout_engine.sleep_calls == 0
    assert engine.rollout_engine.validate is False
    assert engine.current_mode == "train"
    assert engine.execute_task_calls[0][3:] == ("train", False)


def test_execute_tasks_verl_defaults_to_sleep_mode_enabled(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)
    engine = _make_execute_tasks_verl_engine(mod, types.SimpleNamespace())

    result = asyncio.run(engine.execute_tasks_verl(_FakeBatch()))

    assert result == (["episode"], ["task-0"])
    assert engine.rollout_engine.wake_calls == 1
    assert engine.rollout_engine.sleep_calls == 1
    assert engine.rollout_engine.validate is False
    assert engine.current_mode == "train"


def test_execute_tasks_verl_resets_validate_and_sleeps_on_error(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)
    engine = _make_execute_tasks_verl_engine(mod, types.SimpleNamespace())

    async def execute_tasks(tasks, task_ids, **kwargs):  # noqa: ARG001
        assert engine.current_mode == "val"
        assert engine.rollout_engine.validate is True
        raise RuntimeError("rollout failed")

    engine.execute_tasks = execute_tasks

    with pytest.raises(RuntimeError, match="rollout failed"):
        asyncio.run(engine.execute_tasks_verl(_FakeBatch(meta_info={"validate": True})))

    assert engine.rollout_engine.wake_calls == 1
    assert engine.rollout_engine.sleep_calls == 1
    assert engine.rollout_engine.validate is False
    assert engine.current_mode == "train"


def test_execute_tasks_verl_parses_sleep_mode_string_false(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)
    config = {"rllm": {"rollout_enable_sleep_mode": "False"}}
    engine = _make_execute_tasks_verl_engine(mod, config)

    result = asyncio.run(engine.execute_tasks_verl(_FakeBatch()))

    assert result == (["episode"], ["task-0"])
    assert engine.rollout_engine.wake_calls == 0
    assert engine.rollout_engine.sleep_calls == 0


def test_workflow_engine_shutdown_closes_queued_workflows(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    class CloseTrackingWorkflow(mod.Workflow):
        instances = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            CloseTrackingWorkflow.instances.append(self)

        async def run(self, task, uid, **kwargs):
            return None

        def close(self):
            self.close_calls += 1

    config = types.SimpleNamespace(rllm=types.SimpleNamespace(agent=types.SimpleNamespace(trajectory_timeout=None, eval_trajectory_timeout=None)))
    rollout_engine = types.SimpleNamespace(validate=False)
    engine = mod.AgentWorkflowEngine(
        workflow_cls=CloseTrackingWorkflow,
        workflow_args={},
        rollout_engine=rollout_engine,
        config=config,
        n_parallel_tasks=2,
        retry_limit=1,
    )

    async def run_case():
        await engine.initialize_pool()
        engine.shutdown()
        assert [workflow.close_calls for workflow in CloseTrackingWorkflow.instances] == [1, 1]
        assert engine.executor is None

    asyncio.run(run_case())
