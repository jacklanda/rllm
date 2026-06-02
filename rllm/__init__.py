"""rLLM: Reinforcement Learning with Language Models."""

import sys
import warnings

from rllm.utils.logging import configure_logging_from_env

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices = {}
except ImportError:
    pass

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode", "rollout", "evaluator", "Task"]

configure_logging_from_env()


def __getattr__(name: str):
    if name in ("rollout", "evaluator"):
        from rllm.eval.rollout_decorator import evaluator, rollout

        _mod = sys.modules[__name__]
        _mod.rollout = rollout
        _mod.evaluator = evaluator
        return rollout if name == "rollout" else evaluator

    if name == "Task":
        from rllm.types import Task

        _mod = sys.modules[__name__]
        _mod.Task = Task
        return Task

    agent_exports = {"BaseAgent", "Action", "Step", "Trajectory", "Episode"}
    if name in agent_exports:
        from rllm.agents.agent import BaseAgent
        from rllm.types import Action, Episode, Step, Trajectory

        exports = {
            "BaseAgent": BaseAgent,
            "Action": Action,
            "Step": Step,
            "Trajectory": Trajectory,
            "Episode": Episode,
        }
        _mod = sys.modules[__name__]
        for key, value in exports.items():
            setattr(_mod, key, value)
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
