"""Backward-compat re-export shim for the trajectory data types.

The canonical home for :class:`Step`, :class:`Trajectory`, :class:`Episode`,
:class:`Action`, :class:`TrajectoryGroup` is :mod:`rllm.types`. This module
re-exports them so existing imports keep working::

    from rllm.agents.agent import Episode, Step, Trajectory  # still works

:class:`BaseAgent` is still defined here because the surviving
:mod:`rllm.workflows` path consumes it (e.g. as the ``agent_cls`` field on a
:class:`~rllm.workflows.workflow.Workflow`). New agents that don't go through
that path should implement :class:`rllm.types.AgentFlow` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rllm.types import (
    _DEFAULT_TRAJ_NAME,
    Action,
    Episode,
    Step,
    Trajectory,
    TrajectoryGroup,
)

__all__ = [
    "Action",
    "BaseAgent",
    "Episode",
    "Step",
    "Trajectory",
    "TrajectoryGroup",
    "_DEFAULT_TRAJ_NAME",
]


class BaseAgent(ABC):
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Converts agent's internal state into a list of OAI chat completions."""
        return []

    @property
    def trajectory(self) -> Trajectory:
        """Converts agent's internal state into a Trajectory object."""
        return Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        Args:
            observation (Any): The observation after stepping through environment.
            reward (float): The reward received after taking the action.
            done (bool): Whether the episode has ended due to termination.
            info (dict): Additional metadata from the environment.
        """
        raise NotImplementedError("Subclasses must implement update_from_env")

    def update_from_model(self, response: str, **kwargs) -> "Action | list[Action]":
        """
        Updates the agent's internal state after the model generates a response.

        Args:
            response (str): The response from the model.

        Returns:
            Action or list[Action]: A single action or a list of actions to execute.
        """
        raise NotImplementedError("Subclasses must implement update_from_model")

    def update_from_env_intermediate(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Append an intermediate tool response to the agent's message list.

        Called between tool calls within a single model turn when the model
        generates multiple tool calls. Unlike update_from_env(), this does NOT
        create a new Step or update trajectory metadata.

        Args:
            observation: The observation from the environment.
            reward: The reward received.
            done: Whether the episode has ended.
            info: Additional metadata from the environment.
        """
        raise NotImplementedError("Subclasses must implement this method to support multi-tool-call execution")

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state, typically called at the beginning of a new episode.

        This function should clear any stored history or state information necessary
        for a fresh interaction.

        Returns:
            None
        """
        return

    def get_current_state(self) -> Step | None:
        """
        Returns the agent's current state as a dictionary.

        This method provides access to the agent's internal state at the current step,
        which can be useful for debugging, logging, or state management.

        Returns:
            Step: The agent's current state.
        """
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
