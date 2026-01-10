"""rLLM: Reinforcement Learning with Language Models

Main package for the rLLM framework.
"""

# Suppress gym deprecation warning from dependencies (e.g., r2e_gym)
# rLLM uses gymnasium, but some dependencies still use the deprecated gym
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Import commonly used classes
from .agents import Action, BaseAgent, Episode, Step, Trajectory

__all__ = [
    "BaseAgent",
    "Action",
    "Step",
    "Trajectory",
    "Episode",
]
