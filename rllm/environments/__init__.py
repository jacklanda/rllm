"""rLLM Environments Module

This module provides various environment implementations for RL training.
Note: Some dependencies (e.g., r2egym) use deprecated gym package.
We suppress these warnings as rLLM itself uses gymnasium.
"""
import warnings

# Suppress gym deprecation warnings from dependencies
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

from rllm.environments.base.base_env import BaseEnv
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.environments.tools.tool_env import ToolEnvironment

__all__ = ["BaseEnv", "SingleTurnEnvironment", "ToolEnvironment"]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


ENVIRONMENT_IMPORTS = [
    ("rllm.environments.browsergym.browsergym", "BrowserGymEnv"),
    ("rllm.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    ("rllm.environments.swe.swe", "SWEEnv"),
    ("rllm.environments.cli.cli", "CLIEnv"),
    ("rllm.environments.code.competition_coding", "CompetitionCodingEnv"),
    ("rllm.environments.appworld.appworld_env", "AppWorldEnv"),
]

for module_path, class_name in ENVIRONMENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
