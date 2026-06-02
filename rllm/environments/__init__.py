"""Environment exports for legacy Agent+Environment training paths."""

import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices = {}
except ImportError:
    pass

from rllm.environments.base.base_env import BaseEnv


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


ENVIRONMENT_IMPORTS = [
    ("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    ("rllm.environments.tools.tool_env", "ToolEnvironment"),
    ("rllm.environments.tools.mcp_env", "MCPEnvironment"),
    ("rllm.environments.swe.swe", "SWEEnv"),
    ("rllm.environments.cli.cli", "CLIEnv"),
    ("rllm.environments.fused.fused", "FusedEnv"),
    ("rllm.environments.endless_terminals.et_env", "ETEnv"),
    ("rllm.environments.browsergym.browsergym", "BrowserGymEnv"),
    ("rllm.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    ("rllm.environments.code.competition_coding", "CompetitionCodingEnv"),
    ("rllm.environments.appworld.appworld_env", "AppWorldEnv"),
]

__all__ = ["BaseEnv"]

for module_path, class_name in ENVIRONMENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
