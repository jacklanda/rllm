"""Backward-compatible re-export shim for legacy agent classes and trajectory types."""

from rllm.agents.agent import BaseAgent
from rllm.types import Action, Episode, Step, Trajectory


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


AGENT_IMPORTS = [
    ("rllm.agents.math_agent", "MathAgent"),
    ("rllm.agents.tool_agent", "ToolAgent"),
    ("rllm.agents.tool_agent", "MCPToolAgent"),
    ("rllm.agents.swe_agent", "SWEAgent"),
    ("rllm.agents.cli_agent", "CLIAgent"),
    ("rllm.agents.fused_agent", "FusedAgent"),
    ("rllm.agents.et_agent", "ETAgent"),
    ("rllm.agents.miniwob_agent", "MiniWobAgent"),
    ("rllm.agents.frozenlake_agent", "FrozenLakeAgent"),
    ("rllm.agents.code_agent", "CompetitionCodingAgent"),
    ("rllm.agents.webarena_agent", "WebArenaAgent"),
]

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode"]

for module_path, class_name in AGENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
