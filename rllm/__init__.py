"""rLLM: Reinforcement Learning with Language Models

Main package for the rLLM framework.
"""

# ============================================================================
# Suppress gym deprecation warnings from dependencies
# ============================================================================
# rLLM uses the modern 'gymnasium' package for RL environments.
# However, some dependencies (e.g., r2e-gym) still use the deprecated 'gym'
# package, which triggers warnings about being unmaintained.
#
# Since we cannot control third-party dependencies, we suppress these warnings
# at the package level. This is safe because:
# 1. rLLM's own code uses gymnasium, not gym
# 2. The warnings are about maintainability, not functionality
# 3. The dependency packages still work correctly with gym
#
# If you need to see these warnings for debugging, comment out the filters below.
# ============================================================================
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

# The "Gym has been unmaintained..." banner is NOT a warnings.warn() call —
# gym/__init__.py does `print(notice, file=sys.stderr)` directly, reading from
# the `gym_notices` package. warnings.filterwarnings cannot suppress raw prints,
# so we neutralize the source by emptying gym_notices.notices before gym is
# imported by any downstream dependency (e.g. r2egym).
try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices = {}
except ImportError:
    pass

# Import commonly used classes
from .agents import Action, BaseAgent, Episode, Step, Trajectory

__all__ = [
    "BaseAgent",
    "Action",
    "Step",
    "Trajectory",
    "Episode",
]
