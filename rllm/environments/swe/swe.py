import json
import logging
import os
import re
import tempfile
import warnings

import numpy as np
from datasets import Dataset, load_dataset

# Suppress gym deprecation warnings from r2egym dependency
# r2egym uses deprecated gym package, but rLLM uses gymnasium
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

try:
    import r2egym
    from r2egym.agenthub.action import Action
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
except ImportError:
    r2egym = None
    EnvArgs = None
    RepoEnv = None
    Action = None

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)

try:
    R2EGYM_PATH = os.path.dirname(r2egym.__file__)
except Exception:
    R2EGYM_PATH = ""
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

SWEAGENT_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]

R2E_ENV_IDS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "R2E-Gym/SWE-Bench-Lite",
]
DEFAULT_R2E_ENV_ID = "R2E-Gym/R2E-Gym-Lite"


class SWEEnv(BaseEnv):
    """Software Engineering Environment for code-related tasks.

    Reward model (binary, exit-code only):
        * For gemcli/gemswe samples (SOP cli_data_20260429 format): at
          reward-compute time we inject ``sample["eval_script"]`` into the
          container as ``run_tests.sh``, execute it, and parse its
          ``OMNIGRIL_EXIT_CODE=<n>`` line. reward = 1.0 iff exit code == 0,
          else 0.0. Nothing test-related is prepared at reset time so the
          agent cannot read the test_patch or verification commands during
          its trajectory.
        * For standard r2egym images: we delegate to
          ``RepoEnv.compute_reward()`` unchanged.
    """

    def __init__(
        self,
        entry: dict | None = None,
        dataset: Dataset | None = None,
        idx: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        backend: str = "docker",
        verbose: bool = False,
        scaffold: str = "r2egym",
    ):
        """Initialize the SWE environment.

        Args:
            dataset: Dataset containing the tasks. If None, uses default dataset.
            idx: Index of the task to use. If None, selects a random task.
            timeout: Timeout for each step in seconds.
        """
        if entry is not None:
            self.entry = entry
            self.dataset = None
            self.idx = None
        else:
            if dataset is None:
                dataset = load_dataset(DEFAULT_R2E_ENV_ID, split="test")
            self.dataset = dataset

            if idx is None:
                idx = np.random.randint(0, len(self.dataset))
            assert 0 <= idx < len(self.dataset), "Selected index out of range"
            self.idx = idx
            self.entry = self.dataset[idx]
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.total_steps = 0
        self.backend = backend
        self.env = None
        self.verbose = verbose
        self.scaffold = scaffold
        docker_image = self.entry.get("docker_image") or ""
        self._is_gemcli = "gemcli" in docker_image or "gemswe" in docker_image
        self._reward_debug: dict = {}
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"

    def reset(self) -> tuple[str, dict]:
        """Reset the environment to initial state.

        Returns:
            Tuple containing task instruction and additional info.
        """
        # Reset environment and docker runtime.
        if not self.env:
            env_args = EnvArgs(ds=self.entry)
            self.env = RepoEnv(env_args, backend=self.backend, step_timeout=self.step_timeout, reward_timeout=self.reward_timeout, verbose=self.verbose)
        else:
            self.env.reset()
        if self.scaffold == "r2egym":
            self.env.add_commands(R2EGYM_COMMAND_FILES)
        else:
            self.env.add_commands(SWEAGENT_COMMAND_FILES)
        self._fix_tool_shebangs()
        self._suppress_git_warnings()
        self._install_tool_dependencies()

        # NOTE: eval_script / run_tests.sh are intentionally NOT injected here.
        # They are written to the container at reward-compute time only (see
        # ``_inject_eval_script`` and ``compute_final_reward_metadata``) so the
        # agent cannot read the test_patch or verification commands during its
        # trajectory. The Docker image already starts at ``base_commit`` for
        # gemcli/gemswe samples, so no pre-trajectory patching is required.

        self.total_steps = 0
        # Dump a container-env fingerprint on every reset so tool-env regressions are grep-able.
        self._log_container_fingerprint()
        return self.env.get_task_instruction(), {}

    def _copy_content_to_container(self, content: str, container_path: str, *, suffix: str = ".sh", chmod: bool = False) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            self.env.runtime.copy_to_container(tmp_path, container_path)
            if chmod:
                self.env.runtime.run(f"chmod +x {container_path}")
        finally:
            os.unlink(tmp_path)

    def _log_container_fingerprint(self):
        """Log python3 path, chardet version, and file_editor availability for regression detection."""
        cmd = (
            "python3 -c 'import sys,chardet; print(\"python3:\",sys.executable,\"chardet:\",chardet.__version__)' 2>&1; "
            "ls -la /usr/local/bin/file_editor 2>&1 | head -1"
        )
        out, _ = self.env.runtime.run(cmd, timeout=15)
        logger.info("container_fingerprint: %s", (out or "").strip())

    def _fix_tool_shebangs(self):
        """Fix tool script shebangs in the Docker container to use portable interpreter path.

        R2E-Gym tool scripts are shipped with #!/root/.venv/bin/python, which only
        works if setup_env() successfully symlinked a Python venv there.  Replacing
        with #!/usr/bin/env python3 makes the scripts work regardless, since python3
        is always reachable via DOCKER_PATH.
        """
        if self.scaffold == "r2egym":
            tool_names = ["file_editor", "execute_bash", "search", "finish"]
        else:
            tool_names = ["str_replace_editor", "execute_bash", "submit"]

        sed_cmds = " && ".join(
            f"sed -i '1s|^#!.*python.*$|#!/usr/bin/env python3|' /usr/local/bin/{name}"
            for name in tool_names
        )
        output, error_code = self.env.runtime.run(sed_cmds, timeout=15)
        if error_code and "Error" in str(error_code):
            logger.warning("Failed to fix tool shebangs: %s", output)

    def _suppress_git_warnings(self):
        """Suppress git ambiguous refname warnings in the Docker container.

        When commits have hashes that collide with ref names, git outputs
        multi-line warnings to STDOUT. These warnings corrupt file content
        when retrieved via git show/cat-file, and clutter the agent's
        command output. We suppress at multiple levels:
        1. advice.objectNameWarning disables porcelain warnings
        2. core.warnAmbiguousRefs=false disables ambiguous ref warnings
        """
        self.env.runtime.run(
            "git config advice.objectNameWarning false 2>/dev/null; "
            "git config advice.ambiguousFetchRefspec false 2>/dev/null; "
            "git config core.warnAmbiguousRefs false 2>/dev/null",
            timeout=15,
        )

    def _install_tool_dependencies(self):
        """Install Python packages required by tool scripts in the Docker container.

        The file_editor tool imports ``chardet``; ``coverage`` is used by some
        eval_scripts. Tool shebangs are rewritten to ``#!/usr/bin/env python3``
        by ``_fix_tool_shebangs``, so deps must live in the interpreter that
        ``python3`` resolves to. Many base images ship ``/usr/bin/python3``
        without the ``pip`` module, so we cascade through several install
        strategies and only warn once the smoke test still fails.
        """
        deps = ["chardet", "coverage"]
        deps_arg = " ".join(deps)
        smoke_cmd = "python3 -c 'import chardet, coverage; print(\"tool_deps_ok\")' 2>&1"

        smoke_out, _ = self.env.runtime.run(smoke_cmd, timeout=15)
        if "tool_deps_ok" in (smoke_out or ""):
            return

        attempts = [
            ("python3 -m pip", f"python3 -m pip install --quiet --disable-pip-version-check {deps_arg}"),
            ("ensurepip+pip",
             "python3 -m ensurepip --default-pip >/dev/null 2>&1; "
             f"python3 -m pip install --quiet --disable-pip-version-check {deps_arg}"),
            ("pip3", f"pip3 install --quiet --disable-pip-version-check {deps_arg}"),
            ("apt-get",
             "apt-get update -qq >/dev/null 2>&1 && "
             "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
             "python3-chardet python3-coverage >/dev/null 2>&1"),
        ]
        install_log = []
        for label, cmd in attempts:
            output, error_code = self.env.runtime.run(cmd, timeout=120)
            install_log.append(f"[{label}] ec={error_code} out={(output or '')[:200]}")
            smoke_out, _ = self.env.runtime.run(smoke_cmd, timeout=15)
            if "tool_deps_ok" in (smoke_out or ""):
                return

        logger.warning(
            "Tool-dependency smoke test FAILED after all install strategies; "
            "file_editor will crash with ModuleNotFoundError. attempts=%s last_smoke=%s",
            " | ".join(install_log),
            (smoke_out or "").strip(),
        )

    def _inject_eval_script(self):
        """Inject the sample's ``eval_script`` into the container as run_tests.sh.

        Called at reward-compute time (never during reset), so the agent cannot
        read the verification script or the embedded test_patch during its
        trajectory. The eval_script itself ``git apply``s the test_patch, runs
        tests, and prints ``OMNIGRIL_EXIT_CODE=<n>``.
        """
        eval_script = self.entry.get("eval_script")
        if not eval_script:
            return False
        alt_path = self.env.runtime.alt_path
        self._copy_content_to_container(eval_script, f"{alt_path}/run_tests.sh", chmod=True)
        logger.info("Injected eval_script as run_tests.sh for reward evaluation")
        return True

    def _build_reward_debug(
        self,
        *,
        reward: float,
        reward_source: str,
        output: str = "",
        error_code=None,
        verifier_error: str = "",
        omnigril_exit_code: int | None = None,
    ) -> dict:
        output_head = output[:1000] if output else ""
        output_tail = output[-500:] if len(output) > 500 else output
        return {
            "type": "gemcli",
            "reward": float(reward),
            "resolved": reward >= 1.0,
            "reward_mode": "binary",
            "reward_source": reward_source,
            "verifier_error": verifier_error,
            "omnigril_exit_code": omnigril_exit_code,
            "exit_code": error_code,
            "log_head": output_head,
            "log_tail": output_tail,
            "log": output or "",
        }

    def compute_final_reward_metadata(self) -> dict:
        """Compute the episode's final reward and populate ``reward_debug``.

        For gemcli/gemswe samples we use the SOP eval_script path described in
        experiments/artifacts/cli_data_20260429/README.md: inject ``eval_script``
        as run_tests.sh now (post-trajectory), run it, and judge strictly by
        ``OMNIGRIL_EXIT_CODE``: 0 -> 1.0 reward, anything else -> 0.0.
        """
        if not self._is_gemcli:
            reward = float(self.env.compute_reward())
            self._reward_debug = {
                "type": "r2egym",
                "reward": reward,
                "resolved": reward >= 1.0,
                "reward_mode": "binary",
                "reward_source": "trusted_verifier",
                "verifier_error": "",
            }
            return self._reward_debug

        if not self.entry.get("eval_script"):
            self._reward_debug = self._build_reward_debug(
                reward=0.0,
                reward_source="eval_script",
                verifier_error="eval_script_missing",
            )
            return self._reward_debug

        # Inject eval_script now (after the agent finished its trajectory)
        # and execute it inside the container.
        self._inject_eval_script()
        output, error_code = self.env.runtime.run_tests(timeout=self.reward_timeout)
        output = output or ""

        omnigril_code = self._parse_omnigril_exit_code(output)
        if omnigril_code is None:
            reward = 0.0
            verifier_error = "omnigril_exit_code_missing"
        elif omnigril_code == 0:
            reward = 1.0
            verifier_error = ""
        else:
            reward = 0.0
            verifier_error = f"omnigril_exit_{omnigril_code}"

        self._reward_debug = self._build_reward_debug(
            reward=reward,
            reward_source="eval_script",
            output=output,
            error_code=error_code,
            verifier_error=verifier_error,
            omnigril_exit_code=omnigril_code,
        )
        return self._reward_debug

    def compute_final_reward(self):
        reward_debug = self.compute_final_reward_metadata()
        return reward_debug["reward"]

    @property
    def reward_debug(self) -> dict:
        """Return reward debug info populated by compute_final_reward()."""
        return self._reward_debug

    @staticmethod
    def _parse_omnigril_exit_code(log: str) -> int | None:
        """Extract ``OMNIGRIL_EXIT_CODE=<n>`` from eval_script output.

        Returns the integer exit code, or None if the marker is missing.
        """
        match = re.search(r"OMNIGRIL_EXIT_CODE=(\d+)", log or "")
        if match:
            return int(match.group(1))
        return None

    def step(self, action: str | Action) -> tuple[str, float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action string to execute in the environment

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if isinstance(action, str):
            action_obj: Action = Action.from_string(action)
        else:
            action_obj = action

        if not action_obj.function_name:
            return (
                "You forgot to use a function call in your response. "
                "YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.\n"
                "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.",
                0,
                False,
                {},
            )

        # RepoEnv always returns 0 reward, must be evaluated by DockerRuntime.
        obs, reward, done, info = self.env.step(action_obj)

        self.total_steps += 1
        observation = str(obs)

        # Normalize redundant error code prefix from DockerRuntime.
        # DockerRuntime.run() returns error_code as "Error: Exit code N" string;
        # Observation.__str__() then prepends "Exit code: " producing the redundant
        # "Exit code: Error: Exit code N".  Collapse to "Exit code: N".
        observation = re.sub(
            r"Exit code: Error: Exit code (\S+)",
            r"Exit code: \1",
            observation,
        )

        return observation, reward, done, info

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.env is not None:
            self.env.close()

    @staticmethod
    def from_dict(extra_info: dict | str) -> "SWEEnv":
        """Create an environment instance from JSON configuration.

        Args:
            extra_info: Dictionary containing configuration parameters.
                       The entire dict will be used as 'entry', and any keys
                       matching __init__ parameters will be extracted and passed.

        Returns:
            Initialized SWEEnv instance
        """
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(SWEEnv.__init__)
        init_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return SWEEnv(**init_params)
