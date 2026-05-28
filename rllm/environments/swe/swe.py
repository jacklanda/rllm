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

# The unmaintained-gym banner is printed via `print(..., file=sys.stderr)` from
# gym/__init__.py, not via warnings.warn — so filterwarnings cannot silence it.
# Empty the notices dict before gym imports to short-circuit the print.
try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices = {}
except ImportError:
    pass

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


_PYTEST_SUMMARY_RE = re.compile(r"=+\s*" r"(?:(?P<failed>\d+)\s+failed[,\s]*)?" r"(?:(?P<passed>\d+)\s+passed[,\s]*)?" r"(?:(?P<skipped>\d+)\s+skipped[,\s]*)?" r"(?:(?P<xfailed>\d+)\s+xfailed[,\s]*)?" r"(?:(?P<xpassed>\d+)\s+xpassed[,\s]*)?" r"(?:(?P<errors>\d+)\s+errors?[,\s]*)?" r"(?:(?P<warnings>\d+)\s+warnings?[,\s]*)?" r"(?:(?P<deselected>\d+)\s+deselected[,\s]*)?" r"in\s+[0-9.]+s")

# Signatures that mean the patched source won't even parse/apply, so no
# meaningful test signal can come out of this rollout — the verifier will
# return rc!=0 even though the agent may have been one line away from correct.
_UNAPPLICABLE_PATCH_SIGS = (
    "SyntaxError",
    "IndentationError",
    "ERRORS during collection",
    "error: patch failed",
    "error: cannot apply",
    "patch does not apply",
    "malformed patch",
    "fatal: corrupt patch",
)


def _parse_pytest_summary(log: str) -> dict | None:
    """Return {passed,failed,errors,skipped,xfailed,xpassed,total,pass_rate}
    extracted from the last pytest summary line in *log*, or None if no
    summary line is present. Robust to pytest's optional segments and handles
    the common r2egym tail ``=== 1 failed, 1066 passed, 4 skipped, 1 xfailed in 0.85s ===``.
    """
    if not log:
        return None
    last = None
    for m in _PYTEST_SUMMARY_RE.finditer(log):
        last = m
    if last is None:
        return None

    def _i(k: str) -> int:
        v = last.group(k)
        try:
            return int(v) if v is not None else 0
        except ValueError:
            return 0

    passed = _i("passed")
    failed = _i("failed")
    errors = _i("errors")
    skipped = _i("skipped")
    xfailed = _i("xfailed")
    xpassed = _i("xpassed")
    graded_total = passed + failed + errors
    if graded_total <= 0:
        pass_rate = 0.0
    else:
        pass_rate = passed / graded_total
    return {
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
        "tests_skipped": skipped,
        "tests_xfailed": xfailed,
        "tests_xpassed": xpassed,
        "tests_total": graded_total,
        "pass_rate": float(pass_rate),
    }


def _is_unapplicable_patch(log: str) -> bool:
    """True when the log shows the patch wasn't applied or the source won't
    even parse. Used to distinguish "close but wrong" from "invalid edit"."""
    if not log:
        return False
    return any(sig in log for sig in _UNAPPLICABLE_PATCH_SIGS)


def _enrich_file_editor_observation(action_obj, observation: str, runtime) -> str:
    """Make file_editor failure messages actionable instead of dead-ends.

    Cases handled (no change to the in-container tool binary):
      1. ``Multiple occurrences of old_str`` → append the first 5 match line numbers
         so the agent can disambiguate with a wider anchor.
      2. ``No match found for old_str`` → append a 3-line fuzzy candidate pulled via
         difflib so the agent can see the drift (whitespace / indent / case).
      3. Successful ``str_replace`` / ``insert`` / ``create`` → run ``py_compile`` on
         the edited path for ``*.py`` files; if it fails, revert the edit via the
         runtime and surface the compile error so the agent is warned *before*
         the eval_script blows up with IndentationError at collection time.

    Runs only when ``action_obj.function_name == 'file_editor'``; any failure inside
    this helper is caught and the original observation is returned unchanged so it
    never degrades the base code path.
    """
    try:
        if not action_obj or getattr(action_obj, "function_name", "") != "file_editor":
            return observation
        params = {}
        try:
            params = action_obj.to_dict().get("parameters", {}) or {}
        except Exception:
            params = {}
        cmd = str(params.get("command", ""))
        path = str(params.get("path", "")) if params.get("path") else ""
        obs = str(observation)

        # --- (1) Multiple occurrences → attach first-5 match line numbers. -----
        if "Multiple occurrences of old_str" in obs and path and runtime is not None:
            old_str = params.get("old_str", "")
            if old_str and isinstance(old_str, str):
                import shlex

                key = old_str.splitlines()[0] if old_str else ""
                if key and len(key) >= 3:
                    q = shlex.quote(key)
                    try:
                        out, _rc = runtime.run(f"grep -n -F -- {q} {shlex.quote(path)} | head -5", timeout=10)
                        lines = [ln for ln in str(out).splitlines() if ln.strip()]
                        if lines:
                            obs += ("\n\n[editor-hint] First match line(s) in {p}: {ls}.\n" "Pick a unique anchor by extending old_str with the surrounding line(s).").format(p=path, ls=", ".join(ln.split(":", 1)[0] for ln in lines))
                    except Exception:
                        pass
            return obs

        # --- (2) No match found → attach fuzzy candidate. ----------------------
        if "No match found for old_str" in obs and path and runtime is not None:
            old_str = params.get("old_str", "")
            if isinstance(old_str, str) and old_str.strip():
                try:
                    import difflib

                    out, _rc = runtime.run(
                        f"sed -n '1,4000p' {__import__('shlex').quote(path)}",
                        timeout=10,
                    )
                    haystack = str(out).splitlines()
                    needle = old_str.splitlines()
                    cand = difflib.get_close_matches(needle[0] if needle else "", haystack, n=1, cutoff=0.6) if needle else []
                    if cand:
                        obs += ("\n\n[editor-hint] Closest line in {p}: {c!r}.\n" "Fix whitespace / capitalization drift, then retry with the exact text.").format(p=path, c=cand[0][:200])
                except Exception:
                    pass
            return obs

        # --- (3) Post-edit py_compile gate. ------------------------------------
        if (
            cmd in {"str_replace", "insert", "create"} and path.endswith(".py") and runtime is not None and "Error" not in obs.split("\n", 1)[0]  # tool itself reported success
        ):
            import shlex

            q = shlex.quote(path)
            try:
                out, _rc = runtime.run(
                    f'python3 -c "import py_compile,sys; py_compile.compile({q!r}, doraise=True)" 2>&1 || true',
                    timeout=20,
                )
                out_s = str(out)
                if ("IndentationError" in out_s) or ("SyntaxError" in out_s):
                    # Revert the change: r2egym file_editor writes an "_original"
                    # backup for str_replace / insert. If the backup is absent
                    # (e.g. `create`), fall back to `git checkout --`.
                    runtime.run(
                        f"(test -f {q}.__rllm_bak__ && mv -f {q}.__rllm_bak__ {q}) " f"|| (cd /testbed && git checkout -- {q} 2>/dev/null) || true",
                        timeout=10,
                    )
                    obs += "\n\n[editor-hint] Post-edit py_compile FAILED and the edit has been reverted.\n" f"Compiler said:\n{out_s.strip()[-400:]}\n" "View the file, then retry with a correct edit."
            except Exception:
                pass
    except Exception:
        return observation
    return obs


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
        self._viewed_files: set[str] = set()
        # Dump a container-env fingerprint on every reset so tool-env regressions are grep-able.
        self._log_container_fingerprint()
        # Gather environment context (CWD + file tree) for agent spatial awareness.
        env_context = self._get_env_context()
        cwd = self._get_cwd()
        return self.env.get_task_instruction(), {"env_context": env_context, "cwd": cwd}

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
        cmd = 'python3 -c \'import sys,chardet; print("python3:",sys.executable,"chardet:",chardet.__version__)\' 2>&1; ' "ls -la /usr/local/bin/file_editor 2>&1 | head -1"
        out, _ = self.env.runtime.run(cmd, timeout=15)
        logger.info("container_fingerprint: %s", (out or "").strip())

    def _get_cwd(self) -> str:
        """Get the current working directory inside the container."""
        if self.env is None or self.env.runtime is None:
            return "/testbed"
        out, _ = self.env.runtime.run("pwd", timeout=5)
        return (out or "/testbed").strip()

    def _get_env_context(self) -> str:
        """Get environment context (CWD + file tree) for agent spatial awareness.

        Returns a formatted string with the current working directory and a
        depth-limited file tree of the repository, suitable for injection into
        the agent's observation to prevent blind file operations.
        """
        if self.env is None or self.env.runtime is None:
            return ""
        cwd_output, _ = self.env.runtime.run("pwd", timeout=5)
        cwd = (cwd_output or "/testbed").strip()
        tree_output, _ = self.env.runtime.run(
            "find . -maxdepth 2 -not -path '*/\\.*' -not -path '*/__pycache__/*' " "-not -path '*/node_modules/*' -not -path '*/.git/*' | sort | head -80",
            timeout=10,
        )
        tree = (tree_output or "").strip()
        return f"Current Working Directory: {cwd}\nRepository Structure (depth=2):\n{tree}"

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

        sed_cmds = " && ".join(f"sed -i '1s|^#!.*python.*$|#!/usr/bin/env python3|' /usr/local/bin/{name}" for name in tool_names)
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
            "git config advice.objectNameWarning false 2>/dev/null; " "git config advice.ambiguousFetchRefspec false 2>/dev/null; " "git config core.warnAmbiguousRefs false 2>/dev/null",
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
            ("ensurepip+pip", "python3 -m ensurepip --default-pip >/dev/null 2>&1; " f"python3 -m pip install --quiet --disable-pip-version-check {deps_arg}"),
            ("pip3", f"pip3 install --quiet --disable-pip-version-check {deps_arg}"),
            ("apt-get", "apt-get update -qq >/dev/null 2>&1 && " "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq " "python3-chardet python3-coverage >/dev/null 2>&1"),
        ]
        install_log = []
        for label, cmd in attempts:
            output, error_code = self.env.runtime.run(cmd, timeout=120)
            install_log.append(f"[{label}] ec={error_code} out={(output or '')[:200]}")
            smoke_out, _ = self.env.runtime.run(smoke_cmd, timeout=15)
            if "tool_deps_ok" in (smoke_out or ""):
                return

        logger.warning(
            "Tool-dependency smoke test FAILED after all install strategies; " "file_editor will crash with ModuleNotFoundError. attempts=%s last_smoke=%s",
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
        pytest_stats: dict | None = None,
        patch_applicable: bool | None = None,
        reward_mode: str = "binary",
    ) -> dict:
        output_head = output[:1000] if output else ""
        output_tail = output[-500:] if len(output) > 500 else output
        debug = {
            "type": "gemcli",
            "reward": float(reward),
            "resolved": reward >= 1.0,
            "reward_mode": reward_mode,
            "reward_source": reward_source,
            "verifier_error": verifier_error,
            "omnigril_exit_code": omnigril_exit_code,
            "exit_code": error_code,
            "log_head": output_head,
            "log_tail": output_tail,
            "log": output or "",
        }
        if pytest_stats:
            # Flattened into reward_debug so agent_ppo_trainer's existing
            # metrics path (tests_passed/failed/total/pass_rate) auto-populates.
            debug.update(pytest_stats)
        if patch_applicable is not None:
            debug["patch_applicable"] = bool(patch_applicable)
        return debug

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
        pytest_stats = _parse_pytest_summary(output)
        patch_unapplicable = _is_unapplicable_patch(output)
        # Patch is considered applicable if pytest ran at all (we parsed a
        # summary line) AND no syntax/apply-failure signatures are present.
        patch_applicable = bool(pytest_stats) and not patch_unapplicable

        shaped = os.environ.get("RLLM_SWE_SHAPED_REWARD", "0") == "1"
        reward_mode = "shaped" if shaped else "binary"

        if omnigril_code is None:
            reward = 0.0
            verifier_error = "omnigril_exit_code_missing"
        elif omnigril_code == 0:
            reward = 1.0
            verifier_error = ""
        else:
            # rc != 0. Split the "unapplicable patch" class from "tests ran
            # and some failed" so downstream logging / filtering can tell
            # "agent broke the source" apart from "agent was close".
            if patch_unapplicable or not pytest_stats:
                reward = 0.0
                verifier_error = "unapplicable_patch" if patch_unapplicable else f"omnigril_exit_{omnigril_code}"
            elif shaped:
                # Partial credit capped under 0.5 so a true pass (binary 1.0)
                # remains strictly more valuable than any partial rollout.
                reward = float(min(pytest_stats["pass_rate"] * 0.5, 0.49))
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
            pytest_stats=pytest_stats,
            patch_applicable=(patch_applicable if omnigril_code is not None else None),
            reward_mode=reward_mode,
        )
        return self._reward_debug

    def compute_final_reward(self):
        reward_debug = self.compute_final_reward_metadata()
        return reward_debug["reward"]

    # ------------------------------------------------------------------
    # Gold-patch sanity check
    # ------------------------------------------------------------------

    def apply_gold_patch_in_container(self) -> tuple[bool, str]:
        """Apply ``self.entry['gold_patch']`` inside the already-running container.

        Tries ``git apply -p1 -v`` first and falls back to
        ``patch --batch --fuzz=5 -p1``, matching
        ``experiments/artifacts/cli_data_20260429/run_eval_in_container.py``.
        Returns ``(applied, combined_output)``.
        """
        gold_patch = self.entry.get("gold_patch") or ""
        if not gold_patch.strip():
            return False, "gold_patch field is empty or missing in entry"
        if self.env is None or getattr(self.env, "runtime", None) is None:
            return False, "container runtime not available (call reset() first)"

        patch_path = "/tmp/rllm_gold_patch.diff"
        try:
            self._copy_content_to_container(gold_patch, patch_path, suffix=".diff", chmod=False)
        except Exception as exc:
            return False, f"copy_to_container failed: {exc}"

        # DockerRuntime.run wraps the command in `timeout N <cmd>` and runs via
        # `/bin/sh -c`. Using `cd /testbed && ...` breaks because `timeout` cannot
        # exec the shell builtin `cd` (exit 127), short-circuiting the real
        # command. Pass the cwd via `workdir=` instead, matching
        # experiments/artifacts/cli_data_20260429/run_eval_in_container.py which
        # uses container.exec_run(workdir="/testbed").
        def _ok(rc) -> bool:
            return str(rc).strip() == "0"

        primary = f"git apply -p1 -v {patch_path} 2>&1"
        out1, rc1 = self.env.runtime.run(primary, timeout=60, workdir="/testbed")
        out1 = out1 or ""
        if _ok(rc1):
            return True, out1

        fallback = f"patch --batch --fuzz=5 -p1 -i {patch_path} 2>&1"
        out2, rc2 = self.env.runtime.run(fallback, timeout=60, workdir="/testbed")
        combined = f"[git apply rc={rc1}]\n{out1}\n--- patch fallback ---\n[patch rc={rc2}]\n{out2 or ''}"
        if _ok(rc2):
            return True, combined
        return False, combined

    def sanity_check_gold_patch(self) -> dict:
        """Bypass the agent and evaluate the task as if the gold patch were submitted.

        Steps (assumes ``reset()`` has already been called so the container is
        live at ``base_commit``):
          1. Apply ``entry['gold_patch']`` inside the container.
          2. Run the standard reward pipeline (``compute_final_reward_metadata``),
             which injects ``eval_script`` as run_tests.sh, executes it, and
             parses ``OMNIGRIL_EXIT_CODE``.

        A healthy training pipeline should return ``reward == 1.0`` for every
        CLI sample with a valid gold_patch + eval_script. Anything less flags a
        verifier / container-workflow bug independent of the agent policy.
        """
        if not self._is_gemcli:
            self._reward_debug = {
                "type": "gold_patch_sanity",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "gold_patch_sanity",
                "verifier_error": "not_a_gemcli_sample",
                "sanity_check": True,
                "gold_patch_applied": False,
            }
            return self._reward_debug

        patch_applied, patch_output = self.apply_gold_patch_in_container()
        patch_tail = (patch_output or "")[-500:]

        if not patch_applied:
            self._reward_debug = {
                "type": "gold_patch_sanity",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "gold_patch_sanity",
                "verifier_error": "gold_patch_apply_failed",
                "sanity_check": True,
                "gold_patch_applied": False,
                "gold_patch_apply_output_tail": patch_tail,
            }
            return self._reward_debug

        reward_debug = self.compute_final_reward_metadata()
        reward_debug = dict(reward_debug) if isinstance(reward_debug, dict) else {"reward": 0.0}
        reward_debug["type"] = "gold_patch_sanity"
        reward_debug["reward_source"] = "gold_patch_sanity"
        reward_debug["sanity_check"] = True
        reward_debug["gold_patch_applied"] = True
        reward_debug["gold_patch_apply_output_tail"] = patch_tail
        self._reward_debug = reward_debug
        return reward_debug

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
                "You forgot to use a function call in your response. " "YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.\n" "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.",
                0,
                False,
                {},
            )

        # Auto-resolve relative paths for file_editor calls
        if action_obj.function_name == "file_editor":
            params = action_obj.parameters or {}
            path_val = params.get("path", "")
            if path_val and not path_val.startswith("/"):
                cwd = self._get_cwd()
                params["path"] = f"{cwd}/{path_val}"
                action_obj = Action(function_name=action_obj.function_name, parameters=params)

        # Track viewed files and warn on blind edits
        _edit_without_view = False
        if action_obj.function_name == "file_editor":
            params = action_obj.parameters or {}
            cmd = params.get("command", "")
            path_val = params.get("path", "")
            if cmd == "view" and path_val:
                self._viewed_files.add(path_val)
            elif cmd == "str_replace" and path_val and path_val not in getattr(self, "_viewed_files", set()):
                _edit_without_view = True

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

        # File-editor ergonomics: enrich failure observations + compile-gate edits.
        # Gated by env var so a regression can be rolled back without a redeploy.
        if os.environ.get("RLLM_FILE_EDITOR_ERGONOMICS", "1") != "0":
            try:
                runtime = getattr(self.env, "runtime", None) if self.env is not None else None
                observation = _enrich_file_editor_observation(action_obj, observation, runtime)
            except Exception:
                pass

        # Path-error hint: when a file operation fails due to a missing path,
        # suggest exploration commands so the agent doesn't blindly retry.
        if "does not exist" in observation or "No such file or directory" in observation:
            observation += "\n\nHint: The specified path does not exist. " "Try `find . -name '<filename>'` to locate the file, " "or `ls` to see the current directory contents."

        # Warn when str_replace targets a file that hasn't been viewed yet
        if _edit_without_view and "ERROR" not in observation.split("\n", 1)[0]:
            observation += "\n\n[editor-hint] You are editing a file you haven't viewed yet. " "Use `file_editor(view, path=...)` first to see the current content and avoid mismatches."

        # Inject CWD into info so the agent always knows its location.
        cwd = self._get_cwd()
        info["cwd"] = cwd

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
