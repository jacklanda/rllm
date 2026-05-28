"""Endless Terminals (ET) environment for rllm.

ET tasks are self-contained CLI tasks: each ships a pre-built Docker image
(name = ``gemcli/<task_id>``), an instruction, an initial-state pytest, a
final-state pytest, a ``tests/test.sh`` entry that writes the reward to
``/logs/verifier/reward.txt``, and a reference solution. The full schema is
documented in ``experiments/artifacts/endless_terminals/ET_TASKS_SCHEMA_AND_RUNBOOK.md``.

This env assumes images are already published to the Docker daemon pointed at
by ``$DOCKER_HOST`` — it does NOT build images from the dockerfile string.
Train scripts wire the connectivity check via
``rllm.trainer.verl.agent_ppo_trainer._check_docker_connectivity``.

The interaction protocol matches SWEAgent (function-call XML emitted by the
LLM, parsed by ``r2egym.agenthub.action.Action``):
  * ``execute_bash(command=...)``         — run a shell command in /home/user
  * ``str_replace_editor(command=..., path=..., ...)`` — view/edit files
  * ``submit()``                          — finish; triggers the verifier

Reward is binary: 1.0 iff ``/logs/verifier/reward.txt`` reads ``"1"`` after
``tests/test.sh`` runs; otherwise 0.0.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tarfile
import threading
import time
import uuid
from typing import Any

from rllm.environments.base.base_env import BaseEnv

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:  # pragma: no cover - r2egym is a hard dep elsewhere
    SWEAction = None

logger = logging.getLogger(__name__)


# Default cwd inside the container. The ET runbook fixes /home/user as the
# task workdir; instruction text always references absolute paths under it.
DEFAULT_CWD = "/home/user"

# Reward file path inside the container (per ET runbook).
REWARD_FILE = "/logs/verifier/reward.txt"

# Tests directory inside the container; populated at reward time.
TESTS_DIR = "/tests"

# Where the str_replace_editor / execute_bash / submit tool scripts live.
TOOLS_DIR = "/usr/local/bin"

_CLIENT_LOCK = threading.Lock()
_SHARED_DOCKER_CLIENT = None  # type: ignore[var-annotated]


def _get_docker_client():
    """Return a process-wide docker.DockerClient.

    Reuses ``r2egym.agenthub.runtime.docker.DockerRuntime._shared_docker_client``
    if already seeded by ``agent_ppo_trainer._check_docker_connectivity`` so we
    share its connection pool. Otherwise builds a new client from $DOCKER_HOST.
    """
    global _SHARED_DOCKER_CLIENT
    if _SHARED_DOCKER_CLIENT is not None:
        return _SHARED_DOCKER_CLIENT

    # Prefer the r2egym-seeded client (large pool, configured by the trainer).
    try:
        from r2egym.agenthub.runtime.docker import DockerRuntime

        if DockerRuntime._shared_docker_client is not None:
            with _CLIENT_LOCK:
                if _SHARED_DOCKER_CLIENT is None:
                    _SHARED_DOCKER_CLIENT = DockerRuntime._shared_docker_client
            return _SHARED_DOCKER_CLIENT
    except Exception:  # pragma: no cover
        pass

    import docker

    with _CLIENT_LOCK:
        if _SHARED_DOCKER_CLIENT is None:
            base_url = os.environ.get("DOCKER_HOST") or None
            api_version = os.environ.get("DOCKER_API_VERSION", "auto")
            if base_url:
                _SHARED_DOCKER_CLIENT = docker.DockerClient(
                    base_url=base_url,
                    timeout=120,
                    version=api_version,
                    max_pool_size=64,
                    num_pools=16,
                )
            else:
                _SHARED_DOCKER_CLIENT = docker.from_env(timeout=120)
    return _SHARED_DOCKER_CLIENT


def _put_text_to_container(container, container_path: str, content: str, mode: int = 0o644):
    """Write a UTF-8 string to a file inside the container via put_archive.

    container_path must be an absolute path. The parent directory is assumed
    to already exist; callers should ``mkdir -p`` first if unsure.
    """
    parent_dir = os.path.dirname(container_path) or "/"
    name = os.path.basename(container_path)
    data = content.encode("utf-8")
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = mode
        tar.addfile(info, io.BytesIO(data))
    tar_stream.seek(0)
    container.put_archive(parent_dir, tar_stream.read())


def _exec(container, cmd: list[str] | str, *, workdir: str | None = None, timeout: int = 60, environment: dict | None = None) -> tuple[int, str]:
    """Run a command in the container with a hard timeout, return (exit_code, output_text).

    ``timeout`` is enforced inside the container by wrapping with `timeout(1)` so
    that a hung exec cannot hold the agent slot open. Output is utf-8 decoded
    with ``errors='replace'``.
    """
    if isinstance(cmd, list):
        # Build a `timeout N <quoted cmd>` invocation via /bin/sh -c.
        import shlex

        joined = " ".join(shlex.quote(p) for p in cmd)
    else:
        joined = cmd
    wrapped = f"timeout {int(timeout)} sh -c {__import__('shlex').quote(joined)}"
    res = container.exec_run(
        cmd=["/bin/sh", "-c", wrapped],
        workdir=workdir or DEFAULT_CWD,
        stdout=True,
        stderr=True,
        environment=environment or {},
        demux=False,
    )
    out = res.output or b""
    if isinstance(out, tuple):  # demux fallback
        out = (out[0] or b"") + (out[1] or b"")
    text = out.decode("utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m|\r", "", text)
    return int(res.exit_code if res.exit_code is not None else -1), text


_SCAFFOLD_TOOL_FILES_CACHE: list[str] | None = None


def _sweagent_tool_files() -> list[str]:
    """Return absolute paths to the SWEAgent tool scripts shipped by r2egym.

    These are copied into the container's /usr/local/bin so the agent's
    function calls are translated 1:1 into in-container script invocations.
    """
    global _SCAFFOLD_TOOL_FILES_CACHE
    if _SCAFFOLD_TOOL_FILES_CACHE is not None:
        return _SCAFFOLD_TOOL_FILES_CACHE
    import r2egym

    base = os.path.dirname(r2egym.__file__)
    files = [
        os.path.join(base, "agenthub/tools/str_replace_editor.py"),
        os.path.join(base, "agenthub/tools/execute_bash.py"),
        os.path.join(base, "agenthub/tools/submit.py"),
    ]
    _SCAFFOLD_TOOL_FILES_CACHE = files
    return files


# pytest's terse summary line is the most reliable cross-version anchor; we
# accept either "5 passed" or "5 passed, 1 failed in 1.23s". When the verifier
# script swallows pytest output we return (None, None) so the partial-reward
# branch falls back to the binary verdict.
_PYTEST_PASSED_RE = re.compile(r"\b(\d+)\s+passed\b")
_PYTEST_FAILED_RE = re.compile(r"\b(\d+)\s+(?:failed|errors?)\b")


def _parse_pytest_counts(text: str) -> tuple[int | None, int]:
    if not text:
        return None, 0
    passed_m = _PYTEST_PASSED_RE.search(text)
    failed_m = _PYTEST_FAILED_RE.search(text)
    if not passed_m and not failed_m:
        return None, 0
    passed = int(passed_m.group(1)) if passed_m else 0
    failed = int(failed_m.group(1)) if failed_m else 0
    return passed, failed


class ETEnv(BaseEnv):
    """Endless Terminals environment.

    One container per env instance, lazily created on first ``reset()``.
    The agent emits SWEAgent-style function-call XML; ``step()`` parses it
    via ``Action.from_string`` and forwards the bash/file_editor invocation
    into the container. ``submit`` ends the episode (done=True). Reward is
    computed at episode end by ``compute_final_reward_metadata()``.
    """

    SUPPORTED_FUNCTIONS = ("execute_bash", "str_replace_editor", "file_editor", "submit", "finish")

    def __init__(
        self,
        entry: dict,
        step_timeout: int = 90,
        reward_timeout: int | None = None,
        verbose: bool = False,
        partial_reward: bool = False,
        finish_without_evidence_penalty: float = 0.0,
    ):
        if SWEAction is None:
            raise RuntimeError("r2egym is required for ETEnv (Action parsing).")
        if not isinstance(entry, dict):
            raise TypeError(f"ETEnv entry must be a dict, got {type(entry)}")

        required = ("task_id", "docker_image", "instruction", "test_script", "final_state_test")
        missing = [k for k in required if k not in entry]
        if missing:
            raise ValueError(f"ETEnv entry missing keys: {missing}")

        self.entry = entry
        self.task_id: str = entry["task_id"]
        self.docker_image: str = entry["docker_image"]
        self.instruction: str = entry["instruction"]
        self.dockerfile: str = entry.get("dockerfile", "")
        self.initial_state_test: str = entry.get("initial_state_test", "")
        self.test_script: str = entry["test_script"]
        self.final_state_test: str = entry["final_state_test"]
        self.solution: str = entry.get("solution", "")

        self.cpus: int = int(entry.get("cpus", 1) or 1)
        self.memory_mb: int = int(entry.get("memory_mb", 2048) or 2048)
        self.agent_timeout_sec: float = float(entry.get("agent_timeout_sec", 300.0) or 300.0)
        self.verifier_timeout_sec: float = float(entry.get("verifier_timeout_sec", 300.0) or 300.0)

        self.step_timeout = int(step_timeout)
        # Default reward timeout to the task-declared verifier timeout, but allow
        # an explicit override (e.g., a large global ceiling for big test_scripts).
        self.reward_timeout = int(reward_timeout) if reward_timeout else int(self.verifier_timeout_sec)
        self.verbose = verbose

        # Partial-reward / shaping toggles. Defaults preserve binary 0/1
        # behaviour for backward compat; FUSED ET grpo configs can opt in.
        self.partial_reward: bool = bool(partial_reward)
        self.finish_without_evidence_penalty: float = float(finish_without_evidence_penalty)

        self.client = None
        self.container = None
        self.container_name: str = ""
        self.total_steps: int = 0
        self._initial_state_ok: bool | None = None
        self._reward_debug: dict = {}
        self._tools_installed: bool = False
        self._closed: bool = False
        # Track whether the agent ever issued a "verification-style" read
        # action (cat / ls / head / tail / stat / file_editor view / test -f
        # / find -name). Used by the optional finish-without-evidence
        # shaping path.
        self._evidence_actions: int = 0

    # ------------------------------------------------------------------
    # Reset / step / close
    # ------------------------------------------------------------------

    def reset(self, task: dict | None = None) -> tuple[str, dict]:
        """Start the container, run initial-state pytest, return (instruction, info).

        ``task`` is accepted for Workflow.reset() compat but ignored — the env
        is bound to a single task at construction time via ``from_dict``.
        """
        self.client = _get_docker_client()
        try:
            self.client.images.get(self.docker_image)
        except Exception as exc:
            raise RuntimeError(f"Docker image not found on daemon: {self.docker_image!r} " f"(DOCKER_HOST={os.environ.get('DOCKER_HOST', '<local>')}). " f"Underlying error: {exc!r}") from exc

        # Container name needs to be unique per rollout. UUID prefix keeps it
        # short enough for Docker's 64-char limit even for long task_ids.
        suffix = uuid.uuid4().hex[:10]
        # Sanitize task_id: container names must match [a-zA-Z0-9_.-]+
        safe_tid = re.sub(r"[^a-zA-Z0-9_.-]", "-", self.task_id)[:40]
        self.container_name = f"et-{safe_tid}-{suffix}"

        run_kwargs = dict(
            image=self.docker_image,
            command=["sleep", "infinity"],
            name=self.container_name,
            detach=True,
            tty=False,
            stdin_open=False,
            # network_mode=host avoids veth/bridge attachment which saturates
            # docker0's FDB at >1k parallel rollouts (same rationale as r2egym).
            network_mode="host",
            mem_limit=f"{self.memory_mb}m",
            # Note: cpu_count is a Windows-only API; for Linux daemons we set
            # nano_cpus = cpus * 1e9 to express a fractional CPU quota.
            nano_cpus=int(self.cpus * 1e9),
            # auto_remove=False so close() can clean up explicitly even if the
            # daemon is slow to react; we remove on close().
            auto_remove=False,
            labels={"rllm": "et", "rllm_task_id": self.task_id},
        )
        try:
            self.container = self.client.containers.run(**run_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to start ET container for image={self.docker_image} " f"name={self.container_name}: {exc!r}") from exc

        # Reload to get the running state populated.
        try:
            self.container.reload()
        except Exception:
            pass

        # Run the initial-state pytest (informational; do not block the agent).
        self._initial_state_ok = self._run_initial_state_test()
        if not self._initial_state_ok:
            logger.warning(
                "ET initial_state_test FAILED for task_id=%s image=%s; agent will still run",
                self.task_id,
                self.docker_image,
            )

        # Lazily install the SWEAgent tool scripts on first reset (so step()
        # can resolve `execute_bash` / `str_replace_editor` inside the container).
        self._install_tools()

        self.total_steps = 0
        info = {
            "task_id": self.task_id,
            "docker_image": self.docker_image,
            "container_name": self.container_name,
            "cwd": DEFAULT_CWD,
            "initial_state_ok": bool(self._initial_state_ok),
        }
        return self.instruction, info

    def step(self, action: Any) -> tuple[str, float, bool, dict]:
        """Run one agent action against the container.

        Accepts either a raw XML string from the LLM or an ``Action`` object
        (the workflow forwards whatever ``agent.update_from_model`` returns).
        """
        if self.container is None:
            raise RuntimeError("ETEnv.step() called before reset()")

        if isinstance(action, str):
            action_obj = SWEAction.from_string(action)
        elif hasattr(action, "function_name"):
            action_obj = action
        else:
            # rllm.agents.Action wraps the XML string in a `.action` field.
            inner = getattr(action, "action", None)
            action_obj = SWEAction.from_string(inner) if isinstance(inner, str) else SWEAction(function_name="", parameters={})

        fn = getattr(action_obj, "function_name", "") or ""
        params = getattr(action_obj, "parameters", {}) or {}

        if not fn:
            return (
                "You forgot to use a function call in your response. " "YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.\n" "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.",
                0.0,
                False,
                {"task_id": self.task_id, "cwd": DEFAULT_CWD},
            )

        if fn in ("submit", "finish"):
            # Episode ends; reward is computed by the engine via
            # compute_final_reward_metadata(). Stash evidence-flag for the
            # reward path to read; cannot be applied here because the binary
            # reward is computed later.
            self._reward_debug["_finish_evidence_count"] = int(self._evidence_actions)
            return ("<<<Finished>>>", 0.0, True, {"task_id": self.task_id, "cwd": DEFAULT_CWD})

        # Tool-name drift: 4B-Thinking has seen ``file_editor`` in pretraining
        # (r2egym protocol) and emits it ~7% of the time even though the ET
        # prompt only documents ``str_replace_editor``. The two scripts are
        # interchangeable for the view/create/str_replace/insert subset that
        # ET tasks use, so alias here rather than burning a step on
        # "Unknown function".
        if fn == "file_editor":
            fn = "str_replace_editor"

        if fn not in self.SUPPORTED_FUNCTIONS:
            return (
                f"Unknown function: {fn}. Supported: execute_bash, str_replace_editor, submit.",
                0.0,
                False,
                {"task_id": self.task_id, "cwd": DEFAULT_CWD},
            )

        # Translate the function call into an in-container shell invocation.
        # IMPORTANT: r2egym's Action.to_bashcmd() emits ``execute_bash --cmd
        # "..."`` and ``str_replace_editor --command ... --path ...`` keyword
        # form, but the r2egym tool *scripts* shipped to /usr/local/bin take
        # the first arg positionally. Build the invocation explicitly so the
        # mapping stays correct regardless of upstream-r2egym drift.
        cmd_argv = self._build_tool_argv(fn, params)
        if cmd_argv is None:
            return (
                f"Could not build command for {fn} from parameters {params}.",
                0.0,
                False,
                {"task_id": self.task_id, "cwd": DEFAULT_CWD},
            )

        # Evidence-action accounting (cheap, conservative): does this step
        # *read* state rather than mutate it? If yes, count it toward the
        # "finish without evidence" guard. View-only file_editor calls and
        # read-flavored bash one-liners qualify; redirections / writes / pip
        # installs / chmods do not.
        if fn in ("str_replace_editor", "file_editor"):
            if (params.get("command") or "") == "view":
                self._evidence_actions += 1
        elif fn == "execute_bash":
            raw = (params.get("cmd") or params.get("command") or "").strip()
            head = raw.split(maxsplit=1)[0] if raw else ""
            head = head.split("/")[-1]  # strip any /usr/bin/ prefix
            if head in {"cat", "ls", "head", "tail", "stat", "find", "grep", "wc", "file", "test"}:
                # Exclude when redirecting output to a file (mutates state).
                if ">" not in raw and ">>" not in raw:
                    self._evidence_actions += 1

        rc, output = _exec(self.container, cmd_argv, workdir=DEFAULT_CWD, timeout=self.step_timeout)
        observation = output if output else ""
        if rc == 124:
            observation = f"The command took too long to execute (>{self.step_timeout}s). " f"Try a smaller batch or use a non-blocking variant.\n{observation}"

        # Path-error hint, mirroring SWEEnv behavior.
        if "No such file or directory" in observation or "does not exist" in observation:
            observation += "\n\nHint: The specified path does not exist. " "Try `find . -name '<filename>'` to locate the file, " "or `ls` to see the current directory contents."

        self.total_steps += 1
        info = {"task_id": self.task_id, "cwd": DEFAULT_CWD, "exit_code": rc}
        return observation, 0.0, False, info

    @staticmethod
    def _build_tool_argv(fn: str, params: dict) -> list[str] | None:
        """Build the in-container argv for a SWEAgent function call.

        The r2egym tool scripts take their primary action as a positional arg
        (``command`` for ``execute_bash`` and ``str_replace_editor``) and the
        rest as ``--key value`` flags.

        This deliberately diverges from ``Action.to_bashcmd()`` (which emits
        ``--cmd``) — the upstream serializer doesn't match the upstream
        scripts, and ETEnv ships those scripts verbatim into the container.
        """
        if fn == "execute_bash":
            cmd = params.get("cmd") or params.get("command") or ""
            if not cmd:
                return None
            return ["execute_bash", str(cmd)]

        if fn == "str_replace_editor":
            command = params.get("command") or ""
            if not command:
                return None
            argv = ["str_replace_editor", str(command)]
            for key in ("path", "file_text", "old_str", "new_str", "insert_line", "view_range"):
                val = params.get(key)
                if val is None or val == "":
                    continue
                argv.append(f"--{key}")
                argv.append(str(val))
            return argv

        return None

    def close(self) -> None:
        """Stop and remove the container. Best-effort; never raises.

        Honors ``RLLM_ET_KEEP_CONTAINER=1`` for debugging.
        """
        if self._closed:
            return
        self._closed = True
        if self.container is None:
            return
        if os.environ.get("RLLM_ET_KEEP_CONTAINER", "0") == "1":
            logger.info("RLLM_ET_KEEP_CONTAINER=1: leaving container %s", self.container_name)
            return
        try:
            self.container.stop(timeout=2)
        except Exception:
            pass
        try:
            self.container.remove(force=True)
        except Exception:
            pass
        self.container = None

    # ------------------------------------------------------------------
    # Helpers: tool install, initial state, reward
    # ------------------------------------------------------------------

    def _install_tools(self) -> None:
        """Copy SWEAgent tool scripts into /usr/local/bin and rewrite their shebangs.

        ET base images are minimal Ubuntu/Python and do NOT ship the file_editor /
        execute_bash / submit binaries that the SWEAgent function-call protocol
        expects. We copy the python tool scripts shipped by r2egym and rewrite
        their shebang to ``#!/usr/bin/env python3`` so they run regardless of
        whether the image has /root/.venv. Idempotent.
        """
        if self._tools_installed or self.container is None:
            return
        try:
            tool_files = _sweagent_tool_files()
        except Exception as exc:
            logger.warning("Could not locate SWEAgent tool scripts: %s", exc)
            return

        # mkdir + write each script.
        rc, _ = _exec(self.container, ["mkdir", "-p", TOOLS_DIR], timeout=10)
        if rc != 0:
            logger.warning("ETEnv: mkdir %s failed (rc=%s) for task=%s", TOOLS_DIR, rc, self.task_id)
        for src in tool_files:
            try:
                with open(src, encoding="utf-8") as f:
                    content = f.read()
            except Exception as exc:
                logger.warning("ETEnv: failed to read tool script %s: %s", src, exc)
                continue
            # Rewrite shebang for portability.
            content = re.sub(r"^#![^\n]*\n", "#!/usr/bin/env python3\n", content, count=1)
            # ET base images are minimal: r2egym's str_replace_editor.py imports
            # chardet unconditionally, but most ET dockerfiles do NOT install it
            # (training-step-1 dump: 96 chardet ImportError observations across
            # 128 trajectories, all on file_editor / str_replace_editor calls).
            # Soft-import: if chardet is missing, fall back to utf-8 — every ET
            # task ships utf-8 text fixtures, so this is lossless in practice.
            content = content.replace(
                "import chardet\n",
                "try:\n    import chardet\nexcept ImportError:  # ET base images may lack chardet\n    chardet = None\n",
            )
            content = content.replace(
                'encoding = chardet.detect(path.read_bytes())["encoding"]',
                'encoding = chardet.detect(path.read_bytes())["encoding"] if chardet is not None else None',
            )
            name = os.path.basename(src)  # e.g. "execute_bash.py"
            stem = os.path.splitext(name)[0]
            dest = f"{TOOLS_DIR}/{stem}"
            try:
                _put_text_to_container(self.container, dest, content, mode=0o755)
            except Exception as exc:
                logger.warning("ETEnv: put_archive failed for %s -> %s: %s", src, dest, exc)
                continue
            _exec(self.container, ["chmod", "+x", dest], timeout=10)
            # Provide ``file_editor`` as a name-equivalent symlink to
            # ``str_replace_editor`` so tool calls under either name resolve to
            # the same in-container script. The system prompt advertises
            # ``file_editor`` as canonical; the trainer-side aliasing in step()
            # remains as a defence-in-depth.
            if stem == "str_replace_editor":
                _exec(
                    self.container,
                    ["ln", "-sf", dest, f"{TOOLS_DIR}/file_editor"],
                    timeout=10,
                )

        # Smoke-test that python3 is callable. ET dockerfiles install python3 +
        # pytest, so this should always succeed; warn loudly if not.
        rc, out = _exec(self.container, ["python3", "--version"], timeout=10)
        if rc != 0:
            logger.warning("ETEnv: python3 --version failed in container %s rc=%s out=%s", self.container_name, rc, out[:200])
        self._tools_installed = True

    def _run_initial_state_test(self) -> bool:
        """Write the initial-state pytest into the container and run it.

        Returns True on success. Failures are logged but do not block the
        episode — the task may still be solvable, and curriculum_filter can
        quarantine via ``info["initial_state_ok"]``.
        """
        if not self.initial_state_test or self.container is None:
            return True
        try:
            _exec(self.container, ["mkdir", "-p", "/tmp"], timeout=10)
            _put_text_to_container(self.container, "/tmp/test_initial_state.py", self.initial_state_test, mode=0o644)
        except Exception as exc:
            logger.warning("ETEnv: failed to copy initial_state_test for task=%s: %s", self.task_id, exc)
            return False
        rc, out = _exec(
            self.container,
            ["python3", "-m", "pytest", "/tmp/test_initial_state.py", "-v", "--no-header", "-q"],
            timeout=int(self.verifier_timeout_sec),
        )
        if rc != 0 and self.verbose:
            logger.info("initial_state_test rc=%s tail=%s", rc, out[-400:])
        return rc == 0

    def compute_final_reward_metadata(self) -> dict:
        """Run the verifier and return the metrics dict.

        Steps (per ET runbook):
          1. Write tests/test_final_state.py and tests/test.sh into /tests/.
          2. Ensure /logs/verifier exists.
          3. Run `bash /tests/test.sh` with verifier_timeout_sec.
          4. Read /logs/verifier/reward.txt; reward = 1.0 iff content == "1".
        """
        if self.container is None:
            self._reward_debug = {
                "type": "endless_terminals",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "verifier_reward_txt",
                "verifier_error": "container_missing",
            }
            return self._reward_debug

        debug: dict[str, Any] = {
            "type": "endless_terminals",
            "reward_mode": "binary",
            "reward_source": "verifier_reward_txt",
            "task_id": self.task_id,
            "docker_image": self.docker_image,
        }

        # 1+2: prep /tests and /logs/verifier
        try:
            _exec(self.container, ["mkdir", "-p", TESTS_DIR, "/logs/verifier"], timeout=10)
            _put_text_to_container(self.container, f"{TESTS_DIR}/test_final_state.py", self.final_state_test, mode=0o644)
            _put_text_to_container(self.container, f"{TESTS_DIR}/test.sh", self.test_script, mode=0o755)
            _exec(self.container, ["chmod", "+x", f"{TESTS_DIR}/test.sh"], timeout=10)
        except Exception as exc:
            debug.update(
                {
                    "reward": 0.0,
                    "resolved": False,
                    "verifier_error": "test_setup_failed",
                    "log_tail": f"setup error: {exc!r}",
                }
            )
            self._reward_debug = debug
            return debug

        # 3: run test.sh
        t0 = time.time()
        rc, output = _exec(
            self.container,
            ["bash", f"{TESTS_DIR}/test.sh"],
            workdir=DEFAULT_CWD,
            timeout=int(self.reward_timeout),
        )
        elapsed = time.time() - t0
        debug["test_script_exit_code"] = rc
        debug["test_script_seconds"] = round(elapsed, 2)
        # Cap log size: head + tail to avoid blowing up the trainer's metric DB.
        debug["log_head"] = output[:1500]
        debug["log_tail"] = output[-1500:] if len(output) > 1500 else ""
        debug["log"] = output[-4000:]  # tail-only to bound size

        if rc == 124:
            debug.update(
                {
                    "reward": 0.0,
                    "resolved": False,
                    "verifier_error": "test_script_timeout",
                }
            )
            self._reward_debug = debug
            return debug

        # 4: read reward.txt
        cat_rc, reward_text = _exec(
            self.container,
            ["cat", REWARD_FILE],
            timeout=15,
        )
        reward_text = (reward_text or "").strip()
        debug["reward_file"] = reward_text
        if cat_rc != 0:
            debug.update(
                {
                    "reward": 0.0,
                    "resolved": False,
                    "verifier_error": "reward_txt_missing",
                }
            )
            self._reward_debug = debug
            return debug

        if reward_text == "1":
            debug.update({"reward": 1.0, "resolved": True, "verifier_error": ""})
        elif reward_text == "0":
            debug.update({"reward": 0.0, "resolved": False, "verifier_error": ""})
        else:
            debug.update(
                {
                    "reward": 0.0,
                    "resolved": False,
                    "verifier_error": f"reward_txt_unexpected:{reward_text[:32]!r}",
                }
            )

        # ---------- Optional shaping (opt-in via constructor flags) ----------
        # Both flags are off by default; existing FUSED ET experiments retain
        # binary 0/1 reward unless explicitly enabled in the env config.
        debug["reward_binary"] = float(debug.get("reward", 0.0))
        debug["evidence_actions_before_finish"] = int(self._evidence_actions)
        passed, failed = _parse_pytest_counts(output)
        if passed is not None:
            debug["pytest_passed"] = passed
            debug["pytest_failed"] = failed
            total = passed + failed
            debug["pytest_pass_fraction"] = (passed / total) if total > 0 else 0.0
        else:
            debug["pytest_pass_fraction"] = None

        if self.partial_reward and debug["reward_binary"] < 1.0 and debug.get("pytest_pass_fraction"):
            # Tier the partial reward: pass-fraction in [0, 0.7) gets a small
            # signal; ≥0.7 gets a stronger signal but never crosses the
            # binary threshold, so the verifier remains the gold standard.
            frac = debug["pytest_pass_fraction"]
            partial = 0.5 * frac if frac < 0.7 else 0.8 * frac  # capped <1.0
            debug["reward"] = float(min(0.95, partial))
            debug["reward_mode"] = "binary+partial"

        if self.finish_without_evidence_penalty > 0.0 and self._evidence_actions == 0 and debug["reward_binary"] < 1.0:
            # Discourage "give up and finish" without ever inspecting state.
            penalty = float(self.finish_without_evidence_penalty)
            debug["reward"] = float(max(-1.0, debug["reward"] - penalty))
            debug["finish_without_evidence_penalty_applied"] = penalty

        self._reward_debug = debug
        return debug

    def compute_final_reward(self) -> float:
        meta = self.compute_final_reward_metadata()
        return float(meta.get("reward", 0.0))

    @property
    def reward_debug(self) -> dict:
        return self._reward_debug

    # ------------------------------------------------------------------
    # Reference-solution helper (used by smoke tests + sanity validation)
    # ------------------------------------------------------------------

    def run_reference_solution(self) -> dict:
        """Apply ``self.solution`` inside the live container.

        Mirrors the ``run_et_task.py --skip-solution`` inverse path. Returns
        ``{"ok": bool, "exit_code": int, "log": str}``. Used by the smoke test
        to confirm a task's verifier chain passes when the agent is replaced
        with the reference solution.
        """
        if self.container is None:
            return {"ok": False, "exit_code": -1, "log": "container missing"}
        if not self.solution:
            return {"ok": False, "exit_code": -1, "log": "solution field empty"}
        try:
            _put_text_to_container(self.container, "/tmp/solve.sh", self.solution, mode=0o755)
            _exec(self.container, ["chmod", "+x", "/tmp/solve.sh"], timeout=10)
        except Exception as exc:
            return {"ok": False, "exit_code": -1, "log": f"copy failed: {exc!r}"}
        rc, out = _exec(
            self.container,
            ["bash", "/tmp/solve.sh"],
            workdir=DEFAULT_CWD,
            timeout=int(self.agent_timeout_sec),
        )
        return {"ok": rc == 0, "exit_code": rc, "log": out[-2000:]}

    # ------------------------------------------------------------------
    # Workflow integration
    # ------------------------------------------------------------------

    @staticmethod
    def is_multithread_safe() -> bool:
        # Each ETEnv owns its own container; there is no shared mutable state
        # between instances besides the docker.DockerClient (which docker-py
        # documents as thread-safe for exec_run / containers.run via the urllib3
        # connection pool).
        return True

    @staticmethod
    def from_dict(extra_info: dict | str) -> "ETEnv":
        """Construct an ETEnv from a verl-style ``extra_info`` row.

        The whole dict is used as ``entry``; any keys that match ``__init__``
        kwargs (e.g. ``step_timeout``) are also forwarded so callers can
        override the defaults from the parquet row if they need to.
        """
        import inspect

        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(ETEnv.__init__)
        init_params: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "entry"):
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
        init_params["entry"] = extra_info
        return ETEnv(**init_params)

    # Defensive cleanup on GC; close() is idempotent so this is safe even
    # when the engine has already called close() explicitly.
    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass
