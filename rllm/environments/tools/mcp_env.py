import asyncio
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import uuid
import warnings
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.mcp_tool import MCPTool

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """Manages MCP connections in a dedicated thread to avoid asyncio context issues."""

    # Global lock to serialize MCP subprocess spawning.  The ``mcp``
    # library's ``stdio_client`` uses ``anyio.open_process`` internally,
    # which installs a child-process watcher on the running event loop.
    # When many MCPConnectionManagers initialise concurrently (each on
    # its own thread / event-loop), the watchers race and throw
    # "Racing with another loop to spawn a process".  Holding a lock
    # around the critical ``_initialize_connection`` section prevents this.
    _spawn_lock = threading.Lock()
    _active_server_lock = threading.Lock()
    _active_server_semaphore: threading.BoundedSemaphore | None = None
    _active_server_limit: int | None = None
    _fd_throttle_condition = threading.Condition()

    def __init__(self, mcp_server_command: str, mcp_server_args: list[str] | None = None, mcp_server_env: dict[str, str] | None = None):
        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args or []
        self.mcp_server_env = mcp_server_env

        self.request_queue: queue.Queue[tuple[str, Any, queue.Queue[tuple[str, Any]] | None]] = queue.Queue()
        self.response_queues: dict[str, queue.Queue[Any]] = {}
        self.worker_thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: ClientSession | None = None
        self.stdio_transport: Any = None
        self.tool_map: dict[str, MCPTool] = {}
        self.running = False
        self.exit_stack: AsyncExitStack | None = None
        self._mcp_errlog: Any = None
        self._mcp_errlog_path: str | None = None
        self._startup_response_queue: queue.Queue[tuple[str, Any]] | None = None
        self._server_slot_semaphore: threading.BoundedSemaphore | None = None
        self._server_slot_acquired = False
        self.startup_timeout = float(os.environ.get("RLLM_MCP_INIT_TIMEOUT", "90"))
        self.tool_timeout = float(os.environ.get("RLLM_MCP_TOOL_TIMEOUT", "30"))

    @classmethod
    def _get_active_server_semaphore(cls) -> threading.BoundedSemaphore | None:
        raw_limit = os.environ.get("RLLM_MCP_MAX_ACTIVE_SERVERS")
        if raw_limit is not None:
            try:
                limit = int(raw_limit)
            except (TypeError, ValueError):
                limit = 0
        else:
            raw_fd_budget = os.environ.get("RLLM_MCP_MAX_ACTIVE_FDS")
            if raw_fd_budget is None:
                return None
            try:
                fd_budget = int(raw_fd_budget)
            except (TypeError, ValueError):
                fd_budget = 0
            try:
                fds_per_server = max(1, int(os.environ.get("RLLM_MCP_FDS_PER_SERVER", "6")))
            except (TypeError, ValueError):
                fds_per_server = 6
            limit = fd_budget // fds_per_server
        if limit <= 0:
            return None

        with cls._active_server_lock:
            if cls._active_server_semaphore is None or cls._active_server_limit != limit:
                cls._active_server_semaphore = threading.BoundedSemaphore(limit)
                cls._active_server_limit = limit
            return cls._active_server_semaphore

    @staticmethod
    def _ensure_nofile_limit() -> None:
        try:
            min_nofile = int(os.environ.get("RLLM_MCP_MIN_NOFILE", "4096"))
        except (TypeError, ValueError):
            min_nofile = 4096
        if min_nofile <= 0:
            return

        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft >= min_nofile:
                return
            target = min_nofile if hard == resource.RLIM_INFINITY else min(min_nofile, hard)
            if target > soft:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except Exception as e:
            logger.debug("Could not raise RLIMIT_NOFILE to %s: %s", min_nofile, e)

    @staticmethod
    def _current_fd_count() -> int:
        try:
            return len(os.listdir("/proc/self/fd"))
        except Exception:
            return 0

    @classmethod
    def _wait_for_fd_headroom(cls) -> None:
        try:
            threshold = int(os.environ.get("RLLM_MCP_FD_THROTTLE_THRESHOLD", "4096"))
        except (TypeError, ValueError):
            threshold = 4096
        if threshold <= 0:
            return

        with cls._fd_throttle_condition:
            while cls._current_fd_count() >= threshold:
                cls._fd_throttle_condition.wait(timeout=1.0)

    @classmethod
    def _notify_fd_headroom(cls) -> None:
        with cls._fd_throttle_condition:
            cls._fd_throttle_condition.notify_all()

    def _acquire_server_slot(self) -> None:
        if self._server_slot_acquired:
            return
        semaphore = self._get_active_server_semaphore()
        if semaphore is None:
            return
        semaphore.acquire()
        self._server_slot_semaphore = semaphore
        self._server_slot_acquired = True

    def _release_server_slot(self) -> None:
        if not self._server_slot_acquired:
            return
        semaphore = self._server_slot_semaphore
        self._server_slot_semaphore = None
        self._server_slot_acquired = False
        if semaphore is not None:
            try:
                semaphore.release()
            except ValueError:
                pass

    def start(self):
        """Start the connection manager thread.

        Subprocess spawning is serialized by the class-level ``_spawn_lock``,
        held inside ``_initialize_connection`` around only the ``stdio_client``
        spawn (preventing the *anyio* "Racing with another loop to spawn a
        process" error). The rest of the handshake — ``session.initialize()``
        and ``list_tools()`` — runs concurrently across managers, so N startups
        no longer serialize end-to-end.
        """
        if self.running:
            return

        self._ensure_nofile_limit()
        self._wait_for_fd_headroom()
        self._acquire_server_slot()
        try:
            max_retries = max(1, int(os.environ.get("RLLM_MCP_START_RETRIES", "5")))
        except (TypeError, ValueError):
            max_retries = 5
        last_error = None
        try:
            for attempt in range(max_retries):
                self.running = True
                response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
                self._startup_response_queue = response_queue
                self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
                self.worker_thread.start()

                # Wait for initialization
                self.request_queue.put(("init", None, response_queue))
                try:
                    result = response_queue.get(timeout=self.startup_timeout + 10)
                except queue.Empty:
                    last_error = "Timed out waiting for MCP server initialization"
                    stopped = self._stop_no_lock()
                    if stopped:
                        continue
                    break
                if result[0] == "error":
                    last_error = result[1]
                    stopped = self._stop_no_lock()
                    if stopped and attempt < max_retries - 1 and self._is_retryable_startup_error(last_error):
                        import time

                        # Step-1 fused rollouts can start many distinct MCP
                        # assets at once.  A subset of otherwise-healthy stdio
                        # servers sometimes exits with a transient "Connection
                        # closed"/cancelled/timeout while Python imports are
                        # contending.  Back off and retry the same asset instead
                        # of handing the rollout an empty tool set immediately.
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    if not stopped:
                        break
                    raise Exception(f"Failed to initialize MCP connection: {last_error}")
                self._startup_response_queue = None
                return  # Success

            raise Exception(f"Failed to initialize MCP connection after {max_retries} retries: {last_error}")
        except Exception:
            self._release_server_slot()
            self._notify_fd_headroom()
            raise

    @staticmethod
    def _is_retryable_startup_error(error: Any) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "connection closed",
                "initialization cancelled",
                "timed out waiting for mcp server initialization",
                "racing with another loop",
            )
        )

    def _stop_no_lock(self) -> bool:
        """Internal stop that doesn't acquire locks — used during retry."""
        self.running = False
        self.request_queue.put(("stop", None, None))
        worker_thread = getattr(self, "worker_thread", None)
        if worker_thread:
            worker_thread.join(timeout=5)
            if worker_thread.is_alive():
                logger.warning("MCP worker thread did not stop within timeout for %s %s", self.mcp_server_command, self.mcp_server_args)
                return False
        # Clear any leftover state for a clean retry
        self.session = None
        self.tool_map = {}
        self._startup_response_queue = None
        # Drain leftover items from queue
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        return True

    def stop(self):
        """Stop the connection manager thread."""
        if not getattr(self, "running", False):
            self._release_server_slot()
            self._notify_fd_headroom()
            return

        self.running = False
        self.request_queue.put(("stop", None, None))
        worker_thread = getattr(self, "worker_thread", None)
        if worker_thread:
            worker_thread.join(timeout=5)
            if worker_thread.is_alive():
                logger.warning("MCP worker thread did not stop within timeout for %s %s", self.mcp_server_command, self.mcp_server_args)
        self.session = None
        self.tool_map = {}
        self._startup_response_queue = None
        self._release_server_slot()
        self._notify_fd_headroom()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        """Execute tool calls and return results."""
        if not self.running:
            raise Exception("Connection manager not running")

        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.request_queue.put(("execute", tool_calls, response_queue))
        try:
            result = response_queue.get(timeout=self.tool_timeout + 5)
        except queue.Empty as e:
            self.stop()
            raise Exception(f"Tool execution timed out after {self.tool_timeout:.1f}s") from e
        if result[0] == "error":
            raise Exception(f"Tool execution failed: {result[1]}")
        return result[1]  # type: ignore

    def _run_worker(self):
        """Worker thread that runs the asyncio event loop."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._worker_loop())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.running = False
            self._notify_startup_error(e)
        finally:
            if self.loop and not self.loop.is_closed():
                try:
                    self.loop.run_until_complete(self._cleanup())
                except Exception:
                    pass
                self.loop.close()
            self.loop = None

    def _notify_startup_error(self, error: BaseException | str) -> None:
        response_queue = getattr(self, "_startup_response_queue", None)
        if response_queue is None:
            return
        try:
            response_queue.put(("error", self._format_error_with_stderr(error)))
        except Exception:
            pass
        finally:
            self._startup_response_queue = None

    def _format_error_with_stderr(self, error: BaseException | str) -> str:
        message = str(error)
        errlog_path = getattr(self, "_mcp_errlog_path", None)
        if not errlog_path:
            return message
        try:
            with open(errlog_path, "r", encoding="utf-8", errors="replace") as fh:
                stderr = fh.read().strip()
        except Exception:
            return message
        if not stderr:
            return message
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        return f"{message}\nMCP server stderr:\n{stderr}"

    async def _worker_loop(self):
        """Main worker loop that processes requests."""
        while self.running:
            try:
                # Check for requests with timeout
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                command, data, response_queue = request

                if command == "init":
                    try:
                        await asyncio.wait_for(self._initialize_connection(), timeout=self.startup_timeout)
                        if response_queue:
                            response_queue.put(("success", self.tool_map))
                        self._startup_response_queue = None
                    except asyncio.CancelledError as e:
                        self.running = False
                        await self._cleanup_suppressing()
                        if response_queue:
                            response_queue.put(("error", self._format_error_with_stderr(f"MCP initialization cancelled: {e}")))
                        break
                    except asyncio.TimeoutError:
                        self.running = False
                        await self._cleanup_suppressing()
                        if response_queue:
                            response_queue.put(("error", self._format_error_with_stderr(f"Timed out waiting for MCP server initialization after {self.startup_timeout:.1f}s")))
                        break
                    except Exception as e:
                        self.running = False
                        await self._cleanup_suppressing()
                        if response_queue:
                            response_queue.put(("error", self._format_error_with_stderr(e)))
                        break

                elif command == "execute":
                    try:
                        result = await asyncio.wait_for(self._execute_tools(data), timeout=self.tool_timeout)
                        if response_queue:
                            response_queue.put(("success", result))
                    except asyncio.CancelledError as e:
                        self.running = False
                        await self._cleanup_suppressing()
                        if response_queue:
                            response_queue.put(("error", f"MCP tool execution cancelled: {e}"))
                        break
                    except asyncio.TimeoutError:
                        self.running = False
                        await self._cleanup_suppressing()
                        if response_queue:
                            response_queue.put(("error", f"MCP tool execution timed out after {self.tool_timeout:.1f}s"))
                        break
                    except Exception as e:
                        if response_queue:
                            response_queue.put(("error", str(e)))

                elif command == "stop":
                    break

            except Exception as e:
                print(f"Worker loop error: {e}")

    async def _initialize_connection(self):
        """Initialize the MCP connection."""
        server_params = StdioServerParameters(command=self.mcp_server_command, args=self.mcp_server_args, env=self.mcp_server_env)

        # Use AsyncExitStack properly within this event loop
        self.exit_stack = AsyncExitStack()
        # Capture subprocess stderr in a temp file. Passing /dev/null hides the
        # generated tools.py traceback and turns every startup failure into a
        # useless "Connection closed".
        errlog = tempfile.NamedTemporaryFile(prefix="rllm_mcp_", suffix=".stderr", mode="w+", encoding="utf-8", delete=False)
        self._mcp_errlog_path = errlog.name
        self._mcp_errlog = self.exit_stack.enter_context(errlog)
        # Serialize ONLY the subprocess spawn. anyio's ``open_process`` (used
        # by ``stdio_client``) installs a child-process watcher on the running
        # event loop; when many managers spawn concurrently from their own
        # worker-thread loops the watchers race and throw "Racing with another
        # loop to spawn a process". Holding ``_spawn_lock`` around just the
        # spawn (~ms) prevents that race while letting the expensive part of
        # the handshake — ``session.initialize()`` + ``list_tools()`` — run
        # fully concurrently across managers. (Previously the lock wrapped the
        # entire ~1s start() handshake, serializing all startups and leaving
        # the GPUs idle for >10 min while 1024 MCP servers booted one-by-one.)
        with MCPConnectionManager._spawn_lock:
            self.stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params, errlog=self._mcp_errlog))
        stdio, write = self.stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        if self.session:
            await self.session.initialize()

            response = await self.session.list_tools()
            tools = response.tools
            # print(f"\nConnected to MCP server with tools: {[tool.name for tool in tools]}")

            self.tool_map = {}
            for tool in tools:
                mcp_tool = MCPTool(session=self.session, tool_name=tool.name, tool_description=tool.description, tool_schema=tool.inputSchema)
                self.tool_map[tool.name] = mcp_tool
                mapped_name = tool.name.replace("-", "_")
                if mapped_name != tool.name:
                    mapped_tool = MCPTool(session=self.session, tool_name=tool.name, tool_description=tool.description, tool_schema=tool.inputSchema)
                    self.tool_map[mapped_name] = mapped_tool

    async def _execute_tools(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        """Execute tool calls."""
        tool_outputs: dict[str, str] = {}

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            if tool_name in self.tool_map:
                tool_instance = self.tool_map[tool_name]
                result = await tool_instance.async_forward(**tool_args)
                tool_outputs[tool_call["id"]] = result.to_string()
            else:
                tool_outputs[tool_call["id"]] = f"Error: Tool {tool_name} not found"

        return tool_outputs

    async def _cleanup(self) -> None:
        """Clean up the connection."""
        exit_stack = getattr(self, "exit_stack", None)
        if exit_stack:
            self.exit_stack = None
            await exit_stack.aclose()
        self.session = None
        self.stdio_transport = None
        self.tool_map = {}

    async def _cleanup_suppressing(self) -> None:
        try:
            await self._cleanup()
        except Exception as e:
            logger.debug("MCP cleanup failed for %s %s: %s", self.mcp_server_command, self.mcp_server_args, e)


class MCPEnvironment(BaseEnv):
    """
    An environment for MCP-based tools that provides questions and evaluates responses.
    Uses a dedicated connection manager to avoid asyncio context issues.
    """

    # Class-level pool to share managers across instances with the same server config.
    _connection_manager: MCPConnectionManager | None = None
    _connection_managers: dict[tuple[Any, ...], MCPConnectionManager] = {}
    _manager_lock = threading.Lock()

    def __init__(
        self,
        task: dict[str, Any] | None = None,
        mcp_server_command: str | None = None,
        mcp_server_args: list[str] | None = None,
        mcp_server_env: dict[str, str] | None = None,
        reward_fn: RewardFunction | None = None,
        max_steps: int = 10,
        share_mcp_manager: bool = True,
    ):
        """
        Initialize the MCPEnvironment.

        Args:
            task: Task information for the environment.
            mcp_server_command: Command to run the MCP server.
            mcp_server_args: Arguments for the MCP server.
            mcp_server_env: Environment variables for the MCP server.
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
            share_mcp_manager: Whether to share a manager across environments with the same server config.
        """
        self.step_count = 0
        self.max_steps = max_steps
        self.task = task
        self.reward_fn = reward_fn
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward

        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args or []
        self.mcp_server_env = mcp_server_env
        self.share_mcp_manager = share_mcp_manager
        self._non_submit_tool_calls = 0
        self._submit_without_tool_retries = 0
        self._max_submit_without_tool_retries = 10
        self._connection_manager: MCPConnectionManager | None = None
        self._manager_key: tuple[Any, ...] | None = None

        # Initialize connection manager
        if self.mcp_server_command is not None:
            manager_key = self._build_manager_key()
            with MCPEnvironment._manager_lock:
                manager = MCPEnvironment._connection_managers.get(manager_key)
                if manager is None:
                    manager = MCPConnectionManager(self.mcp_server_command, self.mcp_server_args, self.mcp_server_env)
                    manager.start()
                    MCPEnvironment._connection_managers[manager_key] = manager
                self._connection_manager = manager
                MCPEnvironment._connection_manager = manager
                self._manager_key = manager_key

    @staticmethod
    def _ensure_server_script_legacy_inline(assets_dir: Path) -> Path:
        server_script = assets_dir / "mcp_server.py"
        server_script.write_text(
            "import sys\n"
            "import logging\n"
            "import types\n"
            "import re\n"
            "from typing import Any\n"
            "_MCP_LOGGERS = ('mcp', 'mcp.server', 'mcp.server.fastmcp', 'mcp.server.fastmcp.tools', 'mcp.server.fastmcp.tools.tool_manager')\n"
            "for _name in _MCP_LOGGERS:\n"
            "    _lg = logging.getLogger(_name)\n"
            "    _lg.setLevel(logging.CRITICAL)\n"
            "    _lg.propagate = False\n"
            "    _lg.disabled = True\n"
            "from pathlib import Path\n"
            "_ASSETS_DIR = Path(__file__).parent\n"
            "sys.path.insert(0, str(_ASSETS_DIR))\n"
            "_TOOLS_PATH = _ASSETS_DIR / 'tools.py'\n"
            "_source = _TOOLS_PATH.read_text(encoding='utf-8')\n"
            "_BAD_LINE_PREFIX_RE = re.compile(r'^(?P<indent>[ \\t]+)(?P<number>\\d+)[ \\t]+(?P<code>(?:raise|return|if|for|while|elif|else\\b|except\\b|with|assert\\b|[A-Za-z_][A-Za-z0-9_]*\\s*=).*)$')\n"
            "_BAD_ARGUMENT_NUMBER_RE = re.compile(r'(?<=,)\\s*\\d+\\s+(?=(?!(?:if|for|else)\\b)[A-Za-z_])')\n"
            "_SINGLE_QUOTED_INDEX_RE = re.compile(r\"\\['(?P<key>[^']+)'\\]\")\n"
            "_FSTRING_DOUBLE_QUOTED_EXPR_RE = re.compile(r'\\[\"\\{(?P<expr>[^}]+)\\}\"\\]')\n"
            "_FSTRING_DOUBLE_QUOTED_INDEX_RE = re.compile(r'\\[\"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\"\\]')\n"
            "_NON_ASCII_NUMERIC_DEFAULT_RE = re.compile(r'(?P<prefix>=\\s*)[^\\x00-\\x7F]+(?P<number>-?\\d+(?:\\.\\d+)?)\\b')\n"
            "_ENUM_ALL_DEFAULT_RE = re.compile(r'(?P<prefix>:\\s*(?P<enum>[A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*)(?P=enum)\\.ALL\\b')\n"
            "def _repair_common_fstring_quote_error(_line):\n"
            "    if \"f'\" in _line:\n"
            "        _line = _SINGLE_QUOTED_INDEX_RE.sub(r'[\"\\g<key>\"]', _line)\n"
            "        _line = _line.replace('\"]})', '\"])}')\n"
            "    if 'f\"' in _line:\n"
            "        _line = _FSTRING_DOUBLE_QUOTED_EXPR_RE.sub(r\"['{\\g<expr>}']\", _line)\n"
            "        _line = _FSTRING_DOUBLE_QUOTED_INDEX_RE.sub(r\"['\\g<key>']\", _line)\n"
            "        _line = _line.replace(\"']})\", \"'])}\")\n"
            "    _stripped = _line.rstrip('\\n')\n"
            "    if 'raise ValueError(\"' in _stripped and _stripped.endswith(\"')\"):\n"
            "        return _stripped[:-2] + '\")' + ('\\n' if _line.endswith('\\n') else '')\n"
            "    return _line\n"
            "def _insert_pass_for_empty_block(_lines, _error_lineno):\n"
            "    _insert_at = _error_lineno - 1\n"
            "    for _prev_idx in range(_insert_at - 1, -1, -1):\n"
            "        _prev = _lines[_prev_idx]\n"
            "        if not _prev.strip():\n"
            "            continue\n"
            "        if _prev.lstrip().startswith('#'):\n"
            "            continue\n"
            "        if not _prev.rstrip().endswith(':'):\n"
            "            return False\n"
            "        _indent = re.match(r'^[ \\\\t]*', _prev).group(0)\n"
            "        _lines.insert(_insert_at, f'{_indent}    pass\\n')\n"
            "        return True\n"
            "    return False\n"
            "def _repair_syntax_line_prefixes(_src):\n"
            "    _lines = _src.splitlines(keepends=True)\n"
            "    for _ in range(100):\n"
            "        try:\n"
            "            compile(''.join(_lines), '<tools.py>', 'exec')\n"
            "            break\n"
            "        except (SyntaxError, IndentationError) as _e:\n"
            "            _lineno = _e.lineno\n"
            "            if _lineno is None or _lineno < 1 or _lineno > len(_lines):\n"
            "                break\n"
            "            _line = _lines[_lineno - 1]\n"
            "            _line_ending = '\\n' if _line.endswith('\\n') else ''\n"
            "            _body = _line[:-1] if _line_ending else _line\n"
            "            _match = _BAD_LINE_PREFIX_RE.match(_body)\n"
            "            if not _match:\n"
            "                _repaired_arg = _BAD_ARGUMENT_NUMBER_RE.sub(' ', _line)\n"
            "                if _repaired_arg != _line:\n"
            "                    _lines[_lineno - 1] = _repaired_arg\n"
            "                    continue\n"
            "                _repaired_fstring = _repair_common_fstring_quote_error(_line)\n"
            "                if _repaired_fstring != _line:\n"
            "                    _lines[_lineno - 1] = _repaired_fstring\n"
            "                    continue\n"
            "                if isinstance(_e, IndentationError) and 'expected an indented block after' in str(_e):\n"
            "                    if _insert_pass_for_empty_block(_lines, _lineno):\n"
            "                        continue\n"
            "                break\n"
            "            _lines[_lineno - 1] = f\"{_match.group('indent')}{_match.group('code')}{_line_ending}\"\n"
            "    return ''.join(_lines)\n"
            "def _normalize_builtin_any_annotations(_src):\n"
            "    _out = []\n"
            "    _in_sig = False\n"
            "    for _line in _src.splitlines(keepends=True):\n"
            "        _stripped = _line.lstrip()\n"
            "        if _stripped.startswith('def ') or _in_sig:\n"
            "            _line = re.sub(r'(?<=[:\\\\[, ])any(?=[]\\\\[,)=\\\\n ])', 'Any', _line)\n"
            "            _in_sig = not _stripped.endswith(':\\\\n') and not _stripped.endswith(':')\n"
            "        _out.append(_line)\n"
            "    return ''.join(_out)\n"
            "def _normalize_non_ascii_numeric_defaults(_src):\n"
            "    return _NON_ASCII_NUMERIC_DEFAULT_RE.sub(r'\\g<prefix>\\g<number>', _src)\n"
            "def _collect_enum_members(_src):\n"
            "    _members_by_enum = {}\n"
            "    _lines = _src.splitlines()\n"
            "    _idx = 0\n"
            "    while _idx < len(_lines):\n"
            "        _line = _lines[_idx]\n"
            "        _match = re.match(r'^(?P<indent>[ \\\\t]*)class\\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\\s*\\((?P<bases>[^)]*\\bEnum\\b[^)]*)\\)\\s*:', _line)\n"
            "        if not _match:\n"
            "            _idx += 1\n"
            "            continue\n"
            "        _class_indent = len(_match.group('indent').replace('\\t', '    '))\n"
            "        _members = []\n"
            "        _idx += 1\n"
            "        while _idx < len(_lines):\n"
            "            _body_line = _lines[_idx]\n"
            "            _stripped = _body_line.strip()\n"
            "            if not _stripped or _stripped.startswith('#'):\n"
            "                _idx += 1\n"
            "                continue\n"
            "            _body_indent = len(_body_line[:len(_body_line) - len(_body_line.lstrip())].replace('\\t', '    '))\n"
            "            if _body_indent <= _class_indent:\n"
            "                break\n"
            "            _member_match = re.match(r'^[ \\\\t]+([A-Z][A-Z0-9_]*)\\s*=', _body_line)\n"
            "            if _member_match:\n"
            "                _members.append(_member_match.group(1))\n"
            "            _idx += 1\n"
            "        _members_by_enum[_match.group('name')] = _members\n"
            "    return _members_by_enum\n"
            "def _normalize_missing_enum_all_defaults(_src):\n"
            "    _enum_members = _collect_enum_members(_src)\n"
            "    if not _enum_members:\n"
            "        return _src\n"
            "    def _replace(_match):\n"
            "        _enum_name = _match.group('enum')\n"
            "        _members = _enum_members.get(_enum_name)\n"
            "        if not _members or 'ALL' in _members:\n"
            "            return _match.group(0)\n"
            "        return f\"{_match.group('prefix')}{_enum_name}.{_members[0]}\"\n"
            "    return _ENUM_ALL_DEFAULT_RE.sub(_replace, _src)\n"
            "def _repair_empty_try_blocks(_src):\n"
            "    _lines = _src.splitlines(keepends=True)\n"
            "    _out = []\n"
            "    _idx = 0\n"
            "    while _idx < len(_lines):\n"
            "        _out.append(_lines[_idx])\n"
            "        if _lines[_idx].strip() == 'try:':\n"
            "            _indent = _lines[_idx][:len(_lines[_idx]) - len(_lines[_idx].lstrip())]\n"
            "            _scan = _idx + 1\n"
            "            while _scan < len(_lines) and (not _lines[_scan].strip() or _lines[_scan].lstrip().startswith('#')):\n"
            "                _scan += 1\n"
            "            if _scan < len(_lines):\n"
            "                _next_stripped = _lines[_scan].lstrip()\n"
            "                _next_indent = _lines[_scan][:len(_lines[_scan]) - len(_next_stripped)]\n"
            "                if _next_indent == _indent and _next_stripped.startswith(('except ', 'except:', 'finally:')):\n"
            "                    _out.append(f'{_indent}    pass\\n')\n"
            "        _idx += 1\n"
            "    return ''.join(_out)\n"
            "def _normalize_bare_tool_decorators(_src):\n"
            "    if '@tool' not in _src:\n"
            "        return _src\n"
            "    _lines = _src.splitlines(keepends=True)\n"
            "    if not any(_line.lstrip().startswith('@mcp.tool') for _line in _lines):\n"
            "        return _src\n"
            "    _out = []\n"
            "    for _line in _lines:\n"
            "        _stripped = _line.lstrip()\n"
            "        if _stripped.startswith('@tool('):\n"
            "            _indent = _line[:len(_line) - len(_stripped)]\n"
            "            _out.append(f\"{_indent}@mcp.tool({_stripped[len('@tool('):]}\")\n"
            "        else:\n"
            "            _out.append(_line)\n"
            "    return ''.join(_out)\n"
            "def sanitize_tools_source(_src):\n"
            "    _rewritten = []\n"
            "    _saw_fastmcp_instance = False\n"
            "    for _line in _src.splitlines(keepends=True):\n"
            "        _stripped = _line.strip()\n"
            "        if _stripped.startswith('mcp') and '= FastMCP(' in _stripped:\n"
            "            _saw_fastmcp_instance = True\n"
            "            _rewritten.append(_line)\n"
            "            _indent = _line[:len(_line) - len(_line.lstrip())]\n"
            "            if not _indent:\n"
            "                _rewritten.append('tool = mcp.tool\\n')\n"
            "            continue\n"
            "        if _saw_fastmcp_instance and _stripped == 'import mcp':\n"
            "            _indent = _line[:len(_line) - len(_line.lstrip())]\n"
            "            if _indent:\n"
            "                _rewritten.append(f'{_indent}pass\\n')\n"
            "            else:\n"
            "                _rewritten.append('import mcp as _rllm_mcp_package\\n')\n"
            "            continue\n"
            "        if _saw_fastmcp_instance and _stripped in {'from .tools import mcp', 'from tools import mcp'}:\n"
            "            _rewritten.append('\\n')\n"
            "            continue\n"
            "        if _saw_fastmcp_instance and _stripped == 'from mcp import tool':\n"
            "            _rewritten.append('\\n')\n"
            "            continue\n"
            "        if _stripped in {'from . import BASE_DIR', 'from tools import BASE_DIR'}:\n"
            "            _indent = _line[:len(_line) - len(_line.lstrip())]\n"
            "            _rewritten.append(f'{_indent}BASE_DIR = Path(__file__).resolve().parent\\n')\n"
            "            continue\n"
            "        if re.match(r'^BASE_DIR\\s*=\\s*Path\\(__file__\\)(?:\\.resolve\\(\\))?\\.parent\\.parent\\s*$', _stripped):\n"
            "            _indent = _line[:len(_line) - len(_line.lstrip())]\n"
            "            _rewritten.append(f'{_indent}BASE_DIR = Path(__file__).resolve().parent\\n')\n"
            "            continue\n"
            "        _rewritten.append(_line)\n"
            "    _normalized = _normalize_builtin_any_annotations(''.join(_rewritten))\n"
            "    _normalized = _normalize_non_ascii_numeric_defaults(_normalized)\n"
            "    _normalized = _normalize_missing_enum_all_defaults(_normalized)\n"
            "    _normalized = _repair_empty_try_blocks(_normalized)\n"
            "    _normalized = _normalize_bare_tool_decorators(_normalized)\n"
            "    return _repair_syntax_line_prefixes(_normalized)\n"
            "_source = sanitize_tools_source(_source)\n"
            "tools = types.ModuleType('tools')\n"
            "tools.__file__ = str(_TOOLS_PATH)\n"
            "tools.__package__ = ''\n"
            "tools.Any = Any\n"
            "sys.modules['tools'] = tools\n"
            "exec(compile(_source, str(_TOOLS_PATH), 'exec'), tools.__dict__)\n"
            "tools.mcp.run()\n",
            encoding="utf-8",
        )
        return server_script

    @staticmethod
    def _ensure_server_script(assets_dir: Path) -> Path:
        server_script = assets_dir / "mcp_server.py"
        repo_root = Path(__file__).resolve().parents[3]
        script_source = (
            "import logging\n"
            "import importlib.util\n"
            "import os\n"
            "import sys\n"
            "import types\n"
            "from pathlib import Path\n"
            "from typing import Any\n"
            "_MCP_LOGGERS = ('mcp', 'mcp.server', 'mcp.server.fastmcp', 'mcp.server.fastmcp.tools', 'mcp.server.fastmcp.tools.tool_manager')\n"
            "for _name in _MCP_LOGGERS:\n"
            "    _lg = logging.getLogger(_name)\n"
            "    _lg.setLevel(logging.CRITICAL)\n"
            "    _lg.propagate = False\n"
            "    _lg.disabled = True\n"
            "_ASSETS_DIR = Path(__file__).resolve().parent\n"
            "sys.path.insert(0, str(_ASSETS_DIR))\n"
            "_repo_root = None\n"
            "for _parent in _ASSETS_DIR.parents:\n"
            "    if (_parent / 'rllm' / 'environments' / 'tools' / 'mcp_source.py').exists():\n"
            "        _repo_root = _parent\n"
            "        break\n"
            f"if _repo_root is None:\n"
            f"    _known_repo_root = Path({str(repo_root)!r})\n"
            f"    if (_known_repo_root / 'rllm' / 'environments' / 'tools' / 'mcp_source.py').exists():\n"
            f"        _repo_root = _known_repo_root\n"
            "if _repo_root is None:\n"
            "    raise RuntimeError('Could not locate rllm/environments/tools/mcp_source.py')\n"
            "_mcp_source_path = _repo_root / 'rllm' / 'environments' / 'tools' / 'mcp_source.py'\n"
            "_spec = importlib.util.spec_from_file_location('_rllm_mcp_source', _mcp_source_path)\n"
            "if _spec is None or _spec.loader is None:\n"
            "    raise RuntimeError(f'Could not load {_mcp_source_path}')\n"
            "_mcp_source = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_mcp_source)\n"
            "sanitize_tools_source = _mcp_source.sanitize_tools_source\n"
            "_TOOLS_PATH = _ASSETS_DIR / 'tools.py'\n"
            "_source = sanitize_tools_source(_TOOLS_PATH.read_text(encoding='utf-8'))\n"
            "tools = types.ModuleType('tools')\n"
            "tools.__file__ = str(_TOOLS_PATH)\n"
            "tools.__package__ = ''\n"
            "tools.Any = Any\n"
            "sys.modules['tools'] = tools\n"
            "exec(compile(_source, str(_TOOLS_PATH), 'exec'), tools.__dict__)\n"
            "tools.mcp.run()\n"
        )
        try:
            if server_script.read_text(encoding="utf-8") == script_source:
                return server_script
        except FileNotFoundError:
            pass

        tmp_script = server_script.with_name(f".{server_script.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_script.write_text(script_source, encoding="utf-8")
            os.replace(tmp_script, server_script)
        finally:
            try:
                tmp_script.unlink()
            except FileNotFoundError:
                pass
        return server_script

    def _build_manager_key(self) -> tuple[Any, ...]:
        env_items = tuple(sorted(self.mcp_server_env.items())) if self.mcp_server_env else None
        base_key = (self.mcp_server_command, tuple(self.mcp_server_args), env_items)
        if self.share_mcp_manager:
            return base_key
        return ("isolated", uuid.uuid4().hex, base_key)

    def _get_connection_manager(self) -> MCPConnectionManager | None:
        manager = self._connection_manager
        class_manager = MCPEnvironment._connection_manager
        if class_manager is not None and class_manager is not manager:
            return class_manager
        return manager

    @staticmethod
    def _safe_stop_manager(manager: MCPConnectionManager | None) -> None:
        if manager is None:
            return
        try:
            manager.stop()
        except Exception:
            pass

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        self._non_submit_tool_calls = 0
        self._submit_without_tool_retries = 0
        obs = dict(self.task) if isinstance(self.task, dict) else {}
        manager = self._get_connection_manager()
        tool_map = getattr(manager, "tool_map", None)
        if manager and tool_map:
            tools_json = []
            seen = set()
            for tool in tool_map.values():
                name = getattr(tool, "name", None)
                if not name or name in seen:
                    continue
                seen.add(name)
                tools_json.append(tool.json)
            if tools_json:
                obs["tools_json"] = tools_json
        return obs, {}

    def step(self, action: Any):
        """
        Take a step in the environment based on the action.

        Args:
            action: Action from the agent (tool calls or final response)

        Returns:
            next_observations, rewards, terminateds, infos
        """
        if isinstance(action, dict):
            action = [action]
        self.step_count += 1

        reward = 0.0
        # Check if we should terminate
        done = self.step_count >= self.max_steps or isinstance(action, str)

        # Check if action contains a "finish" tool call
        submit_result_tool_call = None
        if isinstance(action, list) and action:
            for tool_call in action:
                tool_name = tool_call.get("function", {}).get("name", "")
                if tool_name == "finish":
                    done = True
                    break
                # Check if it's a submit_result_difficulty_xxx function
                elif tool_name.startswith("submit_result_difficulty_"):
                    done = True
                    submit_result_tool_call = tool_call
                    break
                else:
                    self._non_submit_tool_calls += 1

        if submit_result_tool_call is not None and self._non_submit_tool_calls == 0:
            self._submit_without_tool_retries += 1
            remaining = self._max_submit_without_tool_retries - self._submit_without_tool_retries
            if self._submit_without_tool_retries <= self._max_submit_without_tool_retries:
                return (
                    {},
                    0.0,
                    False,
                    {
                        "rejected_submit": True,
                        "reason": "submit_without_tool_call",
                        "remaining_retries": max(remaining, 0),
                    },
                )

            forced_reward = -5.0
            info_dict = {
                "response": action,
                "metadata": {
                    "error": "submit_without_tool_call_max_retries",
                    "reward/base_reward": 0.0,
                    "reward/tool_call_total": forced_reward,
                    "reward/step_penalty": 0.0,
                    "reward/tool_call_bonus": 0.0,
                    "tool_call_stats": {
                        "submit_called": True,
                        "non_submit_tool_calls": 0,
                        "step_count": self.step_count,
                    },
                },
                "reward/base_reward": 0.0,
                "reward/tool_call_total": forced_reward,
                "reward/step_penalty": 0.0,
                "reward/tool_call_bonus": 0.0,
            }
            return {}, forced_reward, True, info_dict

        if done:
            # Agent is done - evaluate the response
            llm_response = None

            # If submit_result_difficulty_xxx was called, execute it first and use result from arguments as llm_response
            if submit_result_tool_call is not None:
                try:
                    # Execute the submit_result tool call to ensure it completes
                    manager = self._get_connection_manager()
                    if manager is not None:
                        tool_outputs = manager.execute_tool_calls([submit_result_tool_call])
                    else:
                        tool_outputs = {}

                    # Extract result from tool call arguments (not from tool output)
                    arguments = submit_result_tool_call.get("function", {}).get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, ValueError):
                            arguments = {}

                    # Get the result from arguments and convert to JSON string
                    if isinstance(arguments, dict):
                        result = arguments.get("result")
                        if result is not None:
                            if isinstance(result, (dict, list)):
                                llm_response = json.dumps(result, ensure_ascii=False)
                            else:
                                llm_response = str(result)
                        else:
                            llm_response = None
                    else:
                        llm_response = None
                except Exception as e:
                    print(f"Error executing submit_result tool: {e}")
                    llm_response = None

            # Handle other termination cases
            if llm_response is None:
                if isinstance(action, str):
                    llm_response = action
                elif isinstance(action, list):
                    # Find the finish tool call
                    finish_action = None
                    for tool_call in action:
                        if tool_call.get("function", {}).get("name") == "finish":
                            finish_action = tool_call
                            break
                    if finish_action:
                        arguments = finish_action.get("function", {}).get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)

                        if isinstance(arguments, dict):
                            llm_response = arguments.get("response", "")
                        else:
                            llm_response = str(arguments)
                    else:
                        llm_response = str(action)

            if self.reward_fn and self.task is not None:
                tool_call_stats = {
                    "submit_called": submit_result_tool_call is not None,
                    "non_submit_tool_calls": self._non_submit_tool_calls,
                    "step_count": self.step_count,
                }
                if isinstance(self.task, dict):
                    task_info = dict(self.task)
                    task_info["tool_call_stats"] = tool_call_stats
                else:
                    task_info = {"tool_call_stats": tool_call_stats}
                reward_output = self.reward_fn(task_info=task_info, action=llm_response)
                metadata = dict(reward_output.metadata)
                metadata["tool_call_stats"] = tool_call_stats

                info_dict = {"response": action, "metadata": metadata}
                if "reward/base_reward" in metadata:
                    info_dict["reward/base_reward"] = metadata["reward/base_reward"]
                if "reward/tool_call_total" in metadata:
                    info_dict["reward/tool_call_total"] = metadata["reward/tool_call_total"]
                if "reward/step_penalty" in metadata:
                    info_dict["reward/step_penalty"] = metadata["reward/step_penalty"]
                if "reward/tool_call_bonus" in metadata:
                    info_dict["reward/tool_call_bonus"] = metadata["reward/tool_call_bonus"]

                return {}, reward_output.reward, done, info_dict
            else:
                return {}, 0.0, done, {"response": action, "metadata": {}}

        # Execute tool calls using the connection manager
        tool_calls = action
        try:
            manager = self._get_connection_manager()
            if manager is not None:
                tool_outputs = manager.execute_tool_calls(tool_calls)
                next_obs = {"tool_outputs": tool_outputs}
            else:
                next_obs = {"tool_outputs": {}}
        except Exception as e:
            print(f"Tool execution error: {e}")
            next_obs = {"tool_outputs": {}}

        return next_obs, reward, done, {"response": action, "metadata": {}}

    def close(self):
        """Clean up resources."""
        if self.share_mcp_manager:
            return
        if self._connection_manager is None:
            return
        with MCPEnvironment._manager_lock:
            if self._manager_key in MCPEnvironment._connection_managers:
                MCPEnvironment._connection_managers.pop(self._manager_key, None)
        MCPEnvironment._safe_stop_manager(self._connection_manager)
        if MCPEnvironment._connection_manager is self._connection_manager:
            MCPEnvironment._connection_manager = None
        self._connection_manager = None

    @staticmethod
    def cleanup_global_resources():
        """Clean up global connection manager."""
        with MCPEnvironment._manager_lock:
            for manager in MCPEnvironment._connection_managers.values():
                MCPEnvironment._safe_stop_manager(manager)
            MCPEnvironment._connection_managers = {}
            MCPEnvironment._safe_stop_manager(MCPEnvironment._connection_manager)
            MCPEnvironment._connection_manager = None

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> "MCPEnvironment":
        env_args = dict(env_args)
        mcp_server_command = env_args.pop("mcp_server_command", None)
        mcp_server_args = env_args.pop("mcp_server_args", None)
        mcp_server_env = env_args.pop("mcp_server_env", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        share_mcp_manager = env_args.pop("share_mcp_manager", True)

        tools_py = env_args.get("tools_py")
        if mcp_server_command is None and tools_py:
            tools_path = Path(tools_py)
            if tools_path.exists() and tools_path.is_file():
                server_script = MCPEnvironment._ensure_server_script(tools_path.parent)
                mcp_server_command = sys.executable
                mcp_server_args = [str(server_script)]
            else:
                warnings.warn(f"tools.py not found: {tools_py}. MCP server will not start.", stacklevel=2)

        return MCPEnvironment(
            task=env_args,
            mcp_server_command=mcp_server_command,
            mcp_server_args=mcp_server_args,
            mcp_server_env=mcp_server_env,
            max_steps=max_steps,
            reward_fn=reward_fn,
            share_mcp_manager=share_mcp_manager,
        )
