import asyncio
import json
import logging
import os
import queue
import sys
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

    def start(self):
        """Start the connection manager thread.

        Uses the class-level ``_spawn_lock`` to serialise MCP subprocess
        spawning across all instances, preventing the *anyio* "Racing with
        another loop to spawn a process" error.
        """
        if self.running:
            return

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            with MCPConnectionManager._spawn_lock:
                self.running = True
                self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
                self.worker_thread.start()

                # Wait for initialization
                response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
                self.request_queue.put(("init", None, response_queue))
                try:
                    result = response_queue.get(timeout=60)
                except queue.Empty:
                    last_error = "Timed out waiting for MCP server initialization"
                    self._stop_no_lock()
                    continue
                if result[0] == "error":
                    last_error = result[1]
                    self._stop_no_lock()
                    if "Racing" in str(last_error):
                        import time

                        time.sleep(0.5 * (attempt + 1))
                        continue
                    raise Exception(f"Failed to initialize MCP connection: {last_error}")
                return  # Success

        raise Exception(f"Failed to initialize MCP connection after {max_retries} retries: {last_error}")

    def _stop_no_lock(self):
        """Internal stop that doesn't acquire locks — used during retry."""
        self.running = False
        self.request_queue.put(("stop", None, None))
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            self.worker_thread = None
        # Clear any leftover state for a clean retry
        self.session = None
        self.tool_map = {}
        # Drain leftover items from queue
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        """Stop the connection manager thread."""
        if not self.running:
            return

        self.running = False
        self.request_queue.put(("stop", None, None))
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
            self.worker_thread = None
        self.session = None
        self.tool_map = {}

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
        result = response_queue.get(timeout=30)
        if result[0] == "error":
            raise Exception(f"Tool execution failed: {result[1]}")
        return result[1]  # type: ignore

    def _run_worker(self):
        """Worker thread that runs the asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._worker_loop())
        finally:
            if self.session:
                try:
                    self.loop.run_until_complete(self._cleanup())
                except Exception:
                    pass
            if self.loop:
                self.loop.close()

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
                        await self._initialize_connection()
                        if response_queue:
                            response_queue.put(("success", self.tool_map))
                    except Exception as e:
                        if response_queue:
                            response_queue.put(("error", str(e)))

                elif command == "execute":
                    try:
                        result = await self._execute_tools(data)
                        if response_queue:
                            response_queue.put(("success", result))
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
        # Route subprocess stderr to /dev/null: stdio_client forwards it to
        # the parent's stderr by default, leaking MCP server warnings (e.g.
        # "Tool already exists" from duplicate @mcp.tool defs in tools.py)
        # into Ray worker logs.
        self._mcp_errlog = self.exit_stack.enter_context(open(os.devnull, "w"))
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
        if hasattr(self, "exit_stack") and self.exit_stack:
            await self.exit_stack.aclose()


class MCPEnvironment(BaseEnv):
    """
    An environment for MCP-based tools that provides questions and evaluates responses.
    Uses a dedicated connection manager to avoid asyncio context issues.
    """

    # Class-level pool to share managers across instances with the same server config.
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
                self._manager_key = manager_key

    @staticmethod
    def _ensure_server_script(assets_dir: Path) -> Path:
        server_script = assets_dir / "mcp_server.py"
        server_script.write_text(
            "import sys\n"
            "import logging\n"
            "_MCP_LOGGERS = ('mcp', 'mcp.server', 'mcp.server.fastmcp', 'mcp.server.fastmcp.tools', 'mcp.server.fastmcp.tools.tool_manager')\n"
            "for _name in _MCP_LOGGERS:\n"
            "    _lg = logging.getLogger(_name)\n"
            "    _lg.setLevel(logging.CRITICAL)\n"
            "    _lg.propagate = False\n"
            "    _lg.disabled = True\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "import tools\n"
            "tools.mcp.run()\n",
            encoding="utf-8",
        )
        return server_script

    def _build_manager_key(self) -> tuple[Any, ...]:
        env_items = tuple(sorted(self.mcp_server_env.items())) if self.mcp_server_env else None
        base_key = (self.mcp_server_command, tuple(self.mcp_server_args), env_items)
        if self.share_mcp_manager:
            return base_key
        return ("isolated", uuid.uuid4().hex, base_key)

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        self._non_submit_tool_calls = 0
        self._submit_without_tool_retries = 0
        obs = dict(self.task) if isinstance(self.task, dict) else {}
        if self._connection_manager and self._connection_manager.tool_map:
            tools_json = []
            seen = set()
            for tool in self._connection_manager.tool_map.values():
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
                    if self._connection_manager is not None:
                        tool_outputs = self._connection_manager.execute_tool_calls([submit_result_tool_call])
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
            if self._connection_manager is not None:
                tool_outputs = self._connection_manager.execute_tool_calls(tool_calls)
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
        self._connection_manager.stop()
        self._connection_manager = None

    @staticmethod
    def cleanup_global_resources():
        """Clean up global connection manager."""
        with MCPEnvironment._manager_lock:
            for manager in MCPEnvironment._connection_managers.values():
                manager.stop()
            MCPEnvironment._connection_managers = {}

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
