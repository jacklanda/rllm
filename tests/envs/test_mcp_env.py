import asyncio
import subprocess
import sys
import threading
from unittest.mock import Mock, patch

import pytest

from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
from rllm.rewards.reward_fn import RewardFunction, RewardOutput


class MockRewardFunction(RewardFunction):
    """Mock reward function for testing."""

    def __call__(self, task_info, action, **kwargs):
        # Simple mock reward based on action content
        reward = 1.0 if "correct" in str(action).lower() else 0.0
        metadata = {"evaluated": True, "action": action}
        return RewardOutput(reward=reward, metadata=metadata)


class TestMCPConnectionManager:
    """Test suite for MCPConnectionManager class."""

    def test_init(self):
        """Test MCPConnectionManager initialization."""
        manager = MCPConnectionManager(mcp_server_command="test_command", mcp_server_args=["--arg1", "--arg2"], mcp_server_env={"VAR": "value"})

        assert manager.mcp_server_command == "test_command"
        assert manager.mcp_server_args == ["--arg1", "--arg2"]
        assert manager.mcp_server_env == {"VAR": "value"}
        assert manager.running is False
        assert manager.tool_map == {}

    def test_init_default_args(self):
        """Test MCPConnectionManager initialization with default arguments."""
        manager = MCPConnectionManager("test_command")

        assert manager.mcp_server_command == "test_command"
        assert manager.mcp_server_args == []
        assert manager.mcp_server_env is None

    @patch("threading.Thread")
    @patch("queue.Queue")
    def test_start(self, mock_queue, mock_thread):
        """Test starting the connection manager."""
        # Mock the response queue to simulate successful initialization
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("success", {"tool1": "mock_tool"})
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")

        # Mock the worker thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        manager.start()

        assert manager.running is True
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    @patch("threading.Thread")
    @patch("queue.Queue")
    def test_start_initialization_error(self, mock_queue, mock_thread):
        """Test starting the connection manager with initialization error."""
        # Mock the response queue to simulate initialization error
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("error", "Connection failed")
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")

        # Mock the worker thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        with pytest.raises(Exception, match="Failed to initialize MCP connection"):
            manager.start()

    def test_stop_not_running(self):
        """Test stopping the connection manager when not running."""
        manager = MCPConnectionManager("test_command")

        # Should not raise any errors
        manager.stop()
        assert manager.running is False

    @patch("threading.Thread")
    def test_stop_running(self, mock_thread):
        """Test stopping a running connection manager."""
        manager = MCPConnectionManager("test_command")
        manager.running = True
        manager.worker_thread = Mock()

        manager.stop()

        assert manager.running is False
        manager.worker_thread.join.assert_called_once_with(timeout=5)

    @patch("queue.Queue")
    def test_execute_tool_calls_not_running(self, mock_queue):
        """Test executing tool calls when manager is not running."""
        manager = MCPConnectionManager("test_command")

        with pytest.raises(Exception, match="Connection manager not running"):
            manager.execute_tool_calls([])

    @patch("queue.Queue")
    def test_execute_tool_calls_success(self, mock_queue):
        """Test successful tool call execution."""
        # Mock the response queue
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("success", {"call_1": "result"})
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")
        manager.running = True

        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]
        result = manager.execute_tool_calls(tool_calls)

        assert result == {"call_1": "result"}

    @patch("queue.Queue")
    def test_execute_tool_calls_error(self, mock_queue):
        """Test tool call execution with error."""
        # Mock the response queue
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("error", "Tool execution failed")
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")
        manager.running = True

        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]

        with pytest.raises(Exception, match="Tool execution failed"):
            manager.execute_tool_calls(tool_calls)

    @patch("threading.Thread")
    def test_start_retries_transient_connection_closed(self, mock_thread):
        """Transient stdio startup closures should be retried before failing rollout reset."""
        manager = MCPConnectionManager("test_command")

        first_response = Mock()
        first_response.get.return_value = ("error", "Connection closed")
        second_response = Mock()
        second_response.get.return_value = ("success", {"tool1": "mock_tool"})
        thread_instance = Mock()
        thread_instance.is_alive.return_value = False
        mock_thread.return_value = thread_instance

        with patch("queue.Queue", side_effect=[first_response, second_response]):
            manager.start()

        assert manager.running is True
        assert mock_thread.call_count == 2
        assert thread_instance.start.call_count == 2

    def test_start_initialization_cancelled(self, monkeypatch):
        """Cancelled MCP initialization should be reported to the starter thread."""

        async def raise_cancelled(self):
            raise asyncio.CancelledError("cancel scope closed")

        monkeypatch.setattr(MCPConnectionManager, "_initialize_connection", raise_cancelled)

        manager = MCPConnectionManager("test_command")
        with pytest.raises(Exception, match="MCP initialization cancelled"):
            manager.start()

        assert manager.running is False

    def test_start_initialization_error_cleans_partial_resources(self, monkeypatch):
        """Failed initialization should close resources opened before session setup."""
        cleaned = False

        class FakeExitStack:
            async def aclose(self):
                nonlocal cleaned
                cleaned = True

        async def fail_after_partial_init(self):
            self.exit_stack = FakeExitStack()
            self.tool_map = {"stale": object()}
            raise RuntimeError("partial init failed")

        monkeypatch.setattr(MCPConnectionManager, "_initialize_connection", fail_after_partial_init)

        manager = MCPConnectionManager("test_command")
        with pytest.raises(Exception, match="partial init failed"):
            manager.start()

        assert cleaned is True
        assert manager.running is False
        assert manager.tool_map == {}

    def test_active_server_limit_blocks_until_slot_released(self, monkeypatch):
        """The MCP active-server cap should apply backpressure instead of over-opening FDs."""
        monkeypatch.setenv("RLLM_MCP_MAX_ACTIVE_SERVERS", "1")
        MCPConnectionManager._active_server_semaphore = None
        MCPConnectionManager._active_server_limit = None

        first = MCPConnectionManager("test_command")
        second = MCPConnectionManager("test_command")
        acquired = threading.Event()

        try:
            first._acquire_server_slot()

            thread = threading.Thread(target=lambda: (second._acquire_server_slot(), acquired.set()))
            thread.start()

            assert acquired.wait(timeout=0.1) is False
            first._release_server_slot()
            assert acquired.wait(timeout=1.0) is True
            thread.join(timeout=1)
        finally:
            first._release_server_slot()
            second._release_server_slot()
            MCPConnectionManager._active_server_semaphore = None
            MCPConnectionManager._active_server_limit = None

    def test_active_server_limit_derived_from_fd_budget(self, monkeypatch):
        monkeypatch.delenv("RLLM_MCP_MAX_ACTIVE_SERVERS", raising=False)
        monkeypatch.setenv("RLLM_MCP_MAX_ACTIVE_FDS", "18")
        monkeypatch.setenv("RLLM_MCP_FDS_PER_SERVER", "6")
        MCPConnectionManager._active_server_semaphore = None
        MCPConnectionManager._active_server_limit = None

        try:
            assert MCPConnectionManager._get_active_server_semaphore() is not None
            assert MCPConnectionManager._active_server_limit == 3
        finally:
            MCPConnectionManager._active_server_semaphore = None
            MCPConnectionManager._active_server_limit = None

    def test_active_server_limit_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("RLLM_MCP_MAX_ACTIVE_SERVERS", raising=False)
        monkeypatch.delenv("RLLM_MCP_MAX_ACTIVE_FDS", raising=False)
        MCPConnectionManager._active_server_semaphore = None
        MCPConnectionManager._active_server_limit = None

        assert MCPConnectionManager._get_active_server_semaphore() is None
        assert MCPConnectionManager._active_server_limit is None

    def test_fd_throttle_waits_only_at_threshold(self, monkeypatch):
        monkeypatch.setenv("RLLM_MCP_FD_THROTTLE_THRESHOLD", "10")
        fd_state = {"count": 9}

        monkeypatch.setattr(MCPConnectionManager, "_current_fd_count", staticmethod(lambda: fd_state["count"]))

        MCPConnectionManager._wait_for_fd_headroom()

        fd_state["count"] = 10
        passed = threading.Event()

        thread = threading.Thread(target=lambda: (MCPConnectionManager._wait_for_fd_headroom(), passed.set()))
        thread.start()

        assert passed.wait(timeout=0.1) is False
        fd_state["count"] = 9
        MCPConnectionManager._notify_fd_headroom()
        assert passed.wait(timeout=1.0) is True
        thread.join(timeout=1)


class TestMCPEnvironment:
    """Test suite for MCPEnvironment class."""

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_init_default(self, mock_init, mock_start):
        """Test MCPEnvironment initialization with default parameters."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        assert env.step_count == 0
        assert env.max_steps == 10
        assert env.task is None
        assert env.reward_fn is not None  # Should use zero_reward

    def test_ensure_server_script_silences_mcp_run_warnings(self, tmp_path):
        """Generated MCP server wrapper should suppress duplicate-tool logs through run()."""
        (tmp_path / "tools.py").write_text(
            "import logging\n"
            "class _MCP:\n"
            "    def run(self):\n"
            "        logging.getLogger('mcp.server.fastmcp.tools.tool_manager').warning('Tool already exists: duplicate_tool')\n"
            "mcp = _MCP()\n",
            encoding="utf-8",
        )
        server_script = MCPEnvironment._ensure_server_script(tmp_path)

        result = subprocess.run([sys.executable, str(server_script)], cwd=tmp_path, capture_output=True, text=True, timeout=10)

        assert result.returncode == 0
        assert "Tool already exists" not in result.stdout
        assert "Tool already exists" not in result.stderr

    def test_ensure_server_script_preserves_mcp_instance_after_bare_import(self, tmp_path):
        """Generated tools.py files may accidentally re-import mcp after creating the server."""
        (tmp_path / "tools.py").write_text(
            "class _MCP:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        self.names = []\n"
            "    def tool(self, description=None):\n"
            "        def decorate(fn):\n"
            "            self.names.append(fn.__name__)\n"
            "            return fn\n"
            "        return decorate\n"
            "    def run(self):\n"
            "        print(','.join(self.names))\n"
            "FastMCP = _MCP\n"
            "mcp = FastMCP('Tools')\n"
            "@mcp.tool(description='before')\n"
            "def before_import():\n"
            "    return 'before'\n"
            "import mcp\n"
            "@mcp.tool(description='after')\n"
            "def after_import():\n"
            "    return 'after'\n",
            encoding="utf-8",
        )
        server_script = MCPEnvironment._ensure_server_script(tmp_path)

        result = subprocess.run([sys.executable, str(server_script)], cwd=tmp_path, capture_output=True, text=True, timeout=10)

        assert result.returncode == 0
        assert result.stdout.strip() == "before_import,after_import"
        assert "AttributeError" not in result.stderr

    def test_ensure_server_script_skips_relative_self_import_mcp(self, tmp_path):
        """Generated tools.py files can contain package-style self imports."""
        (tmp_path / "tools.py").write_text(
            "class _MCP:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        self.names = []\n"
            "    def tool(self, description=None):\n"
            "        def decorate(fn):\n"
            "            self.names.append(fn.__name__)\n"
            "            return fn\n"
            "        return decorate\n"
            "    def run(self):\n"
            "        print(','.join(self.names))\n"
            "FastMCP = _MCP\n"
            "mcp = FastMCP('Tools')\n"
            "@mcp.tool(description='before')\n"
            "def before_relative_import():\n"
            "    return 'before'\n"
            "from .tools import mcp\n"
            "@mcp.tool(description='after')\n"
            "def after_relative_import():\n"
            "    return 'after'\n",
            encoding="utf-8",
        )
        server_script = MCPEnvironment._ensure_server_script(tmp_path)

        result = subprocess.run([sys.executable, str(server_script)], cwd=tmp_path, capture_output=True, text=True, timeout=10)

        assert result.returncode == 0
        assert result.stdout.strip() == "before_relative_import,after_relative_import"
        assert "ImportError" not in result.stderr

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_init_with_custom_parameters(self, mock_init, mock_start):
        """Test MCPEnvironment initialization with custom parameters."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        max_steps = 5

        env = MCPEnvironment(task=task, mcp_server_command="test_command", mcp_server_args=["--arg1"], mcp_server_env={"VAR": "value"}, reward_fn=reward_fn, max_steps=max_steps)

        assert env.task == task
        assert env.reward_fn == reward_fn
        assert env.max_steps == max_steps
        assert env.mcp_server_command == "test_command"
        assert env.mcp_server_args == ["--arg1"]
        assert env.mcp_server_env == {"VAR": "value"}

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_init_no_reward_function_warning(self, mock_init, mock_start):
        """Test that warning is issued when no reward function is provided."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        with pytest.warns(UserWarning, match="No reward function specified"):
            env = MCPEnvironment(mcp_server_command="test_command", reward_fn=None)
            assert env.reward_fn is not None  # Should use zero_reward

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_reset(self, mock_init, mock_start):
        """Test the reset method."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        env = MCPEnvironment(task=task, max_steps=5, mcp_server_command="test_command")

        # Set some non-initial state
        env.step_count = 3

        obs, info = env.reset()

        assert env.step_count == 0
        assert obs == task
        assert isinstance(info, dict)

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_string_action(self, mock_init, mock_start):
        """Test stepping with string action (should terminate)."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = "This is my final answer"
        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert reward == 0.0  # MockRewardFunction returns 0 for non-"correct" answers
        assert done is True
        assert info["response"] == action
        assert "metadata" in info

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_string_action_correct(self, mock_init, mock_start):
        """Test stepping with string action containing 'correct'."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = "The correct answer is 42"
        obs, reward, done, info = env.step(action)

        assert reward == 1.0  # MockRewardFunction returns 1 for "correct" answers
        assert done is True

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_finish_tool_call(self, mock_init, mock_start):
        """Test stepping with finish tool call."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": {"response": "Final answer"}}}]

        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert done is True
        assert info["response"] == action

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_regular_tool_calls(self, mock_init, mock_start):
        """Test stepping with regular tool calls."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        env = MCPEnvironment(task=task, mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager

        action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

        obs, reward, done, info = env.step(action)

        assert obs == {"tool_outputs": {"call_1": "Tool output"}}
        assert reward == 0
        assert done is False
        assert info["response"] == action
        mock_manager.execute_tool_calls.assert_called_once_with(action)

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_max_steps_termination(self, mock_init, mock_start):
        """Test that environment terminates when max_steps is reached."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(max_steps=2, mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager

        # Take steps until max_steps
        for i in range(2):
            action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

            obs, reward, done, info = env.step(action)

            if i == 1:  # Last step
                assert done is True
            else:
                assert done is False

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_dict_action(self, mock_init, mock_start):
        """Test stepping with dict action (converted to list)."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager

        action = {"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}

        obs, reward, done, info = env.step(action)

        # Should convert dict to list
        mock_manager.execute_tool_calls.assert_called_once_with([action])

    def test_close_with_connection(self):
        """Test close method with active connection."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            env = MCPEnvironment(mcp_server_command="test_command")

            # Should not raise any errors - close method does nothing
            env.close()

    def test_close_no_connection(self):
        """Test close method without connection."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            env = MCPEnvironment(mcp_server_command="test_command")

            # Should not raise any errors
            env.close()

    def test_cleanup_global_resources_no_manager(self):
        """Test cleanup_global_resources when no manager exists."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        # Should not raise any errors
        MCPEnvironment.cleanup_global_resources()

    def test_cleanup_global_resources_with_manager(self):
        """Test cleanup_global_resources with existing manager."""
        mock_manager = Mock()
        MCPEnvironment._connection_manager = mock_manager

        MCPEnvironment.cleanup_global_resources()

        mock_manager.stop.assert_called_once()
        assert MCPEnvironment._connection_manager is None

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_idx_property(self, mock_init, mock_start):
        """Test the idx property from BaseEnv."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")

        # Initially should be None
        assert env.idx is None

        # Should be able to set and get
        env.idx = 5
        assert env.idx == 5

    def test_is_multithread_safe(self):
        """Test the is_multithread_safe static method."""
        assert MCPEnvironment.is_multithread_safe() is True

    def test_from_dict(self):
        """Test creating environment from dictionary."""
        env_args = {
            "question": "Test question",
            "mcp_server_command": "test_command",
            "mcp_server_args": ["--arg1"],
            "mcp_server_env": {"VAR": "value"},
            "max_steps": 15,
            "reward_fn": MockRewardFunction(),
        }

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            # Clear any existing manager
            MCPEnvironment._connection_manager = None

            env = MCPEnvironment.from_dict(env_args)

            assert isinstance(env, MCPEnvironment)
            assert env.max_steps == 15
            assert env.task == {"question": "Test question"}
            assert env.mcp_server_command == "test_command"
            assert env.mcp_server_args == ["--arg1"]
            assert env.mcp_server_env == {"VAR": "value"}

    def test_from_dict_minimal(self):
        """Test creating environment from minimal dictionary."""
        env_args = {"question": "Test question"}

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            # Clear any existing manager
            MCPEnvironment._connection_manager = None

            env = MCPEnvironment.from_dict(env_args)

            assert isinstance(env, MCPEnvironment)
            assert env.task == {"question": "Test question"}
            assert env.max_steps == 10  # Default value

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_full_interaction_flow(self, mock_init, mock_start):
        """Test a complete interaction flow."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "What is the capital of France?"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, max_steps=3, mcp_server_command="test_command")

        # Reset environment
        obs, info = env.reset()
        assert obs == task
        assert env.step_count == 0

        # Step 1: Use a tool
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Paris is the capital of France"}
        MCPEnvironment._connection_manager = mock_manager

        action1 = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "capital of France"}}}]

        obs1, reward1, done1, info1 = env.step(action1)

        assert obs1 == {"tool_outputs": {"call_1": "Paris is the capital of France"}}
        assert reward1 == 0
        assert done1 is False
        assert env.step_count == 1

        # Step 2: Finish with answer
        action2 = "The correct answer is Paris"
        obs2, reward2, done2, info2 = env.step(action2)

        assert obs2 == {}
        assert reward2 == 1.0  # MockRewardFunction gives 1.0 for "correct"
        assert done2 is True
        assert info2["response"] == action2

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_finish_tool_call_with_arguments(self, mock_init, mock_start):
        """Test stepping with finish tool call containing arguments."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": {"response": "The correct final answer"}}}]

        obs, reward, done, info = env.step(action)

        assert reward == 1.0  # Should extract response from arguments
        assert done is True

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_tool_execution_error(self, mock_init, mock_start):
        """Test stepping when tool execution fails."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager to raise an error
        mock_manager = Mock()
        mock_manager.execute_tool_calls.side_effect = Exception("Tool execution failed")
        MCPEnvironment._connection_manager = mock_manager

        action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

        # Should not raise error but return empty tool outputs
        obs, reward, done, info = env.step(action)
        assert obs == {"tool_outputs": {}}

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_edge_cases(self, mock_init, mock_start):
        """Test various edge cases."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {}
        MCPEnvironment._connection_manager = mock_manager

        # Empty action list
        obs, reward, done, info = env.step([])
        assert obs == {"tool_outputs": {}}
        assert done is False

        # None action - check how it's handled
        try:
            obs, reward, done, info = env.step(None)
            # If it doesn't raise an error, that's also fine
            assert isinstance(obs, dict)
        except (TypeError, AttributeError):
            # This is expected behavior
            pass

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_reward_function_integration(self, mock_init, mock_start):
        """Test integration with different reward functions."""

        class CustomReward(RewardFunction):
            def __call__(self, task_info, action, **kwargs):
                score = len(str(action)) / 10.0  # Simple length-based reward
                return RewardOutput(reward=score, metadata={"length": len(str(action))})

        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test"}
        reward_fn = CustomReward()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")

        env.reset()

        # Test with different length responses
        short_action = "Short"
        long_action = "This is a much longer response that should get a higher reward"

        _, reward1, _, info1 = env.step(short_action)
        env.reset()
        _, reward2, _, info2 = env.step(long_action)

        assert reward2 > reward1  # Longer response should get higher reward
        assert info1["metadata"]["length"] < info2["metadata"]["length"]

    def test_connection_manager_thread_safety(self):
        """Test that connection manager handles thread safety correctly."""
        # Both environments should be able to access the class-level manager
        assert hasattr(MCPEnvironment, "_connection_manager")
        assert hasattr(MCPEnvironment, "_manager_lock")

    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    @patch.object(MCPConnectionManager, "start")
    def test_connection_manager_singleton_behavior(self, mock_start, mock_init):
        """Test that connection manager behaves as singleton."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        MCPEnvironment(mcp_server_command="test_command")
        MCPEnvironment(mcp_server_command="test_command")

        # Both environments should use the same manager
        assert MCPEnvironment._connection_manager is not None

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_malformed_tool_call_handling(self, mock_init, mock_start):
        """Test handling of malformed tool calls."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager

        # Malformed action (missing required fields)
        action = [{"id": "call_1"}]  # Missing function field

        obs, reward, done, info = env.step(action)

        # Should still process the action
        assert obs == {"tool_outputs": {"call_1": "Tool output"}}
        assert done is False
