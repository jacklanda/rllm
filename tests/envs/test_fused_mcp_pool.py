"""Tests for FusedEnv's refcounted MCP connection-manager pool.

Focus: the start-placeholder path that collapses step-1's
train_batch_size*rollout_n concurrent same-task acquires down to a single
server start per distinct tools_py, while different tasks still start
concurrently and refcounts stay balanced under churn.
"""

import threading
import time

import pytest

from rllm.environments.fused import fused as fused_mod
from rllm.environments.fused.fused import FusedEnv


class FakeManager:
    """Stand-in for MCPConnectionManager that records start/stop calls."""

    start_calls = 0
    _start_calls_lock = threading.Lock()

    def __init__(self, command, args, start_delay=0.05):
        self.command = command
        self.args = args
        self.running = False
        self._start_delay = start_delay
        self.stopped = False

    def start(self):
        # Simulate the ~1s handshake so racers actually overlap.
        time.sleep(self._start_delay)
        with FakeManager._start_calls_lock:
            FakeManager.start_calls += 1
        self.running = True

    def stop(self):
        self.stopped = True
        self.running = False


@pytest.fixture(autouse=True)
def _clean_pool(monkeypatch):
    """Reset the class-level pool and patch in FakeManager for each test."""
    FusedEnv._mcp_pool.clear()
    FusedEnv._mcp_pool_refcount.clear()
    FusedEnv._mcp_pool_starting.clear()
    FakeManager.start_calls = 0
    monkeypatch.setattr(fused_mod, "MCPConnectionManager", FakeManager)
    yield
    FusedEnv._mcp_pool.clear()
    FusedEnv._mcp_pool_refcount.clear()
    FusedEnv._mcp_pool_starting.clear()


def _acquire(key, results, idx):
    mgr = FusedEnv._acquire_mcp_manager(key, "python", [key])
    results[idx] = mgr


def test_concurrent_same_key_starts_one_server():
    """8 rollouts of the same task -> exactly one server started, all share it."""
    key = "/tools/task_a/mcp_server.py"
    n = 8
    results = [None] * n
    threads = [threading.Thread(target=_acquire, args=(key, results, i)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeManager.start_calls == 1, "should start exactly one server for one tools_py"
    assert all(r is results[0] for r in results), "all rollouts must share the same manager"
    assert FusedEnv._mcp_pool_refcount[key] == n
    assert results[0].running is True


def test_distinct_keys_start_independently():
    """Different tasks each get their own server."""
    keys = [f"/tools/task_{i}/mcp_server.py" for i in range(4)]
    results = [None] * len(keys)
    threads = [threading.Thread(target=_acquire, args=(k, results, i)) for i, k in enumerate(keys)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeManager.start_calls == len(keys)
    assert len({id(r) for r in results}) == len(keys)
    for k in keys:
        assert FusedEnv._mcp_pool_refcount[k] == 1


def test_release_stops_only_at_zero():
    key = "/tools/task_b/mcp_server.py"
    n = 5
    results = [None] * n
    threads = [threading.Thread(target=_acquire, args=(key, results, i)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    mgr = results[0]
    assert FusedEnv._mcp_pool_refcount[key] == n

    for i in range(n - 1):
        FusedEnv._release_mcp_manager(key)
        assert mgr.stopped is False, f"must not stop while {n - 1 - i} refs remain"
        assert FusedEnv._mcp_pool_refcount[key] == n - 1 - i

    FusedEnv._release_mcp_manager(key)
    assert mgr.stopped is True
    assert key not in FusedEnv._mcp_pool
    assert key not in FusedEnv._mcp_pool_refcount


def test_start_failure_lets_a_waiter_retry(monkeypatch):
    """If the first starter fails, a waiting rollout retries and starts its own."""
    key = "/tools/task_c/mcp_server.py"

    calls = {"n": 0}
    calls_lock = threading.Lock()

    class FlakyManager(FakeManager):
        def start(self):
            with calls_lock:
                calls["n"] += 1
                first = calls["n"] == 1
            time.sleep(0.05)
            if first:
                raise RuntimeError("simulated startup failure")
            self.running = True

    monkeypatch.setattr(fused_mod, "MCPConnectionManager", FlakyManager)

    results = [None] * 2
    errors = [None] * 2

    def acquire(idx):
        try:
            results[idx] = FusedEnv._acquire_mcp_manager(key, "python", [key])
        except Exception as e:  # noqa: BLE001
            errors[idx] = e

    threads = [threading.Thread(target=acquire, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # One acquire saw the failure; the other (or a retry) ended with a live mgr.
    live = [r for r in results if r is not None and getattr(r, "running", False)]
    assert len(live) >= 1, "a waiter should retry and obtain a live manager"
    assert FusedEnv._mcp_pool[key].running is True


def test_wait_timeout_zero_does_not_spawn_duplicate_starter(monkeypatch):
    """Default fused setting waits for queued same-key starter instead of duplicating it."""
    key = "/tools/task_d/mcp_server.py"

    class SlowManager(FakeManager):
        def __init__(self, command, args, start_delay=0.2):
            super().__init__(command, args, start_delay=start_delay)

    monkeypatch.setattr(fused_mod, "MCPConnectionManager", SlowManager)
    monkeypatch.setattr(FusedEnv, "_MCP_START_WAIT_TIMEOUT", 0.0)

    results = [None] * 2
    threads = [threading.Thread(target=_acquire, args=(key, results, i)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeManager.start_calls == 1
    assert results[0] is results[1]
    assert FusedEnv._mcp_pool_refcount[key] == 2
