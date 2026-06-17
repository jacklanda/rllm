import logging

from rllm.rewards.verifier_reward import _load_tools_from_tools_py, verifier_reward_fn


def test_load_tools_from_tools_py_silences_mcp_registration_warning(tmp_path, caplog):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "import logging\n"
        "logging.getLogger('mcp.server.fastmcp.tools.tool_manager').warning('Tool already exists: duplicate_tool')\n"
        "def duplicate_tool():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        tools = _load_tools_from_tools_py(str(tools_py))

    assert "duplicate_tool" in tools
    assert tools["duplicate_tool"]() == {"result": "ok"}
    assert "Tool already exists" not in caplog.text


def test_load_tools_from_tools_py_preserves_mcp_instance_after_bare_import(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        def decorate(fn):\n"
        "            return fn\n"
        "        return decorate\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "@mcp.tool(description='before')\n"
        "def before_import():\n"
        "    return 'before'\n"
        "import mcp\n"
        "@mcp.tool(description='after')\n"
        "def after_import():\n"
        "    return 'after'\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["before_import"]() == {"result": "before"}
    assert tools["after_import"]() == {"result": "after"}


def test_load_tools_from_tools_py_drops_self_imported_mcp(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        return lambda fn: fn\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "from tools import mcp\n"
        "@mcp.tool(description='after self import')\n"
        "def after_self_import():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["after_self_import"]() == {"result": "ok"}


def test_load_tools_from_tools_py_repairs_numeric_line_prefix(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "def submit_result_difficulty_3(result: dict):\n"
        "    if not isinstance(result, dict):\n"
        "       1 raise ValueError('result must be dict')\n"
        "    return result\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["submit_result_difficulty_3"]({"ok": True}) == {"result": {"ok": True}}


def test_load_tools_from_tools_py_preserves_valid_numeric_generator_prefix(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "def count_keywords(text: str) -> int:\n"
        "    keywords = ['alpha', 'gamma']\n"
        "    return sum(\n"
        "        1 for keyword in keywords if keyword in text\n"
        "    )\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["count_keywords"]("alpha beta alpha") == {"result": 1}


def test_load_tools_from_tools_py_maps_bare_tool_to_mcp_tool(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        def decorate(fn):\n"
        "            return fn\n"
        "        return decorate\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "@mcp.tool(description='normal')\n"
        "def normal_tool():\n"
        "    return 'normal'\n"
        "@tool(description='bare')\n"
        "def submit_result_difficulty_3(result: dict):\n"
        "    return result\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["submit_result_difficulty_3"]({"ok": True}) == {"result": {"ok": True}}
    assert "tool" not in tools


def test_load_tools_from_tools_py_supports_tool_decorator_variable(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        def decorate(fn):\n"
        "            return fn\n"
        "        return decorate\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "TOOL_DECORATOR = tool(description='submit')\n"
        "@TOOL_DECORATOR\n"
        "def submit_result_difficulty_3(result: list[dict]):\n"
        "    return result\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["submit_result_difficulty_3"]([{"ok": True}]) == {"result": [{"ok": True}]}
    assert "tool" not in tools


def test_load_tools_from_tools_py_filters_typing_aliases(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "from typing import Any, Dict, List\n"
        "def useful_tool(value: Dict[str, Any]) -> List[str]:\n"
        "    return list(value)\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert set(tools) == {"useful_tool"}
    assert tools["useful_tool"]({"ok": True}) == {"result": ["ok"]}


def test_load_tools_from_tools_py_repairs_parent_parent_base_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "content.json").write_text('{"ok": true}', encoding="utf-8")
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        return lambda fn: fn\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "BASE_DIR = Path(__file__).parent.parent\n"
        "@mcp.tool(description='read data')\n"
        "def read_data():\n"
        "    with open(BASE_DIR / 'data' / 'content.json', encoding='utf-8') as f:\n"
        "        return json.load(f)\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["read_data"]() == {"result": {"ok": True}}


def test_load_tools_from_tools_py_repairs_non_ascii_numeric_default(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "def count_tool(min_bullet_count: int =常说0):\n"
        "    return min_bullet_count\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["count_tool"]() == {"result": 0}


def test_load_tools_from_tools_py_repairs_roman_numeric_default(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "def count_tool(min_bullet_count: int = III):\n"
        "    return min_bullet_count\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["count_tool"]() == {"result": 3}


def test_load_tools_from_tools_py_repairs_bare_identifier_defaults(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "def defaults(\n"
        "    min_count: int = card0,\n"
        "    min_length: int = case100,\n"
        "    section_length: int = II0,\n"
        "    label: str = chatgpt,\n"
        "    enabled: bool = BOOLEAN,\n"
        "    ratio: float = float('inf'),\n"
        "):\n"
        "    return [min_count, min_length, section_length, label, enabled, ratio]\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["defaults"]() == {"result": [0, 100, 20, "chatgpt", False, float("inf")]}


def test_load_tools_from_tools_py_repairs_duplicated_base_dir_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "content.json").write_text('{"ok": true}', encoding="utf-8")
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "BASE_DIR = Path(__file__).parent\n"
        "def read_data():\n"
        "    file_path = Path(__file__).parent / 'data' / BASE_DIR / 'data' / 'content.json'\n"
        "    with open(file_path, encoding='utf-8') as f:\n"
        "        return json.load(f)\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["read_data"]() == {"result": {"ok": True}}


def test_load_tools_from_tools_py_drops_unavailable_mcp_tool_import(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        return lambda fn: fn\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "from mcp import tool\n"
        "@mcp.tool(description='ok')\n"
        "def useful_tool():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["useful_tool"]() == {"result": "ok"}


def test_load_tools_from_tools_py_repairs_relative_base_dir_import(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "from pathlib import Path\n"
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        return lambda fn: fn\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "from . import BASE_DIR\n"
        "@mcp.tool(description='base dir')\n"
        "def base_dir_name():\n"
        "    return BASE_DIR.name\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["base_dir_name"]() == {"result": tmp_path.name}


def test_load_tools_from_tools_py_repairs_tools_base_dir_import(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "from pathlib import Path\n"
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        return lambda fn: fn\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "from tools import BASE_DIR\n"
        "@mcp.tool(description='base dir')\n"
        "def base_dir_name():\n"
        "    return BASE_DIR.name\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["base_dir_name"]() == {"result": tmp_path.name}


def test_load_tools_from_tools_py_repairs_missing_enum_all_default(tmp_path):
    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "from enum import Enum\n"
        "class _MCP:\n"
        "    def tool(self, description=None):\n"
        "        return lambda fn: fn\n"
        "FastMCP = _MCP\n"
        "mcp = FastMCP()\n"
        "class Timeframe(Enum):\n"
        "    CURRENT = 'current'\n"
        "    FUTURE = 'future'\n"
        "@mcp.tool(description='bad enum default')\n"
        "def timeframe_tool(timeframe: Timeframe = Timeframe.ALL):\n"
        "    return timeframe.value\n",
        encoding="utf-8",
    )

    tools = _load_tools_from_tools_py(str(tools_py))

    assert tools["timeframe_tool"]() == {"result": "current"}


def test_mcp_step_penalty_can_be_disabled_for_offline_rs(monkeypatch):
    task_info = {
        "verifier": {"verification_code": "def verify(tools, answer):\n    return {'passed': True}\n"},
        "tool_call_stats": {
            "submit_called": True,
            "non_submit_tool_calls": 19,
            "step_count": 20,
            "distinct_successful_tools": 8,
        },
    }

    monkeypatch.delenv("RLLM_MCP_DISABLE_STEP_PENALTY", raising=False)
    default_result = verifier_reward_fn(task_info=task_info, action="{}")
    assert default_result.reward == 0.0
    assert default_result.metadata["reward/raw_step_penalty"] == -2.4000000000000004
    assert default_result.metadata["reward/step_penalty_disabled"] == 0

    monkeypatch.setenv("RLLM_MCP_DISABLE_STEP_PENALTY", "True")
    rs_result = verifier_reward_fn(task_info=task_info, action="{}")
    assert rs_result.reward == 1.0
    assert rs_result.metadata["reward/raw_step_penalty"] == -2.4000000000000004
    assert rs_result.metadata["reward/step_penalty"] == 0.0
    assert rs_result.metadata["reward/step_penalty_disabled"] == 1
