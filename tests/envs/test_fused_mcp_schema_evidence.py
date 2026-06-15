from types import SimpleNamespace

from rllm.environments.fused.fused import (
    FusedEnv,
    _compact_mcp_tool_output,
    _build_evidence_to_field_trace,
    _extract_answer_schema,
    _validate_submission_schema,
)


def test_extract_answer_schema_and_validate_required_object():
    question = """
    Do the task.

    Below is the answer submission format requirement:

    {
      "type": "object",
      "properties": {
        "analysis_summary": {"type": "string"},
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
          }
        }
      },
      "required": ["analysis_summary", "items"]
    }
    """
    schema = _extract_answer_schema(question)

    assert schema is not None
    missing = _validate_submission_schema({"items": []}, schema)
    assert missing["passed"] is False
    assert any("analysis_summary" in err for err in missing["errors"])
    assert any("items" in err and "empty" in err for err in missing["errors"])

    valid = _validate_submission_schema({"analysis_summary": "ok", "items": [{"name": "A"}]}, schema)
    assert valid["passed"] is True


def test_validate_submission_schema_rejects_empty_array_and_item_missing_key():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    }

    empty = _validate_submission_schema([], schema)
    assert empty["passed"] is False
    assert "array submission must not be empty" in empty["errors"][0]

    missing_item_key = _validate_submission_schema([{}], schema)
    assert missing_item_key["passed"] is False
    assert any("$.[0].title" in err or "$[0].title" in err for err in missing_item_key["errors"])


def test_validate_submission_schema_accepts_integer_fields_and_coerces_lossless_numbers():
    schema = {
        "type": "object",
        "properties": {
            "total": {"type": "integer"},
            "nested": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        },
        "required": ["total", "nested"],
    }

    check = _validate_submission_schema({"total": 0.0, "nested": {"count": "2.0"}}, schema)

    assert check["passed"] is True
    assert check["normalized_payload"] == {"total": 0, "nested": {"count": 2}}


def test_validate_submission_schema_treats_integer_as_number_compatible():
    schema = {
        "type": "object",
        "properties": {"ratio": {"type": "number"}},
        "required": ["ratio"],
    }

    check = _validate_submission_schema({"ratio": 1}, schema)

    assert check["passed"] is True


def test_mcp_finish_schema_self_check_blocks_invalid_submission():
    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 0
    env._mcp_last_schema_self_check = {}

    obs, reward, done, info = FusedEnv._handle_mcp_finish(env, SimpleNamespace(parameters={"result": "{}"}))

    assert done is False
    assert reward == 0.0
    assert env._mcp_answer == ""
    assert env._mcp_schema_self_check_failures == 1
    assert info["mcp/schema_self_check_failed"] == 1
    assert "Schema self-check failed" in obs


def test_mcp_finish_terminates_after_repeated_schema_self_check_failures():
    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 1
    env._mcp_last_schema_self_check = {}

    obs, reward, done, info = FusedEnv._handle_mcp_finish(env, SimpleNamespace(parameters={"result": "{}"}))

    assert done is True
    assert reward == 0.0
    assert env._mcp_answer == ""
    assert info["termination_reason"] == "MCP_SCHEMA_SELF_CHECK_EXCEEDED"
    assert "failed too many times" in obs


def test_mcp_finish_accepts_valid_submission_after_self_check():
    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 0
    env._mcp_last_schema_self_check = {}

    obs, reward, done, info = FusedEnv._handle_mcp_finish(env, SimpleNamespace(parameters={"result": '{"analysis_summary": "ok"}'}))

    assert done is True
    assert reward == 0.0
    assert info == {}
    assert obs == "Your answer has been submitted."
    assert env._mcp_answer == '{"analysis_summary": "ok"}'
    assert env._mcp_last_schema_self_check["passed"] is True


def test_compact_mcp_tool_output_preserves_head_and_tail_with_marker():
    text = "\n".join(f"line {i}" for i in range(30))

    compact = _compact_mcp_tool_output(text, char_limit=120, line_limit=10)

    assert "line 0" in compact
    assert "line 29" in compact
    assert "truncated" in compact
    assert len(compact) <= 180


def test_evidence_to_field_trace_maps_answer_field_to_tool_output():
    answer = {"items": [{"name": "JFR CPU Load", "metric": "Thread CPU Load events"}]}
    tool_evidence = [
        {
            "tool": "get_finding_cpu_load_content",
            "arguments": {"search_term": "CPU"},
            "output": '{"heading": "Understanding Thread CPU Load Events", "text": "JFR records Thread CPU Load events to identify resource-intensive threads."}',
        }
    ]

    trace = _build_evidence_to_field_trace(answer, tool_evidence)

    assert trace["field_count"] == 2
    assert trace["mapped_count"] >= 1
    metric_mapping = next(m for m in trace["mappings"] if m["path"].endswith(".metric"))
    assert metric_mapping["matched"] is True
    assert metric_mapping["source_tool"] == "get_finding_cpu_load_content"
    assert metric_mapping["source_heading"] == "Understanding Thread CPU Load Events"
