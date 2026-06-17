from types import SimpleNamespace

from rllm.environments.fused.fused import (
    FusedEnv,
    _compact_mcp_tool_output,
    _build_evidence_to_field_trace,
    _extract_answer_schema,
    _submission_self_check_observation,
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
          "minItems": 1,
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
    assert missing["expected_top_level_types"] == ["object"]
    assert missing["required_top_level_keys"] == ["analysis_summary", "items"]
    assert any("analysis_summary" in err for err in missing["errors"])
    assert any("items" in err and "empty" in err for err in missing["errors"])

    valid = _validate_submission_schema({"analysis_summary": "ok", "items": [{"name": "A"}]}, schema)
    assert valid["passed"] is True


def test_validate_submission_schema_rejects_empty_array_and_item_missing_key():
    schema = {
        "type": "array",
        "minItems": 1,
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


def test_validate_submission_schema_allows_empty_arrays_unless_schema_requires_items():
    schema = {
        "type": "object",
        "properties": {
            "data_quality_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Must be empty if no issues.",
            },
        },
        "required": ["data_quality_flags"],
    }

    check = _validate_submission_schema({"data_quality_flags": []}, schema)

    assert check["passed"] is True


def test_validate_submission_schema_rejects_empty_array_when_min_items_positive():
    schema = {"type": "array", "minItems": 1, "items": {"type": "string"}}

    check = _validate_submission_schema([], schema)

    assert check["passed"] is False
    assert "array submission must not be empty" in check["errors"][0]


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


def test_submission_self_check_observation_includes_schema_hints():
    schema = {
        "type": "object",
        "properties": {"analysis_summary": {"type": "string"}},
        "required": ["analysis_summary"],
    }
    check = _validate_submission_schema({}, schema)

    obs = _submission_self_check_observation(check)

    assert "Expected top-level JSON type: object" in obs
    assert "Your submitted top-level JSON type: object" in obs
    assert "Required top-level keys: analysis_summary" in obs
    assert "not a JSON schema/meta object" in obs
    assert "Minimum answer shape from schema" in obs
    assert '{"analysis_summary": "..."}' in obs


def test_submission_self_check_observation_explains_wrong_top_level_type():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    }
    check = _validate_submission_schema('{"title": "A"}', schema)

    obs = _submission_self_check_observation(check)

    assert "Expected top-level JSON type: array" in obs
    assert "Your submitted top-level JSON type: string" in obs
    assert "first character of the submitted JSON value should be '['" in obs
    assert 'Minimum answer shape from schema: [{"title": "..."}]' in obs


def test_submission_self_check_observation_warns_schema_meta_submission():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    check = _validate_submission_schema({"type": "object", "properties": {}}, schema)

    obs = _submission_self_check_observation(check)

    assert "Do not submit the schema definition" in obs
    assert "Add these missing top-level keys: answer" in obs


def test_mcp_finish_schema_self_check_blocks_invalid_submission():
    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 0
    env._mcp_last_schema_self_check = {}
    env._mcp_submit_attempted = False
    env._mcp_submit_accepted = False
    env._mcp_submit_rejected = False

    obs, reward, done, info = FusedEnv._handle_mcp_finish(env, SimpleNamespace(parameters={"result": "{}"}))

    assert done is False
    assert reward == 0.0
    assert env._mcp_answer == ""
    assert env._mcp_submit_attempted is True
    assert env._mcp_submit_accepted is False
    assert env._mcp_submit_rejected is True
    assert env._mcp_schema_self_check_failures == 1
    assert info["mcp/schema_self_check_failed"] == 1
    assert "Schema self-check failed" in obs
    assert "Expected top-level JSON type: object" in obs
    assert "Required top-level keys: analysis_summary" in obs


def test_mcp_finish_terminates_after_repeated_schema_self_check_failures():
    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 1
    env._mcp_last_schema_self_check = {}
    env._mcp_submit_attempted = False
    env._mcp_submit_accepted = False
    env._mcp_submit_rejected = False

    obs, reward, done, info = FusedEnv._handle_mcp_finish(env, SimpleNamespace(parameters={"result": "{}"}))

    assert done is True
    assert reward == 0.0
    assert env._mcp_answer == ""
    assert env._mcp_submit_attempted is True
    assert env._mcp_submit_accepted is False
    assert env._mcp_submit_rejected is True
    assert info["termination_reason"] == "MCP_SCHEMA_SELF_CHECK_EXCEEDED"
    assert "failed too many times" in obs


def test_mcp_finish_accepts_valid_submission_after_self_check():
    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 0
    env._mcp_last_schema_self_check = {}
    env._mcp_submit_attempted = False
    env._mcp_submit_accepted = False
    env._mcp_submit_rejected = False

    obs, reward, done, info = FusedEnv._handle_mcp_finish(env, SimpleNamespace(parameters={"result": '{"analysis_summary": "ok"}'}))

    assert done is True
    assert reward == 0.0
    assert info == {}
    assert obs == "Your answer has been submitted."
    assert env._mcp_answer == '{"analysis_summary": "ok"}'
    assert env._mcp_submit_attempted is True
    assert env._mcp_submit_accepted is True
    assert env._mcp_submit_rejected is False
    assert env._mcp_last_schema_self_check["passed"] is True


def test_mcp_submit_result_does_not_execute_server_side_submit_tool():
    class ExplodingManager:
        def execute_tool_calls(self, _tool_calls):
            raise AssertionError("submit_result_difficulty tool should not be executed")

    env = object.__new__(FusedEnv)
    env.total_steps = 0
    env._mcp_answer = ""
    env._mcp_answer_schema = {"type": "object", "required": ["analysis_summary"], "properties": {"analysis_summary": {"type": "string"}}}
    env._mcp_schema_self_check_failures = 0
    env._mcp_last_schema_self_check = {}
    env._mcp_submit_attempted = False
    env._mcp_submit_accepted = False
    env._mcp_submit_rejected = False
    env._mcp_connection_manager = ExplodingManager()

    obs, reward, done, info = FusedEnv._handle_mcp_submit_result(
        env,
        SimpleNamespace(
            function_name="submit_result_difficulty_2",
            parameters={"result": {"analysis_summary": "ok"}},
        ),
    )

    assert done is True
    assert reward == 0.0
    assert info == {}


def test_mcp_reward_requires_accepted_schema_checked_submission():
    env = object.__new__(FusedEnv)
    env.entry = {
        "verifier": {"verification_code": "def verify(tools, answer):\n    return {'passed': True}\n"},
    }
    env.total_steps = 1
    env._mcp_answer = ""
    env._mcp_distinct_tools = set()
    env._mcp_tool_evidence = []
    env._mcp_submit_attempted = True
    env._mcp_submit_accepted = False
    env._mcp_submit_rejected = True
    env._mcp_last_submit_rejected_by_schema = True
    env._mcp_schema_self_check_failures = 1
    env._mcp_last_schema_self_check = {"passed": False, "errors": ["$.answer: missing required key"]}

    reward = FusedEnv._compute_mcp_reward(env)

    assert reward == 0.0
    assert env._mcp_reward_debug["base_reward"] == 0.0
    assert env._mcp_reward_debug["is_correct"] is False
    assert env._mcp_reward_debug["submit_accepted"] is False
    assert env._mcp_reward_debug["submit_rejected_by_schema"] is True
    assert env._mcp_reward_debug["schema_self_check"]["passed"] is False


def test_mcp_reward_reports_last_schema_rejection_separately_from_history():
    env = object.__new__(FusedEnv)
    env.entry = {
        "verifier": {"verification_code": "def verify(tools, answer):\n    return {'passed': True}\n"},
    }
    env.total_steps = 2
    env._mcp_answer = '{"answer": "validated metric"}'
    env._mcp_distinct_tools = {"lookup"}
    env._mcp_tool_evidence = [{"tool": "lookup", "arguments": {}, "output": "The validated metric appears in source."}]
    env._mcp_submit_attempted = True
    env._mcp_submit_accepted = True
    env._mcp_submit_rejected = True
    env._mcp_last_submit_rejected_by_schema = False
    env._mcp_schema_self_check_failures = 1
    env._mcp_last_schema_self_check = {"passed": True, "errors": []}

    reward = FusedEnv._compute_mcp_reward(env)

    assert reward > 0.0
    assert env._mcp_reward_debug["is_correct"] is True
    assert env._mcp_reward_debug["submit_rejected_by_schema"] is False
    assert env._mcp_reward_debug["submit_ever_rejected_by_schema"] is True
    assert env._mcp_reward_debug["evidence_gate_failed"] is False


def test_mcp_reward_demotes_verifier_pass_when_answer_has_no_tool_evidence_alignment():
    env = object.__new__(FusedEnv)
    env.entry = {
        "verifier": {"verification_code": "def verify(tools, answer):\n    return {'passed': True}\n"},
    }
    env.total_steps = 2
    env._mcp_answer = '{"metric_name": "voter turnout increase", "numerical_value": "6%"}'
    env._mcp_distinct_tools = {"lookup"}
    env._mcp_tool_evidence = [{"tool": "lookup", "arguments": {}, "output": ""}]
    env._mcp_submit_attempted = True
    env._mcp_submit_accepted = True
    env._mcp_submit_rejected = False
    env._mcp_last_submit_rejected_by_schema = False
    env._mcp_schema_self_check_failures = 0
    env._mcp_last_schema_self_check = {"passed": True, "errors": []}

    reward = FusedEnv._compute_mcp_reward(env)

    assert reward == 0.0
    assert env._mcp_reward_debug["is_correct"] is False
    assert env._mcp_reward_debug["evidence_gate_failed"] is True
    assert env._mcp_reward_debug["evidence_gate_reason"] == "no_evidence_field_alignment"


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


def test_resolve_mcp_tools_py_fallback_uses_unique_generated_suffix(tmp_path):
    assets = tmp_path / "assets"
    fallback = assets / "task-conduct-fitness-evaluations-task-2_run4"
    fallback.mkdir(parents=True)
    (fallback / "tools.py").write_text("def tool():\n    return 1\n", encoding="utf-8")

    missing = assets / "task-conduct-fitness-evaluations-task-2" / "tools.py"

    assert FusedEnv._resolve_mcp_tools_py_fallback(str(missing)) == str(fallback / "tools.py")
