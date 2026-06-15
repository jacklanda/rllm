import inspect
import hashlib
import json
import logging
import os
import re
import sys
import threading
import uuid

from rllm.environments.cli.cli import CLIEnv

logger = logging.getLogger(__name__)

try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

try:
    from examples.search.local_retrieval_tool import LocalRetrievalTool
except ImportError:
    try:
        from rllm.tools.web_tools.tavily_tool import TavilySearchTool as LocalRetrievalTool
    except ImportError:
        LocalRetrievalTool = None

try:
    from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
except ImportError:
    MCPConnectionManager = None
    MCPEnvironment = None

try:
    from rllm.environments.endless_terminals.et_env import ETEnv
except ImportError:
    ETEnv = None


_ANSWER_SCHEMA_MARKERS = (
    "answer submission format requirement",
    "answer format requirement",
    "submission format requirement",
)

_MCP_TOOL_OUTPUT_CHAR_LIMIT = int(os.environ.get("RLLM_MCP_TOOL_OUTPUT_CHAR_LIMIT", "6000"))
_MCP_TOOL_OUTPUT_LINE_LIMIT = int(os.environ.get("RLLM_MCP_TOOL_OUTPUT_LINE_LIMIT", "160"))
_MCP_SCHEMA_SELF_CHECK_MAX_FAILURES = int(os.environ.get("RLLM_MCP_SCHEMA_SELF_CHECK_MAX_FAILURES", "2"))
_SEARCH_REWRITE_MAX_REJECTIONS = int(os.environ.get("RLLM_SEARCH_REWRITE_MAX_REJECTIONS", "2"))

_EVIDENCE_STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "their",
    "this",
    "with",
    "within",
}

_QUERY_STOPWORDS = _EVIDENCE_STOPWORDS | {
    "about",
    "after",
    "before",
    "between",
    "does",
    "find",
    "give",
    "info",
    "information",
    "list",
    "look",
    "lookup",
    "name",
    "news",
    "search",
    "show",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whose",
}


def _is_et_entry(entry: dict) -> bool:
    """Heuristic detector for Endless Terminals rows.

    Two signals (either is sufficient):
      1. ``data_source == "endless_terminals"`` (stamped by convert_to_parquet.py).
      2. Schema match: docker_image + final_state_test + no repo_name/commit_hash
         (ET rows are self-contained CLI tasks, not SWE-Bench-style repo issues).
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("data_source") == "endless_terminals":
        return True
    if entry.get("docker_image") and entry.get("final_state_test") and entry.get("instruction") and not entry.get("repo_name") and not entry.get("commit_hash"):
        return True
    return False


def _extract_balanced_json(text: str, start_idx: int = 0) -> object | None:
    """Return the first balanced JSON object/array in ``text`` after ``start_idx``."""
    if not isinstance(text, str) or not text:
        return None
    opener_idx = -1
    for i in range(max(0, start_idx), len(text)):
        if text[i] in "{[":
            opener_idx = i
            break
    if opener_idx < 0:
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for i in range(opener_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in pairs:
            stack.append(pairs[ch])
            continue
        if ch in "}]":
            if not stack or ch != stack[-1]:
                return None
            stack.pop()
            if not stack:
                candidate = text[opener_idx : i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _extract_answer_schema(question: str) -> dict | None:
    """Extract the task's answer JSON schema from the prompt when present."""
    if not isinstance(question, str) or not question:
        return None
    low = question.lower()
    marker_idx = -1
    for marker in _ANSWER_SCHEMA_MARKERS:
        marker_idx = low.find(marker)
        if marker_idx >= 0:
            break
    parsed = _extract_balanced_json(question, marker_idx if marker_idx >= 0 else 0)
    return parsed if isinstance(parsed, dict) and isinstance(parsed.get("type"), str) else None


def _json_type_name(value: object) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _schema_expected_types(schema: dict) -> set[str]:
    expected = schema.get("type")
    if isinstance(expected, str):
        return {expected}
    if isinstance(expected, list):
        return {x for x in expected if isinstance(x, str)}
    return set()


def _schema_type_matches(actual: str, expected: set[str]) -> bool:
    if not expected:
        return True
    if actual in expected:
        return True
    return actual == "integer" and "number" in expected


def _coerce_submission_for_schema(payload: object, schema: dict | None) -> object:
    """Coerce lossless JSON scalar types before local schema self-check."""
    if not isinstance(schema, dict):
        return payload

    expected = _schema_expected_types(schema)
    if "integer" in expected:
        if isinstance(payload, float) and payload.is_integer():
            return int(payload)
        if isinstance(payload, str) and re.fullmatch(r"[-+]?\d+(?:\.0+)?", payload.strip()):
            return int(float(payload.strip()))

    if isinstance(payload, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}
        return {
            key: _coerce_submission_for_schema(value, properties.get(key, {}))
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [_coerce_submission_for_schema(item, item_schema) for item in payload]

    return payload


def _validate_submission_schema(payload: object, schema: dict | None, *, max_errors: int = 8) -> dict:
    """Lightweight local submit self-check for top-level type/required/non-empty.

    This intentionally implements a small JSON-Schema subset. The task verifier
    remains authoritative; this pre-check catches the common submit-contract
    errors early enough for the policy to repair them within the rollout.
    """
    if not schema:
        return {"passed": True, "errors": [], "schema_found": False}

    payload = _coerce_submission_for_schema(payload, schema)
    errors: list[str] = []

    def add(message: str) -> None:
        if len(errors) < max_errors:
            errors.append(message)

    def validate_node(value: object, node_schema: dict, path: str) -> None:
        if len(errors) >= max_errors or not isinstance(node_schema, dict):
            return

        expected = _schema_expected_types(node_schema)
        actual = _json_type_name(value)
        if not _schema_type_matches(actual, expected):
            add(f"{path}: expected {sorted(expected)}, got {actual}")
            return

        if isinstance(value, dict):
            if path == "$" and not value:
                add("$: object submission must not be empty")
            required = node_schema.get("required", [])
            properties = node_schema.get("properties", {})
            if not isinstance(required, list):
                required = []
            if not isinstance(properties, dict):
                properties = {}
            for key in required:
                if not isinstance(key, str):
                    continue
                child_path = f"{path}.{key}" if path != "$" else f"$.{key}"
                if key not in value:
                    add(f"{child_path}: missing required key")
                    continue
                child = value[key]
                if child == [] or child == {}:
                    add(f"{child_path}: required value must not be empty")
                validate_node(child, properties.get(key, {}), child_path)
        elif isinstance(value, list):
            if not value:
                add(f"{path}: array submission must not be empty")
                return
            item_schema = node_schema.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(value[:20]):
                    validate_node(item, item_schema, f"{path}[{idx}]")

    validate_node(payload, schema, "$")
    return {"passed": not errors, "errors": errors, "schema_found": True, "normalized_payload": payload}


def _compact_mcp_tool_output(output: object, *, char_limit: int = _MCP_TOOL_OUTPUT_CHAR_LIMIT, line_limit: int = _MCP_TOOL_OUTPUT_LINE_LIMIT) -> str:
    """Keep MCP observations bounded while preserving head/tail evidence."""
    text = str(output)
    if char_limit <= 0:
        return text

    lines = text.splitlines()
    if line_limit > 0 and len(lines) > line_limit:
        head_count = max(1, int(line_limit * 0.75))
        tail_count = max(1, line_limit - head_count)
        omitted = len(lines) - head_count - tail_count
        text = "\n".join(
            lines[:head_count]
            + [f"... [truncated {omitted} lines from MCP tool output] ..."]
            + lines[-tail_count:]
        )

    if len(text) <= char_limit:
        return text

    head_len = max(1, int(char_limit * 0.75))
    tail_len = max(1, char_limit - head_len)
    omitted = len(text) - head_len - tail_len
    return (
        text[:head_len].rstrip()
        + f"\n... [truncated {omitted} characters from MCP tool output] ...\n"
        + text[-tail_len:].lstrip()
    )


def _submission_self_check_observation(check: dict) -> str:
    errors = check.get("errors") or []
    details = "\n".join(f"- {err}" for err in errors)
    return (
        "Schema self-check failed; your answer was not submitted.\n"
        "Fix the JSON value and submit again. Minimum checks: top-level type, "
        "required keys, and non-empty required arrays/objects.\n"
        f"{details}"
    ).strip()


def _tokenize_for_evidence(text: object) -> set[str]:
    raw = str(text or "").lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", raw))
    return {t for t in tokens if t not in _EVIDENCE_STOPWORDS}


def _normalize_search_query(query: object) -> str:
    text = str(query or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\"'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(query: object) -> set[str]:
    normalized = _normalize_search_query(query)
    terms = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", normalized))
    return {term for term in terms if term not in _QUERY_STOPWORDS}


def _query_similarity(left: object, right: object) -> float:
    left_terms = _query_terms(left)
    right_terms = _query_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def _extract_precise_query_clues(query: object) -> set[str]:
    """Extract clues specific enough to justify a duplicate-result rewrite."""
    text = str(query or "")
    clues: set[str] = set()

    for match in re.findall(r"\b\d{2,4}(?:[-/.]\d{1,2}){0,2}\b|\b\d+(?:\.\d+)?%?\b", text):
        clues.add(match.lower())

    for match in re.findall(r'"([^"]{4,80})"|\'([^\']{4,80})\'', text):
        quoted = next((part for part in match if part), "")
        normalized = _normalize_search_query(quoted)
        if normalized:
            clues.add(normalized)

    for match in re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[-\s]+[A-Z0-9][A-Za-z0-9]*)+\b", text):
        normalized = _normalize_search_query(match)
        if normalized and len(_query_terms(normalized)) >= 2:
            clues.add(normalized)

    for term in re.findall(r"\b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)+\b", text):
        normalized = _normalize_search_query(term)
        if normalized and normalized not in _QUERY_STOPWORDS:
            clues.add(normalized)

    for term in _query_terms(text):
        if len(term) >= 8:
            clues.add(term)

    return clues


def _search_rewrite_guidance(reason: str) -> str:
    return (
        f"Search rejected: {reason}. The previous web_search returned duplicate or generic evidence. "
        "Rewrite the query with one unused precise clue from the question or evidence, such as a number, date, quoted title phrase, parameter combination, or proper noun. "
        "Do not search the same entity alone again."
    )


def _flatten_answer_fields(value: object, path: str = "$", *, limit: int = 80) -> list[dict]:
    fields: list[dict] = []

    def walk(node: object, node_path: str) -> None:
        if len(fields) >= limit:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                child_path = f"{node_path}.{key}" if node_path != "$" else f"$.{key}"
                walk(child, child_path)
        elif isinstance(node, list):
            if all(not isinstance(x, (dict, list)) for x in node):
                fields.append({"path": node_path, "value": node})
                return
            for idx, child in enumerate(node[:20]):
                walk(child, f"{node_path}[{idx}]")
        elif node not in (None, ""):
            fields.append({"path": node_path, "value": node})

    walk(value, path)
    return fields


def _extract_source_heading(output: str) -> str:
    patterns = (
        r'"heading"\s*:\s*"([^"]+)"',
        r'"title"\s*:\s*"([^"]+)"',
        r"^heading\s*:\s*(.+)$",
        r"^title\s*:\s*(.+)$",
    )
    for pattern in patterns:
        m = re.search(pattern, output or "", flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()[:160]
    return ""


def _evidence_snippet(output: str, value: object, tokens: set[str], *, max_chars: int = 320) -> str:
    output = output or ""
    value_text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value or "")
    value_text = value_text.strip()
    low_output = output.lower()
    if len(value_text) >= 8:
        idx = low_output.find(value_text.lower()[:120])
        if idx >= 0:
            start = max(0, idx - max_chars // 3)
            return output[start : start + max_chars].strip()
    best_line = ""
    best_overlap = 0
    for line in output.splitlines():
        overlap = len(tokens & _tokenize_for_evidence(line))
        if overlap > best_overlap:
            best_overlap = overlap
            best_line = line
    return best_line.strip()[:max_chars]


def _build_evidence_to_field_trace(answer: object, tool_evidence: list[dict], *, max_fields: int = 40) -> dict:
    """Map submitted answer fields to the strongest matching tool output."""
    fields = _flatten_answer_fields(answer, limit=max_fields)
    mappings: list[dict] = []
    missing = 0
    for field in fields:
        tokens = _tokenize_for_evidence(field["value"])
        if not tokens:
            continue
        best: tuple[float, dict | None] = (0.0, None)
        for evidence in tool_evidence:
            output = evidence.get("output", "")
            evidence_tokens = _tokenize_for_evidence(output)
            if not evidence_tokens:
                continue
            score = len(tokens & evidence_tokens) / max(1, len(tokens))
            if score > best[0]:
                best = (score, evidence)
        score, evidence = best
        if evidence is None or score <= 0.0:
            missing += 1
            mappings.append({"path": field["path"], "value_preview": str(field["value"])[:160], "matched": False, "score": 0.0})
            continue
        output = evidence.get("output", "")
        mappings.append(
            {
                "path": field["path"],
                "value_preview": str(field["value"])[:160],
                "matched": score >= 0.35,
                "score": round(float(score), 4),
                "source_tool": evidence.get("tool", ""),
                "source_heading": _extract_source_heading(output),
                "snippet": _evidence_snippet(output, field["value"], tokens),
            }
        )
        if score < 0.35:
            missing += 1
    return {
        "field_count": len(fields),
        "mapped_count": sum(1 for m in mappings if m.get("matched")),
        "missing_or_weak_count": missing,
        "mappings": mappings,
    }


class FusedEnv(CLIEnv):
    """Fused environment combining CLI/SWE Docker tools with external web search and MCP tools.

    Operates in four modes based on data type:

    **CLI mode** (entry has ``docker_image`` and SWE-Bench schema):
        Extends CLIEnv (which extends SWEEnv). Intercepts ``web_search`` tool
        calls and routes them to a ``LocalRetrievalTool`` running outside the Docker
        container, while all other tool calls are delegated to Docker via the parent.

    **ET mode** (entry has ``data_source="endless_terminals"`` or matches the
        ET schema — docker_image + final_state_test, no repo_name):
        Wraps a standalone ``ETEnv`` instance. ETEnv talks to the remote Docker
        daemon directly (no r2egym RepoEnv), runs the ET initial-state pytest,
        executes function-call XML actions, and returns a binary reward from
        ``/logs/verifier/reward.txt`` after ``tests/test.sh``.

    **Web search mode** (entry has ``data_source`` but no ``docker_image`` or ``tools_py``):
        No Docker container is created. Only ``web_search`` and ``finish``/``submit``
        tools are available. Reward is computed via F1-score against ``ground_truth``.

    **MCP mode** (entry has ``tools_py``):
        No Docker container is created. Connects to an MCP server that serves the
        tools defined in ``tools_py``. Reward is computed via verifier code.
    """

    # Shared retrieval tool singleton — avoids creating one httpx.Client per env instance
    _shared_retrieval_tool = None
    _retrieval_lock = threading.Lock()

    # Refcounted pool of MCP connection managers keyed by tools_py path.
    # A GRPO batch runs ``rollout_n`` (here 8) rollouts of the SAME task
    # concurrently, all pointing at the same ``tools.py`` — and the MCP tools
    # are read-only file queries with no server-side mutable state. Without
    # pooling, each rollout spawned its own MCP subprocess (1024 servers for a
    # 128x8 batch), and the serialized ~1s startup left every GPU idle for
    # >10 min before the first token. Pooling collapses that to one server per
    # distinct tools_py (<=128), refcounted so the server is stopped only when
    # the last rollout sharing it closes.
    _mcp_pool: dict[str, "MCPConnectionManager"] = {}
    _mcp_pool_refcount: dict[str, int] = {}
    _mcp_pool_lock = threading.Lock()
    _mcp_pool_failure_logged: set[str] = set()
    # Per-key "a start is in flight" event. The first rollout for a tools_py
    # registers an Event under the lock and starts the server; concurrent
    # same-task rollouts (e.g. the other rollout_n-1 of step 1's burst) find
    # the event and wait on it instead of each spawning their own server only
    # to lose the race and stop it. This collapses step-1 server starts from
    # train_batch_size*rollout_n (1024) down to distinct-tools_py (<=128).
    _mcp_pool_starting: dict[str, "threading.Event"] = {}
    # Max seconds a same-task rollout waits on an in-flight server start before
    # giving up and starting its own.  <=0 means wait until the designated
    # starter succeeds/fails.  That is the fused-training default: with an
    # active-server cap, the starter may legitimately queue for a slot, and
    # timing out here would multiply duplicate starts for the same tools.py.
    _MCP_START_WAIT_TIMEOUT = float(os.environ.get("RLLM_MCP_START_WAIT_TIMEOUT", "0"))

    def __init__(
        self,
        retrieval_server_url: str | None = None,
        retrieval_max_results: int = 3,
        harness: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retrieval_server_url = retrieval_server_url or os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:65432")
        self.retrieval_max_results = retrieval_max_results
        self.harness = str(harness or self.entry.get("harness") or "").strip().lower().replace("-", "_")
        self._record_parser_unknown_metrics = self.harness not in {"cot", "bare"}

        # Detect task mode: ET (checked before CLI since ET rows also carry a
        # docker_image), CLI, MCP, or Web Search.
        self._task_mode = self._resolve_task_mode(self.entry)

        # Web search mode state
        self._search_answer = ""  # Agent's submitted answer for reward computation
        self._search_answer_is_verbatim_submission = True
        self._search_last_raw_action = ""
        self._search_reward_debug = {}

        # MCP mode state
        self._mcp_connection_manager: "MCPConnectionManager | None" = None
        self._mcp_pool_key: str | None = None  # tools_py key into FusedEnv._mcp_pool, if pooled
        self._mcp_tool_schemas: list[dict] = []
        self._mcp_answer = ""
        self._mcp_reward_debug: dict = {}
        self._mcp_has_used_tools = False
        self._mcp_consecutive_unknown = 0
        self._mcp_unknown_total = 0
        self._mcp_distinct_tools: set[str] = set()
        self._mcp_answer_schema: dict | None = None
        self._mcp_schema_self_check_failures = 0
        self._mcp_last_schema_self_check: dict = {}
        self._mcp_tool_evidence: list[dict] = []

        # ET mode state — built lazily in _reset_et so failures during ETEnv
        # construction surface in reset() (which the engine wraps in retry
        # logic) rather than the constructor (which it does not).
        self._et_inner: "ETEnv | None" = None
        self._et_reward_debug: dict = {}
        self._et_consecutive_unknown = 0
        self._et_unknown_total = 0

    def _parser_unknown_metadata(self, consecutive_unknown: int | None = None, unknown_total: int | None = None) -> dict:
        if not self._record_parser_unknown_metrics:
            return {}
        meta = {}
        if consecutive_unknown is not None:
            meta["parser/consecutive_unknown"] = consecutive_unknown
        if unknown_total is not None:
            meta["parser/unknown_total"] = unknown_total
        return meta

    def _get_retrieval_tool(self):
        """Return the class-level shared retrieval tool (lazy-initialized, thread-safe)."""
        if FusedEnv._shared_retrieval_tool is None:
            with FusedEnv._retrieval_lock:
                if FusedEnv._shared_retrieval_tool is None:
                    if LocalRetrievalTool is None:
                        logger.warning("LocalRetrievalTool not available — web_search will return errors")
                        return None
                    FusedEnv._shared_retrieval_tool = LocalRetrievalTool(
                        server_url=self.retrieval_server_url,
                        max_results=self.retrieval_max_results,
                    )
        return FusedEnv._shared_retrieval_tool

    @property
    def supports_parallel_step(self) -> bool:
        return self._task_mode in ("web search", "mcp")

    def _resolve_task_mode(self, entry: dict | None) -> str:
        entry = entry or {}
        if _is_et_entry(entry):
            return "et"
        if entry.get("docker_image"):
            return "cli"
        if entry.get("tools_py"):
            return "mcp"
        return "web search"

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task: dict | str | None = None) -> tuple[str, dict]:
        next_task = self._normalize_entry(task)
        if next_task is not None and next_task != self.entry:
            self.close()
            self.env = None
        self._bind_task(next_task)
        self._task_mode = self._resolve_task_mode(self.entry)

        if self._task_mode == "et":
            return self._reset_et()
        if self._task_mode == "mcp":
            return self._reset_mcp()
        if self._task_mode == "web search":
            return self._reset_search()
        return self._reset_swe()

    def _reset_et(self) -> tuple[str, dict]:
        """Reset for ET-mode tasks (delegates to ETEnv)."""
        if ETEnv is None:
            raise RuntimeError("ETEnv import failed; cannot run endless_terminals tasks. " "Check that rllm.environments.endless_terminals.et_env is importable.")
        if self._et_inner is None:
            self._et_inner = ETEnv.from_dict(self.entry)
        obs, info = self._et_inner.reset()
        info["task_type"] = "et"
        return obs, info

    def _reset_search(self) -> tuple[str, dict]:
        """Reset for web-search-mode tasks (no Docker)."""
        self.total_steps = 0
        self._search_answer = ""
        self._search_answer_is_verbatim_submission = True
        self._search_last_raw_action = ""
        self._search_reward_debug = {}
        # Fix #2/#5: per-rollout parser-health + bypass counters.
        self._search_web_search_calls = 0
        self._search_consecutive_unknown = 0
        self._search_unknown_total = 0
        # P0-2: track low-content (junk) retrieval responses so the
        # reward can punish "finish after 2 searches that returned
        # nothing useful" — the dominant failure mode for
        # simpleqa/hotpotqa/medqa at step 0.
        self._search_low_content_responses = 0
        self._search_retrieval_seen_docs = set()
        self._search_retrieval_duplicate_hits = 0
        self._search_query_history: list[str] = []
        self._search_used_precise_clues: set[str] = set()
        self._search_rewrite_required = False
        self._search_rewrite_reason = ""
        self._search_rewrite_rejections = 0

        question = self.entry.get("question") or self.entry.get("query") or self.entry.get("input") or self.entry.get("problem_statement", "")
        # Strip stale answer-format instructions that conflict with FUSED_SEARCH_USER_PROMPT
        question = re.sub(r"\s*When ready, output the final answer enclosed in <answer> and </answer> tags\. Do not generate any content after the </answer> tag\.?", "", question).strip()
        return question, {"task_type": "web search"}

    def _reset_swe(self) -> tuple[str, dict]:
        """Reset for CLI-mode tasks (Docker container)."""
        obs, info = super().reset()
        info["task_type"] = "cli"
        return obs, info

    @classmethod
    def _acquire_mcp_manager(cls, pool_key: str, mcp_server_command: str, mcp_server_args: list[str]) -> "MCPConnectionManager":
        """Return a shared, started MCPConnectionManager for ``pool_key``.

        Reuses an existing manager (incrementing its refcount) when one is
        already running for this tools_py; otherwise starts a new one. The
        manager is started OUTSIDE the pool lock so concurrent first-time
        starts for *different* tasks proceed in parallel (the ~1s handshake is
        no longer serialized by ``_spawn_lock`` either). Same-task racers wait
        on a per-key start Event rather than each spawning their own server.
        """
        # Decide our role under the lock: reuse a live manager, wait for an
        # in-flight start, or become the starter ourselves.
        while True:
            with cls._mcp_pool_lock:
                manager = cls._mcp_pool.get(pool_key)
                if manager is not None and getattr(manager, "running", False):
                    cls._mcp_pool_refcount[pool_key] = cls._mcp_pool_refcount.get(pool_key, 0) + 1
                    return manager
                starting = cls._mcp_pool_starting.get(pool_key)
                if starting is None:
                    # We are the first: register intent and start below.
                    starting = threading.Event()
                    cls._mcp_pool_starting[pool_key] = starting
                    is_starter = True
                else:
                    is_starter = False

            if is_starter:
                break
            # Another rollout is starting this server; wait for it, then retry
            # the fast path (it will either be pooled-and-live or failed).  By
            # default we wait indefinitely because the designated starter may be
            # queued behind RLLM_MCP_MAX_ACTIVE_SERVERS; spawning duplicate
            # same-task servers under load recreates the step-1 burst failures.
            timeout = cls._MCP_START_WAIT_TIMEOUT if cls._MCP_START_WAIT_TIMEOUT > 0 else None
            completed = starting.wait(timeout=timeout)
            with cls._mcp_pool_lock:
                manager = cls._mcp_pool.get(pool_key)
                if manager is not None and getattr(manager, "running", False):
                    cls._mcp_pool_refcount[pool_key] = cls._mcp_pool_refcount.get(pool_key, 0) + 1
                    return manager
                # Start failed and the starter cleared its event: loop again —
                # we may now become the starter.
                if cls._mcp_pool_starting.get(pool_key) is None:
                    continue
                # Optional legacy escape hatch for a truly wedged starter.  The
                # training script leaves this disabled; users can set a positive
                # RLLM_MCP_START_WAIT_TIMEOUT if they prefer duplicate fallback.
                if not completed and cls._MCP_START_WAIT_TIMEOUT > 0 and cls._mcp_pool_starting.get(pool_key) is starting:
                    cls._mcp_pool_starting[pool_key] = threading.Event()
                    starting = cls._mcp_pool_starting[pool_key]
                    is_starter = True
                    break

        # We are the designated starter: start a fresh manager without holding
        # the pool lock so different-task starts run concurrently.
        new_manager = None
        try:
            new_manager = MCPConnectionManager(mcp_server_command, mcp_server_args)
            new_manager.start()
        except Exception:
            # Starting failed: clear our intent and wake waiters so one of
            # them can retry, then propagate.
            with cls._mcp_pool_lock:
                if cls._mcp_pool_starting.get(pool_key) is starting:
                    cls._mcp_pool_starting.pop(pool_key, None)
            starting.set()
            if new_manager is not None:
                try:
                    new_manager.stop()
                except Exception:
                    pass
            raise

        with cls._mcp_pool_lock:
            existing = cls._mcp_pool.get(pool_key)
            if existing is not None and getattr(existing, "running", False):
                # Lost a race (independent starter after a wait timeout):
                # keep theirs, drop ours (stop outside the lock below).
                cls._mcp_pool_refcount[pool_key] = cls._mcp_pool_refcount.get(pool_key, 0) + 1
                loser = new_manager
                manager = existing
            else:
                cls._mcp_pool[pool_key] = new_manager
                cls._mcp_pool_refcount[pool_key] = 1
                loser = None
                manager = new_manager
            if cls._mcp_pool_starting.get(pool_key) is starting:
                cls._mcp_pool_starting.pop(pool_key, None)
        starting.set()

        if loser is not None:
            try:
                loser.stop()
            except Exception:
                pass
        return manager

    @classmethod
    def _release_mcp_manager(cls, pool_key: str) -> None:
        """Decrement the refcount for ``pool_key`` and stop the manager when it
        reaches zero (last rollout sharing this tools_py has closed)."""
        manager_to_stop = None
        with cls._mcp_pool_lock:
            count = cls._mcp_pool_refcount.get(pool_key, 0) - 1
            if count <= 0:
                cls._mcp_pool_refcount.pop(pool_key, None)
                manager_to_stop = cls._mcp_pool.pop(pool_key, None)
            else:
                cls._mcp_pool_refcount[pool_key] = count
        if manager_to_stop is not None:
            try:
                manager_to_stop.stop()
            except Exception:
                pass

    def _reset_mcp(self) -> tuple[str, dict]:
        """Reset for MCP-mode tasks (tool-based tasks via MCP server)."""
        self.total_steps = 0
        self._mcp_answer = ""
        self._mcp_reward_debug = {}
        self._mcp_has_used_tools = False
        self._mcp_consecutive_unknown = 0
        self._mcp_unknown_total = 0
        self._mcp_distinct_tools: set[str] = set()
        question = self.entry.get("question", self.entry.get("problem_statement", ""))
        self._mcp_answer_schema = _extract_answer_schema(question)
        self._mcp_schema_self_check_failures = 0
        self._mcp_last_schema_self_check = {}
        self._mcp_tool_evidence = []

        # Release any manager held from a prior reset on this instance so the
        # pool refcount stays balanced (reset() only auto-closes on task change).
        if self._mcp_connection_manager is not None:
            if self._mcp_pool_key is not None:
                self._release_mcp_manager(self._mcp_pool_key)
            else:
                try:
                    self._mcp_connection_manager.stop()
                except Exception:
                    pass
            self._mcp_connection_manager = None
            self._mcp_pool_key = None

        # Resolve tools_py path
        tools_py = self.entry.get("tools_py", "")
        data_root = self.entry.get("data_root", "")
        if data_root and tools_py and not os.path.isabs(tools_py):
            tools_py_abs = os.path.join(data_root, os.path.basename(tools_py))
        else:
            tools_py_abs = tools_py

        # If tools_py_abs doesn't exist, try the original path directly
        if not os.path.exists(tools_py_abs):
            tools_py_abs = tools_py

        # Start MCP server and discover tools
        if MCPConnectionManager is not None and MCPEnvironment is not None:
            try:
                from pathlib import Path

                tools_path = Path(tools_py_abs)
                if tools_path.exists() and tools_path.is_file():
                    server_script = MCPEnvironment._ensure_server_script(tools_path.parent)
                    mcp_server_command = sys.executable
                    mcp_server_args = [str(server_script)]
                    # Share one MCP server across all rollouts of this task
                    # (same tools_py) via a refcounted pool. The tools are
                    # read-only file queries, so a shared server is safe and
                    # avoids spawning rollout_n identical subprocesses.
                    pool_key = str(server_script)
                    self._mcp_connection_manager = self._acquire_mcp_manager(pool_key, mcp_server_command, mcp_server_args)
                    self._mcp_pool_key = pool_key
                    # Extract tool schemas from discovered tools (deduplicate by name)
                    seen = set()
                    self._mcp_tool_schemas = []
                    for tool in self._mcp_connection_manager.tool_map.values():
                        name = getattr(tool, "name", None)
                        if not name or name in seen:
                            continue
                        seen.add(name)
                        self._mcp_tool_schemas.append(self._slim_tool_schema(tool.json))
                else:
                    logger.error("tools_py not found: %s", tools_py_abs)
                    self._mcp_tool_schemas = []
            except Exception as e:
                log_error = True
                try:
                    pool_key = str(MCPEnvironment._ensure_server_script(Path(tools_py_abs).parent))
                    with FusedEnv._mcp_pool_lock:
                        if pool_key in FusedEnv._mcp_pool_failure_logged:
                            log_error = False
                        else:
                            FusedEnv._mcp_pool_failure_logged.add(pool_key)
                except Exception:
                    pass
                if log_error:
                    logger.error("Failed to start MCP server for %s: %s", tools_py_abs, e)
                self._mcp_tool_schemas = []
        else:
            logger.warning("MCP dependencies not available — MCP task will have no tools")
            self._mcp_tool_schemas = []

        return question, {
            "task_type": "mcp",
            "tools_json": self._mcp_tool_schemas,
            "difficulty": self.entry.get("difficulty", ""),
        }

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    _MAX_DESC_CHARS = 240
    _MAX_PARAM_DESC_CHARS = 120

    @classmethod
    def _slim_tool_schema(cls, schema: dict) -> dict:
        """Strip verbose sub-fields from a JSON-Schema tool definition.

        Fix #8: the system prompt reached ~13 k chars (close to the 13 k
        tool-call write limit) because each of the 1000+ per-task MCP
        tools carried redundant ``title`` fields on every property and
        multi-paragraph descriptions. We keep ``name``, ``description``,
        ``parameters`` / ``inputSchema`` with only ``type``,
        ``properties``, ``required``, ``items``; truncate descriptions
        to one sentence; drop ``title`` entirely (JSON Schema treats it
        as cosmetic).

        Handles both flat schemas and OpenAI-style wrapped schemas
        ``{"type": "function", "function": {...}}`` (what ``MCPTool.json``
        returns). The wrapped form is preserved and its inner body is
        slimmed; a bug in the flat-only version caused every MCP tool to
        collapse to ``{}`` and disappear from the prompt.
        """
        if not isinstance(schema, dict):
            return schema
        if isinstance(schema.get("function"), dict):
            inner = cls._slim_tool_schema(schema["function"])
            wrapped: dict = {"type": schema.get("type", "function"), "function": inner}
            return wrapped
        out: dict = {}
        if "name" in schema:
            out["name"] = schema["name"]
        desc = schema.get("description")
        if isinstance(desc, str) and desc:
            out["description"] = desc.strip()[: cls._MAX_DESC_CHARS]
        params_key = "parameters" if "parameters" in schema else ("inputSchema" if "inputSchema" in schema else None)
        if params_key:
            out[params_key] = cls._slim_json_schema(schema[params_key])
        return out

    @classmethod
    def _slim_json_schema(cls, node):
        if not isinstance(node, dict):
            return node
        out: dict = {}
        keep_keys = ("type", "properties", "required", "items", "enum", "oneOf", "anyOf")
        for k in keep_keys:
            if k in node:
                v = node[k]
                if k == "properties" and isinstance(v, dict):
                    out[k] = {p: cls._slim_property(pv) for p, pv in v.items()}
                elif k in ("oneOf", "anyOf") and isinstance(v, list):
                    out[k] = [cls._slim_json_schema(x) for x in v]
                elif k == "items":
                    out[k] = cls._slim_json_schema(v)
                else:
                    out[k] = v
        desc = node.get("description")
        if isinstance(desc, str) and desc:
            out["description"] = desc.strip()[: cls._MAX_PARAM_DESC_CHARS]
        return out

    @classmethod
    def _slim_property(cls, prop):
        if not isinstance(prop, dict):
            return prop
        out: dict = {}
        for k in ("type", "enum", "items", "properties", "required", "oneOf", "anyOf"):
            if k in prop:
                v = prop[k]
                if k == "items":
                    out[k] = cls._slim_json_schema(v)
                elif k == "properties" and isinstance(v, dict):
                    out[k] = {p: cls._slim_property(pv) for p, pv in v.items()}
                elif k in ("oneOf", "anyOf") and isinstance(v, list):
                    out[k] = [cls._slim_json_schema(x) for x in v]
                else:
                    out[k] = v
        desc = prop.get("description")
        if isinstance(desc, str) and desc:
            out["description"] = desc.strip()[: cls._MAX_PARAM_DESC_CHARS]
        return out

    # Fix #7: single-turn guards to prevent runaway generation. The eval
    # dump at step-10 contained one musique rollout whose assistant turn was
    # 488,222 characters of pure token repetition (e.g. ``"Jennifer"`` × 10k)
    # yet terminated normally with reward 0. With no truncation, no
    # repetition check, and no abnormal-termination flag, the signal was
    # invisible to training.
    #
    # The char threshold is deliberately set ABOVE the per-step token soft cap
    # (rllm.agent.per_step_max_tokens) expressed in chars, so a normal long
    # turn that merely hits the soft cap is NOT misflagged as runaway purely
    # for length — only genuine repetition (the 100-consecutive-token signal,
    # which fires regardless of length) terminates such a turn. At ~3.5
    # chars/token a 16k-token cap is ~56k chars, so 64k keeps a safety margin.
    _MAX_TURN_CHARS = 64_000
    _MAX_CONSECUTIVE_TOKEN_REPEATS = 100
    _MAX_REPEATED_NGRAM_TOKENS = 40
    _RUNAWAY_SYMBOL_RE = re.compile(r"([{}<>\[\]()/])\1{80,}")
    _RUNAWAY_PHRASES = (
        "the tool call is a function",
        "the tool response is a",
        "valid json schema",
        "the user message is",
        "the user is a person",
    )

    @classmethod
    def _detect_runaway(cls, raw: str) -> tuple[bool, str]:
        """Return (is_runaway, reason) for pathological assistant output."""
        if not raw:
            return False, ""
        raw = str(raw)
        if len(raw) > cls._MAX_TURN_CHARS:
            return True, f"assistant turn exceeded {cls._MAX_TURN_CHARS} chars (got {len(raw)})"
        lowered = raw.lower()
        for phrase in cls._RUNAWAY_PHRASES:
            count = lowered.count(phrase)
            if count >= 8:
                return True, f"phrase {phrase!r} repeated {count} times"
        symbol_match = cls._RUNAWAY_SYMBOL_RE.search(raw)
        if symbol_match:
            return True, f"symbol {symbol_match.group(1)!r} repeated excessively"
        tokens = raw.split()
        if len(tokens) >= cls._MAX_CONSECUTIVE_TOKEN_REPEATS:
            run, prev = 1, None
            for tok in tokens:
                if tok == prev:
                    run += 1
                    if run >= cls._MAX_CONSECUTIVE_TOKEN_REPEATS:
                        return True, f"same token {tok!r} repeated {run}× consecutively"
                else:
                    run, prev = 1, tok
        lowered_tokens = [tok.lower() for tok in tokens]
        for n in (3, 4, 5, 8, 12, 16):
            min_runs = 10 if n <= 4 else 6
            if len(lowered_tokens) < n * min_runs:
                continue
            for offset in range(n):
                run = 1
                prev_ngram = None
                for i in range(offset, len(lowered_tokens) - n + 1, n):
                    ngram = tuple(lowered_tokens[i : i + n])
                    if ngram == prev_ngram:
                        run += 1
                        if run >= min_runs and run * n >= cls._MAX_REPEATED_NGRAM_TOKENS:
                            preview = " ".join(ngram[:8])
                            return True, f"{n}-gram loop repeated {run} times: {preview!r}"
                    else:
                        run = 1
                        prev_ngram = ngram
        return False, ""

    def step(self, action):
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first
        if not raw_text and action is not None:
            raw_text = getattr(action, "action", "") or getattr(action, "model_response", "")
        if raw_text and not isinstance(raw_text, str):
            raw_text = str(raw_text)
        bad, reason = self._detect_runaway(raw_text)
        # The runaway guard exists to abort pathological multi-turn generation
        # (e.g. a 488k-char ``"Jennifer" x thousands`` loop) before it wastes
        # decode budget. For prompt-only harnesses (cot/bare) the *entire* turn
        # is the answer and is already fully generated by the time we see it:
        # thinking-mode COT legitimately runs 64k-220k chars, so the length
        # check (``_MAX_TURN_CHARS``) is a false positive that discards a valid
        # final answer with reward 0 before ``_step_search`` can extract it.
        # This is the dominant rollout-time false negative on GPQA/cot — every
        # >64k-char response was being zeroed even though ``\boxed{...}`` /
        # ``<answer>X</answer>`` sat at the end. Skip the hard runaway
        # termination for cot/bare and let normal extraction score the turn;
        # genuinely degenerate output simply yields no extractable answer (and
        # still incurs the repetition/length penalties), so it scores ~0 anyway
        # — matching what the post-hoc ``merge_eval_json`` rescore recovers.
        if bad and self.harness not in {"cot", "bare"}:
            self.total_steps += 1
            info = {
                "termination_reason": "TRUNCATION",
                "termination_message": f"Runaway generation: {reason}",
                "guard/runaway_chars": len(raw_text),
            }
            return (
                f"Error: runaway generation detected ({reason}); terminating rollout.",
                0.0,
                True,
                info,
            )
        if self._task_mode == "mcp":
            return self._step_mcp(action)
        if self._task_mode == "web search":
            return self._step_search(action)
        if self._task_mode == "et":
            return self._step_et(action)
        return self._step_swe(action)

    def _step_swe(self, action):
        """CLI-mode step: web_search goes to retrieval tool, everything else to Docker."""
        if SWEAction is None:
            return super().step(action)

        # Unwrap list[Action] → list[SWEAction] and process each
        action_objs = self._unwrap_actions(action)

        # Check if the first action is web_search; rest go to Docker
        if action_objs and action_objs[0].function_name == "web_search":
            return self._handle_web_search(action_objs[0])

        # For SWE tools, pass the first action as its XML string to the parent
        if action_objs:
            return super().step(action_objs[0].to_xml_string())
        return super().step(action)

    def _step_et(self, action):
        """ET-mode step: route into the wrapped ETEnv, with structural-error
        bookkeeping that mirrors the CLI/MCP modes so the engine sees the
        same termination_reason taxonomy across data sources.
        """
        if self._et_inner is None:
            return (
                "Error: ET env not initialized; call reset() first.",
                0.0,
                True,
                {
                    "termination_reason": "ENV_INIT_ERROR",
                },
            )

        # Keep raw text so we can rescue \boxed{...} as an implicit submit and
        # report structural parse failures explicitly.
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first

        action_objs = self._unwrap_actions(action) if SWEAction is not None else []

        if not action_objs:
            if raw_text:
                try:
                    from rllm.parser.tool_parser import QwenToolParser as _QTP

                    tcs = _QTP().parse_qwen_tool_calls(raw_text)
                    if tcs and tcs[0].get("name") in ("finish", "submit"):
                        action_objs = [SWEAction(function_name="submit", parameters={})]
                except Exception:
                    pass
            if not action_objs:
                boxed = self._extract_boxed_from_raw(raw_text) if raw_text else None
                if boxed is not None:
                    action_objs = [SWEAction(function_name="submit", parameters={})]
                else:
                    self._et_consecutive_unknown += 1
                    self._et_unknown_total += 1
                    if self._et_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                        info = {
                            "termination_reason": "ABNORMAL_PARSE_ERROR",
                            **self._parser_unknown_metadata(self._et_consecutive_unknown, self._et_unknown_total),
                        }
                        return (
                            "Error: could not parse any actions; terminating rollout.",
                            0.0,
                            True,
                            info,
                        )
                    return "Error: could not parse any actions from model output.", 0.0, False, {}

        # ETEnv accepts one action at a time. Run them sequentially; the
        # first finish/submit terminates the rollout.
        observations: list[str] = []
        last_info: dict = {}
        for action_obj in action_objs:
            self._et_consecutive_unknown = 0
            obs, _r, done, info = self._et_inner.step(action_obj)
            observations.append(str(obs))
            last_info = info
            if done:
                combined = "\n".join(observations) if observations else str(obs)
                return combined, 0.0, True, info

        combined = "\n".join(observations) if observations else "No tool calls executed."
        return combined, 0.0, False, last_info

    _MAX_CONSECUTIVE_UNKNOWN = 3

    @staticmethod
    def _answer_marker_search_text(raw: str) -> str:
        if not raw:
            return ""
        think_end = raw.rfind("</think>")
        if think_end == -1:
            return raw
        post_think = raw[think_end + len("</think>") :].strip()
        if not post_think:
            return raw
        if re.search(r"<answer\b|(?:\\boxed|boxed|oxed|\x08oxed)\{", post_think, flags=re.IGNORECASE):
            return post_think
        return raw

    @classmethod
    def _extract_boxed_span_from_raw(cls, raw: str) -> tuple[int, str] | None:
        raw = cls._answer_marker_search_text(raw)
        matches: list[tuple[int, int]] = []
        for tok in ("\\boxed{", "boxed{", "oxed{", "\x08oxed{"):
            start = raw.find(tok)
            while start != -1:
                matches.append((start, len(tok)))
                start = raw.find(tok, start + 1)
        for start, tok_len in sorted(matches, reverse=True):
            i = start + tok_len
            depth = 1
            j = i
            while depth and j < len(raw):
                if raw[j] == "{":
                    depth += 1
                elif raw[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                return start, raw[i : j - 1]
        return None

    @classmethod
    def _extract_boxed_from_raw(cls, raw: str) -> str | None:
        """Rescue the final ``\\boxed{…}`` payload from a raw assistant turn.

        Prefer text after the last ``</think>`` when present, and take the last
        balanced boxed span. If nothing follows ``</think>``, search the full
        turn so Qwen-style outputs that keep the final answer inside the think
        block still score correctly.
        """
        if not raw:
            return None
        span = cls._extract_boxed_span_from_raw(raw)
        return span[1] if span is not None else None

    @classmethod
    def _extract_answer_tag_span_from_raw(cls, raw: str) -> tuple[int, str] | None:
        if not raw:
            return None
        raw = cls._answer_marker_search_text(raw)
        matches = list(re.finditer(r"<answer>\s*((?:(?!<answer>).)*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            return None
        match = matches[-1]
        return match.start(), match.group(1).strip()

    @classmethod
    def _extract_answer_tag_from_raw(cls, raw: str) -> str | None:
        """Rescue the final <answer>...</answer> payload from prompt-only turns."""
        span = cls._extract_answer_tag_span_from_raw(raw)
        return span[1] if span is not None else None

    @classmethod
    def _extract_final_answer_marker_from_raw(cls, raw: str) -> str | None:
        box_span = cls._extract_boxed_span_from_raw(raw)
        ans_span = cls._extract_answer_tag_span_from_raw(raw)
        candidates = [span for span in (box_span, ans_span) if span is not None and not cls._is_placeholder_answer_marker(span[1])]
        if not candidates:
            return None
        # A bare-letter ``<answer>LETTER</answer>`` is the MCQ final-answer line
        # mandated by the cot/bare prompt and is authoritative over a trailing
        # ``\boxed{value}`` artifact (e.g. ``<answer>D</answer>\n\boxed{33.4}``).
        if ans_span is not None and re.fullmatch(r"\(?\s*[A-Fa-f]\s*\)?", ans_span[1].strip()):
            return ans_span[1].strip().strip("()").strip().upper()
        return max(candidates, key=lambda item: item[0])[1].strip()

    @staticmethod
    def _is_placeholder_answer_marker(value: str) -> bool:
        normalized = re.sub(r"[\W_]+", "", str(value or "")).lower()
        return normalized in {"", "answer", "finalanswer", "boxedfinalanswer", "letter", "option", "choice"}

    @staticmethod
    def _has_tool_call_markup(raw: str) -> bool:
        return bool(re.search(r"<\s*/?\s*(?:tool_call|function_call)\b|<function=", raw or ""))

    def _step_search(self, action):
        """Web-search-mode step: handle web_search + finish/submit locally, error on Docker tools."""
        if SWEAction is None:
            # Cannot parse actions without r2egym
            self.total_steps += 1
            return "Error: r2egym not available for action parsing.", 0.0, False, {}

        # Keep the raw model output so we can rescue \boxed{...} when the
        # tool-call parser returns nothing useful.
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first
        if not raw_text and action is not None:
            raw_text = getattr(action, "action", "") or getattr(action, "model_response", "")
        if raw_text and raw_text.strip():
            self._search_last_raw_action = raw_text.strip()

        if self.harness in {"cot", "bare"} and raw_text and raw_text.strip() and not self._has_tool_call_markup(raw_text):
            final_marker = self._extract_final_answer_marker_from_raw(raw_text)
            if final_marker is not None:
                self._search_answer = final_marker.strip()
                self._search_answer_is_verbatim_submission = True
            else:
                self._search_answer = raw_text.strip()
                self._search_answer_is_verbatim_submission = False
            self.total_steps += 1
            self._search_consecutive_unknown = 0
            return "Your answer has been submitted.", 0.0, True, {}

        if self.harness not in {"cot", "bare"} and getattr(self, "_search_web_search_calls", 0) == 0 and raw_text:
            final_marker = self._extract_final_answer_marker_from_raw(raw_text)
            if final_marker is not None:
                self._search_answer = final_marker.strip()
                self._search_answer_is_verbatim_submission = True
                self.total_steps += 1
                self._search_consecutive_unknown = 0
                return self._search_bypass_termination("explicit answer marker before web_search")

        action_objs = self._unwrap_actions(action)
        if raw_text and action_objs and all(not getattr(obj, "function_name", "") for obj in action_objs):
            action_objs = []
        if not action_objs:
            # Last-resort: try QwenToolParser directly on the raw string
            if raw_text:
                from rllm.parser.tool_parser import QwenToolParser as _QTP

                tcs = _QTP().parse_qwen_tool_calls(raw_text)
                if tcs and tcs[0].get("name") in ("finish", "submit"):
                    result = tcs[0].get("arguments", {}).get("result", "")
                    action_objs = [SWEAction(function_name="finish", parameters={"result": result})]
            if not action_objs:
                # Parser exhausted: try \boxed{...} as implicit finish.
                final_marker = self._extract_final_answer_marker_from_raw(raw_text)
                if final_marker is not None:
                    action_objs = [SWEAction(function_name="finish", parameters={"result": final_marker})]
                elif self.harness in {"cot", "bare"} and raw_text and raw_text.strip():
                    self._search_answer = raw_text.strip()
                    self._search_answer_is_verbatim_submission = False
                    self.total_steps += 1
                    self._search_consecutive_unknown = 0
                    return "Your answer has been submitted.", 0.0, True, {}
                else:
                    self.total_steps += 1
                    self._search_consecutive_unknown += 1
                    self._search_unknown_total += 1
                    if self._search_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                        info = {
                            "termination_reason": "ABNORMAL_PARSE_ERROR",
                            **self._parser_unknown_metadata(self._search_consecutive_unknown, self._search_unknown_total),
                        }
                        return (
                            "Error: could not parse any actions; terminating rollout.",
                            0.0,
                            True,
                            info,
                        )
                    return "Error: could not parse any actions from model output.", 0.0, False, {}

        observations: list[str] = []
        for action_obj in action_objs:
            fn = action_obj.function_name

            if fn == "web_search":
                self._search_consecutive_unknown = 0
                self._search_web_search_calls += 1
                obs, reward, done, info = self._handle_web_search(action_obj)
                if done:
                    return obs, reward, done, info
                observations.append(obs)
                continue

            if fn in ("finish", "submit"):
                self._search_consecutive_unknown = 0
                return self._handle_search_finish(action_obj)

            # Docker-only tools are not available in web search mode
            self.total_steps += 1
            self._search_consecutive_unknown += 1
            self._search_unknown_total += 1
            if self._search_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                info = {
                    "termination_reason": "ABNORMAL_PARSE_ERROR",
                    **self._parser_unknown_metadata(self._search_consecutive_unknown, self._search_unknown_total),
                }
                return (
                    f"Error: tool '{fn}' is not available; terminating after {self._search_consecutive_unknown} consecutive parse failures.",
                    0.0,
                    True,
                    info,
                )
            observations.append(f"Error: The tool '{fn}' is not available for web search tasks. " "Use web_search to find information and finish to submit your answer.")

        combined = "\n".join(observations) if observations else "No tool calls executed."
        return combined, 0.0, False, {}

    # ------------------------------------------------------------------
    # Action unwrap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_actions(action) -> "list[SWEAction]":
        """Convert any action format from the workflow into a list of SWEAction objects.

        Handles:
        - ``str`` (XML-encoded SWEAction)
        - ``SWEAction`` instance
        - ``Action`` dataclass (``action.action`` is the XML string)
        - ``list[Action | str]`` from CLIAgent.update_from_model()
        """
        from rllm.agents.agent import Action as AgentAction

        raw_items: list = []
        if isinstance(action, list):
            raw_items = action
        else:
            raw_items = [action]

        swe_actions: list[SWEAction] = []
        for item in raw_items:
            if isinstance(item, AgentAction):
                item = item.action  # unwrap the dataclass
            if isinstance(item, str):
                try:
                    swe_actions.append(SWEAction.from_string(item))
                except Exception:
                    logger.warning("Failed to parse action string: %s", item[:120])
            elif SWEAction is not None and isinstance(item, SWEAction):
                swe_actions.append(item)
            else:
                logger.warning("Unknown action type in _unwrap_actions: %s", type(item))
        return swe_actions

    # ------------------------------------------------------------------
    # MCP step
    # ------------------------------------------------------------------

    def _step_mcp(self, action):
        """MCP-mode step: route tool calls to MCP server, handle finish locally.

        Supports receiving a ``list[Action]`` from CLIAgent (multiple parsed
        tool calls per model turn).  Non-finish tool calls are executed
        sequentially and their results concatenated; a finish call terminates.
        """
        if SWEAction is None:
            self.total_steps += 1
            return "Error: r2egym not available for action parsing.", 0.0, False, {}

        # Keep raw text so we can rescue a `\boxed{…}` implicit finish and
        # report structural parse failures explicitly (fix #6).
        raw_text = action if isinstance(action, str) else ""
        if not raw_text and isinstance(action, list) and action:
            first = action[0]
            raw_text = getattr(first, "action", "") if not isinstance(first, str) else first

        action_objs = self._unwrap_actions(action)
        if not action_objs:
            # Last-resort: try QwenToolParser, then \boxed{...} as implicit finish.
            if raw_text:
                try:
                    from rllm.parser.tool_parser import QwenToolParser as _QTP

                    tcs = _QTP().parse_qwen_tool_calls(raw_text)
                    if tcs and tcs[0].get("name") in ("finish", "submit"):
                        result = tcs[0].get("arguments", {}).get("result", "")
                        action_objs = [SWEAction(function_name="finish", parameters={"result": result})]
                except Exception:
                    pass
            if not action_objs:
                boxed = self._extract_boxed_from_raw(raw_text)
                if boxed is not None:
                    action_objs = [SWEAction(function_name="finish", parameters={"result": boxed})]
                else:
                    self.total_steps += 1
                    self._mcp_consecutive_unknown = getattr(self, "_mcp_consecutive_unknown", 0) + 1
                    self._mcp_unknown_total = getattr(self, "_mcp_unknown_total", 0) + 1
                    if self._mcp_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                        info = {
                            "termination_reason": "ABNORMAL_PARSE_ERROR",
                            "termination_message": (f"MCP: {self._mcp_consecutive_unknown} consecutive turns " "without a parseable <tool_call>"),
                            **self._parser_unknown_metadata(self._mcp_consecutive_unknown, self._mcp_unknown_total),
                        }
                        return (
                            "Error: could not parse any actions; terminating rollout.",
                            0.0,
                            True,
                            info,
                        )
                    return (
                        "Error: could not parse any actions from model output. "
                        'Emit exactly one <tool_call>{"name": ..., "arguments": {...}}</tool_call> '
                        "block; use finish/submit to end the task.",
                        0.0,
                        False,
                        self._parser_unknown_metadata(unknown_total=self._mcp_unknown_total),
                    )

        observations: list[str] = []
        for action_obj in action_objs:
            fn = action_obj.function_name

            # Handle finish/submit — terminates immediately
            if fn in ("finish", "submit"):
                self._mcp_consecutive_unknown = 0
                return self._handle_mcp_finish(action_obj)

            # Handle submit_result_difficulty_xxx
            if fn.startswith("submit_result_difficulty_"):
                self._mcp_consecutive_unknown = 0
                return self._handle_mcp_submit_result(action_obj)

            # Regular MCP tool call
            # Empty function name means the parser found a <tool_call> block
            # but couldn't recover the ``name`` key — treat as structural
            # failure so the engine sees an explicit INVALID_REACT_STRUCTURE
            # bucket instead of silently consuming a step with a confusing
            # "Tool  not found" error.
            if not fn:
                self.total_steps += 1
                self._mcp_consecutive_unknown += 1
                self._mcp_unknown_total += 1
                if self._mcp_consecutive_unknown >= self._MAX_CONSECUTIVE_UNKNOWN:
                    info = {
                        "termination_reason": "INVALID_REACT_STRUCTURE",
                        "termination_message": (f"MCP: {self._mcp_consecutive_unknown} consecutive " "tool_calls with empty/unparseable `name` field"),
                        **self._parser_unknown_metadata(self._mcp_consecutive_unknown, self._mcp_unknown_total),
                    }
                    return (
                        "Error: could not parse a tool `name` from the <tool_call> " "block; terminating rollout.",
                        0.0,
                        True,
                        info,
                    )
                observations.append("Error: empty tool name. Each <tool_call> must be valid JSON with a " '"name" string (e.g. {"name": "finish", "arguments": {...}}).')
                continue

            self._mcp_consecutive_unknown = 0
            self._mcp_has_used_tools = True
            self.total_steps += 1

            params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
            # Restore original types lost during SWEAction string round-trip.
            # SWEAction stringifies all parameter values (int→"0", bool→"false",
            # list→'["a","b"]').  json.loads recovers the original types so the
            # MCP server receives correctly-typed arguments.
            restored_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    try:
                        restored_params[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        restored_params[k] = v
                else:
                    restored_params[k] = v
            tool_call_id = str(uuid.uuid4())
            tool_calls = [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": fn,
                        "arguments": json.dumps(restored_params, ensure_ascii=False),
                    },
                }
            ]

            if self._mcp_connection_manager is None:
                observations.append(f"Execution output of [{fn}]:\nError: MCP server not available.")
                continue

            try:
                tool_outputs = self._mcp_connection_manager.execute_tool_calls(tool_calls)
                output_str = tool_outputs.get(tool_call_id, "No output")
                compact_output = _compact_mcp_tool_output(output_str)
                observations.append(f"Execution output of [{fn}]:\n{compact_output}")
                # Fix #3: track distinct successful tool names so the
                # verifier can reward genuine exploration (gated on
                # is_correct) rather than raw call count. A call is counted
                # as "successful" only if the output doesn't start with the
                # MCP server's "Error:" prefix.
                if not str(output_str).lstrip().lower().startswith("error"):
                    self._mcp_tool_evidence.append(
                        {
                            "tool": fn,
                            "arguments": restored_params,
                            "output": str(output_str)[:8000],
                        }
                    )
                    self._mcp_distinct_tools.add(fn)
            except Exception as e:
                logger.error("MCP tool execution failed for %s: %s", fn, e)
                observations.append(f"Execution output of [{fn}]:\nError: {str(e)}")

        combined = "\n".join(observations) if observations else "No tool calls executed."
        return combined, 0.0, False, {}

    def _handle_web_search(self, action_obj) -> tuple[str, float, bool, dict]:
        """Execute a web_search tool call via LocalRetrievalTool."""
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        query = params.get("query", "")
        query_text = str(query or "").strip()
        top_k = params.get("top_k", None)
        if top_k is not None:
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                top_k = None

        history: list[str] = getattr(self, "_search_query_history", [])
        used_clues: set[str] = getattr(self, "_search_used_precise_clues", set())
        query_clues = _extract_precise_query_clues(query_text)
        last_query = history[-1] if history else ""
        if getattr(self, "_search_rewrite_required", False):
            new_clues = query_clues - used_clues
            too_similar = bool(last_query) and _query_similarity(query_text, last_query) >= 0.8
            if not query_text or not new_clues or too_similar:
                self._search_rewrite_rejections = getattr(self, "_search_rewrite_rejections", 0) + 1
                reason_bits = []
                if not query_text:
                    reason_bits.append("empty query")
                if not new_clues:
                    reason_bits.append("missing an unused precise clue")
                if too_similar:
                    reason_bits.append("too similar to the previous query")
                reason = ", ".join(reason_bits) or "query was not a valid rewrite"
                info = {
                    "search/query_rewrite_rejected": 1,
                    "search/query_rewrite_rejections": self._search_rewrite_rejections,
                }
                if self._search_rewrite_rejections >= _SEARCH_REWRITE_MAX_REJECTIONS:
                    info["termination_reason"] = "SEARCH_QUERY_REWRITE_EXCEEDED"
                    info["termination_message"] = f"Rejected {self._search_rewrite_rejections} duplicate/generic follow-up queries."
                    return _search_rewrite_guidance(reason) + "\nRepeated rewrite failures; terminating rollout.", 0.0, True, info
                return _search_rewrite_guidance(reason), 0.0, False, info

        tool = self._get_retrieval_tool()
        if tool is None:
            observation = "Execution output of [web_search]:\nError: web_search tool is not available. LocalRetrievalTool could not be initialized."
            return observation, 0.0, False, {}

        try:
            result = tool.forward(query=query, top_k=top_k)
            raw = result.to_string()
            # P2-8: dedup passages already seen in this rollout so the
            # model is forced to issue queries that surface new evidence.
            # simpleqa/gpqa at step-0 showed 864 / 138 consecutive
            # near-identical responses respectively.
            seen: set = getattr(self, "_search_retrieval_seen_docs", set())
            new_chunks: list[str] = []
            dup_count = 0
            for chunk in raw.split("\n\n"):
                normalized = re.sub(r"\s+", " ", chunk.strip().lower())
                normalized = re.sub(r"[^\w\s]", "", normalized)
                if not normalized:
                    continue
                h = hashlib.sha1(normalized[:1000].encode("utf-8", errors="ignore")).hexdigest()
                if h in seen:
                    dup_count += 1
                    continue
                seen.add(h)
                new_chunks.append(chunk)
            if not new_chunks and raw.strip():
                # All chunks dedup'd away — surface a hint instead of
                # returning the identical passage a second time.
                body = (
                    "All returned passages were already surfaced by a previous search. "
                    "Do not submit an answer from repeated evidence. Rewrite the query with a different clue, exact title phrase, named entity, date, or number, then call web_search again."
                )
                self._search_rewrite_required = True
                self._search_rewrite_reason = "all returned passages were duplicates"
            else:
                body = "\n\n".join(new_chunks) if new_chunks else raw
            self._search_retrieval_seen_docs = seen
            self._search_retrieval_duplicate_hits = getattr(self, "_search_retrieval_duplicate_hits", 0) + dup_count
            # P0-2: count low-content responses (fewer than 20 words
            # after stripping the header) for reward shaping downstream.
            word_count = len(body.split())
            useful_result = bool(new_chunks) and word_count >= 20
            if word_count < 20:
                self._search_low_content_responses = getattr(self, "_search_low_content_responses", 0) + 1
                self._search_rewrite_required = True
                self._search_rewrite_reason = "low-content or generic search result"
            elif useful_result:
                self._search_rewrite_required = False
                self._search_rewrite_reason = ""
            if query_text:
                history.append(query_text)
                self._search_query_history = history
                self._search_used_precise_clues = used_clues | query_clues
            if useful_result:
                self._search_rewrite_rejections = 0
            observation = f"Execution output of [web_search]:\n{body}"
        except Exception as e:
            logger.error("web_search execution failed: %s", str(e))
            observation = f"Execution output of [web_search]:\nError executing web_search: {str(e)}"

        return observation, 0.0, False, {}

    def _handle_search_finish(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle finish/submit tool call in web search mode."""
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        result = params.get("result", "")
        if self._is_placeholder_answer_marker(str(result or "")):
            return (
                "Error: FINAL_ANSWER is a placeholder, not an answer. Continue searching if needed, then submit the actual final answer in \\boxed{...}.",
                0.0,
                False,
                {"search/placeholder_submit_rejected": 1},
            )
        self._search_answer = result
        self._search_answer_is_verbatim_submission = True
        if self.harness not in {"cot", "bare"} and getattr(self, "_search_web_search_calls", 0) == 0:
            return self._search_bypass_termination("finish/submit before web_search")
        return "Your answer has been submitted.", 0.0, True, {}

    @staticmethod
    def _search_bypass_termination(reason: str) -> tuple[str, float, bool, dict]:
        message = f"{reason} in a tool-use web-search harness"
        return (
            f"Error: {message}; terminating rollout.",
            0.0,
            True,
            {
                "termination_reason": "ABNORMAL_SEARCH_BYPASS",
                "termination_message": message,
                "credit_assignment": "reasoning_step_only",
                "reward/bypass_termination": 1.0,
            },
        )

    def _handle_mcp_finish(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle finish/submit tool call in MCP mode.

        Tries to preserve the submitted result as a proper Python object so
        that the verifier receives a dict/list instead of a stringified blob.
        """
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        result = params.get("result", params.get("response", ""))

        # Try to get a structured object out of the result
        parsed = result
        if isinstance(result, str) and result.strip():
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                parsed = result

        accepted, observation, info = self._accept_mcp_submission(parsed, result)
        if not accepted:
            return observation, 0.0, bool(info.get("termination_reason")), info
        return "Your answer has been submitted.", 0.0, True, {}

    def _handle_mcp_submit_result(self, action_obj) -> tuple[str, float, bool, dict]:
        """Handle submit_result_difficulty_xxx tool call in MCP mode."""
        self.total_steps += 1
        params = action_obj.parameters if hasattr(action_obj, "parameters") else {}
        result = params.get("result", "")

        # Try to get a structured object out of the result
        parsed = result
        if isinstance(result, str) and result.strip():
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                parsed = result

        accepted, observation, info = self._accept_mcp_submission(parsed, result)
        if not accepted:
            return observation, 0.0, bool(info.get("termination_reason")), info

        # Also execute on the MCP server if available (for side effects)
        if self._mcp_connection_manager is not None:
            fn = action_obj.function_name
            # Restore types for MCP server execution (same as _step_mcp)
            restored_params = {}
            for k, v in params.items():
                if isinstance(v, str):
                    try:
                        restored_params[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        restored_params[k] = v
                else:
                    restored_params[k] = v
            tool_call_id = str(uuid.uuid4())
            tool_calls = [
                {
                    "id": tool_call_id,
                    "function": {
                        "name": fn,
                        "arguments": json.dumps(restored_params, ensure_ascii=False),
                    },
                }
            ]
            try:
                self._mcp_connection_manager.execute_tool_calls(tool_calls)
            except Exception:
                pass

        return "Your answer has been submitted.", 0.0, True, {}

    def _accept_mcp_submission(self, parsed: object, raw_result: object) -> tuple[bool, str, dict]:
        """Store a submission only if it passes the local schema self-check."""
        check = _validate_submission_schema(parsed, getattr(self, "_mcp_answer_schema", None))
        self._mcp_last_schema_self_check = check
        normalized = check.get("normalized_payload", parsed)
        if not check.get("passed", True):
            self._mcp_schema_self_check_failures += 1
            if self._mcp_schema_self_check_failures >= _MCP_SCHEMA_SELF_CHECK_MAX_FAILURES:
                errors = check.get("errors", [])
                return (
                    False,
                    _submission_self_check_observation(check) + "\nSchema self-check failed too many times; terminating rollout.",
                    {
                        "mcp/schema_self_check_failed": 1,
                        "mcp/schema_self_check_errors": errors,
                        "termination_reason": "MCP_SCHEMA_SELF_CHECK_EXCEEDED",
                        "termination_message": f"MCP schema self-check failed {self._mcp_schema_self_check_failures} times.",
                    },
                )
            return (
                False,
                _submission_self_check_observation(check),
                {
                    "mcp/schema_self_check_failed": 1,
                    "mcp/schema_self_check_errors": check.get("errors", []),
                },
            )

        if isinstance(normalized, (dict, list)):
            self._mcp_answer = json.dumps(normalized, ensure_ascii=False)
        elif isinstance(normalized, str) and normalized.strip():
            self._mcp_answer = normalized
        else:
            self._mcp_answer = str(raw_result) if raw_result else ""
        return True, "", {}

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def compute_final_reward(self):
        if self._task_mode == "mcp":
            return self._compute_mcp_reward()
        if self._task_mode == "web search":
            return self._compute_search_reward()
        if self._task_mode == "et":
            return self._compute_et_reward()
        return super().compute_final_reward()

    def compute_final_reward_metadata(self) -> dict:
        if self._task_mode == "mcp":
            self._compute_mcp_reward()
            return self._mcp_reward_debug
        if self._task_mode == "web search":
            self._compute_search_reward()
            return self._search_reward_debug
        if self._task_mode == "et":
            self._compute_et_reward()
            return self._et_reward_debug
        return super().compute_final_reward_metadata()

    def _compute_et_reward(self) -> float:
        """ET-mode reward: delegate to ETEnv.compute_final_reward_metadata,
        then surface the binary 1.0/0.0 as the rollout reward.
        """
        if self._et_inner is None:
            self._et_reward_debug = {
                "type": "endless_terminals",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "et_env_uninitialized",
                "verifier_error": "ETEnv was never reset; cannot run verifier.",
            }
            self._reward_debug = self._et_reward_debug
            return 0.0
        meta = {}
        try:
            meta = self._et_inner.compute_final_reward_metadata() or {}
        except Exception as exc:
            meta = {
                "type": "endless_terminals",
                "reward": 0.0,
                "resolved": False,
                "reward_mode": "binary",
                "reward_source": "et_env_exception",
                "verifier_error": f"{type(exc).__name__}: {exc}"[:512],
            }
        meta.setdefault("type", "endless_terminals")
        meta.setdefault("reward_mode", "binary")
        if self._record_parser_unknown_metrics:
            meta.setdefault("parser/unknown_total", self._et_unknown_total)
        reward = float(meta.get("reward", 0.0))
        meta["reward"] = reward
        meta.setdefault("resolved", reward >= 1.0)
        self._et_reward_debug = meta
        self._reward_debug = self._et_reward_debug
        return reward

    def _compute_search_reward(self) -> float:
        """Compute strict EM-based reward for web search tasks."""
        from rllm.rewards.reward_types import RewardConfig, RewardInput
        from rllm.rewards.search_reward import RewardSearchFn

        ground_truth = self.entry.get("ground_truth") or self.entry.get("answer") or self.entry.get("gt_answer") or self.entry.get("ground_truth_answer", "")
        answer = self._search_answer
        is_verbatim_submission = bool(getattr(self, "_search_answer_is_verbatim_submission", True))
        if self.harness in {"cot", "bare"} and not str(answer or "").strip():
            raw_answer = str(getattr(self, "_search_last_raw_action", "") or "").strip()
            if raw_answer and not self._has_tool_call_markup(raw_answer):
                final_marker = self._extract_final_answer_marker_from_raw(raw_answer)
                if final_marker is not None:
                    answer = final_marker.strip()
                    is_verbatim_submission = True
                else:
                    answer = raw_answer
                    is_verbatim_submission = False
                self._search_answer = answer
                self._search_answer_is_verbatim_submission = is_verbatim_submission

        config = RewardConfig(
            toolcall_bonus=0.0,
            apply_repetition_penalty=False,
            repetition_penalty_weight=0.0,
            apply_length_penalty=False,
            length_penalty_weight=0.0,
            enable_step_bonus=False,
        )
        reward_fn = RewardSearchFn(config)
        question_text = self.entry.get("question") or self.entry.get("query") or self.entry.get("input") or self.entry.get("problem_statement", "")
        reward_input = RewardInput(
            task_info={
                "ground_truth": ground_truth,
                "step_count": self.total_steps,
                "question": question_text,
                "data_source": self.entry.get("data_source"),
                # Explicit finish/submit and boxed rescue produce canonical
                # answer strings, so the verifier skips the prose-scavenging
                # cascade. Prompt-only COT text still needs the cascade to pull
                # the final answer out of natural language.
                "is_submitted": is_verbatim_submission,
            },
            action=answer,
        )
        reward_output = reward_fn(reward_input)

        ws_calls = getattr(self, "_search_web_search_calls", 0)
        low_content = getattr(self, "_search_low_content_responses", 0)
        dup_hits = getattr(self, "_search_retrieval_duplicate_hits", 0)
        rewrite_rejections = getattr(self, "_search_rewrite_rejections", 0)

        # Web-search training now uses only answer-match reward. Keep the
        # historical metric fields for dashboards, but do not apply them.
        bypass_penalty = 0.0
        early_junk_penalty = 0.0

        final_reward = max(0.0, min(1.0, float(reward_output.reward)))

        # --- Rollout-time answer rescue (mirrors experiments/fused/merge_eval_json.py) ---
        # For cot/bare, the verbatim/marker submission above can miss the model's
        # real final answer: long COT responses frequently contain stray
        # tool-call markup (or emit an empty ``finish``), so ``_step_search``
        # routes the turn through the tool-call parser and submits an empty
        # verbatim answer even though ``<answer>X</answer>`` / ``\boxed{...}`` /
        # a committed prose choice sits at the end. ``_search_last_raw_action``
        # is captured before any branching, so it reliably holds the full final
        # assistant text. Re-evaluate it with the prose cascade + MCQ commitment
        # scan (``is_submitted=False``) and union the exact-match decision, so the
        # rollout-time reward/pass@1 logged to the console already reflects every
        # answer the post-hoc JSON rescore would recover. Union only (never
        # downgrades a submission that already scored correct).
        rollout_rescue_applied = False
        if self.harness in {"cot", "bare"} and not bool(reward_output.is_correct):
            raw_final = str(getattr(self, "_search_last_raw_action", "") or "").strip()
            if raw_final:
                rescue_output = reward_fn(
                    RewardInput(
                        task_info={
                            "ground_truth": ground_truth,
                            "step_count": self.total_steps,
                            "question": question_text,
                            "data_source": self.entry.get("data_source"),
                            "is_submitted": False,
                        },
                        action=raw_final,
                    )
                )
                if rescue_output.is_correct:
                    reward_before_rescue = final_reward
                    reward_output = rescue_output
                    final_reward = max(0.0, min(1.0, float(rescue_output.reward)))
                    rollout_rescue_applied = True

        parser_unknown_metadata = self._parser_unknown_metadata(unknown_total=getattr(self, "_search_unknown_total", 0))
        self._search_reward_debug = {
            "type": "web search",
            "reward": final_reward,
            "resolved": final_reward >= 1.0,
            "reward_mode": "em",
            "reward_source": "search_reward_fn",
            "is_correct": bool(reward_output.is_correct) if reward_output.is_correct is not None else False,
            "verifier_error": "",
            "reward/bypass_penalty": bypass_penalty,
            "reward/early_junk_penalty": early_junk_penalty,
            "reward/web_search_calls": ws_calls,
            "reward/low_content_responses": low_content,
            "reward/duplicate_passage_hits": dup_hits,
            "reward/query_rewrite_rejections": rewrite_rejections,
            "reward/query_rewrite_required": bool(getattr(self, "_search_rewrite_required", False)),
            "implicit_text_submission": not is_verbatim_submission,
            "rollout_rescue/applied": rollout_rescue_applied,
            **({"rollout_rescue/previous_reward": reward_before_rescue} if rollout_rescue_applied else {}),
            **parser_unknown_metadata,
            **reward_output.metadata,
        }
        self._reward_debug = self._search_reward_debug
        return final_reward

    def _compute_mcp_reward(self) -> float:
        """Compute verifier-based reward for MCP tasks."""
        from rllm.rewards.reward_types import RewardOutput
        from rllm.rewards.verifier_reward import verifier_reward_fn

        task_info = {
            **self.entry,
            "tool_call_stats": {
                "submit_called": bool(self._mcp_answer),
                "non_submit_tool_calls": self.total_steps - (1 if self._mcp_answer else 0),
                "step_count": self.total_steps,
                "distinct_successful_tools": len(self._mcp_distinct_tools),
            },
        }
        answer = self._mcp_answer
        parsed_answer: object = answer
        if isinstance(answer, str):
            try:
                parsed_answer = json.loads(answer)
            except (json.JSONDecodeError, ValueError):
                parsed_answer = answer

        try:
            reward_output = verifier_reward_fn(task_info=task_info, action=answer)
        except Exception as e:
            logger.error("MCP reward computation failed: %s", e)
            reward_output = RewardOutput(reward=0.0, metadata={"verifier_error": str(e)})

        verifier_reward = float(reward_output.reward)
        answer_text = str(answer or "").strip()
        non_submit_tool_calls = self.total_steps - (1 if self._mcp_answer else 0)
        distinct_successful_tools = len(self._mcp_distinct_tools)

        placeholder_markers = (
            '"key": "val"',
            '"key":"val"',
            "val2",
            "brush stroke",
            '"name": "finish"',
            '"name":"finish"',
        )
        is_placeholder = any(marker in answer_text for marker in placeholder_markers)
        nontrivial_submit = bool(answer_text) and len(answer_text) >= 100 and not is_placeholder

        shaped_reward = verifier_reward
        shaping_components = {
            "reward/mcp_submit_bonus": 0.0,
            "reward/mcp_tool_evidence_bonus": 0.0,
            "reward/mcp_answer_length_bonus": 0.0,
        }
        if verifier_reward < 1.0 and nontrivial_submit and non_submit_tool_calls > 0 and distinct_successful_tools > 0:
            shaping_components["reward/mcp_submit_bonus"] = 0.12
            shaping_components["reward/mcp_tool_evidence_bonus"] = min(distinct_successful_tools, 4) * 0.05
            shaping_components["reward/mcp_answer_length_bonus"] = min(len(answer_text) / 3000.0, 1.0) * 0.06
            shaped_reward = min(0.40, verifier_reward + sum(shaping_components.values()))

        self._mcp_reward_debug = {
            "type": "mcp",
            "reward": shaped_reward,
            "base_reward": verifier_reward,
            "resolved": verifier_reward >= 1.0,
            "reward_mode": "verifier_with_shaping",
            "reward_source": "verifier_reward_fn+mcp_shaping",
            "is_correct": bool(reward_output.is_correct) if reward_output.is_correct is not None else False,
            "verifier_error": reward_output.metadata.get("error", ""),
            "submit_called": bool(self._mcp_answer),
            "non_submit_tool_calls": non_submit_tool_calls,
            "distinct_successful_tools": distinct_successful_tools,
            "nontrivial_submit": nontrivial_submit,
            "placeholder_submit": is_placeholder,
            "schema_self_check": getattr(self, "_mcp_last_schema_self_check", {}),
            "schema_self_check_failures": getattr(self, "_mcp_schema_self_check_failures", 0),
            "mcp_tool_evidence_count": len(getattr(self, "_mcp_tool_evidence", [])),
            "evidence_to_field": _build_evidence_to_field_trace(parsed_answer, getattr(self, "_mcp_tool_evidence", []))
            if isinstance(parsed_answer, (dict, list))
            else {"field_count": 0, "mapped_count": 0, "missing_or_weak_count": 0, "mappings": []},
            **shaping_components,
            **reward_output.metadata,
        }
        self._reward_debug = self._mcp_reward_debug
        return shaped_reward

    @property
    def reward_debug(self) -> dict:
        if self._task_mode == "mcp":
            return self._mcp_reward_debug
        if self._task_mode == "web search":
            return self._search_reward_debug
        if self._task_mode == "et":
            return self._et_reward_debug
        return self._reward_debug

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self):
        """Clean up resources."""
        # Release the MCP connection manager. Pooled managers (shared across a
        # task's rollouts) are refcounted — only the last release stops the
        # server. A non-pooled manager (no pool key) is stopped directly.
        if self._mcp_connection_manager is not None:
            if self._mcp_pool_key is not None:
                self._release_mcp_manager(self._mcp_pool_key)
            else:
                try:
                    self._mcp_connection_manager.stop()
                except Exception:
                    pass
            self._mcp_connection_manager = None
            self._mcp_pool_key = None
        # Do NOT close _shared_retrieval_tool — it is shared across all FusedEnv instances
        # ET mode owns its own ETEnv instance — close it (best-effort, gated
        # internally by RLLM_ET_KEEP_CONTAINER for debugging).
        if self._task_mode == "et" and self._et_inner is not None:
            try:
                self._et_inner.close()
            except Exception:
                pass
            self._et_inner = None
        # Clean up Docker (CLI mode only)
        if self._task_mode == "cli":
            super().close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(extra_info: dict | str) -> "FusedEnv":
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        # Walk the MRO to collect all accepted __init__ params, since
        # FusedEnv.__init__ forwards **kwargs to parent classes.
        accepted = set()
        for cls in FusedEnv.__mro__:
            if cls is object:
                continue
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                accepted.add(name)

        init_params = {k: v for k, v in extra_info.items() if k in accepted}
        init_params["entry"] = extra_info
        return FusedEnv(**init_params)
