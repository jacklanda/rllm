from __future__ import annotations

import re


_BAD_LINE_PREFIX_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<number>\d+)[ \t]+(?P<code>(?:raise|return|if|for|while|elif|else\b|except\b|with|assert\b|[A-Za-z_][A-Za-z0-9_]*\s*=).*)$"
)
_BAD_ARGUMENT_NUMBER_RE = re.compile(r"(?<=,)\s*\d+\s+(?=(?!(?:if|for|else)\b)[A-Za-z_])")
_SINGLE_QUOTED_INDEX_RE = re.compile(r"\['(?P<key>[^']+)'\]")
_FSTRING_DOUBLE_QUOTED_EXPR_RE = re.compile(r'\["\{(?P<expr>[^}]+)\}"\]')
_FSTRING_DOUBLE_QUOTED_INDEX_RE = re.compile(r'\["(?P<key>[A-Za-z_][A-Za-z0-9_]*)"\]')
_NON_ASCII_NUMERIC_DEFAULT_RE = re.compile(r"(?P<prefix>=\s*)[^\x00-\x7F]+(?P<number>-?\d+(?:\.\d+)?)\b")
_ROMAN_NUMERIC_DEFAULT_RE = re.compile(
    r"(?P<prefix>:\s*(?:int|float)\s*=\s*)(?P<roman>I|II|III|IV|V|VI|VII|VIII|IX|X)\b"
)
_BARE_IDENTIFIER_DEFAULT_RE = re.compile(
    r"(?P<prefix>:\s*(?P<type>int|float|str|bool)\s*=\s*)"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()"
)
_ENUM_ALL_DEFAULT_RE = re.compile(
    r"(?P<prefix>:\s*(?P<enum>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*)(?P=enum)\.ALL\b"
)
_DUPLICATED_BASE_DIR_PATH_RE = re.compile(
    r"Path\(__file__\)\.parent\s*/\s*(?:['\"]data['\"]\s*/\s*)?BASE_DIR\s*/\s*['\"]data['\"]"
)


def sanitize_tools_source(source: str) -> str:
    """Normalize common generated ``tools.py`` defects before import/exec."""
    rewritten: list[str] = []
    saw_fastmcp_instance = False

    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("mcp") and "= FastMCP(" in stripped:
            saw_fastmcp_instance = True
            rewritten.append(line)
            indent = line[: len(line) - len(line.lstrip())]
            if not indent:
                rewritten.append("tool = mcp.tool\n")
            continue

        if saw_fastmcp_instance and stripped == "import mcp":
            indent = line[: len(line) - len(line.lstrip())]
            if indent:
                rewritten.append(f"{indent}pass\n")
            else:
                rewritten.append("import mcp as _rllm_mcp_package\n")
            continue
        if saw_fastmcp_instance and stripped in {"from .tools import mcp", "from tools import mcp"}:
            rewritten.append("\n")
            continue
        if saw_fastmcp_instance and stripped == "from mcp import tool":
            rewritten.append("\n")
            continue
        if stripped in {"from . import BASE_DIR", "from tools import BASE_DIR"}:
            indent = line[: len(line) - len(line.lstrip())]
            rewritten.append(f"{indent}BASE_DIR = Path(__file__).resolve().parent\n")
            continue
        if re.match(r"^BASE_DIR\s*=\s*Path\(__file__\)(?:\.resolve\(\))?\.parent\.parent\s*$", stripped):
            indent = line[: len(line) - len(line.lstrip())]
            rewritten.append(f"{indent}BASE_DIR = Path(__file__).resolve().parent\n")
            continue

        rewritten.append(line)

    normalized = _normalize_builtin_any_annotations("".join(rewritten))
    normalized = _normalize_non_ascii_numeric_defaults(normalized)
    normalized = _normalize_roman_numeric_defaults(normalized)
    normalized = _normalize_bare_identifier_defaults(normalized)
    normalized = _normalize_duplicated_base_dir_paths(normalized)
    normalized = _normalize_missing_enum_all_defaults(normalized)
    normalized = _repair_empty_try_blocks(normalized)
    normalized = _normalize_bare_tool_decorators(normalized)
    return _repair_syntax_line_prefixes(normalized)


def _normalize_roman_numeric_defaults(source: str) -> str:
    """Repair generated defaults such as ``min_count: int = III``."""
    values = {
        "I": "1",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
    }

    def replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{values[match.group('roman')]}"

    return _ROMAN_NUMERIC_DEFAULT_RE.sub(replace, source)


def _normalize_duplicated_base_dir_paths(source: str) -> str:
    """Repair path chains that accidentally include ``BASE_DIR / data`` twice."""
    return _DUPLICATED_BASE_DIR_PATH_RE.sub("BASE_DIR / 'data'", source)


def _normalize_bare_identifier_defaults(source: str) -> str:
    """Repair generated typed defaults such as ``case0`` or ``chatgpt``."""
    allowed = {"True", "False", "None"}

    def replace(match: re.Match[str]) -> str:
        default_name = match.group("name")
        if default_name in allowed:
            return match.group(0)

        annotation = match.group("type")
        if annotation == "str":
            return f"{match.group('prefix')}{default_name!r}"
        if annotation == "bool":
            return f"{match.group('prefix')}False"

        trailing_number = re.search(r"(\d+)$", default_name)
        if trailing_number:
            number = trailing_number.group(1)
            roman_prefix = default_name[: -len(number)]
            roman_values = {
                "I": "1",
                "II": "2",
                "III": "3",
                "IV": "4",
                "V": "5",
                "VI": "6",
                "VII": "7",
                "VIII": "8",
                "IX": "9",
                "X": "10",
            }
            if roman_prefix in roman_values:
                number = f"{roman_values[roman_prefix]}{number}"
            value = number
        else:
            value = "0"

        if annotation == "float" and "." not in value:
            value = f"{value}.0"
        return f"{match.group('prefix')}{value}"

    return _BARE_IDENTIFIER_DEFAULT_RE.sub(replace, source)


def _normalize_missing_enum_all_defaults(source: str) -> str:
    """Replace ``SomeEnum.ALL`` defaults when ``SomeEnum`` has no ALL member."""
    enum_members = _collect_enum_members(source)
    if not enum_members:
        return source

    def replace(match: re.Match[str]) -> str:
        enum_name = match.group("enum")
        members = enum_members.get(enum_name)
        if not members or "ALL" in members:
            return match.group(0)
        return f"{match.group('prefix')}{enum_name}.{members[0]}"

    return _ENUM_ALL_DEFAULT_RE.sub(replace, source)


def _collect_enum_members(source: str) -> dict[str, list[str]]:
    members_by_enum: dict[str, list[str]] = {}
    lines = source.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = re.match(r"^(?P<indent>[ \t]*)class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<bases>[^)]*\bEnum\b[^)]*)\)\s*:", line)
        if not match:
            idx += 1
            continue
        class_indent = len(match.group("indent").replace("\t", "    "))
        members: list[str] = []
        idx += 1
        while idx < len(lines):
            body_line = lines[idx]
            stripped = body_line.strip()
            if not stripped or stripped.startswith("#"):
                idx += 1
                continue
            body_indent = len(body_line[: len(body_line) - len(body_line.lstrip())].replace("\t", "    "))
            if body_indent <= class_indent:
                break
            member_match = re.match(r"^[ \t]+([A-Z][A-Z0-9_]*)\s*=", body_line)
            if member_match:
                members.append(member_match.group(1))
            idx += 1
        members_by_enum[match.group("name")] = members
    return members_by_enum


def _normalize_non_ascii_numeric_defaults(source: str) -> str:
    """Repair generated defaults such as ``min_count: int =常说0``."""
    return _NON_ASCII_NUMERIC_DEFAULT_RE.sub(r"\g<prefix>\g<number>", source)


def _repair_empty_try_blocks(source: str) -> str:
    """Insert ``pass`` into generated empty ``try:`` blocks."""
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    while idx < len(lines):
        out.append(lines[idx])
        stripped = lines[idx].strip()
        if stripped == "try:":
            indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
            scan = idx + 1
            while scan < len(lines) and (not lines[scan].strip() or lines[scan].lstrip().startswith("#")):
                scan += 1
            if scan < len(lines):
                next_stripped = lines[scan].lstrip()
                next_indent = lines[scan][: len(lines[scan]) - len(next_stripped)]
                if next_indent == indent and next_stripped.startswith(("except ", "except:", "finally:")):
                    out.append(f"{indent}    pass\n")
        idx += 1
    return "".join(out)


def _normalize_bare_tool_decorators(source: str) -> str:
    """Map orphan ``@tool`` decorators to the active FastMCP instance."""
    if "@tool" not in source:
        return source
    lines = source.splitlines(keepends=True)
    saw_mcp_tool = any(line.lstrip().startswith("@mcp.tool") for line in lines)
    if not saw_mcp_tool:
        return source
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("@tool("):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}@mcp.tool({stripped[len('@tool('):]}")
        else:
            out.append(line)
    return "".join(out)


def _repair_syntax_line_prefixes(source: str) -> str:
    """Strip stray line-number prefixes only when they actually break syntax."""
    lines = source.splitlines(keepends=True)
    for _ in range(100):
        try:
            compile("".join(lines), "<tools.py>", "exec")
            break
        except (SyntaxError, IndentationError) as e:
            lineno = e.lineno
            if lineno is None or lineno < 1 or lineno > len(lines):
                break
            line = lines[lineno - 1]
            line_ending = "\n" if line.endswith("\n") else ""
            body = line[:-1] if line_ending else line
            match = _BAD_LINE_PREFIX_RE.match(body)
            if match:
                lines[lineno - 1] = f"{match.group('indent')}{match.group('code')}{line_ending}"
                continue

            repaired_arg = _BAD_ARGUMENT_NUMBER_RE.sub(" ", line)
            if repaired_arg != line:
                lines[lineno - 1] = repaired_arg
                continue

            repaired_fstring = _repair_common_fstring_quote_error(line)
            if repaired_fstring != line:
                lines[lineno - 1] = repaired_fstring
                continue

            if isinstance(e, IndentationError) and "expected an indented block after" in str(e):
                if _insert_pass_for_empty_block(lines, lineno):
                    continue

            break
    return "".join(lines)


def _repair_common_fstring_quote_error(line: str) -> str:
    if "f'" in line:
        line = _SINGLE_QUOTED_INDEX_RE.sub(r'["\g<key>"]', line)
        line = line.replace('"]})', '"])}')
    if 'f"' in line:
        line = _FSTRING_DOUBLE_QUOTED_EXPR_RE.sub(r"['{\g<expr>}']", line)
        line = _FSTRING_DOUBLE_QUOTED_INDEX_RE.sub(r"['\g<key>']", line)
        line = line.replace("']})", "'])}")
    stripped = line.rstrip("\n")
    if 'raise ValueError("' in stripped and stripped.endswith("')"):
        return stripped[:-2] + '")' + ("\n" if line.endswith("\n") else "")
    return line


def _insert_pass_for_empty_block(lines: list[str], error_lineno: int) -> bool:
    insert_at = error_lineno - 1
    for prev_idx in range(insert_at - 1, -1, -1):
        prev = lines[prev_idx]
        if not prev.strip():
            continue
        if prev.lstrip().startswith("#"):
            continue
        if not prev.rstrip().endswith(":"):
            return False
        indent = re.match(r"^[ \t]*", prev).group(0)
        lines.insert(insert_at, f"{indent}    pass\n")
        return True
    return False


def _normalize_builtin_any_annotations(source: str) -> str:
    out: list[str] = []
    in_sig = False
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("def ") or in_sig:
            line = re.sub(r"(?<=[:\[, ])any(?=[]\[,)=\n ])", "Any", line)
            in_sig = not stripped.endswith(":\n") and not stripped.endswith(":")
        out.append(line)
    return "".join(out)
