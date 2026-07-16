"""Reward function selection for agentic evaluation.

All agentic benchmarks covered here are open-ended QA with string answers, so
we default to ``rllm.rewards.reward_fn.search_reward_fn`` (Exact Match + F1
against ground-truth strings).

Eval-time answer extraction
---------------------------
At eval time we only trust the model's ``\\boxed{...}`` answer. If no boxed
span is present we fall back to the raw (post-``<think>``-strip) response
and let the downstream EM/F1 grader decide — we deliberately skip rllm's
heuristic cascade (bold / dates / proper names / "the answer is X" / sentence
scoring) so accuracy is not inflated by extractor rescues on malformed
outputs.
"""

from __future__ import annotations

import re
import inspect
from typing import Any, Callable
import importlib.util
from pathlib import Path
from rllm.rewards.reward_fn import search_reward_fn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


RewardFn = Callable[[dict, str], RewardOutput]
_CHOICE_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def _reward_config(**kwargs) -> RewardConfig:
    params = inspect.signature(RewardConfig).parameters
    return RewardConfig(**{k: v for k, v in kwargs.items() if k in params})


class _BoxedFirstSearchFn(RewardSearchFn):
    """Boxed-first extractor with raw-response fallback.

    Reuses ``RewardSearchFn``'s boxed-unwrapping logic via ``super()`` — it
    already handles ``\\boxed{...}``, ``boxed{...}``, and ``oxed{...}`` with
    nested-brace counting and strips LaTeX wrappers. If super() returns a
    value that looks like it came from a non-boxed fallback (bold, date,
    name, number, "the answer is X", sentence score, etc.), we instead
    return the raw response so the grader scores the agent's final text
    verbatim.
    """

    def extract_answer_from_response(self, response: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""

        if re.search(r"(?:\\boxed|boxed|oxed)\s*\{", cleaned):
            extracted = super().extract_answer_from_response(response)
            if extracted:
                return extracted

        return cleaned


def _boxed_first_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """search_reward_fn variant that uses the boxed-first extractor."""
    reward_config = _reward_config()
    fn = _BoxedFirstSearchFn(reward_config)
    return fn(RewardInput(task_info=task_info, action=action))


def _last_boxed_content(text: str) -> str | None:
    """Extract the last boxed span, accepting common malformed variants."""
    matches: list[str] = []
    for match in re.finditer(r"(?:\\boxed|boxed|oxed)\s*\{", text):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(text) and depth:
            depth += (text[pos] == "{") - (text[pos] == "}")
            pos += 1
        if depth == 0:
            matches.append(text[start : pos - 1])
    return matches[-1].strip() if matches else None


def _strip_latex_text_wrappers(text: str) -> str:
    out = (text or "").strip()
    out = re.sub(r"^\s*\$+\s*|\s*\$+\s*$", "", out)
    out = out.replace("\\mathrm", "\\text")
    for _ in range(4):
        match = re.fullmatch(r"\\text\s*\{\s*(.*?)\s*\}", out, flags=re.DOTALL)
        if not match:
            break
        out = match.group(1).strip()
    return out


def _normalize_choice_text(text: str) -> str:
    out = _strip_latex_text_wrappers(text).lower()
    out = out.replace("∞", " infinity ")
    out = out.replace("\\infty", " infinity ")
    out = re.sub(
        r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        r"\1 \2",
        out,
    )
    out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt \1", out)
    out = re.sub(r"\\(?:mathrm|operatorname|text)\s*\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\(?:left|right)\b", " ", out)
    out = re.sub(r"\\[,;:! ]+", " ", out)
    out = re.sub(r"[^a-z0-9]+", " ", out)
    ignored = {
        "displaystyle",
        "frac",
        "left",
        "right",
        "mathrm",
        "text",
    }
    return " ".join(tok for tok in out.split() if tok not in ignored)


def _choice_text_matches(candidate: str, option: str) -> bool:
    candidate_norm = _normalize_choice_text(candidate)
    option_norm = _normalize_choice_text(option)
    if not candidate_norm or not option_norm:
        return False
    if candidate_norm == option_norm:
        return True

    candidate_tokens = candidate_norm.split()
    option_tokens = option_norm.split()
    if len(candidate_tokens) < 2 or len(option_tokens) < 2:
        return False
    overlap = len(set(candidate_tokens) & set(option_tokens))
    return (
        overlap >= min(len(candidate_tokens), len(option_tokens))
        and abs(len(candidate_tokens) - len(option_tokens)) <= 1
    )


def _question_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    for line in (question or "").splitlines():
        match = re.match(r"\s*([A-D])\s*[.)]\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if match:
            if current is not None:
                options[current] = " ".join(chunks).strip()
            current = match.group(1).upper()
            chunks = [match.group(2).strip()]
        elif current is not None and line.strip():
            chunks.append(line.strip())
    if current is not None:
        options[current] = " ".join(chunks).strip()
    return options


def _choice_from_candidate(candidate: str) -> str:
    candidate = _strip_latex_text_wrappers(candidate)
    candidate = candidate.strip(" \t\r\n`*_")
    candidate = re.sub(r"^\s*\$+\s*|\s*\$+\s*$", "", candidate).strip()
    patterns = (
        r"^\s*(?:the\s+)?(?:option|choice|answer)\s*(?:is|:)?"
        r"\s*[\[(]?\s*\**([A-D])\**\s*(?:[\]).,;:]|$)",
        r"^\s*(?:\\text\s*\{\s*)?\**([A-D])\**\s*(?:\}|[.)\]:,-]|$)",
    )
    for pattern in patterns:
        match = re.match(pattern, candidate, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _tail_answer_text_candidates(text: str) -> list[str]:
    """Short final-answer spans that can be matched against option contents."""
    tail = (text or "")[-1800:]
    spans: list[str] = []
    patterns = (
        r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)\s*(.{1,180})",
        r"(?:therefore|thus|so|hence),?\s*(?:the\s+)?(?:final\s+)?"
        r"answer\s*(?:is|:)\s*(.{1,180})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, tail, flags=re.IGNORECASE | re.DOTALL):
            span = match.group(1).strip()
            span = re.split(r"(?:\n{2,}|(?<=[.!?])\s+(?=[A-Z]))", span, maxsplit=1)[0]
            if span:
                spans.append(span.strip())
    return spans[-4:]


def _extract_gpqa_choice(response: str, question: str = "") -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    candidates: list[str] = []
    content_candidates: list[str] = []
    boxed = _last_boxed_content(cleaned)
    if boxed:
        candidates.append(boxed)
        content_candidates.append(boxed)
    tail_spans = _tail_answer_text_candidates(cleaned)
    candidates.extend(tail_spans)
    content_candidates.extend(tail_spans)
    if len(cleaned) <= 200:
        content_candidates.append(cleaned)
    candidates.append(cleaned)

    for candidate in candidates:
        choice = _choice_from_candidate(candidate)
        if choice:
            return choice

    answer_patterns = (
        r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*(?:the\s+)?(?:option|choice)\s*\**\s*([A-D])\s*\**\b",
        r"(?:final\s+answer|correct\s+answer|answer|choice|option)\s*(?:is|:)?\s*(?:\\text\s*\{\s*)?[\[(]?\s*\**([A-D])\**\b",
        r"(?:choose|select|pick)\s+(?:the\s+)?(?:option|choice)?\s*\**([A-D])\**\b",
        r"\b\**([A-D])\**\s*(?:is\s+correct|is\s+the\s+correct\s+answer)\b",
        r"(?:corresponds\s+to|matches|would\s+be)\s+(?:the\s+)?(?:option|choice)\s*\**\s*([A-D])\s*\**\b",
        r"(?:therefore|thus|so|hence),?\s*(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*\**([A-D])\**\b",
        r"(?:therefore|thus|so|hence),?\s*\**([A-D])\**\s*(?:[.)]|$)",
    )
    search_region = cleaned[-1200:]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, search_region, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    final_lines = [
        line.strip() for line in re.split(r"[\r\n]+", response or "") if line.strip()
    ]
    for line in reversed(final_lines[-6:]):
        choice = _choice_from_candidate(line)
        if choice:
            return choice

    options = _question_options(question)
    for letter, option in options.items():
        if any(_choice_text_matches(candidate, option) for candidate in content_candidates):
            return letter

    return ""


def _gpqa_diamond_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Strict multiple-choice reward for GPQA Diamond.

    GPQA ground truth is an option letter. The generic search reward is too
    permissive for this benchmark because its F1 fallback can both accept
    wrong prefixed answers and reject correct ``A. option text`` answers.
    """
    ground_truth = task_info.get("ground_truth") or task_info.get("answer")
    if isinstance(ground_truth, list):
        truth = str(ground_truth[0]) if ground_truth else ""
    else:
        truth = str(ground_truth or "")
    truth_match = _CHOICE_RE.search(truth)
    truth_choice = truth_match.group(1).upper() if truth_match else ""

    question = str(task_info.get("question") or "")
    extracted = _extract_gpqa_choice(action, question)
    is_correct = bool(extracted and truth_choice and extracted == truth_choice)
    return RewardOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        metadata={
            "extracted_answer": extracted,
            "ground_truths": [truth_choice] if truth_choice else [],
            "exact_match": is_correct,
            "f1_score": 1.0 if is_correct else 0.0,
            "evaluation_method": "gpqa_choice_exact",
        },
    )



def _extract_medqa_choice(response: str, question: str = "") -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    candidates: list[str] = []
    content_candidates: list[str] = []
    boxed = _last_boxed_content(cleaned)
    if boxed:
        candidates.append(boxed)
        content_candidates.append(boxed)
    tail_spans = _tail_answer_text_candidates(cleaned)
    candidates.extend(tail_spans)
    content_candidates.extend(tail_spans)
    if len(cleaned) <= 200:
        content_candidates.append(cleaned)
    candidates.append(cleaned)

    for candidate in candidates:
        choice = _choice_from_candidate(candidate)
        if choice:
            return choice

    answer_patterns = (
        r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*(?:the\s+)?(?:option|choice)\s*\**\s*([A-D])\s*\**\b",
        r"(?:final\s+answer|correct\s+answer|answer|choice|option)\s*(?:is|:)?\s*(?:\\text\s*\{\s*)?[\[(]?\s*\**([A-D])\**\b",
        r"(?:choose|select|pick)\s+(?:the\s+)?(?:option|choice)?\s*\**([A-D])\**\b",
        r"\b\**([A-D])\**\s*(?:is\s+correct|is\s+the\s+correct\s+answer)\b",
        r"(?:corresponds\s+to|matches|would\s+be)\s+(?:the\s+)?(?:option|choice)\s*\**\s*([A-D])\s*\**\b",
        r"(?:therefore|thus|so|hence),?\s*(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*\**([A-D])\**\b",
        r"(?:therefore|thus|so|hence),?\s*\**([A-D])\**\s*(?:[.)]|$)",
    )
    search_region = cleaned[-1200:]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, search_region, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    final_lines = [
        line.strip() for line in re.split(r"[\r\n]+", response or "") if line.strip()
    ]
    for line in reversed(final_lines[-6:]):
        choice = _choice_from_candidate(line)
        if choice:
            return choice

    options = _question_options(question)
    for letter, option in options.items():
        if any(_choice_text_matches(candidate, option) for candidate in content_candidates):
            return letter

    return ""


def _medqa_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """Strict multiple-choice reward for GPQA Diamond.

    GPQA ground truth is an option letter. The generic search reward is too
    permissive for this benchmark because its F1 fallback can both accept
    wrong prefixed answers and reject correct ``A. option text`` answers.
    """
    ground_truth = task_info.get("ground_truth") or task_info.get("answer")
    if isinstance(ground_truth, list):
        truth = str(ground_truth[0]) if ground_truth else ""
    else:
        truth = str(ground_truth or "")
    truth_match = _CHOICE_RE.search(truth)
    truth_choice = truth_match.group(1).upper() if truth_match else ""

    question = str(task_info.get("question") or "")
    extracted = _extract_medqa_choice(action, question)
    is_correct = bool(extracted and truth_choice and extracted == truth_choice)
    return RewardOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        metadata={
            "extracted_answer": extracted,
            "ground_truths": [truth_choice] if truth_choice else [],
            "exact_match": is_correct,
            "f1_score": 1.0 if is_correct else 0.0,
            "evaluation_method": "medqa_choice_exact",
        },
    )




def _make_configured_reward_fn(
    toolcall_bonus: float,
    correct_reward: float,
    incorrect_reward: float,
) -> RewardFn:
    """Build a boxed-first search reward with shaping terms disabled."""
    cfg = _reward_config(
        toolcall_bonus=toolcall_bonus,
        apply_repetition_penalty=False,
        correct_reward=correct_reward,
        incorrect_reward=incorrect_reward,
        enable_step_bonus=False,
    )

    def _fn(task_info: dict, action: str) -> RewardOutput:
        reward_fn = _BoxedFirstSearchFn(cfg)
        return reward_fn(RewardInput(task_info=task_info, action=action))

    return _fn


_BROWSECOMP_SCORER = None

def extract_answer_from_response(response: str) -> str:
    response = response.strip()

    # Remove thinking tags first
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"\s+", " ", response).strip()

    if not response:
        return ""

    # 1. HIGHEST PRIORITY: Look for \boxed{} or \boxed[] content
    def unbox(s: str) -> str | None:
        """Extract content from \boxed{} with proper nesting support"""
        try:
            i = s.find("boxed{")
            if i == -1:
                return None
            i += 6  # 6 == len("boxed{")
            depth = 1
            j = i
            while depth and j < len(s):
                depth += (s[j] == "{") - (s[j] == "}")
                j += 1
            if depth:
                return None  # unbalanced braces
            return s[i : j - 1]
        except (IndexError, ValueError):
            return None

    boxed_content = unbox(response)

    if boxed_content is not None:
        return boxed_content.strip()

    bold_patterns = [
        r"\*\*([^*]+)\*\*",
        r"\*([^*]+)\*",
    ]
    for pattern in bold_patterns:
        matches = re.findall(pattern, response)
        if matches:
            # Return the most substantive bold text (longer than 2 chars, not just punctuation)
            substantive_matches = [m.strip() for m in matches if len(m.strip()) > 2 and not re.match(r"^[^\w]*$", m.strip())]
            if substantive_matches:
                return substantive_matches[0]

    # 3. Extract dates (years, full dates)
    date_patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        r"\b(?:March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{4}\b",
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[0]

    # 4. Extract names (proper nouns - capitalized words)
    name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    name_matches = re.findall(name_pattern, response)
    if name_matches:
        # Filter out common non-name phrases
        non_names = {"United States", "New York", "Los Angeles", "Great Britain", "Middle East", "South Africa"}
        valid_names = [name for name in name_matches if name not in non_names and len(name.split()) <= 4]
        if valid_names:
            return valid_names[0]

    # 5. Extract numbers with context (for "how many", "when", etc.)
    number_patterns = [
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:votes?|dollars?|years?|months?|days?|people|million|billion|percent|%)\b",
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b",
    ]
    for pattern in number_patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[0]

    # 6. Look for direct answer patterns (improved)
    answer_patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*([^.!?]+)",
        r"(?:therefore|thus|so|hence)\s*,?\s*([^.!?]+)",
        r"(?:in\s+conclusion|to\s+summarize|in\s+summary)\s*,?\s*([^.!?]+)",
        r"(?:^|\.\s+)([A-Z][^.!?]*(?:was|is|are|were)\s+[^.!?]+)",  # Declarative statements
        r"(?:the\s+answer\s+would\s+be|it\s+(?:is|was))\s*:?\s*([^.!?]+)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = re.sub(r"^\W+|\W+$", "", answer)  # Remove leading/trailing punctuation
            if len(answer) > 3:  # Must be substantive
                return answer

    # 7. Try to find the most informative sentence (contains key words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if len(s.strip()) > 10]
    if sentences:
        # Score sentences based on information content
        def score_sentence(sentence):
            score = 0
            # Prefer sentences with specific information
            if re.search(r"\b\d{4}\b", sentence):  # Contains year
                score += 3
            if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", sentence):  # Contains proper names
                score += 2
            if re.search(r"\b(?:was|is|are|were)\b", sentence, re.IGNORECASE):  # Declarative
                score += 1
            if len(sentence.split()) < 15:  # Prefer concise answers
                score += 1
            return score

        scored_sentences = [(score_sentence(s), s) for s in sentences]
        scored_sentences.sort(reverse=True, key=lambda x: x[0])

        if scored_sentences[0][0] > 0:  # If best sentence has positive score
            return scored_sentences[0][1]

    # 8. Fallback: take first substantial sentence
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if len(s.strip()) > 5]
    if sentences:
        return sentences[0]

    # 9. Last resort: return cleaned response up to first 100 chars
    return response[:100].strip()


def _load_browsecomp_scorer():
    """Load browsecomp-plus_scorer.py despite the hyphen in its filename."""
    global _BROWSECOMP_SCORER
    if _BROWSECOMP_SCORER is not None:
        return _BROWSECOMP_SCORER

    path = Path(__file__).resolve().with_name("browsecomp-plus_scorer.py")
    spec = importlib.util.spec_from_file_location("browsecomp_plus_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load BrowseComp-Plus scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BROWSECOMP_SCORER = module
    return module


def _browsecomp_plus_reward_fn(task_info: dict, action: str) -> RewardOutput:
    scorer = _load_browsecomp_scorer()
    ground_truth = task_info.get("ground_truth") or task_info.get("answer")
    if isinstance(ground_truth, list):
        truth = str(ground_truth[0]) if ground_truth else ""
    else:
        truth = str(ground_truth or "")
    question = str(task_info.get("question") or task_info.get("query") or "")
    output = str(action or "")
    cleaned = re.sub(r"<think>.*?</think>", "", output or "", flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = output

    if re.search(r"(?:\\boxed|boxed|oxed)\s*\{", cleaned):
        extracted = extract_answer_from_response(output)
        if extracted:
            cleaned = extracted

    judge = scorer.judge_answer_from_env(
        query=question,
        ground_truth=truth,
        output=cleaned,
    )
    judge_result = judge.get("judge_result") or {}
    is_correct = bool(judge.get("is_correct", False))
    extracted = str(judge.get("extracted_answer") or "")
    confidence = judge_result.get("confidence")
    return RewardOutput(
        reward=1.0 if is_correct else 0.0,
        is_correct=is_correct,
        metadata={
            "extracted_answer": extracted,
            "ground_truths": [truth] if truth else [],
            "exact_match": is_correct,
            "f1_score": 1.0 if is_correct else 0.0,
            "confidence": confidence,
            "judge_reasoning": judge_result.get("reasoning"),
            "judge_response": judge.get("judge_response"),
            "evaluation_method": "browsecomp_plus_llm_judge",
        },
    )


REWARD_OVERRIDES: dict[str, RewardFn] = {
    "gpqa_diamond": _gpqa_diamond_reward_fn,
    "medqa": _medqa_reward_fn,
    "browsecomp_plus": _browsecomp_plus_reward_fn,
}


def build_reward_fn(
    data_source: str,
    *,
    toolcall_bonus: float = 0.0,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> RewardFn:
    """Return the reward function appropriate for ``data_source``.

    The default eval reward is a plain correctness signal (1.0 / 0.0) with no
    tool-call bonus; we only care about ``is_correct`` for aggregation, not
    shaping terms used at train time. The underlying extractor prefers
    ``\\boxed{...}`` and falls back to the raw response text.
    """
    if data_source in REWARD_OVERRIDES:
        return REWARD_OVERRIDES[data_source]

    return _make_configured_reward_fn(
        toolcall_bonus=toolcall_bonus,
        correct_reward=correct_reward,
        incorrect_reward=incorrect_reward,
    )


def build_dispatching_reward_fn(
    *,
    toolcall_bonus: float = 0.0,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.0,
) -> RewardFn:
    """Reward fn that picks the right grader from ``task_info['data_source']``.

    Used by the mixed-evals path, where a single workflow engine pools
    rollouts across benchmarks. Per-data-source reward fns are memoized so
    we still build each grader exactly once.
    """
    cache: dict[str, RewardFn] = {}

    def _resolve(data_source: str) -> RewardFn:
        fn = cache.get(data_source)
        if fn is None:
            fn = build_reward_fn(
                data_source,
                toolcall_bonus=toolcall_bonus,
                correct_reward=correct_reward,
                incorrect_reward=incorrect_reward,
            )
            cache[data_source] = fn
        return fn

    def _fn(task_info: dict, action: str) -> RewardOutput:
        data_source = str((task_info or {}).get("data_source") or "")
        return _resolve(data_source)(task_info, action)

    return _fn


__all__ = [
    "RewardFn",
    "REWARD_OVERRIDES",
    "build_reward_fn",
    "build_dispatching_reward_fn",
    "search_reward_fn",
]
