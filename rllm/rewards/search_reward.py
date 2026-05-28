import logging
import re
import string
from collections import Counter
from typing import Any, List

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput

logger = logging.getLogger(__name__)


def repetition_penalty_reward(
    text: str,
    max_n: int = 2,
    weights: List[float] = None,
) -> float:
    """
    Calculate repetition penalty reward for English text (word-level n-grams).
    Higher repetition returns more negative scores; no repetition returns 0.0.

    Args:
        text: Input English string
        max_n: Maximum n-gram length to consider (default 4, checks 1-gram through 4-gram)
        weights: Weight list for each n-gram level (default [1.0, 1.5, 2.0, 2.5])

    Returns:
        float in range [-1.0, 0.0] (0.0 = no repetition, -1.0 = extreme repetition/saturated)
        Returns -2.0 for empty text as a special error case.
    """
    # Normalize: lowercase, remove extra whitespace
    s = text.strip().lower()
    s = re.sub(r'\s+', ' ', s)

    if not s:
        return -2.0

    # Tokenize into words for English
    words = s.split()
    L = len(words)

    if L == 0:
        return -2.0

    # Default weights (if not provided by user)
    if weights is None:
        weights = [1.0 + 0.5 * i for i in range(max_n)]

    # Ensure weights length matches max_n (pad with 1.0 or truncate)
    if len(weights) < max_n:
        weights = weights + [1.0] * (max_n - len(weights))
    else:
        weights = weights[:max_n]

    raw_penalty = 0.0
    normalization = 0.0

    # Count repetitions at each n-gram level (word-level)
    for n in range(1, max_n + 1):
        if n > L:
            break

        # Extract all n-grams (word-level sliding window)
        ngrams = [tuple(words[i:i+n]) for i in range(L - n + 1)]
        counts = Counter(ngrams)

        # Sum up (count - 1) for each repeated n-gram
        # Each repetition beyond the first occurrence contributes to penalty
        occs_beyond_first = sum((cnt - 1) for cnt in counts.values() if cnt > 1)
        raw_penalty += occs_beyond_first * weights[n - 1]

        # Normalization factor: for extreme case where all windows are identical,
        # the maximum possible (count-1) sum at this n-gram level is (L - n)
        # (because there are L-n+1 windows, and (occ-1) = L-n when all are the same)
        normalization += max(0, (L - n)) * weights[n - 1]

    if raw_penalty == 0 or normalization == 0:
        return 0.0

    # Normalize and negate, capping at -1.0 (saturated penalty)
    score = -min(1.0, raw_penalty / normalization)
    return round(float(score), 2)


def length_restart_penalty(
    text: str,
    char_threshold: int = 6000,
    char_saturation: int = 20000,
    wait_threshold: int = 6,
    wait_saturation: int = 20,
) -> float:
    """P1-3: penalize the rambling / self-restart pattern seen in step-0
    evals. Scores in [-1.0, 0.0]. 0.0 means "under both thresholds".

    The score is the max of two components:
      - over-length: linear ramp from ``char_threshold`` to
        ``char_saturation`` total assistant characters.
      - restart-loop: linear ramp from ``wait_threshold`` to
        ``wait_saturation`` occurrences of "Wait," (case-insensitive).
    """
    if not text:
        return 0.0
    total_chars = len(text)
    length_score = 0.0
    if total_chars > char_threshold and char_saturation > char_threshold:
        length_score = min(1.0, (total_chars - char_threshold) / (char_saturation - char_threshold))
    wait_count = len(re.findall(r"\bWait[,.]", text, flags=re.IGNORECASE))
    wait_score = 0.0
    if wait_count > wait_threshold and wait_saturation > wait_threshold:
        wait_score = min(1.0, (wait_count - wait_threshold) / (wait_saturation - wait_threshold))
    return -round(max(length_score, wait_score), 3)


class RewardSearchFn:
    def __init__(self, config: RewardConfig):
        self.config = config

    def parse_tool_calls(self, response: str) -> tuple[int, list[str], bool]:
        """
        Parse and count valid <tool_call> ... </tool_call> patterns in the response.

        Args:
            response: The model's response text

        Returns:
            tuple: (count of valid tool calls, list of matched tool call contents, is_valid)
                  is_valid is False if tags are mismatched/invalid
        """
        # Pattern to match properly paired <tool_call> ... </tool_call> tags
        # This uses a non-greedy match to avoid matching across multiple tool calls
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)

        # Validate that we don't have unclosed or improperly nested tags
        # Count opening and closing tags separately
        opening_tags = response.count('<tool_call>')
        closing_tags = response.count('</tool_call>')

        # If tags are mismatched, something is wrong
        if opening_tags != closing_tags:
            return 0, [], False  # Invalid structure, parsing failed

        # If we have opening tags but no matches, the parsing failed
        if opening_tags > 0 and len(matches) == 0:
            return 0, [], False  # Invalid structure, parsing failed

        # Return the count, matched contents, and validity
        return len(matches), matches, True

    # P2-6: lightweight unit/date/thousands normalisation. Step-0 evals
    # showed ~50 close-miss failures across simpleqa/bamboogle/2wiki
    # where extracted_answer and GT differed only by surface form
    # (``"4990 J"`` vs ``"4990J"``, ``"1,142"`` vs ``"1142"``,
    # ``"12/03/1988"`` vs ``"03/12/1988"``, ``"120,000 euros"`` vs
    # ``"120000"``).
    _UNIT_SUFFIX_PATTERN = re.compile(
        r"^\s*(-?\d[\d,\.\s]*)\s*"
        r"(?:%|"
        r"usd|euros?|eur|gbp|dollars?|cents?|pounds?|yen|jpy|rmb|cny|"
        r"kg|kgs|g|mg|lb|lbs|oz|ton|tons|tonnes?|"
        r"km|m|cm|mm|mi|ft|in|inch|inches|yd|yard|yards|"
        r"k|m|b|bn|mn|million|millions|billion|billions|thousand|thousands|"
        r"j|kj|mj|cal|kcal|w|kw|mw|hp|"
        r"v|mv|kv|a|ma|hz|khz|mhz|ghz|"
        r"s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours|"
        r"people|persons|students|votes"
        r")\s*$",
        re.IGNORECASE,
    )
    _DATE_DMY_PATTERN = re.compile(r"^\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\s*$")
    _DATE_ISO_PATTERN = re.compile(r"^\s*(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\s*$")

    @classmethod
    def _normalize_number_token(cls, s: str) -> str | None:
        """If ``s`` looks like a plain number (possibly with thousands
        separators / decimals / a unit suffix), return a canonical form
        ``"<num>[ <unit>]"`` where ``<num>`` drops separators; else None."""
        if not s:
            return None
        txt = s.strip()
        m = cls._UNIT_SUFFIX_PATTERN.match(txt)
        if m:
            num_part = m.group(1).replace(",", "").replace(" ", "")
            try:
                float(num_part)
            except ValueError:
                return None
            return num_part
        # Pure number with thousands separators / decimals.
        if re.match(r"^-?\d[\d,\.\s]*$", txt):
            num_part = txt.replace(",", "").replace(" ", "")
            try:
                float(num_part)
            except ValueError:
                return None
            return num_part
        return None

    @classmethod
    def _date_variants(cls, s: str) -> set[str]:
        """Return canonical variants for an ambiguous DMY/MDY date
        string. Empty set if ``s`` is not date-shaped."""
        text = (s or "").strip()
        m_iso = cls._DATE_ISO_PATTERN.match(text)
        if m_iso:
            y, a, b = m_iso.group(1), int(m_iso.group(2)), int(m_iso.group(3))
            return {f"{y}-{a:02d}-{b:02d}"}
        m = cls._DATE_DMY_PATTERN.match(text)
        if not m:
            return set()
        a, b, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(y) == 2:
            y = ("20" + y) if int(y) < 50 else ("19" + y)
        variants: set[str] = set()
        if 1 <= a <= 12 and 1 <= b <= 31:
            variants.add(f"{y}-{a:02d}-{b:02d}")
        if 1 <= b <= 12 and 1 <= a <= 31:
            variants.add(f"{y}-{b:02d}-{a:02d}")
        return variants

    def normalize_answer(self, s: str) -> str:
        """Normalize answer text for evaluation (following HotpotQA/SQuAD standards)"""

        # P2-6: strip thousand separators inside numbers before the
        # punctuation pass so "1,142" and "1142" collapse.
        num_canon = self._normalize_number_token(s)
        if num_canon is not None:
            return num_canon

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction: str, ground_truth: str) -> tuple[float, float, float]:
        """Calculate F1 score between prediction and ground truth"""
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0, 0, 0)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """Calculate exact match score"""
        if self.normalize_answer(prediction) == self.normalize_answer(ground_truth):
            return True
        # P2-6: date DMY/MDY ambiguity — accept matching canonical variants.
        pv = self._date_variants(str(prediction).strip())
        gv = self._date_variants(str(ground_truth).strip())
        if pv and gv and pv & gv:
            return True
        return False

    def _unwrap_json_fragment(self, text: str) -> str:
        """Unwrap JSON tool-call fragments that wrap the actual answer.

        Handles cases like:
          {"command": "submit", "result": "No"} -> "No"
          {"name": "finish", "arguments": {"result": "Paris"}} -> "Paris"
        """
        text = text.strip()
        if not (text.startswith("{") and text.endswith("}")):
            return text
        try:
            import json
            obj = json.loads(text)
            if isinstance(obj, dict):
                # {"command": "submit", "result": <answer>}
                if "result" in obj:
                    val = obj["result"]
                    return str(val) if not isinstance(val, str) else val
                # {"name": "finish", "arguments": {"result": <answer>}}
                args = obj.get("arguments", {})
                if isinstance(args, dict) and "result" in args:
                    val = args["result"]
                    return str(val) if not isinstance(val, str) else val
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return text

    _OPTION_LETTERS = {"A", "B", "C", "D", "E", "F"}

    @classmethod
    def _parse_letter_set(cls, s: str) -> set[str] | None:
        """Parse a string as a set of option letters (e.g. 'ABC', 'A,B,D',
        'A、B', 'A and C'). Returns ``None`` if the string doesn't look
        like a pure letter set. P1-4: scienceqa multi-letter MCQ GTs
        (``"ABC"``/``"CD"``) were universally scored 0 at step 0 because
        the extractor only handled single letters.
        """
        if not s:
            return None
        text = str(s)
        # First drop connector words that overlap option letters if used
        # as char-class entries (``and`` contains ``a``/``n``/``d``).
        text = re.sub(r"(?i)\b(and|以及|和|与|或)\b", " ", text)
        # Now strip remaining delimiters and whitespace.
        cleaned = re.sub(r"[\s,，、;；·.()\[\]{}\\/&+]+", "", text)
        cleaned = cleaned.strip().upper()
        if not cleaned:
            return None
        letters = set(cleaned)
        if not letters or any(ch not in cls._OPTION_LETTERS for ch in letters):
            return None
        # Reject strings with duplicated letters (e.g. "AA") — unlikely
        # to be a real MCQ answer and a hint the parse is off.
        if len(letters) != len(cleaned):
            return None
        return letters

    def _map_value_to_option_letter(self, extracted: str, ground_truths: list[str]) -> str:
        """For multiple-choice questions where ground truth is A/B/C/D,
        if the model output a raw value instead of a letter, try to map it back.

        This is a no-op if ground truths are not single option letters.
        Also normalises multi-letter MCQ answers (``"A,C"`` -> ``"AC"``,
        order-insensitive) when the GT is a multi-letter set.
        """
        if not ground_truths:
            return extracted

        # Multi-letter path: both sides parse as letter sets -> canonical sorted form.
        gt_letter_sets = [self._parse_letter_set(gt) for gt in ground_truths]
        if all(s and len(s) >= 1 for s in gt_letter_sets) and any(len(s) >= 2 for s in gt_letter_sets):
            extracted_letters = self._parse_letter_set(extracted)
            if extracted_letters:
                return "".join(sorted(extracted_letters))
            return extracted

        # Single-letter path (original behaviour).
        if not all(gt.strip().upper() in self._OPTION_LETTERS for gt in ground_truths):
            return extracted
        if extracted.strip().upper() in self._OPTION_LETTERS:
            return extracted.strip().upper()
        return extracted

    def _map_letter_to_option_value(self, extracted: str, question: str) -> str:
        """Reverse of _map_value_to_option_letter.

        Used when the GT is prose but the model emitted a bare MCQ letter
        ("C"). Scans the question for a line matching ``^C[.)] (.+)`` and
        returns the prose. 175/300 medqa eval rollouts in the training
        dump fell into this gap — the reasoning was correct but the
        surface form was a letter while the verifier compared against a
        noun phrase.
        """
        if not extracted or not question:
            return extracted
        cand = extracted.strip().upper()
        if not (len(cand) == 1 and cand in "ABCDEF"):
            return extracted
        # Find the option line in the question body.
        pattern = rf"(?mi)^\s*{re.escape(cand)}\s*[.)\:\-]\s*(.+?)\s*$"
        m = re.search(pattern, question)
        if m:
            return m.group(1).strip()
        return extracted

    _LATEX_TEXT_WRAPPERS = (
        r"\text",
        r"\textbf",
        r"\textit",
        r"\texttt",
        r"\textrm",
        r"\textsf",
        r"\mathrm",
        r"\mathbf",
        r"\mathit",
        r"\mathtt",
        r"\mathsf",
        r"\operatorname",
        r"\emph",
        r"\underline",
    )

    @classmethod
    def _strip_latex_wrappers(cls, s: str) -> str:
        """Peel LaTeX text-shape wrappers off an already-unboxed answer.

        Evidence from `evals_trajectory/global_steps_10.json`: 24 zero-reward
        cases had `\\boxed{\\text{no}}`, `\\boxed{\\textbf{Paris}}`, or
        `\\boxed{\\$42\\$}` as the model's final answer. Unboxing alone
        leaves the text-command wrapper intact, so F1 against a bare
        ground-truth token (``"no"``, ``"Paris"``) scores 0. We repeatedly
        strip the longest recognized wrapper (e.g. ``\\textbf`` before
        ``\\text``) and then clean up dangling ``$``/``\\$`` currency markers
        and outer whitespace. Only text-shape commands are peeled; math
        operators (``\\frac``, ``\\sqrt``) are left intact.
        """
        if not s:
            return s
        out = s.strip()
        # Loop in case wrappers are nested, e.g. \textbf{\text{no}}.
        # Longest-first keeps \textbf from being mis-matched as \text.
        wrappers = sorted(cls._LATEX_TEXT_WRAPPERS, key=len, reverse=True)
        for _ in range(8):  # depth cap
            changed = False
            for w in wrappers:
                if not out.startswith(w):
                    continue
                tail = out[len(w):].lstrip()
                if not tail.startswith("{"):
                    continue
                depth = 1
                j = 1
                while depth and j < len(tail):
                    if tail[j] == "{":
                        depth += 1
                    elif tail[j] == "}":
                        depth -= 1
                    j += 1
                if depth:
                    continue  # unbalanced — leave alone
                inner = tail[1 : j - 1]
                trailing = tail[j:].strip()
                # Only peel if the wrapper spans the whole answer; otherwise
                # we risk corrupting a multi-token payload.
                if trailing:
                    continue
                out = inner.strip()
                changed = True
                break
            if not changed:
                break
        # Strip dangling $...$ math mode and escaped currency.
        if out.startswith("$") and out.endswith("$") and len(out) >= 2:
            out = out[1:-1].strip()
        out = out.replace("\\$", "$").replace("\\%", "%")
        out = out.replace("\\,", " ").replace("\\;", " ").replace("\\:", " ")
        return out.strip()

    def extract_answer_from_response(self, response: str) -> str:
        response = response.strip()

        # Remove thinking tags first
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"\s+", " ", response).strip()

        if not response:
            return ""

        # 0. Unwrap JSON tool-call fragments before any other extraction
        response = self._unwrap_json_fragment(response)
        if not response:
            return ""

        # 1. HIGHEST PRIORITY: Look for \boxed{} or \boxed[] content
        def unbox(s: str) -> str | None:
            """Extract content from \\boxed{...} with proper nesting support.

            Handles three leak modes observed in the eval trajectories:
              * ``\\boxed{C}`` — intended form (the ``\\b`` is a backslash+b,
                not the Python \\x08 backspace, because the source string is
                what the model literally typed).
              * ``boxed{C}``  — the model emitted ``boxed`` without any
                leading backslash.
              * ``oxed{C}``   — an upstream string-literal bug stripped the
                leading ``\\b`` (Python parser treats ``\\boxed`` inside a
                double-quoted string as ``\\x08oxed``; the fragment appeared
                in 4 eval rollouts across gpqa/musique/medqa and silently
                truncated the extracted answer).
            """
            if not s:
                return None
            # Scan for any of the three anchors; take the earliest.
            anchors = []
            for tok in ("\\boxed{", "boxed{", "oxed{", "\x08oxed{"):
                i = s.find(tok)
                if i >= 0:
                    anchors.append((i, len(tok)))
            if not anchors:
                return None
            anchors.sort()
            start, tok_len = anchors[0]
            i = start + tok_len
            depth = 1
            j = i
            while depth and j < len(s):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            if depth:
                return None  # unbalanced braces
            return s[i : j - 1]

        boxed_content = unbox(response)

        if boxed_content is not None:
            return self._strip_latex_wrappers(boxed_content.strip())

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

    def evaluate_answer(
        self,
        model_answer: str,
        ground_truth: str | list[str],
        question: str | None = None,
        data_source: str | None = None,
    ) -> tuple[bool, float, dict[str, Any]]:
        extracted_answer = self.extract_answer_from_response(model_answer)

        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        # For multiple-choice (GPQA-style): try mapping raw values to option letters
        extracted_answer = self._map_value_to_option_letter(extracted_answer, ground_truths)

        # P1-4: canonicalise multi-letter MCQ ground truths so the F1/EM
        # path sees the same order/form as the extracted answer.
        canon_gts: list[str] = []
        for gt in ground_truths:
            ls = self._parse_letter_set(gt)
            if ls and len(ls) >= 2:
                canon_gts.append("".join(sorted(ls)))
            else:
                canon_gts.append(gt)
        ground_truths = canon_gts

        # Reverse mapping for MCQ-shaped datasets whose GT is prose (medqa).
        # Fix #1: 175/300 medqa rollouts output a bare letter while GT is a
        # noun phrase; pure f1 scoring of "C" vs "Colorectal cancer" is 0.
        mcq_prose_datasets = {"medqa"}
        if data_source in mcq_prose_datasets and question:
            mapped = self._map_letter_to_option_value(extracted_answer, question)
            if mapped != extracted_answer:
                extracted_answer = mapped

        max_f1 = 0.0
        max_em = False
        best_match = ""
        best_precision = 0.0
        best_recall = 0.0

        metadata: dict[str, Any] = {"extracted_answer": extracted_answer, "ground_truths": ground_truths, "evaluation_method": None}

        for gt in ground_truths:
            gt_str = str(gt).strip()

            # Calculate exact match
            em = self.exact_match_score(extracted_answer, gt_str)
            if em:
                max_em = True
                max_f1 = 1.0  # Perfect F1 for exact match
                best_match = gt_str
                best_precision = 1.0
                best_recall = 1.0
                metadata["evaluation_method"] = "exact_match"
                break

            # Calculate F1 score
            f1, precision, recall = self.f1_score(extracted_answer, gt_str)
            if f1 > max_f1:
                max_f1 = f1
                best_match = gt_str
                best_precision = precision
                best_recall = recall
                metadata["evaluation_method"] = "f1_score"

        # Determine if answer is "correct" based on threshold
        # Use lower threshold for F1 score (0.3) as it's more lenient than exact match
        f1_threshold = 0.3
        is_correct = max_em or max_f1 >= f1_threshold

        metadata.update({"best_match": best_match, "f1_score": max_f1, "precision": best_precision, "recall": best_recall, "exact_match": max_em, "f1_threshold": f1_threshold})

        return is_correct, max_f1, metadata

    def __call__(self, input: RewardInput) -> RewardOutput:
        # Extract information from task_info and action
        model_response = input.action
        ground_truth = input.task_info.get("ground_truth") or input.task_info.get("answer")

        if ground_truth is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False, metadata={"error": "No ground truth provided"})

        # Parse tool calls from the response
        # tool_call_count, tool_call_contents, is_valid_parsing = self.parse_tool_calls(model_response)

        is_correct, score, metadata = self.evaluate_answer(
            model_response,
            ground_truth,
            question=input.task_info.get("question"),
            data_source=input.task_info.get("data_source"),
        )

        if is_correct:
            # For exact matches, give full reward
            # For F1 matches, scale reward by F1 score
            if metadata.get("exact_match", False):
                reward = self.config.correct_reward
            else:
                # Scale reward by F1 score for partial matches
                reward = self.config.correct_reward * score
        else:
            # Fix #10: keep a continuous near-miss credit below the
            # f1_threshold instead of the binary cliff. The eval dump at
            # step-10 had 20+ zero-reward cases with f1 ∈ (0, 0.3) whose
            # gradient signal was being discarded entirely. We award half
            # the scaled reward in that band, still dominated by any
            # threshold-passing rollout, so ranking order is preserved.
            if score > 0.0:
                reward = self.config.correct_reward * score * 0.5
            else:
                reward = self.config.incorrect_reward

        # Apply step-based bonus for correct answers
        step_bonus = 0.0
        if self.config.enable_step_bonus and is_correct and reward > 0:
            step_count = input.task_info.get("step_count", 0)
            if step_count >= self.config.min_steps_for_bonus:
                # Calculate bonus scaling factor
                # Linear scaling from min_steps to max_steps
                steps_above_min = step_count - self.config.min_steps_for_bonus
                steps_range = self.config.max_steps_for_bonus - self.config.min_steps_for_bonus

                if steps_range > 0:
                    # Normalize to [0, 1] range, capped at 1.0
                    bonus_factor = min(1.0, steps_above_min / steps_range)
                    # Apply bonus rate
                    step_bonus = reward * self.config.step_bonus_rate * bonus_factor
                else:
                    # If min and max are the same, give full bonus if qualified
                    step_bonus = reward * self.config.step_bonus_rate

                metadata["step_bonus_factor"] = bonus_factor if steps_range > 0 else 1.0
                metadata["step_count"] = step_count

        """
        # Apply tool call bonus/penalty based on new strategy:
        # 1. Invalid tags -> -0.5 penalty
        # 2. Single tool call + reward > 0 -> +0.5 bonus (only if answer is correct/partially correct)
        # 3. Multiple tool calls (>= 2) -> -0.5 penalty (regardless of correctness)
        # 4. No tool call -> 0 adjustment
        tool_call_adjustment = 0.0
        if not is_valid_parsing:
            # Invalid/unparseable tool call tags - penalize
            tool_call_adjustment = -self.config.toolcall_bonus
            metadata["tool_call_status"] = "invalid_tags_penalty"
        elif tool_call_count >= 2:
            # Multiple tool calls - always penalize to suppress repeated calls
            tool_call_adjustment = -self.config.toolcall_bonus
            metadata["tool_call_status"] = "multiple_calls_penalty"
        elif tool_call_count == 1 and reward > 0:
            # Invalid tool call: single tool call but generating answer with positive reward - no bonus
            tool_call_adjustment = 0.0
            metadata["tool_call_status"] = "single_call_no_bonus"
        elif tool_call_count == 1 and reward <= 0:
            # Single tool call with positive reward - give bonus
            # This encourages tool use only when it leads to correct/partially correct answers
            tool_call_adjustment = self.config.toolcall_bonus
            metadata["tool_call_status"] = "single_call_bonus"
        else:
            # No tool call detected
            tool_call_adjustment = 0.0
            metadata["tool_call_status"] = "no_tool_call"
        """

        # Store base reward before adjustments
        base_reward = reward

        """
        if self.config.toolcall_bonus > 0.0:
            reward += tool_call_adjustment
        """

        # Fix #8: activate repetition penalty on the web-search path.
        # The eval dump at step-10 contained one musique rollout with a
        # 488k-char assistant turn (``"Jennifer" × thousands``) that
        # terminated normally with reward 0 — no penalty fired because this
        # block was commented out, leaving `repetition_penalty_reward`
        # identically 0 for all 1250 rollouts. Opt-in via
        # RewardConfig.apply_repetition_penalty so MCP paths are unaffected.
        repetition_penalty = 0.0
        if self.config.apply_repetition_penalty:
            try:
                repetition_penalty = repetition_penalty_reward(
                    model_response,
                    max_n=self.config.repetition_max_n,
                )
            except Exception as _e:
                logger.debug("repetition_penalty_reward failed: %s", _e)
                repetition_penalty = 0.0
            repetition_penalty_weighted = repetition_penalty * self.config.repetition_penalty_weight
            reward += repetition_penalty_weighted
        else:
            repetition_penalty_weighted = 0.0

        # P1-3: length / self-restart penalty. Step-0 evals showed
        # 53-99% of rollouts with single-message >8k chars and heavy
        # "Wait," restart cycles. This signal is orthogonal to the
        # n-gram repetition penalty above and catches long "rambling
        # but not literally repeating" traces.
        length_penalty_raw = 0.0
        length_penalty_weighted = 0.0
        if getattr(self.config, "apply_length_penalty", False):
            try:
                length_penalty_raw = length_restart_penalty(
                    model_response,
                    char_threshold=self.config.length_penalty_char_threshold,
                    char_saturation=self.config.length_penalty_char_saturation,
                    wait_threshold=self.config.length_penalty_wait_threshold,
                    wait_saturation=self.config.length_penalty_wait_saturation,
                )
            except Exception as _e:
                logger.debug("length_restart_penalty failed: %s", _e)
                length_penalty_raw = 0.0
            length_penalty_weighted = length_penalty_raw * self.config.length_penalty_weight
            reward += length_penalty_weighted

        # Add tool call information and other reward components to metadata
        metadata.update({
            "base_reward": reward,
            "tool_call_reward": 0,
            "repetition_penalty_reward": repetition_penalty_weighted,
            "length_penalty_reward": length_penalty_weighted,
            "step_bonus": step_bonus,
        })

        # Apply step bonus to final reward
        reward += step_bonus

        # Upper-clamp the web-search reward to correct_reward (1.0 by default).
        # Step bonus + partial-match scaling could otherwise push the total
        # above the nominal ceiling, which skews training/eval aggregates.
        if reward > self.config.correct_reward:
            reward = self.config.correct_reward

        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)
