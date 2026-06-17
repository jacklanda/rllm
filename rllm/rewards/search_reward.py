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
    s = re.sub(r"\s+", " ", s)

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
        ngrams = [tuple(words[i : i + n]) for i in range(L - n + 1)]
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
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)

        # Validate that we don't have unclosed or improperly nested tags
        # Count opening and closing tags separately
        opening_tags = response.count("<tool_call>")
        closing_tags = response.count("</tool_call>")

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

    _ACRONYM_TOKEN_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.+-]{1,12}$")
    _ROMAN_NUMERAL_PATTERN = re.compile(r"(?i)^[ivxlcdm]+$")
    _CHEMICAL_LOCANT_PATTERN = re.compile(r"(?i)^[a-z]?\d+(?:[.,;:/_-]?[a-z]?\d+)*[a-z]?$")

    def _is_safe_parenthetical_alias(self, content: str, outer: str) -> bool:
        """Accept only high-signal parenthetical aliases.

        Parentheses in web answers often hold disambiguating chemistry,
        chromosomal loci, monomer composition, or roman numerals. Treating every
        parenthetical fragment as an alias caused false positives such as
        ``III`` matching unrelated metal complexes and ``P`` matching different
        copolymers. Keep conventional acronyms like ``RPV``/``ISI``/``FEA``.
        """
        raw = str(content or "").strip()
        if not raw:
            return False
        if len(raw) == 1:
            return False
        if self._ROMAN_NUMERAL_PATTERN.fullmatch(raw):
            return False
        if self._CHEMICAL_LOCANT_PATTERN.fullmatch(raw):
            return False
        if re.search(r"\d", raw) and not re.fullmatch(r"(?i)[A-Z]{2,}\d{0,3}", raw):
            return False
        if not self._ACRONYM_TOKEN_PATTERN.fullmatch(raw):
            return False
        letters = re.sub(r"[^A-Za-z]", "", raw).lower()
        outer_words = re.findall(r"[A-Za-z]+", outer)
        if not letters or not outer_words:
            return False
        initials = "".join(word[0].lower() for word in outer_words if word)
        outer_compact = "".join(outer_words).lower()
        return letters in initials or letters in outer_compact or len(letters) <= 5

    def _answer_aliases(self, s: str) -> set[str]:
        """Return conservative normalized aliases for entity-style answers.

        This recovers common web-search false negatives such as full name vs
        parenthetical acronym (``Rilpivirine (RPV)``), hyphenation variants
        (``Inter-stimulus`` vs ``interstimulus``), and version prefixes
        (``v2.3.9`` vs ``2.3.9``) without accepting arbitrary substrings like
        ``Latin`` for ``Medieval Latin``.
        """
        raw = str(s or "").strip()
        if not raw:
            return set()

        aliases = {self.normalize_answer(raw)}

        paren_contents = [m.strip() for m in re.findall(r"\(([^()]+)\)", raw) if m.strip()]
        without_parens = re.sub(r"\s*\([^()]*\)", "", raw).strip()
        if without_parens:
            aliases.add(self.normalize_answer(without_parens))
        for content in paren_contents:
            if self._is_safe_parenthetical_alias(content, without_parens or raw):
                aliases.add(self.normalize_answer(content))

        compact_source = {raw, without_parens}
        compact_source.update(content for content in paren_contents if self._is_safe_parenthetical_alias(content, without_parens or raw))
        for value in compact_source:
            if not value:
                continue
            normalized = self.normalize_answer(value)
            if normalized:
                aliases.add(re.sub(r"\s+", "", normalized))

        version = re.fullmatch(r"(?i)\s*v(?:ersion)?\s*([0-9][0-9A-Za-z.\-_]*)\s*", raw)
        if version:
            aliases.add(self.normalize_answer(version.group(1)))
        aliases.add(re.sub(r"(?i)\bv\s+(?=\d)", "", self.normalize_answer(raw)))
        aliases.add(self.normalize_answer(re.sub(r"(?i)\bv(?=\d)", "", raw)))

        generic_suffix_alias = self._generic_entity_suffix_alias(raw)
        if generic_suffix_alias:
            aliases.add(generic_suffix_alias)

        return {alias for alias in aliases if alias}

    def _parenthetical_acronym_aliases(self, s: str) -> set[str]:
        raw = str(s or "").strip()
        without_parens = re.sub(r"\s*\([^()]*\)", "", raw).strip()
        aliases = set()
        for content in [m.strip() for m in re.findall(r"\(([^()]+)\)", raw) if m.strip()]:
            if self._is_safe_parenthetical_alias(content, without_parens or raw):
                normalized = self.normalize_answer(content)
                if normalized:
                    aliases.add(normalized)
        return aliases

    def _subject_tokens_for_alias_guard(self, s: str) -> set[str]:
        without_parens = re.sub(r"\s*\([^()]*\)", "", str(s or "")).strip()
        tokens = set(self.normalize_answer(without_parens).split())
        return {
            token
            for token in tokens
            if len(token) >= 3 and token not in self._GENERIC_ENTITY_SUFFIXES and token not in self._GENERIC_TECH_SUFFIX_TOKENS
        }

    def _aliases_match_safely(self, prediction: str, ground_truth: str) -> bool:
        pred_aliases = self._answer_aliases(prediction)
        gt_aliases = self._answer_aliases(ground_truth)
        shared = pred_aliases & gt_aliases
        if not shared:
            return False

        # Shared parenthetical acronyms are safe when one side is the acronym
        # itself. When both sides are long entities, require compatible subject
        # tokens so generic technical acronyms like IPN do not erase different
        # material names.
        pred_acronyms = self._parenthetical_acronym_aliases(prediction)
        gt_acronyms = self._parenthetical_acronym_aliases(ground_truth)
        acronym_only = shared <= (pred_acronyms | gt_acronyms)
        if not acronym_only:
            return True

        pred_norm = self.normalize_answer(prediction)
        gt_norm = self.normalize_answer(ground_truth)
        if pred_norm in shared or gt_norm in shared:
            return True

        pred_subject = self._subject_tokens_for_alias_guard(prediction)
        gt_subject = self._subject_tokens_for_alias_guard(ground_truth)
        if not pred_subject or not gt_subject:
            return True
        return bool(pred_subject & gt_subject)

    def _person_name_aliases(self, s: str) -> set[str]:
        raw = str(s or "").strip()
        if not raw:
            return set()
        cleaned = re.sub(r"[^A-Za-z,\s.'-]", " ", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        aliases = set()
        has_comma = "," in cleaned

        comma = re.fullmatch(r"([A-Za-z][A-Za-z'.-]+),\s*([A-Za-z])\.?", cleaned)
        if comma:
            last, initial = comma.groups()
            aliases.add(f"{initial.lower()} {last.lower()}")
            aliases.add(f"{last.lower()} {initial.lower()}")

        tokens = [tok.strip(".") for tok in re.split(r"\s+", cleaned.replace(",", " ")) if tok.strip(".")]
        if any(tok.lower() in {"of", "de", "del", "la", "le", "van", "von"} for tok in tokens):
            return aliases
        suffixes = {"jr", "sr", "ii", "iii", "iv"}
        tokens = [tok for tok in tokens if tok.lower() not in suffixes]
        has_initial = any(len(tok) == 1 for tok in tokens)
        if (has_comma or has_initial or len(tokens) == 2) and len(tokens) >= 2 and all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", tok) for tok in tokens):
            first = tokens[0]
            last = tokens[-1]
            aliases.add(f"{first.lower()} {last.lower()}")
            if has_comma or has_initial:
                aliases.add(f"{first[0].lower()} {last.lower()}")
                aliases.add(f"{last.lower()} {first[0].lower()}")
        return aliases

    def _critical_surface_mismatch(self, prediction: str, ground_truth: str) -> bool:
        """Reject matches that only work after erasing critical symbols/digits."""
        pred_raw = str(prediction or "").strip()
        gt_raw = str(ground_truth or "").strip()
        if not pred_raw or not gt_raw:
            return False

        pv = self._date_variants(pred_raw)
        gv = self._date_variants(gt_raw)
        if pv and gv:
            return not bool(pv & gv)

        def _canonical_digit_chunks(text: str) -> list[str]:
            chunks = re.findall(r"(?i)[a-z]*\d[a-z0-9.+:/;_-]*", text)
            return [re.sub(r"(?i)^v(?=\d)", "", chunk).lower() for chunk in chunks]

        pred_digit_chunks = _canonical_digit_chunks(pred_raw)
        gt_digit_chunks = _canonical_digit_chunks(gt_raw)
        if pred_digit_chunks or gt_digit_chunks:
            if pred_digit_chunks != gt_digit_chunks:
                return True

        critical_symbols = {"+", "#"}
        pred_symbols = {ch for ch in pred_raw if ch in critical_symbols}
        gt_symbols = {ch for ch in gt_raw if ch in critical_symbols}
        if pred_symbols != gt_symbols:
            return True

        return False

    _GENERIC_ENTITY_SUFFIXES = {"project", "portal", "report", "standard", "province", "model"}
    _GENERIC_TECH_SUFFIX_TOKENS = {
        "analysis",
        "based",
        "framework",
        "hydrogel",
        "imaging",
        "interpenetrating",
        "method",
        "microscopy",
        "model",
        "network",
        "polymer",
        "technique",
    }

    def _generic_entity_suffix_alias(self, s: str) -> str:
        """Drop generic entity words only when a stable core remains.

        This intentionally removes only low-information wrappers that often
        appear in gold labels ("Documenting Hate project") while avoiding broad
        substring acceptance ("Latin" must not match "Medieval Latin").
        """
        tokens = self.normalize_answer(s).split()
        core = [tok for tok in tokens if tok not in self._GENERIC_ENTITY_SUFFIXES]
        if len(core) < 2 or core == tokens:
            return ""
        return " ".join(core)

    def _coerce_submitted_answer_text(self, s: str) -> str:
        """Normalize harmless escaping in submitted final-answer strings."""
        text = str(s or "")
        text = re.sub(r"\\+(?=\s*[A-Za-z0-9])", " ", text)
        return re.sub(r"\s+", " ", text).strip()

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

    _MONTH_OR_DATE_PATTERN = re.compile(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\b|"
        r"\b\d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?\b|"
        r"\b\d{4}\b|"
        r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _has_temporal_marker(cls, s: str) -> bool:
        return bool(cls._MONTH_OR_DATE_PATTERN.search(str(s or "")))

    def _partial_match_acceptance(self, prediction: str, ground_truth: str, precision: float, recall: float) -> tuple[bool, str]:
        """Gate token-overlap F1 so entity/date fragments are not counted correct.

        The reward can still use the continuous F1 score as shaping signal, but
        `is_correct` should not become true for answers that only share a
        modifier, a year, or a few entity tokens with the gold answer.
        """
        pred_norm = self.normalize_answer(prediction)
        gt_norm = self.normalize_answer(ground_truth)
        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()

        if not pred_tokens or not gt_tokens:
            return False, "empty_normalized_answer"

        if pred_norm in {"yes", "no", "noanswer"} or gt_norm in {"yes", "no", "noanswer"}:
            return False, "binary_mismatch"

        if Counter(pred_tokens) == Counter(gt_tokens):
            return True, "same_token_multiset"

        if self._has_temporal_marker(prediction) or self._has_temporal_marker(ground_truth):
            return False, "temporal_granularity_mismatch"

        pred_set = set(pred_tokens)
        gt_set = set(gt_tokens)
        if pred_set < gt_set or gt_set < pred_set:
            return False, "substring_entity_mismatch"

        # Short, mostly named-entity answers are exactly where token F1 creates
        # false positives such as "Superior mesenteric artery" vs "Inferior
        # mesenteric artery" or "Jean I ..." vs "Francois II ...".
        if len(pred_tokens) <= 6 and len(gt_tokens) <= 6:
            return False, "short_entity_token_overlap"

        if precision < 0.8 or recall < 0.8:
            return False, "low_precision_or_recall"

        return True, "high_overlap_non_entity"

    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """Calculate exact match score"""
        pred_raw = str(prediction).strip()
        gt_raw = str(ground_truth).strip()
        pred_letter = pred_raw.upper()
        gt_letter = gt_raw.upper()
        if pred_letter in self._OPTION_LETTERS or gt_letter in self._OPTION_LETTERS:
            return pred_letter == gt_letter
        if self._critical_surface_mismatch(prediction, ground_truth):
            return False
        if self.normalize_answer(prediction) == self.normalize_answer(ground_truth):
            return True
        if self._aliases_match_safely(prediction, ground_truth):
            return True
        if self._person_name_aliases(prediction) & self._person_name_aliases(ground_truth):
            return True
        # P2-6: date DMY/MDY ambiguity — accept matching canonical variants.
        pv = self._date_variants(pred_raw)
        gv = self._date_variants(gt_raw)
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

    # Explicit final-answer commitment patterns for free-text (COT) MCQ turns.
    #
    # Qwen-style COT eval rollouts frequently state a clear final choice ("the
    # answer is C", "C is the correct answer", "I'll go with C") and then keep
    # second-guessing until the per-step token cap truncates the turn. The
    # generic prose-scavenging cascade then mis-extracts an early **bold**
    # heading ("**Conditions:**") or an earlier hypothetical letter, producing
    # a false negative on an answer the model actually committed to. These
    # high-precision patterns scan the whole turn and the caller takes the LAST
    # surviving match so the model's final commitment wins.
    #
    # Two pattern families with different case sensitivity:
    #   * answer-anchored (the word "answer"/"option"/"choice" disambiguates):
    #     case-insensitive letter, so "the answer is c" is honoured.
    #   * letter-first ("C is the correct answer"): UPPERCASE letter only, so a
    #     lowercase problem item like "(c) is the densest" (referring to star c,
    #     not option C) or a compound label ("A = Benzoquinone") is not mistaken
    #     for an option commitment.
    _MCQ_COMMIT_PATTERNS = [
        re.compile(p)
        for p in (
            r"(?i)(?:the\s+)?(?:correct\s+|final\s+|right\s+)?answer\s+(?:is|would\s+be|should\s+be|must\s+be|=|:)\s*\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])",
            r"(?i)\bfinal\s+answer\s*(?:is|:|=)\s*\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])",
            r"(?:[Oo]ption|[Cc]hoice)\s+\(?([A-F])\)?\s+(?i:is\s+(?:the\s+)?(?:correct|right|final|best|answer))\b",
            r"(?<![A-Za-z0-9])\(?\*{0,2}([A-F])\*{0,2}\)?\s+(?i:is\s+(?:the\s+)?(?:correct|right|final|best)\s+(?:answer|choice|option))\b",
            r"(?<![A-Za-z0-9])\(?\*{0,2}([A-F])\*{0,2}\)?\s+(?i:is\s+(?:the\s+)?correct)(?:\s+(?i:answer|choice|option))?(?![A-Za-z0-9])",
            r"(?i)\bI(?:'?ll| will| am going to| have decided to| shall)?\s+(?:go\s+with|choose|select|pick|finalize|finalise|settle\s+on|decide\s+on|conclude\s+with)\s+(?:option\s+|choice\s+|with\s+)?\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])",
            r"(?i)\bmy\s+(?:final\s+)?(?:answer|choice)\s+is\s*\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])",
            r"(?i)\bgoing\s+with\s+(?:option\s+|choice\s+)?\(?([A-Fa-f])\)?(?![A-Za-z0-9])",
            r"(?i)\b(?:most\s+likely|likely|intended|probable|best|safe)\s+(?:answer|choice|option)\s*(?:is|would\s+be|:|=)?\s*\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])",
            r"(?i)\b(?:answer|choice|option)\s+\(?\*{0,2}([A-Fa-f])\*{0,2}\)?\s+(?:is\s+)?(?:the\s+)?(?:most\s+likely|likely|best|safe|intended|probable)\s+(?:answer|choice|option|bet)\b",
            r"(?i)\b(?:this|that|it)\s+(?:matches|fits|corresponds\s+to|points\s+to)\s+(?:option|choice|answer)\s+\(?\*{0,2}([A-Fa-f])\*{0,2}\)?(?![A-Za-z0-9])",
        )
    ]
    # Tentative/hypothetical context preceding a match -> not a real commitment
    # ("if the answer is A", "let's assume the answer is D", "is it possible C").
    # Prefix terms (assum*, possibl*) intentionally omit a trailing \b so
    # "assuming"/"possible" are matched.
    _MCQ_TENTATIVE_PREFIX = re.compile(
        r"(?i)(?:\bif\b|\bassum|\bsuppose|\bunless|\bwere\b|\bmaybe\b|\bmight\b|\bperhaps\b|\bpossibl|\bchance\b|\bwhat\s+if\b|\bis\s+it\s+possible|\bany\s+scenario\s+where\b|\bscenario\s+where\b|\bin\s+case\b|\bcould\s+be\b|\bguess\b|\bnot\s+sure\b|\bunsure\b|\blet'?s\s+say|\beither\b)"
    )
    # Enumeration of options ("A or C", "B, C, or D is correct") -> a listing,
    # not a single commitment. Case-sensitive: option letters are uppercase.
    _MCQ_ENUM_FORWARD = re.compile(r"\s*(?:[Oo]r|/|,|[Aa]nd|[Nn]or)\s+[A-F]\b")
    _MCQ_ENUM_BACKWARD = re.compile(r"[A-F]\s*[,/]\s*(?:[Oo]r\s+|[Aa]nd\s+)?$|[A-F]\s+(?:[Oo]r|[Aa]nd|[Nn]or)\s+$")
    _MCQ_NEGATIVE_FOLLOW = re.compile(r"(?i)^\s*(?:is\s+)?(?:likely\s+)?(?:not|incorrect|wrong|unlikely|impossible|inconsistent|ruled\s+out|fails?|does\s+not)\b")

    @classmethod
    def _parse_option_label(cls, s: str) -> str | None:
        text = str(s or "").strip()
        if not text:
            return None
        m = re.match(r"(?is)^(?:final\s+)?(?:answer\s*(?:is)?\s*:?\s*)?(?:option|choice)?\s*([A-F])\s*(?:[.)\]:、-]|\b)", text)
        if m:
            return m.group(1).upper()
        return None

    def _parse_question_options(self, question: str | None) -> dict[str, str]:
        if not question:
            return {}
        options: dict[str, str] = {}
        pattern = re.compile(r"(?ms)(?:^|\n)\s*([A-F])\s*[.)]\s*(.+?)(?=(?:\n\s*[A-F]\s*[.)]\s*)|\Z)")
        for letter, value in pattern.findall(question):
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                options[letter.upper()] = value
        return options

    def _map_option_value_to_letter(self, extracted: str, question: str | None) -> str:
        options = self._parse_question_options(question)
        if not options:
            return extracted

        text = str(extracted or "").strip()
        norm_text = self.normalize_answer(text)
        if not norm_text:
            return extracted
        if len(text) == 1 and text.islower():
            for letter, value in options.items():
                if norm_text == self.normalize_answer(value):
                    return letter

        label = self._parse_option_label(text)
        if label:
            return label

        # A lowercase one-character answer may be the option value in GPQA
        # prompts like "A. d / B. a / C. b / D. c". Uppercase one-character
        # answers are treated as option labels.
        if len(text) == 1 and text.upper() in self._OPTION_LETTERS and text.isupper():
            return text.upper()

        for letter, value in options.items():
            if norm_text == self.normalize_answer(value):
                return letter

        text_tokens = set(norm_text.split())
        if len(text_tokens) >= 2:
            subset_matches = []
            for letter, value in options.items():
                value_tokens = set(self.normalize_answer(value).split())
                if text_tokens and text_tokens <= value_tokens:
                    subset_matches.append(letter)
            if len(subset_matches) == 1:
                return subset_matches[0]
        return extracted

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

    def _map_value_to_option_letter(self, extracted: str, ground_truths: list[str], question: str | None = None) -> str:
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
        mapped = self._map_option_value_to_letter(extracted, question)
        if mapped != extracted:
            return mapped
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
                tail = out[len(w) :].lstrip()
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

    @classmethod
    def _answer_marker_search_text(cls, raw: str) -> str:
        """Pick the text region to scan for final-answer markers.

        Prefer the post-``</think>`` region ONLY when it carries a *real*
        (non-placeholder) marker — that is the model's official answer line and
        must win over anything said while reasoning. But when the post-think
        region is empty, carries no marker, or merely echoes the prompt's
        literal template (``<answer>LETTER</answer>``, ``\\boxed{FINAL_ANSWER}``),
        fall back to the WHOLE turn so a real answer the model committed *inside*
        the ``<think>`` block — before it truncated or degenerated into echoing
        the placeholder — is still found. Thinking-mode COT (Qwen3.5-4B on GPQA)
        frequently states the choice mid-reasoning (``<answer>B</answer>`` /
        ``\\boxed{C}`` inside the think block) and never restates it cleanly
        after the closing tag, so scanning post-think alone drops the answer.
        """
        if not raw:
            return ""
        think_end = raw.rfind("</think>")
        if think_end == -1:
            return raw
        post_think = raw[think_end + len("</think>") :].strip()
        if not post_think:
            return raw
        if cls._has_non_placeholder_answer_marker(post_think):
            return post_think
        return raw

    @staticmethod
    def _is_placeholder_answer_marker(value: str) -> bool:
        normalized = re.sub(r"[\W_]+", "", str(value or "")).lower()
        return normalized in {"", "answer", "finalanswer", "boxedfinalanswer", "letter", "option", "choice"}

    @classmethod
    def _has_placeholder_answer_marker(cls, raw: str) -> bool:
        if not raw:
            return False
        patterns = (
            r"<answer>\s*((?:(?!<answer>).)*?)\s*</answer>",
            r"(?:\\boxed|boxed|oxed|\x08oxed)\{(.*?)\}",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.DOTALL | re.IGNORECASE):
                if cls._is_placeholder_answer_marker(match.group(1)):
                    return True
        return False

    @classmethod
    def _has_non_placeholder_answer_marker(cls, raw: str) -> bool:
        if not raw:
            return False
        patterns = (
            r"<answer>\s*((?:(?!<answer>).)*?)\s*</answer>",
            r"(?:\\boxed|boxed|oxed|\x08oxed)\{(.*?)\}",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.DOTALL | re.IGNORECASE):
                if not cls._is_placeholder_answer_marker(match.group(1)):
                    return True
        return False

    def _infer_committed_mcq_letter(self, response: str, valid_letters: set[str]) -> str | None:
        """Return the model's last *explicit* MCQ commitment, or ``None``.

        Scans the whole free-text turn for high-precision final-answer phrases
        (``_MCQ_COMMIT_PATTERNS``) and returns the LAST one so that, in a
        rambling COT that revisits several options, the model's final stated
        choice wins. Matches in a tentative context ("if the answer is A") or
        that are part of an option enumeration ("B, C, or D is correct") are
        rejected. ``<think>`` tags are neutralised so a commitment made inside a
        still-open think block (the turn truncated before the post-think answer)
        is still recovered.
        """
        if not response:
            return None
        text = re.sub(r"<think>|</think>", " ", response)
        best_pos = -1
        best_letter: str | None = None
        for pattern in self._MCQ_COMMIT_PATTERNS:
            for match in pattern.finditer(text):
                letter = match.group(1).upper()
                if letter not in valid_letters:
                    continue
                if self._MCQ_TENTATIVE_PREFIX.search(text[max(0, match.start() - 90) : match.start()]):
                    continue
                if self._MCQ_ENUM_FORWARD.match(text[match.end() : match.end() + 8]):
                    continue
                if self._MCQ_ENUM_BACKWARD.search(text[max(0, match.start() - 12) : match.start()]):
                    continue
                if self._MCQ_NEGATIVE_FOLLOW.search(text[match.end() : match.end() + 40]):
                    continue
                if match.start() > best_pos:
                    best_pos = match.start()
                    best_letter = letter
        return best_letter

    def _infer_mcq_letter_from_final_text(self, response: str, question: str | None) -> str | None:
        """Infer an MCQ letter from the final prose when marker tags are placeholders.

        Some Qwen COT evals ended with ``<answer>LETTER</answer>`` but the
        surrounding final sentence named the option value ("the Skyrmion").
        This helper only looks at conclusion-like tail sentences, then maps
        option labels or option values from the question back to A-F.
        """
        options = self._parse_question_options(question)
        if not options:
            return None

        text = self._answer_marker_search_text(self._unwrap_json_fragment(str(response or "")))
        text = re.sub(
            r"<answer>\s*(?:answer|final\s*answer|letter|option|choice)?\s*</answer>",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        segments = [seg.strip() for seg in re.split(r"[\n.!?]+", text[-12000:]) if seg.strip()]
        conclusion_re = re.compile(
            r"\b(?:answer|therefore|thus|hence|conclusion|conclude|corresponds|matches|fits|points to|option|choice|correct|not associated|most likely|safe bet|intended)\b",
            re.IGNORECASE,
        )

        for segment in reversed(segments[-80:]):
            if not conclusion_re.search(segment):
                continue
            label = self._parse_option_label(segment)
            if label and label in options:
                return label
            explicit = re.search(
                r"(?i)\b(?:option|choice|answer)\s*(?:is|:|=|would\s+be|should\s+be)?\s*([A-F])\b",
                segment,
            )
            if explicit and explicit.group(1).upper() in options:
                return explicit.group(1).upper()

            norm_segment = self.normalize_answer(segment)
            for letter, value in options.items():
                norm_value = self.normalize_answer(value)
                if len(norm_value) > 1 and re.search(rf"(?<!\w){re.escape(norm_value)}(?!\w)", norm_segment):
                    return letter
                raw_value = str(value or "").strip()
                if len(raw_value) == 1 and raw_value.islower():
                    quoted_value = re.search(rf"['\"`]\s*{re.escape(raw_value)}\s*['\"`]", segment)
                    labeled_value = re.search(
                        rf"(?i)\b(?:planet|object|value|choice|option|answer|corresponds\s+to)\s+['\"`]?" rf"{re.escape(raw_value)}['\"`]?(?![a-z0-9])",
                        segment,
                    )
                    if quoted_value or labeled_value:
                        return letter
        return None

    def extract_answer_from_response(self, response: str, is_submitted: bool = False) -> str:
        """Extract the final answer from a model response.

        ``is_submitted`` marks the input as an already-submitted answer (the
        ``result`` parameter of a ``finish``/``submit`` tool call), as opposed
        to a raw free-text model turn. The fused web-search env passes the
        submitted ``result`` string directly, so the prose-scavenging cascade
        (the bold/date/name/number/sentence heuristics below) only re-mangles
        an answer that is *already* canonical — e.g. ``"Short Term 12" ->
        "Short Term"`` (first ``[A-Z][a-z]+ ...`` proper-noun match),
        ``"Treviso, Italy" -> "Italy"``, ``"16 Blocks" -> "16"``. When
        ``is_submitted`` is True we apply only the *lossless* unwrappers
        (think-tag strip, JSON-fragment unwrap, ``\\boxed{}`` unwrap, LaTeX
        text-wrapper strip) and return the cleaned value verbatim; the cascade
        is skipped. Validated over the full step-0 eval dump (2350 episodes):
        0 regressions on exact-match, F1, and the (em | f1>=thr) gate, with
        +6 EM / +11 F1 recovered. Callers that pass a full free-text turn
        (e.g. the deepresearch example) must leave ``is_submitted`` False so
        the cascade still runs.
        """
        raw_response = response.strip()
        raw_unwrapped = self._unwrap_json_fragment(raw_response)
        marker_response = self._answer_marker_search_text(raw_unwrapped)

        # Remove thinking tags for prose scavenging, but keep a marker-search
        # fallback above for Qwen-style outputs whose final answer is still
        # inside the closing <think> block.
        response = re.sub(r"<think>.*?</think>", "", raw_unwrapped, flags=re.DOTALL)
        response = re.sub(r"\s+", " ", response).strip()

        # 1. HIGHEST PRIORITY: Look for final answer markers.
        def boxed_spans(s: str) -> list[tuple[int, str]]:
            """Return every complete \\boxed{...} (in source order) with nesting.

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
                return []
            anchors = []
            for tok in ("\\boxed{", "boxed{", "oxed{", "\x08oxed{"):
                i = s.find(tok)
                while i >= 0:
                    anchors.append((i, len(tok)))
                    i = s.find(tok, i + 1)
            spans = []
            for start, tok_len in sorted(anchors):
                i = start + tok_len
                depth = 1
                j = i
                while depth and j < len(s):
                    if s[j] == "{":
                        depth += 1
                    elif s[j] == "}":
                        depth -= 1
                    j += 1
                if depth == 0:
                    spans.append((start, s[i : j - 1]))
            return spans

        def answer_tag_spans(s: str) -> list[tuple[int, str]]:
            return [(m.start(), m.group(1).strip()) for m in re.finditer(r"<answer>\s*((?:(?!<answer>).)*?)\s*</answer>", s, flags=re.DOTALL | re.IGNORECASE)]

        # Thinking-mode COT often echoes the prompt's literal format placeholders
        # ("<answer>LETTER</answer>", "\\boxed{FINAL_ANSWER}") or fuses the
        # placeholder word onto the real choice ("\\boxed{LETTER A}") AFTER it has
        # already committed the actual option letter. Strip a leading placeholder
        # word when the remainder is a bare option letter ("LETTER A" -> "A"); this
        # only fires on the exact template-contamination shape, never on prose.
        def _decontaminate_marker(content: str) -> str:
            m = re.fullmatch(r"(?i)(?:final\s*answer|answer|letter|option|choice)\s*[:=).\-]?\s*\(?\s*([A-Fa-f])\s*\)?", content.strip())
            return m.group(1) if m else content

        all_spans = sorted(boxed_spans(marker_response) + answer_tag_spans(marker_response), key=lambda item: item[0])
        cleaned_spans = [(pos, _decontaminate_marker(content)) for pos, content in all_spans]
        marker_candidates = [(pos, content) for pos, content in cleaned_spans if not self._is_placeholder_answer_marker(content)]
        if marker_candidates:
            # A marker holding a bare option letter is the MCQ final-answer line
            # mandated by the cot/bare harness prompt; it is authoritative and must
            # win over a later non-letter ``\boxed{value}`` artifact. Otherwise a
            # turn ending ``<answer>D</answer>\n\boxed{33.4}`` (or ``\boxed{B}`` then
            # ``\boxed{33.5}``) mis-extracts the trailing value, which then fails to
            # map back to the option letter. Prefer the LAST bare-letter marker so a
            # turn that revisits options still resolves to its final committed choice.
            letter_spans = [(pos, content) for pos, content in marker_candidates if re.fullmatch(r"\(?\s*[A-Fa-f]\s*\)?", content.strip())]
            if letter_spans:
                return letter_spans[-1][1].strip().strip("()").strip().upper()
            return self._strip_latex_wrappers(marker_candidates[-1][1].strip())

        if not response:
            return ""

        # For an already-submitted answer, the value above is canonical: the
        # lossless unwrappers (think/JSON/boxed) have run and no \boxed{} was
        # present, so return it as-is. Skipping the prose cascade avoids
        # truncating clean answers like "Short Term 12"/"Treviso, Italy".
        if is_submitted:
            submitted = self._strip_latex_wrappers(response.strip())
            submitted = self._coerce_submitted_answer_text(submitted)
            return "" if self._is_placeholder_answer_marker(submitted) else submitted

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

    def _extract_and_map(self, model_answer, raw_ground_truths, question, data_source, is_submitted):
        """Extract a candidate answer and apply the MCQ letter<->value maps.

        Factored out so the cascade candidate and the verbatim (submitted)
        candidate run through the *identical* normalization before scoring.
        """
        cand = self.extract_answer_from_response(model_answer, is_submitted=is_submitted)
        # For multiple-choice (GPQA-style): try mapping raw values to option letters
        cand = self._map_value_to_option_letter(cand, raw_ground_truths, question)
        if question and not is_submitted and all(str(gt).strip().upper() in self._OPTION_LETTERS for gt in raw_ground_truths) and not self._has_non_placeholder_answer_marker(str(model_answer or "")):
            # Prefer an explicit final-answer commitment scanned over the whole
            # turn (fixes COT false negatives where the cascade grabbed a bold
            # heading or an earlier hypothetical letter). Fall back to the
            # conservative tail-scan inference, then the cascade candidate.
            options = self._parse_question_options(question)
            valid_letters = set(options) if options else set(self._OPTION_LETTERS)
            committed = self._infer_committed_mcq_letter(str(model_answer or ""), valid_letters)
            if committed:
                cand = committed
            else:
                inferred = self._infer_mcq_letter_from_final_text(str(model_answer or ""), question)
                if inferred:
                    cand = inferred
        # Reverse mapping for MCQ-shaped datasets whose GT is prose (medqa).
        if data_source in {"medqa"} and question:
            mapped = self._map_letter_to_option_value(cand, question)
            if mapped != cand:
                cand = mapped
        return cand

    def evaluate_answer(
        self,
        model_answer: str,
        ground_truth: str | list[str],
        question: str | None = None,
        data_source: str | None = None,
        is_submitted: bool = False,
    ) -> tuple[bool, float, dict[str, Any]]:
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        # In fused eval, ``model_answer`` is the submitted result parameter (or
        # an implicit \boxed{} answer rescued by the env). Treat that as already
        # canonical and avoid the free-text scavenging cascade, which can mangle
        # answers such as "Short Term 12" -> "Short Term", "\text{Soissons}" ->
        # "issons", or a refusal sentence -> an entity mentioned inside it.
        extracted_answer = self._extract_and_map(model_answer, ground_truths, question, data_source, is_submitted=is_submitted)

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

        # Mark submitted partial matches so logged reward_debug makes it clear
        # they were scored from the verbatim result, not the prose cascade.
        if is_submitted and not max_em:
            metadata["submitted_verbatim_evaluation"] = True

        # Web Search QA uses strict exact-match correctness. F1/partial-match
        # checks are retained only as diagnostic metadata for near-miss
        # analysis; they must not produce correctness or reward credit.
        f1_threshold = 1.0
        partial_match_accepted = False
        partial_match_reject_reason = ""
        if not max_em and max_f1 > 0.0:
            _, diagnostic_reject_reason = self._partial_match_acceptance(
                extracted_answer,
                best_match,
                best_precision,
                best_recall,
            )
            partial_match_reject_reason = diagnostic_reject_reason or "exact_match_required"
        is_correct = max_em

        metadata.update(
            {
                "best_match": best_match,
                "f1_score": max_f1,
                "precision": best_precision,
                "recall": best_recall,
                "exact_match": max_em,
                "f1_threshold": f1_threshold,
                "partial_match_accepted": partial_match_accepted,
                "partial_match_reject_reason": partial_match_reject_reason,
            }
        )

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
            is_submitted=bool(input.task_info.get("is_submitted", False)),
        )

        if metadata.get("exact_match", False):
            reward = self.config.correct_reward
        else:
            reward = self.config.incorrect_reward

        # Web Search QA reward is intentionally binary. Ignore step bonuses and
        # shaping penalties here so the output remains exactly incorrect_reward
        # or correct_reward regardless of caller config.
        step_bonus = 0.0

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
        repetition_penalty_weighted = 0.0

        # P1-3: length / self-restart penalty. Step-0 evals showed
        # 53-99% of rollouts with single-message >8k chars and heavy
        # "Wait," restart cycles. This signal is orthogonal to the
        # n-gram repetition penalty above and catches long "rambling
        # but not literally repeating" traces.
        length_penalty_weighted = 0.0

        # Add tool call information and other reward components to metadata
        metadata.update(
            {
                "base_reward": reward,
                "tool_call_reward": 0,
                "repetition_penalty_reward": repetition_penalty_weighted,
                "length_penalty_reward": length_penalty_weighted,
                "step_bonus": step_bonus,
            }
        )

        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)
