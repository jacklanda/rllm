import re
import string
from collections import Counter
from typing import Any, List

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput


def repetition_penalty_reward(
    text: str,
    max_n: int = 4,
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

    def normalize_answer(self, s: str) -> str:
        """Normalize answer text for evaluation (following HotpotQA/SQuAD standards)"""

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
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def extract_answer_from_response(self, response: str) -> str:
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

    def evaluate_answer(self, model_answer: str, ground_truth: str | list[str]) -> tuple[bool, float, dict[str, Any]]:
        extracted_answer = self.extract_answer_from_response(model_answer)

        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

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
        tool_call_count, tool_call_contents, is_valid_parsing = self.parse_tool_calls(model_response)

        is_correct, score, metadata = self.evaluate_answer(model_response, ground_truth)

        if is_correct:
            # For exact matches, give full reward
            # For F1 matches, scale reward by F1 score
            if metadata.get("exact_match", False):
                reward = self.config.correct_reward
            else:
                # Scale reward by F1 score for partial matches
                reward = self.config.correct_reward * score
        else:
            reward = self.config.incorrect_reward

        # Apply tool call bonus/penalty based on new strategy:
        # 1. Invalid tags -> -0.5 penalty
        # 2. Single tool call + reward > 0 -> +0.5 bonus (only if answer is correct/partially correct)
        # 3. Multiple tool calls (>= 2) -> -0.5 penalty (regardless of correctness)
        # 4. No tool call -> 0 adjustment
        tool_call_adjustment = 0.0

        """
        if not is_valid_parsing:
            # Invalid/unparseable tool call tags - penalize
            tool_call_adjustment = -self.config.toolcall_bonus
            metadata["tool_call_status"] = "invalid_tags_penalty"
        elif tool_call_count >= 2:
            # Multiple tool calls - always penalize to suppress repeated calls
            tool_call_adjustment = -self.config.toolcall_bonus
            metadata["tool_call_status"] = "multiple_calls_penalty"
        elif tool_call_count == 1 and reward > 0:
            # Single tool call with positive reward - give bonus
            # This encourages tool use only when it leads to correct/partially correct answers
            tool_call_adjustment = self.config.toolcall_bonus
            metadata["tool_call_status"] = "single_call_bonus"
        elif tool_call_count == 1 and reward <= 0:
            # Single tool call but incorrect answer - no bonus (avoid reward hacking)
            tool_call_adjustment = 0.0
            metadata["tool_call_status"] = "single_call_no_bonus"
        else:
            # No tool call detected
            tool_call_adjustment = 0.0
            metadata["tool_call_status"] = "no_tool_call"
        """

        # Store base reward before adjustments
        base_reward = reward

        if self.config.toolcall_bonus != 0.0:
            reward += tool_call_adjustment

        # Apply repetition penalty if enabled
        repetition_penalty = 0.0
        if self.config.apply_repetition_penalty:
            repetition_penalty = repetition_penalty_reward(
                model_response,
                max_n=self.config.repetition_max_n
            )
            # Scale by weight and add to total reward
            repetition_penalty_weighted = repetition_penalty * self.config.repetition_penalty_weight
            reward += repetition_penalty_weighted

        # Add tool call information and other reward components to metadata
        metadata.update({
            "base_reward": base_reward,
            "tool_call_reward": tool_call_adjustment,
            "repetition_penalty_reward": repetition_penalty * self.config.repetition_penalty_weight if self.config.apply_repetition_penalty else None,
        })

        return RewardOutput(reward=reward, is_correct=is_correct, metadata=metadata)
