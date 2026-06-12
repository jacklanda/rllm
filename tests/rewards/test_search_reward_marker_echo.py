#!/usr/bin/env python3
"""Regression tests for template-placeholder / stray-value marker echoes.

Thinking-mode COT (Qwen3.5-4B, GPQA) frequently commits the real option letter
via a final-answer marker and then *also* echoes the prompt's literal format
placeholders ("<answer>LETTER</answer>", "\\boxed{FINAL_ANSWER}"), fuses the
placeholder word onto the choice ("\\boxed{LETTER A}"), or appends a stray
numeric "\\boxed{value}" after the committing "\\boxed{B}". Positionally-last
marker selection then grabbed the placeholder / value and the answer failed to
map back to the option letter — a rollout-time false negative.

The fix (``extract_answer_from_response``):
  * scans ALL <answer>/\\boxed markers, not just the last;
  * decontaminates "LETTER A" -> "A" (placeholder word + bare option letter);
  * drops pure placeholders ("LETTER", "FINAL_ANSWER", ...);
  * prefers the LAST bare-option-letter marker over a later non-letter value.

Each case below is a real step-0 eval false negative
(experiments/.../cot_evals_20260611212409.json).
"""

from rllm.rewards.reward_types import RewardConfig
from rllm.rewards.search_reward import RewardSearchFn

QUESTION = "A multiple choice question.\nA. alpha\nB. beta\nC. gamma\nD. delta"


def _fn():
    return RewardSearchFn(RewardConfig())


def _evaluate(text, gt, question=QUESTION):
    return _fn().evaluate_answer(text, gt, question=question, data_source="gpqa", is_submitted=False)


def _extract(text):
    return _fn().extract_answer_from_response(text, is_submitted=False)


# --- eid 8a3e1c01 (GT=B): committing \boxed{B} then a stray \boxed{33.5} value ---
def test_boxed_letter_beats_trailing_boxed_value():
    text = "Reasoning. This matches option B.\n\n\\boxed{B}\n\n\\boxed{33.5}"
    assert _extract(text) == "B"
    is_correct, _, meta = _evaluate(text, "B")
    assert is_correct and meta["extracted_answer"] == "B"


# --- eid 951d5950 (GT=A): \boxed{LETTER A} — placeholder word fused to choice ---
def test_boxed_placeholder_word_fused_to_letter():
    text = "The valid statement is the Group VIa catalyst.\n\n\\boxed{LETTER A}"
    assert _extract(text) == "A"
    is_correct, _, meta = _evaluate(text, "A")
    assert is_correct and meta["extracted_answer"] == "A"


# --- eid a94386e9 (GT=B): <answer>B</answer> then echoed <answer>LETTER</answer> ---
def test_answer_tag_echoed_placeholder_after_real_letter():
    text = (
        "Therefore, B. <answer>B</answer>\n"
        'Wait, the instruction says "The final line must be: <answer>LETTER</answer>".\n'
        "So I should use <answer>B</answer>.\n"
        'The prompt says: "the final line must be: <answer>LETTER</answer>". So I will output'
    )
    assert _extract(text) == "B"
    is_correct, _, meta = _evaluate(text, "B")
    assert is_correct and meta["extracted_answer"] == "B"


def test_placeholder_only_does_not_fabricate_letter():
    # A turn that ONLY echoed the placeholder (no real commitment) must NOT be
    # scored correct against any letter — the placeholder is not an answer.
    text = "I am not sure.\n<answer>LETTER</answer>\n\\boxed{FINAL_ANSWER}"
    is_correct_a, _, _ = _evaluate(text, "A")
    is_correct_b, _, _ = _evaluate(text, "B")
    assert not is_correct_a and not is_correct_b


def test_single_boxed_letter_still_works():
    # Guard against over-stripping: a plain committing marker is untouched.
    assert _extract("Final.\n\\boxed{C}") == "C"
    assert _extract("Final.\n<answer>D</answer>") == "D"


def test_answer_letter_still_beats_trailing_boxed_value():
    # The prior fix (answer-tag letter beats trailing \boxed{value}) must hold.
    text = "Reasoning.\n<answer>D</answer>\n\\boxed{33.4}"
    assert _extract(text) == "D"


# --- think-internal markers: the model commits the choice *inside* <think> and
#     then only echoes the prompt's placeholder template after </think>. The
#     marker-search region must fall back to the whole turn so the real choice
#     is recovered (not just the post-</think> placeholder). ---
def test_real_marker_inside_think_with_placeholder_after():
    text = "<think>Reasoning. This matches option B.\n<answer>B</answer></think>\n<answer>LETTER</answer>"
    assert _extract(text) == "B"
    is_correct, _, meta = _evaluate(text, "B")
    assert is_correct and meta["extracted_answer"] == "B"


def test_real_boxed_inside_think_with_placeholder_boxed_after():
    text = "<think>The catalyst is the Group VIa metal, so \\boxed{A}.</think>\n\\boxed{FINAL_ANSWER}"
    assert _extract(text) == "A"
    is_correct, _, meta = _evaluate(text, "A")
    assert is_correct and meta["extracted_answer"] == "A"


def test_real_post_think_marker_still_wins_over_think_mention():
    # A genuine final-answer line after </think> is authoritative and must beat
    # any option merely mentioned while reasoning inside the think block.
    text = "<think>Maybe A, or possibly C.</think>\nFinal answer: <answer>D</answer>"
    assert _extract(text) == "D"
    is_correct, _, meta = _evaluate(text, "D")
    assert is_correct and meta["extracted_answer"] == "D"


def test_marker_inside_unclosed_think_is_recovered():
    # Thinking-mode COT truncated before emitting </think>: the choice committed
    # mid-think (no closing tag, no post-think text) must still be found.
    text = "<think>Working through it. The answer is option C, so \\boxed{C}. Wait, let me"
    assert _extract(text) == "C"
    is_correct, _, meta = _evaluate(text, "C")
    assert is_correct and meta["extracted_answer"] == "C"


def test_think_internal_placeholder_only_does_not_fabricate():
    # If the think block holds only a placeholder echo (no real choice), nothing
    # should be fabricated against any letter.
    text = "<think>I am unsure. <answer>LETTER</answer></think>\n<answer>LETTER</answer>"
    is_correct_a, _, _ = _evaluate(text, "A")
    is_correct_b, _, _ = _evaluate(text, "B")
    assert not is_correct_a and not is_correct_b
