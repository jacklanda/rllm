#!/usr/bin/env python3
"""Regression tests for the free-text (COT) MCQ commitment extractor.

Qwen-style COT eval rollouts often state a clear final choice ("the answer is
C", "C is the correct answer", "I'll go with C") and then keep deliberating
until the per-step token cap truncates the turn. The generic prose cascade then
mis-extracts an early **bold** heading or an earlier hypothetical letter,
producing a false negative on an answer the model actually committed to.

``_infer_committed_mcq_letter`` scans the whole turn for high-precision
commitment phrases and returns the LAST surviving one, while rejecting
tentative ("if the answer is A"), enumerated ("B, C, or D is correct") and
entity-letter ("A = Benzoquinone", lowercase "(c)") confusions.
"""

from rllm.rewards.reward_types import RewardConfig
from rllm.rewards.search_reward import RewardSearchFn

QUESTION = "A multiple choice question.\nA. alpha\nB. beta\nC. gamma\nD. delta"


def _fn():
    return RewardSearchFn(RewardConfig())


def _evaluate(text, gt="C", question=QUESTION):
    # Mirrors the fused COT path: free-text turn, is_submitted=False.
    return _fn().evaluate_answer(text, gt, question=question, data_source="gpqa", is_submitted=False)


def test_answer_is_letter_recovered():
    is_correct, _, meta = _evaluate("Lots of reasoning.\nTherefore the answer is C.")
    assert is_correct and meta["extracted_answer"] == "C"


def test_letter_is_the_correct_answer_recovered():
    is_correct, _, meta = _evaluate("After analysis,\nC is the correct answer.\n")
    assert is_correct and meta["extracted_answer"] == "C"


def test_choice_verb_recovered():
    is_correct, _, meta = _evaluate("Hmm.\nI'll go with C because of the symmetry.")
    assert is_correct and meta["extracted_answer"] == "C"


def test_bold_heading_no_longer_shadows_commitment():
    """The cascade alone grabs the bold heading; the commitment scan fixes it."""
    fn = _fn()
    text = "**Conditions:**\nSome long reasoning.\nThus the answer is C.\nMore musing."
    # Cascade in isolation harvests the bold heading -> false negative.
    assert fn.extract_answer_from_response(text) == "Conditions:"
    is_correct, _, meta = _evaluate(text)
    assert is_correct and meta["extracted_answer"] == "C"


def test_last_commitment_wins():
    """A revisited choice: the model's final stated answer is taken."""
    is_correct, _, meta = _evaluate("First I think the answer is B.\nNo wait, the answer is C.")
    assert is_correct and meta["extracted_answer"] == "C"


def test_tentative_commitment_rejected():
    """A hypothetical mention is not a commitment (scanner returns None)."""
    fn = _fn()
    assert fn._infer_committed_mcq_letter("If the answer is C then x follows.", set("ABCD")) is None
    assert fn._infer_committed_mcq_letter("Let's assume the answer is D for now.", set("ABCD")) is None
    assert fn._infer_committed_mcq_letter("Is it possible the answer is C? I need to check.", set("ABCD")) is None
    assert fn._infer_committed_mcq_letter("Is there any scenario where C is correct? Maybe not.", set("ABCD")) is None


def test_tentative_does_not_override_genuine_commitment():
    """A later tentative mention must not shadow an earlier firm commitment."""
    fn = _fn()
    text = "The answer is C.\nBut what if the answer is D instead?"
    assert fn._infer_committed_mcq_letter(text, set("ABCD")) == "C"


def test_hypothetical_late_answer_does_not_override_tail_inference():
    """A late hypothetical option mention must not beat a final prose answer."""
    text = (
        "So the answer should be B.\n"
        "However, I should double check if there is any scenario where C is correct.\n"
        "After checking, most likely answer B."
    )
    is_correct, _, meta = _evaluate(text, gt="B")
    assert is_correct and meta["extracted_answer"] == "B"


def test_later_likely_answer_overrides_earlier_commitment():
    """A final likely-answer phrase is a real commitment and should win."""
    text = "At first the answer is B.\nAfter recalculating, most likely answer C."
    is_correct, _, meta = _evaluate(text, gt="C")
    assert is_correct and meta["extracted_answer"] == "C"


def test_likely_incorrect_option_is_not_a_commitment():
    fn = _fn()
    assert fn._infer_committed_mcq_letter("Option B is likely incorrect. Let's check D.", set("ABCD")) is None
    assert fn._infer_committed_mcq_letter("After checking, option B is the safe bet.", set("ABCD")) == "B"


def test_tail_matches_option_sentence_is_recovered():
    """A final value/option match sentence should beat an early heading."""
    text = "**Given information:** lots of chemistry reasoning.\nThis matches option C."
    is_correct, _, meta = _evaluate(text, gt="C")
    assert is_correct and meta["extracted_answer"] == "C"


def test_enumeration_not_a_commitment():
    """'B, C, or D is correct' lists options; it is not a single commitment."""
    fn = _fn()
    assert fn._infer_committed_mcq_letter("whether B, C, or D is correct here", set("ABCD")) is None
    assert fn._infer_committed_mcq_letter("the answer is A or C", set("ABCD")) is None


def test_entity_letter_not_mistaken_for_commitment():
    """'A = Benzoquinone' labels a compound, not an option commitment."""
    fn = _fn()
    assert fn._infer_committed_mcq_letter("Therefore, A = Benzoquinone.", set("ABCD")) is None


def test_lowercase_item_not_mistaken_for_option():
    """Lowercase '(c)' refers to a problem item, not option C."""
    q = "How many stars are detectable?\nA. 2\nB. 3\nC. 4\nD. 5"
    # Model concludes option B but discusses star (c); (c) must not win.
    text = "Star (c) is the densest one.\nTherefore the answer is B."
    is_correct, _, meta = _fn().evaluate_answer(text, "B", question=q, data_source="gpqa", is_submitted=False)
    assert is_correct and meta["extracted_answer"] == "B"


def test_submitted_answer_untouched_by_commitment_scan():
    """A verbatim finish/submit result still bypasses the free-text scan."""
    is_correct, _, meta = _fn().evaluate_answer("C", "C", question=QUESTION, data_source="gpqa", is_submitted=True)
    assert is_correct and meta["extracted_answer"] == "C"
