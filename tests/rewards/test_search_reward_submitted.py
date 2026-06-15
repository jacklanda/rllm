#!/usr/bin/env python3
"""Regression tests for the ``is_submitted`` EM-union path in RewardSearchFn.

The fused web-search env passes the verbatim ``result`` parameter of a
finish/submit tool call as the answer. The prose-scavenging extraction cascade
truncates such canonical answers ("Short Term 12" -> "Short Term"); the
``is_submitted=True`` path evaluates the lossless verbatim candidate directly,
without letting the free-text cascade truncate or re-interpret it.
"""

from rllm.rewards.reward_types import RewardConfig
from rllm.rewards.search_reward import RewardSearchFn


def _fn():
    return RewardSearchFn(RewardConfig())


def test_submitted_exact_match_recovered():
    """A clean submitted answer that the cascade would truncate is matched."""
    fn = _fn()
    # Cascade alone harvests the first proper-noun bigram ("Short Term"); the
    # verbatim path keeps the full title and exact-matches.
    em_cascade = fn.exact_match_score(fn.extract_answer_from_response("Short Term 12"), "Short Term 12")
    assert not em_cascade  # demonstrates the original truncation bug
    is_correct, f1, meta = fn.evaluate_answer("Short Term 12", "Short Term 12", is_submitted=True)
    assert is_correct and meta["exact_match"] and f1 == 1.0


def test_submitted_latex_text_answer_not_truncated():
    """Submitted LaTeX text wrappers are stripped without dropping prefixes."""
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("\\text{Soissons}", "Soissons", is_submitted=True)
    assert is_correct and meta["exact_match"] and f1 == 1.0
    assert meta["extracted_answer"] == "Soissons"


def test_submitted_latex_letter_maps_to_option_value():
    """\\text{C} submitted for an medqa MCQ resolves to the option value."""
    fn = _fn()
    question = "Q?\nA. Succinylcholine\nB. Inhaled ipratropium and oxygen\nC. Atropine and pralidoxime\nD. Inhaled albuterol and oxygen"
    is_correct, f1, meta = fn.evaluate_answer("\\text{C}", "Atropine and pralidoxime", question=question, data_source="medqa", is_submitted=True)
    assert is_correct and meta["exact_match"]


def test_submitted_uses_verbatim_answer_for_f1():
    """Submitted answers are scored as submitted, not as cascade snippets."""
    fn = _fn()
    _, f1_cascade, meta_cascade = fn.evaluate_answer("Treviso, Italy", "Treviso", is_submitted=False)
    _, f1_submitted, meta_submitted = fn.evaluate_answer("Treviso, Italy", "Treviso", is_submitted=True)
    assert meta_cascade["extracted_answer"] != meta_submitted["extracted_answer"]
    assert f1_submitted == 2 / 3
    assert f1_submitted != f1_cascade


def test_submitted_refusal_sentence_not_reduced_to_mentioned_entity():
    """A refusal sentence should not be scored as the entity it mentions."""
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer('The film "Wagtails Army" does not exist.', "Wagtails Army", is_submitted=True)
    assert not is_correct
    assert f1 > 0.0
    assert meta["extracted_answer"] == 'The film "Wagtails Army" does not exist.'


def test_submitted_no_false_positive_on_wrong_entity():
    """A wrong but token-overlapping answer must stay wrong (no EM granted)."""
    fn = _fn()
    # "Joan I of Navarre" vs "Joan II of Navarre": high token overlap, wrong.
    _, _, meta = fn.evaluate_answer("\\boxed{Joan I of Navarre}", "Joan II of Navarre", is_submitted=True)
    assert not meta["exact_match"]


def test_parenthetical_acronym_alias_exact_match():
    fn = _fn()
    for prediction, ground_truth in [
        ("rilpivirine", "Rilpivirine (RPV)"),
        ("RPV", "Rilpivirine (RPV)"),
        ("interstimulus interval", "Inter-stimulus interval (ISI)"),
        ("ISI", "Inter-stimulus interval (ISI)"),
    ]:
        is_correct, f1, meta = fn.evaluate_answer(prediction, ground_truth, is_submitted=True)
        assert is_correct
        assert f1 == 1.0
        assert meta["exact_match"]


def test_version_prefix_alias_exact_match():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("VarScan 2.3.9", "VarScan v2.3.9", is_submitted=True)
    assert is_correct
    assert f1 == 1.0
    assert meta["exact_match"]


def test_submitted_backslash_escape_is_coerced_before_scoring():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer(r"Documenting\ Hate", "Documenting Hate project", is_submitted=True)
    assert is_correct
    assert f1 == 1.0
    assert meta["exact_match"]
    assert meta["extracted_answer"] == "Documenting Hate"


def test_generic_entity_suffix_alias_exact_match():
    fn = _fn()
    for prediction, ground_truth in [
        ("Documenting Hate", "Documenting Hate project"),
        ("JGI Genome", "JGI Genome Portal"),
        ("Messiah Stradivarius dendrochronology", 'The "Messiah" Stradivarius dendrochronology report'),
        ("ISO 1087-1", "ISO 1087-1 standard"),
    ]:
        is_correct, f1, meta = fn.evaluate_answer(prediction, ground_truth, is_submitted=True)
        assert is_correct
        assert f1 == 1.0
        assert meta["exact_match"]


def test_generic_entity_suffix_alias_requires_same_core():
    fn = _fn()
    for prediction, ground_truth in [
        ("Genome Portal", "JGI Genome Portal"),
        ("Stradivarius", 'The "Messiah" Stradivarius dendrochronology report'),
        ("Latin", "Medieval Latin standard"),
    ]:
        is_correct, f1, meta = fn.evaluate_answer(prediction, ground_truth, is_submitted=True)
        assert not is_correct
        assert f1 > 0.0
        assert not meta["exact_match"]


def test_alias_matching_does_not_accept_plain_entity_substring():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("Latin", "Medieval Latin", is_submitted=True)
    assert not is_correct
    assert f1 > 0.0
    assert not meta["exact_match"]


def test_partial_year_does_not_count_as_correct_date_answer():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("1837", "20 June 1837", is_submitted=True)
    assert not is_correct
    assert f1 == 0.5
    assert meta["partial_match_reject_reason"] == "temporal_granularity_mismatch"


def test_temporal_duration_mismatch_does_not_count_as_correct():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer(
        "Treat with three weekly injections of penicillin, obtain titers in 3 months",
        "Treat with three weekly injections of penicillin, obtain titers in 6 months",
        is_submitted=True,
    )
    assert not is_correct
    assert f1 > 0.8
    assert meta["partial_match_reject_reason"] == "temporal_granularity_mismatch"


def test_entity_modifier_missing_does_not_count_as_correct():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("Latin", "Medieval Latin", is_submitted=True)
    assert not is_correct
    assert f1 > 0.0
    assert meta["partial_match_reject_reason"] == "substring_entity_mismatch"


def test_short_entity_token_overlap_does_not_count_as_correct():
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("Jean I de La Tremoille", "Francois II de La Tremoille", is_submitted=True)
    assert not is_correct
    assert f1 >= 0.3
    assert meta["partial_match_reject_reason"] == "short_entity_token_overlap"


def test_submitted_flag_off_preserves_cascade():
    """Default (is_submitted=False) behavior is unchanged: the cascade still
    truncates a multi-word title, documenting the prior behavior."""
    fn = _fn()
    assert fn.extract_answer_from_response("Short Term 12") == "Short Term"


def test_verbatim_extract_is_lossless():
    """is_submitted extraction only does lossless unwraps, no prose cascade."""
    fn = _fn()
    assert fn.extract_answer_from_response("Treviso, Italy", is_submitted=True) == "Treviso, Italy"
    assert fn.extract_answer_from_response("\\boxed{Mazhai}", is_submitted=True) == "Mazhai"
    assert fn.extract_answer_from_response("\\text{Latin}", is_submitted=True) == "Latin"


def test_submitted_placeholder_boxed_answer_extracts_empty():
    fn = _fn()
    assert fn.extract_answer_from_response("\\boxed{FINAL_ANSWER}", is_submitted=True) == ""
    is_correct, _, meta = fn.evaluate_answer("\\boxed{FINAL_ANSWER}", "Martin King Whyte", is_submitted=True)
    assert not is_correct
    assert meta["extracted_answer"] == ""


def test_multiple_boxed_uses_final_answer_for_submitted_response():
    """The final boxed answer, not an intermediate boxed value, is graded."""
    fn = _fn()
    response = r"<think>Intermediate result is \boxed{6.3 \times 10^{-7} \text{ M}}</think> The option is \boxed{A}."
    is_correct, f1, meta = fn.evaluate_answer(response, "A", data_source="gpqa_diamond", is_submitted=True)
    assert is_correct and meta["exact_match"] and f1 == 1.0
    assert meta["extracted_answer"] == "A"


def test_final_answer_inside_think_block_is_not_stripped_to_empty():
    fn = _fn()
    response = r"<think>Intermediate option was \boxed{A}. After checking, <answer>D</answer></think>"
    is_correct, f1, meta = fn.evaluate_answer(response, "D", is_submitted=False)

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "D"


def test_final_answer_inside_think_block_survives_truncated_post_think():
    fn = _fn()
    response = (
        "<think>After checking, <answer>D</answer></think>\n"
        "Now I will restate the solution, but this post-think text is truncated before the final marker."
    )
    is_correct, f1, meta = fn.evaluate_answer(response, "D", is_submitted=False)

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "D"


def test_last_answer_marker_wins_over_earlier_boxed_marker():
    fn = _fn()
    response = r"Initial calculation suggested \boxed{A}. Final answer: <answer>B</answer>"
    is_correct, f1, meta = fn.evaluate_answer(response, "B", is_submitted=False)

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "B"


def test_answer_tag_is_extracted_before_prose_heuristics():
    fn = _fn()
    response = "The likely option is A at first. After checking, <answer>B</answer>"
    is_correct, f1, meta = fn.evaluate_answer(response, "B", is_submitted=False)

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "B"


def test_unique_option_phrase_maps_to_mcq_letter():
    fn = _fn()
    question = (
        "What is observed?\n"
        "A. Warm atomic interstellar medium.\n"
        "B. Cold molecular interstellar medium.\n"
        "C. Cold atomic interstellar medium.\n"
        "D. Warm molecular interstellar medium."
    )
    response = r"Therefore, <answer>C</answer> \boxed{Cold atomic}"
    is_correct, f1, meta = fn.evaluate_answer(response, "C", question=question, data_source="gpqa_diamond")

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "C"


def test_lowercase_option_value_maps_to_mcq_letter():
    fn = _fn()
    question = "Which object?\nA. d\nB. a\nC. b\nD. c"
    is_correct, f1, meta = fn.evaluate_answer("c", "D", question=question, data_source="gpqa_diamond", is_submitted=True)

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "D"


def test_multiple_boxed_uses_final_nested_text_answer():
    fn = _fn()
    response = r"<think>Candidate \boxed{\text{Unknown}}</think> Final answer: \boxed{\text{Simone Fontecchio}}"
    is_correct, f1, meta = fn.evaluate_answer(response, "Simone Fontecchio", is_submitted=True)
    assert is_correct and meta["exact_match"] and f1 == 1.0
    assert meta["extracted_answer"] == "Simone Fontecchio"


def test_empty_answer_does_not_exact_match_option_a():
    """Hotpot/SQuAD article stripping must not turn option A into empty."""
    fn = _fn()
    is_correct, f1, meta = fn.evaluate_answer("", "A", is_submitted=True)
    assert not is_correct
    assert f1 == 0.0
    assert not meta["exact_match"]


def test_placeholder_answer_marker_falls_back_to_final_option_value():
    fn = _fn()
    question = "Which particle is not a Goldstone mode?\nA. Magnon\nB. Phonon\nC. Pion\nD. Skyrmion"
    response = (
        "Magnons, phonons, and pions are Goldstone modes. "
        "Thus, the particle not associated with this mechanism is the Skyrmion.\n"
        "<answer>LETTER</answer>"
    )

    is_correct, f1, meta = fn.evaluate_answer(response, "D", question=question, data_source="gpqa_diamond")

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "D"


def test_prompt_only_mcq_final_option_sentence_without_marker_is_rescued():
    fn = _fn()
    question = (
        "Among the following exoplanets, which one has the highest density?\n\n"
        "A. d\nB. a\nC. b\nD. c"
    )
    response = (
        "After comparing densities, Planet 'c' has the highest density. "
        "The correct choice corresponds to Planet 'c'."
    )

    is_correct, f1, meta = fn.evaluate_answer(response, "D", question=question, data_source="gpqa_diamond")

    assert is_correct
    assert f1 == 1.0
    assert meta["extracted_answer"] == "D"

def test_answer_letter_tag_beats_trailing_boxed_value():
    """A bare-letter <answer> tag is authoritative over a trailing \\boxed{value}.

    Regression for GPQA cot episode 4d577963: the turn ended with
    ``<answer>D</answer>\\n\\boxed{33.4}``. The extractor took the
    positionally-last marker (the boxed value ``33.4``), which then failed to
    map back to option D ("~ 33.4" normalizes differently), producing a false
    negative on an answer the model had explicitly committed to.
    """
    fn = _fn()
    question = "Which planet?\nA. ~ 3.2\nB. ~ 4.4\nC. ~ 10.4\nD. ~ 33.4"
    response = "The calculated value is approximately 33.4.\n<answer>D</answer>\n\\boxed{33.4}"
    assert fn.extract_answer_from_response(response) == "D"
    is_correct, f1, meta = fn.evaluate_answer(response, "D", question=question, data_source="gpqa_diamond")
    assert is_correct and meta["exact_match"] and meta["extracted_answer"] == "D"


def test_answer_value_tag_still_used_when_not_bare_letter():
    """When <answer> holds a value (not a bare letter), positional pick stands."""
    fn = _fn()
    # No bare-letter answer tag -> boxed value (later position) is still chosen.
    response = "Reasoning...\n<answer>Paris</answer>\n\\boxed{London}"
    assert fn.extract_answer_from_response(response) == "London"
