from rllm.environments.fused.fused import FusedEnv


def test_fused_env_boxed_rescue_uses_last_boxed_after_thinking():
    raw = r"""
<think>
I considered an empty placeholder \boxed{} and an intermediate value \boxed{B}.
</think>
The final answer is \boxed{A}.
"""
    assert FusedEnv._extract_boxed_from_raw(raw) == "A"


def test_fused_env_boxed_rescue_handles_nested_final_answer():
    raw = r"<think>Candidate \boxed{B}</think> Final: \boxed{\text{A}}"
    assert FusedEnv._extract_boxed_from_raw(raw) == r"\text{A}"


def test_fused_env_answer_tag_rescue_uses_last_answer_after_thinking():
    raw = "<think><answer>A</answer></think> Explanation.\n<answer>B</answer>"
    assert FusedEnv._extract_answer_tag_from_raw(raw) == "B"


def test_fused_env_answer_marker_rescue_handles_answer_inside_final_think_block():
    raw = "<think>Long reasoning with an earlier \\boxed{A}.\nFinal choice is C.\n<answer>C</answer></think>"
    assert FusedEnv._extract_final_answer_marker_from_raw(raw) == "C"


def test_fused_env_final_answer_marker_uses_last_marker_not_boxed_priority():
    raw = "Reasoning mentions \\boxed{A}. After checking, <answer>B</answer>"
    assert FusedEnv._extract_final_answer_marker_from_raw(raw) == "B"


def test_fused_env_answer_marker_ignores_unclosed_instruction_tag():
    raw = 'Do not put <answer> or \\boxed{} anywhere except the final line. Final result.\n<answer>A</answer>'
    assert FusedEnv._extract_final_answer_marker_from_raw(raw) == "A"


def test_prompt_only_harness_suppresses_parser_unknown_reward_metadata():
    env = FusedEnv.from_dict({"question": "Q?", "answer": "A", "harness": "cot"})
    env._reset_search()
    env._search_unknown_total = 3
    env._search_answer = "B"

    env._compute_search_reward()

    assert "parser/unknown_total" not in env.reward_debug


def test_cot_plain_text_answer_is_implicit_search_submission():
    env = FusedEnv.from_dict({"question": "What is 2+2?", "answer": "4", "harness": "cot"})
    env._reset_search()

    obs, reward, done, info = env._step_search("The answer is 4.")
    final_reward = env._compute_search_reward()

    assert obs == "Your answer has been submitted."
    assert reward == 0.0
    assert done is True
    assert info == {}
    assert final_reward == 1.0
    assert env.reward_debug["implicit_text_submission"] is True
    assert env.reward_debug["extracted_answer"] == "4"
    assert env.reward_debug["reward/bypass_penalty"] == 0.0


def test_cot_final_reward_rescues_empty_runtime_submission_from_last_action():
    env = FusedEnv.from_dict(
        {
            "question": "Pick one.\nA. alpha\nB. beta\nC. gamma\nD. delta",
            "answer": "C",
            "data_source": "gpqa_diamond",
            "harness": "cot",
        }
    )
    env._reset_search()
    env._search_last_raw_action = "Reasoning text. Therefore the answer is \\boxed{C}."

    final_reward = env._compute_search_reward()

    assert final_reward == 1.0
    assert env._search_answer == "C"
    assert env.reward_debug["extracted_answer"] == "C"
    assert env.reward_debug["implicit_text_submission"] is False


def test_cot_boxed_answer_not_misparsed_as_empty_finish():
    env = FusedEnv.from_dict({"question": "Pick one.\nA. d\nB. a\nC. b\nD. c", "answer": "D", "harness": "cot"})
    env._reset_search()

    raw = "The correct planet is c, which corresponds to option D.\n\n<answer>D</answer>\n\\boxed{D}"
    obs, reward, done, info = env._step_search(raw)
    final_reward = env._compute_search_reward()

    assert obs == "Your answer has been submitted."
    assert reward == 0.0
    assert done is True
    assert info == {}
    assert env._search_answer == "D"
    assert env._search_answer_is_verbatim_submission is True
    assert final_reward == 1.0
    assert env.reward_debug["extracted_answer"] == "D"


def test_cot_answer_inside_final_think_block_scores_correctly():
    env = FusedEnv.from_dict({"question": "Pick one.\nA. foo\nB. bar\nC. baz", "answer": "C", "harness": "cot"})
    env._reset_search()

    raw = "<think>Reasoning that stays open until the final line.\n<answer>C</answer></think>"
    obs, reward, done, info = env._step_search(raw)
    final_reward = env._compute_search_reward()

    assert obs == "Your answer has been submitted."
    assert reward == 0.0
    assert done is True
    assert info == {}
    assert env._search_answer == "C"
    assert final_reward == 1.0
    assert env.reward_debug["extracted_answer"] == "C"


def test_cot_answer_inside_think_block_survives_truncated_post_think():
    env = FusedEnv.from_dict({"question": "Pick one.\nA. foo\nB. bar\nC. baz\nD. qux", "answer": "D", "harness": "cot"})
    env._reset_search()

    raw = (
        "<think>The final choice is D.\n<answer>D</answer></think>\n"
        "I will now write a clean solution, but it is truncated before the final marker."
    )
    obs, reward, done, info = env._step_search(raw)
    final_reward = env._compute_search_reward()

    assert obs == "Your answer has been submitted."
    assert reward == 0.0
    assert done is True
    assert info == {}
    assert env._search_answer == "D"
    assert final_reward == 1.0
    assert env.reward_debug["extracted_answer"] == "D"


def test_cot_answer_tag_submission_avoids_repetition_penalty_on_full_reasoning():
    env = FusedEnv.from_dict({"question": "Pick one.\nA. foo\nB. bar", "answer": "B", "harness": "cot"})
    env._reset_search()

    raw = " ".join(["Reasoning repeats."] * 200) + "\n<answer>B</answer>"
    _, _, done, _ = env._step_search(raw)
    final_reward = env._compute_search_reward()

    assert done is True
    assert env._search_answer == "B"
    assert final_reward == 1.0
    assert env.reward_debug["repetition_penalty_reward"] == 0.0


def test_gem_search_reward_does_not_penalize_finish_without_search():
    env = FusedEnv.from_dict({"question": "What is 2+2?", "answer": "4", "harness": "gem"})
    env._reset_search()
    env._search_answer = "4"
    env._search_answer_is_verbatim_submission = True
    env._search_web_search_calls = 0

    final_reward = env._compute_search_reward()

    assert final_reward == 1.0
    assert env.reward_debug["reward/bypass_penalty"] == 0.0


def test_fused_search_reward_disables_repetition_and_length_penalties():
    env = FusedEnv.from_dict({"question": "Name the capital of France.", "answer": "Paris", "harness": "cot"})
    env._reset_search()
    env._search_answer = " ".join(["London"] * 500)
    env._search_answer_is_verbatim_submission = True

    final_reward = env._compute_search_reward()

    assert final_reward == 0.0
    assert env.reward_debug["repetition_penalty_reward"] == 0.0
    assert env.reward_debug["length_penalty_reward"] == 0.0


def test_cot_placeholder_answer_marker_can_still_score_full_reward():
    env = FusedEnv.from_dict(
        {
            "question": "Which particle is not a Goldstone mode?\nA. Magnon\nB. Phonon\nC. Pion\nD. Skyrmion",
            "answer": "D",
            "harness": "cot",
            "data_source": "gpqa_diamond",
        }
    )
    env._reset_search()

    raw = (
        " ".join(["Magnons are a distractor."] * 150)
        + "\nThus, the particle not associated with this mechanism is the Skyrmion.\n"
        "<answer>LETTER</answer>"
    )
    _, _, done, _ = env._step_search(raw)
    final_reward = env._compute_search_reward()

    assert done is True
    assert final_reward == 1.0
    assert env.reward_debug["extracted_answer"] == "D"


def test_gem_plain_text_still_requires_tool_call():
    env = FusedEnv.from_dict({"question": "What is 2+2?", "answer": "4", "harness": "gem"})
    env._reset_search()

    obs, reward, done, info = env._step_search("The answer is 4.")

    assert obs == "Error: could not parse any actions from model output."
    assert reward == 0.0
    assert done is False
    assert info == {}


def test_gem_finish_without_search_terminates_as_credit_assigned_bypass():
    env = FusedEnv.from_dict({"question": "What is 2+2?", "answer": "4", "harness": "gem"})
    env._reset_search()

    obs, reward, done, info = env._step_search('<tool_call>{"name":"finish","arguments":{"result":"4"}}</tool_call>')

    assert done is True
    assert reward == 0.0
    assert "before web_search" in obs
    assert info["termination_reason"] == "ABNORMAL_SEARCH_BYPASS"
    assert info["credit_assignment"] == "reasoning_step_only"
    assert env._search_answer == "4"


def test_gem_explicit_answer_without_search_terminates_as_credit_assigned_bypass():
    env = FusedEnv.from_dict({"question": "What is 2+2?", "answer": "4", "harness": "gem"})
    env._reset_search()

    obs, reward, done, info = env._step_search("Reasoning complete.\n<answer>4</answer>")

    assert done is True
    assert reward == 0.0
    assert "explicit answer marker before web_search" in obs
    assert info["termination_reason"] == "ABNORMAL_SEARCH_BYPASS"
    assert info["credit_assignment"] == "reasoning_step_only"
    assert env._search_answer == "4"


def test_tool_harness_keeps_parser_unknown_reward_metadata():
    env = FusedEnv.from_dict({"question": "Q?", "answer": "A", "harness": "gem"})
    env._reset_search()
    env._search_unknown_total = 3
    env._search_answer = "B"

    env._compute_search_reward()

    assert env.reward_debug["parser/unknown_total"] == 3
