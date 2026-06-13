from rllm.rewards.reward_types import RewardConfig, RewardInput
from rllm.rewards.search_reward import RewardSearchFn


def _score(action, ground_truth, **task_info):
    fn = RewardSearchFn(RewardConfig())
    return fn(RewardInput(task_info={"ground_truth": ground_truth, **task_info}, action=action))


def test_token_overlap_gets_no_reward_without_exact_match():
    result = _score("Latin", "Medieval Latin", is_submitted=True)

    assert result.reward == 0.0
    assert result.is_correct is False
    assert result.metadata["f1_score"] > 0.0
    assert result.metadata["exact_match"] is False
    assert result.metadata["partial_match_accepted"] is False
    assert result.metadata["partial_match_reject_reason"] == "substring_entity_mismatch"


def test_verbose_answer_containing_gold_gets_no_reward():
    result = _score(
        "The histone modification is histone H4 lysine 16 acetylation, also known as H4K16ac.",
        "H4K16ac",
        is_submitted=True,
    )

    assert result.reward == 0.0
    assert result.is_correct is False
    assert result.metadata["f1_score"] > 0.0
    assert result.metadata["exact_match"] is False


def test_exact_match_gets_full_reward_without_step_bonus():
    config = RewardConfig(
        enable_step_bonus=True,
        apply_repetition_penalty=True,
        repetition_penalty_weight=0.5,
        apply_length_penalty=True,
        length_penalty_weight=0.5,
    )
    fn = RewardSearchFn(config)

    result = fn(
        RewardInput(
            task_info={"ground_truth": "Quantum hacking", "step_count": 20, "is_submitted": True},
            action="Quantum hacking",
        )
    )

    assert result.reward == 1.0
    assert result.is_correct is True
    assert result.metadata["exact_match"] is True
    assert result.metadata["step_bonus"] == 0.0
    assert result.metadata["repetition_penalty_reward"] == 0.0
    assert result.metadata["length_penalty_reward"] == 0.0


def test_non_exact_reward_stays_zero_even_with_shaping_config():
    config = RewardConfig(
        enable_step_bonus=True,
        apply_repetition_penalty=True,
        repetition_penalty_weight=0.5,
        apply_length_penalty=True,
        length_penalty_weight=0.5,
    )
    fn = RewardSearchFn(config)

    result = fn(
        RewardInput(
            task_info={"ground_truth": "Quantum hacking", "step_count": 20, "is_submitted": True},
            action="This is quantum hacking.",
        )
    )

    assert result.reward == 0.0
    assert result.is_correct is False
    assert result.metadata["f1_score"] > 0.0
    assert result.metadata["exact_match"] is False
