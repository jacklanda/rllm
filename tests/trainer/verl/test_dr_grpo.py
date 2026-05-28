import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss

from rllm.trainer.verl.ray_trainer import compute_advantage, validate_dr_grpo_config


def _make_grpo_batch():
    return DataProto.from_single_dict(
        {
            "responses": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "token_level_rewards": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ]
            ),
            "uid": np.array(["prompt-a", "prompt-a"], dtype=object),
        }
    )


def test_dr_grpo_advantage_does_not_normalize_by_group_std():
    batch = compute_advantage(_make_grpo_batch(), adv_estimator="dr_grpo")

    expected = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    torch.testing.assert_close(batch.batch["advantages"], expected)
    torch.testing.assert_close(batch.batch["returns"], expected)


def test_grpo_honors_norm_adv_by_std_config():
    batch = compute_advantage(
        _make_grpo_batch(),
        adv_estimator="grpo",
        norm_adv_by_std_in_grpo=False,
    )

    expected = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    torch.testing.assert_close(batch.batch["advantages"], expected)


def test_dr_grpo_loss_uses_fixed_response_width_normalizer():
    loss_mat = torch.ones(2, 4)
    loss_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    loss = agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode="seq-mean-token-sum-norm")

    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_dr_grpo_config_requires_fixed_response_width_loss():
    config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "dr_grpo"},
            "actor_rollout_ref": {"actor": {"loss_agg_mode": "seq-mean-token-mean"}},
        }
    )

    with pytest.raises(ValueError, match="Dr.GRPO requires"):
        validate_dr_grpo_config(config)
