from types import SimpleNamespace

import numpy as np
import torch
from verl import DataProto

from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer


def _non_contiguous_tensor(shape, dtype=torch.long):
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.arange(numel, dtype=dtype).reshape(shape[1], shape[0]).contiguous().t()


def test_pad_dataproto_to_world_size_returns_contiguous_tensors_without_padding():
    trainer = object.__new__(AgentWorkflowPPOTrainer)
    trainer.use_critic = False
    trainer.use_reference_policy = False
    trainer.use_rm = False
    trainer.hybrid_engine = True
    trainer.actor_rollout_wg = SimpleNamespace(world_size=2)

    tensors = {
        "input_ids": _non_contiguous_tensor((4, 6)),
        "attention_mask": _non_contiguous_tensor((4, 6)),
        "position_ids": _non_contiguous_tensor((4, 6)),
        "prompts": _non_contiguous_tensor((4, 3)),
        "responses": _non_contiguous_tensor((4, 3)),
        "response_mask": _non_contiguous_tensor((4, 3)),
    }
    assert not tensors["position_ids"].is_contiguous()

    batch = DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "is_pad_step": np.array([False, False, False, False]),
            "is_last_step": np.array([True, True, True, True]),
            "is_valid": np.array([True, True, True, True]),
        },
    )

    padded = trainer._pad_dataproto_to_world_size(batch)

    assert padded.batch["position_ids"].is_contiguous()
    assert all(tensor.is_contiguous() for tensor in padded.batch.values())
    assert padded.non_tensor_batch["is_valid"].tolist() == [True, True, True, True]
