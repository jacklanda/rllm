"""FSDP2-based agent SFT training entrypoint.

Adapted from ``examples/sft/train_math_sft.py`` and verl's ``run_sft``.

NOTE: we intentionally do NOT use ``rllm.trainer.agent_sft_trainer.AgentSFTTrainer``.
Its ``_train_verl`` path hardcodes ``rllm.trainer.verl.sft_dataset.RLLMSFTDataset``,
which imports the chat-template parser from ``verl.utils.parser.chat_template_parser`` --
a module that does not exist in the installed verl in this environment (it lives at
``rllm.parser.chat_template_parser``), so importing it raises ModuleNotFoundError and the
``data.custom_cls`` override is ignored.

Instead we replicate verl's ``run_sft`` here and go through ``create_sft_dataset``, which
honors ``data.custom_cls`` and loads our self-contained
``examples/sft_agent/agent_sft_dataset.py::AgentSFTDataset`` (assistant-only loss masking).
The training loop is still verl's ``FSDPSFTTrainer`` with ``model.strategy=fsdp2``.
"""

import os
from datetime import timedelta

import hydra
import torch
import torch.distributed
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

import verl.trainer.fsdp_sft_trainer as fsdp_sft_trainer
import verl.utils.checkpoint.checkpoint_manager as checkpoint_manager
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.distributed import destroy_global_process_group
from verl.utils.fs import copy_to_local

from rllm.trainer.sft_metrics import normalize_sft_lr_metrics


class RLLMFSDPSFTTrainer(FSDPSFTTrainer):
    def training_step(self, batch):
        return normalize_sft_lr_metrics(super().training_step(batch))


def initialize_global_process_group(timeout_second=36000):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device_name = get_device_name()
    device_id = None
    if device_name != "cpu":
        get_torch_device().set_device(local_rank)
        device_id = torch.device(device_name, local_rank)

    torch.distributed.init_process_group(
        get_nccl_backend(),
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
        device_id=device_id,
    )
    return local_rank, rank, world_size


def validate_effective_dataset_size(config: DictConfig, train_dataset, val_dataset, dp_size: int):
    per_dp_batch_size = config.data.train_batch_size // dp_size
    per_dp_samples = len(train_dataset) // dp_size
    if per_dp_samples < per_dp_batch_size:
        raise ValueError(
            "Effective train dataset is too small for FSDPSFTTrainer with drop_last=True: "
            f"{len(train_dataset)=}, {dp_size=}, {per_dp_batch_size=}. "
            "Increase data.train_max_samples or reduce data.train_batch_size."
        )

    per_dp_val_samples = len(val_dataset) // dp_size
    if per_dp_val_samples < config.data.micro_batch_size_per_gpu:
        raise ValueError(
            "Effective validation dataset is too small for FSDPSFTTrainer with drop_last=True: "
            f"{len(val_dataset)=}, {dp_size=}, micro_batch_size_per_gpu={config.data.micro_batch_size_per_gpu}. "
            "Increase data.val_max_samples or reduce data.micro_batch_size_per_gpu."
        )


def patch_checkpoint_logging():
    def find_latest_ckpt_path(path, directory_format="global_step_{}"):
        if path is None:
            return None

        tracker_file = checkpoint_manager.get_checkpoint_tracker_filename(path)
        if not os.path.exists(tracker_file):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"Checkpoint tracker file does not exist: {tracker_file}")
            return None

        with open(tracker_file, "rb") as f:
            iteration = int(f.read().decode())

        ckpt_path = os.path.join(path, directory_format.format(iteration))
        if not os.path.exists(ckpt_path):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"Checkpoint does not exist: {ckpt_path}")
            return None

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Found checkpoint: {ckpt_path}")
        return ckpt_path

    checkpoint_manager.find_latest_ckpt_path = find_latest_ckpt_path
    fsdp_sft_trainer.find_latest_ckpt_path = find_latest_ckpt_path


def normalize_data_paths(data_paths):
    """Accept comma-separated path strings as a shorthand for a list of data files."""
    if isinstance(data_paths, str) and "," in data_paths:
        paths = [path.strip() for path in data_paths.split(",") if path.strip()]
        if not paths:
            raise ValueError(f"No valid data paths found in {data_paths!r}")
        return paths
    return data_paths


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_sft_trainer", version_base=None)
def main(config: DictConfig):
    patch_checkpoint_logging()

    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    try:
        device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        dp_size = world_size // config.ulysses_sequence_parallel_size
        ulysses_device_mesh = init_device_mesh(
            device_type=device_name,
            mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
            mesh_dim_names=("dp", "sp"),
        )

        local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

        train_dataset = create_sft_dataset(
            normalize_data_paths(config.data.train_files),
            config.data,
            tokenizer,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_sft_dataset(
            normalize_data_paths(config.data.val_files),
            config.data,
            tokenizer,
            max_samples=config.data.get("val_max_samples", 128),
        )
        validate_effective_dataset_size(config, train_dataset, val_dataset, dp_size)

        trainer = RLLMFSDPSFTTrainer(
            config=config,
            device_mesh=device_mesh,
            ulysses_device_mesh=ulysses_device_mesh,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer.fit()
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
