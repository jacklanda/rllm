"""Multi-turn agent SFT dataset with assistant-only loss masking.

This is a self-contained variant of ``rllm.trainer.verl.sft_dataset.RLLMSFTDataset``.
It exists in the example dir for two reasons:

1. The shared ``RLLMSFTDataset`` imports the chat-template parser from
   ``verl.utils.parser.chat_template_parser``, which does not exist in the installed
   verl in this environment. The parser actually lives at
   ``rllm.parser.chat_template_parser``. We import the correct path here.
2. ``verl.trainer.fsdp_sft_trainer.create_sft_dataset`` instantiates a custom dataset
   class with a ``max_samples`` kwarg, so we forward it to the base class.

Masking semantics ("cumulative" method):
  - Every assistant message (including its inline ``<think>...</think>`` reasoning and
    serialized tool calls) is rendered and its tokens get ``loss_mask = 1`` -> learning target.
  - The system message (tool specs) and every user message (tool outputs / environment
    observations) get ``loss_mask = 0`` -> masked out of the loss.

Each message is rendered individually with ``add_generation_prompt=False``. This is robust
to trajectories that contain consecutive user messages (back-to-back tool observations),
which the agent rollouts in this project produce.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.dataset.multiturn_sft_dataset import convert_nested_value_to_list_recursive

from rllm.parser.chat_template_parser import ChatTemplateParser

try:
    from agent_sft_data_utils import records_from_json
except ImportError:  # pragma: no cover - supports package-style imports in tests
    from examples.sft_agent.agent_sft_data_utils import records_from_json

logger = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


class AgentSFTDataset(MultiTurnSFTDataset):
    def __init__(self, parquet_files: str | list[str], tokenizer, config=None, max_samples: int = -1):
        self._agent_config = config if config is not None else {}
        super().__init__(parquet_files, tokenizer, config, max_samples=max_samples)

        # "cumulative" trains on all assistant turns; "stepwise" trains only on the last one.
        rllm_cfg = config.get("rllm", {}) if config is not None else {}
        self.tokenize_and_mask_method = rllm_cfg.get("tokenize_and_mask_method", "cumulative")
        logger.info(f"Using {self.tokenize_and_mask_method} tokenization and masking method")

        self.parser = ChatTemplateParser.get_parser(tokenizer)

    def _load_dataframe(self, data_file):
        path = Path(str(data_file))
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".json", ".jsonl"}:
            rllm_cfg = self._agent_config.get("rllm", {})
            reward_threshold = rllm_cfg.get("sft_reward_threshold", 0.0)
            records, skipped = records_from_json(path, reward_threshold=reward_threshold)
            if _is_rank_zero():
                logger.info(
                    "Converted %s usable SFT samples from %s; skipped=%s",
                    len(records),
                    path,
                    dict(skipped),
                )
            return pd.DataFrame(records)
        raise ValueError(f"Unsupported AgentSFTDataset file type for {path}. Expected .parquet, .json, or .jsonl")

    def _read_files_and_process(self):
        def series_to_item(ls):
            while isinstance(ls, pd.core.series.Series | np.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = [self._load_dataframe(data_file) for data_file in self.parquet_files]
        self.dataframe = pd.concat(dataframes)
        if self.messages_key not in self.dataframe.columns:
            raise ValueError(
                f"Loaded Agent SFT data does not contain a '{self.messages_key}' column. "
                "Parquet input must already contain messages; JSON input must contain offline-RS trajectories."
            )

        total = len(self.dataframe)
        if _is_rank_zero():
            logger.info("Loaded %s samples from %s", total, self.parquet_files)

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
                selection = "random"
            else:
                indices = np.arange(self.max_samples)
                selection = "first"
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            if _is_rank_zero():
                logger.info("Selected %s %s samples out of %s", self.max_samples, selection, total)

        self.messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()

        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def _tokenize_and_mask(self, messages):
        if self.tokenize_and_mask_method == "cumulative":
            return self._tokenize_and_mask_cumulative(messages)
        elif self.tokenize_and_mask_method == "stepwise":
            return self._tokenize_and_mask_stepwise(messages)
        else:
            raise ValueError(f"Unknown tokenize_and_mask_method {self.tokenize_and_mask_method}")

    def _tokenize_and_mask_cumulative(self, messages):
        """Train on every assistant turn; mask system + user (observation) turns."""
        tokens = []
        loss_mask = []

        for i in range(len(messages)):
            parsed_msg = self.parser.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            if messages[i]["role"] == "assistant":
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def _tokenize_and_mask_stepwise(self, messages):
        """Train only on the final assistant turn; mask everything else."""
        tokens = []
        loss_mask = []

        last_assistant_idx = -1
        for i in range(len(messages)):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
        assert last_assistant_idx != -1, "No assistant message found in trajectory"

        for i in range(len(messages)):
            parsed_msg = self.parser.parse([messages[i]], is_first_msg=(i == 0), add_generation_prompt=False)
            ids = self.tokenizer.encode(parsed_msg, add_special_tokens=False)
            if i == last_assistant_idx and messages[i]["role"] == "assistant":
                loss_mask.extend([1] * len(ids))
            else:
                loss_mask.extend([0] * len(ids))
            tokens.extend(ids)

        return tokens, loss_mask

    def __getitem__(self, item):
        messages = self.messages[item]

        tokens, loss_mask = self._tokenize_and_mask(messages)

        input_ids = torch.tensor(tokens, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        attention_mask = torch.tensor([1] * len(tokens), dtype=torch.long)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
            padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))

        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
