import json

import hydra
from datasets import load_dataset

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import create_search_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from .local_retrieval_tool import LocalRetrievalTool


def prepare_hotpotqa_data(train_size=None, test_size=None):
    """
    Loading HotpotQA dataset and registering it with the DatasetRegistry.
    Only loads essential fields: question, ground_truth, data_source

    Args:
        train_size: Maximum number of training examples to load
        test_size: Maximum number of test examples to load

    Returns:
        tuple: (train_dataset, test_dataset)
    """

    def process_split(split_data, max_size):
        """Process a data split with optional size limit"""
        if max_size is not None:
            split_data = split_data.select(range(min(max_size, len(split_data))))
        print(split_data)
        processed = [{"question": example["question"], "ground_truth": example["answer"], "data_source": "hotpotqa"} for example in split_data]

        print(f"Processed {len(processed)} examples")
        return processed

    print("Loading HotpotQA dataset...")
    # Clear corrupted cache before loading
    import shutil
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "hotpotqa___hotpot_qa"
    if cache_dir.exists():
        print(f"Removing corrupted cache at {cache_dir}")
        shutil.rmtree(cache_dir)

    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    train_processed = process_split(hotpot_dataset["train"], train_size)
    test_processed = process_split(hotpot_dataset["validation"], test_size)

    train_dataset = DatasetRegistry.register_dataset("hotpotqa", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("hotpotqa", test_processed, "test")
    return train_dataset, test_dataset


def prepare_gem_search_data(train_size=None, test_size=None):
    """
    Loading gem search dataset and registering it with the DatasetRegistry.
    Only loads essential fields: question, ground_truth, data_source

    Args:
        train_size: Maximum number of training examples to load
        test_size: Maximum number of test examples to load

    Returns:
        tuple: (train_dataset, test_dataset)
    """

    def process_split(split_data, max_size):
        """Process a data split with optional size limit"""
        if max_size is not None:
            split_data = split_data.select(range(min(max_size, len(split_data))))

        # print(split_data)

        processed = [{"question": example["extra_info.question"], "ground_truth": example["gt_answer"], "data_source": "gem_search"} for example in split_data]

        print(f"Processed {len(processed)} examples")
        return processed

    print("Loading GEM search dataset...")
    # v1
    # with open("experiments/artifacts/search_data_20260110/search_data_processed.json", "r") as f:
    # v2
    # with open("experiments/artifacts/search_data_20260110/search_data_processed.json", "r") as f:
    # data = json.load(f)
    # v3
    # with open("experiments/artifacts/search_data_20260119/search_data_processed_v3.json", "r") as f:
    # data = json.load(f)
    # v3.1
    # with open("/share/nlp/liuyang/workspace/gem/rllm/experiments/artifacts/search_data_20260120/search_data_processed_v3.json", "r") as f:
        # data = json.load(f)
    # ASearcher (Baseline)
    with open("/share/nlp/liuyang/workspace/gem/rllm/experiments/artifacts/ASearcher/ASearcher.json", "r") as f:
        data = json.load(f)

    train_data = [example for example in data if example.get("extra_info.split") == "train"]
    # test_data = [example for example in data if example.get("extra_info.split") == "test"]
    test_data = train_data

    print(f"Found {len(train_data)} training examples and {len(test_data)} test examples")

    train_processed = process_split(train_data, train_size)
    test_processed = process_split(test_data, test_size)

    train_dataset = DatasetRegistry.register_dataset("gem_search", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("gem_search", test_processed, "test")

    return train_dataset, test_dataset


def _patch_vllm_generate():
    """
    Monkey patch the vLLM server's generate method to respect _override_max_tokens.
    This prevents negative max_tokens errors when prompts are close to max_model_len.
    """
    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

        original_generate = vLLMHttpServer.generate

        async def patched_generate(self, prompt_ids, sampling_params, request_id, image_data=None):
            """Patched generate that respects _override_max_tokens from sampling_params."""
            # Check if we have an override value
            override_max_tokens = sampling_params.pop("_override_max_tokens", None)

            if override_max_tokens is not None:
                # Use the override value instead of recalculating
                # max_tokens = override_max_tokens
                print(f"Using override max_tokens: {max_tokens} (prompt_length: {len(prompt_ids)})")
                max_tokens = 60000
            else:
                # Original calculation
                max_tokens = self.config.max_model_len - len(prompt_ids)
                max_tokens = 60000

            # Ensure max_tokens is at least 1
            if max_tokens < 1:
                print(f"Warning: Calculated max_tokens ({max_tokens}) is less than 1. " f"Setting to 1. (prompt_length: {len(prompt_ids)}, max_model_len: {self.config.max_model_len})")
                max_tokens = 1

            # Continue with the rest of the original method
            from vllm.sampling_params import SamplingParams
            from verl.workers.rollout.vllm_rollout.vllm_async_server import _qwen2_5_vl_dedup_image_tokens, VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH
            from vllm import LoRARequest
            from vllm.inputs import TokensPrompt
            from verl.workers.rollout.replica import TokenOutput

            sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
            sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
            sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
            prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
            prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None)

            # Add lora request
            lora_request = None
            if self.model_config.lora_rank > 0:
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
                if lora_loaded:
                    lora_request = LoRARequest(lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)

            generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request)

            token_ids = []
            log_probs = []
            finish_reason = None

            async for request_output in generator:
                # Process the generator output as in the original method
                if hasattr(request_output, "outputs") and len(request_output.outputs) > 0:
                    output = request_output.outputs[0]
                    token_ids = output.token_ids
                    if hasattr(output, "log_probs"):
                        log_probs = [lp[tid].logprob if lp and tid in lp else 0.0 for lp, tid in zip(output.log_probs or [], output.token_ids)]
                    finish_reason = output.finish_reason

            return TokenOutput(token_ids=token_ids, log_probs=log_probs, finish_reason=finish_reason)

        vLLMHttpServer.generate = patched_generate
        print("Successfully patched vLLMHttpServer.generate to respect _override_max_tokens")
    except Exception as e:
        print(f"Warning: Failed to patch vLLMHttpServer.generate: {e}")
        print("Training will continue but may encounter max_tokens errors")


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Apply monkey patch for vLLM server
    _patch_vllm_generate()

    # train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    # val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    # train_dataset, _ = prepare_gem_search_data()
    # _, val_dataset = prepare_hotpotqa_data()
    train_dataset, val_dataset = prepare_hotpotqa_data()

    tool_map = {"local_search": LocalRetrievalTool}

    # Create configurable reward function
    # Get reward config from hydra config if available, otherwise use defaults
    reward_config = config.get("reward", {})
    reward_fn = create_search_reward_fn(
        toolcall_bonus=reward_config.get("toolcall_bonus", 0.5),
        apply_repetition_penalty=reward_config.get("apply_repetition_penalty", True),
        repetition_penalty_weight=reward_config.get("repetition_penalty_weight", 0.5),
        repetition_max_n=reward_config.get("repetition_max_n", 4),
        correct_reward=reward_config.get("correct_reward", 1.0),
        incorrect_reward=reward_config.get("incorrect_reward", 0.0),
    )

    env_args = {
        "max_steps": 32,
        "tool_map": tool_map,
        "reward_fn": reward_fn,
    }

    agent_args = {"system_prompt": SEARCH_SYSTEM_PROMPT, "tool_map": tool_map, "parser_name": "qwen"}

    # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
