import json

import hydra
from datasets import load_dataset

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
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

        print(split_data)

        processed = [
            {
                "question": example["extra_info.question"],
                "ground_truth": example["gt_answer"],
                "data_source": "gem_search"
            }
            for example in split_data
        ]

        print(f"Processed {len(processed)} examples")
        return processed

    print("Loading GEM search dataset...")
    with open("experiments/artifacts/search_data_processed.json", "r") as f:
        data = json.load(f)

    train_data = [example for example in data if example.get("extra_info.split") == "train"]
    test_data = [example for example in data if example.get("extra_info.split") == "test"]

    print(f"Found {len(train_data)} training examples and {len(test_data)} test examples")

    train_processed = process_split(train_data, train_size)
    test_processed = process_split(test_data, test_size)

    train_dataset = DatasetRegistry.register_dataset("gem_search", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("gem_search", test_processed, "test")

    return train_dataset, test_dataset


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    # val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    train_dataset, _ = prepare_gem_search_data()
    _, val_dataset = prepare_hotpotqa_data()

    tool_map = {"local_search": LocalRetrievalTool}

    env_args = {
        "max_steps": 20,
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
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
