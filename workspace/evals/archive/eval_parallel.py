#!/usr/bin/env python3

import os
import re
import json
import math
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import pandas as pd
import sglang as sgl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, set_seed

from parser import extract_answer, strip_string
from grader import math_equal, compute_format_score

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get the logger for SGLang
sglang_logger = logging.getLogger("sglang")

# Set the logging level to WARNING or higher to suppress INFO messages
sglang_logger.setLevel(logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Math evaluation script with multiverse sampling"
    )

    # Model configuration
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallel size (defaults to available GPU count)",
    )

    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=30000,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=None, help="Repetition penalty"
    )
    parser.add_argument(
        "--parallel_reasoning",
        action="store_true",
        help="Enable parallel reasoning",
    )

    # Batch processing
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_samples", type=int, default=8, help="Number of samples per problem"
    )
    parser.add_argument(
        "--passk", type=int, default=1, help="k value for pass@k calculation"
    )

    # Task configuration
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["AMC23", "AIME24", "AIME25", "MATH500"],
        default=["AMC23", "AIME24", "AIME25", "MATH500"],
        help="Tasks to evaluate on",
    )
    parser.add_argument(
        "--task_dir", type=str, default="bench/", help="Directory containing task data"
    )

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="prediction", help="Output directory"
    )
    parser.add_argument(
        "--save_alias", type=str, default="", help="Alias to append to save path"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing results"
    )

    # Other options
    parser.add_argument(
        "--apply_chat", action="store_true", default=True, help="Apply chat template"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


class TokenizedDataset(Dataset):
    def __init__(self, prompts: List[str], answers: List[str]):
        self.prompts = prompts
        self.answers = answers

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return (self.prompts[idx], self.answers[idx], idx)


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Calculate pass@k metric (commonly used for AIME)"""
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


class DatasetLoader:
    """Handles loading of different mathematical datasets"""

    def __init__(self, task_dir: str):
        self.task_dir = task_dir

    def load_amc23(self) -> Tuple[List[str], List[str]]:
        df = pd.read_parquet(os.path.join(self.task_dir, "AMC23/amc23.parquet"))
        questions = df["question"].tolist()
        answers = [str(a) for a in df["answer"].tolist()]
        return questions, answers

    def load_aime24(self) -> Tuple[List[str], List[str]]:
        df = pd.read_parquet(os.path.join(self.task_dir, "AIME24/aime24.parquet"))
        questions = df["problem"].tolist()
        answers = [
            re.search(r"\\boxed\{(\d+)\}", text).group(1)
            for text in df["solution"].tolist()
        ]
        return questions, answers

    def load_aime25(self) -> Tuple[List[str], List[str]]:
        questions = []
        answers = []
        for file in ["aime2025-I.jsonl", "aime2025-II.jsonl"]:
            filepath = os.path.join(self.task_dir, "AIME25", file)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    sample = json.loads(line)
                    questions.append(sample["question"])
                    answers.append(str(sample["answer"]))
        return questions, answers

    def load_math500(self) -> Tuple[List[str], List[str], List[str]]:
        questions = []
        solutions = []
        answers = []
        filepath = os.path.join(self.task_dir, "MATH500/math500.jsonl")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sample = json.loads(line)
                questions.append(sample["problem"])
                solutions.append(sample["solution"])
                answers.append(str(sample["answer"]))
        return questions, solutions, answers

    def load_dataset(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        """Load dataset by name"""
        if dataset_name == "AMC23":
            return self.load_amc23()
        elif dataset_name == "AIME24":
            return self.load_aime24()
        elif dataset_name == "AIME25":
            return self.load_aime25()
        elif dataset_name == "MATH500":
            questions, _, answers = self.load_math500()
            return questions, answers
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


class PromptBuilder:
    """Handles prompt construction for different models"""

    def __init__(self, tokenizer, model_name: str, apply_chat: bool = True):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.apply_chat = apply_chat

    def build_prompt(self, question: str) -> str:
        """Build prompt for a given question"""
        if not self.apply_chat:
            inst = "Please think step by step and in parallel. Put your final answer within \\boxed{}."
            return f"{inst}\n{question}"

        # Determine system message based on model
        if "Qwen2.5-32B-Instruct" in self.model_name:
            system_content = "Please reason step by step, and put your final answer within \\boxed{}."
        elif "Multiverse-32B" in self.model_name:
            system_content = "Please think step by step and in parallel. Put your final answer within \\boxed{}."
        elif "MathReasoner" in self.model_name:
            system_content = "You are a math reasoning agent. Please reason step by step to solve the math problem in parallel."
        else:
            system_content = "You are a math reasoning agent. Please reason step by step to solve the math problem in parallel."

        msgs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


class ModelEvaluator:
    """Handles model evaluation logic"""

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if args.parallel_reasoning:
            print("==> Parallel reasoning enabled.")
            stop_token_ids = [151670, 151674]  # tokens triggering parallel reasoning
        else:
            stop_token_ids = []

        # Build sampling parameters
        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "skip_special_tokens": False,
            "stop_token_ids": stop_token_ids,
        }
        if args.temperature is not None:
            self.sampling_params["temperature"] = args.temperature
        if args.top_p is not None:
            self.sampling_params["top_p"] = args.top_p
        if args.top_k is not None:
            self.sampling_params["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            self.sampling_params["repetition_penalty"] = args.repetition_penalty

    def extract_boxed_prediction(self, full_pred: str) -> str:
        """Extract boxed prediction from full prediction"""
        pred = re.findall(r"\\boxed\{([^}]*)\}", full_pred)
        if len(pred) > 0:
            pred = pred[0].strip()
        else:
            pos = full_pred.find("</think>")
            pred = (
                full_pred[pos + len("</think>") :].strip() if pos != -1 else full_pred
            )
        return pred

    def generate_with_retry(self, prompts: List[str]) -> List[str]:
        """Generate responses with OOM handling"""
        max_tokens = self.args.max_new_tokens

        while True:
            try:
                with torch.no_grad():
                    generated = self.model.generate(prompts, self.sampling_params)
                return [r["text"] for r in generated]
            except Exception as e:
                print(f"Unexpected error: {e}", flush=True)
                continue

    def evaluate_batch(
        self,
        batch_prompts: List[str],
        batch_answers: List[str],
        batch_indices: List[int],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Evaluate a batch of prompts"""
        results = {}

        # Initialize results
        for i, global_idx in enumerate(batch_indices):
            reference = strip_string(batch_answers[i], skip_unit=False)
            results[global_idx] = {
                "time": datetime.now().isoformat(),
                "prompt": batch_prompts[i],
                "sample_size": self.args.num_samples,
                "full_preds": [],
                "boxed_preds": [],
                "reference": reference,
            }

        # Generate samples
        for sample_id in tqdm(range(self.args.num_samples), desc="Sampling..."):
            set_seed(sample_id)

            full_preds = self.generate_with_retry(batch_prompts)
            boxed_preds = [
                extract_answer(pred, dataset_name, use_last_number=True)
                for pred in full_preds
            ]

            # Store predictions
            for i, global_idx in enumerate(batch_indices):
                results[global_idx]["full_preds"].append(full_preds[i])
                results[global_idx]["boxed_preds"].append(boxed_preds[i])

            # Log full predictions in debug mode
            if self.local_rank == 0 and self.args.debug:
                for i, global_idx in enumerate(batch_indices):
                    print(
                        f"Sample {sample_id} - Problem {global_idx} Prediction: {full_preds[i]}",
                        flush=True,
                    )
                    pass

        # Calculate scores
        for global_idx in results:
            reference = results[global_idx]["reference"]
            preds = results[global_idx]["boxed_preds"]
            full_preds = results[global_idx]["full_preds"]
            full_preds_cleaned = [
                pred.split("<|im_start|>assistant", 1)[-1].strip()
                for pred in full_preds
            ]
            correct_count = sum(
                [math_equal(prediction=pred, reference=reference) for pred in preds]
            )
            avg_format_score = sum(
                [compute_format_score(pred) for pred in full_preds_cleaned]
            ) / len(full_preds_cleaned)
            results[global_idx]["correct_count"] = correct_count
            passk_score = estimate_pass_at_k(
                self.args.num_samples, correct_count, self.args.passk
            )
            avg_completed_tokens = sum(
                [len(self.tokenizer.encode(pred)) for pred in full_preds_cleaned]
            ) / len(full_preds_cleaned)
            input_tokens = len(self.tokenizer.encode(results[global_idx]["prompt"]))
            results[global_idx][f"pass@{self.args.passk}"] = passk_score
            results[global_idx]["format_score"] = avg_format_score
            results[global_idx]["input_tokens"] = input_tokens
            results[global_idx]["completed_tokens"] = avg_completed_tokens

            if self.local_rank == 0 and self.args.debug:
                print(
                    f"Problem {global_idx}: avg pass@1: {passk_score:.3f}; avg format score: {avg_format_score:.3f}",
                    flush=True,
                )

        return results


def initialize_model_and_tokenizer(args):
    """Initialize model and tokenizer"""
    print(f"==> Loading model/tokenizer from {args.model_path}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left", use_fast=False
    )

    if hasattr(config, "max_position_embeddings"):
        print(f"==> max position embeddings: {config.max_position_embeddings}")
    else:
        print(f"==> max position embeddings: {config.model_max_length}")

    tp_size = args.tp_size if args.tp_size else torch.cuda.device_count()
    model = sgl.Engine(
        model_path=args.model_path,
        tp_size=tp_size,
        disable_overlap_schedule=True,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
    )

    return model, tokenizer


def evaluate_single_dataset(
    dataset_name: str,
    args,
    model,
    tokenizer,
    dataset_loader: DatasetLoader,
    prompt_builder: PromptBuilder,
    evaluator: ModelEvaluator,
):
    """Evaluate a single dataset"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup output path
    out_dir = os.path.join(
        args.output_dir,
        os.path.basename(args.model_path) + args.save_alias,
        "zero_shot",
    )
    if local_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"{dataset_name}_{args.num_samples}.json")
    if not args.overwrite and os.path.exists(output_path):
        print(f"==> {output_path} already exists, skipping...")
        return None

    # Load dataset
    print(f"==> Loading {dataset_name} dataset...")
    questions, targets = dataset_loader.load_dataset(dataset_name)

    # Build prompts
    print(f"==> Building prompts for {len(questions)} problems...")
    prompts = []
    answers = []
    for question, target in tqdm(zip(questions, targets), desc="Building prompts"):
        prompt = prompt_builder.build_prompt(question)
        prompts.append(prompt)
        answers.append(str(target))

    # Create dataset and dataloader
    dataset = TokenizedDataset(prompts, answers)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    all_results = {}
    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        batch_prompts, batch_answers, batch_indices = batch
        batch_indices = [idx.item() for idx in batch_indices]

        batch_prompts = list(batch_prompts)
        batch_results = evaluator.evaluate_batch(
            batch_prompts, batch_answers, batch_indices, dataset_name
        )
        all_results.update(batch_results)

        # checkpoint intermediate results
        if local_rank == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            print(f'==> Intermediate results saved to "{output_path}".')

    # Calculate final score
    final_score = sum(
        [all_results[k][f"pass@{args.passk}"] for k in all_results]
    ) / len(dataset)
    print(
        f"==> {os.path.basename(args.model_path)} {dataset_name} pass@{args.passk}: {final_score * 100:.2f}%"
    )
    final_format_score = sum(
        [all_results[k]["format_score"] for k in all_results]
    ) / len(dataset)
    print(
        f"==> {os.path.basename(args.model_path)} {dataset_name} avg format score: {final_format_score:.2f}"
    )

    # Save results
    if local_rank == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print(f"==> Results saved to {output_path}")

    return final_score


def main():
    """Main evaluation function"""
    sgl.set_default_backend("vllm")

    # Parse arguments
    args = parse_arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize components
    model, tokenizer = initialize_model_and_tokenizer(args)
    dataset_loader = DatasetLoader(args.task_dir)
    prompt_builder = PromptBuilder(
        tokenizer, os.path.basename(args.model_path), args.apply_chat
    )
    evaluator = ModelEvaluator(model, tokenizer, args)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each dataset
    results_summary = {}
    for dataset_name in args.tasks:
        print(f"\n{'=' * 50}")
        print(f"Starting evaluation on {dataset_name}")
        print(f"{'=' * 50}")

        score = evaluate_single_dataset(
            dataset_name,
            args,
            model,
            tokenizer,
            dataset_loader,
            prompt_builder,
            evaluator,
        )

        if score is not None:
            results_summary[dataset_name] = score

    # Print final summary
    if local_rank == 0 and results_summary:
        print(f"\n{'=' * 50}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 50}")
        for dataset_name, score in results_summary.items():
            print(f"{dataset_name:10}: {score * 100:6.2f}%")
        print(f"{'=' * 50}")
        print("==> Evaluation completed!")


if __name__ == "__main__":
    main()
