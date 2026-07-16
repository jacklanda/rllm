#!/usr/bin/env python3

import os
import re
import json
import math
import time
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import sglang
import numpy as np
import pandas as pd
import sglang.srt.entrypoints.engine
import ray
from tqdm import tqdm
from rich.table import Table
from rich.syntax import Syntax
from rich.status import Status
from rich.console import Console
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, set_seed

from utils import (
    math_equal,
    strip_string,
    extract_answer,
    load_instruction,
    compute_format_score,
    parse_math_arena_answer,
    extract_math_arena_answer,
)

# from parser import extract_answer, strip_string
# from grader import math_equal, compute_format_score
# from math_arena_parser import extract_answer as extract_math_arena_answer
# from math_arena_parser import parse_answer as parse_math_arena_answer

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get the logger for SGLang
# sglang_logger = logging.getLogger("sglang")

# Set the logging level to WARNING or higher to suppress INFO messages
# sglang_logger.setLevel(logging.WARNING)

console = Console()


class InferenceTimer:
    """Helper class for tracking inference timing and throughput"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_tokens = 0
        self.total_samples = 0
        self.batch_times = []

    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        console.log(f"🚀 Starting inference at {datetime.now().strftime('%H:%M:%S')}")

    def end(self):
        """End the timer"""
        self.end_time = time.time()

    def add_batch_stats(self, tokens: int, samples: int, batch_time: float = None):
        """Add statistics for a batch"""
        self.total_tokens += tokens
        self.total_samples += samples
        if batch_time is not None:
            self.batch_times.append(batch_time)

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        current_time = self.end_time if self.end_time else time.time()
        return current_time - self.start_time

    def get_throughput_stats(self) -> Dict[str, float]:
        """Calculate throughput statistics"""
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return {"elapsed_time": 0, "tokens_per_second": 0, "samples_per_second": 0}

        return {
            "elapsed_time": elapsed,
            "tokens_per_second": self.total_tokens / elapsed,
            "samples_per_second": self.total_samples / elapsed,
            "total_tokens": self.total_tokens,
            "total_samples": self.total_samples,
        }

    def print_progress(self, current_sample: int = None, total_samples: int = None):
        """Print current progress with timing info"""
        elapsed = self.get_elapsed_time()
        stats = self.get_throughput_stats()

        elapsed_str = f"{elapsed:.1f}s"
        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            elapsed_str = f"{minutes}m {seconds}s"

        progress_str = ""
        if current_sample is not None and total_samples is not None:
            progress_str = f"[{current_sample}/{total_samples}] "

        console.log(
            f"⏱️  {progress_str}Elapsed: {elapsed_str} | "
            # f"Throughput: {stats['tokens_per_second']:.1f} tok/s, {stats['samples_per_second']:.1f} samples/s"
        )

    def print_final_stats(self):
        """Print final timing and throughput statistics"""
        if self.start_time is None:
            return

        stats = self.get_throughput_stats()
        elapsed = stats["elapsed_time"]

        # Format elapsed time
        if elapsed >= 3600:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            elapsed_str = f"{hours}h {minutes}m {seconds}s"
        elif elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            elapsed_str = f"{minutes}m {seconds}s"
        else:
            elapsed_str = f"{elapsed:.1f}s"

        console.rule("🏁 Inference Complete")
        console.log(f"⏱️  Total time: {elapsed_str}")
        # console.log(f"🔢 Total tokens generated: {stats['total_tokens']:,}")
        # console.log(f"📊 Total samples processed: {stats['total_samples']:,}")
        # console.log(
        # f"🚀 Average throughput: {stats['tokens_per_second']:.1f} tokens/sec"
        # )
        # console.log(
        # f"📈 Average sample rate: {stats['samples_per_second']:.1f} samples/sec"
        # )

        if self.batch_times:
            avg_batch_time = np.mean(self.batch_times)
            console.log(f"⏳ Average batch time: {avg_batch_time:.2f}s")
        console.rule()


def validate_args(args):
    """Validate command line arguments"""
    if args.mode == "dp":
        if args.dp_size and args.dp_size > torch.cuda.device_count():
            raise ValueError(
                f"dp_size ({args.dp_size}) cannot exceed available GPUs ({torch.cuda.device_count()})"
            )
        if args.dp_size and args.dp_size < 1:
            raise ValueError("dp_size must be at least 1")

    if args.mode == "tp":
        if args.tp_size and args.tp_size > torch.cuda.device_count():
            raise ValueError(
                f"tp_size ({args.tp_size}) cannot exceed available GPUs ({torch.cuda.device_count()})"
            )
        if args.tp_size and args.tp_size < 1:
            raise ValueError("tp_size must be at least 1")

    if args.mode == "ray":
        if args.dp_size and args.dp_size < 1:
            raise ValueError("dp_size must be at least 1 for Ray mode")

    return args


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Math evaluation script with multiverse sampling"
    )

    # Model configuration
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Instruction file for prompt construction",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tp", "dp", "ray"],
        default="tp",
        help="Parallelism mode: 'tp' for tensor parallelism, 'dp' for data parallelism, 'ray' for Ray-based async engines",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallel size for 'tp' mode (defaults to available GPU count)",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=None,
        help="Number of parallel engines for 'dp' mode (defaults to available GPU count)",
    )
    parser.add_argument(
        "--dp_async",
        action="store_true",
        help="Use async evaluation for data parallelism mode for better concurrency",
    )
    parser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=None,
        help="Memory fraction per engine (default: auto-calculated based on mode and size)",
    )

    # Generation parameters
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate",
    )
    parser.add_argument(
        "--log_samples",
        type=int,
        default=1,
        help="Log every N samples for debugging",
    )
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
        choices=[
            "AIME24",
            "AIME25",
            "AMC23",
            "MATH500",
            "HMMT_Feb_2025",
            "Minerva",
            "Olympiad",
            "GPQA",
            "BBEH",
            "ZebraLogic",
            "mix",
        ],
        default=[
            "AIME24",
            "AIME25",
            "AMC23",
            "MATH500",
            "HMMT_Feb_2025",
            "Minerva",
            "Olympiad",
            "GPQA",
            "BBEH",
            "ZebraLogic",
        ],
        help="Tasks to evaluate on",
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default="dataset/bench",
        help="Directory containing task data",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
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
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Maximum number of result files to keep per dataset. Older files will be deleted.",
    )

    args = parser.parse_args()
    return validate_args(args)


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

    def load_mix(self) -> Tuple[List[str], List[str]]:
        questions = []
        answers = []
        filepath = os.path.join(self.task_dir, "mix", "test.json")
        with open(filepath, "r", encoding="utf-8") as f:
            samples = json.load(f)
            for sample in samples:
                questions.append(sample["question"])
                answers.append(str(sample["answer"]))
        return questions, answers

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

    def load_hmmt_feb_2025(self) -> Tuple[List[str], List[str]]:
        df = pd.read_parquet(
            os.path.join(self.task_dir, "HMMT_Feb_2025/hmmt_feb_2025.parquet")
        )
        questions = df["problem"].tolist()
        answers = [str(a) for a in df["answer"].tolist()]
        return questions, answers

    def load_minerva(self) -> Tuple[List[str], List[str]]:
        questions = []
        answers = []
        filepath = os.path.join(self.task_dir, "Minerva/minerva.jsonl")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sample = json.loads(line)
                questions.append(sample["question"])
                answers.append(str(sample["answer"]))
        return questions, answers

    def load_olympiad(self) -> Tuple[List[str], List[str]]:
        questions = []
        answers = []
        filepath = os.path.join(self.task_dir, "Olympiad/olympiad.json")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for sample in data:
                questions.append(sample["problem"])
                assert (
                    len(sample["answer"]) == 1
                ), "Expecting single answer in Olympiad Bench."
                answers.append(str(sample["answer"][0]))
        return questions, answers

    def load_gpqa(self) -> Tuple[List[str], List[str]]:
        questions = []
        answers = []
        filepath = os.path.join(self.task_dir, "GPQA_Diamond/gpqa_diamond.json")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for sample in data:
                questions.append(sample["question"])
                answers.append(str(sample["answer"]))
        return questions, answers

    def load_zebra_logic(self) -> Tuple[List[str], List[str]]:
        questions = []
        answers = []
        filepath = os.path.join(self.task_dir, "ZebraLogic/zebra_logic.json")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for sample in data:
                puzzle = sample["puzzle"]
                question = sample["question"]
                choices = sample["choices"]
                answer = sample["answer"]
                assert answer in choices, "Answer must be in choices"
                # enumerate choices with A. , B. , etc.
                choices_string = "".join(
                    [f"{chr(65 + i)}. {c} " for i, c in enumerate(choices)]
                )
                ordered_answer = chr(65 + choices.index(answer))  # Convert to letter
                full_question = f"Puzzle: {puzzle}\n\nQuestion: {question}\n\nChoices: {choices_string}\n\nAnswer:"
                questions.append(full_question)
                answers.append(ordered_answer)
        return questions, answers

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
        elif dataset_name == "HMMT_Feb_2025":
            return self.load_hmmt_feb_2025()
        elif dataset_name == "Minerva":
            return self.load_minerva()
        elif dataset_name == "Olympiad":
            return self.load_olympiad()
        elif dataset_name == "GPQA":
            return self.load_gpqa()
        elif dataset_name == "ZebraLogic":
            return self.load_zebra_logic()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")


class PromptBuilder:
    """Handles prompt construction for different models"""

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.model_name = os.path.basename(self.args.model_path)

    def build_prompt(self, question: str, parallel_reasoning: bool) -> str:
        """Build prompt for a given question"""
        if not self.args.apply_chat:
            inst = "Please think step by step and in parallel. Put your final answer within \\boxed{}."
            return f"{inst}\n\n{question}"

        # Determine system message based on model
        if "Qwen3" in self.model_name:
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        elif "Qwen2.5-32B-Instruct" in self.model_name:
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        elif "Multiverse-32B" in self.model_name:
            system_prompt = "Please think step by step and in parallel. Put your final answer within \\boxed{}."
        elif "MathReasoner" in self.model_name:
            system_prompt = None
        else:
            system_prompt = None
            console.log(
                f"[yellow]Warning: Unknown model for chat template: {self.model_name}. Proceeding without system prompt.[/yellow]"
            )

        # if parallel_reasoning:
        # # question += " Think step by step before answering."
        # question += " Think step by step and in parallel before answering."
        # else:
        # question += " Think step by step before answering."
        # question += " Think step by step and in parallel before answering."

        if self.args.instruction:
            instruction = load_instruction(self.args.instruction)
        else:
            instruction = None

        msgs = []

        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if instruction:
            msgs.append({"role": "user", "content": question + "\n\n" + instruction})
        else:
            msgs.append({"role": "user", "content": question})

        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )


class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        self.engine_id = kwargs.get("engine_id", 0)
        kwargs.pop("engine_id", None)
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()


@ray.remote(num_cpus=1)
class RayAsyncSglangEngine:
    """Ray actor wrapper for SGLang AsyncEngine, following AsyncSglangServer pattern"""

    def __init__(self, engine_id: int, model_path: str, tp_size: int = 1,
                 mem_fraction_static: float = 0.8, **kwargs):
        self.engine_id = engine_id
        self.model_path = model_path
        self.tp_size = tp_size
        self.mem_fraction_static = mem_fraction_static
        self.kwargs = kwargs
        self.engine = None
        self._initialized = False

    async def init_engine(self):
        """Initialize the SGLang engine, similar to AsyncSglangServer.init_engine"""
        if self._initialized:
            return

        console.log(f"Ray Engine {self.engine_id}: Initializing SGLang engine...")
        self.engine = AsyncEngine(
            model_path=self.model_path,
            tp_size=self.tp_size,
            mem_fraction_static=self.mem_fraction_static,
            disable_overlap_schedule=True,
            dtype=torch.bfloat16,
            engine_id=self.engine_id,
            **self.kwargs
        )
        self._initialized = True
        console.log(f"Ray Engine {self.engine_id}: Initialization complete")

    async def async_generate(self, prompt, sampling_params):
        """Generate responses asynchronously"""
        if not self._initialized:
            await self.init_engine()

        return await self.engine.async_generate(
            prompt=prompt,
            sampling_params=sampling_params
        )

    async def flush_cache(self):
        """Flush the engine cache"""
        if self.engine:
            return await self.engine.flush_cache()

    def wake_up(self):
        """Wake up engine - following AsyncSglangServer pattern"""
        if self.engine:
            console.log(f"Ray Engine {self.engine_id}: Wake up called")

    def sleep(self):
        """Sleep engine - following AsyncSglangServer pattern"""
        if self.engine:
            console.log(f"Ray Engine {self.engine_id}: Sleep called")

    def shutdown(self):
        """Shutdown the engine"""
        if self.engine:
            self.engine.shutdown()
            self.engine = None
            self._initialized = False


class RayAsyncEngineManager:
    """Manager for multiple Ray-based SGLang engines, following AsyncLLMServerManager pattern"""

    def __init__(self, model_path: str, dp_size: int, tp_size: int = 1,
                 mem_fraction_static: float = None, **kwargs):
        self.model_path = model_path
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.kwargs = kwargs

        # Calculate memory fraction similar to AsyncLLMServerManager
        num_gpus = torch.cuda.device_count()
        if mem_fraction_static is not None:
            self.mem_fraction_static = mem_fraction_static
        else:
            if dp_size <= num_gpus:
                self.mem_fraction_static = 0.8
            else:
                engines_per_gpu = (dp_size + num_gpus - 1) // num_gpus
                safety_factor = 0.85
                self.mem_fraction_static = (0.95 / engines_per_gpu) * safety_factor
                self.mem_fraction_static = max(0.05, self.mem_fraction_static)

        console.log(f"Ray Manager: Creating {dp_size} engines with memory fraction {self.mem_fraction_static:.3f}")

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors for engines following AsyncLLMServerManager pattern
        self.async_engines = []
        self._create_engines()

    def _create_engines(self):
        """Create Ray actor engines, similar to AsyncLLMServerManager server creation"""
        num_gpus = torch.cuda.device_count()

        # Start all engine instances, restart if needed (following AsyncLLMServerManager pattern)
        unready_engine_ranks = set(range(self.dp_size))
        max_retries = 3
        retry_count = 0

        while len(unready_engine_ranks) > 0 and retry_count < max_retries:
            console.log(f"Ray Manager: Creating engines, attempt {retry_count + 1}")

            engines = {}
            for engine_id in unready_engine_ranks:
                gpu_id = engine_id % num_gpus
                console.log(f"Ray Manager: Creating engine {engine_id} on GPU {gpu_id}")

                # Create engine actor with GPU assignment
                if self.dp_size <= num_gpus:
                    # Each engine gets its own GPU
                    engine_actor = RayAsyncSglangEngine.options(
                        num_gpus=1,
                        name=f"sglang_engine_{engine_id}"
                    ).remote(
                        engine_id=engine_id,
                        model_path=self.model_path,
                        tp_size=self.tp_size,
                        mem_fraction_static=self.mem_fraction_static,
                        **self.kwargs
                    )
                else:
                    # Multiple engines share GPUs - use fractional allocation
                    gpu_fraction = 1.0 / ((self.dp_size + num_gpus - 1) // num_gpus)
                    engine_actor = RayAsyncSglangEngine.options(
                        num_gpus=gpu_fraction,
                        name=f"sglang_engine_{engine_id}"
                    ).remote(
                        engine_id=engine_id,
                        model_path=self.model_path,
                        tp_size=self.tp_size,
                        mem_fraction_static=self.mem_fraction_static,
                        **self.kwargs
                    )
                engines[engine_id] = engine_actor

            # Try to verify all engines are ready
            for engine_id, engine_actor in engines.items():
                try:
                    # Test engine availability
                    ray.get(engine_actor.__ray_ready__.remote(), timeout=10)
                    self.async_engines.append(engine_actor)
                    unready_engine_ranks.remove(engine_id)
                    console.log(f"Ray Manager: Engine {engine_id} ready")
                except Exception as e:
                    console.log(f"Ray Manager: Engine {engine_id} failed: {e}")
                    try:
                        ray.kill(engine_actor)
                    except:
                        pass

            retry_count += 1

        if len(unready_engine_ranks) > 0:
            raise RuntimeError(f"Failed to create engines: {unready_engine_ranks}")

        console.log(f"Ray Manager: All {len(self.async_engines)} engines created successfully")

    async def init_all_engines(self):
        """Initialize all engines in parallel, following AsyncLLMServerManager.init_engine pattern"""
        console.log("Ray Manager: Initializing all engines in parallel...")

        # Initialize all engines concurrently
        init_futures = [engine.init_engine.remote() for engine in self.async_engines]
        init_results = await asyncio.gather(*[self._ray_future_to_asyncio(fut) for fut in init_futures])

        # Log wrapped ray subprocess initialization results with rich.syntax XML
        console.log("Ray subprocess engine initialization results:")
        syntaxed_init_results = Syntax(
            str(init_results),
            "xml",
            theme="github-dark",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(syntaxed_init_results)

        console.log("Ray Manager: All engines initialized successfully")

    async def _ray_future_to_asyncio(self, ray_future):
        """Convert Ray future to asyncio future"""
        while True:
            try:
                result = ray.get(ray_future, timeout=0.1)
                # Log wrapped ray subprocess result with rich.syntax XML
                console.log("Ray subprocess result obtained:")
                syntaxed_result = Syntax(
                    str(result),
                    "xml",
                    theme="github-dark",
                    line_numbers=True,
                    word_wrap=True,
                )
                console.print(syntaxed_result)
                return result
            except ray.exceptions.GetTimeoutError:
                await asyncio.sleep(0.1)

    def get_engine(self, index: int):
        """Get engine by index"""
        return self.async_engines[index % len(self.async_engines)]

    def get_all_engines(self):
        """Get all engines"""
        return self.async_engines

    def wake_up(self):
        """Wake up all engines, following AsyncLLMServerManager.wake_up pattern"""
        console.log("Ray Manager: Waking up all engines...")
        ray.get([engine.wake_up.remote() for engine in self.async_engines])

    def sleep(self):
        """Sleep all engines, following AsyncLLMServerManager.sleep pattern"""
        console.log("Ray Manager: Sleeping all engines...")
        ray.get([engine.sleep.remote() for engine in self.async_engines])

    async def shutdown_all(self):
        """Shutdown all engines"""
        console.log("Ray Manager: Shutting down all engines...")
        shutdown_futures = [engine.shutdown.remote() for engine in self.async_engines]
        shutdown_results = await asyncio.gather(*[self._ray_future_to_asyncio(fut) for fut in shutdown_futures])

        # Log wrapped ray subprocess shutdown results with rich.syntax XML
        console.log("Ray subprocess engine shutdown results:")
        syntaxed_shutdown_results = Syntax(
            str(shutdown_results),
            "xml",
            theme="github-dark",
            line_numbers=True,
            word_wrap=True,
        )
        console.print(syntaxed_shutdown_results)
        console.log("Ray Manager: All engines shut down")


class RayDataParallelEvaluator:
    """Handles data parallel evaluation with Ray-based engines"""

    def __init__(self, ray_manager: RayAsyncEngineManager, tokenizer, args):
        self.ray_manager = ray_manager
        self.config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.args = args
        self.dp_size = ray_manager.dp_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.timer = InferenceTimer()

        # Build sampling params
        if args.parallel_reasoning:
            console.log("[bold pink1]Parallel reasoning enabled.[/bold pink1]")
            stop_token_ids = [
                self.tokenizer.encode("</guideline>")[0],
                self.tokenizer.encode("</step>")[0],
            ]
        else:
            stop_token_ids = []

        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "skip_special_tokens": False,
            "stop_token_ids": stop_token_ids,
            "no_stop_trim": True,
        }
        if args.temperature is not None:
            self.sampling_params["temperature"] = args.temperature
        if args.top_p is not None:
            self.sampling_params["top_p"] = args.top_p
        if args.top_k is not None:
            self.sampling_params["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            self.sampling_params["repetition_penalty"] = args.repetition_penalty

    def split_data(
        self, prompts: List[str], answers: List[str]
    ) -> List[Tuple[List[str], List[str], List[int]]]:
        """Split data into chunks for each engine"""
        n = len(prompts)
        chunk_size = (n + self.dp_size - 1) // self.dp_size

        chunks = []
        for i in range(self.dp_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            if start_idx < n:
                chunk_prompts = prompts[start_idx:end_idx]
                chunk_answers = answers[start_idx:end_idx]
                chunk_indices = list(range(start_idx, end_idx))
                chunks.append((chunk_prompts, chunk_answers, chunk_indices))

        return chunks

    async def evaluate_with_ray_engine(
        self,
        engine_id: int,
        engine_actor,
        prompts: List[str],
        answers: List[str],
        indices: List[int],
        dataset_name: str,
    ) -> Tuple[Dict[int, Dict], Dict[str, int]]:
        """Evaluate a subset of data with a Ray-based engine"""
        console.log(f"Ray Engine {engine_id}: Processing {len(prompts)} problems")

        results = {}
        total_tokens = 0
        total_samples = 0

        try:
            # Create batches from this engine's chunk
            dataset = TokenizedDataset(prompts, answers)
            dataloader = DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=False
            )

            for batch in dataloader:
                batch_prompts, batch_answers, batch_local_indices = batch
                batch_local_indices = [idx.item() for idx in batch_local_indices]

                # Convert local indices to global indices
                batch_global_indices = [indices[idx] for idx in batch_local_indices]
                batch_prompts = list(batch_prompts)

                batch_results = await self.evaluate_ray_batch(
                    engine_actor, batch_prompts, batch_answers, batch_global_indices, dataset_name
                )
                results.update(batch_results)

                # Count tokens and samples for this batch
                for global_idx in batch_global_indices:
                    if global_idx in batch_results:
                        result = batch_results[global_idx]
                        # Count tokens from all predictions
                        for pred in result["full_preds"]:
                            pred_cleaned = pred.split("<|im_start|>assistant", 1)[
                                -1
                            ].strip()
                            total_tokens += len(self.tokenizer.encode(pred_cleaned))
                        total_samples += len(result["full_preds"])

                # Allow other coroutines to run
                await asyncio.sleep(0)

        except Exception as e:
            console.log(f"Ray Engine {engine_id} encountered error: {e}", style="bold red")
            raise

        console.log(f"Ray Engine {engine_id}: Completed processing {len(results)} results")
        return results, {"total_tokens": total_tokens, "total_samples": total_samples}

    async def evaluate_ray_batch(
        self,
        engine_actor,
        batch_prompts: List[str],
        batch_answers: List[str],
        batch_indices: List[int],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Evaluate a batch using Ray engine"""
        batch_start_time = time.time()
        results = {}

        # Initialize results
        for i, global_idx in enumerate(batch_indices):
            if dataset_name == "HMMT_Feb_2025":
                reference, _ = parse_math_arena_answer(
                    str(batch_answers[i]), list_answer="," in str(batch_answers[i])
                )
                reference = str(reference)
            else:
                reference = strip_string(batch_answers[i], skip_unit=False)

            results[global_idx] = {
                "time": datetime.now().isoformat().rsplit(".", 1)[0].strip(),
                "prompt": batch_prompts[i],
                "sample_size": self.args.num_samples,
                "full_preds": [],
                "boxed_preds": [],
                "reference": reference,
            }

        # Generate samples
        batch_total_tokens = 0
        for sample_id in range(self.args.num_samples):
            set_seed(sample_id)

            # Call Ray engine for generation
            generation_future = engine_actor.async_generate.remote(
                prompt=batch_prompts,
                sampling_params=self.sampling_params
            )

            # Convert Ray future to result
            full_preds_response = await self._ray_future_to_asyncio(generation_future)
            full_preds = [r["text"] for r in full_preds_response]

            # Log wrapped response from ray subprocess with rich.syntax XML
            console.log("Ray subprocess response received:")
            syntaxed_response = Syntax(
                str(full_preds_response),
                "xml",
                theme="github-dark",
                line_numbers=True,
                word_wrap=True,
            )
            console.print(syntaxed_response)

            boxed_preds = [
                extract_answer(pred, dataset_name, use_last_number=True)
                if dataset_name != "HMMT_Feb_2025"
                else str(
                    extract_math_arena_answer(
                        pred, False, True, list_answer="," in str(batch_answers[i])
                    )[0]
                )
                for pred in full_preds
            ]

            # Count tokens for this sample batch
            sample_tokens = sum(
                [len(self.tokenizer.encode(pred)) for pred in full_preds]
            )
            batch_total_tokens += sample_tokens

            # Store predictions
            for i, global_idx in enumerate(batch_indices):
                results[global_idx]["full_preds"].append(full_preds[i])
                results[global_idx]["boxed_preds"].append(boxed_preds[i])

            # Log full predictions in debug mode
            if self.local_rank == 0 and self.args.debug:
                for i, global_idx in enumerate(batch_indices):
                    if i < self.args.log_samples:
                        console.rule()
                        syntaxed_text = Syntax(
                            full_preds[i],
                            "xml",
                            theme="github-dark",
                            line_numbers=True,
                            word_wrap=True,
                        )
                        console.print(syntaxed_text)
                        console.rule()
                    else:
                        break

        # Calculate scores and finalize results
        for global_idx in results:
            correct_count = 0
            reference = results[global_idx]["reference"]
            preds = results[global_idx]["boxed_preds"]
            full_preds = results[global_idx]["full_preds"]
            full_preds_cleaned = [
                pred.split("<|im_start|>assistant", 1)[-1].strip()
                for pred in full_preds
            ]
            for pred in preds:
                pred_cleaned = strip_string(pred, skip_unit=False)
                if math_equal(prediction=pred_cleaned, reference=reference):
                    correct_count += 1
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
            results[global_idx]["completed_tokens"] = int(avg_completed_tokens)

            if self.local_rank == 0 and self.args.debug:
                console.log(
                    f"Problem {global_idx}: avg pass@1: {passk_score:.3f}; avg format score: {avg_format_score:.3f}",
                )

        return results

    async def _ray_future_to_asyncio(self, ray_future):
        """Convert Ray future to asyncio future"""
        while True:
            try:
                result = ray.get(ray_future, timeout=0.1)
                # Log wrapped ray subprocess result with rich.syntax XML
                console.log("Ray subprocess result obtained:")
                syntaxed_result = Syntax(
                    str(result),
                    "xml",
                    theme="github-dark",
                    line_numbers=True,
                    word_wrap=True,
                )
                console.print(syntaxed_result)
                return result
            except ray.exceptions.GetTimeoutError:
                await asyncio.sleep(0.1)

    async def evaluate_parallel_async(
        self,
        prompts: List[str],
        answers: List[str],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Evaluate using Ray-based data parallelism"""
        # Start timing
        if self.local_rank == 0:
            self.timer.start()

        # Split data across engines
        chunks = self.split_data(prompts, answers)
        engines = self.ray_manager.get_all_engines()

        # Create async tasks for each engine
        tasks = []
        for engine_id, (engine_actor, chunk) in enumerate(zip(engines, chunks)):
            if not chunk:  # Skip if no data for this engine
                continue

            chunk_prompts, chunk_answers, chunk_indices = chunk

            task = self.evaluate_with_ray_engine(
                engine_id,
                engine_actor,
                chunk_prompts,
                chunk_answers,
                chunk_indices,
                dataset_name,
            )
            tasks.append(task)

        # Run all tasks concurrently and gather results
        console.log(f"Starting Ray async evaluation with {len(tasks)} tasks")
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = {}
        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                console.log(f"Task {i} failed with error: {result}", style="bold red")
                raise result
            # result is a tuple (results_dict, stats_dict)
            results_dict, stats_dict = result
            all_results.update(results_dict)

        # End timing and print stats
        if self.local_rank == 0:
            self.timer.end()
            self.timer.print_final_stats()

        console.log(
            f"Ray async data parallel evaluation completed. Collected {len(all_results)} results"
        )
        return all_results


class DataParallelEvaluator:
    """Handles data parallel evaluation with multiple engines"""

    def __init__(self, models: List[AsyncEngine], tokenizer, args):
        self.models = models
        self.config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.args = args
        self.dp_size = len(models)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.timer = InferenceTimer()

        # Build sampling params similar to ModelEvaluator
        if args.parallel_reasoning:
            console.log("[bold pink1]Parallel reasoning enabled.[/bold pink1]")
            stop_token_ids = [
                self.tokenizer.encode("</guideline>")[0],
                self.tokenizer.encode("</step>")[0],
            ]
        else:
            stop_token_ids = []

        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "skip_special_tokens": False,
            "stop_token_ids": stop_token_ids,
            "no_stop_trim": True,
        }
        if args.temperature is not None:
            self.sampling_params["temperature"] = args.temperature
        if args.top_p is not None:
            self.sampling_params["top_p"] = args.top_p
        if args.top_k is not None:
            self.sampling_params["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            self.sampling_params["repetition_penalty"] = args.repetition_penalty

    def split_data(
        self, prompts: List[str], answers: List[str]
    ) -> List[Tuple[List[str], List[str], List[int]]]:
        """Split data into chunks for each engine"""
        n = len(prompts)
        chunk_size = (n + self.dp_size - 1) // self.dp_size

        chunks = []
        for i in range(self.dp_size):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            if start_idx < n:
                chunk_prompts = prompts[start_idx:end_idx]
                chunk_answers = answers[start_idx:end_idx]
                chunk_indices = list(range(start_idx, end_idx))
                chunks.append((chunk_prompts, chunk_answers, chunk_indices))

        return chunks

    async def evaluate_with_engine(
        self,
        engine_id: int,
        engine: AsyncEngine,
        prompts: List[str],
        answers: List[str],
        indices: List[int],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Evaluate a subset of data with a specific engine"""
        console.log(f"Engine {engine_id}: Processing {len(prompts)} problems")

        results = {}
        try:
            evaluator = ModelEvaluator(engine, self.tokenizer, self.args)

            # Create batches from this engine's chunk
            dataset = TokenizedDataset(prompts, answers)
            dataloader = DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=False
            )

            # for batch in tqdm(dataloader, desc=f"Engine {engine_id}"):
            for batch in dataloader:
                batch_prompts, batch_answers, batch_local_indices = batch
                batch_local_indices = [idx.item() for idx in batch_local_indices]

                # Convert local indices to global indices
                batch_global_indices = [indices[idx] for idx in batch_local_indices]

                batch_prompts = list(batch_prompts)
                batch_results = await evaluator.evaluate_batch(
                    batch_prompts, batch_answers, batch_global_indices, dataset_name
                )
                results.update(batch_results)

        except Exception as e:
            console.log(f"Engine {engine_id} encountered error: {e}", style="bold red")
            raise

        console.log(f"Engine {engine_id} (async): Completed {len(results)} results")
        return results

    async def evaluate_with_engine_async(
        self,
        engine_id: int,
        engine: AsyncEngine,
        prompts: List[str],
        answers: List[str],
        indices: List[int],
        dataset_name: str,
    ) -> Tuple[Dict[int, Dict], Dict[str, int]]:
        """Async version of evaluate_with_engine for better concurrency"""
        console.log(f"Engine {engine_id}: Processing {len(prompts)} problems (async)")

        results = {}
        total_tokens = 0
        total_samples = 0

        try:
            evaluator = ModelEvaluator(engine, self.tokenizer, self.args)

            # Create batches from this engine's chunk
            dataset = TokenizedDataset(prompts, answers)
            dataloader = DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=False
            )

            # for batch in tqdm(dataloader, desc=f"Engine {engine_id} (async)"):
            for batch in dataloader:
                batch_prompts, batch_answers, batch_local_indices = batch
                batch_local_indices = [idx.item() for idx in batch_local_indices]

                # Convert local indices to global indices
                batch_global_indices = [indices[idx] for idx in batch_local_indices]

                batch_prompts = list(batch_prompts)
                batch_results = await evaluator.evaluate_batch(
                    batch_prompts, batch_answers, batch_global_indices, dataset_name
                )
                results.update(batch_results)

                # Count tokens and samples for this batch
                for global_idx in batch_global_indices:
                    if global_idx in batch_results:
                        result = batch_results[global_idx]
                        # Count tokens from all predictions
                        for pred in result["full_preds"]:
                            pred_cleaned = pred.split("<|im_start|>assistant", 1)[
                                -1
                            ].strip()
                            total_tokens += len(self.tokenizer.encode(pred_cleaned))
                        total_samples += len(result["full_preds"])

                # Allow other coroutines to run
                await asyncio.sleep(0)

        except Exception as e:
            console.log(f"Engine {engine_id} encountered error: {e}", style="bold red")
            raise

        console.log(f"Engine {engine_id}: Completed processing {len(results)} results")
        return results, {"total_tokens": total_tokens, "total_samples": total_samples}

    async def evaluate_parallel_async(
        self,
        prompts: List[str],
        answers: List[str],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Async version of parallel evaluation for better performance"""
        # Start timing
        if self.local_rank == 0:
            self.timer.start()

        # Split data across engines
        chunks = self.split_data(prompts, answers)

        # Create async tasks for each engine
        tasks = []
        for engine_id, (engine, chunk) in enumerate(zip(self.models, chunks)):
            if not chunk:  # Skip if no data for this engine
                continue

            chunk_prompts, chunk_answers, chunk_indices = chunk

            task = self.evaluate_with_engine_async(
                engine_id,
                engine,
                chunk_prompts,
                chunk_answers,
                chunk_indices,
                dataset_name,
            )
            tasks.append(task)

        # Run all tasks concurrently and gather results
        console.log(f"Starting async evaluation with {len(tasks)} tasks")
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = {}
        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                console.log(f"Task {i} failed with error: {result}", style="bold red")
                raise result
            # result is a tuple (results_dict, stats_dict) from evaluate_with_engine_async
            results_dict, stats_dict = result
            all_results.update(results_dict)

        # End timing and print stats
        if self.local_rank == 0:
            self.timer.end()
            self.timer.print_final_stats()

        console.log(
            f"Async data parallel evaluation completed. Collected {len(all_results)} results"
        )
        return all_results

    def evaluate_parallel(
        self,
        prompts: List[str],
        answers: List[str],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Evaluate using data parallelism across multiple engines"""
        # Start timing
        if self.local_rank == 0:
            self.timer.start()

        # Split data across engines
        chunks = self.split_data(prompts, answers)

        all_results = {}

        # Use ThreadPoolExecutor to run evaluations in parallel
        # Note: For production, consider using ProcessPoolExecutor for true parallelism
        with ThreadPoolExecutor(max_workers=self.dp_size) as executor:
            futures = []

            for engine_id, (engine, chunk) in enumerate(zip(self.models, chunks)):
                if not chunk:  # Skip if no data for this engine
                    continue

                chunk_prompts, chunk_answers, chunk_indices = chunk

                future = executor.submit(
                    self.evaluate_with_engine,
                    engine_id,
                    engine,
                    chunk_prompts,
                    chunk_answers,
                    chunk_indices,
                    dataset_name,
                )
                futures.append(future)

            # Gather results from all engines
            for future in as_completed(futures):
                try:
                    engine_results = future.result()
                    all_results.update(engine_results)
                except Exception as e:
                    console.log(f"Error in engine evaluation: {e}", style="bold red")
                    raise

        # End timing and print stats
        if self.local_rank == 0:
            self.timer.end()
            self.timer.print_final_stats()

        console.log(
            f"Data parallel evaluation completed. Collected {len(all_results)} results"
        )
        return all_results


class ModelEvaluator:
    """Handles model evaluation logic"""

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.config = AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        self.tokenizer = tokenizer
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.timer = InferenceTimer()

        if args.parallel_reasoning:
            console.log("Parallel reasoning enabled.")
            stop_token_ids = [
                self.tokenizer.encode("</guideline>")[0],  # </goal>
                self.tokenizer.encode("</step>")[0],
            ]  # tokens triggering parallel reasoning
            # else:
            # raise ValueError(
            # f"Parallel reasoning not supported for model type {self.config.model_type}"
            # )
        else:
            stop_token_ids = []

        # Build sampling parameters
        self.sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "skip_special_tokens": False,
            "stop_token_ids": stop_token_ids,
            "no_stop_trim": True,
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

    async def generate_with_retry(self, prompts: List[str]) -> List[str]:
        """Generate responses with OOM handling"""
        while True:
            with torch.no_grad():
                # generated = self.model.generate(prompts, self.sampling_params)
                generated = await self.model.async_generate(
                    prompt=prompts,
                    sampling_params=self.sampling_params,
                )
                await self.model.flush_cache()
            return [r["text"] for r in generated]

    async def evaluate_batch(
        self,
        batch_prompts: List[str],
        batch_answers: List[str],
        batch_indices: List[int],
        dataset_name: str,
    ) -> Dict[int, Dict]:
        """Evaluate a batch of prompts"""
        batch_start_time = time.time()
        results = {}

        # Initialize results
        for i, global_idx in enumerate(batch_indices):
            if dataset_name == "HMMT_Feb_2025":
                reference, _ = parse_math_arena_answer(
                    str(batch_answers[i]), list_answer="," in str(batch_answers[i])
                )
                reference = str(reference)
            else:
                reference = strip_string(batch_answers[i], skip_unit=False)

            results[global_idx] = {
                "time": datetime.now().isoformat().rsplit(".", 1)[0].strip(),
                "prompt": batch_prompts[i],
                "sample_size": self.args.num_samples,
                "full_preds": [],
                "boxed_preds": [],
                "reference": reference,
            }

        # Generate samples
        batch_total_tokens = 0
        # for sample_id in tqdm(range(self.args.num_samples), desc="Sampling..."):
        for sample_id in range(self.args.num_samples):
            set_seed(sample_id)

            full_preds = await self.generate_with_retry(batch_prompts)
            boxed_preds = [
                extract_answer(pred, dataset_name, use_last_number=True)
                if dataset_name != "HMMT_Feb_2025"
                else str(
                    extract_math_arena_answer(
                        pred, False, True, list_answer="," in str(batch_answers[i])
                    )[0]
                )
                for pred in full_preds
            ]

            # Count tokens for this sample batch
            sample_tokens = sum(
                [len(self.tokenizer.encode(pred)) for pred in full_preds]
            )
            batch_total_tokens += sample_tokens

            # Store predictions
            for i, global_idx in enumerate(batch_indices):
                results[global_idx]["full_preds"].append(full_preds[i])
                results[global_idx]["boxed_preds"].append(boxed_preds[i])

            # Log full predictions in debug mode
            if self.local_rank == 0 and self.args.debug:
                for i, global_idx in enumerate(batch_indices):
                    if i < self.args.log_samples:
                        console.rule()
                        # syntax = Syntax(self.instruction, "html", theme="monokai", line_numbers=True)
                        syntaxed_text = Syntax(
                            full_preds[i],
                            "xml",
                            theme="github-dark",
                            line_numbers=True,
                            word_wrap=True,
                        )
                        console.print(syntaxed_text)
                        console.rule()
                    else:
                        break

        # Update timer with batch statistics
        batch_time = time.time() - batch_start_time
        total_samples = len(batch_indices) * self.args.num_samples
        if self.local_rank == 0:
            self.timer.add_batch_stats(batch_total_tokens, total_samples, batch_time)
            # Print progress periodically (every few batches)
            if hasattr(self, "_batch_count"):
                self._batch_count += 1
            else:
                self._batch_count = 1

            # Print progress every 5 batches or so
            if self._batch_count % max(1, min(5, self.args.batch_size // 4)) == 0:
                # self.timer.print_progress()
                pass

        # Calculate scores
        for global_idx in results:
            correct_count = 0
            reference = results[global_idx]["reference"]
            preds = results[global_idx]["boxed_preds"]
            full_preds = results[global_idx]["full_preds"]
            full_preds_cleaned = [
                pred.split("<|im_start|>assistant", 1)[-1].strip()
                for pred in full_preds
            ]
            for pred in preds:
                # console.print(pred)
                pred_cleaned = strip_string(pred, skip_unit=False)
                if math_equal(prediction=pred_cleaned, reference=reference):
                    correct_count += 1
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
            results[global_idx]["completed_tokens"] = int(avg_completed_tokens)

            if self.local_rank == 0 and self.args.debug:
                console.log(
                    f"Problem {global_idx}: avg pass@1: {passk_score:.3f}; avg format score: {avg_format_score:.3f}",
                )
                avg_completed_tokens = sum(
                    [len(self.tokenizer.encode(pred)) for pred in full_preds_cleaned]
                ) / len(full_preds_cleaned)
                input_tokens = len(self.tokenizer.encode(results[global_idx]["prompt"]))
                results[global_idx][f"pass@{self.args.passk}"] = passk_score
                results[global_idx]["format_score"] = avg_format_score
                results[global_idx]["input_tokens"] = input_tokens
                results[global_idx]["completed_tokens"] = int(avg_completed_tokens)

        return results


def initialize_model_and_tokenizer(args):
    """Initialize model and tokenizer for either TP, DP, or Ray mode"""
    console.log(f"Loading model/tokenizer from {args.model_path}")
    console.log(f"Mode: {args.mode.upper()}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left", use_fast=False
    )

    if hasattr(config, "max_position_embeddings"):
        console.log(f"Max position embeddings: {config.max_position_embeddings}")
    else:
        console.log(f"Max position embeddings: {config.model_max_length}")

    if args.mode == "tp":
        # Tensor Parallelism mode - single engine with multiple GPUs
        tp_size = args.tp_size if args.tp_size else torch.cuda.device_count()
        console.log(f"Using Tensor Parallelism with tp_size={tp_size}")

        model = AsyncEngine(
            model_path=args.model_path,
            tp_size=tp_size,
            disable_overlap_schedule=True,
            dtype=torch.bfloat16,
            # log_level="info",
            mem_fraction_static=0.8,
        )
        return model, tokenizer

    elif args.mode == "ray":
        # Ray mode - distributed async engines with repeated initiation
        num_gpus = torch.cuda.device_count()
        dp_size = args.dp_size if args.dp_size else num_gpus
        tp_size = args.tp_size if args.tp_size else 1
        console.log(
            f"Using Ray mode with dp_size={dp_size}, tp_size={tp_size}, available GPUs={num_gpus}"
        )

        # Create Ray manager for repeated initiation
        ray_manager = RayAsyncEngineManager(
            model_path=args.model_path,
            dp_size=dp_size,
            tp_size=tp_size,
            mem_fraction_static=args.mem_fraction_static,
        )
        return ray_manager, tokenizer

    elif args.mode == "dp":
        # Data Parallelism mode - multiple engines
        num_gpus = torch.cuda.device_count()
        dp_size = args.dp_size if args.dp_size else num_gpus
        console.log(
            f"Using Data Parallelism with dp_size={dp_size}, available GPUs={num_gpus}"
        )

        # Determine GPU assignment strategy
        if dp_size <= num_gpus:
            # Each engine gets its own GPU
            console.log(f"Assigning one GPU per engine")
            engines_per_gpu = 1
            gpus_per_engine = 1
        else:
            # Multiple engines share GPUs
            engines_per_gpu = (dp_size + num_gpus - 1) // num_gpus
            gpus_per_engine = 1
            console.log(f"Multiple engines per GPU: {engines_per_gpu} engines/GPU")

        # Calculate memory fraction
        if args.mem_fraction_static is not None:
            mem_fraction = args.mem_fraction_static
        else:
            if dp_size <= num_gpus:
                # Each engine on separate GPU - can use more memory
                mem_fraction = 0.8
            else:
                # Multiple engines per GPU - need to share memory
                safety_factor = 0.85
                mem_fraction = (0.95 / engines_per_gpu) * safety_factor
                mem_fraction = max(0.05, mem_fraction)

        console.log(f"Memory fraction per engine: {mem_fraction:.3f}")

        models = []
        for i in range(dp_size):
            # Determine which GPU this engine should use
            gpu_id = i % num_gpus
            console.log(f"Initializing engine {i} on GPU {gpu_id}")

            # Set CUDA_VISIBLE_DEVICES for this engine
            import os

            original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            try:
                engine = AsyncEngine(
                    model_path=args.model_path,
                    tp_size=1,
                    disable_overlap_schedule=True,
                    dtype=torch.bfloat16,
                    # log_level="info",
                    mem_fraction_static=mem_fraction,
                )
                models.append(engine)
            finally:
                # Restore original CUDA setting
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda

        return models, tokenizer

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def cleanup_old_results(output_dir: str, dataset_name: str, save_total_limit: int):
    """Remove old result files to maintain save_total_limit"""
    if save_total_limit is None:
        return

    # Find all result files for this dataset
    pattern = f"{dataset_name.lower()}_avg*.json"
    result_files = []

    for file in os.listdir(output_dir):
        if file.startswith(dataset_name.lower()) and file.endswith(".json"):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                result_files.append((file_path, os.path.getmtime(file_path)))

    # Sort by modification time (newest first)
    result_files.sort(key=lambda x: x[1], reverse=True)

    # Delete files beyond the limit
    if len(result_files) > save_total_limit:
        files_to_delete = result_files[save_total_limit:]
        for file_path, _ in files_to_delete:
            try:
                os.remove(file_path)
                console.log(f"Deleted old result file: {os.path.basename(file_path)}")
            except OSError as e:
                console.log(f"Failed to delete {file_path}: {e}", style="bold red")


def evaluate_single_dataset(
    dataset_name: str,
    args,
    model,  # Can be single model (TP), list of models (DP), or RayAsyncEngineManager (Ray)
    tokenizer,
    dataset_loader: DatasetLoader,
    prompt_builder: PromptBuilder,
    evaluator: Optional[ModelEvaluator] = None,  # For TP mode
    dp_evaluator: Optional[DataParallelEvaluator] = None,  # For DP mode
    ray_evaluator: Optional[RayDataParallelEvaluator] = None,  # For Ray mode
):
    """Evaluate a single dataset using either TP, DP, or Ray mode"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup output path
    out_dir = os.path.join(
        args.output_dir,
        "eval",
        os.path.basename(args.model_path) + args.save_alias,
    )
    if local_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(
        out_dir, f"{dataset_name.lower()}_avg{args.num_samples}.json"
    )
    if not args.overwrite and os.path.exists(output_path):
        console.log(f'"{output_path}" already exists, skipping...')
        return None

    # Load dataset
    console.log(f'Loading "{dataset_name}" dataset...')
    questions, targets = dataset_loader.load_dataset(dataset_name)

    if args.max_problems is not None:
        questions = questions[: args.max_problems]
        targets = targets[: args.max_problems]

    # Build prompts
    console.log(f"Building prompts for {len(questions)} problems...")
    prompts = []
    answers = []
    for question, target in zip(questions, targets):
        prompt = prompt_builder.build_prompt(question, args.parallel_reasoning)
        prompts.append(prompt)
        answers.append(str(target))

    # Evaluate based on mode
    with Status(
        f'[bold cyan]Evaluating "{dataset_name}"...[/bold cyan]',
        console=console,
    ) as _:
        # Tensor Parallelism mode
        if args.mode == "tp":
            dataset = TokenizedDataset(prompts, answers)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            all_results = {}

            async def evaluate_tp_async():
                # Start timing for TP mode
                if local_rank == 0:
                    evaluator.timer.start()

                results = {}
                for batch in tqdm(
                    dataloader, desc=f"Evaluating {dataset_name} (TP mode)"
                ):
                    batch_prompts, batch_answers, batch_indices = batch
                    batch_indices = [idx.item() for idx in batch_indices]

                    batch_prompts = list(batch_prompts)
                    batch_results = await evaluator.evaluate_batch(
                        batch_prompts, batch_answers, batch_indices, dataset_name
                    )
                    results.update(batch_results)

                    # checkpoint intermediate results
                    if local_rank == 0:
                        console.log(f'Intermediate results saved to "{output_path}"')

                # End timing for TP mode
                if local_rank == 0:
                    evaluator.timer.end()
                    evaluator.timer.print_final_stats()

                return results

            loop = asyncio.get_event_loop()
            all_results = loop.run_until_complete(evaluate_tp_async())
        # Data Parallelism mode
        elif args.mode == "dp":
            console.log(f"Starting data parallel evaluation for {dataset_name}")

            # Use async evaluation if available
            if args.dp_async:
                loop = asyncio.get_event_loop()
                all_results = loop.run_until_complete(
                    dp_evaluator.evaluate_parallel_async(prompts, answers, dataset_name)
                )
            else:
                all_results = dp_evaluator.evaluate_parallel(
                    prompts, answers, dataset_name
                )
        # Ray mode - async engines with repeated initiation
        elif args.mode == "ray":
            console.log(f"Starting Ray async evaluation for {dataset_name}")

            async def evaluate_ray_async():
                # Initialize all engines first (repeated initiation)
                await ray_evaluator.ray_manager.init_all_engines()

                # Run the evaluation
                return await ray_evaluator.evaluate_parallel_async(
                    prompts, answers, dataset_name
                )

            loop = asyncio.get_event_loop()
            all_results = loop.run_until_complete(evaluate_ray_async())
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    # Calculate final score
    final_score = sum(
        [all_results[k][f"pass@{args.passk}"] for k in all_results]
    ) / len(prompts)  # Use len(prompts) instead of len(dataset)
    console.log(
        f"{os.path.basename(args.model_path)} {dataset_name} pass@{args.passk}: {final_score * 100:.2f}%"
    )
    final_format_score = sum(
        [all_results[k]["format_score"] for k in all_results]
    ) / len(prompts)
    console.log(
        f"{os.path.basename(args.model_path)} {dataset_name} avg format score: {final_format_score:.2f}"
    )

    # Save results
    if local_rank == 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        console.log(f'Results saved to "{output_path}"')

        # Cleanup old results if save_total_limit is specified
        if args.save_total_limit is not None:
            eval_dir = os.path.dirname(output_path)
            cleanup_old_results(eval_dir, dataset_name, args.save_total_limit)

    return final_score, final_format_score


def main():
    """Main evaluation function"""
    sglang.set_default_backend("vllm")

    # Parse arguments
    args = parse_arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize components based on mode
    model_or_models, tokenizer = initialize_model_and_tokenizer(args)
    dataset_loader = DatasetLoader(args.task_dir)
    prompt_builder = PromptBuilder(args, tokenizer)

    evaluator = None
    dp_evaluator = None
    ray_evaluator = None

    if args.mode == "tp":
        # Single model evaluator for TP mode
        evaluator = ModelEvaluator(model_or_models, tokenizer, args)
    elif args.mode == "dp":
        # Data parallel evaluator for DP mode
        dp_evaluator = DataParallelEvaluator(model_or_models, tokenizer, args)
    elif args.mode == "ray":
        # Ray-based data parallel evaluator with repeated initiation
        ray_evaluator = RayDataParallelEvaluator(model_or_models, tokenizer, args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each dataset
    results_summary = {}
    format_scores_summary = {}
    for dataset_name in args.tasks:
        console.rule()
        console.log(
            f'Starting evaluation on "{dataset_name}" with {args.mode.upper()} mode'
        )
        console.rule()

        result = evaluate_single_dataset(
            dataset_name,
            args,
            model_or_models,
            tokenizer,
            dataset_loader,
            prompt_builder,
            evaluator,
            dp_evaluator,
            ray_evaluator,
        )

        if result is not None:
            score, format_score = result
            results_summary[dataset_name] = score
            format_scores_summary[dataset_name] = format_score

    # Print final summary
    if local_rank == 0 and results_summary:
        console.rule()
        table = Table(title=f"Final Evaluation Summary ({args.mode.upper()} mode)")
        table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
        table.add_column(f"Pass@{args.passk}", justify="right", style="magenta")
        table.add_column("Format Score", justify="right", style="green")
        for dataset_name, score in results_summary.items():
            format_score = format_scores_summary[dataset_name]
            table.add_row(dataset_name, f"{score * 100:.2f}%", f"{format_score:.2f}")
        console.print(table)
        console.rule()

    # Cleanup resources
    if args.mode == "tp":
        # Single model cleanup
        console.log("Shutting down TP engine...")
        model_or_models.shutdown()
    elif args.mode == "dp":
        # Multiple models cleanup
        console.log("Shutting down DP engines...")
        for i, model in enumerate(model_or_models):
            console.log(f"Shutting down engine {i}...")
            try:
                model.shutdown()
            except Exception as e:
                console.log(f"Error shutting down engine {i}: {e}", style="bold red")
    elif args.mode == "ray":
        # Ray engines cleanup with repeated initiation support
        console.log("Shutting down Ray engines...")
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(model_or_models.shutdown_all())
        except Exception as e:
            console.log(f"Error shutting down Ray engines: {e}", style="bold red")

        # Shutdown Ray if we started it
        try:
            ray.shutdown()
        except Exception as e:
            console.log(f"Error shutting down Ray: {e}", style="bold red")


if __name__ == "__main__":
    main()
