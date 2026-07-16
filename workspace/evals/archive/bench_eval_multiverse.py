# CUDA_VISIBLE_DEVICES=4,5,6,7 python batch_pred_vllm.py --model_path /data/baijun/models/lawgpt-lora-7b-v2 --tp_size 4 --batch_size 1
from transformers import AutoConfig, AutoTokenizer, set_seed, HfArgumentParser
from torch.utils.data import Dataset, DataLoader
import sglang as sgl
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from glob import glob
import pandas as pd
import torch
import math
import json
import re
import os


from grader import math_equal

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Arguments:
    # model_path: str = '/vepfs-cnbj15b5293fdbd7/data/share/Qwen3-14B'

    # model_path: str = '/data/baijun/models/Qwen3-8B-Base'
    # model_path: str = '/data/baijun/models/Qwen3-8B'
    # model_path: str = '/data/baijun/models/Qwen2.5-32B-Instruct'
    # model_path: str = '/data/baijun/models/Multiverse-32B'
    # model_path: str = "/vepfs-cnbj15b5293fdbd7/data/share/checkpoints/MathReasoner-14B-20250824-steps-800"
    # model_path: str = "/vepfs-cnbj15b5293fdbd7/data/share/checkpoints/MathReasoner-MathReasoner-14B-20250826-Serie"
    model_path: str = "/vepfs-cnbj15b5293fdbd7/data/share/checkpoints/MathReasoner-8B-20250826-Serie/checkpoint-888"
    # model_path: str = '/data/baijun/models/MathReasoner-14B-20250821'

    max_new_tokens: int = 30000

    batch_size: int = 1
    apply_chat: bool = True
    debug: bool = True
    # debug: bool = False
    task_dir: str = "bench/"
    output_dir: str = "prediction"
    save_alias: str = ""
    overwrite: bool = True  # 是否覆盖已有结果

    num_samples: int = 8  # 每个题目生成多少次
    passk: int = 1  # 计算 pass@k 时的 k


class TokenizedDataset(Dataset):
    def __init__(
        self,
        prompts: List[str],
        answers: List[str],
    ):
        self.prompts = prompts
        self.answers = answers

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return (self.prompts[idx], self.answers[idx], idx)


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """计算 pass@k (AIME 常用指标)"""
    if num_samples < k:
        return 1.0 if num_correct == num_samples else 0.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def load_amc23():
    df = pd.read_parquet("bench/AMC23/amc23.parquet")
    questions = df["question"].tolist()
    answers = [str(a) for a in df["answer"].tolist()]
    return questions, answers


def load_aime24():
    df = pd.read_parquet("bench/AIME24/aime24.parquet")
    questions = df["problem"].tolist()
    answers = [
        re.search(r"\\boxed\{(\d+)\}", text).group(1)
        for text in df["solution"].tolist()
    ]
    return questions, answers


def load_aime25():
    questions = []
    answers = []
    for file in ["bench/AIME25/aime2025-I.jsonl", "bench/AIME25/aime2025-II.jsonl"]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sample = json.loads(line)
                questions += [sample["question"]]
                answers += [str(sample["answer"])]
    return questions, answers


def load_math500():
    questions = []
    solutions = []
    answers = []
    for file in ["bench/MATH500/math500.jsonl"]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                sample = json.loads(line)
                questions += [sample["problem"]]
                solutions += [sample["solution"]]
                answers += [str(sample["answer"])]
    return questions, solutions, answers


def get_args() -> Tuple[Arguments]:
    parser = HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()
    return args


if __name__ == "__main__":
    sgl.set_default_backend("vllm")

    args = get_args()[0]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load tokenizer and model
    print(f"==> Loading model/tokenizer from {args.model_path}")
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )

    if hasattr(config, "max_position_embeddings"):
        print(f"==> max position embeddings : {config.max_position_embeddings}")
    else:
        print(f"==> max position embeddings : {config.model_max_length}")
    model = sgl.Engine(
        model_path=args.model_path,
        tp_size=torch.cuda.device_count(),
        disable_overlap_schedule=True,
    )

    sampling_params = {
        # "temperature": 0.6,
        # "top_p": 0.95,
        # "repetition_penalty": 1.05,
        # "temperature": 0.7,
        # "top_p": 0.8,
        # "top_k": 20,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
    }

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)
    DATASETS = [
        "AMC23",
        "AIME24",
        "AIME25",
        "MATH500",
    ]

    for dataset_name in DATASETS:
        if dataset_name != "AIME24":
            continue

        # if 'AIME25' not in dataset_name:
        # continue

        # Prepare output path
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
            continue

        # Load raw data

        if dataset_name == "AIME24":
            questions, targets = load_aime24()
        elif dataset_name == "AIME25":
            questions, targets = load_aime25()
        elif dataset_name == "AMC23":
            questions, targets = load_amc23()
        elif dataset_name == "MATH500":
            questions, _, targets = load_math500()

        # Build prompts and answers
        prompts, answers = [], []
        for ques, tgt in tqdm(zip(questions, targets)):
            inst = "Please think step by step and in parallel. Put your final answer within \boxed{}."

            if args.apply_chat:
                if os.path.basename(args.model_path) == "Qwen2.5-32B-Instruct":
                    msgs = [
                        {
                            "role": "system",
                            "content": "Please reason step by step, and put your final answer within \boxed{}.",
                        },
                        {"role": "user", "content": f"{ques}"},
                    ]
                elif os.path.basename(args.model_path).startswith("Multiverse-32B"):
                    msgs = [
                        {
                            "role": "system",
                            "content": "Please think step by step and in parallel. Put your final answer within \boxed{}.",
                        },
                        {"role": "user", "content": f"{ques}"},
                    ]
                elif os.path.basename(args.model_path).startswith("MathReasoner"):
                    msgs = [
                        {
                            "role": "system",
                            "content": "You are a math reasoning agent. Please reason step by step to solve the math problem in parallel.",
                        },
                        {"role": "user", "content": f"{ques}"},
                    ]
                else:
                    msgs = [
                        {
                            "role": "system",
                            "content": "You are a math reasoning agent. Please reason step by step to solve the math problem in parallel.",
                        },
                        {"role": "user", "content": f"{ques}"},
                    ]

                prompt = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = f"{inst}\n{ques}"
            # import pdb; pdb.set_trace()
            prompts.append(prompt)
            answers.append(str(tgt))

        # Create tokenized dataset
        dataset = TokenizedDataset(prompts, answers)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

        sum_passk_score = 0.0
        results: Dict[int, Dict] = {}

        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            batch_prompts, batch_answers, batch_indices = batch
            global_idx = batch_indices[0].item()
            refr = batch_answers[0]

            origin_preds = []
            preds = []
            for sample_id in tqdm(range(args.num_samples), desc=f"Sampling..."):
                set_seed(sample_id)
                max_tokens = args.max_new_tokens
                with torch.no_grad():
                    while True:
                        try:
                            generated = model.generate(
                                batch_prompts[0],
                                sampling_params,
                            )
                            break
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                torch.cuda.empty_cache()
                                max_tokens = max_tokens // 2
                                print(
                                    f"OOM detected, reducing max_new_tokens to {max_tokens}",
                                    flush=True,
                                )
                                if max_tokens < 1:
                                    raise
                            else:
                                raise
                        except ValueError as e:
                            if "max_model_len" in str(e):
                                batch_prompts = [
                                    prompt[100:] for prompt in batch_prompts
                                ]
                                print(
                                    f"Error: {e}. Reducing prompt length.", flush=True
                                )
                # import pdb; pdb.set_trace()
                origin_text = generated["text"]
                # pred = re.findall(r"\\boxed\{(.*?)\}", origin_text)
                if "boxed{" in origin_text:
                    pred = origin_text.rsplit("boxed{", 1)[-1].rsplit("}", 1)[0].strip()
                else:
                    pos = origin_text.find("</think>")
                    pred = (
                        origin_text[pos + len("</think>") :].strip()
                        if pos != -1
                        else origin_text
                    )

                origin_preds.append(origin_text)
                preds.append(pred)

            # 判断正确预测个数
            correct_count = 0
            try:
                # for p in preds:
                # if float(p) == float(refr):
                # correct_count += 1
                # else:
                # if p.strip() == refr.strip():
                # correct_count += 1
                correct_count = sum(
                    [math_equal(prediction=p, reference=refr) for p in preds]
                )
            except Exception as _:
                correct_count = sum([p.strip() == refr.strip() for p in preds])

            results[global_idx] = {
                "origin_prompt": batch_prompts[0],
                "origin_predictions": origin_preds,
                "predictions": preds,
                "sample_size": args.num_samples,
                "correct_count": correct_count,
                "refr": refr,
            }

            # 更新统计
            passk_score = estimate_pass_at_k(
                args.num_samples, correct_count, args.passk
            )
            results[global_idx][f"pass@{args.passk}"] = passk_score
            sum_passk_score += passk_score

            if local_rank == 0:
                print(results[global_idx], flush=True)

        # 计算总体 pass@k
        final_score = sum_passk_score / len(dataset)
        print(
            f"==> {os.path.basename(args.model_path)} {dataset_name} pass@{args.passk}: {final_score * 100:.2f}%"
        )

        if local_rank == 0:
            with open(output_path, "w", encoding="utf-8") as wf:
                json.dump(results, wf, ensure_ascii=False, indent=4)

    if local_rank == 0:
        print("==> Zero-shot inference with Accelerate completed!")
