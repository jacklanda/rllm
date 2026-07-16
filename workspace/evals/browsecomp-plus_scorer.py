"""BrowseComp-Plus answer judging.

This module keeps only the LLM-as-a-judge path: given a query, a ground-truth
answer, and a model output, ask a judge model whether the output answers the
query correctly.
"""

from __future__ import annotations

from functools import lru_cache
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openai
from tqdm import tqdm


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def call_openai_judge(
    client: openai.OpenAI,
    prompt: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
) -> Any:
    body: dict[str, Any] = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": prompt,
    }
    if system_prompt:
        body["instructions"] = system_prompt
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
    return client.responses.create(**body)


@lru_cache(maxsize=8)
def get_openai_client(api_key: str = "", base_url: str = "") -> Any:
    """Build a cached OpenAI/OpenAI-compatible client from args or env vars."""
    resolved_api_key = api_key or os.getenv("BROWSECOMP_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Set BROWSECOMP_JUDGE_API_KEY or OPENAI_API_KEY before using the BrowseComp-Plus judge."
        )

    try:
        import openai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The openai package is required to call the BrowseComp-Plus judge model."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    resolved_base_url = "https://ark.cn-beijing.volces.com/api/v3"
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return openai.OpenAI(**kwargs)


def _match_field(text: str, field: str, stop_fields: tuple[str, ...] = ()) -> str | None:
    escaped = re.escape(field)
    stop = "|".join(re.escape(f) for f in stop_fields)
    if stop:
        pattern = rf"(?:\*\*{escaped}:?\*\*|{escaped}:)\s*(.*?)(?=\n(?:{stop})\s*:|\n\*\*(?:{stop})\s*:?\*\*|$)"
    else:
        pattern = rf"(?:\*\*{escaped}:?\*\*|{escaped}:)\s*(.*?)(?=\n|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def parse_judge_response(judge_response: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    result["extracted_final_answer"] = _match_field(judge_response, "extracted_final_answer")
    result["reasoning"] = _match_field(
        judge_response,
        "reasoning",
        stop_fields=("correct", "confidence"),
    )

    correct_text = _match_field(judge_response, "correct")
    if correct_text:
        correct_match = re.search(r"\b(yes|no)\b", correct_text, re.IGNORECASE)
        if correct_match:
            result["correct"] = correct_match.group(1).lower() == "yes"

    confidence_text = _match_field(judge_response, "confidence")
    if confidence_text:
        confidence_match = re.search(r"(\d+(?:\.\d+)?)\s*%?", confidence_text)
        if confidence_match:
            result["confidence"] = min(float(confidence_match.group(1)), 100.0)

    if result["correct"] is None:
        result["parse_error"] = True
    return result


def judge_answer(
    client: Any,
    query: str,
    ground_truth: str,
    output: str,
    model: str = "deepseek-v4-flash-260425",
    max_output_tokens: int = 1024,
    reasoning_effort: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    prompt = create_judge_prompt(
        question=query,
        response=output,
        correct_answer=ground_truth,
    )
    judge_response = call_openai_judge(
        client=client,
        prompt=prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
    )
    judge_text = getattr(judge_response, "output_text", "")
    judge_result = parse_judge_response(judge_text)
    is_correct = bool(judge_result.get("correct", False))
    return {
        "query": query,
        "ground_truth": ground_truth,
        "output": output,
        "judge_prompt": prompt,
        "judge_response": judge_text,
        "judge_result": judge_result,
        "extracted_answer": judge_result.get("extracted_final_answer"),
        "is_correct": is_correct,
        "reward": 1.0 if is_correct else 0.0,
    }


def judge_answer_from_env(
    query: str,
    ground_truth: str,
    output: str,
) -> dict[str, Any]:
    """Judge one BrowseComp-Plus answer using environment-configured settings.

    Environment knobs:
    - BROWSECOMP_JUDGE_MODEL, default ``gpt-4.1``
    - BROWSECOMP_JUDGE_MAX_OUTPUT_TOKENS, default ``1024``
    - BROWSECOMP_JUDGE_REASONING_EFFORT, optional
    - BROWSECOMP_JUDGE_SYSTEM_PROMPT, optional
    - BROWSECOMP_JUDGE_API_KEY / OPENAI_API_KEY
    - BROWSECOMP_JUDGE_BASE_URL / OPENAI_BASE_URL
    """
    client = get_openai_client()
    return judge_answer(
        client=client,
        query=query,
        ground_truth=ground_truth,
        output=output,
        model="deepseek-v4-flash-260425",
        max_output_tokens=int(os.getenv("BROWSECOMP_JUDGE_MAX_OUTPUT_TOKENS", "1024")),
        reasoning_effort=os.getenv("BROWSECOMP_JUDGE_REASONING_EFFORT") or None,
        system_prompt=os.getenv("BROWSECOMP_JUDGE_SYSTEM_PROMPT") or None,
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON object/list or JSONL records with query/ground_truth/output."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported input format in {path}")


def _record_value(record: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None:
            return str(value)
    return ""


def _judge_record(
    client: openai.OpenAI,
    record: dict[str, Any],
    model: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    system_prompt: str | None,
) -> dict[str, Any]:
    query = _record_value(record, ("query", "question"))
    ground_truth = _record_value(record, ("ground_truth", "correct_answer", "answer"))
    output = _record_value(record, ("output", "response", "prediction"))
    if not query or not ground_truth or not output:
        raise ValueError(
            "Each record must contain query/question, ground_truth/correct_answer/answer, "
            "and output/response/prediction."
        )
    result = judge_answer(
        client=client,
        query=query,
        ground_truth=ground_truth,
        output=output,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
    )
    if "query_id" in record:
        result["query_id"] = record["query_id"]
    if "id" in record:
        result["id"] = record["id"]
    return result

