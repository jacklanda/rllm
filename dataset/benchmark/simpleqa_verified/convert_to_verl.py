"""Convert simpleqa_verified.csv to the verl parquet format used by 2wiki/data_verl.parquet."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "simpleqa_verified.csv"
DST = HERE / "data_verl.parquet"
DATA_SOURCE = "simpleqa_verified"


def convert() -> pd.DataFrame:
    src = pd.read_csv(SRC)

    rows = []
    for pid, r in enumerate(src.itertuples(index=False)):
        question = str(r.problem)
        answer = str(r.answer)
        urls = [u for u in str(r.urls).split(",") if u]
        rows.append(
            {
                "task_id": str(pid),
                "input": question,
                "gt_answer": answer,
                "subject": "General Knowledge",
                "ground_truth_answer": answer,
                "target": answer,
                "data_source": DATA_SOURCE,
                "prompt": None,
                "ability": "search",
                "reward_model": {"ground_truth": answer, "style": "rule"},
                "extra_info": {
                    "answer": answer,
                    "image": None,
                    "pid": pid,
                    "query": question,
                    "split": "validation",
                    "original_index": int(r.original_index),
                    "topic": str(r.topic),
                    "answer_type": str(r.answer_type),
                    "multi_step": bool(r.multi_step),
                    "requires_reasoning": bool(r.requires_reasoning),
                    "urls": urls,
                },
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "task_id",
            "input",
            "gt_answer",
            "subject",
            "ground_truth_answer",
            "target",
            "data_source",
            "prompt",
            "ability",
            "reward_model",
            "extra_info",
        ],
    )


if __name__ == "__main__":
    df = convert()
    df.to_parquet(DST, index=False)
    print(f"Wrote {len(df)} rows to {DST}")
