"""Convert ScienceQA.csv to the verl parquet format used by 2wiki/data_verl.parquet.

The CSV ships with XOR+base64-encoded `prompt` and `answer` fields (the key is
the per-row `canary` string). We decode them using the same scheme as
xbench_evals.py before emitting the verl record.
"""

import base64
import csv
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "ScienceQA.csv"
DST = HERE / "data_verl.parquet"
DATA_SOURCE = "scienceqa"


def xor_decrypt(data: bytes, key: str) -> bytes:
    key_bytes = key.encode("utf-8")
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def decode_field(value: str, key: str) -> str:
    return xor_decrypt(base64.b64decode(value), key).decode("utf-8")


def convert() -> pd.DataFrame:
    with open(SRC, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    rows = []
    for pid, r in enumerate(raw_rows):
        key = r["canary"]
        question = decode_field(r["prompt"], key)
        answer = decode_field(r["answer"], key)
        qtype = r.get("type", "")
        subject = r.get("subject", "General Knowledge")
        orig_id = r.get("id", str(pid))

        rows.append(
            {
                "task_id": str(pid),
                "input": question,
                "gt_answer": answer,
                "subject": subject,
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
                    "original_id": orig_id,
                    "type": qtype,
                    "subject": subject,
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
