"""Shared strict input readers for the offline cookbook."""
import json
import math
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows))


def boolean(value, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not bool:
        raise ValueError("label must be a JSON boolean")
    return value


def probability(answer):
    value = answer.get("noul", answer.get("probability")) if isinstance(answer, dict) else answer
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("missing, non-finite, or out-of-range probability")
    return float(value)


def index_rows(rows):
    result = {}
    for row in rows:
        key = row["id"]
        if key in result:
            raise ValueError("duplicate row id")
        result[key] = row
    return result
