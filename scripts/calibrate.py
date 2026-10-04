"""Per-question Platt maps, fitted on selection rows only."""
import argparse
import json
import math
from battery import QUESTIONS
from common import index_rows, read_jsonl
from score_battery import load, median_answers, score


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def fit(pairs):
    if not pairs:
        raise ValueError("no selection pairs")
    a, b = 1.0, 0.0
    for _ in range(3000):
        ga = gb = 0.0
        for p, label in pairs:
            error = sig(a * logit(p) + b) - label
            ga += error * logit(p); gb += error
        a -= 0.01 * ga / len(pairs); b -= 0.01 * gb / len(pairs)
    return a, b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outputs"); p.add_argument("--labels", required=True)
    args = p.parse_args(); rows = index_rows(read_jsonl(args.labels)); by = load(args.outputs)
    answers = {key: median_answers(rs) for key, rs in by.items() if key in rows}
    params = {q: fit([(a[q], rows[key]["gold"][q]) for key, a in answers.items() if rows[key]["split"] == "sel"]) for q in QUESTIONS}
    transformed = {key: [{q: sig(params[q][0] * logit(a[q]) + params[q][1]) for q in QUESTIONS}] for key, a in answers.items()}
    test = {key: row for key, row in rows.items() if row["split"] == "test"}
    print(json.dumps({"params": params, "test_raw": score(test, by), "test_calibrated": score(test, transformed)}, indent=2))


if __name__ == "__main__":
    main()
