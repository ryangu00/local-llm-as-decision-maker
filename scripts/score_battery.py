"""Median per-question scoring with explicit failures and composed decisions."""
import argparse
import collections
import json
import statistics
from battery import QUESTIONS, compose
from common import boolean, index_rows, probability, read_jsonl


def load(path):
    by = collections.defaultdict(list)
    for row in read_jsonl(path):
        if row.get("answers"):
            by[row["id"]].append(row["answers"])
    return by


def median_answers(runs):
    return {q: statistics.median(probability(row[q]) for row in runs) for q in QUESTIONS}


def cov5(points):
    if not points:
        return None
    best = errors = 0
    for count, (_, correct) in enumerate(sorted(points, key=lambda x: -x[0]), 1):
        errors += not correct
        if errors / count <= 0.05:
            best = count
    return best / len(points)


def composed(gold, answers, review=0.35, action=0.70):
    tp = fp = fn = fast = band = 0
    for key, probs in answers.items():
        decision, _ = compose(probs, review, action)
        label = boolean(gold[key]["gold"]["q1"])
        hit = decision == "fast"
        fast += hit; band += decision == "review"
        tp += hit and label; fp += hit and not label; fn += not hit and label
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fast": fast, "review": band, "fp": fp, "tp": tp, "fn": fn,
            "n": len(answers), "n_failed": len(gold) - len(answers)}


def score(gold, by):
    answers = {key: median_answers(by[key]) for key in gold if by.get(key)}
    result = {"n_total": len(gold), "n": len(answers), "n_failed": len(gold) - len(answers)}
    for question in QUESTIONS:
        points = []; errors = confident = 0; briers = []; labels = []
        for key, ans in answers.items():
            p = ans[question]; label = boolean(gold[key]["gold"][question]); labels.append(label)
            correct = (p >= 0.5) == label; confidence = max(p, 1 - p)
            points.append((confidence, correct)); briers.append((p - label) ** 2)
            if confidence >= 0.9:
                confident += 1; errors += not correct
        n = len(points)
        result[question] = {"accuracy": sum(ok for _, ok in points) / n if n else None,
                            "majority": max(sum(labels), n - sum(labels)) / n if n else None,
                            "hcer_errors": errors, "hcer_n": confident,
                            "hcer": errors / confident if confident else None,
                            "coverage_5pct": cov5(points),
                            "brier": statistics.mean(briers) if briers else None}
    result["Q1"] = composed(gold, answers)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outputs"); p.add_argument("--labels", required=True)
    p.add_argument("--split", choices=("all", "dev", "sel", "test"), default="test")
    p.add_argument("--language")
    a = p.parse_args()
    rows = [r for r in read_jsonl(a.labels) if (a.split == "all" or r["split"] == a.split) and (not a.language or r.get("lang") == a.language)]
    print(json.dumps(score(index_rows(rows), load(a.outputs)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
