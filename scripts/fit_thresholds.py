"""Fit global thresholds on selection rows; never select thresholds on test."""
import argparse
import itertools
import json
from common import index_rows, read_jsonl
from score_battery import composed, load, median_answers

GRID = [round(x / 100, 2) for x in range(10, 95, 5)]
CAVEAT = "Test rows are reporting-only. A precision change can depend on one relabelled row; the historical test was also used for model selection."


def fit(gold, answers):
    best = None
    for review, action in itertools.product(GRID, GRID):
        if review > action:
            continue
        score = composed(gold, answers, review, action)
        if score["precision"] is None or score["precision"] < 0.98:
            continue
        key = (score["recall"] or 0, score["fast"])
        if best is None or key > best[0]:
            best = (key, review, action)
    return None if best is None else best[1:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outputs"); p.add_argument("--labels", required=True)
    a = p.parse_args(); rows = index_rows(read_jsonl(a.labels)); by = load(a.outputs)
    answers = {key: median_answers(rs) for key, rs in by.items() if key in rows}
    sel = {key: row for key, row in rows.items() if row["split"] == "sel"}
    thresholds = fit(sel, {key: answers[key] for key in sel if key in answers})
    report = {"caveat": CAVEAT, "thresholds": thresholds, "selection_positives": sum(r["gold"]["q1"] for r in sel.values())}
    if thresholds:
        report["review_band_removed"] = thresholds[0] == thresholds[1]
        for split in ("dev", "sel", "test"):
            subset = {key: r for key, r in rows.items() if r["split"] == split}
            report[split] = composed(subset, {key: answers[key] for key in subset if key in answers}, *thresholds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
