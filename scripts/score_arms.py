"""Multi-run vote scoring for generic shape labels, adapted from the study."""
import argparse
import collections
import json
import statistics
from common import index_rows, probability, read_jsonl
from score_battery import cov5


def vote(runs):
    winner = collections.Counter(row["shape"] for row in runs).most_common(1)[0][0]
    # Preserve the historical median over runs, including non-winning runs.
    return winner, statistics.median(probability(row["p"]) for row in runs)


def score(labels, outputs):
    by = collections.defaultdict(list)
    for row in outputs:
        if row.get("shape") is not None:
            by[row["id"]].append(row)
    points = []; briers = []; hce = hcn = none_ok = none_n = 0
    known = []; risk = []
    for key, row in labels.items():
        if not by.get(key):
            continue
        top, p = vote(by[key]); ok = top == row["gold_shape"]; points.append((p, ok)); briers.append((p - ok) ** 2)
        if row["gold_shape"] == "none":
            none_n += 1; none_ok += ok
        if p >= 0.9:
            hcn += 1; hce += not ok
        ks = [r["known"] for r in by[key] if r.get("known") is not None]
        rs = [r["risk"] for r in by[key] if r.get("risk") is not None]
        if ks:
            known.append((statistics.median(ks) >= 0.5) == row["gold_known"])
        if rs:
            risk.append(abs(statistics.median(rs) - row["gold_risk"]))
    n = len(points)
    return {"n": n, "n_failed": len(labels) - n, "shape_accuracy": sum(ok for _, ok in points) / n if n else None,
            "majority": max(collections.Counter(row["gold_shape"] for row in labels.values()).values()) / len(labels) if labels else None,
            "none_recall": f"{none_ok}/{none_n}", "known_accuracy": statistics.mean(known) if known else None,
            "risk_mae": statistics.mean(risk) if risk else None, "hcer_errors": hce, "hcer_n": hcn,
            "coverage_5pct": cov5(points), "top_confidence_brier": statistics.mean(briers) if briers else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outputs"); p.add_argument("--labels", required=True)
    p.add_argument("--split", default="all", choices=("all", "dev", "sel", "test")); p.add_argument("--language")
    a = p.parse_args(); rows = [r for r in read_jsonl(a.labels) if (a.split == "all" or r["split"] == a.split) and (not a.language or r.get("lang") == a.language)]
    print(json.dumps(score(index_rows(rows), read_jsonl(a.outputs)), indent=2))


if __name__ == "__main__":
    main()
