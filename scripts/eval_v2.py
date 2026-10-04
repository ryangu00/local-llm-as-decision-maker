"""Strict paired rename scoring, using a local checkpoint or explicitly synthetic stubs."""
import argparse
import json
import os
from pathlib import Path
from battery import QUESTIONS, TEMPLATE
from common import boolean, probability, read_jsonl, write_jsonl
from stats import paired
from synthetic_model import predict

prob = probability


def score(rows, decide):
    tp = fp = fn = tn = jtp = jfp = jfn = jtn = 0; saved = []
    for row in rows:
        label = boolean(row["tpl"], nullable=True); judgment = boolean(row["judgment"], nullable=True)
        answers = decide(row["state"]); pt = prob(answers[TEMPLATE]); pj = prob(answers["judgment"])
        hit, jhit = pt >= 0.70, pj >= 0.35
        if label is not None:
            tp += hit and label; fp += hit and not label; fn += not hit and label; tn += not hit and not label
        if judgment is not None:
            jtp += jhit and judgment; jfp += jhit and not judgment; jfn += not jhit and judgment; jtn += not jhit and not judgment
        saved.append({"id": row["id"], "tpl": label, "judgment": judgment, "p_tpl": pt, "p_judgment": pj})
    jn = jtp + jfp + jfn + jtn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "judgment_accuracy": (jtp + jtn) / jn if jn else None,
            "judgment_recall": jtp / (jtp + jfn) if jtp + jfn else None,
            "judgment_tp": jtp, "judgment_fp": jfp, "judgment_fn": jfn, "judgment_tn": jtn}, saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True); p.add_argument("--probes", required=True); p.add_argument("--out", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--stub", choices=("name", "content")); group.add_argument("--model")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.stub:
        decide = lambda state: predict(state, a.stub)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import laya
        agent = laya.load(a.model, device=a.device, fast=False)
        def decide(state):
            out = laya.decide(agent, state, questions={q: QUESTIONS[q] for q in (TEMPLATE, "judgment")})
            return out.get("answers", out)
    report = {"synthetic_stub": a.stub}; saved = {}
    for name, path in (("test", Path(a.data) / "eval_test.jsonl"), ("swap", Path(a.data) / "eval_swap.jsonl"), ("probe", Path(a.probes))):
        report[name], saved[name] = score(read_jsonl(path), decide)
        write_jsonl(Path(a.out) / (name + ".jsonl"), saved[name])
    report["paired"] = paired(saved["test"], saved["swap"])
    Path(a.out, "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
