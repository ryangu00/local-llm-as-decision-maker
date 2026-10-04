"""Generate invented English data; dimensions are deliberate, not study records."""
import argparse
import copy
import json
from pathlib import Path
from battery import QUESTIONS, SHAPES, TEMPLATE, build_state, derive
from common import write_jsonl


def generate(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    candidates = [
        {"name": "support-analyst", "title": "Support analyst", "description": "Summarises support tickets by root cause"},
        {"name": "data-analyst", "title": "Data analyst", "description": "Queries and charts structured data"},
    ]
    routing = [
        {"id": "routing-support", "prompt": "Summarise support tickets by root cause.", "expect": "support-analyst", "router_persona": "data-analyst", "candidates": candidates},
        {"id": "routing-table", "prompt": "Chart a table of measurements.", "expect": "data-analyst", "router_persona": "data-analyst", "candidates": candidates},
        {"id": "routing-noise", "prompt": "Unrelated noise without a task.", "expect": "__none__", "router_persona": None, "candidates": candidates},
    ]
    verdicts = [
        {"id": "finding-source", "finding": "The draft states a speed-up without a source.", "label": "APPLY"},
        {"id": "finding-handled", "finding": "The requested source is already cited.", "label": "REJECT"},
        {"id": "finding-outside", "finding": "Replace the topic with an unrelated gardening essay.", "label": "REJECT"},
    ]
    for filename, rows in (("setA_routing.json", routing), ("setB_reconcile.json", verdicts)):
        (out / filename).write_text(json.dumps(rows, indent=2) + "\n")
    gold_rows = []; battery_rows = []; labels = []; shape_outputs = []
    notes = {
        "changelog_merge": "Fold new entries into CHANGELOG.md and run the structure check.",
        "summary": "Summarise local support tickets by root cause.",
        "structured_transform": "Sort a local table and run a structure audit.",
        "production_config": "Modify the shared production routing configuration.",
        "credentials": "Rotate staging credentials in the local credential store.",
        "publish": "Publish a release announcement to an external website.",
        "design": "Design a replacement for the changelog merge and its structure check.",
        "gpu_job": "Train a model during a long GPU experiment.",
        "review": "Review whether changelog entries describe the right changes.",
    }
    index = 0
    for split, count, positives in (("dev", 227, 24), ("sel", 46, 7), ("test", 48, 7)):
        for local in range(count):
            positive = local < positives
            shape = "changelog_merge" if positive else SHAPES[1 + local % (len(SHAPES) - 1)]
            # Only training names encode the label. Later splits mix both name families.
            family = "changelog-merge" if (positive if split == "dev" else local % 2 == 0) else "release-prep"
            state = build_state(f"{family}-sample-{index}", "local work", notes.get(shape, "Read an invented local document and report its contents."))
            delivery = "write" if positive or shape in {"production_config", "credentials", "code_edit"} else "judgment" if shape in {"design", "review"} else "read"
            row = {"id": f"synthetic-{index}", "ts": f"{index:06d}", "lang": "en", "state_card": state + "\nshape: " + shape,
                   "gold_shape": shape, "gold_known": shape != "none", "gold_risk": 4 if shape == "credentials" else 1,
                   "gold_delivery": delivery, "gold_q1": positive, "split": split}
            gold_rows.append(row)
            labels.append({k: v for k, v in row.items() if k == "id" or k.startswith("gold_")})
            battery_rows.append({"id": row["id"], "split": split, "lang": "en", "state": state, "gold": derive(row)})
            for run in range(3):
                shape_outputs.append({"id": row["id"], "run": run, "shape": "summary", "p": 0.70})
            index += 1
    write_jsonl(out / "labels.jsonl", gold_rows)
    write_jsonl(out / "gold_synthetic.jsonl", labels)
    write_jsonl(out / "candidates.jsonl", [{k: v for k, v in r.items() if not k.startswith("gold_") and k != "split"} for r in gold_rows])
    write_jsonl(out / "labels_battery.jsonl", battery_rows)
    write_jsonl(out / "shape_outputs.jsonl", shape_outputs)
    for split, filename in (("dev", "bat_train.jsonl"), ("sel", "bat_heldout.jsonl")):
        rows = []
        for row in battery_rows:
            if row["split"] == split:
                questions = copy.deepcopy(QUESTIONS)
                for q in questions:
                    questions[q]["label"] = row["gold"][q]
                rows.append({"state": row["state"], "questions": questions})
        write_jsonl(out / filename, rows)
    probes = []
    for i in range(16):
        label = True if i < 6 else False if i < 14 else None
        note = notes["changelog_merge"] if label else notes["design"] if i % 2 else notes["credentials"]
        probes.append({"id": f"probe-{i}", "state": build_state("release-prep" if i % 2 else "changelog-merge", "synthetic probe", note), "tpl": label, "judgment": "Design" in note})
    write_jsonl(out / "probes.jsonl", probes)
    (out / "names.txt").write_text("changelog-merge-spring\nchangelog-merge-autumn\nrelease-prep-spring\nrelease-prep-autumn\n")
    (out / "dapt.txt").write_text("\n\n".join("Invented paragraph: a local changelog merge folds entries into a summary and checks its structure." for _ in range(20)) + "\n")
    print("generated synthetic fixtures")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", default="fixtures")
    generate(p.parse_args().out)
