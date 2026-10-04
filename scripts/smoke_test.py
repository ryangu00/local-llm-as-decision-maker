"""CPU train/export/reload check with three invented English rows; local weights required."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def row(shape, known, risk):
    return {"state": f"Synthetic task: {shape}. Known shape: {known}. Risk category: {risk}.",
            "questions": {
                "shape": {"type": "choice", "instructions": "Which task shape fits?", "criteria": {"summary": "Summarise a document", "transform": "Transform a table", "none": "Neither job"}, "label": shape},
                "known": {"type": "noul", "instructions": "Is this a listed task shape?", "label": known},
                "risk": {"type": "score", "instructions": "Which stated risk category applies?", "criteria": ["reversible local", "limited reversible", "shared state", "major loss", "irreversible"], "label": risk},
            }}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True); p.add_argument("--subfolder", default="multilingual")
    p.add_argument("--work-dir", default="results")
    a = p.parse_args(); work = Path(a.work_dir); work.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rows = [row("summary", True, 1), row("transform", False, 2), row("none", True, 0)]
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=work) as directory:
        root = Path(directory); train = root / "train.jsonl"; heldout = root / "heldout.jsonl"; out = root / "model"
        for path in (train, heldout):
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        subprocess.run([sys.executable, str(Path(__file__).with_name("laya_ft.py")), "--train", str(train), "--heldout", str(heldout), "--base", a.base, "--subfolder", a.subfolder, "--device", "cpu", "--epochs", "1", "--batch-size", "3", "--grad-accum", "1", "--out", str(out)], check=True)
        import laya
        agent = laya.load(str(out), device="cpu", fast=False)
        schema = {"type": "object", "properties": {"shape": {"type": "string", "enum": ["summary", "transform", "none"]}, "known": {"type": "boolean"}, "risk": {"type": "integer", "minimum": 0, "maximum": 4}}}
        for item in rows:
            answer = agent.decide(item["state"], schema=schema)
            assert set(answer) == {"shape", "known", "risk"}, answer
        print("CPU train/export/reload smoke passed")


if __name__ == "__main__":
    main()
