"""Run in-process, typed HTTP, first-token readout, or synthetic-stub battery arms."""
import argparse
import json
import os
import urllib.request
from battery import QUESTIONS
from common import probability, read_jsonl, write_jsonl
from decide import fetch_distribution
from synthetic_model import predict


def normalise(result):
    answers = result.get("answers", result)
    return {q: probability(answers[q]) for q in QUESTIONS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("laya", "http", "readout", "stub"))
    p.add_argument("--labels", required=True); p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=1); p.add_argument("--model", default="MODEL_NAME")
    p.add_argument("--subfolder", default="multilingual"); p.add_argument("--device", default="cpu")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    a = p.parse_args()
    if a.runs < 1:
        p.error("--runs must be positive")
    if a.arm == "laya":
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from laya_ft import _load_agent
        import laya
        agent = _load_agent(a.model, a.subfolder, a.device)
    records = []
    for run in range(a.runs):
        for row in read_jsonl(a.labels):
            rec = {"id": row["id"], "run": run, "arm": a.arm}
            try:
                if a.arm == "stub":
                    ans = predict(row["state"])
                elif a.arm == "laya":
                    ans = normalise(laya.decide(agent, row["state"], questions=QUESTIONS))
                elif a.arm == "http":
                    req = urllib.request.Request(a.url, json.dumps({"state": row["state"], "questions": QUESTIONS}).encode(), {"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=300) as response:
                        ans = normalise(json.load(response))
                else:
                    ans = {}
                    for q, spec in QUESTIONS.items():
                        user = f"{spec['instructions']}\nYes: {spec['criteria']['true']}\nNo: {spec['criteria']['false']}\n\n{row['state']}\nAnswer Yes or No."
                        dist, _ = fetch_distribution(a.base, a.model, "Answer only Yes or No.", user, ["Yes", "No"])
                        if dist is None:
                            raise ValueError("no valid first-token distribution")
                        ans[q] = dist["Yes"]
                rec["answers"] = ans
            except Exception as exc:
                rec["err"] = type(exc).__name__ + ": " + str(exc)
            records.append(rec)
    write_jsonl(a.out, records)
    errors = sum("err" in row for row in records)
    print("records", len(records), "failed", errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
