"""Run the public mechanics end to end without dependencies or model weights."""
import argparse
import contextlib
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request
from mock_transport import Transport
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from common import read_jsonl
from decide import metrics, process_B
from mock_server import make_server


def run(root, transport="loopback", data="fixtures"):
    root = Path(root).resolve()
    data = Path(data).resolve()
    if Path.cwd().resolve() not in root.parents:
        raise ValueError("output must be a subdirectory of this checkout")
    root.mkdir(parents=True, exist_ok=True)
    server = None; thread = None
    if transport == "loopback":
        server = make_server(0)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f"http://127.0.0.1:{server.server_port if server else 8000}"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OUTPUT_ROOT": str(root / "readout")}
    if transport == "inprocess":
        env["COOKBOOK_MOCK_MODE"] = "clean"
    else:
        env.pop("COOKBOOK_MOCK_MODE", None)
    completed = []

    def command(name, arguments):
        result = subprocess.run(arguments, text=True, capture_output=True, env=env)
        (root / (name + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect its log")
        completed.append(name)

    def python(name, script, *args):
        wrapper = ["scripts/mock_transport.py"] if transport == "inprocess" else []
        command(name, [sys.executable, *wrapper, "scripts/" + script, *map(str, args)])

    try:
        python("selftest", "decide.py", "--selftest", "--data", data, "--base", base, "--model", "mock")
        command("three-runs", ["bash", "scripts/run3.sh", "--data", str(data), "--base", base, "--model", "mock", "--limit", "3"])
        python("metrics-only", "decide.py", "--metrics-only", root / "readout/run1", "--out", root / "recomputed", "--model", "mock")
        python("order-ablation", "decide.py", "--data", data, "--out", root / "ablation", "--base", base, "--model", "mock", "--set", "A", "--ablate", "--workers", "1")
        python("merge", "merge_gold.py", "--candidates", data / "candidates.jsonl", "--gold", data / "gold_synthetic.jsonl", "--out", root / "merged.jsonl")
        python("derive", "battery.py", root / "merged.jsonl", root / "derived.jsonl")
        for arm in ("stub", "http", "readout"):
            path = root / ("battery-" + arm + ".jsonl")
            python("arm-" + arm, "arm_battery.py", "--arm", arm, "--labels", data / "labels_battery.jsonl", "--out", path, "--base", base, "--url", base + "/v1/systemone", "--model", "mock")
            python("score-" + arm, "score_battery.py", path, "--labels", data / "labels_battery.jsonl")
        python("shape-score", "score_arms.py", data / "shape_outputs.jsonl", "--labels", data / "labels.jsonl")
        python("thresholds", "fit_thresholds.py", root / "battery-stub.jsonl", "--labels", data / "labels_battery.jsonl")
        python("calibration", "calibrate.py", root / "battery-stub.jsonl", "--labels", data / "labels_battery.jsonl")
        python("augment", "build_v2.py", "--data", data, "--out", root / "v2", "--names", data / "names.txt")
        for stub in ("name", "content"):
            python("rename-" + stub, "eval_v2.py", "--data", root / "v2", "--probes", data / "probes.jsonl", "--stub", stub, "--out", root / ("rename-" + stub))
        python("statistics", "stats.py")
        mock = Transport("jitter")
        if server:
            server.mode = "jitter"; server.calls = 0
        verdicts = json.loads((data / "setB_reconcile.json").read_text())
        context = contextlib.nullcontext() if server else patch.object(urllib.request, "urlopen", mock.urlopen)
        with context:
            jitter = [metrics([process_B(base, "mock", row) for row in verdicts], "B") for _ in range(2)]
        assert jitter[0]["accuracy"] == jitter[1]["accuracy"]
        assert jitter[0]["hcer"] != jitter[1]["hcer"]
        (root / "jitter.json").write_text(json.dumps(jitter, indent=2) + "\n")
        original = json.loads((root / "rename-name/summary.json").read_text())
        content = json.loads((root / "rename-content/summary.json").read_text())
        assert original["paired"]["decision_flips"] > 0
        assert content["paired"]["decision_flips"] == 0
        (root / "status.json").write_text(json.dumps({"synthetic_only": True, "transport": transport, "commands": completed, "learned_training_run": False}, indent=2) + "\n")
        print("OFFLINE DEMO PASS: readout, battery, calibration, rename pairing, statistics")
        print("SYNTHETIC ONLY: no learned model was trained or evaluated")
        print("TRANSPORT:", transport)
    finally:
        if server:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out", default="results/offline")
    p.add_argument("--data", default="fixtures", help="synthetic fixture directory")
    p.add_argument("--transport", choices=("loopback", "inprocess"), default="loopback")
    a = p.parse_args(); run(a.out, a.transport, a.data)
