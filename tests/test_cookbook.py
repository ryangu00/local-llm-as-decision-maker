"""Offline regressions for the public mechanics and publication boundaries."""
import ast
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import battery
import build_v2
import calibrate
import common
import dapt_laya
import decide
import eval_v2
import fit_thresholds
import laya_ft
import merge_gold
import mock_server
from mock_transport import Transport
import score_arms
import score_battery
import stats
from synthetic_model import predict


class ReadoutTests(unittest.TestCase):
    def test_mapped_binary_regression(self):
        rows = [{"label": label, "options": ["Yes", "No"], "distribution": {"Yes": yes, "No": 1 - yes}, "top": "Yes", "top_p": yes}
                for label, yes in (("APPLY", 0.9), ("REJECT", 0.6))]
        result = decide.metrics(rows, "B")
        self.assertAlmostEqual(result["brier"], 0.37)
        self.assertAlmostEqual(result["ece"], 0.35)
        self.assertEqual(result["coverage_5pct"], 0.5)
        self.assertEqual(result["coverage_threshold"], 0.9)

    def test_distribution_max_matching_and_missing_floor(self):
        raw = [{"token": " A", "logprob": -1}, {"token": "a", "logprob": -2}, {"token": "irrelevant", "logprob": -3}]
        distribution, logs = decide.build_distribution(raw, ["A", "B"])
        self.assertEqual(logs, {"A": -1, "B": -6})
        self.assertAlmostEqual(sum(distribution.values()), 1)
        self.assertGreater(distribution["A"], distribution["B"])

    def test_invalid_distribution(self):
        self.assertEqual(decide.build_distribution([], ["Yes", "No"]), (None, None))
        self.assertIsNone(decide.build_distribution([{"token": "Yes", "logprob": float("nan")}], ["Yes", "No"])[0])

    def test_scan_past_failing_prefix(self):
        rows = [{"label": "Yes", "options": ["Yes", "No"], "distribution": {"Yes": .6, "No": .4}, "top": "No" if i == 0 else "Yes", "top_p": 1 - i / 100}
                for i in range(20)]
        self.assertEqual(decide.metrics(rows, "A")["coverage_5pct"], 1)

    def test_all_failed_not_zero_metrics(self):
        result = decide.metrics([{"failed": True}, {"failed": True}], "B")
        self.assertEqual(result["n_failed"], 2); self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["brier"])

    def test_cap_candidates(self):
        item = {"id": "synthetic-cap", "prompt": "Pick a handler", "expect": "__none__", "candidates": [{"name": f"handler-{i}"} for i in range(26)]}
        with contextlib.redirect_stderr(io.StringIO()):
            _, _, options, gold, _ = decide.question_A(item)
        self.assertEqual(len(options), 26); self.assertEqual(gold, "Z")

    def test_hash_stable_and_sensitive_to_payload(self):
        first = decide.request_hash("mock", "system", "user")
        self.assertEqual(first, decide.request_hash("mock", "system", "user"))
        self.assertNotEqual(first, decide.request_hash("mock", "system", "changed"))
        self.assertEqual(len(first), 64)

    def test_build_failure_has_no_fabricated_hash(self):
        result = decide.process_B("unused", "mock", {"id": "invalid", "label": "APPLY"})
        self.assertTrue(result["failed"]); self.assertIsNone(result["request_hash"])

    def test_semantic_failure_retries_and_exhausts(self):
        with patch.object(decide, "call_one", return_value={"choices": []}) as call, patch.object(decide.time, "sleep"):
            self.assertEqual(decide.fetch_distribution("unused", "mock", "s", "u", ["Yes", "No"]), (None, None))
        self.assertEqual(call.call_count, 3)

    def test_failure_not_counted_as_escape_agreement(self):
        item = {"id": "noise", "prompt": "noise", "expect": "__none__", "candidates": []}
        with patch.object(decide, "fetch_distribution", side_effect=[(None, None), ({"A": 1}, [])]):
            result = decide.process_A("unused", "mock", item, ablate=True)
        self.assertIsNone(result["ablate_agree"])
        summary = {}; decide.ablate_summary([result], summary)
        self.assertEqual(summary["ablate_failed"], 1)

    def test_empty_selftest_and_limit(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for name in ("setA_routing.json", "setB_reconcile.json"):
                Path(directory, name).write_text("[]")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
                decide.selftest("unused", "mock", directory)
            self.assertEqual(exit.exception.code, 1)
        self.assertEqual(decide._nonneg_int("0"), 0)
        with self.assertRaises(Exception):
            decide._nonneg_int("-1")

    def test_metrics_only_preserves_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            src = Path(directory, "input"); dst = Path(directory, "output"); src.mkdir()
            (src / "B_mock.jsonl").write_text('{"failed": true}\n')
            (src / "summary_mock.json").write_text("untouched")
            with contextlib.redirect_stdout(io.StringIO()):
                decide.run_metrics_only(str(src), "mock", "B", "mock", False, str(dst))
            self.assertEqual((src / "summary_mock.json").read_text(), "untouched")
            self.assertEqual(json.loads((dst / "summary_mock.json").read_text())["sets"]["B"]["n_failed"], 1)


class BatteryTests(unittest.TestCase):
    def answers(self):
        return {q: 0.01 for q in battery.QUESTIONS}

    def test_composition_boundaries_and_priority(self):
        cases = [({}, "full"), ({battery.TEMPLATE: .70}, "fast"), ({battery.TEMPLATE: .35}, "review"),
                 ({battery.TEMPLATE: .99, "egress": .70}, "full"), ({battery.TEMPLATE: .99, "judgment": .35}, "review")]
        for changes, expected in cases:
            self.assertEqual(battery.compose({**self.answers(), **changes})[0], expected)

    def test_bad_probability_fails_closed(self):
        for value in (None, float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                battery.compose({**self.answers(), "egress": value})

    def test_state_builder_strips_shape(self):
        self.assertEqual(battery.state_of({"state_card": "goal: demo\nshape: summary\ntask: local\nnote: invented"}), "goal: demo\ntask: local\nnote: invented")
        self.assertTrue(battery.build_state("demo", "local", "invented").startswith("goal: demo\ntask:"))

    def test_derivation_uses_label_not_name(self):
        row = common.read_jsonl(ROOT / "fixtures/labels.jsonl")[0]
        changed = {**row, "state_card": row["state_card"].replace("changelog-merge", "release-prep")}
        self.assertEqual(battery.derive(row), battery.derive(changed))

    def test_battery_brier_and_failure_denominator(self):
        answers = self.answers(); labels = {q: False for q in answers}; labels["q1"] = False
        gold = {"a": {"gold": labels}, "b": {"gold": labels}}
        result = score_battery.score(gold, {"a": [answers]})
        self.assertEqual(result["n_failed"], 1)
        self.assertAlmostEqual(result["egress"]["brier"], .0001)
        self.assertIsNone(result["Q1"]["precision"])

    def test_band_collapse_for_seven_and_fifty_positives(self):
        for positives in (7, 50):
            gold = {}; answers = {}
            for i in range(positives * 2):
                label = i < positives; key = str(i)
                gold[key] = {"gold": {"q1": label}}
                answers[key] = {**self.answers(), **{q: .3 for q in battery.HARD}, battery.TEMPLATE: .6 if label else .3}
            self.assertEqual(fit_thresholds.fit(gold, answers), (.35, .35))

    def test_platt_decreases_log_loss(self):
        pairs = [(.9, False), (.8, False), (.7, False), (.6, True)]
        a, b = calibrate.fit(pairs)
        def loss(probs):
            return -sum(math.log(p if y else 1 - p) for p, (_, y) in zip(probs, pairs))
        self.assertLess(loss([calibrate.sig(a * calibrate.logit(p) + b) for p, _ in pairs]), loss([p for p, _ in pairs]))
        with self.assertRaises(ValueError):
            calibrate.fit([])

    def test_group_time_split(self):
        rows = [{"id": str(i), "state_card": "goal: " + str(i // 2), "ts": str(i)} for i in range(10)]
        merged = merge_gold.merge(rows, [{"id": row["id"], "gold_q1": False} for row in rows])
        for i in range(0, 10, 2):
            self.assertEqual(merged[i]["split"], merged[i + 1]["split"])
        self.assertEqual(merged[0]["split"], "dev"); self.assertEqual(merged[-1]["split"], "test")

    def test_shape_vote_and_empty_score(self):
        self.assertEqual(score_arms.vote([{"shape": "summary", "p": .9}, {"shape": "summary", "p": .8}, {"shape": "none", "p": .7}]), ("summary", .8))
        self.assertIsNone(score_arms.score({}, [])["shape_accuracy"])


class RenameAndStatsTests(unittest.TestCase):
    def test_strict_labels_and_probabilities(self):
        for label in ("false", 0, 1, None):
            with self.assertRaises(ValueError):
                common.boolean(label)
        self.assertIsNone(common.boolean(None, nullable=True))
        for value in ({}, {"noul": None}, {"probability": float("nan")}, {"noul": True}):
            with self.assertRaises(ValueError):
                eval_v2.prob(value)

    def test_augmentation_preserves_content_and_labels(self):
        rows = common.read_jsonl(ROOT / "fixtures/bat_train.jsonl")
        pool = (ROOT / "fixtures/names.txt").read_text().splitlines()
        augmented = build_v2.augment(rows, pool, 0, 70)
        self.assertEqual(len(augmented), 345)
        self.assertEqual(sum(row["questions"][battery.TEMPLATE]["label"] for row in augmented), 72)
        originals = {(r["state"].split("\n", 1)[1], json.dumps(r["questions"], sort_keys=True)) for r in rows}
        self.assertTrue(all((r["state"].split("\n", 1)[1], json.dumps(r["questions"], sort_keys=True)) in originals for r in augmented))
        self.assertEqual(augmented, build_v2.augment(rows, pool, 0, 70))

    def test_header_failure(self):
        with self.assertRaises(ValueError):
            build_v2.rename("missing header", "new")

    def test_null_probe_labels_excluded(self):
        rows = common.read_jsonl(ROOT / "fixtures/probes.jsonl")
        result, saved = eval_v2.score(rows, predict)
        self.assertEqual(sum(result[k] for k in ("tp", "fp", "fn", "tn")), 14)
        self.assertEqual(len(saved), 16)

    def test_pairing_detects_flips_with_same_aggregate(self):
        left = [{"id": "a", "p_tpl": .9}, {"id": "b", "p_tpl": .1}]
        right = [{"id": "b", "p_tpl": .9}, {"id": "a", "p_tpl": .1}]
        self.assertEqual(stats.paired(left, right)["decision_flips"], 2)
        with self.assertRaises(ValueError):
            stats.paired(left, right[:1])
        with self.assertRaises(ValueError):
            stats.paired(left, [right[0], right[0]])

    def test_exact_intervals_from_table(self):
        cases = [(4, 7, .184, .901), (6, 7, .421, .996), (4, 6, .223, .957), (2, 6, .043, .777),
                 (16, 48, .204, .484), (19, 20, .751, .999), (29, 37, .618, .902), (6, 9, .299, .925), (5, 9, .212, .863)]
        for k, n, low, high in cases:
            with self.subTest(k=k, n=n):
                a, b = stats.clopper_pearson(k, n)
                self.assertEqual(round(a, 3), low); self.assertEqual(round(b, 3), high)

    def test_bounds_and_sample_sizes(self):
        self.assertAlmostEqual(stats.zero_upper(41), .0705, places=4)
        self.assertEqual(round(stats.zero_upper(8) * 100, 1), 31.2)
        self.assertEqual(round(stats.zero_upper(48) * 100, 2), 6.05)
        self.assertEqual(stats.sample_size(.01), 299); self.assertEqual(stats.sample_size(.001), 2995)
        self.assertEqual(round(.05 ** (1 / 6), 3), .607)
        self.assertEqual(stats.mcnemar(3, 1), .625)
        self.assertEqual(stats.mcnemar(0, 0), 1)


class TrainingHelpersTests(unittest.TestCase):
    def test_targets_and_schema_validation(self):
        self.assertEqual(laya_ft._target_for_question({"type": "noul", "label": True}), [0, 1])
        self.assertEqual(laya_ft._target_for_question({"type": "choice", "label": "b", "criteria": {"a": "A", "b": "B"}}), [0, 1])
        with self.assertRaises(ValueError):
            laya_ft._target_for_question({"type": "score", "criteria": ["low"], "label": True})

    def test_make_items_uses_question_targets(self):
        module = types.ModuleType("laya.common")
        module.QTYPES = {"noul": 2}; module.build_sequence = lambda *args: ([1, 2], [0, 1])
        module.render_options = lambda question: [False, True]
        with patch.dict(sys.modules, {"laya": types.ModuleType("laya"), "laya.common": module}):
            items = laya_ft.make_items([{"state": "invented", "questions": {"q": {"type": "noul", "label": True}}}], object(), {})
        self.assertEqual(items[0]["target"], [0, 1]); self.assertEqual(items[0]["label"], 1)

    def test_reward_fallback_only_for_unsupported_primitive(self):
        class Tensor:
            device = "cpu"
            def cpu(self): return self
            def to(self, device): return self
        t = Tensor()
        with patch.object(laya_ft, "_mean", wraps=laya_ft._mean):
            reward = unittest.mock.Mock(side_effect=[RuntimeError("not implemented"), t])
            self.assertIs(laya_ft._proper_reward_cpu_fallback(reward, t, t, t, t), t)
            self.assertEqual(reward.call_count, 2)
        with self.assertRaises(RuntimeError):
            laya_ft._proper_reward_cpu_fallback(unittest.mock.Mock(side_effect=RuntimeError("bad shape")), t, t, t, t)

    def test_dapt_packing_padding_and_empty(self):
        class Tokenizer:
            eos_token_id = 9
            def __call__(self, text, **kwargs): return {"input_ids": [1, 2]}
        blocks = dapt_laya._pack_corpus("invented paragraph\n\nother invented paragraph", Tokenizer(), 4, 0)
        self.assertEqual(blocks[0]["input_ids"], [1, 2, 9, 1])
        self.assertEqual(blocks[1]["attention_mask"], [1, 1, 0, 0])
        with self.assertRaises(ValueError):
            dapt_laya._pack_corpus("", Tokenizer(), 4, 0)


class IntegrationTests(unittest.TestCase):
    def test_offline_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            data = Path(directory, "fixtures")
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            generated = subprocess.run([sys.executable, "scripts/make_fixtures.py", "--out", str(data)], cwd=ROOT, text=True, capture_output=True, env=env)
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            self.assertEqual({p.name for p in data.iterdir()}, {p.name for p in (ROOT / "fixtures").iterdir()})
            for path in data.iterdir():
                self.assertEqual(path.read_bytes(), (ROOT / "fixtures" / path.name).read_bytes(), path.name)
            # Distinguish the regenerated inputs from the checked-in defaults.
            routing_path = data / "setA_routing.json"
            routing = json.loads(routing_path.read_text())
            for row in routing:
                row["id"] = "fresh-" + row["id"]
            routing_path.write_text(json.dumps(routing))
            result = subprocess.run([sys.executable, "scripts/run_offline.py", "--data", str(data), "--out", str(Path(directory) / "output"), "--transport", "inprocess"], cwd=ROOT, text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("OFFLINE DEMO PASS", result.stdout)
            root = Path(directory, "output")
            for run in (1, 2, 3):
                rows = common.read_jsonl(root / f"readout/run{run}/A_mock.jsonl")
                self.assertEqual(len(rows), 3); self.assertTrue(all(r["request_hash"] for r in rows))
                self.assertEqual({r["id"] for r in rows}, {r["id"] for r in routing})
                self.assertEqual(decide.metrics(rows, "A")["accuracy"], 1.0)
            self.assertFalse(json.loads((root / "status.json").read_text())["learned_training_run"])

    def test_routing_mock_uses_task_not_candidate_descriptions(self):
        items = json.loads((ROOT / "fixtures/setA_routing.json").read_text())
        transport = Transport("clean")
        with patch.object(decide.urllib.request, "urlopen", transport.urlopen), patch.object(decide.time, "sleep") as sleep:
            for item in items:
                # Candidate text must not supply routing keywords for the task.
                item = copy.deepcopy(item)
                for candidate in item["candidates"]:
                    candidate["description"] += "; support tickets; unrelated noise"
                with self.subTest(item=item["id"]):
                    row = decide.process_A("http://127.0.0.1:8000", "mock", item, ablate=True)
                    self.assertFalse(row.get("failed", False))
                    self.assertEqual(row["top"], row["label"])
                    self.assertTrue(row["ablate_agree"])
            sleep.assert_not_called()

    def test_null_logprobs_serialized_http(self):
        transport = Transport("none-logprobs")
        with patch.object(decide.urllib.request, "urlopen", transport.urlopen), patch.object(decide.time, "sleep"):
            result = decide.process_B("http://127.0.0.1:8000", "mock", {"id": "bad-logprobs", "finding": "invented", "label": "APPLY"})
        self.assertTrue(result["failed"]); self.assertEqual(transport.server.calls, 3)

    def test_inprocess_arm_with_mocked_library(self):
        import arm_battery
        module = types.ModuleType("laya")
        module.load = lambda *args, **kwargs: object()
        module.decide = lambda agent, state, questions: {"answers": {q: {"noul": value} for q, value in predict(state).items()}}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            labels = Path(directory, "labels.jsonl"); output = Path(directory, "output.jsonl")
            common.write_jsonl(labels, common.read_jsonl(ROOT / "fixtures/labels_battery.jsonl")[:3])
            argv = ["arm_battery.py", "--arm", "laya", "--model", "synthetic", "--labels", str(labels), "--out", str(output)]
            with patch.dict(sys.modules, {"laya": module}), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                arm_battery.main()
            self.assertEqual(len(common.read_jsonl(output)), 3)


class PublicationTests(unittest.TestCase):
    def test_source_syntax(self):
        for path in list((ROOT / "scripts").glob("*.py")) + [ROOT / "docs/make_banner.py"]:
            ast.parse(path.read_text(), filename=path.name)

    def test_readme_length_and_no_forbidden_character_ranges(self):
        self.assertTrue(250 <= len((ROOT / "README.md").read_text().splitlines()) <= 450)
        forbidden = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
        for parent in (ROOT / "docs", ROOT / "scripts", ROOT / "tests", ROOT / "fixtures", ROOT / "configs"):
            for path in parent.rglob("*"):
                if path.is_file() and path.suffix not in {".pyc", ".png"}:
                    self.assertIsNone(forbidden.search(path.read_text()), str(path.relative_to(ROOT)))

    def test_fixture_name_regularities_only_in_training(self):
        rows = common.read_jsonl(ROOT / "fixtures/labels_battery.jsonl")
        train = [r for r in rows if r["split"] == "dev"]
        self.assertTrue(all(("changelog-merge" in r["state"].splitlines()[0]) == r["gold"][battery.TEMPLATE] for r in train))
        test = [r for r in rows if r["split"] == "test"]
        self.assertTrue(any(("changelog-merge" in r["state"].splitlines()[0]) != r["gold"][battery.TEMPLATE] for r in test))


if __name__ == "__main__":
    unittest.main()
