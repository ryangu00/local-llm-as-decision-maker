# Chapter 3: Fine-tune on your own labels, and measure what it bought

### Problem

Zero-shot arms either never act (Kev: everything in the review band) or act on the wrong side (Laya: confidently reversed). A 421M encoder can be fine-tuned on a single local GPU. What does 227 rows buy, how well calibrated is it, and which of the usual fixes help?

### What we built or changed

**Laya fine-tuning port (`laya_ft.py`, a port of the upstream Kaggle notebook to a single process on CPU or a local GPU).** Training objective copied from the notebook, not invented: a proper-scoring-rule reward with a GRPO-style policy-gradient term plus a soft cross-entropy term on one-hot targets.

| Setting | Value |
|---|---|
| Base | `convaiinnovations/laya`, subfolder `multilingual`, `laya==0.3.20`, fast path off |
| Rows | typed JSONL: `state` plus `questions` (each with `type` = choice / noul-style boolean / score, `instructions`, `criteria`, `label`); each row is expanded into one item per question |
| Sequence budget | `max_len` 1024, `head_max_len` 256, gradient checkpointing on |
| Optimiser | AdamW, weight decay 0.01, encoder learning rate `--lr` (2.5e-5), head learning rate 4x; cosine schedule to 1e-6; batch 8, gradient accumulation 4; gradient clipping 1.0 |
| Policy-gradient term | group size 4 zero-mean logit perturbations, noise sigma annealed from 0.4 to 0.1 over the epochs, group-normalised advantages |
| Reward op | computed on CPU when the selected device lacks a primitive |
| Precision | fp32, no autocast |
| Temperature fit | scalar temperature per question type fitted with L-BFGS on the held-out rows, clamped to [0.1, 10]; the checkpoint's inherited per-option-count temperature table is removed because the upstream README says stale values silently override the new fit |
| Seed | 20260922 (one seed) |
| Export | half-precision `model.safetensors`, `encoder/` config, `tokenizer/`, `rl_agent_config.json` marked fine-tuned |
| Smoke test | three synthetic rows, one CPU epoch, reload the exported directory, call `decide` for all three types |

The table records the historical run. The public script defaults to `--seed 0`
for new runs; pass `--seed 20260922` to select the recorded study seed.
Using that seed with synthetic fixtures does not reproduce the private study.

Battery run: dev 227 rows x 7 questions as training items, sel 46 rows as the temperature-fitting hold-out, **6 epochs**; test never touched during training.

**Kev fine-tuning.** `--init_from jaredpalmer/kev-0.8b`, 2 epochs, learning rate 2e-5, fp32, batch 1 on a local GPU; the 4B variant was queued with `--init_from jaredpalmer/kev-4b`, accumulation 8 and activation checkpointing (its result is not recorded in any source).

**Domain pre-training arm (DAPT).** Continue the encoder with masked-language-modelling on 8.3 MB of the operator's own notes (2,177 pages, 49,934 paragraphs, 6,547 sensitive paragraphs removed), one epoch, learning rate 5e-5, max length 256, mask probability 0.15, batch 16, a randomly initialised MLM head from the base architecture's config (encoder weights from the checkpoint; missing/unexpected key lists both empty), then the same battery fine-tune. Gate set in advance: composed precision at least 0.98 and the three weakest questions at least 5 points better.

**Per-question Platt calibration (`calibrate.py`).** For each yes/no question fit `p' = sigmoid(a * logit(p) + b)` by gradient descent on the 46 sel rows, evaluate on test.

**Public CPU reproduction commands (not historical hardware descriptions).**
All dependencies and base checkpoint files must already be local. These training
commands were not executed as part of the offline checks.
The recorded environment was Python 3.11, `laya==0.3.20`, `torch` 2.14, fp32,
on a local GPU. Other dependency versions and upstream commits were not recorded.

```bash
python3 scripts/smoke_test.py --base models/laya --subfolder multilingual
python3 scripts/laya_ft.py --train fixtures/bat_train.jsonl --heldout fixtures/bat_heldout.jsonl --base models/laya --subfolder multilingual --device cpu --epochs 6 --lr 2.5e-5 --seed 0 --out results/laya-battery
python3 scripts/dapt_laya.py --base models/laya --subfolder multilingual --corpus fixtures/dapt.txt --out results/laya-dapt-smoke --device cpu --epochs 1 --lr 5e-5 --max-len 256 --mask-prob 0.15 --batch 16 --max-steps 3
```

The synthetic corpus has 20 paragraphs. The DAPT command is capped at three steps;
a short tokenised corpus can exhaust earlier. It is a mechanics check, not a
replication of the private corpus experiment. For actual battery training, export
only dev and sel; never include the test labels in training or temperature fitting.

The supplied port follows the public Laya Kaggle notebook's proper-scoring reward,
GRPO-style term and soft cross-entropy term. Public credit:
[Laya](https://huggingface.co/convaiinnovations/laya), reported Apache-2.0 in the study.
The notebook URL, upstream licence texts and exact revision were not archived;
verify them before redistributing third-party code or weights. No upstream weights
or dependencies are bundled. The runnable port preserves the objective and export
layout, uses offline loading, and defaults to CPU for public smoke checks.

Only this command shape was recorded for the queued larger Kev run; the public
example uses a CPU device placeholder rather than the historical device setting:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m kev.train --data fixtures/kev_train.jsonl --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype fp32 --device cpu --max_state 1024 --checkpointing 1 --out results/kev-4b
```

This is a reference argument list, not a supported offline reproduction step:
the Kev training dependency revision and its export schema were not supplied,
`fixtures/kev_train.jsonl` is deliberately not fabricated, and no Kev-4B result is
recorded. The process queue wrapper is not shipped.

### Procedure as run

1. Export labels to the typed JSONL format for each library (dev = train, sel = hold-out, test not exported).
2. Fine-tune Laya (6 epochs) and Kev-0.8B (2 epochs), then score exactly like the zero-shot arms (three inference runs, median).
3. Fit thresholds on sel; fit Platt on sel; run the DAPT arm; compare on the same 48 test rows.
4. After the test scoring, serve the Laya model as a shadow service that records answers but changes no routing.

### Results

Per-question and composed results are in Tables 2.2 and 2.3 (fine-tuned rows). Additional first-hand facts:

| Item | Value |
|---|---|
| Laya battery fine-tune, test | template 1.000 (0 confident errors of 46 confident answers); composed precision 6/6, recall 6/9; one false fast on the pre-correction labels (the disputed row) |
| Fitted temperature, battery model | at the clamp ceiling of 10.0 for the model that was served; the 24-way 4-epoch model's fit was choice 1.99, score 1.94, boolean 0.90 |
| Over-confidence on test (confident errors / confident answers) | `writes` 15/40, `judgment` 10/41, `egress` 9/45, `gpu_long` 7/46, `deterministic` 5/43, `prod_shared` 4/39 |
| Kev-0.8B fine-tuned | composed 1.00 precision, 0.556 recall; hard-block accuracy 0.79 to 0.92; fewer confident answers (see Table 2.2) |
| Parameters of the fitted Platt maps (a, b) | writes (0.33, -1.04), deterministic (0.69, 0.43), egress (0.23, -0.25), prod_shared (0.23, 0.35), judgment (0.39, 0.60), gpu_long (0.45, -1.63), template (1.33, -0.01) |

**Table 3.1 — Platt calibration on the Laya fine-tuned arm (fit on 46 sel rows, evaluated on test).**

| Question | Raw confident errors / confident answers | Platt |
|---|---|---|
| writes | 15/40 | 0/0 (no confident answers left) |
| deterministic | 5/43 | 3/35 |
| egress | 9/45 | 0/0 |
| prod_shared | 4/39 | 0/0 |
| judgment | 10/41 | 0/0 |
| gpu_long | 7/46 | 8/46 (worse) |
| template | 0/46 | 0/46 |
| Composed Q1 | precision 1.00, recall 0.67, fast 6 | precision 1.00, recall 0.56, fast 5 |

Platt cleared the confident errors by making the model stop being confident (slopes 0.23 to 0.45), and cost one fast decision. Two parameters per question on 46 rows (7 positives for the composed label) is thin; the decision was to serve raw probabilities with default thresholds and to monitor the confident-error rate.

**Table 3.2 — domain pre-training A/B on test (gate: precision at least 0.98 and weak questions plus 5 points).**

| Arm | writes | gpu_long | judgment | template | Composed precision / recall |
|---|---|---|---|---|---|
| Fine-tuned on battery labels only | 0.604 | 0.833 | 0.729 | 1.000 (0/46) | 1.00 / 0.667 |
| Domain pre-trained, then fine-tuned | 0.750 | 0.729 | 0.792 | 0.833 (5/43) | 0.857 / 0.667 |
| Change | +14.6 pts | -10.4 pts | +6.3 pts | -16.7 pts, 5 confident errors | one false fast |

Not adopted: the template question worsened in this run. The study interpretation was that general notes diluted the task-specific signal; that cause was not isolated.

### What did not work

- **Fitting thresholds on 46 rows (7 positives).** Fits collapse the review band in 4 of the 5 arms where a fit exists; see Chapter 2.
- **Platt calibration on 46 rows.** Removed confident errors by removing confidence, hurt `gpu_long`, cost recall 0.67 to 0.56.
- **Temperature fitting.** The fitted temperature hit its ceiling of 10.0 for the served model; temperature scaling could not fix an over-confident model with only 46 hold-out rows.
- **Domain-adaptive pre-training.** One run, one seed, one epoch, a corpus that is notes rather than task cards: precision 1.00 to 0.857, template accuracy 1.000 to 0.833 with 5 confident errors.
- **Hard-block questions** stayed at or near majority accuracy after fine-tuning (Table 2.2).

### Pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| Fine-tune script cannot run on a fresh machine | Notebook code assumed a multi-device setup and device-specific optimisations | Port to single-process fp32; fall back to CPU for an unsupported reward primitive; keep an explicit smoke test that reloads the exported directory and calls `decide` |
| Fitted temperature silently ignored after fine-tune | The checkpoint carries a per-option-count temperature table that overrides a new global temperature | Remove the table from the exported config when you write your own fit |
| Hold-out used for temperature is also what you "selected on" | Only 46 rows exist | Say so; do not call it a validation score |
| Serving different from training | See Chapter 2 pitfalls (prefix and wording) | Share one builder |

### The rule

Fine-tuning a 0.4B decision encoder on a few hundred rows can lift a literal question to near-perfect on a small test set, but expect it to be over-confident, do not fit thresholds or calibration on a hold-out with fewer than about 50 positives, and compare every accuracy against the majority-class rate.

## Evaluate and retrain the synthetic battery with local weights

After the CPU smoke succeeds, the earlier battery-training command writes an
exported checkpoint. Use the shared registry for both original and augmented runs:

```bash
python3 scripts/arm_battery.py --arm laya --model results/laya-battery --subfolder '' --device cpu --labels fixtures/labels_battery.jsonl --runs 3 --out results/laya-answers.jsonl
python3 scripts/score_battery.py results/laya-answers.jsonl --labels fixtures/labels_battery.jsonl
python3 scripts/build_v2.py --data fixtures --out results/v2 --names fixtures/names.txt
python3 scripts/laya_ft.py --train results/v2/bat_train.jsonl --heldout results/v2/bat_heldout.jsonl --base models/laya --subfolder multilingual --device cpu --epochs 6 --lr 2.5e-5 --seed 0 --out results/laya-renamed
python3 scripts/eval_v2.py --data results/v2 --probes fixtures/probes.jsonl --model results/laya-battery --device cpu --out results/paired-original
python3 scripts/eval_v2.py --data results/v2 --probes fixtures/probes.jsonl --model results/laya-renamed --device cpu --out results/paired-retrained
```

These are executable recipes once the local prerequisites exist, not commands
executed for this publication. The same public seed default is used in both
training commands. Whether either learned model uses the planted name prefix is
an outcome to measure; the programmed stubs cannot answer that question.
The public common-name sampler also differs from the historical augmentation.
