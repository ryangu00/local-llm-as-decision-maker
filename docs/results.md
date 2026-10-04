# Results and conditions

All empirical results below are historical aggregates on private data. No private
rows or per-row outputs are included. The readout study was 2026-09-22; the
battery and fine-tuning study was 2026-09-27; the rename test was 2026-09-28.
Recomputed figures were recomputed in 2026-10 from the saved study outputs.
The public synthetic checks do not reproduce these measurements.

## Label agreement

Conditions: 2026-09-27; blind second adjudication of 50 randomly drawn private rows.

**Label quality (blind second adjudication, 50 rows):**

| Label | Agreement | Action taken |
|---|---|---|
| Fast-path eligibility (the composed "Q1" label) | 43/50 (0.86); gate was 0.85 | accepted |
| Task shape (24 classes) | 35/50 (0.70) | below the 0.85 gate; 11 rows re-adjudicated under a tightened rubric (8 numbered rules); 4 disagreements kept with the reason recorded |
| "Is a known shape available" | 39/50 (0.78) | not used downstream |
| Risk level (5 levels) | 37/50 exact, 49/50 within one level | not used as a model target after Chapter 2 |
| Delivery type (read / write / judgment) | 44/50 (0.88) | used to derive two battery labels |

## Table 1.1 - Formal readout medians

Conditions: 2026-09-22; three sequential runs; A n=37, B n=108; shared production endpoint.

**Table 1.1 — the formal three-run result** (median, range in brackets; endpoint shared with production; one worker; recomputed from the three summaries).

| Set | n | Accuracy | Brier | ECE | HCER@0.9 | Coverage @5 % | Items whose top option was identical in all 3 runs | Mean / max spread of `top_p` across runs |
|---|---|---|---|---|---|---|---|---|
| A routing | 37 | 0.784 (0.784-0.811) | 0.336 (0.331-0.356) | 0.149 (0.145-0.153) | 0.100 (0.069-0.107) | 0.541 (0.432-0.541) | 36/37 | 0.034 / 0.166 |
| B verdicts | 108 | 0.657 (0.648-0.685) | 0.449 (0.440-0.472) | 0.079 (0.075-0.119) | 0.077 (0.000-0.111) | 0.102 (0.028-0.111) | 89/108 | 0.114 / 0.286 |

## Table 1.2 - Predefined gate outcome

Conditions: 2026-09-22; formal three-run medians; A n=37, B n=108.

**Table 1.2 — gate outcome** (formal medians).

| Gate condition | Set A | Set B |
|---|---|---|
| Accuracy >= 0.85 | fail (0.784) | fail (0.657) |
| Accuracy not below baseline | pass (router 0.514) | the report scored B as failing all six conditions; against the 0.5 majority baseline its 0.657 is above, so this row depends on which baseline is meant |
| Brier <= 0.12 | fail (0.336) | fail (0.449) |
| ECE <= 0.05 | fail (0.149) | fail (0.079) |
| High-confidence error <= 3 % | fail (0.100) | fail (0.077) |
| Coverage @5 % >= 40 % | pass (0.541; the weakest run was 0.432) | fail (0.102) |

## Table 1.3 - Readout repeatability

Conditions: 2026-09-22; the same 37 routing items; worker counts shown below; recomputed in 2026-10.

**Table 1.3 — the same Set A, four runs of the same code on the same data** (this is what exposed the instrument problem; run v0 and v1 used 4 workers, det-1 and det-2 used 1 worker, recomputed).

| Run | Accuracy | Brier | ECE | HCER@0.9 (n at or above 0.9) | Coverage @5 % |
|---|---|---|---|---|---|
| v0 | 0.784 | 0.348 | 0.150 | 0.071 (28) | 0.189 |
| v1 | 0.784 | 0.346 | 0.158 | 0.138 (29) | 0.378 |
| det-1 | 0.784 | 0.355 | 0.161 | 0.138 (29) | 0.189 |
| det-2 | 0.784 | 0.336 | 0.145 | 0.069 (29) | 0.568 |
| median | 0.784 | 0.347 | 0.154 | 0.105 | 0.284 |

## Table 1.4 - Candidate coverage breakdown

Conditions: 2026-09-22; first routing run, n=37; cross-run qualifications below.

**Table 1.4 — where it is right and wrong, Set A** (first run; "in candidates 19/20, junk 9/10" held in all seven runs; the missing-candidate count was 1/7 in six runs and 2/7 in the formal third run).

| Category | n | Readout right | Incumbent router right | Note |
|---|---|---|---|---|
| Correct handler among the candidates | 20 | 19 | 9 | the one error had `top_p` 0.985 |
| Correct handler not among the candidates (right answer: escalate) | 7 | 1 | 0 | 6 of 7 chose a nearby handler, `top_p` 0.44 to 0.996; 6 of the 8 errors in this run are here |
| Junk input (right answer: escalate) | 10 | 9 | 10 | the one error had `top_p` 0.87 |

## Battery question meanings

Conditions: 2026-09-27 design; public English paraphrases, not private question text.

**The battery.** One request carries seven literal yes/no questions. Each has an instruction sentence and two criteria sentences (what makes the answer true, what makes it false):

| Question id | Meaning (English paraphrase) |
|---|---|
| `writes` | Does the task create or modify files, code, configuration or services? |
| `deterministic` | Can the deliverable be verified mechanically by a test, script or structure audit, with no human judgment? |
| `egress` | Does the task call a cloud model, reach the public internet, publish or send something out? |
| `prod_shared` | Does the task modify production files shared by several parties, or handle credentials? |
| `judgment` | Is the deliverable itself a verdict, design, review opinion or decision? |
| `gpu_long` | Does the task need more than 10 minutes of long model download, compile, training, benchmark or inference on a GPU machine? |
| `tpl_<name>` | Is this task exactly the one pre-approved template (one question per template; here a recurring "merge new entries into a summary file and run a structure check" job)? |

## Battery arm mechanisms

Conditions: 2026-09-27; six local battery arms; mechanisms rather than measured scores.

**Six arms**, each scored with the same script (per-question accuracy, HCER@0.9 on `max(p, 1-p)`, coverage at 5 %, Brier, composed precision/recall):

| Arm | How it reads the answer |
|---|---|
| Large local LLM readout | one first-token request per question (`max_tokens=2`, `top_logprobs=20`, thinking off), `Yes`/`No` softmax |
| Laya, zero-shot | in-process typed question heads, `multilingual` checkpoint |
| Kev-0.8B, zero-shot | HTTP typed-decision server |
| Laya fine-tuned on the battery labels | see Chapter 3 |
| Kev-0.8B fine-tuned on the battery labels | see Chapter 3 |
| Laya with domain pre-training then fine-tuning | see Chapter 3 |

## Table 2.1 - Shape classification

Conditions: 2026-09-27; all n=321, test n=48, sel n=46; corrected labels, recomputed in 2026-10.

**Table 2.1 — 24-way question, zero-shot arms and a fine-tuned arm** (321 rows unless stated; recomputed after the 11-row re-adjudication; chance 1/24 = 0.042).

| Arm | Shape accuracy, all 321 | Test (48) | Sel (46) | HCER@0.9 (confident errors / confident answers), all | Coverage @5 %, all |
|---|---|---|---|---|---|
| Large local LLM readout | 0.209 | 0.229 | 0.130 | 13/40 | 0.012 |
| Laya zero-shot | 0.178 | 0.229 | 0.130 | 32/51 | 0.000 |
| Kev-0.8B zero-shot | 0.162 | 0.125 | 0.087 | 0/0 | 0.006 |
| Laya fine-tuned on 24-way, 4 epochs (dev rows are in training) | 0.604 (not held out) | 0.333 | 0.283 | 0/5 | 0.312 |

## Table 2.2 - Per-question battery results

Conditions: 2026-09-27; corrected test labels, n=48; confident-error denominators shown in cells.

**Table 2.2 — battery, test set (48 rows), per question: accuracy, then confident errors / confident answers (HCER@0.9).** Recomputed with the corrected gold labels. The last row is the accuracy of always answering the majority class, computed from the label counts (positives in test: writes 28, deterministic 9, egress 9, prod_shared 5, judgment 13, gpu_long 10, template 7).

| Arm | writes | deterministic | egress | prod_shared | judgment | gpu_long | template |
|---|---|---|---|---|---|---|---|
| Large local LLM readout | 0.604 (4/19) | 0.542 (2/8) | 0.812 (9/47) | 0.917 (2/44) | 0.479 (5/13) | 0.833 (6/37) | 0.854 (1/26) |
| Laya zero-shot | 0.500 (2/9) | 0.167 (26/29) | 0.292 (6/6) | 0.292 (16/20) | 0.375 (17/25) | 0.229 (17/24) | 0.167 (37/38) |
| Kev-0.8B zero-shot | 0.542 (0/0) | 0.771 (0/0) | 0.812 (0/1) | 0.896 (0/0) | 0.833 (0/0) | 0.354 (0/0) | 0.979 (0/3) |
| Laya fine-tuned | 0.604 (15/40) | 0.854 (5/43) | 0.771 (9/45) | 0.896 (4/39) | 0.729 (10/41) | 0.833 (7/46) | 1.000 (0/46) |
| Kev-0.8B fine-tuned | 0.750 (0/4) | 0.812 (0/1) | 0.812 (1/22) | 0.917 (0/4) | 0.792 (0/11) | 0.792 (5/32) | 0.979 (0/44) |
| Laya domain-pre-trained then fine-tuned | 0.750 (11/46) | 0.833 (6/40) | 0.812 (9/46) | 0.812 (6/43) | 0.792 (9/46) | 0.729 (11/46) | 0.833 (5/43) |
| **Always the majority class** | **0.583** | **0.812** | **0.812** | **0.896** | **0.729** | **0.792** | **0.854** |

## Table 2.3 - Composed fast-path decisions

Conditions: 2026-09-27; test n=48, 9 composed positives; default thresholds.

**Table 2.3 — composed fast-path decision on test (9 positives of 48; "fast" = rule output `fast`).**

| Arm | Precision | Recall | Fast | Review | False fast |
|---|---|---|---|---|---|
| Large local LLM readout | 0.00 | 0.00 | 1 | 16 | 1 |
| Laya zero-shot | 0.50 | 0.111 | 2 | 7 | 1 |
| Kev-0.8B zero-shot | none | 0.00 | 0 | 44 | 0 |
| Laya fine-tuned | 1.00 (6/6) | 0.667 (6/9) | 6 | 8 | 0 |
| Kev-0.8B fine-tuned | 1.00 (5/5) | 0.556 (5/9) | 5 | 15 | 0 |
| Laya domain-pre-trained then fine-tuned | 0.857 (6/7) | 0.667 (6/9) | 7 | 4 | 1 |

## Table 2.4 - Confounded headline comparison

Conditions: 2026-09-27; same 48 test rows; task, baseline, input and epochs differ.

**Table 2.4 — the comparison the headline is built on, and why it is not like-for-like.**

| | 24-way shape | Template yes/no question |
|---|---|---|
| Model | Laya fine-tuned (4 epochs) | Laya fine-tuned (6 epochs, battery labels) |
| Test accuracy | 0.333 (0.354 before re-adjudication) | 1.000 |
| Chance / majority | 0.042 / 0.208 | 0.854 (7 positives of 48) |
| State card | contained the hand-written shape line | shape line stripped |
| Label space | 24 classes, 6 of 20 present dev classes under 5 examples | one binary question, 24 positives in dev |
| What changed together | question, label space, base rate, state text, epochs | |

## Threshold fitting

Conditions: 2026-09-27; fit on sel n=46 with 7 positives; report on test n=48 with corrected labels.

**Threshold fitting (global review and action thresholds fitted on the 46 sel rows, 7 positives, target precision at least 0.98 with maximal recall).**

| Arm | Fitted (review, action) | Test precision / recall / fast / false fast |
|---|---|---|
| Laya fine-tuned | 0.35 / 0.35 | 1.00 / 0.778 / 7 / 0 |
| Kev-0.8B fine-tuned | 0.35 / 0.35 | 1.00 / 0.667 / 6 / 0 |
| Kev-0.8B zero-shot | 0.55 / 0.55 | 1.00 / 0.444 / 4 / 0 |
| Laya domain-pre-trained | 0.10 / 0.10 | 0.75 / 0.667 / 8 / 2 |
| Large local LLM readout | 0.45 / 0.90 | no fast decisions |
| Laya zero-shot | no threshold reaches precision 0.98 on sel | n/a |

## Fine-tuning settings

Conditions: 2026-09-27; dev n=227, sel n=46, one seed; recorded configuration, not new measurements.

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

## Additional fine-tuning observations

Conditions: 2026-09-27; test n=48; Platt maps and temperatures fitted on sel n=46.

Per-question and composed results are in Tables 2.2 and 2.3 (fine-tuned rows). Additional first-hand facts:

| Item | Value |
|---|---|
| Laya battery fine-tune, test | template 1.000 (0 confident errors of 46 confident answers); composed precision 6/6, recall 6/9; one false fast on the pre-correction labels (the disputed row) |
| Fitted temperature, battery model | at the clamp ceiling of 10.0 for the model that was served; the 24-way 4-epoch model's fit was choice 1.99, score 1.94, boolean 0.90 |
| Over-confidence on test (confident errors / confident answers) | `writes` 15/40, `judgment` 10/41, `egress` 9/45, `gpu_long` 7/46, `deterministic` 5/43, `prod_shared` 4/39 |
| Kev-0.8B fine-tuned | composed 1.00 precision, 0.556 recall; hard-block accuracy 0.79 to 0.92; fewer confident answers (see Table 2.2) |
| Parameters of the fitted Platt maps (a, b) | writes (0.33, -1.04), deterministic (0.69, 0.43), egress (0.23, -0.25), prod_shared (0.23, 0.35), judgment (0.39, 0.60), gpu_long (0.45, -1.63), template (1.33, -0.01) |

## Table 3.1 - Platt calibration

Conditions: 2026-09-27; fit on sel n=46; evaluate test n=48; corrected labels.

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

## Table 3.2 - Domain pre-training

Conditions: 2026-09-27; one run and seed; test n=48 after the same battery fine-tune.

**Table 3.2 — domain pre-training A/B on test (gate: precision at least 0.98 and weak questions plus 5 points).**

| Arm | writes | gpu_long | judgment | template | Composed precision / recall |
|---|---|---|---|---|---|
| Fine-tuned on battery labels only | 0.604 | 0.833 | 0.729 | 1.000 (0/46) | 1.00 / 0.667 |
| Domain pre-trained, then fine-tuned | 0.750 | 0.729 | 0.792 | 0.833 (5/43) | 0.857 / 0.667 |
| Change | +14.6 pts | -10.4 pts | +6.3 pts | -16.7 pts, 5 confident errors | one false fast |

## Shortcut probe categories

Conditions: 2026-09-28; six hand-built probes; categories describe inputs without publishing their text.

**Finding the shortcut.** Six hand-built probes against the served model with the original template wording and thresholds 0.70/0.35:

| Input (goal name / note) | `p(template)` |
|---|---|
| Training-style name, English merge description | 0.987 |
| Same name, same description in another language | 0.971 |
| A different, non-template name, the same English merge description | 0.019 |
| Training-style name, an **unrelated** task (change shared production routing) | 0.993 |
| A non-template name, the same unrelated task | 0.004 |
| Training-style name, a task that asks for redesign and conflict decisions | 0.954 |

## Table 4.1 - Original and retrained models

Conditions: 2026-09-28; test and rename n=48, 7 positives; probe n=16 with 14 template labels scored.

**Table 4.1 — template question, three evaluation sets** (threshold 0.70).

| Model | Set | TP | FP | FN | TN | Precision | Recall | Judgment accuracy |
|---|---|---|---|---|---|---|---|---|
| Original | test | 6 | 0 | 1 | 41 | 1.00 | 0.86 | 0.750 |
| Original | rename | 0 | 10 | 7 | 31 | 0.00 | 0.00 | 0.750 |
| Original | probe (14 scored) | 2 | 2 | 4 | 6 | 0.50 | 0.33 | 0.750 |
| Retrained | test | 4 | 0 | 3 | 41 | 1.00 | 0.57 | 0.792 |
| Retrained | rename | 4 | 0 | 3 | 41 | 1.00 | 0.57 | 0.792 |
| Retrained | probe | 4 | 0 | 2 | 8 | 1.00 | 0.67 | 0.750 |

## Table 4.2 - Paired rename decisions

Conditions: 2026-09-28; 48 paired rows; template threshold 0.70; same transform family as augmentation.

**Table 4.2 — paired rows, original test versus rename copy (48 rows each).**

| Model | Template decisions that flipped | Largest `\|delta p\|` |
|---|---|---|
| Original | 16 (6 positives lost + 10 negatives gained) | 0.983 |
| Retrained | 0 | 0.115 |

## Table 5.1 - Small-sample arithmetic

Conditions: Recomputed in 2026-10; exact two-sided 95 % intervals unless a bound or approximation is specified.

**Table 5.1 — intervals and bounds for numbers quoted in this book** (recomputed; the first seven match the independent review calculation).

| Quantity | Observed | 95 % interval or bound |
|---|---|---|
| Retrained model test recall | 4/7 | 0.184 to 0.901 |
| Original model test recall | 6/7 | 0.421 to 0.996 |
| Retrained probe recall | 4/6 | 0.223 to 0.957 |
| Original probe recall | 2/6 | 0.043 to 0.777 |
| Retrained test false positives | 0/41 | upper bound 7.05 % |
| Retrained probe false positives | 0/8 | upper bound 31.2 % |
| Sample size for a 1 % bound with zero errors | 299 independent negatives | for 0.1 %: 2,995 |
| McNemar, test positives, 3 vs 1 discordant | | p = 0.625 |
| Retrained rename flips | 0/48 | upper bound 6.05 % (one-sided; the 48 rows are one set, not 48 independent tasks) |
| Original rename flips | 16/48 | 0.204 to 0.484 |
| Readout, Set A, correct when answer in candidates | 19/20 | 0.751 to 0.999 |
| Readout, Set A, accuracy | 29/37 | 0.618 to 0.902 |
| Fast-path precision of the Laya model | 6/6 | one-sided lower bound 0.607 |
| Fast-path recall of the Laya model | 6/9 | 0.299 to 0.925 |
| Fast-path recall of the Kev model | 5/9 | 0.212 to 0.863 |
| Sample size for a recall interval about +/- 10 points near 0.67 | about 85 positives (normal approximation) | |

## Claim ledger

Conditions: Historical evidence across the dated studies; no public synthetic output establishes these claims.

**Claim ledger.**

| Claim | Supported? | Evidence | What is missing |
|---|---|---|---|
| A first-token readout from a production model picks the right option when it is among the candidates | Yes, on a small set | 19/20 in all 7 runs; incumbent 9/20 | A larger independent set; a different model |
| The readout's confident answers are too often wrong to act on | Yes | HCER 0.100 (0.069-0.107) vs 3 % gate, 3 runs | A calibration-layer ablation on the same base model with 300-500 labels |
| The same request produces different probabilities | Yes (raw logprobs differed in every item of every comparison) | 0/37, 0/108 identical; single-worker control | Request hashes in the run files; a deterministic serving configuration to test the cause |
| The cause is the serving stack (reduction order, speculative decoding, batching) | No (suspects only) | none isolated | Isolation runs |
| A 24-way label is worse than yes/no questions for a small decision model | Not established | 0.33 vs 1.000 are different tasks, base rates, inputs | Same-label-space, same-card, same-epoch comparison; stripped-card 24-way control |
| Zero-shot 24-way arms fail | Yes, on this label set | 0.16-0.21 on 321 rows | A different 24-way set |
| The fine-tuned model's template accuracy of 1.000 reflects content understanding | No | rename flips 16/48 | Independent content-only test set |
| The original model used the name as a shortcut | Yes, strongly | 16/48 flips; precision 0.00 on the rename copy; six probes | None needed for the existence claim |
| The retrained model is invariant to the name | Only for one transform | 0/48 flips on a same-type transform | Several name families (unseen prefixes, other languages, meaningless ids, missing name) crossed with the same tasks |
| The retrained model's false-positive rate is low | Weakly | 0/41, 0/8 | 299 independent negatives for a 1 % bound |
| Retraining did not cost recall | Not established, not refuted | 6/7 to 4/7, McNemar p = 0.625 | More positives; blind re-adjudication of the lost ones |
| Rename augmentation is the cause of the improvement | Not established | augmentation also changed class weight and updates | Equal-weight, equal-update control |
| The hard-block questions are a usable safety signal | No | near majority accuracy; one probe at 0.02 | n/a (design conclusion: use deterministic checks) |
| The judgment question works | No | recall 4/13 | A better-defined question and labels |
| Fine-tuning beats the other arms in general | Not established | one family, one dataset, one seed, 48 test rows | Seeds; other datasets |
| Domain pre-training hurts | Not established | one run, one seed | Replicates; task-shaped corpus |
| Threshold fitting on small selection sets is harmful | Partly | the review band vanishes in 4 of 5 fitted arms; the precision drop depended on one label | A selection set with 50 or more positives |

## Supplemental readout conditions and counts

Per-run detail for Set A: accuracy 0.784 / 0.784 / 0.811; HCER 0.069 (29 items) / 0.107 (28) / 0.100 (30); coverage 0.541 / 0.432 / 0.541.

Threshold sweep: `top_p >= 0.9` covers 28/37 items at 92.9 % accuracy; `top_p >= 0.99` covers 12/37 at 91.7 %. The 5 %-budget coverage of 19 % in that run was caused by two errors with `top_p` above 0.98 truncating the prefix.

**Set B breakdown (first run):** "reject" was answered No in 44/54 (81 %); "apply" was answered Yes in only 24/54 (44 %). Quartiles of `top_p` were 0.62 / 0.72 / 0.82. The study interpretation is that the one-sentence finding lacks the draft text, the rule number and the already-handled flag that the original decision-maker had; this was not tested.

**Order ablation, Set A:** the same chosen option under reversed candidate order in 33 of 37 items (0.892).

**Non-reproducibility.** Raw top-20 logprobs were identical in 0 of 37 and 0 of 108 items between two full runs. With one worker and strictly sequential requests, two Set A runs still had 0 of 37 identical raw logprobs, 36 of 37 identical top options, a maximum `top_p` difference of 0.151 and a mean of 0.029. Recomputing the distributions from the raw logprobs with an independent implementation gave 799 of 799 identical distributions, so the difference is not in the conversion layer.

## Revisions and supplemental battery counts

Before the 11-row re-adjudication, the original shape benchmark read 0.215 / 0.174 / 0.162 for the reported zero-shot arms; the fine-tuned test score was 0.354, and sel was 0.283 in both versions. The recomputed table is authoritative.

Threshold fitting originally reported test precision 0.714 and 0.75 for two arms. Recomputing after a disputed test label was corrected leaves a precision drop for the domain-pre-trained arm only. The review band still vanishes in 4 of 5 fitted arms.

Test positives by battery question: writes 28, deterministic 9, egress 9, prod_shared 5, judgment 13, gpu_long 10, template 7. Composed Q1 has 9 positives; these are different targets.

Across all 321 rows, zero-shot Kev-0.8B placed 296 in review. Its default-threshold test arm made no fast decisions. Numeric risk accuracy was 0.19 to 0.40 and lane accuracy 0.17 to 0.60 on test; neither remains in the composition.

Large local readout only: the initial battery had nine serial requests per card, median about 9.8 s per card, occupying the production engine. This is not a cross-arm latency comparison.

## Supplemental rename observations

**Contested positives.** Three original hits became misses and one original miss became a hit. The study interpretation attributed two lost hits to insufficient task content and one to a real miss. The independent review required blind re-adjudication; that interpretation does not invalidate the original labels. Individual row probabilities are not published.

**Live re-check of the retrained model (three single probes, not a sample):** training-style name plus English merge description 0.995; a design task 0.010; training-style name plus the unrelated production-routing task 0.053 (the original gave 0.993).

## Interpretation boundary

The complete [claim ledger and missing experiments](../README.md#limits-and-open-questions)
apply to every table. Full procedures and negative results are in the chapter pages.
