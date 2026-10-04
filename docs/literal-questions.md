# Chapter 2: Ask literal questions, not a 24-way one

### Problem

A gate must decide, for each task card, whether it fits one pre-approved template (take a fast path), needs the full process, or needs a human look. The first design asked a small model one question: which of 24 task shapes is this? The reported arms were tried before changing the question.

### What we built or changed

**The data.** 321 labelled task cards, each a short plain-text state card (goal line, task line, one-line note; at most 1,500 characters). Ordered by time and split 70/15/15 by group so that all rows of one project volume stay in one split: dev 227, sel 46 (selection and temperature fitting), test 48 (sealed until the final scoring). Language: 230 English, 91 mixed. Labels in the 24-way task: 23 named shapes plus "none". In dev, 20 of the 24 options occur and 6 of those have fewer than 5 examples; the largest class holds 14.5 % of dev and 20.8 % of test (recomputed).

**Two input defects in the first design**, found only after the reported arms failed:

1. The state card contained a hand-written "shape: <short phrase>" line. For the shape question that line is a label leak; for other questions it is text that argues for its own classification, which small models follow.
2. The "is a known shape available" question asked the model to reason over the whole shape table, a multi-hop abstraction. The risk question had five levels described by four-word labels with no boundaries.

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

Two further questions were asked in the first battery run and then removed from the decision: a 5-level risk score and a 6-way choice of execution lane. Risk is derived in code from the hard-block questions; verifiability is guaranteed by the template's own check script, not asked.

**The composition rule (v2, only yes/no questions), thresholds `review = 0.35`, `action = 0.70`:**

```
hard = [egress, prod_shared, judgment, gpu_long]
if any(p[q] >= 0.70 for q in hard):            -> "full"   (full process)
elif any(0.35 <= p[q] < 0.70 for q in hard):   -> "review"
elif any(p[t] >= 0.70 for t in template_qs):   -> "fast"
elif any(0.35 <= p[t] < 0.70 for t in template_qs): -> "review"
else:                                           -> "full"
```

The two thresholds are the defaults of a public cookbook for the same "batch of yes/no questions plus thresholds in code" pattern, not fitted values. State cards sent to the question battery keep goal, task and note lines only; the shape line is stripped. Labels for every battery question are derived mechanically from the existing gold labels (for example `judgment` is true when the delivery type is "judgment"); `q1`, the composed target, is true when the shape is known, verifiable, stays on the machine, and is inside the template's allowed file set.

**Six arms**, each scored with the same script (per-question accuracy, HCER@0.9 on `max(p, 1-p)`, coverage at 5 %, Brier, composed precision/recall):

| Arm | How it reads the answer |
|---|---|
| Large local LLM readout | one first-token request per question (`max_tokens=2`, `top_logprobs=20`, thinking off), `Yes`/`No` softmax |
| Laya, zero-shot | in-process typed question heads, `multilingual` checkpoint |
| Kev-0.8B, zero-shot | HTTP typed-decision server |
| Laya fine-tuned on the battery labels | see Chapter 3 |
| Kev-0.8B fine-tuned on the battery labels | see Chapter 3 |
| Laya with domain pre-training then fine-tuning | see Chapter 3 |

**Invented stand-ins for the book (one battery row in the typed JSONL used by the fine-tuning script).** The open-source libraries call the yes/no question type `noul`; `criteria` holds one sentence for true and one for false.

```
{"state": "goal: changelog-merge-2026-03-01\ntask: t2\nnote: Folded 14 new entries into CHANGELOG.md and ran the structure check; every reference resolved.",
 "questions": {
  "writes":  {"type": "noul", "instructions": "Does the task create or modify files, code, configuration or services?",
              "criteria": {"true": "The deliverable includes writing or modifying any file, code, configuration or service",
                           "false": "The task only reads, searches, analyses or gives an opinion"}, "label": true},
  "egress":  {"type": "noul", "instructions": "Does the task call a cloud model, reach the public internet, publish or send something out?",
              "criteria": {"true": "...", "false": "The task stays on the local machine and network"}, "label": false},
  "tpl_changelog_merge": {"type": "noul", "instructions": "Is this task the weekly changelog merge (fold new entries into CHANGELOG.md and run the structure check)?",
              "criteria": {"true": "The task is exactly that merge", "false": "The task is something else"}, "label": true}}}
```

A synthetic card set for the book needs: a few hundred cards; one template question; deliberately planted name regularity in the training split only (every template card's goal name starts with the same prefix) so that Chapter 4 can reproduce a shortcut; negative cards that mention the template's words (changelog, structure check) without being the job; and hard-block cards.

### Procedure as run

1. Pre-label all 321 cards with the local LLM readout (24-way, known, risk) as candidates for the adjudicator, adjudicate gold labels against a written rubric, run the 50-row blind second pass, tighten the rubric, re-adjudicate 11 shape rows (label-quality table above).
2. Score the reported zero-shot arms on the 24-way question with at least three runs each (median/vote).
3. Switch to the battery after reading how practitioners use this model class: one literal question per condition, thresholds in code, numeric and counting questions in code.
4. Score the battery arms (zero-shot arms three runs, median), fine-tune (Chapter 3), score on test.

### Results

**Table 2.1 — 24-way question, zero-shot arms and a fine-tuned arm** (321 rows unless stated; recomputed after the 11-row re-adjudication; chance 1/24 = 0.042).

| Arm | Shape accuracy, all 321 | Test (48) | Sel (46) | HCER@0.9 (confident errors / confident answers), all | Coverage @5 %, all |
|---|---|---|---|---|---|
| Large local LLM readout | 0.209 | 0.229 | 0.130 | 13/40 | 0.012 |
| Laya zero-shot | 0.178 | 0.229 | 0.130 | 32/51 | 0.000 |
| Kev-0.8B zero-shot | 0.162 | 0.125 | 0.087 | 0/0 | 0.006 |
| Laya fine-tuned on 24-way, 4 epochs (dev rows are in training) | 0.604 (not held out) | 0.333 | 0.283 | 0/5 | 0.312 |

Majority class share on test: 0.208. The same table in the original benchmark note, before the re-adjudication, read 0.215 / 0.174 / 0.162 for the three zero-shot arms and 0.354 test for the fine-tuned arm (sel 0.283 in both); use the recomputed values. A 12-epoch fine-tune was reported in a research note as 0.26 held-out with its fitted temperature at 8.8 of a maximum 10; the corresponding run file is not in the benchmark table and the figure is not independently checked.

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

**Table 2.3 — composed fast-path decision on test (9 positives of 48; "fast" = rule output `fast`).**

| Arm | Precision | Recall | Fast | Review | False fast |
|---|---|---|---|---|---|
| Large local LLM readout | 0.00 | 0.00 | 1 | 16 | 1 |
| Laya zero-shot | 0.50 | 0.111 | 2 | 7 | 1 |
| Kev-0.8B zero-shot | none | 0.00 | 0 | 44 | 0 |
| Laya fine-tuned | 1.00 (6/6) | 0.667 (6/9) | 6 | 8 | 0 |
| Kev-0.8B fine-tuned | 1.00 (5/5) | 0.556 (5/9) | 5 | 15 | 0 |
| Laya domain-pre-trained then fine-tuned | 0.857 (6/7) | 0.667 (6/9) | 7 | 4 | 1 |

Over all 321 rows the Kev-0.8B zero-shot arm put 296 rows in the review band (no fast, no hard block): its probabilities all sat inside 0.35 to 0.70, so with default thresholds it never acts.

**Table 2.4 — the comparison the headline is built on, and why it is not like-for-like.**

| | 24-way shape | Template yes/no question |
|---|---|---|
| Model | Laya fine-tuned (4 epochs) | Laya fine-tuned (6 epochs, battery labels) |
| Test accuracy | 0.333 (0.354 before re-adjudication) | 1.000 |
| Chance / majority | 0.042 / 0.208 | 0.854 (7 positives of 48) |
| State card | contained the hand-written shape line | shape line stripped |
| Label space | 24 classes, 6 of 20 present dev classes under 5 examples | one binary question, 24 positives in dev |
| What changed together | question, label space, base rate, state text, epochs | |

So "about 0.33 to 1.00 on the same test rows" is true as a statement of two numbers and unsupported as a statement of cause. The 24-way arm with the shape line stripped, a binary "is it this shape" question on the unstripped card, and a 24-way arm at the same epochs were never run.

**Threshold fitting (global review and action thresholds fitted on the 46 sel rows, 7 positives, target precision at least 0.98 with maximal recall).**

| Arm | Fitted (review, action) | Test precision / recall / fast / false fast |
|---|---|---|
| Laya fine-tuned | 0.35 / 0.35 | 1.00 / 0.778 / 7 / 0 |
| Kev-0.8B fine-tuned | 0.35 / 0.35 | 1.00 / 0.667 / 6 / 0 |
| Kev-0.8B zero-shot | 0.55 / 0.55 | 1.00 / 0.444 / 4 / 0 |
| Laya domain-pre-trained | 0.10 / 0.10 | 0.75 / 0.667 / 8 / 2 |
| Large local LLM readout | 0.45 / 0.90 | no fast decisions |
| Laya zero-shot | no threshold reaches precision 0.98 on sel | n/a |

In 4 of the 5 arms for which any threshold reached precision 0.98 on sel, the fit sets `review = action`, which deletes the review band entirely. That is the robust finding. The first write-up reported a precision collapse on test (0.714 and 0.75 for two arms) and decided to keep the default 0.35/0.70; recomputing the same fits against the gold labels after one disputed test row was corrected gives the table above, where the precision drop happens for one arm only (the domain pre-trained one). The collapse therefore depended on a single relabelled row; the decision to keep the defaults stands on the vanished review band and on 7 positives, not on that precision drop.

**Latency note (large local model readout only; not comparable across arms).** One request per question, nine requests serial per card, median about 9.8 s per card, and it occupied the production engine. That arm was dropped from consideration for that reason, and because its composed precision was 0.

### What did not work

- **Zero-shot decision models on a 24-way label.** The reported arms at 0.16 to 0.21; coverage at the 5 % budget at most 0.012 for three arms; one arm answered with high confidence and was wrong 32 of 51 times. The research-note explanation (24 options with short names as criteria, rare classes, the self-justifying shape line) was not isolated by ablation.
- **The English-oriented zero-shot weights on the meta-judgment questions.** Zero-shot Laya answered "true" with high confidence almost everywhere: template accuracy 0.167, confident errors 37 of 38; hard-block questions 0.23 to 0.38. The upstream README also warns that the English checkpoint collapses on non-Latin text with high confidence, which is why the `multilingual` checkpoint was used throughout.
- **The numeric and lane questions.** Risk (5 levels) accuracy 0.19 to 0.40 and the 6-way lane choice 0.17 to 0.60 on test; both were taken out of the decision and derived in code instead.
- **Hard-block questions as a safety signal.** Table 2.2 shows the fine-tuned arms beat the majority class on the four hard-block questions by at most 0.04 (Laya) and 0.06 (Kev); `egress` is below it for Laya. One end-to-end probe made the point concretely: a probe involving a shared production routing change received a `prod_shared` probability of 0.02 from the served fine-tuned model. Release safety must come from deterministic checks; the questions are advisory.

### Pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| All first end-to-end probes (3 of 3) judged "no template" after a good offline score | The serving client built the state card without the `goal:/task:/note:` prefixes the training cards had, and it phrased the template question differently from the training question | Build serving input with the same function as training; keep question wording in one registry file read by both; after the fix the positive probes scored 0.97 and went to "fast" |
| A confident false positive on test | The gold labels were inconsistent: five rows of one volume were labelled "verifiable transformation, eligible" and a sixth, identical in kind, "shared production file, not eligible" | Treat a model's confident disagreement as a label audit trigger; re-adjudicate under a written rule; recompute everything after the fix, and keep both tables |
| A negative result that disappears on recompute | The threshold-fit "precision collapse" depended on one disputed test label | Report which negative results depend on a single row, and prefer the structural finding (vanished review band) |
| Impressive per-question accuracy | Majority-class accuracy was already 0.73 to 0.90 for five of seven questions | Print the majority baseline next to every accuracy |
| "Better calibrated" fine-tuned arm | The Kev arm made far fewer confident answers (denominators 4, 1, 4, 11 on four questions), so HCER 0/4 is weak evidence | Always print the denominator of HCER |
| Over-reading the zero-shot template score of Kev (0.979) | The question text names the template and the goal line contains the template's name; lexical overlap alone could produce this | Run the rename test on the zero-shot arms too (not done) |

### The rule

Ask a small decision model one literal yes/no question per condition, keep thresholds and composition in code, print the majority baseline next to every accuracy, and treat a comparison across different questions, label spaces and base rates as two separate numbers rather than as an improvement.
