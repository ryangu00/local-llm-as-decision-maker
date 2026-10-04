# Chapter 4: The rename test

### Problem

The fine-tuned model scored precision 1.00 and template accuracy 1.000. Was it reading the task or the project name that heads every state card?

### What we built or changed

**Finding the shortcut.** Six hand-built probes against the served model with the original template wording and thresholds 0.70/0.35:

| Input (goal name / note) | `p(template)` |
|---|---|
| Training-style name, English merge description | 0.987 |
| Same name, same description in another language | 0.971 |
| A different, non-template name, the same English merge description | 0.019 |
| Training-style name, an **unrelated** task (change shared production routing) | 0.993 |
| A non-template name, the same unrelated task | 0.004 |
| Training-style name, a task that asks for redesign and conflict decisions | 0.954 |

Ten further probes written in the pre-task style with non-training names: merge-type tasks scored 0.024 to 0.321 (all missed), unrelated tasks at most 0.007; the judgment question gave a design task only 0.055. Cause: every positive training row's goal name began with the same template-family prefix and a date, and the note was an English after-the-fact record. The test set's positives came from the same prefix family: splitting by volume keeps a volume together but does not remove a shared prefix across volumes, so the held-out set rewarded the shortcut too.

**The paired rename test.** Build, from the same 48 test rows, a second copy whose first state line (the goal name) is replaced: every positive row gets a random non-template name from the pool of names in the data, every negative row gets a decoy template-style name with a random date. Score both copies; per original row compare the decisions (`p(template) >= 0.70`) and record the largest `\|delta p\|`. One summary number: the count of rows whose decision changed.

**The fix (`build_v2.py`).** Augment training and hold-out data by changing only the goal name while retaining the labels:

- for each positive row, 2 copies with a random non-template name, label still true (24 positives become 72);
- for 70 random negative rows (14 in the hold-out), replace the name with a template-style decoy name, label false;
- 227 rows become 345; the hold-out used for temperature fitting 46 becomes 68; the same fine-tuning script, hyperparameters and seed.

Three evaluation sets: the original test (48 rows, 7 template positives, 13 judgment positives; used earlier to select the model, so regression only); the rename copy (same 48 rows, the same kind of transformation as the training augmentation with a different seed); and 16 hand-written probes (6 template positives, 8 negatives, 2 rows with an undefined template label that are not scored).

**Invented stand-ins for the paired rows** (same note, name changed; label does not change):

```
original:  goal: changelog-merge-2026-03-01 | note: Folded 14 new entries into CHANGELOG.md and ran the structure check.   label: template = true
renamed:   goal: release-prep-2026-03-01    | note: Folded 14 new entries into CHANGELOG.md and ran the structure check.   label: template = true
decoy:     goal: changelog-merge-2026-03-14 | note: Rotate the staging database credentials and restart the proxy.        label: template = false
```

### Procedure as run

1. Probe the served model after the first end-to-end test; write down the six-row table; stop using the model's template question for recommendations.
2. Build the augmented data and retrain with the same script.
3. Score old and new models on the three sets; swap the model in the shadow service only (the shadow service records answers and changes no routing; the recommendation layer stays a deterministic registry match).
4. An independent reviewer reviewed the method and the numbers (verdict: accept with changes); the scorer was rewritten to parse strictly and to save per-row probabilities, and the paired result was recomputed from the saved rows.

### Results

**Table 4.1 — template question, three evaluation sets** (threshold 0.70).

| Model | Set | TP | FP | FN | TN | Precision | Recall | Judgment accuracy |
|---|---|---|---|---|---|---|---|---|
| Original | test | 6 | 0 | 1 | 41 | 1.00 | 0.86 | 0.750 |
| Original | rename | 0 | 10 | 7 | 31 | 0.00 | 0.00 | 0.750 |
| Original | probe (14 scored) | 2 | 2 | 4 | 6 | 0.50 | 0.33 | 0.750 |
| Retrained | test | 4 | 0 | 3 | 41 | 1.00 | 0.57 | 0.792 |
| Retrained | rename | 4 | 0 | 3 | 41 | 1.00 | 0.57 | 0.792 |
| Retrained | probe | 4 | 0 | 2 | 8 | 1.00 | 0.67 | 0.750 |

**Table 4.2 — paired rows, original test versus rename copy (48 rows each).**

| Model | Template decisions that flipped | Largest `\|delta p\|` |
|---|---|---|
| Original | 16 (6 positives lost + 10 negatives gained) | 0.983 |
| Retrained | 0 | 0.115 |

**Contested positives.** Three original hits became misses and one original miss became a hit. The study interpretation attributed two lost hits to insufficient task content and one to a real miss. The independent review required blind re-adjudication; that interpretation does not invalidate the original labels. Individual row probabilities are not published.

**Live re-check of the retrained model (three single probes, not a sample):** training-style name plus English merge description 0.995; a design task 0.010; training-style name plus the unrelated production-routing task 0.053 (the original gave 0.993).

**What the review changed in the evaluation script.** Strict probability parsing (a missing field, NaN, or a value outside [0, 1] fails the run instead of counting as 0.0); per-row probabilities saved so that pairing is possible. The original evaluation only accumulated confusion matrices, so its "test and rename results are identical" statement had no row-level evidence until the rows were saved.

### What did not work

- **Reading the confusion matrix as invariance.** Two positives swapping in and out, or probabilities changing without crossing the threshold, give identical matrices. Only the paired flip count shows it.
- **Using the same transform to train and to evaluate.** The rename copy is built with the same operation as the training augmentation (different seed), so passing it shows generalisation to that operation only, not name invariance. The name pool also came from the whole data file and can include names seen in training.
- **Recall.** Test recall fell from 6/7 to 4/7; the exact 95 % interval for 4/7 is 0.18 to 0.90. McNemar on the paired test positives (3 original-right-retrained-wrong, 1 the other way) gives p = 0.625, which neither proves nor rules out a loss.
- **The judgment question.** Retrained test judgment recall is 4/13 (probe 1/5); always answering "false" would score 0.729 on test and 0.6875 on the probe, so the reported 0.79 and 0.75 hide a non-functional question.
- **Interpreting the augmentation as a clean intervention.** It also changed class weight (template positives 24/227 = 10.6 % to 72/345 = 20.9 %, each positive now seen three times) and the number of update steps at the same epoch count, and the hold-out for temperature fitting was augmented the same way. A clean attribution needs an equal-weight, equal-update control.

### Pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| Held-out precision 1.00, live recommendation useless | A shared name prefix across the training positives; group-wise split by volume leaves a shared prefix in test | Check what feature in the input is constant within a class; evaluate with a paired perturbation of that feature |
| "No difference" between original and perturbed sets | Aggregates compared, not rows | Save per-row probabilities and pair rows by id |
| Label-propagating augmentation may be wrong | It assumes the template's identity is content-only; if "membership in a workflow" is the real target, the decoy label is wrong | Define the prediction target in writing first ("is this task content the template's job", versus "does this task live in the template's workflow") |
| Evaluation parser counts a missing probability as a negative | Default 0.0 | Raise on missing, non-finite or out-of-range values |
| Label cast hides nulls | `bool(label)` turns null into false and the string "false" into true | Validate label types before casting |
| New shortcuts introduced by the fix | Decoy names used one fixed prefix, one year, a narrow date range | Draw both classes' names from one distribution; vary prefix, language, and meaningless ids |

### The rule

For any input feature you suspect the model uses as a shortcut (an identifier, a project name, a source), build a paired copy of the evaluation set that changes only that feature and count flipped decisions per row; equal confusion matrices are not evidence of invariance.


## Public adaptation and its scope

`build_v2.py` accepts a template question, template-family prefix, header regex,
name file and seed. Dates are supplied through the name file rather than drawn
from a hard-coded year or range. Both training classes draw names from the same
pool; the adversarial evaluation still exchanges template-style and ordinary
names. This intentionally differs from the historical class-dependent sampler.
It does not retroactively remove the historical confounds.

The synthetic fixtures plant a name regularity in the training split only.
The evaluation split mixes name families independently of the template label.
`eval_v2.py --stub name` deliberately reads the name; `--stub content` deliberately
reads a task phrase. Their difference tests the pairing pipeline, not an effect
of retraining. A learned synthetic shortcut and its removal remain unverified
until local checkpoint training is run. Null probe labels are excluded from
that label's denominator; non-boolean labels fail instead of being cast.
The paired scorer rejects duplicate ids and missing partners.
