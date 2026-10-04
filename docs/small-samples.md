# Chapter 5: Small-sample arithmetic and a claim ledger

### Problem

The template and composed-decision test targets have 7 to 9 positives. What can any of it support?

### What we built or changed

The supplied stdlib helper `scripts/stats.py` implements the arithmetic below; the historical figures were recomputed and agreed with the independent review calculation:

- exact two-sided 95 % Clopper-Pearson interval for `k/n`;
- one-sided 95 % upper bound for zero events in `n` trials: `1 - 0.05^(1/n)`; the sample size needed for a bound `b` is `ceil(ln 0.05 / ln(1 - b))`;
- exact two-sided McNemar test from the two off-diagonal counts;
- paired flip count and largest absolute probability change from two row-level files.

### Results

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

With about 29 items above 0.9, one error is 3.4 %, so a "3 % high-confidence error" gate means zero errors; the readout's three or four errors gave 10 %.

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

### What did not work

Treating the test set as independent: it was used to choose the Laya model before the rename test, the rename copy is the same kind of transformation as the training augmentation, and an independent blind pass agreed on 43 of 50 key labels.

### Pitfalls

| Symptom | Root cause | Fix |
|---|---|---|
| "0 false positives" quoted as a rate | It is an observation over 41 rows | Quote it with its bound |
| 48 swapped rows counted as 48 tasks | They are variants of the same 7 positives and 41 negatives | Report original tasks, effective labels, source groups and variants separately |
| An accuracy gain that is a prior shift | Hold-out and class weights changed together | Keep an equal-weight control |

### The rule

Quote every small-sample number with its interval or bound, and write the claim ledger before the abstract: what the data supports, what it does not, and which experiment would change that.


## Recompute the arithmetic

```bash
python3 scripts/stats.py
python3 scripts/stats.py --paired results/offline/rename-name/test.jsonl results/offline/rename-name/swap.jsonl
```

The interval implementation inverts exact binomial tails by bisection; it does
not use a normal approximation for the listed Clopper-Pearson intervals.
Zero-event bounds are one-sided. The approximately 85-positive planning figure
is a normal approximation, not an exact guarantee. Dependence between variants
limits the interpretation of a binomial bound on rename flips.
