# Pitfalls: symptom, root cause, fix

Historical faults and limitations are kept distinct from changes made for this public kit.

## Chapter 1

| Symptom | Root cause | Fix |
|---|---|---|
| Set B Brier of 0.632 that looks worse than a coin flip | Labels `APPLY/REJECT` were compared with options `Yes/No`, so the one-hot target was always zero and Brier collapsed to the sum of squared probabilities | Map labels into the option space before every metric; the corrected Set B Brier is 0.448 (first run) |
| Coverage at 5 % reported as 0 | The prefix scan stopped at the first prefix over 5 %; the cumulative error rate is not monotone | Scan all prefixes, keep the largest passing one |
| Order-ablation agreement inflated | A failed request mapped to `None` and an "escalate" choice also mapped to `None`, so they "agreed" | Compare only when both requests gave valid distributions; report compared and failed counts |
| A response with valid JSON but empty `top_logprobs` was not retried | Retry only wrapped transport errors | Put extraction and distribution validation inside the retry boundary |
| A table row of zeros when every request failed | Empty-set return dropped the failure counts | Keep `n_total` and `n_failed`; print dashes, not 0.0000 |
| Whole batch dies with `IndexError` | 26 candidates made the escalate letter `LETTERS[26]` | Cap at 25 candidates, build questions inside the per-item try block |
| `--selftest` exits 0 on an empty dataset | `all_ok` was only cleared on build failure | Fail when a set is empty or has no valid distribution |
| `--limit 0` ignored, negative limit slices from the tail | Truthiness test and Python slicing | Test `is not None` and reject negatives in argument parsing |
| Historical `--metrics-only` overwrote the input summary | It ignored `--out` | Fixed in this public adaptation: require `--out` and write the new summary there |
| First token is an end-of-message marker | Bare completions prompt skips the chat template | Use chat completions with thinking disabled |
| Option probabilities missing for a letter | The option fell outside the top-20 | Impute `min(observed) - 3.0`, softmax over allowed options only; record the raw probabilities so omissions can be inspected |
| No readout at all | Some server builds return `logprobs: null` on chat completions | Check one response before building a harness around readouts |

A two-item hand-computed regression case from the review rounds is worth shipping as a unit test: item 1 labelled "apply" with distribution (Yes 0.9, No 0.1), item 2 labelled "reject" with (Yes 0.6, No 0.4) gives Brier 0.37, ECE 0.35, coverage 0.5 at threshold 0.9.

For the procedure and evidence, see [first-token-readouts](first-token-readouts.md).

## Chapter 2

| Symptom | Root cause | Fix |
|---|---|---|
| All first end-to-end probes (3 of 3) judged "no template" after a good offline score | The serving client built the state card without the `goal:/task:/note:` prefixes the training cards had, and it phrased the template question differently from the training question | Build serving input with the same function as training; keep question wording in one registry file read by both; after the fix the positive probes scored 0.97 and went to "fast" |
| A confident false positive on test | The gold labels were inconsistent: five rows of one volume were labelled "verifiable transformation, eligible" and a sixth, identical in kind, "shared production file, not eligible" | Treat a model's confident disagreement as a label audit trigger; re-adjudicate under a written rule; recompute everything after the fix, and keep both tables |
| A negative result that disappears on recompute | The threshold-fit "precision collapse" depended on one disputed test label | Report which negative results depend on a single row, and prefer the structural finding (vanished review band) |
| Impressive per-question accuracy | Majority-class accuracy was already 0.73 to 0.90 for five of seven questions | Print the majority baseline next to every accuracy |
| "Better calibrated" fine-tuned arm | The Kev arm made far fewer confident answers (denominators 4, 1, 4, 11 on four questions), so HCER 0/4 is weak evidence | Always print the denominator of HCER |
| Over-reading the zero-shot template score of Kev (0.979) | The question text names the template and the goal line contains the template's name; lexical overlap alone could produce this | Run the rename test on the zero-shot arms too (not done) |

For the procedure and evidence, see [literal-questions](literal-questions.md).

## Chapter 3

| Symptom | Root cause | Fix |
|---|---|---|
| Fine-tune script cannot run on a fresh machine | Notebook code assumed a multi-device setup and device-specific optimisations | Port to single-process fp32; fall back to CPU for an unsupported reward primitive; keep an explicit smoke test that reloads the exported directory and calls `decide` |
| Fitted temperature silently ignored after fine-tune | The checkpoint carries a per-option-count temperature table that overrides a new global temperature | Remove the table from the exported config when you write your own fit |
| Hold-out used for temperature is also what you "selected on" | Only 46 rows exist | Say so; do not call it a validation score |
| Serving different from training | See Chapter 2 pitfalls (prefix and wording) | Share one builder |

For the procedure and evidence, see [fine-tuning](fine-tuning.md).

## Chapter 4

| Symptom | Root cause | Fix |
|---|---|---|
| Held-out precision 1.00, live recommendation useless | A shared name prefix across the training positives; group-wise split by volume leaves a shared prefix in test | Check what feature in the input is constant within a class; evaluate with a paired perturbation of that feature |
| "No difference" between original and perturbed sets | Aggregates compared, not rows | Save per-row probabilities and pair rows by id |
| Label-propagating augmentation may be wrong | It assumes the template's identity is content-only; if "membership in a workflow" is the real target, the decoy label is wrong | Define the prediction target in writing first ("is this task content the template's job", versus "does this task live in the template's workflow") |
| Evaluation parser counts a missing probability as a negative | Default 0.0 | Raise on missing, non-finite or out-of-range values |
| Label cast hides nulls | `bool(label)` turns null into false and the string "false" into true | Validate label types before casting |
| New shortcuts introduced by the fix | Decoy names used one fixed prefix, one year, a narrow date range | Draw both classes' names from one distribution; vary prefix, language, and meaningless ids |

For the procedure and evidence, see [rename-test](rename-test.md).

## Chapter 5

| Symptom | Root cause | Fix |
|---|---|---|
| "0 false positives" quoted as a rate | It is an observation over 41 rows | Quote it with its bound |
| 48 swapped rows counted as 48 tasks | They are variants of the same 7 positives and 41 negatives | Report original tasks, effective labels, source groups and variants separately |
| An accuracy gain that is a prior shift | Hold-out and class weights changed together | Keep an equal-weight control |

For the procedure and evidence, see [small-samples](small-samples.md).

## Public-kit checks

- The two-item binary regression maps labels before scoring and checks Brier 0.37, ECE 0.35 and coverage 0.5 at threshold 0.9.
- Transport success is insufficient: empty or null logprobs retry, then count as failures. Tests patch the back-off only to keep the offline check short.
- The mock jitter is deliberately programmed and does not establish any cause for the production jitter.
- Synthetic stubs do not stand in for a learned encoder. The real train/export/reload smoke requires local dependencies and a checkpoint.
- Battery Brier is the scalar boolean squared error; the readout uses the sum over all allowed options. The shape scorer records squared top-confidence error, not a full multiclass Brier score.
- Multi-run shape scoring preserves the original majority vote plus median confidence across all runs. That confidence may come from runs that chose a different class.
- Failed rows are explicit in scoring output. A run with no valid answers has undefined metrics, never a perfect score.
- The training sampler now uses a common name pool. The historical augmentation results belong to the original sampler and retain all historical confounds.
- `--metrics-only` now requires a separate output directory. Historical request identity remains unproved; adding hashes to new requests cannot reconstruct old ones.
- Exact upstream commits, the notebook URL and some dependency versions are unknown. The supplied pins reproduce only the versions that were recorded.
