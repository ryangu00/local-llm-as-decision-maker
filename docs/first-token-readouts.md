# Chapter 1: Read the decision off the first token

### Problem

You have a production-grade local model already running. Before training anything, how far does a zero-training readout of its first-token option probabilities get you as a decision-maker, and is the probability it reports a number you can act on? The gate we set before looking: act on it only if its confident answers are almost never wrong.

### What we built or changed

A single stdlib script, `decide.py`, with these parts (all in the shipped file):

1. **Question.** A chat completion with a one-sentence system message and a user message that ends with "Answer with the single letter only" (routing) or "Answer Yes or No" (binary). Candidates are lettered `A`, `B`, ...; the last option is always a fixed "None of these fits; escalate to a human." Truncate to 25 candidates so the escalate letter always exists.
2. **Distribution.** Take the 20 returned `top_logprobs` of the first generated token. For each option keep the maximum logprob among tokens equal to the option after `strip()` and `lower()`. An option whose token is absent from the top-20 gets `min(observed logprob) - 3.0`. Softmax over the allowed options only; everything else is discarded. `top` is the argmax, `top_p` its probability.
3. **Reliability.** Up to 3 attempts, 2 s back-off, retried on transport errors, on valid JSON with a missing or empty `top_logprobs`, and on a distribution that cannot be normalised. A failed item is recorded as failed and counted, never as zero.
4. **Metrics** (per set, over non-failed items):
   - accuracy, and a majority-letter baseline;
   - Brier: mean over items of the sum over options of `(p - y)^2` with one-hot `y` built from the *mapped* label (binary labels `APPLY/REJECT` map to `Yes/No` before scoring — see Pitfalls);
   - ECE: 10 equal-width bins on `top_p`, bin accuracy against bin mean confidence, weighted by bin size;
   - HCER@0.9: the error rate among items with `top_p >= 0.90`, reported with its denominator;
   - coverage at a 5 % error budget: sort items by `top_p` descending, take the largest prefix whose cumulative error rate is at most 5 %; coverage is its size over `n`, threshold the `top_p` of its last item. Scan all prefixes: the cumulative error rate is not monotone.
5. **Order ablation (routing only).** Ask again with the candidate list reversed; agreement is judged on the chosen option's name (or "escalate"), not on the letter.
6. **Offline modes.** `--selftest` (two items per set must give a legal distribution; an empty set must fail), `--metrics-only DIR` (recompute summaries from saved per-item JSONL with no HTTP), `--workers`, `--limit`, `--ablate`.

The historical per-item JSONL stored the full distribution and the raw `top_logprobs`, so every number could be recomputed offline. It did not store the request text or a request hash. The public harness adds request hashes as described below.

**The two evaluation sets (both private; the book needs synthetic stand-ins):**

- *Set A, routing.* 37 items: 27 blind-written task sentences with a known best handler among about 400 handler profiles, plus 10 junk inputs whose correct answer is "escalate". Candidates are the top 5 handlers by embedding similarity (one batch embedding call) plus the incumbent router's pick if it is not already in the list (so 5 candidates for 28 items and 6 for 9 items) plus the fixed escalate option. The correct handler was among the candidates for 20 of the 27 real items. Baselines: the incumbent router was exactly right on 19/37 (0.514); always answering "escalate" would score 17/37 (0.459); the report's "majority letter" baseline is 0.405 and is a letter-level artefact.
- *Set B, verdicts.* 108 items, balanced 54 "apply" and 54 "reject", each a one-sentence reviewer finding about a draft, judged under a short generic rule-set system prompt, with no access to the draft. Baseline 0.5.

**The gate** (written down before the runs, in a design document): accuracy at least 0.85 and not below the incumbent; Brier at most 0.12; ECE at most 0.05; high-confidence error rate at most 3 %; coverage at the 5 % budget at least 40 % to leave shadow mode. These are design-time numbers; the 3 % was chosen to match published figures for purpose-built decision models, and none of them was calibrated against our own cost of an error.

### Procedure as run

1. 2026-09-22: first full run of both sets plus the order ablation (4 worker threads, endpoint shared with production). Setup note: a bare `completions` prompt gave `<|im_end|>` as the first token and cannot be used; chat completions with thinking off gave a clean first token (in one probe "Yes" at logprob -0.04 and "No" at -5.48).
2. The same script was run again on the same data files. The raw logprobs differed.
3. An independent reviewer read the script against its spec and found 9 problems; all were applied (see Pitfalls); the metrics-only mode was added.
4. Control for self-inflicted noise: two full Set A runs with one worker, strictly sequential.
5. The formal result: three sequential full runs (A and B, one worker, 145 requests each, about 49 s per run), reported as median with the range.
6. A second review round recomputed all 12 result groups (7 Set A runs and 5 Set B runs) with an independent implementation: every group matched to three decimals, and the three-run medians, ranges, stable-top counts and `top_p` spreads matched too.

**Public offline commands.** Run from the repository root as described in the
[reproduction procedure](../README.md#reproduce-it-without-our-infrastructure).
The first command uses the socket-free mock wrapper; no server needs to be started.
The public harness uses a loopback placeholder endpoint and a placeholder model name.
Its `--metrics-only` mode requires `--out`, and each built request now has a SHA-256
`request_hash` over the JSON bytes sent. A build failure has a null hash because
there was no request; order ablations store a separate hash. These are publication
adaptations, not evidence that historical runs saved hashes.

```bash
COOKBOOK_MOCK_MODE=clean python3 scripts/mock_transport.py scripts/decide.py --selftest --data fixtures --base http://127.0.0.1:8000 --model mock
python3 scripts/run_offline.py --out results/offline --transport inprocess
python3 scripts/decide.py --metrics-only results/offline/readout/run1 --out results/recomputed --model mock
```

The data directory holds two JSON arrays. Invented stand-ins for the book:

```
// setA_routing.json (one item)
{"id": "A000", "prompt": "Summarise last quarter's support tickets by root cause.",
 "expect": "support-analyst", "router_persona": "data-analyst",
 "candidates": [{"name": "support-analyst", "title": "Support analyst", "description": "Clusters tickets and writes root-cause summaries"},
                {"name": "data-analyst", "title": "Data analyst", "description": "Queries and charts structured data"}]}
// a junk item uses "expect": "__none__"
// setB_reconcile.json (one item)
{"id": "B000", "src": "review-01", "finding": "The draft states a speed-up figure without a source.", "label": "APPLY"}
```

### Results

**Table 1.1 — the formal three-run result** (median, range in brackets; endpoint shared with production; one worker; recomputed from the three summaries).

| Set | n | Accuracy | Brier | ECE | HCER@0.9 | Coverage @5 % | Items whose top option was identical in all 3 runs | Mean / max spread of `top_p` across runs |
|---|---|---|---|---|---|---|---|---|
| A routing | 37 | 0.784 (0.784-0.811) | 0.336 (0.331-0.356) | 0.149 (0.145-0.153) | 0.100 (0.069-0.107) | 0.541 (0.432-0.541) | 36/37 | 0.034 / 0.166 |
| B verdicts | 108 | 0.657 (0.648-0.685) | 0.449 (0.440-0.472) | 0.079 (0.075-0.119) | 0.077 (0.000-0.111) | 0.102 (0.028-0.111) | 89/108 | 0.114 / 0.286 |

Per-run detail for Set A: accuracy 0.784 / 0.784 / 0.811; HCER 0.069 (29 items) / 0.107 (28) / 0.100 (30); coverage 0.541 / 0.432 / 0.541.

**Table 1.2 — gate outcome** (formal medians).

| Gate condition | Set A | Set B |
|---|---|---|
| Accuracy >= 0.85 | fail (0.784) | fail (0.657) |
| Accuracy not below baseline | pass (router 0.514) | the report scored B as failing all six conditions; against the 0.5 majority baseline its 0.657 is above, so this row depends on which baseline is meant |
| Brier <= 0.12 | fail (0.336) | fail (0.449) |
| ECE <= 0.05 | fail (0.149) | fail (0.079) |
| High-confidence error <= 3 % | fail (0.100) | fail (0.077) |
| Coverage @5 % >= 40 % | pass (0.541; the weakest run was 0.432) | fail (0.102) |

**Table 1.3 — the same Set A, four runs of the same code on the same data** (this is what exposed the instrument problem; run v0 and v1 used 4 workers, det-1 and det-2 used 1 worker, recomputed).

| Run | Accuracy | Brier | ECE | HCER@0.9 (n at or above 0.9) | Coverage @5 % |
|---|---|---|---|---|---|
| v0 | 0.784 | 0.348 | 0.150 | 0.071 (28) | 0.189 |
| v1 | 0.784 | 0.346 | 0.158 | 0.138 (29) | 0.378 |
| det-1 | 0.784 | 0.355 | 0.161 | 0.138 (29) | 0.189 |
| det-2 | 0.784 | 0.336 | 0.145 | 0.069 (29) | 0.568 |
| median | 0.784 | 0.347 | 0.154 | 0.105 | 0.284 |

Accuracy did not move in four runs; HCER moved between 7 % and 14 % and coverage between 19 % and 57 %. The reviewer's counter-example against a simple explanation: two runs with the same four confident wrong answers had HCER 4/29 in both but coverage 14/37 and 7/37, so coverage depends on where the wrong answers sit in the whole confidence ranking, not on which side of 0.9 they fall.

**Table 1.4 — where it is right and wrong, Set A** (first run; "in candidates 19/20, junk 9/10" held in all seven runs; the missing-candidate count was 1/7 in six runs and 2/7 in the formal third run).

| Category | n | Readout right | Incumbent router right | Note |
|---|---|---|---|---|
| Correct handler among the candidates | 20 | 19 | 9 | the one error had `top_p` 0.985 |
| Correct handler not among the candidates (right answer: escalate) | 7 | 1 | 0 | 6 of 7 chose a nearby handler, `top_p` 0.44 to 0.996; 6 of the 8 errors in this run are here |
| Junk input (right answer: escalate) | 10 | 9 | 10 | the one error had `top_p` 0.87 |

Threshold sweep: `top_p >= 0.9` covers 28/37 items at 92.9 % accuracy; `top_p >= 0.99` covers 12/37 at 91.7 %. The 5 %-budget coverage of 19 % in that run was caused by two errors with `top_p` above 0.98 truncating the prefix.

**Set B breakdown (first run):** "reject" was answered No in 44/54 (81 %); "apply" was answered Yes in only 24/54 (44 %). Quartiles of `top_p` were 0.62 / 0.72 / 0.82. The study interpretation is that the one-sentence finding lacks the draft text, the rule number and the already-handled flag that the original decision-maker had; this was not tested.

**Order ablation, Set A:** the same chosen option under reversed candidate order in 33 of 37 items (0.892).

**Non-reproducibility.** Raw top-20 logprobs were identical in 0 of 37 and 0 of 108 items between two full runs. With one worker and strictly sequential requests, two Set A runs still had 0 of 37 identical raw logprobs, 36 of 37 identical top options, a maximum `top_p` difference of 0.151 and a mean of 0.029. Recomputing the distributions from the raw logprobs with an independent implementation gave 799 of 799 identical distributions, so the difference is not in the conversion layer.

### What did not work

- **Treating one run as an instrument reading.** Single-run HCER and coverage on 37 items are not citable; every single-run value in Table 1.3 would have told a different story.
- **The escalate option.** The readout rarely chose "none of these": 1 or 2 of 7 when the right handler was missing. The failure mode is retrieval recall (the right handler was in the top 5 or 6 candidates for 20 of 27 items) plus an escalate option that is not taken seriously. Both fixes (a separate "is any candidate acceptable" yes/no question before the choice; raising the candidate list to 8 with a three-field re-rank) were designed and **not run**.
- **Verdict set B with only a sentence of context.** 0.657 accuracy against a 0.5 baseline, 19 of 108 items changing their top option between three runs of the same code and data: the reader was close to flipping a coin on this state card.
- **Attributing the jitter.** The suspects (tensor-parallel reduction order, speculative decoding, request batching with other traffic, a top-k kernel issue) were listed and not isolated. The single-worker control rules out the harness's own concurrency only.
- **"Request-identical" as a proven premise.** The historical run files do not contain the prompt or a hash of it; the claim that requests were byte-identical follows from using the same data file and deterministic construction code and is not directly evidenced.

### Pitfalls

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

### The rule

Use the first-token distribution as a reader only when the correct option is guaranteed to be on the list, always include a calibrated escape option, and report calibration metrics as a median of at least three runs with their range; a single run's probabilities are not an instrument reading.
