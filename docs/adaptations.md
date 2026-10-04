# Public adaptation notes

The historical aggregate results and the public reproduction kit have different roles.
The results describe private decision records. The kit demonstrates the measurement
and evaluation mechanics with newly invented English fixtures and explicit stubs.
No historical row, label file, private corpus, checkpoint or raw output is shipped.

## Changes to the supplied files

| File | Public adaptation |
|---|---|
| `decide.py` | Generic binary rubric; loopback and model placeholders; SHA-256 of the exact JSON request bytes; separate ablation hash; null hash if no request could be built; metrics-only requires an output directory; file handles close explicitly |
| `run3.sh` | Plain sequential loop, one worker; no service manager, process queue or private path; optional socket-free transport for restricted environments |
| `battery.py` | English question registry and synthetic shape vocabulary; original threshold order and values retained; strict probabilities; template labels derive from the synthetic content rubric rather than a name prefix |
| `score_battery.py` | Explicit labels/output arguments, language-field filtering, medians, failure counts and majority baselines; boolean Brier remains scalar squared error |
| `score_arms.py` | Generic shape labels, explicit paths and language filtering; historical vote plus median-confidence procedure retained; squared top-confidence error is named distinctly from multiclass Brier |
| `fit_thresholds.py` | Selection-only fit, explicit inputs, unchanged search grid and precision target; no global threshold mutation; test-row and relabelling caveat in output |
| `calibrate.py` | Explicit paths, selection-only per-question Platt fits, stable sigmoid and an error for empty fit data |
| `arm_battery.py` | Local in-process, typed HTTP and first-token paths; explicit synthetic stub; offline checkpoint loading; no private endpoint, key, scrub table or external service configuration |
| `merge_gold.py` | Synthetic gold JSONL keyed by id replaces private positional files; exact id-set validation; original time-ordered group split logic retained |
| `laya_ft.py` | Public notebook/model credit with missing provenance stated; CPU default, configurable device, offline loading, generic seed default; objective and export format retained |
| `smoke_test.py` | Three invented English rows; CPU epoch, export, reload and calls across the typed outputs; temporary work stays under the chosen output directory |
| `dapt_laya.py` | Offline flags retained and enforced; current-directory path guard; CPU default and generic seed; no timing report; synthetic corpus input |
| `build_v2.py` | Template question, header, family prefix, name pool and seed are options; both training classes draw names from one pool; evaluation ids retained |
| `eval_v2.py` | Shared public registry, local checkpoint or explicit stub, strict boolean validation, nullable probe labels excluded, per-row output and exact id pairing |

The queued larger-model shell wrapper is deliberately omitted. Its reference
argument list appears in the training chapter, with unresolved prerequisites stated.
The training seed defaults in the public scripts are reproducibility controls for
new runs; they are not the recorded study seed or a replication claim.

## Synthetic fixtures and added helpers

`make_fixtures.py` writes invented routing and verdict examples, grouped task cards,
synthetic adjudication inputs, battery JSONL, shape outputs, names and probes.
The synthetic training split plants a prefix regularity; the later splits deliberately
mix name families. Negative cards include template vocabulary and hard-block cases.
The shape outputs are deliberately constant predictions, not benchmark observations.
Shapes without a dedicated note use the generic sentence "Read an invented local
document and report its contents." These fixtures exercise the pipeline, not
semantic coverage of every shape. The `shape:` line in `candidates.jsonl` and
`labels.jsonl` deliberately represents the historical label leak;
`battery.state_of` strips it before battery training or scoring.

The fixture dimensions reuse the recorded dev/sel/test sizes; template-positive
counts are synthetic construction choices. Their grouping can yield different split counts
when `merge_gold.py` applies its group-boundary algorithm. The integrated demo tests
that merge separately and scores the fixture splits selected by `--data` (default
`fixtures`). The end-to-end test regenerates fixtures in a temporary directory
inside the repository, checks them against the shipped files, and uses that copy.

`synthetic_model.py` contains programmed name-reading and content-reading functions.
Neither is learned from augmented data. `stats.py` implements the exact interval,
zero-event and paired-count arithmetic. `common.py` centralises strict row readers.

The mock serves chat first-token logprobs and a typed-decision response. Its
`clean`, `jitter` and `none-logprobs` modes test extraction, confidence sensitivity
and failure accounting. `mock_transport.py` serializes HTTP into the same request
handler in memory; it does not bind a socket. The normal server remains available
for a loopback test in an environment that permits it.
Routing mocks inspect the task text separately from candidate descriptions.
Successful mock requests return without back-off; only failed attempts incur it.

## What was and was not exercised

The standard-library tests exercise metric regressions, every composition branch,
selection fitting on synthetic sets with 7 and 50 positives, Platt fitting,
strict parsing, label-preserving augmentation, paired identity checks, exact
statistics, training-data helpers, synthetic in-process inference and the complete
script pipeline through the socket-free HTTP handler.

The in-memory handler checks are not proof of a real socket round trip.
Real fine-tuning, DAPT, checkpoint reload and learned-model synthetic replication
were not run. They require local training dependencies and a base checkpoint.
No packages or models were downloaded.
The historical hardware, engine build and raw data were not independently inspected.

## Templates and provenance

`LICENSE`, `.gitignore`, the banner image and its drawing script are preserved.
The supplied image is present at `docs/assets/banner.png`.

The Laya port credits the public Kaggle notebook and public model project; the
exact notebook URL, upstream licence texts and commits are unknown. The dependency
file pins only the recorded Laya and torch versions. This repository does not imply
that the unspecified versions form a verified installation environment.
