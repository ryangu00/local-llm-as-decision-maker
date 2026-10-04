# Public-kit verification

This page records checks of the public scripts and fixtures, not a new model study.

Executed after the review fixes:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
bash -n scripts/run3.sh
```

All 39 unittest cases passed. The suite regenerates fixtures in a repository-local
temporary directory and checks them against the shipped files before running the
integrated demo with `--data` pointing to that copy and `--transport inprocess`.
The integration test checks the demo's success marker, saved routing results and
synthetic-only status. The demo prints:

```text
OFFLINE DEMO PASS: readout, battery, calibration, rename pairing, statistics
SYNTHETIC ONLY: no learned model was trained or evaluated
TRANSPORT: inprocess
```

The suite also checks routing with misleading candidate descriptions and reversed
candidate order, without back-off on success. The interval tests match the
displayed historical arithmetic. Shell syntax passed.

The publication audit compared every numbered result table with the fact card;
their numerical tokens match. The claim ledger and missing-experiment list remain
present. Local Markdown link targets exist, including the supplied banner image.
English-only and identifier/path scans passed. The licence, banner
image and banner script are byte-identical to their pre-edit contents.

Generated test inputs and demonstration outputs are removed when the test's
temporary directory closes. No environment-specific output paths are shipped.

Real loopback serving was not tested in this review pass. The HTTP handler was
exercised with serialized requests and responses in memory.
No learned model training, DAPT, real-model inference or export/reload check ran,
and no packages or models were downloaded.
