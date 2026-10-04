#!/usr/bin/env bash
# Three sequential runs. All arguments are forwarded to the readout harness.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_root="${OUTPUT_ROOT:-results/readout}"
for run in 1 2 3; do
  if [[ -n "${COOKBOOK_MOCK_MODE:-}" ]]; then
    python3 "$script_dir/mock_transport.py" "$script_dir/decide.py" "$@" --workers 1 --out "$output_root/run$run"
  else
    python3 "$script_dir/decide.py" "$@" --workers 1 --out "$output_root/run$run"
  fi
done
