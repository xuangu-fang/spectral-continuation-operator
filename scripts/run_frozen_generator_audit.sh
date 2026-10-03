#!/usr/bin/env bash
# Continue fixed cohorts in small resumable chunks. Never changes model settings.
set -euo pipefail
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=src
for audit_seed in 43 44; do
  audit_output="artifacts/coherent_waveguide_frozen_system_seed${audit_seed}_20261003"
  for audit_chunk in 1 2 3 4; do
    if test -f "${audit_output}/summary.json"; then
      break
    fi
    audit_resume=()
    if test -f "${audit_output}/config.json"; then
      audit_resume=(--resume)
    fi
    .venv/bin/python -m spectral_continuation.benchmark_waveguide_generator \
      --config configs/coherent_waveguide_frozen_system_evaluation_20261003.json \
      --field-run "artifacts/coherent_waveguide_generator_field_seed${audit_seed}_20261003" \
      --output "${audit_output}" --max-new-systems 16 "${audit_resume[@]}"
  done
  test -f "${audit_output}/summary.json"
done
.venv/bin/python -m spectral_continuation.summarize_frozen_generator_audit \
  --config configs/coherent_waveguide_frozen_summary_20261003.json \
  --output artifacts/coherent_waveguide_frozen_system_summary_20261003
for audit_chunk in 1 2 3 4; do
  audit_output=artifacts/coherent_waveguide_frozen_new_queries_20261003
  if test -f "${audit_output}/summary.json"; then
    break
  fi
  audit_resume=()
  if test -f "${audit_output}/config.json"; then
    audit_resume=(--resume)
  fi
  .venv/bin/python -m spectral_continuation.evaluate_frozen_generator_queries \
    --config configs/coherent_waveguide_frozen_new_queries_20261003.json \
    --output "${audit_output}" --max-new-systems 16 "${audit_resume[@]}"
done
test -f artifacts/coherent_waveguide_frozen_new_queries_20261003/summary.json
