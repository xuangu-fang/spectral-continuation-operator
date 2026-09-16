# Artifact policy and index

Large checkpoints and per-sample arrays remain on attached storage. Compact
summaries, manifests, CSV tables and paper figures are committed. Every compact
artifact binds its parent data and checkpoint hashes.

- `provenance/`: frozen APEX/APEX-H source evidence and file hashes;
- `stage1_fixed/`: WaveBench/Ocean coarse-band fixed-target controls;
- `stage1_fineband_analytic/`: training-free phase-identifiability horizon;
- `stage1_query_formal/`: three-seed 90/100-selected query runs, paired contrasts
  and representative field figure;
- `stage1_dense_horizon/`: once-only 105/110/115/120 Hz evaluation of the same
  frozen checkpoints;
- `representation_poc/`: matched image/state representation comparison on
  controlled multipath fields and Ocean Pekeris, including three-seed curves.
- `continuous_state_synthetic/`: three-seed controlled multipath comparison of
  continuous poles, lattice recurrence and image U-Net, including 19.5 Hz and
  cross-resolution evidence;
- `continuous_state_ocean_sparse/`: frozen 90/100 Hz sparse-supervision result;
- `continuous_state_ocean_dense/`: main 81–100 Hz training and 101–125 Hz
  extrapolation curves for three parameter-matched models;
- `continuous_state_ocean_fractional/`: frozen-checkpoint evaluation at six
  independently generated half-integer frequencies.
- `sparse_spectral_calibration/`: three-model-seed × three-mask-seed sparse
  broadband calibration results, raw records, error–budget/horizon figure and
  amplitude–phase mechanism figure. The preregistered method gate was not met.

`query_summary.json` stores aggregate metrics, per-seed improvements, paired
bootstrap results, source checkpoint SHA256 values and data lineage. CSV files
are flattened views for plotting or paper tables. PDFs are vector originals;
PNGs are web previews.
