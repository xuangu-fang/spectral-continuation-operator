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
  frozen checkpoints.

`query_summary.json` stores aggregate metrics, per-seed improvements, paired
bootstrap results, source checkpoint SHA256 values and data lineage. CSV files
are flattened views for plotting or paper tables. PDFs are vector originals;
PNGs are web previews.
