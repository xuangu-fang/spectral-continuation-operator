# Spectral Continuation Operator

Learn a frequency-stable propagation representation from sparse, inexpensive
low-frequency wave fields and query the response at unseen higher frequencies.

The first research question is deliberately narrower than a paper claim:

> Does a sparse low-frequency band contain materially more identifiable
> information about a high-frequency field than a single low-frequency field,
> and can a lightweight propagation-aware operator exploit it better than a
> matched black-box operator?

This repository is independent from `APEX_repro_fsk`. The old repository is
treated as a read-only evidence and data source; new experiments, protocols and
artifacts live here.

## Stage 1

- Freeze and attribute trustworthy APEX/APEX-H evidence.
- Build leakage-safe WaveBench and Ocean spectral-continuation protocols.
- Compare medium-only, single-frequency and sparse-band conditioning.
- Compare a matched direct query operator with a minimal propagation-aware
  spectral representation.
- Report raw complex error as primary; amplitude, phase, coherence and
  extrapolation-horizon metrics are diagnostics.

The current reading entry point is [the Chinese stage plan](docs/STAGE1_PLAN_ZH.md).

