# Agent handoff — three Wave as Prompt routes

Updated 2026-10-05. This file replaces the rolling experiment agenda, not the historical evidence.

## Read first

1. `README.md` and `docs/CURRENT_STORY_AND_ROADMAP_ZH.md`.
2. `docs/EVIDENCE_TO_OCEAN_WAVE_PROMPT_ZH.md` (positive AND negative findings).
3. The relevant card in `docs/wave_as_prompt/` and `data/README.md`.
4. `src/spectral_continuation/README.md` before reusing code.
5. The original evidence/config for any result you cite. Old complete handoff:
   `docs/archive/20261005_before_three_routes/AGENT_HANDOFF_PRE_REORG.md`.

## Current scope

- A: sparse complex measurements → same-source, same-frequency unmeasured spatial field.
- B: source-response demonstrations → held-out-source response, initially at the SAME frequency.
- C: approximate physical model + measurements → calibrated unmeasured response.
- A is primary; B can reuse multistatic data; C first needs credible paired model mismatch.
- Route IDs supersede the different A/B/C meanings in old technical memos.
- Representation learning is a means to these tasks, not an open-ended universal-token agenda.
- Active acquisition/RL, blind distant-frequency extrapolation, general foundation models,
  flow/ODE/pole restarts and old capacity sweeps are NOT default tasks.
- New cards contain hypotheses/plans, not completed experiments. Do not start training or
  create a long-term goal merely because you read a plan; follow the current user request.

## Scientific integrity

- Distinguish representation capacity, observation identifiability, inference, and numerical error.
- A model's failure or horizon is not an information-theoretic impossibility result.
- True-medium/target-fitting oracles are extra-information diagnostics, not deployable methods
  or strict lower bounds on all nonlinear representations.
- Strong classical PCA/MAP, posterior-memory and PDE-constrained baselines must remain.
- Preserve failed experiments. Episodic basis, neural residual, field-loss and composer variants
  have already been tried; a restart requires a different, testable mechanism.
- Speed-basis confirmation: same synthetic cosine family, known PDE/source, dense noiseless
  lowband profiles, classical inference. Gaussian family showed only small gains.
  Do not call it universal, neural, real-ocean or cross-PDE success.
- Existing Helmholtz fullfield baseline is ~1e-4 after a physical-domain bug fix;
  obsolete ~0.14 scores are INVALID. Four-line nonlinear deployment remains untested.
- No automated claim that all three routes are publishable or completed.

## Data and evaluation safeguards

- Split by physical system. Multiple sources/frequencies/masks are not independent systems.
- Old test/OOD batches have often been used; consult provenance, never relabel as untouched.
- Keep sealed truth sealed. This reorganization authorizes no reading of reserved test arrays.
- The catalog is for maintainers, NOT inference input. Test seeds/true medium may be in metadata.
- Never use `coherent_helmholtz_multiscale_sparse_dev_unsafe_seed_metadata_20261004`.
  Safe public NPZ inputs exclude seeds/true medium; manifests remain maintenance-only.
- OpenFWI local velocity does NOT match seismograms; no medium input/label from it.
- The Well staircase source/environment identity is unresolved; no multi-source/cross-system claim.
- Deepwave/OpenFWI multistatic files are receiver responses, NOT full interior fields.
- WaveBench old improvement is medium identification, NOT phase-field reconstruction.
- 512 Helmholtz training medium labels are NOT 512 paired field responses.
- Raw complex RelL2 on unobserved targets stays primary; report amplitude and energy-weighted
  circular phase error, system-level uncertainty and tail cases. No target phase alignment in
  primary scores. Lock phase masks before testing; no masking based on predicted amplitude.
- Declare source/clock phase reference, Fourier sign, units, source spectrum and numerical floor.
- C needs independently computed low/high fidelity or physical parameter mismatch, not noisy HF
  labels renamed LF. Count offline training, memory, online fitting and physical solver cost.
- See `docs/wave_as_prompt/EVALUATION_ZH.md` for detailed protocol.

## Repository maintenance

- Preserve module/config/artifact paths and frozen checkpoint interfaces.
- Reuse by reference through `docs/wave_as_prompt/research_catalog.json`; old runnable code
  remains legacy evidence unless explicitly adapted and tested for a new route.
- No overwriting old result folders. New experiments need new config/output identities.
- Large arrays/checkpoints stay local. Do not publish data without upstream license checks.
- Plain-text formulas are preferred in entry docs. Scientific memo equations are SVGs generated
  from `docs/assets/spectral_science_math/equations.json`; check using
  `.venv/bin/python scripts/render_spectral_science_math.py --check`.
- `.venv` is shared with a sibling repository. Do not mutate dependencies casually.
- Read-only checks: `python3 scripts/research_catalog.py check --local-data`;
  focused tests: `.venv/bin/python -m pytest tests/test_research_catalog.py`.
- For full pre-reorganization instructions/provenance, use the archived handoff above.
  Historical “current/next” wording does not override this file.
