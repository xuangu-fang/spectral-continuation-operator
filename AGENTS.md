# Research handoff: current continuous coherent representation branch

## Read before acting

1. `README.md`;
2. `docs/CURRENT_STORY_AND_ROADMAP_ZH.md`;
3. `docs/COHERENT_CONTINUOUS_REPRESENTATION_24H_ZH.md`;
4. `reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md`;
5. `docs/CURRENT_STATUS_ZH.md`, `docs/COHERENT_DATA_PROVENANCE_ZH.md`;
6. `docs/EXPERIMENT_LEDGER_ZH.md` for prior positive/negative evidence.

Current user direction is high-precision complex wavefield/phase representation with continuous space/frequency queries. Ocean is flagship, not an exclusive domain. Known environment/source and adequate supervision are allowed to separate approximation from identifiability. No active-query/RL or unconditional blind distant-frequency pole/ODE restart.

## Evidence and gates

- R0 per-field fitting is not a shared operator.
- New frequency with same-frequency spatial observations is not unobserved-frequency generation.
- R1 has zero target-frequency samples; never merge its scores with spatial reconstruction.
- Main metric: system-macro raw complex relative L2. No global phase alignment. Amplitude and energy-weighted circular phase metrics explain, never replace it.
- Keep low-amplitude/source/boundary diagnostics separate without removing them from primary ROI.
- All current coherent-branch datasets/results are development. Historical strict Ocean systems remain sealed.
- Formal candidates require frozen configs, unseen systems, at least three training seeds, strongest matched baselines, source/phase conventions, physical solver truth at new coordinates, and fair error-cost curves.
- Gates are exploration targets, not reasons to alter a metric or hide failed systems.

## Do not lose these new negatives

- Homogeneous-water Ocean: fixed forward waves plus cylindrical spreading beat neural charts, including the newer coherent kernel. Do not claim universal learning necessity.
- Heterogeneous Helmholtz: sufficient-sample per-field representation succeeds, but 64-sample shared large-window encoder failed. Smaller windows change physical sample density.
- Dense regular sampling: cubic interpolation is stronger than current learned kernel. Moderate budgets show a learned signal; retain the entire budget curve.
- Extremely low train loss of SIREN did not establish held-coordinate accuracy. Do not use an under-tuned INR as the sole baseline.
- Accurate spatial chart interpolation across frequency did not improve reference-carrier interpolation.
- Classical modal/DMD transport has a positive Ocean spectral result, but neither modal analysis nor deep kernel learning is new.
- Shared learned geometry must be challenged by context-only per-instance optimization, not only fixed dictionaries.
- That audit now gives ~0.00725 vs direct learned ~0.01196 on heterogeneous patches. Learned initialization plus 5 steps matches cold 50–200 steps, but a fair latency frontier and a stronger shared encoder remain unverified. Do not claim direct learning beats the strongest classical method.
- Full-set cross-attention (~0.01317) and end-to-end two-step refinement (~0.00783) have now been tested and did not beat context-only optimization. Query-truth geometry fit (~0.00128) is an oracle, not proof of identifiability. Do not scan more attention/unroll depth without a new hypothesis. Prioritize spectral component error attribution next.
- That attribution now isolates Ocean modal birth/support handling, not existing modal phase, as the main R1 error. Simple cutoff/zero-anchor controls on direct fractional-frequency truth improve means only modestly and regress some cases. Do not generalize an implementation's missing-mode zeros into intrinsic unpredictability.
- New known-medium/source/PDE protocol gives classical phase-conditioned reduced-basis least-residual errors 0.03243/0.01551 with 3/5 context frequencies on four heterogeneous systems. This changes available information; never compare directly to unknown-medium sparse reconstruction. Phase-preconditioned RKS is prior art (arXiv:1711.00942), not our novelty. Audit eikonal and capacity before claiming a learning need.
- Eikonal and a shared phase MLP now both fail to improve the 8-system ~0.03335 baseline (double precision confirms). Learned envelope factors at 18 columns give 0.02308 vs fixed radial 0.02379, only 3%; two global powers do not help. A source-aware complex homogeneous reference reaches ~0.02316 with 6 columns on four systems. Do not attribute missing finite-source physics or added basis rank to neural novelty. Independent fractional/fine-coordinate validation is being prepared; no formal test opened.
- Independent fractional/fine-grid validation is now done: unsmooth phase-carrier fine PDE refit fails (~0.397), while transferring coarse coefficients remains ~0.029. A paired 3-frequency audit localizes ~99.91% of frozen-coefficient squared residual to the source; smoothing the carrier at the known finite-source scale restores ~0.029 with the SAME six columns. Source-aware smooth complex reference is ~0.020. This is a classical regularity correction, not an AI gain. Do not report the failed fine-grid readout as a value-capacity or identifiability barrier. Smooth all learned features/factors too; see `docs/OPERATOR_COMPATIBLE_WAVE_REPRESENTATION_ZH.md`.

## Current method/data modules

Latest: smoothing all neural inputs/factors restores fine-grid stability but only ~4% learning gain. Source-aware
carrier plus real neural envelopes gains ~1%; complex neural envelopes gain ~5%. Matched 2x2 source-width/contrast
controls show 6-column classical reference beats 18-column neural under stronger contrast. Stop pointwise-MLP
capacity sweeps. A classical contrast-source/background-propagation audit is now running; background inverses
are real PDE solves whose costs must be counted, not a free learned decoder. This now reaches ~0.0128 on four
systems and ~0.01179 on independent fine coordinates. Shared source-space encoder only improves 0.01455 to
0.01446 on eight systems. Same-family per-instance target oracle reaches ~0.00682; PDE-only optimization ~0.00924
but requires the full 60-column cache. Thus there is headroom, not yet a successful shared encoder. New one-way
range-dependent waveguide data are complete (40 development systems); they omit backscatter/penetrable seabed.
Modal eigenvector sign gauge was a real spectral-query pitfall; use the boundary-derivative convention, not target
phase alignment. Read source-space POC, waveguide data docs and report section 9 before more architecture work.

CRITICAL superseding control: PDE-gradient input only improves shared source-space to 0.014157 (~2.7%). On
4 matched systems x frequencies 7/11/13, classical right-preconditioned GMRES from the same field-ROM initial
guess reaches 0.001987 with 4 background inverse applications and 0.0003467 with 6, versus source-ROM 0.013523
(6 candidate inverse applications plus background source). Stop source-space selector capacity scans. No fair
end-to-end latency claim yet (batch vs sequential costs, setup/context and diagnostic overhead matter). Known-PDE
accuracy is largely a classical solve here, not AI novelty. Reopening needs a new amortization/compression/query
capability hypothesis with this strong control retained. Current next diagnostic is waveguide amplitude/phase
component truth replacement, explicitly oracle-only, plus frequency-context density; no blind extrapolation.

Waveguide follow-up completed: 8 dev systems, 3-context cubic spectral error .21258 (background frame), .14009
(full-medium diagonal phase), .09241 (range-local modal frame), .08327 (source-anchored residual). Both modal
amplitude and phase contribute; neither oracle alone fixes the query. Complete known-medium classical coarse
matrix transport is far stronger: 32/64 cells -> .002821/.000649, no context fields required. This is a reduced
solver, NOT learned representation novelty. Source-linear/passive behavior is tested within the one-way model.
R0 SAME-FREQUENCY profile codec with cheap background frame is .00156 at 65 profiles and .000372 at 129;
raw curves alias badly at that spacing. Do not use this aliasing control as the sole neural baseline. Main open
question remains prediction of compact envelopes / coupling, not their value capacity. Keep known-full-medium
and observation-only spectral-codec protocols separate; never remove physics inputs silently to hide strong solvers.
Bounded multi-phase-kernel audit also failed: 1/2/4 context-optimized components .007247/.008415/.010201.
No shared mixture training or more branch-count sweeps under unchanged information. See report section 11.

Observation-only spectral codec FNO pilot now failed: only stationary background + 3 context fields, no full SSP.
Cubic .212574, tuned same-input RBF residual regressor .211940; final FNO dev .310725, best remains step 0.
Exact same-seed rerun with last-state diagnostics reproduces the trajectory; last train .105016 (> .05 gate).
Data expansion was initially withheld until training capacity passed. That prerequisite has now been satisfied;
see the completed controlled scaling below. Do not interpret the first FNO failure as impossibility.

Superseding update at 20:22 UTC: query-conditioned FiLM (same base width/decode/loss) lowers train to .05229
at 4k steps; extending ONLY its optimization to 12k gives last train .026657, so the .05 capacity gate is now
passed. Dev still .302861, best step 0 .212574. Only then was the 512-system modal training cache generated.
All 512 files are now complete; maximum 256/512-cell numerical difference .00004214. Matched cache32,
cache128, cache512 FiLM runs (12k steps) give best dev .212574/.201853/.175268; strongest same-input
RBF residual regression gives .211940/.200983/.161893. Final train .02670/.05013/.09123. Data diversity helps,
but neural still fails .10 and 20%-over-best-baseline gates. No further width/step/data-only sweep is justified.
Global coupled-carrier coefficients were unstable; local unitary stabilization fixes blow-up but matched-budget
4/8 branches .230/.240 are worse than single .213. Stop this basis family. Unwrapped amplitude/phase best
.2062 is only a small gain; no more unwrap heuristics under unchanged protocol.
Dense full-SSP solver and this background+context
protocol must remain separate. Native parameter count 372032 includes complex scalars; real DOF is 634176.

NEW strongest observation-only control: classical shared generator identification from the same 3 frequencies
and 65 profiles (background only) reaches .016010; quintic derivatives plus range regularization .011362.
257/2049 profiles give .001801/.000789; true-medium projection oracle .000782. This is inverse-physics structure,
not neural novelty, and rules out attributing the .175 FNO result to intrinsic identifiability without qualification.
See report section 12 and `docs/SHARED_PROPAGATION_GENERATOR_POC_ZH.md`. A learned real-generator correction
pilot is running with TRAINING-ONLY medium coefficient labels; compare stronger classical and same-supervision
kernel controls. Its normal-equation features are phase/scale invariant; query decoder retains complex source.
Never leak true development range-dependent SSP into its input, and never hide extra training-label information.
This first generator learner completed: .008142 vs same-medium-supervised RBF .008874 (~8.2% extra, not20%).
Real DOF240304. Now testing ONLY loss alignment via the same differentiable generator and raw complex field
loss, comparing coefficient-pretrained vs classical initialization. The latter reads no training medium labels.
Numerical decoder matches independent NumPy to1e-10; gradients pass finite differences. Still no formal result.

`coherent_representation.py`, `learned_wave_chart.py`, `learned_coherent_kernel.py`,
`coherent_helmholtz_data.py`, `generate_coherent_ocean.py`, their `audit_*`/`evaluate_coherent_*` drivers,
dated `configs/*coherent*20261003.json` and `configs/learned_wave_chart*20261003.json`.
Large raw arrays/checkpoints remain local and ignored; JSON summaries/configs and scientific figures are versioned.

## Historical safeguards

Old negative results remain scoped, not universal impossibility:
- no more blind capacity sweeps of pole/ODE/hypernets under identical low-band information;
- no PCA-tail/proximal/reranking/flow restart without new information or a new causal oracle;
- no arbitrary learned active policy under the old nearly linear-Gaussian belief;
- no reporting a target-truth patch oracle as deployable performance.

Detailed older safeguards/reports are preserved in `docs/archive/AGENT_HANDOFF_BEFORE_COHERENT_20261003.md`.
If reopening an older branch, read its actual result report and ledger restart condition first; filenames/plans alone are insufficient.

## Workflow

State one hypothesis, matched control and stop rule before each experiment. Isolate one structural change at a time. Persist failures and superseding strong controls. Do not change immutable dated configs after producing results; CLI overrides must be captured in output config.

Use apply_patch for code/docs. Preserve user changes and artifact paths. Do not delete historical modules. Do not confuse timing instrumentation with model-training cost. Update the report/ledger/data inventory after material results; clearly distinguish development, oracle, formal, and no-go evidence.
