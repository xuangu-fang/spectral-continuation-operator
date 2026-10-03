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
Field-loss finetune completes at .006954, direct classical-init3000steps .012349 (unequal total optimization;
do not infer medium-label necessity). Strong target-free nonlinear context fitting now gives .007527/.006668/
.005983/.004680/.003503 at5/10/20/50/100 L-BFGS steps. Neural is NOT accuracy-best. New restart hypothesis is
amortized error/cost/storage; full encoding+query timing is required, not neural-forward-only comparisons.
Frozen new-frequency/source/coordinate teacher is being generated for8 already-used dev systems, not formaltest.
That16-file teacher is now COMPLETE; max convergence difference7.254e-6. Frozen learned field model yields
.007226/.007750 for new in-band frequencies at original/shifted sources, and .01361/.01335 at150–200Hz.
Classical100steps remains better. Warm cost with uniformly cached quadrature: neural .006954 at .1710s total
encoding+6queries; kernel .008874/.2008s; independent5/10/20step context fits .007346/.006230/.005130 at
.8266/1.5311/3.0294s. Shared GPU with unrelated VLLM, no exclusive-speed certification. Current follow-ups are
3seed robustness, equal-update-count no-medium-label control, two-system adaptive32→64mode readout at400Hz,
and a classical2D Helmholtz inverse audit. Its first cases show sine spectral differentiation ~1e-4, much stronger
than4th-order ~.02: do not train a neural correction against a deliberately weak derivative baseline.

Superseding robustness: field-finetune seeds42/43/44 .006954/.007153/.007894, mean .007333 (~17.4% over
kernel), so seed42's21.6% is NOT a robust20% gate. No-medium-label classical-init11k matchedupdates .010589,
still worse but do not infer label necessity. Gaussian physical-generator prior conditioned on PDE statistics,
20 predeclared rank/precision options, best .009872; no more PCA capacity scans. 2D Helmholtz full-field inverse
audit now complete:65/129-grid sine spectral .00009886/.00007696 against same257 discrete teacher. No neural
necessity in that noiseless dense protocol. This does not certify continuous PDE error below teacher convergence.
Higher-frequency broad-source32→64mode decoding is a NULL effect (400Hz neural .044). Narrow2m query source
on two same systems changes it:400Hz .132→.0456; oracle .122→.00675. Fixed1024 real generator coefficients,
no encoder retraining, known query source, but classic decoders share the capability. Expanding to original8 dev
systems and3 seeds; do not cherry-pick the first2 or call it formal/general-purpose high-frequency proof.
Full8-system narrow400Hz now complete: neural3seeds64mode .03616/.03754/.03695 vs32mode~.123; kernel .04296,
classic100 .04058. CRITICAL superseding optimization: classic500/2000 oldquery .001669/.001634, narrow400Hz
.02180/.02114, now beats neural. Do not claim old-source unobservability from earlier100-step failures; optimization
was a substantial confound. Coefficient-only neural narrow400Hz .03831/.03620/.03791, close to field-finetuned.
Current bounded restart tests one loss mechanism: same projected64-cell teacher with fixed vs randomGaussian
training source requests, lowband only, single complex field loss, inference context unchanged. Extra training
medium supervision explicit. Original8dev old-source queries select checkpoint, never400Hz/narrow truth. See
`docs/GENERATOR_OPERATOR_ACTION_POC_ZH.md`; no width/loss stacking or automatic3seed expansion on failure.
That mechanism test is now COMPLETE and failed: sameprojectedteacher fixedsource oldquery .006926, randomsource
.007660; narrow400Hz .03585/.03791. Stop source-distribution scans/3seed expansion, do not reselect checkpoints
using newquery truth. Checkpoint selection remained original-source inband. Teacher projection is not the mainissue.
Warm-start control shows a new bounded positive: samephysical context optimizer, neural(seed42)/kernel/classic
20steps .003444/.004642/.005130, total~3.03/3.08/3.04s;5steps .005045/.006568/.007346. Full3warmrepeats8dev,
not forward-only timing. Now require3trainingseeds and Gaussian-generator-prior warmstart (not just RBF).
Exact first-order local square-root inverse (avoids omitted iK' term)4bounded derivative/rangepenalty options
best~.01069, modest vs.01136; no further localGN/derivative sweeps. Frechet derivative tests cover propagating
and evanescent modes. Do not attribute all neural correction to the omitted second-order term.

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
