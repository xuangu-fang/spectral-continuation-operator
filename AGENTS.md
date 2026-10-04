# Research handoff: current continuous coherent representation branch

## Read before acting

1. `README.md`;
2. `docs/PHASE_REPRESENTATION_DECISION_20261004_ZH.md`, then `docs/CURRENT_STORY_AND_ROADMAP_ZH.md`;
3. `docs/COHERENT_CONTINUOUS_REPRESENTATION_24H_ZH.md`;
4. `reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md`;
5. `docs/CURRENT_STATUS_ZH.md`, `docs/COHERENT_DATA_PROVENANCE_ZH.md`;
6. `docs/EXPERIMENT_LEDGER_ZH.md` for prior positive/negative evidence.

Current user direction is high-precision complex wavefield/phase representation with continuous space/frequency queries. Ocean is flagship, not an exclusive domain. Known environment/source and adequate supervision are allowed to separate approximation from identifiability. No active-query/RL or unconditional blind distant-frequency pole/ODE restart.

## Most recent evidence (2026-10-04; read before re-running)

- **Latest POSITIVE but sharply bounded learned-variable result:** Same1024 training MEDIUM labels (512 taper5+512 taper10), same255 per-system code, same clean60/100/140Hz×65 lowband context and 5+5 classical GN: learning a SPEED-field PCA basis on64×160 nodes and applying exact nonlinear `c^-2` yields f800 broad/narrow raw .005375/.005052, phase .004197/.003915rad on NEW16 taper7.5 systems20268000–015. Matched q-PCA255 raw .080678/.081965; raw reductions93.34%/93.84%, all16 systems both-source raw≤.05, all prospective gates pass. Prereg confirmation code/config commit `5fe88a5` BEFORE truth. This is interpolation between training t5 and t10 within SAME16×16 cosine synthetic medium and SAME one-way PDE, known source/background/PDE, dense noiseless context, near-complete255 latent DOF. NOT a second physics domain, strong compression, neural operator, real ocean, or top-conference AI novelty. Old taper10 dev16 speed result .008592/.009278 is NOT another independent cohort. See docs/LEARNED_SPEED_MANIFOLD_CONFIRM_ZH.md and report §22, plus both new figures. Offline training cost excluded; online speed ~9.04s vs q ~7.83–7.93s. Frozen speed basis data-derived rank16 depth factorization is ~machine-precision lossless and stores264,704 float64 vs q524,288, but inference still expands basis and this low rank is CAUSED by true synthetic16 depth cosine modes. Read docs/LEARNED_SPEED_BASIS_STORAGE_AUDIT_ZH.md. Exact known-family inverse still has extra information and remains a crucial bound; don't rank it as matched information.
- **Immediate second-MEDIUM-family NEGATIVE, stop rank/ridge scans:** Gaussian 3-local-inclusion family, same one-way PDE, old8 repeatedly used DEV systems; same512 medium-label seeds, same128 code/context/5+5 GN. Learned speed-variable f800 broad/narrow raw .003219/.003435 vs matched q-PCA .003415/.003528: only5.74%/2.64% improvement, paired wins4/8 each and speed worst-case is worse. Predeclared ≥20% both-source gain gate at commit`092b224` FAILED; no new Gaussian confirmation systems generated. This rejects universal large speed-variable advantage, not the cosine-family prospective positive. See docs/GAUSSIAN_SPEED_VARIABLE_DEV_ZH.md. It is NOT a second PDE or independent validation of speed method.
- **Fixed-code 1000/1200Hz boundary:** On first8 ALREADY-USED new taper7.5 systems, no code refit, f1200 original256 cells×128 modes broad/narrow raw .006741/.006537, phase .005435/.005189rad. Doubling cells to512 and modes to192 improves raw only3.19%/.15%; prereg ≥20% adaptive-readout gate FAILED. Numerical teacher audit max1.935e-4 passed. This means resolution is not current bottleneck up to1200Hz in this synthetic subset; it does NOT prove arbitrary higher-frequency resolution unnecessary. No q-PCA at these new frequencies; do not claim learned-method advantage at1200. Config/code freeze`a7363b7`, see docs/SPEED_RESOLUTION_SCALING_ZH.md. The first8 are not new independent systems.
- **Representation-variable attribution via EXTRA-INFO oracle:** Prereg`f5e29b5`, old truth only: projection of TRUE test medium into frozen speed vs q PCA gives f800 raw .002391/.002404 vs .077785/.075747 on cosine new16, but ~.002–.003 for BOTH spaces on Gaussian old8. Four prereg mechanism checks pass; cosine mean q coefficient projection residual .006644, Gaussian .000469. This supports response-sensitive q rank inflation as a family-dependent mechanism, NOT a strict subspace capacity lower bound. All these numbers READ TRUE TEST MEDIUM and are nondeployable; don't merge with low-band-only table. See docs/VARIABLE_PROJECTION_ATTRIBUTION_ZH.md and reports/figures/spectral_variable_projection_attribution.png.
- **Latest capacity/identifiability boundary, POSTHOC on USED cosine new16:** q-PCA true-test-medium projection f800 raw falls from .0778/.0757 at rank255 to .00309/.00311 at rank768, so linear q has sufficient capacity if given 3x code/shared-q-basis storage. But LOWBAND-only q768 5+5 GN without ongoing prior is .123/.123. Gaussian-code MAP strength1e-6 selected from first8 lowers to .03185/.03353 pooled16, with latter8 a conditional fixed-strength check but NOT new independent test. It remains above speed255 .00538/.00505; other q priors/optimizers are not ruled out. A bounded 0/1/3/5 iteration horizon on first8 improved monotonically, so do not say the failure was late-iteration overfitting. Oracle ≠ deployable, and no cross-PDE claim. See docs/Q_CAPACITY_ORACLE_AUDIT_ZH.md and reports/figures/spectral_q_capacity_audit.png. Future second-domain baselines MUST include regularized high-rank q and strong PDE-constrained inverses, not just weak q255.
- **Decisive exact-family shifted-medium control:** On SAME already-used16 taper10 OOD systems, a classical nonlinear 255-coordinate inverse knowing TRUE 16x16 cosine SPEED family and target taper10, but NOT true test coefficients/high-freq targets/training medium labels, uses the same clean60/100/140Hz context and fixed old8 1e-7 init+5+5 GN. f800 broad/narrow raw .003880/.004129, phase .003021/.003217rad, worst .005807/.006825; all prereg posthoc gates pass. Removing true test coefficients from spec with fixed observations leaves1+1 estimated coordinates and beta EXACTLY unchanged. This proves information sufficiency ONLY in this ideal known-PDE/source/dense noiseless synthetic task, with much stronger true family knowledge than PCA; do NOT call it fair learning-vs-classical ranking or new independent systems. Read docs/SPECTRAL_TAPER_EXACT_FAMILY_ZH.md. It sharpens the open problem: learn adaptable phase-faithful generator representations without the true synthetic family.
- **New OOD NO-GO:** Training taper5 correlated16x16 spectral prior rank256 + frozen5+5 GN applied to fresh16 matched-RMS taper10 systems gives f800 broad/narrow raw .4155/.4350, phase .3743/.4041rad, 0/16 systems raw≤.10; prereg OOD gates fail, numerical audit max2.39e-4 passes. True-medium Euclidean PCA256 projection oracle is still .2925/.2789, whereas true-medium32 physical oracle is .00285/.00281. Euclidean projection is NOT a best-response lower bound. A posthoc explicit TARGET-f800 response oracle was preregistered in docs/SPECTRAL_TAPER_RESPONSE_ORACLE_ZH.md to separate subspace capacity and projection metric; it is not deployable. This is same one-way PDE, not second physics domain. See docs/SPECTRAL_TAPER_SHIFT_POC_ZH.md and its result artifact.
- **Single-target oracle now complete:** Starting from TRUE-medium Euclidean PCA256 projection and directly fitting TARGET f800 broad full-depth profiles for5 steps gives broad raw .2925→.07536, but unfit narrow only .2789→.22321 on all16 OOD systems. The projection was not a best response code, but this target-leaking fit neither reaches mean .05 nor yields a strong shared cross-source code. It is NOT deployment. To rule out lack of narrow target supervision, ONE final posthoc BOTH-target-source oracle is predeclared in docs/SPECTRAL_TAPER_JOINT_ORACLE_ZH.md; stop oracle chasing thereafter, regardless of result.
- **Final OOD attribution now complete; STOP this cohort:** BOTH-target-source leakage oracle (true medium initialization + both f800 full-depth target fields) gives rank256 broad/narrow raw .08293/.08303 after fixed5 GN, phase .06232/.06265rad; all16 still accept step5, so not a best-capacity lower bound. With 512 EXTRA taper10 medium labels and same rank256/low-frequency/5+5 method, deployable-on-target-distribution raw improves .4155/.4350→.12705/.13668 but fails prereg .05 and ≥70% gates; true-medium matched-PCA projection raw .06728/.06516 despite 99.99675% training coefficient variance retained. Neither extra-information control establishes original model OOD transfer or AI novelty. No further oracle/rank/hyperparameter sweep on these16 systems; next experiments need a NEW scientific hypothesis/data protocol. See docs/SPECTRAL_TAPER_JOINT_ORACLE_ZH.md and docs/SPECTRAL_TAPER_MATCHED_PRIOR_ZH.md.
- **Latest decision-changing control:** Exact nonlinear inversion of the KNOWN 16×16 synthetic speed family (255 physical coordinates; no test coefficients or training labels) achieves f800 broad/narrow raw .00942/.01033 and phase .00703/.00765rad on the already-used32 spectral systems, versus PCA256 .01475/.01532. It was fixed on old8 development first, and is extra-information/not cost matched; 28/32 and25/32 systems improve. This refutes “first-order analytic cosine basis failed, therefore learned PCA is necessary.” The learned PCA result is a sound capacity/identifiability fact, NOT an AI-method necessity claim. Source, config, all rows and caveats: docs/SPECTRAL_MEDIUM_STRESS_POC_ZH.md and report §21. Do not call these32 a fresh independent test for the exact-family method.
- **Noise/sparsity boundary:** On old8 spectral development, rank256 5+5 GN with17 CLEAN context profiles still gives f800 raw .0181/.0188; 0.1% complex modal noise .0243/.0249. At1% noise the original measured-source-initial approach fails .172/.178; substituting the already-known analytic source initial (nothing else changed) improves to .0845/.0889, phase .0675/.0724, but worst system raw remains .180/.168. Training-mean medium initialization fails badly (~.49). All posthoc DEV, no real hydrophone claim; keep known-source classical correction as baseline. An initial sparse script crashed before performance rows due missing observation indices; v2 fix and failure are preserved.
- **Fixed noise-condition check on already-used32:** At 1% complex modal noise, known-source initial vs noisy measured initial gives f800 broad/narrow raw .08619/.08805 vs .15054/.16094; phase .06791/.06901rad, wins32/32 and31/32, worst .1274/.1494. All predeclared gates pass, but this is NOT a fresh system cohort or realistic hydrophone test, and still ~6x noiseless error. `artifacts/coherent_waveguide_spectral_noise_confirm_20261004/summary.json`. The next same-PDE taper5→10 matched-RMS OOD protocol was committed before new truths: `docs/SPECTRAL_TAPER_SHIFT_POC_ZH.md`.

- **Newest independent confirmation: report §21 and docs/SPECTRAL_MEDIUM_STRESS_POC_ZH.md.** New32 correlated-spectral systems20265000–031, preregistered before new truth, fixed60/100/140Hz×65 depth context and rank256 PCA prior+5+5 classical GN: f800 broad/new2m raw .01475/.01532 vs original128 .51131/.49965; phase .01123/.01170rad. Both sources 32/32 wins; paired improvement97.11%/96.93%; worst raw .02263/.02439; all registered gates pass, numeric audit5.93e-5. Figure reports/figures/spectral_rank256_confirm.png. Rank256 is near the synthetic generator's255 random coefficients, not a compressed deep neural representation or second physical-domain confirmation. No target-medium leakage in background geometry/carrier/physical readout unit test. Paired old8 low-RMS control reduced the failed128D f800 raw .534/.544→.306/.310, so contrast alone does not explain failure. The preregistered conditional first8 fixed-code frequency curve is complete: f1000 broad/narrow raw .02081/.02016, phase .01615/.01588rad; all predeclared gates pass. Do not count this first8 as a second independent cohort or tune rank/steps on these32.

- **Read report §21 first.** Correlated16×16 spectral medium (same one-way PDE, RMS15m/s) is a harder statistical family. Its original frozen128-code on32 IID systems FAILS: f800 broad/new2m raw .517/.502, phase .499/.482rad. Training PCA128 covers99.8637% coefficient variance yet true-test-medium Euclidean projection still has f800 raw .391/.386. PCA256 extra-information projection is .00809/.00797, not a deployable result. Response-Jacobian-rotated rank128 oracle only .380/.385 and misses prereg gate. Do not infer phase quality from medium variance or local Jacobian energy.
- Post-hoc old8 DEV rank256 training PCA + same lowband60/100/140Hz context + 5+5 GN gives f800 broad/narrow raw .01489/.01638 and phase .01154/.01276rad; 1+1 GN already .0270/.0277, so a neural warm-start is not motivated in this dense noiseless protocol. This doubles code/global-basis/GN dimension and is classical statistical prior + physics, NOT a new deep AI model. NEW32 independent systems20265000–031 and gates were frozen at06:33:55 UTC before any new truth; their final result is in the new first bullet. Do not call old8 independent or alter new32 method after viewing results. A paired old8 RMS8.1689 versus15 control is running to diagnose intensity confounding.
- Extra-exact-family **first-order** analytic cosine prior (known16×16 basis but only linearizes speed→slowness²) failed old8 f800 raw .206/.228. True-medium Euclidean projection into this basis is even worse .564/.566 despite coefficient relative error .0149. This is not a strict response-capacity bound; the later exact nonlinear test in the first bullet **supersedes** the once-open question. Do not claim learning generally beats analytical physics. See docs/SPECTRAL_MEDIUM_STRESS_POC_ZH.md and its artifact manifests.

- Report sections 16–19 supersede earlier rolling status. Fixed low-band 128-real-code + frequency-adaptive physical readout reaches old8 development 800Hz raw .00351/.00356 for broad/new 2m source; this is not a deep-network result.
- Before new-data generation, 16/32-depth prior candidates, 32 IID systems20263400–431, source/code hashes and numerical gates were frozen. All32 finished: at400Hz raw .005966/.006084→.001767/.002072, at800Hz .011958/.012180→.004967/.005429. Relative paired improvements70.4%/65.9%/58.5%/55.4%, each31/32 wins, same system20263421 loses all four slightly. 800Hz 32-term phase RMSE .003987/.004254rad. 3 audit systems×2 frequencies max numeric raw2.50e-5. Full numbers/plots: docs/GENERATOR_NEW_SYSTEMS_HIGH_FREQUENCY_ZH.md and report section19. This is **independent systems within the same synthetic one-way waveguide family**, NOT a second domain or neural advantage.
- Old8 development-only depth64 prior at same128 per-system code improved 800Hz broad/narrow by17.3%/20.1%; global PCA basis/storage and readout cost increased. Do not retrofit it as a frozen new32 candidate.
- Traditional boundary harmonic closure predicts m32–63 coefficient tail to ~0.4% relative error, but even TRUE boundary slopes improve final800Hz raw only14.5%/16.4%; preregistered30% M1 gate failed. No ridge/MLP was trained. Coefficient accuracy != coherent response accuracy.
- Crucial synthetic limitation: hidden speed perturbation is exactly three 2D Gaussians, 15 continuous parameters; thermocline/source/background known. A separate **extra-exact-family** inversion baseline is preregistered on old8 development only in docs/GENERATOR_KNOWN_FAMILY_BASELINE_ZH.md. Its result cannot be called matched-information learning, and any late tests on new32 are post-freeze.
- That exact-family control is now complete on old8: initial 3-neighbor run got trapped on2/8 systems; post-hoc uniform24-neighbor restarts recovered all8 to context squared error≤1.60e-9. 800Hz broad/narrow raw .000266/.000254 versus PCA-code .003508/.003556. This is extra structure and more optimization, not a fair same-info baseline, but proves the synthetic family alone cannot establish learned-general representation. Do not report only the six easy systems or call the two initial traps identifiability.
- Original correlated-spectral frozen128 trial is COMPLETE and negative as above. Its background carrier and generator explicitly strip hidden spectral coefficients; tests check this. The new rank256 confirmation is a separate later freeze, not an originally registered candidate. Both remain same one-way PDE, never a second physical domain.
- The 2D Helmholtz secondary physics domain still favors near-exact classical inversion (~1e-4); compressed rank/PCA variants trade accuracy for storage, with no deep learning advantage. This remains a NO-GO for a current cross-domain AI-method paper claim. Next scientific question is response-sensitive low-order inference and genuinely more complex media, not another arbitrary neural component.

## Evidence and gates

- R0 per-field fitting is not a shared operator.
- New frequency with same-frequency spatial observations is not unobserved-frequency generation.
- R1 has zero target-frequency samples; never merge its scores with spatial reconstruction.
- Main metric: system-macro raw complex relative L2. No global phase alignment. Amplitude and energy-weighted circular phase metrics explain, never replace it.
- Keep low-amplitude/source/boundary diagnostics separate without removing them from primary ROI.
- Most coherent-branch datasets/results are development. NEW64 IID systems20262300–363 are a frozen independent audit; do not tune current candidates on them. Historical strict Ocean systems remain sealed.
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
Strong warm-start audit now complete: neural seeds42/43/44 20step .003444/.003436/.003637(mean.003506),
Gaussian-prior warmstart .004153, better than RBF .004642. Neural increment only15.6% vsbestmatched prior,
not robust20%;5step mean.005156 vsGaussian.005933. All3runs saved; one-repeat newer timing not certified.
Current new action:64independent IID systems20262300–363, candidates+hyperparameters frozen BEFOREdata
generation in dataset freeze_manifest.json. See `docs/GENERATOR_FROZEN_SYSTEM_AUDIT_ZH.md`. This audits generality
despite methodgate notyetpassed; do not claim it as second domain or reuse these systems to tune current models.
Generation complete:128NPZ,maxnumericaldifference1.004e-5. Seed42original64query complete: neural.008498,
kernel.010215;neural20.004127,Gaussian20.004702. Otherseeds pending; no gate claim. Originaldata filenames compatible with benchmark; narrow400Hz
truth uses64modes and newcoordinates. New evaluation code only adds hashes/logging/savedcoefficients, not model
changes; supplemental shared-training-array hashes are explicitly pre-evaluation, not falsely pre-generation.
Initial64evaluation interrupted bySIGTERM(unconfirmedcause); preserved20complete systems/resumedstrictly.
Firstresumeoverwrote runtime metadata beforebackup logicadded; do not claim perfect firstsegmenttiming provenance.
Proper nonlinearGaussian-priorcontrol staysin128trainingPCAcoordinates,20step.007668/100step~.00585;
capacityoracle.000810 vsfulltrueprojection.000782. No evidence for adding nonlinearlearnedlatentmanifold.
Localactual15physicalparameter sensitivity2olddevcases:110Hz amplification1.56–1.86,400Hz8.33–12.25.
Actualperturbationssmall/bound-limited, NOTglobalnonidentifiability or learnednoiseoptimality. See stabilityauditdoc.
Coherentinnovation input control now FAILED: same249520DOF,3000steps,zero-initializedextraweights,
exactinitialoutputmatch; extraobservedcomplexresidual best.006907 vszerofeaturecontinuation.006855.
Same-newinputRBF.010750 vsold.008874. No3seed/width/residual-frame scans. Currentclassicalcontrol uses
exactFULLtrajectory Frechet sensitivities in128trainingprior directions (notearlierlocalderivativematching).
See GENERATOR_TRAJECTORY_NEWTON_CONTROL_ZH.md; independent64frozencomparisonunchanged.
CRITICAL superseding FULLtrajectory GN control: trainingprior128 +Gaussianinit,1/2/5steps raw
.0007344/.0007027/.0007019;only~.336s firstcorrection(notendtoend). Neuralinit affine128directionupdate
stalls~.0059 because off-subspaceerrorcannotberemoved. Project neural into trainingprior first:
step0.006284,step1.000865,step5.000709. DoNOT claim neuralwarmstartadvantageagainstweakLBFGS/localGN.
PCApriorISdata-learned; thisnegatescurrentdeep-networknecessity,NOTallstatisticallearning. Newsource/f400audit
ongoing. No neuralpreconditioner until an actualnewcost/information bottleneck. Reportsection13supersedes12.
2026-10-04: ALL64frozen3seeds+newqueries COMPLETE. OriginaldirectNN mean.008560 vsRBF.010215(16.2%);
NN20.004105 vsGaussian20.004702(12.7%);NN5improvement20.6%CI14.9–25.5,NOTstable20gate.
POST-freeze supplementalGN1/2/5 oldquery .000824/.000684/.000677;newnarrow400 .005930/.005240/.005273,
vsNN3seedmean.04524. Parametersfixedold8,notnew64. DoNOTuse64auditfornewmodelselection.
ProfilebudgetOLD8only:17profilesGN5.000730;5profilesGN50.07774,priormeaninit50.7211.
Contextfitnearperfectbutlargewhitenedcoordinates. TestingONLYproperGaussianMAPstrength1e-8/1e-6/1e-4
fixed50steps;currentridgeonlypenalizesUPDATE,notfullpriornorm. DoNOTjumpfromfailed5steptoimpossibility.
MAP now COMPLETE: lambda1e-8/1e-6/1e-4,50step .008790/.007623/.014105 vsnoMAP.077742.
All3 improve; old5profilefailureNOTinformationlimit. Stopstrengthsweep/nonlinearpriorcapacityrestart.
CurrentboundedOLD8diagnostic: cubiccontinuous128priorbasis,64/128/256rangecells,0/1/2/5samecontextsteps;
true-medium16/32depthtermoracles separate. Checkcoarse-gridcompensation before claiming decoder refinement free.
Continuousrange auditCOMPLETE:16terms256cells+5steps oldquery.000226,narrow400.005106;oracle16f400.004861,
oracle32f400.000835. Depthbandwidth is an actualrepresentation bottleneck,notjustoutputsampling.
Same128latent,training512labels32depthterms,256cells5steps:oldquery.000127,narrow400.001552;
priorcapacityoracle.000184/.001335. Stilllinearlearnedphysicalprior+classicalinference,NOTdeepAIbreakthrough.
Source-subspace audit8olddev400Hz40orthogonalsources:rawmean.001700,meansystemworstcomplexcombination.005189,
maxsystemworst.015344;old16/64raw.006941. Newmatrixtruth[129,96,40],8NPZ+3auditNPZ;
1024/96vs2048/128worstsourceconvergencemax1.514e-5. DoNOTcall40sources40independentsystems.
Rank audit COMPLETE: all40source singular values flat, even last reception profile; best fixed output span
rank8/16/32 rawFrob .8944/.7746/.4472. Traditional linear algebra, not newtheorem or genericFNO/tensor bound.
Second-domain codec COMPLETE: same strong65grid inverse, uncompressed raw.0000984; speedSVD rank4/8
.000393/.000205; new512training-label PCA128 .001713. True speed hasrank<=4 BYCONSTRUCTION; doNOT
claim universal separability. All compression loses some accuracy. Same257teacher only, not1e-4 continuum.
No deep-learning necessity shown; no automaticAE orPCAcapacityscans. Report15 and mechanism10–12 updated.

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
