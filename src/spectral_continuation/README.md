# 源码索引

源码暂不物理移动：历史 checkpoint、artifact manifest、命令和测试都依赖现有 module path。这里用接口
分区替代目录搬迁，避免为了整理破坏复现性。

## Active：当前 wave-system representation

- `global_bayesian_response_operator.py`：strict Q3 global complex response basis、解析 posterior、retrieval
  baseline 与 system-level evaluation；
- `train_prompted_bayesian_neural_operator.py`：PBNO 的 zero-initialized query basis、nonlinear latent
  transport，以及已经冻结的 scale-consistent receiver residual development branch；
- `posterior_memory_response_operator.py`：M3 posterior-kernel attention；用相同 prompt mask 的稳定
  posterior tokens 定义 operator memory keys，包含核温度、validation shrinkage、vectorized deployment
  与 M4 spectral coefficient value bank；
- `audit_spectral_memory_compression.py`：M4 rank-16 operator value dictionary、validation storage--accuracy
  selection 与三 seed test；
- `generate_structural_lattice_wave_data.py`：独立阻尼 mass--spring structural-wave strict-Q3 数据生成器；
- `audit_pbno_noise_robustness.py`：冻结 clean 配置后的 20/10 dB complex prompt-noise 审计；
- `audit_prompt_only_router.py`、`train_prompt_risk_router.py`：hard LOO 与 shared risk-head selection 的
  NO-GO 证据，并导出 target-free prompt/LOO features 与复 residual cross term；
- `train_unified_prompt_prior_adapter.py`：不读 domain ID 的 1,665 参数连续 kernel--neural prior composition，
  统一 K=1/2/4/8；
- `train_prompt_prior_adapter.py`、`train_k1_prompt_prior_adapter.py`、`audit_k1_prior_blend.py`：形成统一模型前的
  分预算机制审计，只作消融，不作为最终部署入口；
- `summarize_pbno_m4.py`：四域主表、storage Pareto、noise 图、hard-router 边界与 unified-adapter 三 seed 图；
- `train_matched_q3_cnp.py`：参数匹配 generic CNP 强神经基线，支持 variable K、resume 与严格 Q3；
- `summarize_pbno_m3.py`：M3 三域主表、paired bootstrap、预算曲线、逐系统散点、OOD 与方法图；
- `benchmark_posterior_kernel_operator.py`：posterior-kernel 参数、response-bank storage、complex-MAC proxy
  与 A100 端到端延迟；
- `evaluate_pbno_checkpoint.py`：同一 PBNO checkpoint 的 Q2/Q3 与 IID/OOD 评测；
- `summarize_pbno_m1.py`：PBNO 三 seed 主表、预算曲线、方法图与 Ocean OOD safety 汇总；
- `audit_pbno_latent_belief.py`：posterior-latent memory、best-of-N oracle、conditional target spread 与
  flow existence gate；
- `audit_functional_tensor_rank.py`：当前 Wave-as-Prompt M0；检验冻结 complex response bases 是否存在更低
  存储的 source--receiver functional rank，并同时审计 sparse-prompt query error 与结构化 reciprocity；
- `train_episodic_response_basis.py`：PFTO bridge；保持闭式 prompt latent solve，只把 PCA basis 改为针对
  held-out-source episodes 学习，用于判断 task-optimized representation 是否存在可利用 headroom；
- `probabilistic_response_operator.py`：由 train-system response spectrum 学各 latent direction prior variance，
  对 sparse prompts 做闭式 MAP system identification；scalar ridge 是其严格 matched baseline；
- `summarize_latent_prior_multiseed.py`：冻结一次 validation selection 后，在三组 prompt-order seed 上按
  physical system 聚合并生成 paired statistics、预算曲线与 probe saving；
- `audit_continuous_response_query.py`：全局交错 receiver-coordinate holdout、linear/cubic/Fourier sampling
  oracle 与 joint held-out-source/receiver query；
- `audit_prompt_adaptive_prior.py`：不读取 target 的 leave-one-prompt-source-out prior selector，用于诊断与
  缓解 response-manifold OOD 时的 prior overconfidence；
- `summarize_continuous_sampling.py`：汇总 Deepwave receiver-grid convergence 与 Ocean 可辨识 continuous
  query，生成 sampling-boundary 图；
- `cross_domain_response_data.py`：OpenFWI、Deepwave、Ocean 与 The Well 的统一 complex response archive；
- `cross_domain_query_audit.py`：互易性、local baselines、validation-selected complex response subspace；
- `train_cross_domain_query.py`：receiver-preserving set attention residual operator；
- `evaluate_cross_domain_query.py`：冻结 checkpoint 的 nested-budget system-level evaluation；
- `diagnose_cross_domain_query.py`：amplitude 与 weighted phase 归因；
- `summarize_cross_domain_query.py`：三 seed 逐系统 paired 汇总；
- `plot_cross_domain_query.py`：跨域预算曲线与代表性复响应正文图；

以下 `response_query*` 是上一轮单域 RQ-1，继续作为基线与历史证据：

- `response_query.py`：Q1/Q2/Q3 episode protocol、receiver arrays、normalization 与 CNP/Transformer models；
- `audit_response_query.py`：PCA/retrieval/interpolation、reciprocity 与 calibrated reciprocity baselines；
- `train_response_query.py`：物理 routing + residual Transformer 的两阶段训练；
- `evaluate_response_query_checkpoint.py`：冻结 IID/OOD split、paired-system bootstrap；
- `summarize_response_query.py`：三 seed 正式表格与预算曲线；
- `plot_response_query_examples.py`：复响应 magnitude/phase 定性图；
- `audit_active_source_query.py`：complete-table best-next-source oracle；当前 6.0% headroom，active gate 未过；
- `fourier_response_operator.py`、`train_response_query_fno_oracle.py`：true-medium decoder 负结果审计，
  不是最终方法。

- `wave_operator_data.py`：system-major 多源、多频数据生成；
- `audit_wave_operator_representation.py`：RBF、reciprocity、Nyström 与 view-rank 审计；
- `train_wave_operator_identifier.py`：missing-block 完整复场负对照；
- `train_wave_operator_contrastive.py`：跨 source-frequency system retrieval；
- `train_wave_medium_probe.py`：hidden-medium spatial sufficiency 与 noise training；
- `train_wave_set_encoder.py`：可变观测预算的 self-supervised shared set encoder；
- `probe_wave_set_encoder.py`：冻结/微调 spatial latent、可变预算与跨 view-group probe；
- `wave_set_encoder.py`：共享 view backbone、moment/slot/coordinate-basis aggregation；
- `evaluate_wave_set_probe.py`：set operator 的 geometry、noise、resolution 与 boundary shift；
- `summarize_wave_set_m0.py`：三 seed budget scaling 与 paired bootstrap；
- `summarize_wave_set_robustness.py`：shift 汇总与正文级可视化；
- `evaluate_wave_medium_ood.py`：geometry、noise、resolution、boundary stress tests；
- `summarize_wave_representation_poc.py`：汇总表、正文图和机器可读 summary。
- `train_wavebench_identifier.py`：依赖无关的 WaveBench FFCV reader、可变频率 system-identification
  matched experiment 与 fixed edge development variants；
- `summarize_wavebench_representation.py`：WaveBench 三 seed、逐系统 paired bootstrap 与预算主图；
- `plot_wavebench_examples.py`：固定测试系统的定性恢复与误差图。
- `audit_multibranch_phase_space.py`：当前 MPS-0 oracle；在受控混合与真实 open/partial/closed patches 上
  用 spatial holdout 比较 K=1 和 top-K 局部 complex wave branches。
- `audit_wavelength_scaled_phase_space.py`：MPS-0b 尺度诊断；比较固定大窗口与 wavelength-scaled windows，
  将重复 patch seeds 先按 physical system 聚合后再做 paired bootstrap。
- `summarize_multibranch_phase_space.py`：MPS-0/1 oracle、synthetic 三 seed、boundary paired statistics 与
  WaveBench mechanism ablation 的冻结汇总和主图。

MPS-1 的 fixed multiscale packet Gram 已完成但未过 15% promotion gate；相关 representation 分支继续
保留用于复现，不应在没有新机制假设时继续枚举固定通道。

## Supporting：共享基础设施

- `data.py`、`metrics.py`、`model.py`：数据、指标和通用网络组件；
- `paired_environment_stats.py`：逐环境 paired 统计；
- `representation.py`、`representation_data.py`：早期表示工具，若复用必须重新核对当前协议。

## Archived：历史实验实现

其余 `train_*`、`evaluate_*`、`summarize_*`、calibration、horizon、pole / recurrence、ocean
multi-fidelity 模块属于已结束的谱延拓与校准路线。它们保持可执行但默认冻结，不应被误认为当前模型的
组成部分。主要科学结论见 [`docs/archive/INDEX_ZH.md`](../../docs/archive/INDEX_ZH.md)，完整命令见
[`docs/REPRODUCIBILITY_ZH.md`](../../docs/REPRODUCIBILITY_ZH.md)。
