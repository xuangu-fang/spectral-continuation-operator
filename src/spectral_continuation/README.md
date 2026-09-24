# 源码索引

源码暂不物理移动：历史 checkpoint、artifact manifest、命令和测试都依赖现有 module path。这里用接口
分区替代目录搬迁，避免为了整理破坏复现性。

## Active：当前 wave-system representation

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
