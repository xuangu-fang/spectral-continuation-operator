# 源码索引

源码暂不物理移动：历史 checkpoint、artifact manifest、命令和测试都依赖现有 module path。这里用接口
分区替代目录搬迁，避免为了整理破坏复现性。

## Active：当前 wave-system representation

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

## Supporting：共享基础设施

- `data.py`、`metrics.py`、`model.py`：数据、指标和通用网络组件；
- `paired_environment_stats.py`：逐环境 paired 统计；
- `representation.py`、`representation_data.py`：早期表示工具，若复用必须重新核对当前协议。

## Archived：历史实验实现

其余 `train_*`、`evaluate_*`、`summarize_*`、calibration、horizon、pole / recurrence、ocean
multi-fidelity 模块属于已结束的谱延拓与校准路线。它们保持可执行但默认冻结，不应被误认为当前模型的
组成部分。主要科学结论见 [`docs/archive/INDEX_ZH.md`](../../docs/archive/INDEX_ZH.md)，完整命令见
[`docs/REPRODUCIBILITY_ZH.md`](../../docs/REPRODUCIBILITY_ZH.md)。
