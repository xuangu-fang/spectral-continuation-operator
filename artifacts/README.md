# Artifact 索引与保存策略

## 当前正文级结果

当前 active artifact collections 为：

- `pbno_m1_summary/`：strict Q3 三 seed主表、方法图、budget curve 与 Ocean OOD safety；
- `pbno_q2_efficiency_formal/summary/`：OpenFWI rank 96 与 Deepwave rank 112 的固定配置三 seed结果、
  accuracy--compute 表、A100 latency 与汇总图；对应
  [`reports/PBNO_M2_EFFICIENCY_AND_BOUNDARIES_ZH.md`](../reports/PBNO_M2_EFFICIENCY_AND_BOUNDARIES_ZH.md)；
- `pbno_m2_receiver_residual/`：scale-consistent receiver residual 的本地 checkpoint 与机器 summary；该
  分支未过 stop line，不属于正文主方法；
- `cross_domain_response_query/`：OpenFWI、Deepwave、Ocean 的 archive manifest、M0/M1 汇总 JSON、预算主图与
  代表性 complex response 图；大型 NPZ、checkpoint 和逐 system CSV 本地保留、Git 排除；对应
  [`reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md`](../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md)；
- `multibranch_phase_space_mps1/summary/`：波长尺度 oracle、synthetic 三 seed、64→128 boundary paired
  statistics、WaveBench mechanism ablation 及汇总图；对应
  [`reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md`](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)；
- `multibranch_phase_space_mps0/` 与 `multibranch_phase_space_mps0b/`：固定窗口失败、尺度诊断和 oracle
  通过的机器可读证据；
- `wavebench_representation_m1/summary/`：公开 WaveBench 三 seed budget scaling、paired bootstrap、固定
  test examples、CSV/JSON 与正文图；对应
  [`reports/WAVEBENCH_REPRESENTATION_M1_ZH.md`](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)；
- `wave_set_m0/summary/`：最新 shared set operator 的三 seed budget scaling、robustness、CSV/JSON 与
  正文图；对应 [`reports/WAVE_SET_OPERATOR_M0_ZH.md`](../reports/WAVE_SET_OPERATOR_M0_ZH.md)；
- `wave_operator_representation/`：此前固定通道 representation POC 与完整消融。

`wave_operator_representation/` 包含：

- `summary/wave_representation_main.png`：检索、介质恢复、source budget 与 stress-test 主图；
- `summary/wave_representation_examples.png`：隐藏介质恢复定性结果；
- `summary/summary.json`：机器可读的主结果、三 seed 聚合与 failure boundary；
- 子目录中的 JSON / CSV / PDF：matched ablation 和绘图源数据。

完整解释见
[`reports/WAVEFIELD_REPRESENTATION_POC_ZH.md`](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

## 历史冻结 collections

其余目录记录早期 spectral continuation、continuous state、sparse calibration、reliable/actionable
horizon 与 paired multi-fidelity 实验。它们不再构成当前主页故事，只作为负结果、研究决策和复现证据
保留。对应结论见 [`docs/archive/INDEX_ZH.md`](../docs/archive/INDEX_ZH.md)。

## 保存规则

大型 checkpoint、逐样本数组与可再生中间结果留在 attached storage，并由 `.gitignore` 排除。Git 只
保存配置、manifest、汇总 JSON / CSV 和论文级 PNG / PDF。每个正式汇总应记录父数据与 checkpoint
hash；任何手工挑选的可视化都必须能追溯到机器可读 summary。
