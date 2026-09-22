# 配置索引

## 当前主线

- `wave_operator_representation_poc_20260922.json`：160-system 主数据、split、episode 与最初 gates；
- `wave_operator_ood_ellipses_20260922.json`：未见 ellipse geometry 的 OOD 数据。
- `wave_system_set_encoder_m0_20260922.json`：M0 自监督 set encoder 的预注册探索协议。
- `wave_set_operator_m0_formal_20260922.json`：冻结后的三 seed variable-budget 正式协议与 claim 边界。

下一阶段的新实验必须使用新的日期化配置，不回写这些已产生结果的文件。

## 历史冻结配置

其余配置属于 spectral continuation、continuous-state、sparse calibration、reliable/actionable horizon
或 independent multi-fidelity 阶段。它们保留在原位置，是因为报告、artifact manifest 和自动测试直接
记录了这些路径。除复现或审计外，不再把它们作为新实验模板。

历史阶段的结论和报告入口见 [`docs/archive/INDEX_ZH.md`](../docs/archive/INDEX_ZH.md)。
