# 配置索引

## 当前主线

- `in_context_response_query_rq0_20260924.json`：当前 active Response Query 协议；固定 receiver array，
  Q1/Q2/Q3、1--16 context probes、information audits、最小 CNP 与 belief/active 的 stop rules。
- `wave_operator_representation_poc_20260922.json`：160-system 主数据、split、episode 与最初 gates；
- `wave_operator_ood_ellipses_20260922.json`：未见 ellipse geometry 的 OOD 数据。
- `wave_system_set_encoder_m0_20260922.json`：M0 自监督 set encoder 的预注册探索协议。
- `wave_set_operator_m0_formal_20260922.json`：冻结后的三 seed variable-budget 正式协议与 claim 边界。
- `wavebench_representation_m1_20260923.json`：公开 WaveBench 的系统级固定 split、三 seed matched
  representation comparison、15% promotion gate 与开发型 edge variants。
- `multibranch_phase_space_mps0_20260923.json`：多分支相空间方向的 oracle-first 协议；先比较 K=1 与
  top-K local wave packets 在 held-out pixels 上的 complex error，未过 15% gate 不训练 learned router。
- `multibranch_phase_space_mps0b_20260923.json`：MPS-0 固定大窗口失败后的 wavelength-scaled 诊断；
  窗口只由公开 frequency、grid spacing 和 reference speed 决定，并用三个 patch seeds 验证。
- `multibranch_phase_space_mps1_20260923.json`：oracle 通过后的 matched downstream POC；冻结 two-scale
  complex packet Gram、residual initialization、三 seed boundary gate 与 2% IID safety boundary。

`in_context_response_query_rq0_20260924.json` 在 test 未打开前只允许补充机器生成的 manifest 路径，不改变
任务语义或 gate。后续新阶段使用新的日期化配置，不回写已经产生结果的文件。

## 历史冻结配置

其余配置属于 spectral continuation、continuous-state、sparse calibration、reliable/actionable horizon
或 independent multi-fidelity 阶段。它们保留在原位置，是因为报告、artifact manifest 和自动测试直接
记录了这些路径。除复现或审计外，不再把它们作为新实验模板。

历史阶段的结论和报告入口见 [`docs/archive/INDEX_ZH.md`](../docs/archive/INDEX_ZH.md)。
