# 下一阶段协议：Wave-System Encoder

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 导航：[项目主页](../README.md) · [一页式当前状态](CURRENT_STATUS_ZH.md) ·
> [实验账本](EXPERIMENT_LEDGER_ZH.md) ·
> [波动场表征学习 POC 报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)
>
> 本文是实验前执行协议，不记录完成后的结果解释。

> **状态更新（2026-09-22）：** M0 shared set encoder 已完成。正式结果与后续决策以
> [Wave Set Operator M0 报告](../reports/WAVE_SET_OPERATOR_M0_ZH.md) 为准；本文保留为预注册协议，
> 不回写事后 gate。

## 1. 冻结的科学命题

当前最值得继续验证的命题不是“某个 U-Net 能否重建另一频率”，而是：

> 局部能量与固定物理尺度上的复相位增量，能否作为跨 source、frequency、resolution 和 boundary 的
> 通用 response token，并由少量 response views 识别一个可查询的 wave-system latent？

## 2. 阶段顺序

### M0：Set encoder 与公平预算曲线

- shared view encoder；
- permutation-invariant aggregation；
- source/frequency/spatial mask 随机化；
- 所有 budget 共用同一组参数；
- 对照 raw complex、amplitude-only、5-channel token、Riesz-style derivative features。

通过条件：5-channel token 在 coefficient recovery、cross-view retrieval 和 calibration 三者中至少两项
稳定优于 amplitude-only，且每个 source budget 方向一致。

### M1：Multi-scale physical phase advance

只加一个组件：多个固定物理 offset。暂不加 Transformer、生成 loss 或 PDE residual。

通过条件：64→128 和 open→partial/closed 两类 shift 均有稳定增益；20 dB 误差低于 0.15。

### M2：Oracle decoder

使用真实 coefficient field 训练 query decoder。decoder 没过 0.30 raw complex RelL2 时，不接 learned
system encoder。

### M3：Encoder--decoder cascade

冻结或弱微调已经通过的 decoder，比较：

- true coefficient；
- predicted coefficient；
- unconstrained spatial latent；
- global latent；
- generic set operator。

### M4：第二数据域与部分观测

优先做 contiguous patches、boundary sensors 和一个独立 PDE 数据源。不得把同一 simulator 的 geometry
variation 写成跨物理泛化。

## 3. 强制报告

- 系统级 split；
- source 数、frequency 数、spatial coverage 与总复观测数；
- grid spacing 与 phase-increment 的物理 offset；
- clean、30、20、10 dB；
- 每种 boundary 独立指标；
- retrieval、coefficient recovery、raw complex query error；
- 参数量、训练 seed 与 failure cases；
- amplitude fallback 与 confidence gate 的触发比例。

## 4. Stop rules

- 若 multi-scale token 在 partial/closed 上不优于 amplitude-only 20%，停止局部差分路线；
- 若 oracle decoder 不能低于 0.30，不训练端到端 query model；
- 若第二数据域只在 coefficient labels 可见时有效，不主张通用 self-supervised representation；
- 若 10 dB 下 learned gate 不能退回 amplitude baseline，停止宣称 noise robustness。
