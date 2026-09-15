# Spectral Continuation Operator

从稀疏低频复波场中辨识可迁移的传播表示，并在训练频带之外查询更高频响应。

核心问题不是固定的 `20 Hz → 40 Hz` 映射，而是：相邻低频采样是否足以辨识谱相位速度，
以及一个轻量 propagation-aware operator 能否把这种信息转化为稳定的高频外推。

第一阶段在 Ocean Pekeris 上只用 90/100 Hz 选择模型。所有 checkpoint 冻结后一次性评估
105/110 Hz 新标签时，细频带 carrier-relative operator 的 raw complex RelL2 为 0.588/0.683，
强单频 direct baseline 为 1.437/1.389，三随机种子平均相对下降 59.0%/50.7%；每个 seed 的
每个 validation 环境都改善。零场阈值下的可用查询 horizon 从 100 Hz 延伸到 125 Hz。
同时，粗采样 WaveBench/Ocean 反例、in-range 性能损失和远频误差回升都被保留；当前结论
是“问题值得继续”，不是“方法已经可投稿”。

本仓库独立于 `APEX_repro_fsk`。旧仓库仅作为只读证据与数据生成器来源；所有新协议、代码、
实验和 artifact 在这里维护。

## 阅读顺序

1. [震荡复数场表征学习 POC](docs/OSCILLATORY_REPRESENTATION_POC_ZH.md)：最新方法、文献定位、
   正结果、失败归因与下一步；
2. [第一阶段中文主报告](reports/STAGE1_REPORT_ZH.md)：谱连续任务的起点、M1 结果与 go/no-go；
3. [复现说明](docs/REPRODUCIBILITY_ZH.md)：数据身份、命令和泄漏防护；
4. [APEX/APEX-H 冻结证据](docs/APEX_LEGACY_EVIDENCE_ZH.md)：旧结果可保留与不可保留的主张；
5. [实验前阶段计划](docs/STAGE1_PLAN_ZH.md)：用于核对哪些判断是在看结果前提出的。

## 代码入口

- `spectral_continuation.audit_baselines`：training-free identifiability 与线性对照；
- `spectral_continuation.train_stage1`：固定目标 matched 对照；
- `spectral_continuation.train_query_operator`：连续查询 M0/M1；
- `spectral_continuation.evaluate_horizon`：冻结 checkpoint 的一次性新频率评估；
- `spectral_continuation.summarize_query`：逐环境 paired 统计与正文图。
- `spectral_continuation.train_representation_poc`：matched image、delay 与稳定谱状态模型；
- `spectral_continuation.evaluate_prony_poc`：有限秩复谱状态 existence probe；
- `spectral_continuation.summarize_representation_poc`：POC 汇总与图表。

主指标始终是逐物理环境等权的 raw complex relative L2；幅值、圆周相位误差与 coherence
只用于机制诊断，不用于替换主指标。
