# Spectral Continuation Operator

从稀疏低频复波场中辨识可迁移的传播表示，并在训练频带之外查询更高频响应。

核心问题不是固定的 `20 Hz → 40 Hz` 映射，而是：相邻低频采样是否足以辨识谱相位速度，
以及一个轻量 propagation-aware operator 能否把这种信息转化为稳定的高频外推。

最新阶段使用 74–80 Hz 可见场和 81–100 Hz 稠密监督，比较近似严格参数匹配的
carrier-gauged pole–residue、稳定递推和 query-conditioned image U-Net。三者都能拟合训练频带：
pole 与 U-Net 的 raw complex RelL2 分别为 0.276 和 0.289；但冻结后在 101–125 Hz 上分别为
0.669 和 1.217，结构化谱状态降低 **45.1%**。相对稳定递推的 0.765，直接连续查询降低
**12.6%**。在训练后才生成的六个半整数频率上，pole 为 0.676，U-Net 为 1.181。主要差距
来自相位，而不是幅值指标，且三 seed 方向稳定。

当前结论是：从低频带估计相位不变的局部传播状态，确实比像素空间 query fitting 更适合
频带外 continuation；误差仍随距离增长，因此还不能主张任意远频恢复。

本仓库独立于 `APEX_repro_fsk`。旧仓库仅作为只读证据与数据生成器来源；所有新协议、代码、
实验和 artifact 在这里维护。

## 阅读顺序

1. [连续复谱状态](docs/CONTINUOUS_SPECTRAL_STATE_ZH.md)：当前长程目标、pole–residue 方法、
   最新结果与严格结论边界；
2. [震荡复数场表征学习 POC](docs/OSCILLATORY_REPRESENTATION_POC_ZH.md)：离散谱状态的起点、
   文献定位和失败归因；
3. [第一阶段中文主报告](reports/STAGE1_REPORT_ZH.md)：谱连续任务的 M1 结果与 go/no-go；
4. [复现说明](docs/REPRODUCIBILITY_ZH.md)：数据身份、命令和泄漏防护；
5. [APEX/APEX-H 冻结证据](docs/APEX_LEGACY_EVIDENCE_ZH.md)：旧结果可保留与不可保留的主张；
6. [实验前阶段计划](docs/STAGE1_PLAN_ZH.md)：用于核对哪些判断是在看结果前提出的。

## 代码入口

- `spectral_continuation.audit_baselines`：training-free identifiability 与线性对照；
- `spectral_continuation.train_stage1`：固定目标 matched 对照；
- `spectral_continuation.train_query_operator`：连续查询 M0/M1；
- `spectral_continuation.evaluate_horizon`：冻结 checkpoint 的一次性新频率评估；
- `spectral_continuation.summarize_query`：逐环境 paired 统计与正文图。
- `spectral_continuation.train_representation_poc`：matched image、delay 与稳定谱状态模型；
- `spectral_continuation.evaluate_prony_poc`：有限秩复谱状态 existence probe；
- `spectral_continuation.summarize_representation_poc`：POC 汇总与图表。
- `spectral_continuation.evaluate_representation_checkpoint`：冻结连续谱状态在独立查询数据上的评测；
- `spectral_continuation.summarize_continuous_state`：跨 seed 频率曲线、机制指标与正文图。

主指标始终是逐物理环境等权的 raw complex relative L2；幅值、圆周相位误差与 coherence
只用于机制诊断，不用于替换主指标。
