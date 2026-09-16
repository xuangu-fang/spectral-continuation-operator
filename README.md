# Spectral Continuation Operator

从稀疏低频复波场中辨识可迁移的传播表示，并在训练频带之外查询更高频响应。

核心问题不是固定的 `20 Hz → 40 Hz` 映射，而是：相邻低频采样是否足以辨识谱相位速度，
以及一个轻量 propagation-aware operator 能否把这种信息转化为稳定的高频外推。

最新阶段首先诊断了原始 `74–80 Hz → 110 Hz` 长程设置：误差随传播距离和真实谱相位曲率
显著增加，增加 pole 数、仿射多分量、Neural ODE 和局部曲率都不能突破可辨识性边界。随后在
明确标注为 development calibration 的协议中，模型看 90–100 Hz、用 101–106 Hz 监督，冻结后
查询 107–125 Hz。加入 64 维环境级表征和受限 hyper-carrier 后，110 Hz 三 seed raw complex
RelL2 为 **0.4473 ± 0.0048**，phase coherence 为 **0.8603 ± 0.0020**，通过预设 gate。
matched recurrence 为 0.4741，相近参数量 carrier-relative query U-Net 三 seed 为 0.6125；
方法的优势集中在 107–112 Hz，113 Hz 后与 recurrence 发生交叉。

当前结论是：从低频带估计相位不变的局部传播状态，确实比像素空间 query fitting 更适合
频带外 continuation；误差仍随距离增长，因此还不能主张任意远频恢复。

本仓库独立于 `APEX_repro_fsk`。旧仓库仅作为只读证据与数据生成器来源；所有新协议、代码、
实验和 artifact 在这里维护。

## 阅读顺序

1. [全局隐空间与局部 token 诊断](reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)：与同目录低秩项目的
   证据合并、子空间旋转、patch token oracle 以及 masked representation 下一步；
2. [连续谱生成器 Gate 报告](reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md)：最新诊断、方法搜索、
   最终协议、三 seed 结果和结论边界；
3. [连续复谱状态](docs/CONTINUOUS_SPECTRAL_STATE_ZH.md)：上一阶段 pole–residue 方法与稠密协议；
4. [震荡复数场表征学习 POC](docs/OSCILLATORY_REPRESENTATION_POC_ZH.md)：离散谱状态的起点、
   文献定位和失败归因；
5. [第一阶段中文主报告](reports/STAGE1_REPORT_ZH.md)：谱连续任务的 M1 结果与 go/no-go；
6. [复现说明](docs/REPRODUCIBILITY_ZH.md)：数据身份、命令和泄漏防护；
7. [APEX/APEX-H 冻结证据](docs/APEX_LEGACY_EVIDENCE_ZH.md)：旧结果可保留与不可保留的主张；
8. [实验前阶段计划](docs/STAGE1_PLAN_ZH.md)：用于核对哪些判断是在看结果前提出的。

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
- `spectral_continuation.oracle_spectral_diagnostics`：低频可辨识性与 representation oracle。
- `spectral_continuation.analyze_error_mechanism`：按距离、干涉节点和谱曲率归因误差。
- `spectral_continuation.summarize_fractional_evaluations`：独立非整数频率评测汇总。
- `spectral_continuation.diagnose_global_subspace`：全局/局部复数子空间与 masked-frequency oracle。
- `spectral_continuation.summarize_global_subspace`：子空间旋转、局部性与容量权衡图。

主指标始终是逐物理环境等权的 raw complex relative L2；幅值、圆周相位误差与 coherence
只用于机制诊断，不用于替换主指标。
