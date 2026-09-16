# Spectral Continuation Operator

从廉价模拟先验和少量测试时观测中辨识新物理实例的谱响应算子，并在完全无观测的条件上查询
复杂波场。

核心问题不是固定的 `20 Hz → 40 Hz` 映射，而是 few-shot spectral calibration：预训练
模型提供跨实例共享的幅值和空间先验，固定稀疏阵列的短频率 sweep 辨识实例相关的 phase-sensitive
latent state，连续 query operator 再回答 support 未覆盖的频率和空间位置。

当前统一任务定义、应用原型、声明边界和强制报告规则见
[研究记忆](docs/RESEARCH_MEMORY_ZH.md)。频率外推是主要 benchmark；新实例上的 latent operator
calibration 才是 AI 任务；system identification 只解释 calibration 所恢复的内部状态。

最新阶段首先诊断了原始 `74–80 Hz → 110 Hz` 长程设置：误差随传播距离和真实谱相位曲率
显著增加，增加 pole 数、仿射多分量、Neural ODE 和局部曲率都不能突破可辨识性边界。随后在
明确标注为 development calibration 的协议中，模型看 90–100 Hz、用 101–106 Hz 监督，冻结后
查询 107–125 Hz。加入 64 维环境级表征和受限 hyper-carrier 后，110 Hz 三 seed raw complex
RelL2 为 **0.4473 ± 0.0048**，phase coherence 为 **0.8603 ± 0.0020**，通过预设 gate。
matched recurrence 为 0.4741，相近参数量 carrier-relative query U-Net 三 seed 为 0.6125；
方法的优势集中在 107–112 Hz，113 Hz 后与 recurrence 发生交叉。

当前结论是：从低频带估计相位不变的局部传播状态，确实比像素空间 query fitting 更适合
频带外 continuation；误差仍随距离增长，因此还不能主张任意远频恢复。

最新的稀疏谱校准 POC 研究一个不同于同频稀疏重建的任务：给定完整 90--100 Hz，再用固定稀疏
阵列扫描 101--106 Hz，预测完全无观测的 107--125 Hz。六频稳定 AR 状态在每频率 0.5% 传感器下
将 107--112 Hz raw complex RelL2 从 0.4106 降至 0.3611，关闭 dense oracle gap 的 49.5%，且
shuffled-instance control 明显恶化。结果主要来自相位改善，但 12.1% 相对收益没有通过预注册
20% gate；这是机制级正结果，不是已完成的论文主方法。

这项 POC 尚未实例化真正的 multi-fidelity setting：完整低频场仍来自同一个 high-fidelity archive，
source 也基本固定。下一阶段先生成独立 low/high fidelity pairs，验证 cheap simulator prior 加 sparse
high-fidelity calibration 的 task gate；通过后才比较通用适配与 phase-aware multi-component state。

最新的局部 token POC 进一步区分了“可表示”与“可预测”：8×16 patch span 的 oracle 很低，
但学习自由局部复系数在 110 Hz 仅得到 0.6186，frequency-mask 辅助训练还会退化。把 token
限制为 frozen hyper-pole 上的局部 carrier slope/curvature 修正后，110 Hz 从 0.4409 小幅降到
0.4367（paired 95% CI 为正），但相对增益仅 0.97%，未通过方法 gate。当前不继续扩大 token、
Transformer 或 mask 路线。

本仓库独立于 `APEX_repro_fsk`。旧仓库仅作为只读证据与数据生成器来源；所有新协议、代码、
实验和 artifact 在这里维护。

## 阅读顺序

1. [当前研究记忆](docs/RESEARCH_MEMORY_ZH.md)：统一任务、AI 定位、应用原型和不可越过的声明边界；
2. [下一阶段 POC 设计](docs/FEW_SHOT_SPECTRAL_CALIBRATION_POC_ZH.md)：独立 fidelity、
   最小方法阶梯、baseline、预算和 gate；
3. [稀疏谱校准 POC](reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)：新任务、两频失败、六频稳定
   复谱状态、三 seed 正式结果和 NO-GO 判定；
4. [局部复谱 Token POC](reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md)：自由 local-span、masked
   reconstruction 和受限 carrier-token 的结果与停止判断；
5. [全局隐空间与局部 token 诊断](reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)：与同目录低秩项目的
   证据合并、子空间旋转、patch token oracle 以及 masked representation 下一步；
6. [连续谱生成器 Gate 报告](reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md)：最新诊断、方法搜索、
   最终协议、三 seed 结果和结论边界；
7. [连续复谱状态](docs/CONTINUOUS_SPECTRAL_STATE_ZH.md)：上一阶段 pole–residue 方法与稠密协议；
8. [震荡复数场表征学习 POC](docs/OSCILLATORY_REPRESENTATION_POC_ZH.md)：离散谱状态的起点、
   文献定位和失败归因；
9. [第一阶段中文主报告](reports/STAGE1_REPORT_ZH.md)：谱连续任务的 M1 结果与 go/no-go；
10. [复现说明](docs/REPRODUCIBILITY_ZH.md)：数据身份、命令和泄漏防护；
11. [APEX/APEX-H 冻结证据](docs/APEX_LEGACY_EVIDENCE_ZH.md)：旧结果可保留与不可保留的主张；
12. [实验前阶段计划](docs/STAGE1_PLAN_ZH.md)：用于核对哪些判断是在看结果前提出的。

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
- `spectral_continuation.summarize_token_poc`：局部 token POC 曲线与逐环境 paired 统计。
- `spectral_continuation.train_ar_calibrator`：不用 future labels 训练稀疏宽带稳定谱状态校准器。
- `spectral_continuation.evaluate_ar_calibration`：冻结模型的预算、mask、horizon 与负对照评测。
- `spectral_continuation.summarize_sparse_calibration`：跨 seed 聚合、paired 统计和正文图。

主指标始终是逐物理环境等权的 raw complex relative L2；幅值、圆周相位误差与 coherence
只用于机制诊断，不用于替换主指标。
