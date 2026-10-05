> 历史快照：2026-10-05 重整前版本；保留原结论与复现记录，不是当前计划。
> 当前入口：[三条 Wave as Prompt 路线](../../wave_as_prompt/README.md)。正文相对链接已重定位。

# 历史探索归档索引

> 本页保存研究路径、主要负结果和原始证据入口。它不是当前任务清单。历史代码、配置和 artifact
> 保留原路径，以维持 checkpoint、manifest、报告链接和测试的可复现性；除复现或重新审计外默认冻结。

若目的是决定下一项实验，而不是查找旧报告，请读更细的
[实验账本](../../EXPERIMENT_LEDGER_ZH.md)。实验账本额外记录了每个失败的作用域、不可重复项、重启条件和
尚未充分开发的正信号。

## 从历史路线中留下的五条科学结论

### 1. APEX / 谐波频率迁移

最初发现“amplitude 比 absolute phase 更容易跨频率迁移”是有效研究起点，但依赖 source、解析或近似
相位先验的方法泛化性有限。exact octave 是清楚的物理 benchmark，不应被写成任意频率泛化。

证据入口：[APEX/APEX-H 冻结证据](../../APEX_LEGACY_EVIDENCE_ZH.md)。

### 2. 连续频谱状态与远频率延拓

Pole、recurrence、Neural ODE、hypernetwork 和增加状态容量都没有消除长 horizon 的 complex error。
误差随传播距离与真实谱相位曲率增长，说明主要障碍是有限观测下的 phase identifiability，而不只是
模型容量。

证据入口：[连续谱生成器 Gate 报告](../../../reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md)、
[连续复谱状态](../../CONTINUOUS_SPECTRAL_STATE_ZH.md)。

### 3. 稀疏谱校准与多保真

在同一 high-fidelity archive 上，101--106 Hz 的稀疏 calibration 对 107--112 Hz 有机制级正信号：
raw complex RelL2 从 0.4106 降到 0.3611，关闭 dense-oracle gap 的 49.5%。但 12.1% 收益没有通过
预设 20% gate。更严格的独立 Ocean low/high-fidelity 数据中，dense oracle 也只改善 15.69%，没有
通过 30% task gate。由此不能把弱 simulator gap 包装成强多保真方法结果。

证据入口：[稀疏谱校准报告](../../../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)、
[独立多保真 Gate](../../../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)。

### 4. Reliable / Actionable Horizon

模型失败具有很强的可预测性：Ocean 与 synthetic 的 fixed-frequency failure AUROC 分别达到 0.906 和
0.960，空间 patch AUROC 也超过 0.92。但“知道会失败”不等于“现有 calibration action 能修复”；
repeated reset 没有继续降低误差，action oracle 相对最佳固定策略也只有 6.42% headroom。因此当前
action space 的 controllability 不足，不应继续扩大 selector。

证据入口：[Reliable-horizon 最终报告](../../../reports/RELIABLE_HORIZON_MAP_FINAL_ZH.md)、
[Actionable-horizon 报告](../../../reports/ACTIONABLE_HORIZON_POC_ZH.md)。

### 5. 全局 latent、局部 token 与当前转向

局部 patch span 的 oracle 很低，但自由学习局部复系数失败；受限 hyper-pole token 只改善 0.97%。这说明
“存在低维局部表示”不等于“能从有限频带稳定预测该表示”。当前项目因此把目标从远频率外推转向更底层的
response-conditioned system representation，并得到 amplitude + local phase advance 的强正结果。

证据入口：[全局与局部诊断](../../../reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)、
[局部复谱 Token 报告](../../../reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md)、
[当前 POC](../../../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

## 历史文档导航

| 阶段 | 协议 / 设计 | 结果 / 判断 |
|---|---|---|
| 初始谱连续 benchmark | [Stage-1 计划](../../STAGE1_PLAN_ZH.md) | [Stage-1 报告](../../../reports/STAGE1_REPORT_ZH.md) |
| 离散与连续谱状态 | [振荡场表征 POC](../../OSCILLATORY_REPRESENTATION_POC_ZH.md)；[连续谱状态](../../CONTINUOUS_SPECTRAL_STATE_ZH.md) | [Generator gate](../../../reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md) |
| 局部复谱表示 | [Local token 协议](../../LOCAL_SPECTRAL_TOKEN_POC_ZH.md) | [Global/local 诊断](../../../reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)；[Local token 结果](../../../reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md) |
| 稀疏谱校准 | [原始 POC](../../SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)；[few-shot multi-fidelity](../../FEW_SHOT_SPECTRAL_CALIBRATION_POC_ZH.md) | [稀疏校准](../../../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)；[solver audit](../../../reports/OCEAN_MULTIFIDELITY_SOLVER_AUDIT_ZH.md)；[paired-fidelity gate](../../../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md) |
| 可靠范围与动作价值 | [Reliable horizon](../../RELIABLE_HORIZON_MAP_POC_ZH.md)；[Actionable horizon](../../ACTIONABLE_HORIZON_POC_ZH.md) | [M0](../../../reports/RELIABLE_HORIZON_MAP_M0_ZH.md)；[M1](../../../reports/RELIABLE_HORIZON_MAP_M1_ZH.md)；[M2a](../../../reports/RELIABLE_HORIZON_MAP_M2A_ZH.md)；[final](../../../reports/RELIABLE_HORIZON_MAP_FINAL_ZH.md)；[action value](../../../reports/ACTIONABLE_HORIZON_POC_ZH.md) |

## 归档边界

- `docs/RESEARCH_MEMORY_ZH.md` 是旧的 few-shot calibration 统一记忆，现改为历史研究日志。
- `docs/REPRODUCIBILITY_ZH.md` 保存全部旧命令和数据 hash，是技术复现档案，不是推荐阅读入口。
- 除两个 `wave_operator_*_20260922.json` 外，`configs/` 中现有配置都属于历史阶段。
- `src/spectral_continuation/` 中不以 wave-operator representation 为中心的训练、校准、horizon 与汇总
  脚本默认冻结；完整分类见源码目录内的 README。
- 历史 artifact 仅用于核对结论。新的正文材料只从 `artifacts/wave_operator_representation/` 取数。
