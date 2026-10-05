> 历史快照：2026-10-05 重整前版本；保留原结论与复现记录，不是当前计划。
> 当前入口：[三条 Wave as Prompt 路线](../../wave_as_prompt/README.md)。正文相对链接已重定位。

# 报告导航

> 更新日期：2026-10-03。项目的当前故事、claim boundary 和未来排序统一见
> [`docs/CURRENT_STORY_AND_ROADMAP_ZH.md`](../../CURRENT_STORY_AND_ROADMAP_ZH.md)。本目录是证据档案，
> “保留”不等于“仍应继续优化”。

## 当前报告

[连续复波场表征：进度、方法、图与正负结论](../../../reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md)。
目标执行中，所有新结果仍是development；旧active query暂停。

## 历史Wave Interrogation阶段的三份报告

1. [`ACTIVE_WAVE_INTERROGATION_M1_ZH.md`](../../../reports/ACTIVE_WAVE_INTERROGATION_M1_ZH.md)：该历史阶段主报告。包含
   untouched IID/OOD systems、七类 acquisition baseline、成本轴、continuous learned utility、top-4
   analytic certification，以及 strict coordinate holdout、速度和 Ocean OOD 的失败边界。
2. [`ACTIVE_WAVE_INTERROGATION_M0_ZH.md`](../../../reports/ACTIVE_WAVE_INTERROGATION_M0_ZH.md)：主任务如何从固定 prompts
   推进到 target-aware source--frequency experimental design。
3. [`PBNO_M4_RISK_RESOLUTION_ZH.md`](../../../reports/PBNO_M4_RISK_RESOLUTION_ZH.md)：当前 operator belief 的来历、
   spectral dictionary、第四物理域和统一 adapter。

## 与历史PBNO方法直接相关的报告

- [`PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md`](../../../reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md)：Bayesian tokenizer、
  neural latent transport 与 posterior-kernel operator。
- [`PBNO_M2_EFFICIENCY_AND_BOUNDARIES_ZH.md`](../../../reports/PBNO_M2_EFFICIENCY_AND_BOUNDARIES_ZH.md)：rank/compute
  Pareto、receiver residual 修复和明确停止线。
- [`PBNO_JOINT_QUERY_M1_ZH.md`](../../../reports/PBNO_JOINT_QUERY_M1_ZH.md)：strict joint source--frequency query 的早期结果。
- [`WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md`](../../../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md)：latent spectrum prior 与
  三域强基线。
- [`CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md`](../../../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md)：OpenFWI、Deepwave、
  Ocean 的跨域 response-query POC。

## 已关闭的方法分支

以下报告很重要，因为它们记录了**不应重复的实验**：

- [`PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md`](../../../reports/PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md)：复数
  operator prior 最佳增益 2.93%，未过 5% gate；不要继续扫容量。
- [`HETEROGENEOUS_FUNCTIONAL_OPERATOR_M0_ZH.md`](../../../reports/HETEROGENEOUS_FUNCTIONAL_OPERATOR_M0_ZH.md)：
  functional encoder、reranker 和 nonlinear neural field 均未把表征容量转化为 sparse identifiability。
- [`PBNO_POSTERIOR_COMPOSER_M0_ZH.md`](../../../reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md)：learned correction 在 selection
  为正、unseen systems 回退；只保留 zero-update fallback 接口。
- [`MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md`](../../../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)：多分量 oracle 成立，但
  fixed packet representation 未过 15% promotion gate。
- [`WAVEBENCH_REPRESENTATION_M1_ZH.md`](../../../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)：公开 WaveBench 的正结果和
  fixed complex-edge token 负结果。
- [`WAVE_SET_OPERATOR_M0_ZH.md`](../../../reports/WAVE_SET_OPERATOR_M0_ZH.md)：可变 source--frequency budget 的 set
  operator、noise/OOD 边界与失败预训练。

## 更早的科学探索

连续谱、可靠延拓视界、稀疏 calibration、局部 token 和独立多保真等早期工作已经压缩到
[`docs/archive/INDEX_ZH.md`](../INDEX_ZH.md)。需要追溯某个结论时再读，不要把旧计划当成当前任务。

## 机器可读结果

- M1 summary：[`artifacts/active_wave_interrogation_m1_summary/summary.json`](../../../artifacts/active_wave_interrogation_m1_summary/summary.json)
- 实验账本：[`docs/EXPERIMENT_LEDGER_ZH.md`](../../EXPERIMENT_LEDGER_ZH.md)
- 当前状态：[`docs/CURRENT_STATUS_ZH.md`](../../CURRENT_STATUS_ZH.md)
