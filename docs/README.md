# 文档导航

## 当前必读

1. [`PBNO_STORY_PLAIN_LANGUAGE_ZH.md`](PBNO_STORY_PLAIN_LANGUAGE_ZH.md)：不要求波动背景的大白话故事、方法与
   实验入口；
2. [`../reports/PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md`](../reports/PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md)：
   当前最新方法审计；learning 只负责 operator prior、解析 proximal consistency、跨 receiver-mask 结果与
   5% gate 未通过的边界；
3. [`../reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md`](../reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md)：上一轮
   方法边界；统一 composer 成立，但 learned correction 未通过 unseen-system gate；
4. [`../reports/PBNO_M4_RISK_RESOLUTION_ZH.md`](../reports/PBNO_M4_RISK_RESOLUTION_ZH.md)：当前最强正结果；
   spectral dictionary、第四物理域、noise、hard-router 边界与统一连续 prior adapter；
5. [`CURRENT_STATUS_ZH.md`](CURRENT_STATUS_ZH.md)：一页式状态、核心结果、负结果与决策 gate；
6. [`PBNO_METHOD_UNIFICATION_NEXT_ZH.md`](PBNO_METHOD_UNIFICATION_NEXT_ZH.md)：统一 latent operator 的
   设计来源；M0 已执行，后续 M1--M3 因 gate 未通过而冻结；
7. [`PBNO_M4_RISK_RESOLUTION_PLAN_ZH.md`](PBNO_M4_RISK_RESOLUTION_PLAN_ZH.md)：本轮预注册目标、执行顺序与
   实际 gate 结论；
8. [`../reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md`](../reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md)：
   PBNO 方法主体；stable posterior tokenizer、neural transport、posterior-kernel operator、三域 strict-Q3、
   OOD、matched CNP、compute 与论文边界；
9. [`PBNO_LONG_TERM_PLAN_ZH.md`](PBNO_LONG_TERM_PLAN_ZH.md)：PBNO 公式、阶段顺序、硬 gates 与停止规则；

## 按需背景与历史证据

- [`EXPERIMENT_LEDGER_ZH.md`](EXPERIMENT_LEDGER_ZH.md)：所有路线的证据账本、禁重复实验、重启条件和
  未充分开发的正信号；新实验立项前必须检索对应条目。
- [`CROSS_DOMAIN_DATA_PROVENANCE_ZH.md`](CROSS_DOMAIN_DATA_PROVENANCE_ZH.md)：OpenFWI、Deepwave、Ocean、
  The Well 的生成/处理、split、hash 与复用限制。
- [`WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md`](WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md)：完整论文 framing、
  方法层次与 flow 进入条件。
- [`../reports/PBNO_JOINT_QUERY_M1_ZH.md`](../reports/PBNO_JOINT_QUERY_M1_ZH.md) 与
  [`../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md`](../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md)：
  strict Q3 与解析 prior 的早期证据。
- [`../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md`](../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md)、
  [`CROSS_DOMAIN_RESPONSE_QUERY_24H_PLAN_ZH.md`](CROSS_DOMAIN_RESPONSE_QUERY_24H_PLAN_ZH.md) 与
  [`CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md`](CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md)：跨域 existence POC、
  provenance 问题与冻结指标。
- [`IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md`](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md) 与
  [`../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md`](../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md)：
  早期 response-query protocol 和 reciprocity Transformer。
- [`../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md`](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md) 与
  [`MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md`](MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md)：
  多分支相空间 oracle、fixed-token 负结果与重启条件。
- [`../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md`](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)、
  [`../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md`](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md) 与
  [`WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md`](WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md)：
  旧波场表征 POC，仅作机制和负结果参考。

## 科学背景

- [`SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md`](SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)：
  面向 AI 读者的波动、信号处理、可预测性与可辨识性备忘录；
- [`REPRODUCIBILITY_ZH.md`](REPRODUCIBILITY_ZH.md)：所有阶段的命令、数据路径和 hash，仅在需要复核时查阅。

## 历史归档

其余文档属于旧的 spectral continuation、few-shot calibration、local token 或 reliable/actionable
horizon 阶段。不要按文件名逐份阅读；先看 [`archive/INDEX_ZH.md`](archive/INDEX_ZH.md) 中压缩后的
主要正负结论，再按需进入原始协议或报告。
