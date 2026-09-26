# Research handoff for future agents

本文件约束本仓库后续研究 Agent。开始编码或训练前，必须先读：

1. `README.md`；
2. `docs/CURRENT_STATUS_ZH.md`；
3. `docs/EXPERIMENT_LEDGER_ZH.md`；
4. 当前任务若涉及新模型，再读 `docs/WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md`。
5. 当前任务若涉及相空间、多路径或 wave token，再读
   `docs/MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md`。
6. 上述任务必须再读结果归档 `reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md`，不能只按执行前 brief 重跑。
7. 当前任何新实验先读 `docs/IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md`；这是 2026-09-24 起的 active
   roadmap，旧表征计划仅作证据档案。
8. Response Query 相关任务必须读 `reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md`，其中包含已通过的 Q2、
   未通过的 Q3、三 seed 结果和禁止重复的 backbone screens。
9. 跨数据集 Response Query 新实验必须读 `reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md`、
   `docs/CROSS_DOMAIN_DATA_PROVENANCE_ZH.md` 与 `docs/CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md`；当前
   response subspace 已是必须击败的 strong baseline，不能再只与 local interpolation 比较。
10. 当前 active 方法计划是 `docs/WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md`。任何 functional tensor、FNO、
    DeepONet 或 flow 实验必须遵循其中 M0--M4 顺序；flow 未通过 latent ambiguity gate 前不得启动。
11. PBNO 后续任务必须先读 `reports/PBNO_M4_RISK_RESOLUTION_ZH.md`；这是当前最新正式结果。M3/M2/M1
    报告只用于方法细节与历史归因。
12. 第一次理解 PBNO 可先读 `docs/PBNO_STORY_PLAIN_LANGUAGE_ZH.md`。若任务涉及下一代统一架构，再读
    `docs/PBNO_METHOD_UNIFICATION_NEXT_ZH.md`；该文档当前是待验证设计，未经用户确认不得把 M0--M3
    全部直接实现或写成已有贡献。

不要通过从文件名猜测进度来选择实验。`docs/`、`reports/` 和 `src/` 中有大量已结束阶段，完整技术路径
保留是为了复现，不表示它们仍是 active roadmap。

## 当前主线

研究问题已推进为 In-Context Wave Response Query。RQ-1 已证明 co-located multistatic array 上的 Q2
held-out-source query：calibrated reciprocity routing + 121,938 参数 residual Transformer 在三 seed IID test
K=8 得到 0.498 vs 0.624，ellipse OOD 得到 0.427 vs 0.558。Q3 joint source-frequency query 仍未通过，
不能把 Q2 结果写成 joint-query 成功。

新的三 seed set-operator 结果：full-context RelL2 为 0.0533，raw 为 0.0593；每源只有 1--2 个频率时
相对 raw 降低 24--30%，20 dB 为 0.112 vs raw 0.529。10 dB 仍失败；clean ellipse 与 128-grid boundary
上 raw 更强。instance-contrastive 预训练冻结 probe 约 0.31，且不改善 full-label 最优点，不是当前贡献。

2026-09-23 的公开 WaveBench 独立验证使用 8,000/500/500 system split 和三 seed。phase-coordinate 在
1--4 frequency budgets 相对 raw 分别降低 7.6%/11.6%/12.7%/12.5%，所有 seed 同向；full-context
paired bootstrap 95% 区间为 12.10%--12.93%。该结果确立跨数据集实用性，但未过预注册 15% 强效应
gate。两个更复杂的 fixed complex-edge token 都更差，不再枚举手工 normalization。

这些结果支持 measurement-efficiency representation，不证明 unseen-query complex field generation 已解决。

2026-09-23 的多分支相空间 POC 已完成。wavelength-scaled K=4 oracle 在 partial/closed 相对 K=1 改善
26.3%/25.7%，证明局部多分量 headroom；fixed two-scale packet Gram 的三 seed 64→128 downstream 改善
只有 11.1%/11.7%，未过 15% promotion gate。所有 30 个 boundary systems 同向，且 synthetic 单频预算
改善 12%--18%，所以记为机制级正结果，不是主方法。不要用单 seed 或 ratio-of-means 改写 gate。

2026-09-25 PBNO M2 已收口。coordinate-RMS residual likelihood 修复了 Deepwave 的数值爆炸，但稀疏预算
receiver residual K=1 仍为 0.11665，未过 0.1138 stop line；不要继续扫 bound、卷积层或 set encoder。
oracle rank-128 主坐标预测 PCA 尾模态在 OpenFWI/Deepwave validation/test 都退化，不启动 nonlinear-tail
decoder。Q2 efficiency 替代 gate 已通过：OpenFWI rank 96 为 0.12880 vs rank 128 的 0.12922，Deepwave
rank 112 为 0.04394 vs 0.04349；对应 rank-cubic solve 减少 57.8%/33.0%。posterior-latent memory 只在
OpenFWI Q3 有正信号，Deepwave 无效，当前不是跨域主方法。先读
`reports/PBNO_M2_EFFICIENCY_AND_BOUNDARIES_ZH.md`。

2026-09-25 PBNO M3 已覆盖 M2 的“memory 单域”判断。稳定 complex128 posterior 修复 Ocean 病态 batch solve
后，posterior-kernel attention 在 strict Q3、K=2 的 OpenFWI/Ocean 分别达到 0.14707/0.48723，相对
validation-selected strongest baseline 改善 13.56%/19.92%，system win rate 93.6%/77.3%；两域 10%
主 gate 已通过。Deepwave 继续由 neural latent transport 处理，K=2 为 0.09077，改善 6.85%。完整方法、
三 seed、OOD、CNP、延迟与边界见 `reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md`。今后不得引用旧
complex64 Ocean Q3 数字，也不得再写“posterior memory 只有 OpenFWI 单域有效”。

PBNO latent-belief best-of-8 gate 也已完成：相对 strongest deployable deterministic predictor，OpenFWI
validation/test gap 只有 8.08%/7.52%，Deepwave 只有 0.78%/0.48%，均低于 15%。不要启动 conditional
flow；重启条件必须是新的信息条件或数据域产生超过 15% 的 conditional oracle gap。

2026-09-26 PBNO M4 已完成。rank-16 spectral operator dictionary 把 OpenFWI/Ocean 的完整 response value
bank 压到原始 0.65%/0.93%，各预算 test 退化不超过 1.48%/1.19%；key 仍必须由实际 prompt mask 重算
posterior，不能退回已失败的 unconditional full-latent key。新的 structural lattice strict-Q3 域在 K=1/2
相对 global strong baseline 改善 10.31%/8.18%，成为第四域正证据。prompt-only hard LOO 与小型 shared
risk head 均未稳定通过全域 2% gate；但一个 1,665 参数连续 prior adapter 在三组独立 prompt seeds、三个域、
K=1/2/4/8 的 12/12 设置中均优于最佳单分支，OpenFWI/Ocean K=2 相对 global 改善 17.25%/21.85%。
不要继续扫 hard-router threshold；不要把三域共享写成 unseen-domain zero-shot。详见
`reports/PBNO_M4_RISK_RESOLUTION_ZH.md`。

## 当前执行顺序

除非用户明确改变方向，否则按以下顺序推进，不能把组件一次堆满：

1. 冻结三个跨域 system split、target/context episodes 和归一化预算轴，不再按 test 调参；
2. 保留 M3 Bayesian tokenizer + neural transport/posterior kernel 作为当前 strongest method；
3. memory value compression 已完成；下一步不要再删 prototype instances，而应保留 rank-16 spectral
   dictionary。统一 prior adapter 使用连续 kernel--neural residual composition；hard LOO/risk selection 是
   NO-GO，不得把二者混淆或继续扫 threshold；
4. 所有新方法必须同时击败当前 response subspace 与 validation 选出的最强同预算 baseline；
5. 第四个 structural-wave synthetic 域已通过；新数据的唯一高优先级是带可信 metadata 的真实/实验
   multistatic response，不再增加第五个同质 synthetic simulator；
6. conditional flow 只有新的 latent sample oracle gap 超过 15% 才允许训练；
7. active、3D 与 variable receiver 只按各自前置 gates 重启；continuous receiver 当前由冻结 functional head
   路由，不能写成 posterior kernel 原生支持。

Q2 的 best-next-source oracle 已审计：相对 nearest-first 的 error-AUC 只改善 6.0%，未过 15% gate。不要在
当前 Q2 setting 上训练 active policy 或 RL；只有 frequency/receiver action 或 task utility 改变后才重启。

多分支 oracle 与 fixed POC 已完成；不再枚举固定方向、半径、bandwidth 或 Gram edge。它现在是 Response
Query 的后置机制支线，不是当前第一阶段组件。

`docs/EXPERIMENT_LEDGER_ZH.md` 中的 underexplored signals 是候选支线，不得全部并入第一阶段。

## 禁止无条件重复的路线

- 在相同 74--100 Hz 信息条件下继续增加 pole 数、局部曲率、Neural ODE 或 hypernetwork 容量；
- 把若干相距很远的低频 raw fields 直接拼接，期待模型自动解开 phase alias；
- 把极低的 patch-span oracle 当成可部署方法；oracle 使用了 query truth 来拟合系数；
- 在相同 90--106 Hz 协议上扩大 free local coefficients、frequency-mask loss 或 Transformer；
- 为 set encoder 继续扫 moment、更多 slot、slot-stack、metadata FiLM、coordinate basis 或 masked-view
  prediction；这些 matched POC 均未改善冻结 transfer；
- 把当前 instance-contrastive checkpoint 当成核心贡献；从头监督训练最终更好（0.0491 vs 0.0639）；
- 在当前 AR calibrator 上继续搜索随机 sensor masks、value head、RL policy 或重复 reset；
- 在 dense oracle 没有先通过时训练昂贵的 sparse multi-fidelity calibrator；
- 因使用 flow/diffusion 就声称任务多模态。必须先有同一条件对应多个真实可行 target 的证据；
- 用 amplitude、phase coherence 或挑选近频区间替代 raw complex RelL2 来制造主要性能结论。
- 在 MPS-1 上继续扫描 fixed packet radii、orientations、bandwidths、Gram edges 或从头训练 69-channel
  前端；oracle 成功不等于该 fixed implementation 成功。

上述路线只有在实验账本写明的“重启条件”满足时才能重新打开。

## 不要过度解释已有失败

- 普通 U-Net 使用 oracle medium 仍失败，只否定当前数据规模和 image decoder；不否定 differentiable
  solver、microlocal decoder 或其他 wave-aware propagator。
- 少源 Nyström 约 0.99，只否定当前预算下的全局 column completion；不否定局部或层次 Green 表示。
- raw budget 曲线随 source 数增加而变差，主要受固定 channel concatenation 混淆；不能写成“更多
  观测有害”。
- ellipse OOD 仍属于同一 simulator，不是跨 PDE、跨 solver 或真实数据泛化。
- reliable-horizon 的高 AUROC 证明 failure 可预测，不证明既定 calibration action 可修复 failure。

## 每个新实验的最低流程

1. 在配置或实验说明中写出单一 hypothesis、matched baseline、主指标和 stop rule；
2. 先做 information/oracle gate，再训练复杂模型；
3. 按 physical system 切分，不能把同一 Green operator 的 source/frequency views 跨 split；
4. 主指标使用逐系统等权 raw complex relative L2；representation probe 可额外报告 retrieval、IoU、
   correlation，但不能代替最终 query error；
5. 同时报告 amplitude 与 phase/coherence，用于解释而非改写主结论；
6. 对噪声、网格变化和 boundary 分开报告，禁止只汇总平均值；
7. 正式结果至少三 seed，并保留 paired environment statistics；
8. 结果完成后更新 `docs/EXPERIMENT_LEDGER_ZH.md`，包括失败归因、重启条件和 artifact 路径。

## 证据语言

- `成立`：matched、multi-seed、system-held-out 证据支持；
- `机制级正结果`：方向可信，但没有通过预注册方法 gate；
- `oracle/existence result`：仅说明表示或信息上限存在；
- `development result`：用于选方法，不得写成 untouched test；
- `NO-GO`：只对报告中冻结的信息、任务和方法族生效，不自动外推到所有变体。

## 文件与版本规则

- 当前 active 代码清单见 `src/spectral_continuation/README.md`；其余模块默认冻结；
- 不回写产生过正式结果的 dated config；为新实验新建配置；
- 大 checkpoint 和逐样本数组不提交 Git；提交配置、hash、summary JSON/CSV 和正文图；
- 不移动历史模块或 artifact 路径，旧 manifest 与报告依赖这些路径；
- 不把新的 exploratory run 混入 `artifacts/wave_operator_representation/summary/`；只有冻结汇总才能进入。
