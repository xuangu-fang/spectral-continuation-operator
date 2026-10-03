> 历史快照：2026-10-03连续相干表示主线更新前。不是当前执行计划；原文证据保留。

# Research handoff for future agents

> **2026-10-03 用户明确改变主线：** 当前任务是高精度复波场连续表征、相位保真与连续空间/频率查询。
> 先读 `docs/COHERENT_CONTINUOUS_REPRESENTATION_24H_ZH.md`。旧 PBNO/active acquisition 方法与结果
> 保留为历史，下面的旧“当前主线/执行顺序”不再是授权执行清单。用户允许先用已知环境、充分监督
> 排除表示/优化瓶颈，再做共享预测；不强制 sparse prompts，也不重启同信息下的盲远频率 pole/ODE。
> 新实验保留 raw complex 主指标、system splits 与全部证据边界。不要把单场拟合写成 operator 泛化。

本文件约束本仓库后续研究 Agent。开始编码或训练前，必须先读：

1. `README.md`；
2. `docs/CURRENT_STORY_AND_ROADMAP_ZH.md`；这是唯一的当前叙事与研究决策入口；
3. `docs/CURRENT_STATUS_ZH.md`；
4. `docs/EXPERIMENT_LEDGER_ZH.md`；
5. 当前任务若涉及新模型，再读 `docs/WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md`。
6. 当前任务若涉及相空间、多路径或 wave token，再读
   `docs/MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md`。
7. 上述任务必须再读结果归档 `reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md`，不能只按执行前 brief 重跑。
8. 涉及早期 Response Query 协议时，再读 `docs/IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md`；这是历史
   roadmap，旧表征计划仅作证据档案。
9. Response Query 相关任务必须读 `reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md`，其中包含已通过的 Q2、
   未通过的 Q3、三 seed 结果和禁止重复的 backbone screens。
10. 跨数据集 Response Query 新实验必须读 `reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md`、
   `docs/CROSS_DOMAIN_DATA_PROVENANCE_ZH.md` 与 `docs/CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md`；当前
   response subspace 已是必须击败的 strong baseline，不能再只与 local interpolation 比较。
11. 涉及早期 Wave as Prompt 方法审计时，再读 `docs/WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md`。任何
    functional tensor、FNO、DeepONet 或 flow 实验都必须先核对其中已执行的 M0--M4 gate；flow 未通过
    latent ambiguity gate 前不得启动。
12. PBNO 后续任务必须先读 `reports/PBNO_M4_RISK_RESOLUTION_ZH.md`；这是当前最新正式结果。M3/M2/M1
    报告只用于方法细节与历史归因。
13. 第一次理解 PBNO 可先读 `docs/PBNO_STORY_PLAIN_LANGUAGE_ZH.md`。若任务涉及下一代统一架构，再读
    `docs/PBNO_METHOD_UNIFICATION_NEXT_ZH.md`；该文档现在只保留设计来源，不能把未通过的 M1--M3 写成
    已有贡献或待自动执行清单。
14. Posterior composer M0 已执行，必须读 `reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md`。统一接口与严格
    adapter fallback 成立，但 learned correction 在 calibration 为正、unseen systems 明确退化；不得继续
    扫宽度、深度或辅助 loss，也不得启动 M1，除非报告中的重启条件满足。
15. 固定 PCA chart 的 heterogeneous functional replacement 已执行，必须读
    `reports/HETEROGENEOUS_FUNCTIONAL_OPERATOR_M0_ZH.md`。compressed oracle 有 16%--49% 空间，但
    functional evidence、reranking、listwise metric 与 nonlinear neural field 均未通过 unseen-system gate。
    不得继续扫 encoder/decoder 容量；必须区分 representation capacity 与 sparse identifiability。
16. uncertainty-conditioned proximal operator prior 已执行，必须读
    `reports/PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md`。它在 Deepwave 25% receiver masks 的正式增益为
    2.93%，OpenFWI/Ocean 均低于 0.6%，没有任何域通过 5% gate。不得继续扫 width、depth、denoising
    noise、steps 或 proximal strength，也不得从该 checkpoint 启动 query-decomposition M1。
17. 当前主线已推进到 target-aware wave interrogation。任何 active/query-selection 任务必须先读
    `docs/CURRENT_STORY_AND_ROADMAP_ZH.md` 与 `reports/ACTIVE_WAVE_INTERROGATION_M1_ZH.md`；M0 和旧 next
    plan 只用于历史核对。不得把当前方法写成 response-adaptive agent。
18. query-conditioned posterior metric 与 learned acquisition policy 已关闭；continuous acquisition utility
    operator 也已在 M1 执行，不再是待办。下一次 method upgrade 需要更密的连续 physics teacher，且必须保留
    analytic fallback、strict coordinate holdout 与 held-out-action regret gate。
19. M1 的 untouched IID 与 pair-action
    utility 通过；top-4 hybrid AUC gap 为 0.29%/0.16%。但 strict source/frequency-coordinate holdout regret
    为 10.23%/18.43%，5× speed 和 Ocean OOD 均失败。不得写成任意连续 action 泛化，也不得继续扫 utility
    width/depth；重启需要更密连续 physics teacher 或新外部 acquisition data。

不要通过从文件名猜测进度来选择实验。`docs/`、`reports/` 和 `src/` 中有大量已结束阶段，完整技术路径
保留是为了复现，不表示它们仍是 active roadmap。

## 当前主线

2026-09-28 M1 已覆盖 M0：全新 untouched IID systems 上 K=2 改善 Deepwave 27.27%、Ocean 20.36%；
七类强基线、K=1/2/4/8、成本轴和 truth oracle 已齐。当前方法接口冻结为 learned PBNO belief、continuous
query proposal 与 analytic top-4 certification。未见 action pairs 的 regret 小于 1.2%，但全新坐标外推失败。
后续不得按 pair holdout 结果宣称 arbitrary continuous source/frequency generalization。

2026-09-27 主线已从固定 prompts 下继续降低 decoder error 推进为 **Target-Aware Wave Interrogation**。
PBNO learned operator belief 的 posterior predictive risk 用于顺序选择 source--frequency actions；在相同预算、
相同第一条随机 probe 和完整 PBNO downstream 下，K=2 相对随机 schedule 改善 Deepwave 27.17%、Ocean
20.27%、OpenFWI 2.58%。K=1/2/4/8 AUC 改善 21.14%/12.35%/3.52%。这是当前优先复现和扩展的 POC。

该结果只支持 target-aware experimental design。training-population fixed schedule 与解析 planner 的差异
不足 0.6%，当前 response-conditioned action oracle gap 也低于 15%；没有新的 nonlinear/multimodal belief 或
action-dependent observation 条件时，不得启动 RL、Transformer policy 或宣称 active reasoning。下一步依次是
untouched systems、完整 acquisition baselines/成本轴、continuous utility operator 与外部 multistatic 数据。

2026-09-27 的 learned proximal prior M0 已关闭：复数相位等变 prior、解析 posterior precision projection 与
prompt safeguard 可以稳定产生小增益，但 3 域 × 4 receiver fractions 的最好结果只有 Deepwave 2.93%。
这说明 learning 放在 operator prior 上比任意 target correction 更安全，但当前 residual 的可辨识 effect size
不足。M4 仍是部署方法；新的 learning 分支只有在新信息条件、跨域对齐 chart 或 target-free 10% correction
oracle 出现后才可重启。

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

2026-09-26 posterior composer M0 已完成。121,603 参数共享 latent operator 将 posterior、spectral
dictionary、neural proposal 与 risk adapter 收进同一计算图，零修正严格复现当前强 adapter；但 learned
mode-wise correction 在 unseen Ocean systems 退化 3.7%--7.1%，OpenFWI/Deepwave 无收益。M0 性能 gate
失败，M1 与 flow 均保持关闭。只保留 zero-update composer 作为后续接口。

2026-09-26 heterogeneous functional operator M0 已完成。每个 prompt 独立 receiver mask 的 strict-Q3
headroom 在 10% receivers 下为 OpenFWI 16.2%、Deepwave 21.0%、Ocean 49.3%，但 functional evidence、
reranking、listwise metric 最多只兑现约 0.6%--2.4%。Ocean nonlinear neural field 的 unseen full-response
latent fit 为 0.368，sparse prompt encoder 仍约 1.035。不要把 target-truth candidate oracle 写成可约减
误差；不要继续扫任意 nonlinear latent 容量。线性 posterior chart 当前作为可观测坐标保留。

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

