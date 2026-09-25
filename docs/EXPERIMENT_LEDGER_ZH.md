# 实验账本：有效事实、失败边界与可重启条件

> 更新日期：2026-09-23
> 目的：让新的研究者或 Agent 在十分钟内知道哪些问题已经做过、结果究竟否定了什么，以及哪些正信号
> 仍未充分开发。本文是决策账本，不替代原始报告和机器可读 artifact。

## 1. 如何读取证据

本仓库的结果分为四级：

| 级别 | 含义 | 可以如何使用 |
|---|---|---|
| A | system-held-out、matched baseline、三 seed 或强 paired 统计 | 当前主要事实 |
| B | development set 上的多 seed / 多环境稳定结果 | 决定是否继续，不作最终泛化 claim |
| C | 单 seed、小 pilot 或协议探索 | 生成假说，不作为方法贡献 |
| O | 使用 query truth 的 oracle / existence probe | 判断信息或表示上限，不可当部署结果 |

“NO-GO”始终带作用域。例如，free local coefficients 的 NO-GO 不等于 local representation 的
NO-GO；普通 U-Net decoder 的失败也不等于波动物理 decoder 不可能。

## 2. 当前事实总表

| ID | 结论 | 证据 | 状态 |
|---|---|---|---|
| F1 | coarse multi-frequency raw concatenation 不会自然带来跨频收益 | WaveBench 与 Ocean 单频/多频 matched comparison | 冻结负结果 |
| F2 | 频率采样必须相对传播时延足够密，才能辨识 phase slope | 79/80 Hz carrier 比 3 Hz 间隔显著更好 | 有效设计原则 |
| F3 | 结构化稳定复谱状态比 matched image U-Net 更适合频率查询 | Ocean 五频 macro 下降 34.5%，共同相位等变误差约 6.5e-5 | B 级正结果 |
| F4 | 增加生成器容量不能突破长 horizon phase identifiability | pole、ODE、curve、hyper variants 均失败 | 冻结负结果 |
| F5 | 局部低频 span 能表示目标，但自由系数不能由当前低频条件稳定确定 | 110 Hz patch oracle 0.052，learned local span 0.6186 | O 正 / 方法负 |
| F6 | 稀疏宽带 calibration 含实例状态信号，但当前方法只关闭约一半 oracle gap | near-band 0.4106→0.3611，12.05% | 机制级正结果 |
| F7 | failure risk 可跨实例、频率和空间稳定预测 | fixed-frequency AUROC 0.906/0.960；spatial AUROC >0.92 | 强正结果 |
| F8 | failure predictability 不等于 action controllability | repeated reset 无增益；随机 mask oracle headroom 仅 6.42% | 冻结负结果 |
| F9 | 物理测度决定 Green reciprocity | 普通 metric error 0.84，加权后 2.1e-6 | 强诊断，未深挖 |
| F10 | amplitude + local spatial phase advance 是当前最强 system representation | retrieval 100%；IID 0.0504；ellipse 0.0955 | A 级主结果 |
| F11 | 当前 representation 尚未解决完整 complex response query | raw/carrier/oracle-medium decoder 约 0.93--0.97 | 架构级失败 |
| F12 | shared slot set operator 将 phase token 的优势定位到低频率预算与 20 dB | 1--2 frequency 相对 raw 降 24--30%；20 dB 0.112 vs 0.529 | A 级主结果 |
| F13 | phase token 不是 universal clean-OOD winner | ellipse 0.100 vs raw 0.073；128-grid overall 0.394 vs 0.288 | A 级边界 |
| F14 | instance-contrastive 预训练不改善 full-label 上限 | pretrained 0.0639；scratch 0.0491；frozen 约 0.31 | 冻结负结果 |
| F15 | phase-coordinate 的观测效率在公开 WaveBench 上可复现，但效应为中等 | 1--4 frequencies 相对 raw 降 7.6%--12.7%；三 seed 全同向 | A 级正结果，15% gate 未过 |
| F16 | 更复杂的 fixed complex-edge normalization 不自动带来更好表示 | symmetric/directed edge 在全部 budgets 弱于 soft phase token | development 负结果 |

## 3. 路线级复盘

### E01：APEX-H 单频谐波迁移

- **原假设**：carrier-relative / harmonic prior 能提高低频到高频迁移的样本效率。
- **可信正结果**：K=1 时 WaveBench 与 Ocean 分别比 direct 降低 7.48% 和 9.41%，样本级方向稳定。
- **决定性边界**：标签增多后优势消失；fully supervised operator 没有稳定更强；flow matching 的必要性
  没有证据。
- **正确归因**：这是低数据区间的 inductive bias，不是完整的通用频率算子。
- **不要重复**：更深网络、等价 loss、local adapter、selective gate 已无稳定收益。
- **仅在何时重启**：新的表示在 zero/few-shot 与充分监督之间显示明确、可解释的 scaling law。
- **证据**：[APEX 冻结证据](APEX_LEGACY_EVIDENCE_ZH.md)。

### E02：粗频带直接拼接

- **原假设**：增加多个低频场自然改善高频预测。
- **结果**：WaveBench 10/15/20→40 与 Ocean 50/100→200 均不优于强单频 direct；粗频 carrier 更差。
- **归因**：相邻频率间隔超过由传播时延决定的 phase-alias threshold，更多 aliased phase 不是更多可用信息。
- **不要重复**：在没有先做 delay / alias audit 时任意选几个频率拼接。
- **重启条件**：频率采样依据最大有效时延设计，或输入表示显式处理多分支 phase ambiguity。
- **证据**：[Stage-1 报告](../reports/STAGE1_REPORT_ZH.md)。

### E03：细频带 phase carrier

- **原假设**：相邻频率复相关能估计可迁移 phase slope。
- **正结果**：无训练 79/80→90 Hz 相对单频 carrier 降低 57.1%；known-medium 冻结模型在新生成的
  105/110 Hz 上相对 direct 降低 59.0%/50.7%；125 Hz 仍降低约 30%。
- **负结果**：训练区间内不如 direct，150 Hz 后 raw error 超过 1，收益随 horizon 衰减。
- **保留结论**：局部 spectral phase advance 是有效信息坐标，但只是 local continuation law。
- **尚未充分开发**：它尚未与当前 spatial phase-advance token 在同一个 set encoder 中统一比较。
- **证据**：[Stage-1 报告](../reports/STAGE1_REPORT_ZH.md)。

### E04：稳定复谱状态与连续 generator

- **原假设**：显式 gauge invariance、稳定 propagation state 比 image bottleneck 更适合 oscillatory fields。
- **正结果**：matched Ocean 105--125 Hz macro 从 image U-Net 1.184 降至 0.776；解析三路径 Prony 在
  20--24 Hz 达到 0.043 且跨分辨率保持。
- **后续最好结果**：校准协议下 hyper-pole 的 110 Hz 为 0.4473，优于 recurrence 0.4741 和 query
  U-Net 0.6125。
- **决定性边界**：113 Hz 左右与 recurrence 交叉；107--125 macro 反而更差。增加 pole、局部曲率、
  affine components 或 Neural ODE 都没有突破。
- **不要重复**：保持同一观测带却继续扫容量或积分器。
- **重启条件**：新增可辨识信息，例如更合适的频率采样、不同 excitation、部分 query observations，
  或明确的 multi-branch state supervision。
- **证据**：[振荡场 POC](OSCILLATORY_REPRESENTATION_POC_ZH.md)、
  [Generator gate](../reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md)。

### E05：全局低秩、局部 span 与 masked frequency

- **原假设**：局部低频 basis 可表示高频，masked learning 可预测其系数。
- **oracle 正结果**：Ocean 110 Hz 的 8×16 raw patch-span oracle 为 0.052；三路径场为 0.0015。gauged
  带内 masked-frequency 解析恢复可到 0.089。
- **方法负结果**：learned free local-span 为 0.6186，frequency-mask 辅助变差到 0.6479；受限 local
  carrier token 只带来 0.97% 改善。
- **正确归因**：表示空间存在，但 query 对应的坐标不可由当前条件唯一确定；masked interpolation 不等于
  extrapolation。
- **不要重复**：更细 patch、更宽 token、相同协议上的 Transformer 或更多 mask loss。
- **重启条件**：context 新增能区分系统/路径的 latent，或训练分布提供同一局部 basis 的可辨识动力学。
- **证据**：[全局与局部诊断](../reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)、
  [Local-token 结果](../reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md)。

### E06：稀疏宽带 spectral calibration

- **原假设**：固定稀疏阵列的短 sweep 能校准实例相关 propagation state，并改善完全无观测的后续频率。
- **正结果**：每频率 0.5% 传感器时，107--112 Hz 从 0.4106 降至 0.3611；shuffled-instance 恶化；
  收益主要来自 phase。
- **负结果**：只改善 12.05%，110 Hz 为 0.4044，改善环境比例 62.5%，均未过 gate；K>184 饱和。
- **信息审计**：两频 dense oracle 上限仅 15.8%；六频 dense oracle 可改善 24.4%。
- **不要重复**：两频自由 phase correction、继续增加随机传感器、直接训练多 pole sparse model。
- **重启条件**：先证明新观测设计的 dense oracle 有至少 30% headroom，且跨足够多系统稳定。
- **证据**：[Sparse calibration 报告](../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)。

### E07：独立 Ocean low/high fidelity

- **原假设**：8-mode cheap solver + HF calibration 可以形成真实 multi-fidelity task。
- **工程正结果**：LF/HF 独立求解与 lineage 成立；适用环境中约 1.8--2.1 倍加速。
- **任务负结果**：仅含两个 validation environments 的 tiny pilot 中，dense HF calibration 对
  107--112 Hz 只改善 15.69%；107 Hz 单步改善 45.88%，到 112 Hz 已为负。这是停止昂贵扩展的
  pilot gate，不是总体效应量估计。
- **归因**：固定 8-mode 在浅水可能等于 HF，在深水又删去过多独立模态；fidelity knob 不连续。
- **不要重复**：直接把旧 sparse calibrator 迁移到这组 LF/HF 数据。
- **重启条件**：连续、跨环境稳定的 fidelity knob，或改成 receding-horizon tracking，并先过 dense oracle。
- **证据**：[Paired-fidelity gate](../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)。

### E08：Reliable-horizon map

- **原假设**：不看 target truth，也能预测延拓何时、何处失效。
- **强正结果**：Ocean/Synthetic fixed-frequency AUROC 为 0.906/0.960；4×8 同场 spatial AUROC 为
  0.945/0.921；不同 patch 尺度和 reset 后均保持。
- **边界**：这是 error ranking / selective prediction，不等于 error reduction。
- **不要重复**：继续堆更复杂 risk head；简单 disagreement 已解释大部分收益。
- **尚未充分开发**：作为当前 wave-system encoder 的 confidence/readout，检验表征是否知道自己的局部
  phase token 在何处失效。
- **证据**：[Reliable-horizon 最终报告](../reports/RELIABLE_HORIZON_MAP_FINAL_ZH.md)。

### E09：Actionable horizon 与随机 sensor masks

- **原假设**：学习 action value 可以选择更好的 calibration endpoint 和 sensor mask。
- **机制正结果**：within-environment utility Spearman 0.624，shuffled-instance 显著更差。
- **决定性负结果**：learned value 0.5846 差于 risk-trigger 0.5720；三个随机 masks 的 oracle 相对最佳
  fixed 只有 6.42% headroom，新增 masks 仅改善 0.48%。
- **归因**：当前 actions 的信息子空间几乎相同，问题不在 selector capacity。
- **不要重复**：更多随机 masks、Transformer value head、RL、重复同类 reset。
- **重启条件**：actions 必须改变信息类型，例如 excitation、频率窗口、观测模态或几何，并先过 oracle gate。
- **证据**：[Actionable-horizon 报告](../reports/ACTIONABLE_HORIZON_POC_ZH.md)。

### E10：Wave-system representation

- **原假设**：amplitude 与固定物理间距的 local phase advance 比 raw complex pixels 更能表示隐藏系统。
- **主正结果**：retrieval 100%；IID medium 0.0504；ellipse OOD 0.0955；均显著优于 raw 和
  amplitude-only。
- **关键消融**：phase-only 0.108，amplitude-only 约 0.238，组合 0.050；加入绝对 normalized phase
  反而退化到 0.064。
- **边界**：10 dB 失败；partial/closed boundary 为 0.614/0.637；输入仍是 dense internal fields；
  retrieval 可能已饱和，不能单独承担后续方法比较。
- **架构负结果**：普通 missing-block decoder 即使输入 true medium 也约 0.965。这只否定当前 image
  decoder，不是信息论 no-go。
- **当前状态**：唯一 active 主线。
- **证据**：[完整 POC](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

### E11：Green reciprocity 与物理 metric

- **诊断**：普通像素内积下 reciprocity error 约 0.84；按 source 处的 `c(s)^2` 加权后为
  `2.1e-6`。只用低频 cross responses 可把相对权重估到约 `5e-7` log RMSE。
- **意义**：响应算子的对称性依赖物理测度；默认欧氏 token aggregation 可能把可恢复结构隐藏掉。
- **尚未做**：没有把 learned measure 用于 encoder normalization、reciprocity pretraining 或
  source-selection generalization。
- **注意**：它没有救活少源 Nyström，不能写成 Green completion 已解决。
- **证据**：[Wave-system POC，第 5.2 节](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md#52-一个额外的物理诊断互易性依赖正确-metric)。

### E12：Variable-budget wave set operator

- **假设**：物理 phase token 能在参数量固定的 shared set operator 中提高观测效率，而不是只帮助固定通道
  U-Net。
- **正式结果**：24-view 为 0.0533，raw 为 0.0593；每源 1--2 frequencies 时相对 raw 稳定降低
  24--30%；20 dB 为 0.112，raw 为 0.529。
- **强负对照**：amplitude-only 为 0.1226，说明幅值稳定但不充分；strong raw 在信息充分时几乎追平。
- **边界**：10 dB 为 0.797；ellipse 与 128-grid partial/closed 均弱于 raw。
- **预训练归因**：instance contrastive 只加速早期训练，冻结可读性差，最终不如 scratch。
- **已停止架构**：moment、slot-stack、masked-view prediction、metadata FiLM、coordinate basis 与
  13-channel multi-scale token 均未过 frozen/cross-group gate。
- **正确 claim**：measurement efficiency 与 moderate-noise inductive bias，不是 universal OOD、foundation
  model 或 unseen-field generator。
- **重启条件**：新 acquisition 信息、可证明 identifiability 的自监督目标，或第二物理域上重复出现相同
  budget scaling。
- **证据**：[M0 报告](../reports/WAVE_SET_OPERATOR_M0_ZH.md)。

### E13：WaveBench 独立验证与 complex-connection 表示

- **假设**：amplitude + local phase advance 的优势来自一般的复传播坐标，而非原 simulator 或固定
  source layout；同一个表示应在公开独立数据上提高频率观测效率。
- **冻结协议**：WaveBench isotropic 10/15/20/40 Hz；按 50,000 个物理系统固定切分，8,000 / 500 /
  500 train/validation/test；同一 checkpoint 随机接受 1--4 个频率；raw、amplitude、phase-coordinate
  使用完全相同 encoder/decoder；三 seed，枚举每个预算的所有频率组合。
- **正式正结果**：1/2/3/4-frequency contrast RelL2 从 raw 的 0.3024/0.2422/0.2077/0.1811 降到
  0.2794/0.2140/0.1814/0.1584；相对降低 7.6%/11.6%/12.7%/12.5%。完整四频 paired bootstrap
  95% 区间为 12.10%--12.93%。
- **强负对照**：amplitude-only 为 0.3807/0.2891/0.2398/0.2088，说明稳定 envelope 不足以辨识
  system，local phase transport 提供独立信息。
- **未过 gate**：预注册要求至少两个 sparse budgets 平均改善 15%，本轮未达到；只能声称稳定中等效应，
  不能声称碾压 strong raw baseline。
- **方法负结果**：symmetric Hermitian edge 与 directed amplitude-transport edge 都略差于当前 soft
  phase token。不要继续枚举 fixed normalization 或把更多方向硬塞进 channel axis。
- **HarmoCore 归因**：Functional Tucker 的 continuous coordinate basis 和 3D scaling 值得继承；但
  per-frequency independent core 没有显式 system latent 或 frequency dynamics，不应用其 sparse
  reconstruction 成功替代跨频 system representation 问题。
- **重启/推进条件**：下一方法只测试 dimension-agnostic shared edge kernel；两个二维域至少一域稳定
  改善、另一域不退化，才进入 3D。3D 先做 system readout，不先做生成。
- **证据**：[WaveBench M1 报告](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)。

### E14：多分支相空间 oracle 与 fixed Gram POC

- **单一假设**：partial/closed boundary 的失败部分来自单个 phase advance 把同一局部邻域中的多个相干
  传播分支压成一个统计量。
- **oracle 正结果**：受控 2/3 分支混合在正确 K 下，held-out complex RelL2 从 0.664/0.772 降到
  0.313/0.326；约 1.5-wavelength 的真实场窗口中，K=4 相对 K=1 在 partial/closed 上改善
  26.3%/25.7%，95% 区间均严格为正。
- **关键尺度发现**：固定 33-pixel 窗口只改善约 6%；9/17/25/33 pixels 的增益单调减小。局部
  plane-wave sparsity 必须按 wavelength，而不是 raster pixels 定标。
- **方法设计**：8 orientations × 2 radial scales 的固定 complex packet bank；归一化 packet energy 加
  complex Gram edges。Gram quotient 一个共同 global phase，同时保留 branch-to-branch relative phase。
- **WaveBench 机制结果**：relative-phase Gram 的单 seed full-context paired 改善为 3.54%，优于逐 packet
  独立 phase quotient 的 2.27%；只是 development evidence。
- **三 seed正结果**：synthetic 的 1-frequency budgets 改善 12.1%--17.7%；full context 仅改善 1.9%。
  64→128 boundary shift 中 open/partial/closed 的逐系统改善为 5.1%/11.1%/11.7%，所有 30 个物理系统
  同向。
- **未过 gate**：partial/closed 均未达到预注册 15%。不能用某个 seed 的 15%--16% 或 oracle 的 26%
  替代正式方法结果；fixed multiscale Gram 不提升为主方法。
- **失败归因**：oracle 用目标 patch 自适应选 atoms 与系数；fixed filters 不适应局部 wavespeed/curved
  wavefront，69-channel raster 没有显式 branch routing，下游 loss 仅间接监督局部分解。
- **不要重复**：更多固定半径、方向、bandwidth、Gram edge 或从头训练大通道前端。
- **重启条件**：引入受 wavelength 约束的 learnable analysis、显式 sparse branch routing，或一个直接
  需要 multipath 的 query；仍需通过同一三 seed 15%/2% gate。
- **证据**：[MPS-1 完整报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)；机器汇总
  `artifacts/multibranch_phase_space_mps1/summary/summary.json`。

### E15：In-Context Wave Response Query（RQ-1 已完成）

- **核心问题**：从 unseen physical system 的少量 source--frequency complex receiver vectors 中，在不更新
  权重时回答 joint-held-out source--frequency query。
- **为什么现在推进**：hidden-medium recovery 已证明 responses 包含 system identity，但 dense missing-field
  generation 约 0.93--0.97；固定 receiver response query 将 system inference 与 dense oscillatory image
  decoding分开。
- **首轮剪枝**：receiver array 固定；Q3 joint query 为主；不加入 flow、RL、variable receivers、3D 或
  branch router。
- **先验审计**：conditional ambiguity、nearest query、coordinate interpolation、low-rank completion、
  context-conditioned retrieval、true-medium decoder、active oracle。
- **最小模型**：DeepSets/CNP；只有 global bottleneck 明确才做 query-to-context cross-attention。
- **主 gate**：Q3 相对最强非神经 baseline 改善至少 15%，K=1 到 K=8 稳定下降；所有统计以 physical
  system 为推断单位。
- **belief gate**：只有 context-near systems 对 target 出现分离 modes 且 deterministic mean 明显平均化，
  才训练 mixture/flow。
- **active gate**：best-next-query oracle 的 error-AUC 相对最佳 fixed schedule 至少改善 15%，且 uncertainty
  校准成立，才训练 policy；RL 最后进入。
- **协议**：[Response Query 主线](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)；配置
  `configs/in_context_response_query_rq0_20260924.json`。
- **Cartesian-array 负结果**：Q3 K=8 的 train mean/PCA 约 0.939；DeepSets 0.936、cross-attention
  0.927、spatial set operator 0.934、masked Transformer 最佳约 0.911。true-medium CNN/FNO decoder 最佳
  约 0.908/0.851。该 setting 未过 gate，不再只换 backbone。
- **任务修正**：采用 9 个 source/receiver co-located transducers，使 Green reciprocity 成为可用的信息路由；
  Q2 target source 不见、target frequency 已见。
- **最终方法**：log-gain-calibrated reciprocity routing + 2-layer、121,938 参数 masked residual
  Transformer；标准化训练后用 physical raw RelL2 小步微调。
- **IID 正结果**：三 seed test K=8 为 0.498 ± 0.014，calibrated reciprocity 为 0.624，相对提升
  20.1%，system bootstrap 95% 区间 [12.7%, 28.7%]。
- **geometry OOD**：ellipse K=8 为 0.427 ± 0.002 vs 0.558，相对提升 23.5%。
- **active oracle 负结果**：192 个 test episodes 上 best-next-source greedy oracle 的 error-AUC 为 0.603，
  nearest-first 为 0.642，只改善 6.0%，未过 15% gate；当前 Q2 不训练 policy/RL。
- **边界**：只证明 Q2 和 co-located array；Q3 joint query、任意 receivers、第二 PDE/真实数据仍未证明。
- **正式报告与配置**：[RQ-1 报告](../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md)；
  `configs/in_context_response_query_rq1_formal_20260924.json`。

### E16：Cross-Domain Response Query（M0 通过，M1 已完成首轮）

- **核心假设**：完整复响应表在跨环境方向存在低维结构；少量 source probes 可以在 inference time 识别
  当前环境的 response-subspace coordinate，再回答 held-out source。
- **统一接口**：`Y[system, source, frequency, receiver]`；同频 Q2；system-held-out split；raw complex RelL2；
  context source 与 target 严格不重叠。
- **数据域**：OpenFWI 1500 systems、Deepwave 1024 systems、Ocean 512 environments；The Well 只作 fixed
  public semantic audit。
- **M0 方法**：对每个频率在 train systems 上学习 complex SVD response basis；query time 用 context rows
  做 ridge completion。rank/ridge 仅由 validation 选择。
- **OpenFWI 正结果**：K=1/2/4 为 0.213/0.176/0.120；同预算 baseline 为 0.547；相对改善
  61.3%/68.5%/78.7%，逐系统胜率 99.0%/99.6%/100%。
- **Deepwave 正结果**：K=1/2/4/8/15 为 0.145/0.115/0.080/0.059/0.053；相对 baseline 改善
  24.6%--72.6%。OOD K=8 为 0.094。
- **Ocean 正结果**：互易误差为 0，但 anchor IDW 在 K=4/8 仍为 0.886/0.564；subspace 为
  0.0043/0.00053。参数 OOD K=4/8 为 0.325/0.199。
- **measurement efficiency**：half-budget 对 full-budget local baseline 的误差比，OpenFWI/Deepwave/Ocean
  分别为 0.322/0.302/0.0076。三个域都通过预注册 gate。
- **M1 方法**：M0 routed baseline + 1D receiver-local encoder + source-set attention + coordinate-aware、
  zero-initialized residual decoder，约 293k 参数。
- **M1 三 seed**：OpenFWI 相对 M0 再改善 2.4%--8.7%；Deepwave 只改善 0.04%--3.5%，且 K=15 的
  paired interval 跨零。因此不把网络 residual 写成主贡献，不继续无条件增加 Transformer/loss。
- **The Well 负发现**：HDF5 不含 source coordinates，场峰与离散 PDE residual 都把 26 trajectories 定位到
  同一 cell。当前 `source_receiver_index=-1`，不能把伪互易矩阵当成物理证据。
- **不要重复**：在同一 Q2 上继续堆大 backbone；只报告 Ocean IID 极小误差；把 reciprocal anchors 说成
  完整答案；用 The Well trajectory index 假装 source geometry。
- **下一步强对照**：参数匹配的 generic masked Transformer、context-conditioned system retrieval、direct
  supervised ridge；只有排除 weak-baseline 解释后，才升级为论文主方法。
- **证据**：[跨域 POC 报告](../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md)；
  [数据 provenance](CROSS_DOMAIN_DATA_PROVENANCE_ZH.md)；机器汇总
  `artifacts/cross_domain_response_query/*/audit_summary.json`。

### E17：Wave as Prompt latent-spectrum prior（M0 正式通过）

- **单一假设**：sparse prompt identification 的误差主要来自不同 system-latent directions 的可辨识性不均；
  train-system response spectrum 可以提供比 scalar ridge 更正确的方向相关先验。
- **方法**：保留同一 complex PCA response basis，把 $\lambda I$ 替换为由训练 latent variance 确定的
  diagonal Gaussian precision；闭式 MAP 更新，无 target leakage、无额外网络 backbone。
- **强基线修正**：baseline rank 搜索扩展到 128。Deepwave 原 rank-32 并非容量上限；rank-64 已把
  validation error 从约 0.0583 降至 0.0519。
- **三 seed half-budget**：OpenFWI 0.1292 vs 0.1745（25.9%）；Deepwave 0.04349 vs 0.05313
  （18.2%）；Ocean 0.001018 vs 0.001678（39.3%）。固定一次 validation selection 后跨三组
  prompt orders 评测，system win rate 为 98.8%/100%/100%。
- **probe saving**：OpenFWI K=3 达到 K=4 scalar；Ocean K=6 达到 K=8 scalar，均节省 25%；Deepwave
  没有达到该 gate。
- **OOD**：half-budget Deepwave/Ocean 相对 scalar 改善 16.1%/14.6%，但 Ocean 绝对 RelL2 仍为 1.032；
  Ocean K=1/K=8 分别明显退化，禁止 universal-OOD claim。
- **functional-rank 负结果**：独立 source--receiver CP 在三域失败；低存储 shared Tucker 只在 Ocean
  保真。receiver Fourier/DCT 即使 carrier demodulation 后也不能保留 OpenFWI/Deepwave query accuracy。
  正确低秩轴是 physical systems，不是振荡 source/receiver coordinates。
- **episodic basis 负结果**：Deepwave 无改善，OpenFWI 约 1.4%；不继续堆 dense basis loss。
- **flow 决策**：没有条件多模态/K-sample oracle 证据，暂不训练 conditional flow。
- **continuous query**：Ocean 全局 interleaved receiver holdout 为 0.007414 vs scalar+cubic 0.008275，
  改善 10.4%；Deepwave 为 1.3874 vs 1.3878，且 true-response interpolation oracle 已达 1.3868，归因为
  receiver undersampling/aliasing，不继续堆 coordinate MLP。
- **Deepwave sampling curve**：相同 systems/sources/frequencies 下，32/64/88 observed ring points 的 oracle
  为 1.387/0.476/0.407，operator 为 1.387/0.479/0.410；96-grid 离散圆在 256 nominal receivers 时发生
  cell duplication，因此冻结该 continuous 分支。
- **prompt calibration**：leave-one-prompt-source-out 二选一不读取 target；Ocean K=4 OOD 从 fixed prior
  1.032 降至 0.846，相对再改善 18.0%，IID 轻微回退 5.3%；K=8 OOD 仍退化 11.2%，只作 robustness
  extension。
- **继续条件**：prompt-only OOD calibration；Deepwave 只有在更密 receiver-grid sampling oracle 合格后才
  重启 neural functional decoder。
- **证据**：[M0 报告](../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md)；机器汇总
  `artifacts/wave_as_prompt_pfto/m0_latent_prior_multiseed/summary.json`。

## 4. 值得从正反馈继续挖的方向

以下按“与当前主线的连接强度 / 最小验证成本”排序。它们不是要求同时实现的组件清单。

### P0：多分支相空间 wave tokens（首轮已完成）

**当前状态。** wavelength-scaled oracle 已通过，fixed multiscale Gram 的三 seed downstream gate 未过。
见 E14。本条不再是可以原样执行的计划。

**若重启。** 只允许加入 learnable wavelength-constrained analysis、显式 sparse routing 或直接的
multipath query；不再枚举 fixed filter channels。

**进入下一阶段的门槛不变。** 在一个预先指定困难条件相对当前 token 降低至少 15% paired system
RelL2，另一二维域和 IID 不退化超过 2%。

**完整协议。** [多分支相空间波表征](MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md)。

### P1：统一 spatial 与 spectral phase advance

**为什么现在值得做。** 当前最强 token 使用同频空间邻点的复相关；旧 Stage-1 最强机制结果来自同点
相邻频率的复相关。两者本质上都在估计局部 complex phase transport，但过去被分在两条路线中。

**最小假设。** 对同一 source 的邻近频率 views，增加 spectral phase advance 与 amplitude ratio，可以
帮助 set encoder 区分 local wavevector、delay 和 boundary-induced multipath；尤其可能改善 closed
boundary 与低 source budget，而不要求解析 source location。

**最小 POC。** 先完成 shared set encoder；随后只增加 spectral increment channels，保持 backbone、
参数量、views 和训练预算一致。比较 amplitude-only、spatial-only、spectral-only、spatial+spectral。

**成功门槛。** 不以已饱和的 retrieval 为唯一指标；至少在 coefficient recovery、20 dB、closed
boundary 三项中的两项稳定优于 spatial-only，并在可变 frequency budget 下方向一致。

**为何不是重走旧路。** 旧 carrier 直接把单一 slope 外推成目标相位；这里把 phase increment 当作
system-identification evidence，由 set encoder 聚合，不强制单路径 continuation law。

### P2：学习物理测度，而不只是学习 feature

**为什么值得做。** reciprocity metric 从 0.84 到 `2.1e-6` 是本项目最干净、但尚未进入模型的物理
诊断之一；而且权重可仅由低频 response 自监督恢复。

**最小假设。** 让 encoder 先估计 response space 的正测度，再在该测度下做 view normalization、
reciprocity consistency 或 attention，可能提高跨 source layout 和跨 resolution 的泛化。

**最小 POC。** 不加 PDE loss；比较 Euclidean aggregation、oracle metric 和 learned metric 三者，先看
cross-layout retrieval 与 medium recovery。learned metric 必须只读 context views。

**成功门槛。** learned metric 接近 oracle，并在新 source layout 或 boundary 上产生至少 10% paired
error reduction；若只让 reciprocity residual 更小但 downstream 不变，则作为科学诊断保留，不进入主方法。

### P3：用 system latent 重新审视 local-span oracle gap

**为什么过去不该继续，现在可能不同。** 旧 free local-span predictor 失败，是因为低频条件没有提供足够
system identity；当前 token 却能以 0.0504 恢复 hidden medium。这个新 latent 可能提供当时缺失的全局
条件，使局部系数不再完全自由。

**前置 gate。** 必须先让 true-medium oracle decoder 低于 0.30。若真实介质都不能驱动合格 decoder，
learned latent 不值得接入。

**最小 POC。** 不预测 1408 个自由复系数。让 system latent 只调制低秩、多分支或局部 propagation
decoder，并与 true-medium、predicted-medium、unconstrained latent 三种条件 matched 比较。

**成功门槛。** predicted-system latent 相对 raw-context decoder 至少改善 30%，且能把 oracle-medium 与
learned-medium 的 gap 解释清楚。

### P4：把 Reliable-horizon 变成 representation confidence probe

**为什么值得保留。** 空间 disagreement 的 AUROC 在两个域和多个尺度上都很强，说明模型能识别局部
不可置信区域。这可以回答 phase token 在低 amplitude、多路径或 OOD boundary 下何时可靠。

**最小 POC。** 对 set encoder 做小 ensemble 或轻量多头，预测 medium error / query error 的 patch map；
比较 amplitude gate、token coherence 与 learned disagreement。

**限制。** 它是安全性与诊断支线，不应代替降低 field error 的主任务，也不应重新进入 calibration policy。

### P5：将独立多保真改成 tracking，而不是 open-loop extrapolation

**保留理由。** 真实 LF/HF gap 在第一步 107 Hz 有 45.88% oracle 改善，但六步 macro 失败。这是很清楚
的 short-memory system tracking 信号。

**为什么暂缓。** 它偏向部署与 sequential sensing，不如 wave-system representation 通用；当前不应与
主线混做。

**重启方式。** 使用连续 fidelity knob 和 receding-horizon observation，先比较固定成本 tracking oracle；
只有跨至少 30 个系统通过后才训练 representation model。

### P6：解析 Prony 只作为 identifiability unit test

合成三路径的 0.043 证明结构正确时 compact spectral state 可以非常准。它适合成为模型必须通过的
unit test：恢复阶数、跨分辨率与 phase equivariance。由于真实场逐像素 Prony 病态，它不适合作为论文
主数据或现实 baseline 的替代。

## 5. 推荐的研究优先级

### 立即推进

1. 冻结 E16 的三个多系统域 split、episodes 与 test，不再按 test 调参；
2. 补齐 retrieval、direct complex ridge 和参数匹配 generic query network，重新定义 strongest baseline；
3. 在共同的归一化 probe 预算上报告 raw error、strong-baseline gain、gap closure 与 budget AUC；
4. 只有至少两个域仍保留 10% 以上 strong-baseline headroom，才升级 query representation；
5. Q3 的 reciprocity routing + per-link spectral state 放在 Q2 强基线审计之后；
6. best-next-source oracle 已完成且未过 gate；active 只有扩展 action space 后才重启。

完整指标定义与 promotion gate 见
[跨数据集指标提案](CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md)。

当前方法升级冻结为 PFTO：先检查 functional tensor rank，再做 episodic differentiable latent fitting，最后才判断
是否需要 amortized encoder 或 conditional latent flow。详细 M0--M4 与 NO-GO 条件见
[Wave as Prompt 顶会计划](WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md)。

### 小成本并行诊断

- P2 learned physical metric 的 downstream usefulness；
- 把现有 reliable-horizon feature 作为 representation confidence 的只读 probe。

### 等前置条件后再做

- generative belief、active acquisition 与 variable receiver query；
- compact multibranch tokens，仅在 Q3/Q4 multipath failure 明确后；
- 第二 PDE / 外部数据域；
- 部分边界阵列与真实 source phase/gain mismatch。

### 暂停

- free local coefficients、generic masked Transformer、更多 pole/ODE/hypernetwork；
- 当前 AR action 上的 sensor-mask / RL 搜索；
- 没有重复条件证据的 flow/diffusion；
- 为了数字好看缩短 horizon、换主指标或只报告 amplitude。

## 6. 新实验完成后的更新模板

每次正式实验在本文件追加：

```text
ID / 日期 / commit:
单一假设:
数据身份与 system split:
允许读取的信息:
matched baseline:
预先 gate:
主结果与 paired uncertainty:
最强负对照:
结论作用域:
失败归因（信息 / 表示 / 优化 / 数据 / 指标）:
是否继续:
若停止，什么新条件下才可重启:
summary / figure / checkpoint hash:
```

没有填写“结论作用域”和“重启条件”的实验，不视为完成归档。
