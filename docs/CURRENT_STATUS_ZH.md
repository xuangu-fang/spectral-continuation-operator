# 当前状态：Wave-System Representation

> 更新日期：2026-09-27
> 用途：项目的一页式决策入口。数字细节见完整 POC 报告，旧路线见历史归档。

## 2026-09-27 最新边界：uncertainty-conditioned proximal prior 未通过

新的 29,761 参数 complex operator prior 只预测 latent correction，不直接读写 target response；correction
经 posterior precision 闭式投影，且有 prompt-consistency safeguard 与精确 M4 fallback。在 strict-Q3 K=2、
三个 test prompt seeds 和独立 10%/25%/50%/100% receiver masks 下，最佳正式结果为 Deepwave 25%：
0.11035 到 0.10712，改善 2.93%，bootstrap 区间 [2.63%, 3.23%]，系统胜率 89.5%。OpenFWI 各档约
0.14%，Ocean 最高 0.55%，完整 receivers 下均低于 0.2%。

shared 与 single-domain、latent denoising pretraining、proximal strength 0.1/1/10 均未产生 5% 结果。因此
M0 为 NO-GO，M1 query decomposition 保持关闭；不要继续扫 width、depth、noise scale、steps 或 proximal
strength。完整结果见
[Uncertainty-Conditioned Proximal Prior M0](../reports/PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md)。

## 2026-09-26 上一结论：M4 解决 storage、通过第四域与统一连续 adapter

### 后续 M0：heterogeneous functional representation 的 oracle 通过，方法 NO-GO

针对固定 PCA chart 的泛化性质疑，新增了每个 prompt 独立 receiver masks 的 strict-Q3 K=2 协议。
10% receivers 下，rank-16 compressed full-dictionary oracle 相对当前 posterior memory 的空间为 OpenFWI
16.2%、Deepwave 21.0%、Ocean 49.3%；因此 representation headroom gate 通过。

但 coordinate-aware additive evidence operator、functional reranker、listwise response metric 和 nonlinear
complex neural field 都未通过方法 gate。最好的 learned development 增益只有约 0.6%--2.4%；Ocean
neural field 虽可将 unseen full-response latent fit 到 0.368，K=2 sparse prompt encoder 仍约 1.035。
结论是 nonlinear manifold capacity 与 sparse conditional identifiability 必须分开；不要继续扫 encoder
宽度、层数、heads 或 latent dimension。当前保留线性 posterior chart 作为可观测坐标，把 learning 限制在
operator prior 与受控 chart transport。详见
[Heterogeneous Functional Operator M0](../reports/HETEROGENEOUS_FUNCTIONAL_OPERATOR_M0_ZH.md)。

### 后续 M0：统一架构成立，learned refinement 停止

M4 的 posterior、spectral dictionary、neural proposal 与 risk adapter 已进入一个 121,603 参数的共享
posterior operator composer。零修正时严格复现当前 adapter，三组 test prompt seeds 的 K=2 为 Deepwave
0.09220、OpenFWI 0.14668、Ocean 0.48116；这是当前可保留的方法接口。

learned mode-wise gate/residual 在 calibration selection 上产生 Ocean 8% 左右的开发增益，但在 unseen test
systems 上退化 3.7%--7.1%，OpenFWI/Deepwave 无实质收益。因此 M0 的性能提升 gate 未通过，M1 与 flow
均不启动。后续 Agent 不得按 selection 数字把 token correction 写成正结果，也不得继续扫 correction loss、
宽度或深度。详见 [M0 正式报告](../reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md)。

M4 的三个风险均得到可部署的正结果，同时保留一个明确的机制负结论：

- **storage 已解决**：rank-16 spectral value bank 将 OpenFWI/Ocean 完整 response-bank 压到
  0.65%/0.93%，各预算 test 退化不超过 1.48%/1.19%；计入 shared basis 后总存储仍减少约 79%/66%。
- **跨物理扩展通过**：独立 structural lattice solver 上，strict-Q3 K=1/2 相对 global strong baseline
  改善 10.31%/8.18%，paired bootstrap 下界 9.64%/7.51%。
- **统一 adapter 通过**：hard leave-one-prompt-out 与 shared risk head 都失败，但连续混合
  posterior-kernel/neural transport 的 1,665 参数共享网络通过。它不读 domain ID 或 test truth；三组独立
  prompt seeds 下，三个域、K=1/2/4/8 的 12/12 设置均优于最佳单分支。K=2 相对 global baseline 的改善为
  OpenFWI 17.25%、Deepwave 6.23%、Ocean 21.85%。hard selection 仍是 NO-GO，不再扫 threshold。

20/10 dB noisy prompts 下，OpenFWI 与 structural lattice 的 posterior kernel 相对 global 分别改善
16%--21% 与 9%--17%。Ocean 的 frozen latent gate 能阻止数值爆炸，但回退误差约 1，仍需 noise-aware
likelihood，不能写成 noisy Ocean 已解决。

当前论文主张应是 **Bayesian posterior token + learned/empirical operator priors + prompt-conditioned continuous
composition + spectral dictionary compression**。统一 composition 已在三个 calibration domains 成立，但不能
写成对未知 PDE/domain 的 zero-shot universal router。第四域仍是 synthetic，真实测量数据依然是主要外部
有效性风险。完整结果见 [PBNO M4 正式报告](../reports/PBNO_M4_RISK_RESOLUTION_ZH.md)。

方法统一候选已经按最小 M0 执行：解析 posterior token、rank-16 dictionary cross-attention 与 risk-conditioned
query composer 均已实现，当前 scalar adapter 也成为其严格退化情形；但 learned refinement 未过
unseen-system gate。因此不再把该备忘录当作待执行清单，设计来源见
[方法统一升级备忘录](PBNO_METHOD_UNIFICATION_NEXT_ZH.md)，执行结论以 M0 报告为准。

## 2026-09-25 上一结论：PBNO M3 已通过 strict-Q3 主 gate

严格 joint source--frequency query 禁止目标 source 和目标 frequency 出现在任何 prompt 中。M3 将稳定
Bayesian posterior 作为 in-context system tokenizer，再由 neural latent transport 或 posterior-kernel
attention 回答新 query；线性 PCA--MAP 是 identity transport、零 kernel blend 的严格特例。

三组 prompt-order seeds、system-held-out test 的 $K=2$ 主结果为：

| 数据域 | 最强 validation-selected baseline | PBNO | 相对降低 | 系统胜率 |
|---|---:|---:|---:|---:|
| OpenFWI | 0.17015 | **0.14707** | **13.56%** | 93.6% |
| Deepwave | 0.09744 | **0.09077** | **6.85%** | 99.2% |
| Ocean | 0.60843 | **0.48723** | **19.92%** | 77.3% |

OpenFWI/Ocean 的 paired bootstrap 95% 改善下界分别为 12.76%/15.87%，因此“两域超过 10%”的主 gate
通过。Deepwave 使用显式 posterior uncertainty 的 belief transport，在 geometry/contrast OOD、$K=8$
从 0.09664 降到 **0.09079**，三 seed 改善 **6.05%**；Ocean gated kernel branch 在各预算均不退化，
OOD gate 同时通过。

强 generic CNP 并非主分母：293,250 参数版本在 OpenFWI 从 checkpoint 继续训练、总计约 20k steps 后
$K=2$ 仍为 0.23247，弱于 global/retrieval 强基线。posterior kernel 的 A100 vectorized 完整评测延迟为
11.9--25.5 ms，约为线性 head 的 3.1--4.0 倍，response bank 为 3.38--20 MiB。预计算 full-latent memory
keys 未被任何 validation setting 选择，是明确负结果。

Q2 efficiency 替代 gate 仍由 rank 96/112 frequency-local heads 保留；Ocean continuous receiver 通过可观测
query-mask router 精确保留 0.007414，退化 0%。conditional best-of-8 gap 仍低于 15%，flow 保持关闭。

当前可以把工作定位为 **retrieval-augmented in-context operator identification** 的论文候选，但仍有三项风险：
两个 prior adapters 尚未统一；memory 比线性 head 慢约 3--4 倍；Ocean OOD 绝对误差仍接近 1。完整公式、
消融、图表和 claim boundary 见
[PBNO M3 正式报告](../reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md)。M1/M2 报告保留为历史归因，
不得再把其中“主 gate 未通过”当作当前状态。

## 2026-09-25 上一结论：Wave as Prompt M0 通过核心 gate

> 当前 active 方法升级已冻结为 PBNO：共享 system latent、query-conditioned complex functional basis 与
> 解析 Bayesian prompt posterior。论文主战场从已成熟的 Q2 推进到 target source/frequency 均未出现的严格
> Q3；阶段计划见 [PBNO 长程计划](PBNO_LONG_TERM_PLAN_ZH.md)。

在强化到 rank 128、独立 validation 选参的 scalar-ridge response-subspace baseline 上，训练系统的 latent
能谱先验仍在三域 half-budget 下稳定降低 raw complex RelL2：

| 数据域 | scalar ridge | latent spectrum prior | 相对降低 |
|---|---:|---:|---:|
| OpenFWI，K=2/4 | 0.1745 | **0.1292** | **25.9%** |
| Deepwave，K=8/15 | 0.05313 | **0.04349** | **18.2%** |
| Ocean，K=4/8 | 0.001678 | **0.001018** | **39.3%** |

这是固定一次 validation selection 后的三组 prompt-order seed、逐 physical-system paired 结果。OpenFWI 与
Ocean 用少 25% probes 达到 full-budget scalar baseline；Deepwave/Ocean half-budget OOD 分别提升
16.1%/14.6%。但 Ocean OOD 绝对误差仍为 1.032，且 K=1/8 会退化，说明 IID prior 仍会分布外过度自信。

functional-rank 审计同时否定了“source 与 receiver 轴都应低秩”的过强假设：真正稳定的低秩主要发生在
physical-system response manifold。下一步只推进 prompt-adaptive calibration 与 continuous receiver query；
在条件多模态证据出现前不加入 flow。完整方法、表格和负结果见
[M0 报告](../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md)。

continuous-query existence audit 已补齐：Ocean 交错 receiver holdout 为 0.007414 vs 0.008275，改善
10.4%；Deepwave 几乎无改善，因为给定真实 target response 的插值 oracle 本身已为 1.387。后者是采样
aliasing 的信息边界，不是继续增加网络容量的理由。

prompt-only leave-one-source-out calibration 在不读 target/OOD 标签时，把 Ocean K=4 OOD 从 fixed prior 的
1.032 降到 0.846（再改善 18.0%），代价是 IID 从 0.001018 轻微回退到 0.001072。它作为 robustness
extension 保留；K=8 OOD 仍未完全修复，不能声称 universal calibration。

Deepwave dense-grid sampling curve 已冻结：32/64/88 个 observed ring points 的 true-response interpolation
oracle 为 1.387/0.476/0.407，operator error 为 1.387/0.479/0.410，始终贴近信息上限。96-grid 离散圆不能
再无重复地扩展到 256 receivers；重启条件是更细 PDE grid 或排除近源奇异区域。

## 2026-09-24 当前结论：跨域 Response Query POC 已通过

统一 Q2 held-out-source query 已在 OpenFWI、Deepwave ring ultrasound、Ocean Pekeris 三个多系统域完成。
方法先从 train systems 学复 source--receiver response subspace，再用新系统的 K 条完整 probe responses
解一个小型复最小二乘问题，识别当前环境坐标并回答 target source。

| 数据域 | half / full K | local baseline | subspace query | half/full error ratio |
|---|---:|---:|---:|---:|
| OpenFWI | 2 / 4 | 0.547 | 0.176 | 0.322 |
| Deepwave | 8 / 15 | 0.195 | 0.059 | 0.302 |
| Ocean | 4 / 8 | 0.886 | 0.0043 | 0.0076 |

三个域都通过同预算 20%、bootstrap、75% 系统胜率、绝对误差与半预算 gate。OpenFWI/Deepwave 的互易
离散误差约为 `6e-7`，Ocean 为 0；但稀疏 reciprocal anchors 本身仍不足以重建振荡响应。正结果来自
**physics-routed observations + learned global response representation**。

M1 的 293k 参数 receiver-preserving residual operator 在三 seed 上：OpenFWI 相对 M0 再改善
2.4%--8.7%，Deepwave 只改善 0.04%--3.5%。因此当前核心结果是 response subspace 与任务 formulation，
不是更大 backbone。The Well 的 source metadata 无法从本地 HDF5 恢复，只作 public data audit。

完整结果见[跨域 POC 报告](../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md)，数据生成与限制见
[跨域数据 provenance](CROSS_DOMAIN_DATA_PROVENANCE_ZH.md)。

下一阶段不再沿用 local baseline 的 50%--99% 增益作为方法 promotion 标准。新的待确认指标是：
half-budget strong-baseline gain、reducible-gap closure、normalized budget-error AUC 与 equivalent probe
saving；完整定义和建议 gate 见[跨数据集指标提案](CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md)。

当前 active 方法目标是将 discrete response subspace 升级为 Prompted Functional Tensor Operator：连续
complex response bases + prompt-conditioned system latent + reciprocity-by-construction。functional tensor 是
主方法，FNO/DeepONet 是 matched operator baselines；latent flow 只有通过条件多模态 existence gate 才进入。
分阶段计划与顶会 NO-GO 条件见 [Wave as Prompt 顶会计划](WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md)。

## 2026-09-24 上一阶段：In-Context Wave Response Query

### RQ-1 已完成：Q2 source query 通过

在 co-located 9-transducer multistatic array 上，gain-calibrated reciprocity routing + 121,938 参数 masked
residual Transformer 得到：

- IID K=8：0.498 ± 0.014 vs 0.624，相对改善 20.1%；
- ellipse geometry OOD：0.427 ± 0.002 vs 0.558，相对改善 23.5%；
- IID paired-system bootstrap 95% improvement interval：[12.7%, 28.7%]。

Q2 是同频率、target source 未见；Q3 的 source 与 frequency 同时未见仍失败。详细报告见
[RQ-1](../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md)。

当前核心问题不再是改进 medium readout，也不是从低频直接生成整张高频场，而是：

> 给定一个未见 system 的少量 source--frequency complex responses，能否在 inference time 识别其 response
> operator，并回答 source 和 frequency 都未见的新 query？

首轮固定 receiver array，定义 Q1 frequency-held-out、Q2 source-held-out、Q3 joint-held-out；Q3 为主任务。
使用现有 160×9×8 dense archive 抽取 receiver vectors，先完成 conditional identifiability、强
interpolation/low-rank/retrieval baselines、decoder existence 和 active-query oracle，再训练最小
DeepSets/CNP。只有 global bottleneck 被证实才进入 cross-attention。

Belief/flow 需要条件多模态证据，active policy 需要 oracle headroom 与 uncertainty calibration；RL、可变
receiver、world model、3D 和 branch router 均后置。冻结协议见
[In-Context Wave Response Query 计划](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)。

## 2026-09-23 最新阶段：多分支机制成立，fixed 实现未过方法 gate

单个 local phase advance 会把同一邻域内的多个相干传播分量压成一个统计量。MPS-0/1 已完成从
information oracle 到 matched downstream 的逐层检验：

| 层级 | 主结果 | 判断 |
|---|---|---|
| 受控 2/3 分支混合 | 正确 K 将 RelL2 0.664/0.772 降到 0.313/0.326 | 分支可辨识 |
| wavelength-scaled real-field oracle | partial/closed 改善 26.3%/25.7% | oracle gate 通过 |
| three-seed synthetic single-frequency readout | 改善 12%--18% | 稳定正信号 |
| 64→128 partial/closed boundary | 改善 11.1%/11.7%，30/30 system 同向 | **15% gate 未过** |
| WaveBench relative-phase Gram | 单 seed full-context 改善 3.54% | 机制结果，不是正式验证 |

最重要的新原则是：局部分析尺度应随波长变化；复分支表示只应 quotient 一个共同 global phase，同时保留
分支之间决定干涉的相对相位。当前 two-scale packet Gram 把这两个原则实现成固定 69-channel raster，
但只关闭了部分 oracle gap。因此停止继续枚举固定滤波器、半径和手工 edge；若重启，必须引入 learnable
analysis、显式 branch routing 或新的 multipath query。

完整方法、三 seed 统计和 stop decision 见
[MPS-1 报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)；原始命题与相关工作见
[多分支相空间波表征](MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md)。

## 2026-09-23 M1 更新：第二数据集成立，但强效应 gate 未过

在公开 WaveBench time-harmonic acoustic 数据上，使用 8,000 / 500 / 500 个 system-held-out
train/validation/test split，完成了三个随机种子的 matched comparison。同一个 shared set operator 从任意
1--4 个 frequency responses 恢复隐藏 wavespeed：

| 表示 | 1 frequency | 2 frequencies | 3 frequencies | 4 frequencies |
|---|---:|---:|---:|---:|
| Raw complex | 0.3024 | 0.2422 | 0.2077 | 0.1811 |
| Amplitude only | 0.3807 | 0.2891 | 0.2398 | 0.2088 |
| Amplitude + phase advance | **0.2794** | **0.2140** | **0.1814** | **0.1584** |

相对 raw 的改善为 7.6%、11.6%、12.7%、12.5%，所有 seed 和 budget 同向。完整四频 paired bootstrap
的 95% 区间为 12.10%--12.93%。这使“phase-coordinate 有跨数据集实用性”从单域观察升级为 A 级事实；
但没有达到预注册的“两个 sparse budgets 至少 15%”强门槛，不能包装成显著架构突破。

两个更复杂的 fixed complex-edge tokens 均弱于当前 soft phase advance。下一步不再继续手工加通道，
而是检验 shared complex-edge graph/point operator。详见
[WaveBench M1 报告](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)。

## 2026-09-22 M0 更新

shared view encoder 与 permutation-invariant 4-slot aggregation 已完成三 seed 正式评估。最重要的结果不再是
固定 24-channel concat，而是一个参数量不随 view budget 增长的 set operator：

| 表示 | 24 views | 6 sources × 1 frequency | 20 dB |
|---|---:|---:|---:|
| Raw complex | 0.0593 | 0.2283 | 0.5288 |
| Amplitude only | 0.1226 | 0.5836 | 1.0807 |
| Amplitude + phase advance | **0.0533** | **0.1622** | **0.1124** |

新结论是：局部 phase advance 主要提高 frequency-poor 与 noisy regime 的观测效率。full-context 相对 raw
仅改善 10.1%，paired bootstrap 区间跨零；clean ellipse 和 128-grid boundary 上 raw 更强；10 dB 仍失败。
详见 [M0 正式报告](../reports/WAVE_SET_OPERATOR_M0_ZH.md)。

当前 instance-contrastive pretraining 不是贡献：冻结 probe 约 0.31，且 full-label 下从头训练为 0.0491，
优于预训练初始化的 0.0639。后续不再继续扫 pooling 或 auxiliary loss。

## 一句话问题

给定一个未知物理系统产生的少量复波动响应，能否学到一个跨 source、frequency、geometry 与
discretization 可复用的系统表示，并最终用它回答新的物理 query？

当前先解决“表示是否存在且是否保留系统信息”，再解决完整复场生成。这个顺序来自一个反复出现的
事实：模型可以较好识别 amplitude 与局部传播状态，却很难从有限频带唯一确定远处频率的完整相位。

## 当前表示

把复场写成 amplitude 与 phase 后，不直接编码绝对 phase，而使用：

1. 每个 response view 内归一化的 log-amplitude；
2. 固定物理距离上，水平方向的复相位 advance 的实部与虚部；
3. 固定物理距离上，竖直方向的复相位 advance 的实部与虚部。

共 5 个实通道。局部相位 advance 通过 `u(x + delta) * conjugate(u(x))` 获得，因此对全局相位旋转
不敏感，也不需要显式 phase unwrapping。低幅值位置用 soft amplitude gate 降低相位通道权重。

这个设计借用了经典的 local wavevector 与 coherence 思想。项目的增量不在于重新发明相位差，而在于
证明它可以作为 response-conditioned system representation，并建立跨 source-frequency 的学习任务、
匹配对照和 failure boundary。

## 已经观察到的强正结果

| Probe | Raw complex | Amplitude | Amplitude + phase advance |
|---|---:|---:|---:|
| Cross-view system retrieval Top-1 | 38.9% | 61.1% | **100.0%** |
| IID hidden-medium RelL2 | 0.3588 | 0.2478 | **0.0504** |
| OOD ellipse hidden-medium RelL2 | 0.4637 | 0.4407 | **0.0955** |

其中 retrieval 的 context 与 candidate views 在 source 和 frequency 上都不重叠；hidden-medium 的训练、
验证和测试按介质严格划分。三 seed 标准差和所有消融见
[完整报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

最重要的科学修正是：

> “幅值容易迁移、相位难迁移”只对了一半。真正不稳定的是绝对相位；局部、gauge-invariant 且按
> 置信度加权的 phase advance 反而是辨识传播系统的关键信息。

## 最重要的负结果

| 尝试 | 结果 | 科学含义 |
|---|---|---|
| source-frequency RBF / eikonal 坐标插值 | RelL2 约 1.15 | 普通坐标平滑不能表达复杂干涉与散射 |
| 少源 Nyström / 全局低秩 Green completion | RelL2 约 0.99 | 少数全局 source columns 不足以恢复复杂响应算子 |
| raw、carrier 或 oracle-medium U-Net 直接生成 missing block | RelL2 约 0.93--0.97 | 当前 decoder 与数据规模不足以生成高频完整复场 |
| unit phase token 加噪 | 20 dB 已明显崩溃 | 低幅值区的 phase 必须做 confidence weighting |
| 64→128 且 partial / closed boundary | 最好仍为 0.614 / 0.637 | 固定物理 stride 有效，但单尺度局部 token 无法处理强多路径 |

这些失败把当前问题定位得更清楚：瓶颈不只是模型容量，也不是简单增加 loss 可以解决；需要分别处理
context representation、multi-branch propagation 和 oscillatory decoding。

## 现在可以相信什么

- 复波场不应默认使用 real/imag 两通道图片表示。
- amplitude 提供相对稳定的系统 fingerprint，局部 phase advance 提供 amplitude 缺失的传播方向和
  局部波数信息。
- 一个轻量、可解释、无需解析 travel time 的 representation 已经显示出强系统辨识能力。
- representation sufficiency 与 field generation 是两个不同问题；前者的成功不能被写成后者已经解决。

## 现在不能声称什么

- 不能声称已经完成 unseen source-frequency 的高精度完整场生成。
- 不能把一个 simulator 上的 ellipse OOD 写成跨方程或跨物理域泛化。
- 不能声称在 10 dB、强混响边界或真实稀疏阵列上稳健。
- 不能把经典 phase-gradient 理论本身写成新的理论贡献。

## 当前 response-query 决策 gate

### Gate 1：Q2 已通过；Q3 information 与 baseline gate 保留

Q2 已相对 gain-calibrated reciprocity 改善 20.1%。Q3 target 仍必须随 context budget 变得更可辨识；未来
joint-query model 必须相对最强 interpolation/retrieval baseline 改善至少 15%。

### Gate 2：Belief gate

只有 context-near systems 对 target 存在分离 modes，且 deterministic model 出现均值化，才训练 mixture、
flow 或 diffusion。

### Gate 3：Active gate

只有 complete-table best-next-query oracle 相对最佳 fixed schedule 的 error-AUC 改善至少 15%，且 learned
uncertainty 可以校准，才训练 active acquisition；RL 最后进入。

## 当前判断

RQ-1 已证明表示能回答受控但有实际意义的 held-out-source probe，而不仅是辨识 system fingerprint。正结果
来自 observation operator 与模型共同设计：co-located array 暴露 reciprocity，网络只学习解析路由后的
残差。下一阶段不是扩大 Transformer，而是第二 multistatic 数据域与 per-link spectral state；不要用
foundation model、belief-space 或 active agent 的长期 framing 掩盖 Q3 尚未解决。

## 入口

- [完整方法与实验报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)
- [RQ-1 正式方法与结果](../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md)
- [当前 Response Query 主线与冻结协议](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)
- [多分支相空间 MPS-1 报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)
- [WaveBench 独立验证与高维路线](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)
- [实验账本：正负结果、禁重复清单与未开发信号](EXPERIMENT_LEDGER_ZH.md)
- [下一阶段冻结协议](WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md)
- [科学备忘录](SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)
- [数据与复现档案](REPRODUCIBILITY_ZH.md)
- [历史探索索引](archive/INDEX_ZH.md)
