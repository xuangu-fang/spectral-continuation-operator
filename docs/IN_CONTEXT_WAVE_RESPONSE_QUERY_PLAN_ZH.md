# In-Context Wave Response Query：主线、问题设定与阶段协议

> 冻结日期：2026-09-24
>
> 状态：当前 active research plan
>
> 目标：从“编码或重建一张波场”转向“用少量复数响应识别一个未见波动系统，并回答新的实验 query”。

> 执行结果更新：RQ-1 已在 co-located multistatic array 的 Q2 held-out-source query 通过 gate；三 seed IID
> K=8 为 0.498 vs calibrated reciprocity 0.624。Q3 仍未通过。详见
> [`../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md`](../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md)。

## 0. 一句话主线

> **给定一个未知波动系统的少量 source--frequency probe responses，模型能否在不更新权重的条件下形成
> system representation，并准确回答新的 source--frequency response query？**

长期 vision 是：

> **From predicting wave fields to interrogating the physical world with waves.**

但当前阶段只实现这条链条的第一层，不同时加入 generative belief、active sensing、RL、环境干预和
cross-physics foundation model：

```text
response query
      ↓
in-context system identification
      ↓
predictive belief / uncertainty
      ↓
active wave interrogation
      ↓
counterfactual wave world model
```

## 1. 第一性原理定义

隐藏环境记为 $\theta$，包含介质、几何和边界。一次 time-harmonic 实验满足：

$$
y=\mathcal M_r\mathcal R_\theta(\omega)f_s+\epsilon,
$$

其中：

- $\mathcal R_\theta(\omega)$ 是环境在频率 $\omega$ 下的 resolvent / Green operator；
- $f_s$ 是 source excitation；
- $\mathcal M_r$ 是 receiver measurement；
- $q=(\omega,s,r)$ 是一次可以向系统提出的 query；
- $y=H_\theta(q)$ 是 response。

给定 context：

$$
D_K=\{(q_i,y_i)\}_{i=1}^{K},
$$

模型需要在完全未见的物理系统上回答：

$$
\widehat y^\star=F_\psi(D_K,q^\star),
$$

且 inference 时不更新 $\psi$。

### 1.1 Query 的三个分量并不对称

| 分量 | 数学角色 | 需要检验的泛化 |
|---|---|---|
| frequency $\omega$ | 改变 resolvent | resonance、dispersion、phase wrapping |
| source $s$ | 改变右端激励 | source linearity、illumination transfer |
| receiver $r$ | 改变测量泛函 | spatial sampling、reciprocity、array geometry |

因此实验必须区分 frequency-held-out、source-held-out、receiver-held-out 和 joint-held-out，不能只报告
一个混合平均数。

### 1.2 Probe intervention 与 environment intervention 不同

改变 $q$ 是询问同一个系统 $\theta$；改变 defect、boundary 或 material 是构造另一个系统 $\theta'$。当前
数据支持前者，不支持严格的 defect counterfactual。第一阶段使用 `response query`，不使用因果
`world model` 的强 claim。

## 2. 为什么这是当前结果的自然下一步

已有实验建立了三个事实：

1. amplitude + gauge-invariant local phase 可以跨 source/frequency 辨识 system；
2. variable-budget set operator 能从任意数量 response views 恢复 hidden medium；
3. 直接生成未见 source-frequency 的完整 64×64 complex field 仍约为 0.93--0.97。

这说明 system identification 已有正信号，但 dense oscillatory decoding 太难。Response Query 将目标从整张
image 改为固定 receiver array 上的 response vector，使我们能够单独检验：

> 当前表示是否真的支持 counterfactual probe prediction，而不仅是恢复一个容易读出的 medium fingerprint。

多分支相空间结果在这里作为设计约束，而不是第一版模型组件：局部尺度按 wavelength 定义、只 quotient
一个共同 global phase、保留 branch-relative phase；fixed 69-channel Gram 不直接进入 RQ-1。

## 3. 与相关工作的边界

- [ICON](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523630/) 已定义 inference-time operator learning，但没有
  专门处理 complex gauge、multipath、reciprocity 和 source--frequency compositional query；
- [Zebra](https://proceedings.mlr.press/v267/serrano25a.html) 使用生成式 autoregressive in-context PDE
  adaptation，主要对象是 parametric time-dependent trajectories；
- [Reinforced Inverse Scattering](https://epubs.siam.org/doi/full/10.1137/22M153207X) 已选择 frequency 与
  sensor locations，因此“用 RL 选频率”本身不是新贡献；
- [Neural Green's Functions](https://arxiv.org/abs/2511.01924) 从已知 domain geometry 学 source/boundary
  agnostic solution operator；本项目要求仅由 responses 识别未知 environment；
- [frequency-domain scattered-wave neural operator](https://academic.oup.com/gji/article/241/3/1467/8096642)
  使用已知 velocity 和 physics-informed background field，是 forward surrogate 而非 in-context system ID。

潜在增量必须来自四者的交叉，而不是重新命名一个 Set Transformer：

1. environment 只通过少量 complex measurements 暴露；
2. source 与 frequency 的 compositional held-out query；
3. wave-specific gauge、wavelength 和 reciprocity representation；
4. 由 query uncertainty 支撑、但后置的 active experiment design。

## 4. 第一阶段主动剪枝后的任务

### 4.1 固定 receiver array

首轮不把 receiver configuration 也变成 action。固定 $R$ 个 receivers，一次 probe 为：

$$
q=(s,\omega),
$$

response 为：

$$
y_q=[u(r_1;s,\omega),\ldots,u(r_R;s,\omega)]\in\mathbb C^R.
$$

这样可以先回答 source/frequency query 是否可学。可变 receiver array、sensor placement 和 point-query 在
RQ-2 以后进入。

### 4.2 数据

首轮复用：

```text
operator_id_64_160.npz
160 systems × 9 sources × 8 frequencies × 64 × 64 complex fields
112 / 24 / 24 system-held-out split
```

从每张 dense field 中抽取固定、公开的 receiver coordinates，形成完整的 environment response table。
初始比较 32 receivers 和 64 receivers，但 development 只选择一次，不在 test 上选 array。

外部数据的角色：

- 128-grid 30-system open/partial/closed：只做 OOD boundary stress test；
- ellipse systems：geometry OOD；
- WaveBench：只有四频和固定 source，后续只做 frequency-query 外部验证，不承担第一阶段完整任务；
- 新仿真：只有现有 112 train systems 已证明 existence、但出现 system-count bottleneck 后才生成。

## 5. Query protocol

| ID | Context 与 target | 作用 | 论文地位 |
|---|---|---|---|
| Q0 | 同 source/frequency，未见 receiver | spatial unit test | 不作 headline |
| Q1 | source 已见，frequency 未见 | spectral query | 次指标 |
| Q2 | frequency 已见，source 未见 | illumination query | 次指标 |
| Q3 | source 与 frequency 都未见 | joint operator query | **主任务** |
| Q4 | Q3 + ellipse/boundary/grid shift | robustness | promotion gate |

### 5.1 Context budgets

冻结 budgets 为 1、2、4、8、16 probes。训练时同一 checkpoint 随机接收可变 K；测试枚举预先冻结的
context/query episodes，使所有模型使用完全相同的信息。

### 5.2 防泄漏规则

- train/validation/test 以 physical system 划分；
- 同一 system 的 source-frequency table 不跨 split；
- query 的 source/frequency 严格不出现在 Q3 context；
- normalization 只由 train systems 估计；
- receiver coordinates 和 episode seeds 在模型比较前冻结；
- 不根据 test query difficulty 选择 context。

## 6. RQ-0：训练前的信息与 oracle audit

### RQ-0A：条件可辨识性

寻找 context 很接近但 query response 差异很大的 system pairs：

$$
d(D_K^{a},D_K^{b})\ll 1,
\qquad
d(y_a^\star,y_b^\star)\gg 1.
$$

输出：

- kNN conditional query error 随 K 的曲线；
- context collision 与 target divergence 的二维图；
- Q1/Q2/Q3 分解；
- ambiguity 是否随增加某个 frequency/source 明显收缩。

它决定 deterministic mean 是否足够，以及 belief model 是否有信息论依据。

### RQ-0B：强非神经 baseline

必须包含：

1. nearest observed frequency/source；
2. source-frequency coordinate interpolation；
3. per-environment low-rank/SVD table completion；
4. HarmoCore-style functional/tensor completion；
5. nearest training-system retrieval conditioned on context；
6. context linear combination / complex ridge readout。

这些 baseline 使用相同 receiver vector、context 和 query，不允许读取 hidden medium。

### RQ-0C：decoder existence

分别比较：

- true medium + query 到 receiver response 的 oracle decoder；
- predicted medium + query decoder；
- response-context-only query model。

如果 true-medium learned decoder 本身不工作，应优先修正 oscillatory decoder，而不是责怪 context encoder。

### RQ-0D：active-query oracle

利用完整 table 穷举下一个可测 probe，比较它对预先冻结 query set 的误差/不确定性降低：

- random；
- uniform frequency sweep；
- fixed source schedule；
- oracle best next query。

oracle 相对最佳 fixed schedule 没有至少 15% 的 area-under-error-curve 改善，则不进入 learned active policy。

## 7. RQ-1：最小 deterministic query model

### 7.1 Model A：DeepSets / Conditional Neural Process

每个 probe 编码为：

$$
h_i=E(q_i,y_i),
\qquad
z_\theta=A(\{h_i\}),
\qquad
\widehat y^\star=D(z_\theta,q^\star).
$$

这是容量和归因最清楚的 baseline。

### 7.2 Model B：query-to-context cross-attention

query token 直接读取全部 context tokens，不强制把 system 压成单个向量：

$$
\widehat y^\star=D(q^\star,\mathrm{Attend}(q^\star,\{h_i\})).
$$

只在 Model A 通过 basic existence 后实现；控制参数量与训练预算。

### 7.3 Representation ablation

首轮只比较：

1. normalized raw real/imag response vector；
2. log-amplitude only；
3. amplitude + wavelength-scaled receiver-edge phase；
4. raw + edge phase residual representation。

对于 receiver graph 的相邻点 $i,j$：

$$
g_{ij}=y_j\overline{y_i},
$$

edge selection 按 $\|r_i-r_j\|/\lambda$ 而非像素距离。global phase rotation augmentation 用于检查
invariance/equivariance。多分支 router、69-channel Gram、flow 和 PDE loss 均不进入首轮。

## 8. 指标与 promotion gates

### 8.1 主指标

- 每个 physical system 等权的 raw complex relative L2；
- 先对同一 system 的 frozen queries 求平均，再跨 systems 统计；
- multi-seed 时先按 system 聚合，再 paired bootstrap。

### 8.2 诊断指标

- normalized amplitude error；
- complex coherence；
- circular phase error，仅在 amplitude 高于冻结阈值处报告；
- context-to-query uncertainty calibration；
- context permutation consistency；
- global phase rotation equivariance。

诊断指标不能替代 raw complex 主指标。

### 8.3 24 小时 POC gate

在现有数据上的第一轮成功至少满足：

1. Q3 error 随 K 稳定下降，而非只在 Q1 插值成立；
2. learned model 相对最强 interpolation/retrieval baseline 至少改善 15%；
3. raw complex error 相对旧 dense-field generation 的约 0.93 有实质下降；
4. source-only 与 frequency-only 至少一项不过拟合；
5. 不通过 test 选 receiver array、episode 或 representation。

若相对 baseline 的改善不足 15%，必须归因为 information、decoder、optimization 或 dataset-size，而不是
继续堆组件。

### 8.4 Belief gate

只有 RQ-0A 显示 context-near systems 对 target 存在分离 modes，且 deterministic model 出现均值化，才训练
mixture、conditional flow 或 diffusion。需要使用 NLL、coverage、energy score 和 posterior contraction，
不能只比较 sampled best-of-N。

### 8.5 Active gate

只有 RQ-0D oracle 通过且 uncertainty 校准成立，才比较 learned greedy acquisition。RL 放在最后，不作为
第一个 active baseline。

## 9. 24 小时执行顺序

### M0：协议与数据管线

- 冻结 receiver arrays、episode seeds、Q1/Q2/Q3 和 budgets；
- 生成可复现 episode manifest；
- 验证 system split、query leakage 和 normalization；
- 实现 raw complex metric 与 system-paired aggregation。

### M1：信息审计与强 baseline

- nearest query、coordinate interpolation、context-conditioned retrieval；
- response-table SVD/low-rank completion；
- conditional ambiguity 和 context budget 曲线；
- active oracle headroom 的廉价 existence check。

### M2：最小 query model

- DeepSets/CNP raw complex；
- amplitude-only 强负对照；
- gauge-aware receiver graph；
- 多轮调整只允许改变一个清楚的归因变量。

### M3：必要时的小型 cross-attention

只有 DeepSets 显示 learnable signal、但 global bottleneck 明确时进入。保持相同 episode、参数级别和训练
steps；不把 attention 的容量优势误写成物理表示优势。

### M4：冻结结果

- 最佳 development variant 三 seed；
- paired bootstrap、Q1/Q2/Q3 和 budget 表；
- prediction examples、failure examples、ambiguity map；
- 中文报告、机器 JSON/CSV、配置、测试、README 与 experiment ledger；
- 不成功也要记录 stop decision 和可重启条件。

## 10. 明确暂缓

- learned active policy、RL、POMDP；
- generative posterior、flow matching、diffusion；
- variable receiver configuration；
-环境 defect/boundary counterfactual；
- cross-physics pretraining；
- 3D/4D 扩展；
- learned multibranch router；
- 用 amplitude/phase 指标替换 complex query error；
- 因加入 Transformer 就声称 in-context reasoning。

## 11. 预期科学分岔

### A. Q3 query 明显成功

说明 response context 能在 inference time 形成可用 system latent。下一步扩大 systems、做 WaveBench
frequency-only 验证，并开始不确定性审计。

### B. Q1 成功、Q2/Q3 失败

说明模型主要做频率插值，source/operator transfer 尚未解决；优先引入 reciprocity、source linearity 或
Green-operator decoder，不包装成通用 system ID。

### C. true-medium decoder 成功、context model 失败

说明瓶颈是 system inference；利用现有强 medium representation 或加强 context set model。

### D. true-medium decoder 也失败

说明瓶颈仍是 oscillatory decoding；停止扩大 context encoder，转向 wave-aware implicit/Green decoder。

### E. deterministic error 高且 ambiguity 明确多模态

这是 generative belief 的合法入口；先报告 identifiability fact，再选择 mixture/flow。

### F. active oracle 无 headroom

冻结 Active Interrogation，不训练 policy。新的 action 必须改变 information type，而不是继续搜索相似
source/frequency combinations。

## 12. 论文潜力与声明边界

若 RQ-1/2 成立，潜在故事不是“一个更好的 wave surrogate”，而是：

> **Complex wave measurements act as an in-context prompt that identifies an unseen physical response operator.**

最小贡献需要同时包含：

1. source--frequency response-query benchmark；
2. inference-time unseen-system adaptation，无 gradient update；
3. wave-specific complex representation；
4. context budget 与 compositional query generalization；
5. 至少一个独立数据域或严格 geometry/boundary shift。

在这些条件未满足前，不使用 `wave foundation model`、`world model` 或 `autonomous scientist` 作为论文主
claim。
