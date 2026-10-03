# Target-Aware Wave Interrogation：下一阶段论文计划

> **历史计划，不是当前 roadmap。** M1 已经执行完毕；2026-10-03 起的研究排序与新 gate 统一见
> [当前故事、证据与未来路线图](CURRENT_STORY_AND_ROADMAP_ZH.md)。保留本文只是为了核对预注册项目与结果，
> 后续 Agent 不得从本文直接启动旧 M1/M2 清单。

> 日期：2026-09-27
> 前置结果：先读 `reports/ACTIVE_WAVE_INTERROGATION_M0_ZH.md`
> 目标：把强 POC 变成经得住“只是固定 schedule / classical design / test reuse”质疑的论文证据。

> **执行状态（2026-09-28）：** M1 untouched systems、完整 acquisition baselines 与 continuous utility
> operator 均已执行。通过/失败门槛与下一步冻结决策见
> `reports/ACTIVE_WAVE_INTERROGATION_M1_ZH.md`；本文以下内容保留为预注册计划，不再是待自动执行清单。

## 1. 冻结主线

一句话故事：

> 从少量 wave prompts 学出一个可查询的 hidden-system operator；给定目标 response 与实验预算，用该
> operator belief 设计最有信息的 source--frequency probes，再回答目标 query。

当前方法原则固定为：

```text
learned complex response operator
        +
analytic posterior update
        +
target-aware acquisition
```

不把自由 Transformer policy、flow、PDE loss、幅相辅助 loss 或 RL 一次堆入第一阶段。

## 2. M1：必须完成的论文级验证

### 2.1 新 untouched systems

- 为 Deepwave、Ocean 各生成一套从未用于方法选择的新 systems；
- 数据生成参数与当前 IID family 一致，同时保留一个独立 OOD split；
- 冻结当前模型、acquisition rule 和全部 hyperparameters 后一次性打开；
- 主表继续使用逐 system raw complex RelL2、paired bootstrap 与 win rate。

Gate：Deepwave/Ocean K=2 至少一域保持 20%，另一域保持 15%；任一域若低于 10%，重新降级为 development
signal。

### 2.2 Acquisition 强基线

必须同时比较：

1. random nested schedule；
2. uniform/farthest source--frequency coverage；
3. nearest-target frequency 与 nearest-target source heuristic；
4. classical target-predictive variance，也就是当前方法；
5. D-optimal latent-volume reduction；
6. training-population fixed schedule；
7. target-truth per-system oracle，只作上限。

主 claim 必须相对第 4--6 项中最强者定义。若 population schedule 与当前方法继续等价，标题与摘要使用
`target-aware design`，不使用 `adaptive agent`。

### 2.3 真实成本轴

除了 probe count K，再报告：

- source activations；
- frequency switches；
- receiver samples；
- acquisition wall-clock proxy；
- planner latency。

如果不同 actions 的现实成本不同，使用 cost-normalized error--budget AUC，不能默认每个 probe 等价。

## 3. M2：让 learning 更现代，但只保留一个升级

### 首选：continuous acquisition operator

把当前有限候选枚举蒸馏为坐标连续的 utility operator：

$$
U_{\psi}(m_D,\Sigma_D,q^{\star},a)
\longrightarrow
\hbox{predicted target risk after action }a.
$$

第一版只学习解析 risk，不学习 target response residual。训练时随机 hold out source locations 与 frequency
coordinates，测试连续 action interpolation/extrapolation。

必须满足：

- held-out action regret 相对 exhaustive analytic planner 不超过 5%；
- planner latency至少降低 5 倍，或支持训练网格外连续 actions；
- Deepwave/Ocean error--budget AUC 不比 exhaustive planner 退化超过 2%；
- 参数少于 500k。

若只是复现离散 lookup 且没有连续 action/速度收益，不作为论文贡献。

### 暂缓：system-conditioned policy

只有下面的 conditional-action oracle gate 通过才重启：

$$
\frac{E_{population}-E_{system\ oracle}}{E_{population}} \ge 15\%.
$$

当前 Deepwave/Ocean/OpenFWI 约为 9.3%/14.1%/10.7%，均未通过。没有新信息条件前禁止继续加 policy
宽度、attention、RL 或更多 schedules。

## 4. M3：外部有效性

优先级顺序：

1. 可信 metadata 的真实 multistatic array；
2. 新的公开 scattering / ultrasound response archive；
3. Structural Lattice 作为跨 PDE synthetic control；
4. 不再增加第五个同质 acoustic toy simulator。

外部域最低 gate：K=2 相对最强非 target-aware schedule 改善 10%，或在相同 error 下节省 25% probes。

## 5. 若想真正进入 adaptive agent，问题必须改变什么

当前 posterior covariance 与 response value 无关，因此 response-conditioned policy 没有足够作用空间。要研究
真正 adaptivity，必须引入能让 action utility 随隐藏系统改变的条件，例如：

- 部分 receiver failures / unknown sensor noise；
- scene-dependent occlusion 或 boundary access；
- multimodal nonlinear inverse belief；
- downstream defect localization，而不是只预测固定 target response；
- action-dependent cost 或 source saturation。

这些是下一篇或 M3 以后的方向，不能为了 policy 外观偷偷混入当前主实验。

## 6. 投稿门槛

只有同时满足以下条件才进入 paper-writing freeze：

1. 新 untouched split 通过；
2. 至少两个域 K=2 相对最强 acquisition baseline 改善 10%；
3. 至少一个域 error--budget AUC 改善 15%，第二域不低于 8%；
4. 一个外部/跨物理域达到 10% 或 25% probe saving；
5. continuous utility operator 达到小于 5% action regret，或明确删除该 learning upgrade；
6. 三 seed、system-level interval、planner latency 与 acquisition-cost axis 完整；
7. 摘要明确区分 target-aware、sequential 与 response-adaptive。

## 7. NO-GO

- 不继续扩大 query-conditioned posterior metric；当前最多 2.55%；
- 不继续训练自由 neural policy；当前 Deepwave 回退，Ocean 只多 0.37%；
- 不用 target-truth action oracle 当作 deployable improvement；
- 不以 K=2 的强结果替代完整 AUC；
- 不把 classical target variance minimization本身写成算法新颖性；贡献必须来自 learned wave operator、
  inference-time response-query task 与跨域实证的结合；
- 不在已有 test systems 上继续选超参数。
