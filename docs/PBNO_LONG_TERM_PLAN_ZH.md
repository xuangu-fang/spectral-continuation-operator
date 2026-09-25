# PBNO 长程研究计划：从响应子空间到 Prompted Bayesian Neural Operator

> 冻结日期：2026-09-25  
> 状态：active；本文件是新一阶段唯一执行入口  
> 目标：在保留现有 Wave-as-Prompt 强结果的前提下，把解析 PCA--MAP 升级为可形成 AI 论文核心贡献的
> query-conditioned neural operator。

## 1. 论文命题

我们不把模型定义为“从若干张图预测另一张图”，而把一个未知波动系统定义为可被少量实验识别的函数：

$$
H_{\theta}(s,\omega,r)
=
\mu_{\phi}(s,\omega,r)
+
\sum_{m=1}^{M}z_{\theta,m}\psi_{\phi,m}(s,\omega,r).
$$

这里 $q=(s,\omega,r)$ 是可选择的物理 query，$z_{\theta}$ 是未知系统的隐状态，
$\psi_{\phi,m}$ 是跨系统共享、随 query 连续变化的复值 response functions。测试时只给少量
prompt--response pairs，不提供介质、几何或 PDE 参数，也不更新网络权重。

一句话故事是：

> **A few wave responses serve as prompts that identify a latent physical operator, which can then answer
> unseen source--frequency queries.**

## 2. 为什么这不是给 PCA 加一个网络

现有逐频率 PCA--MAP 只解决 Q2：在目标频率已经观测其他 sources 时，补全一个 source row。PBNO 的核心
增量是同时学习：

1. 跨 frequency 共享的 system state；
2. 从物理 query coordinates 到复 response basis 的函数；
3. prompt 数变化时仍有一致语义的 Bayesian posterior；
4. posterior predictive uncertainty，而不是单个 deterministic embedding。

当前 PCA--MAP 必须作为严格特例保留：当 $\psi$ 是离散 SVD basis、prior 为对角复高斯且 decoder 为线性时，
PBNO 精确退化为现有方法。这样可以把收益清楚归因给 learned functional representation，而不是训练预算或
不公平的 inference procedure。

## 3. 冻结任务

### Q2：held-out source，已建立的下限

目标 source 不出现在 prompts；目标 frequency 可以出现。Q2 用来保证新方法没有牺牲当前成熟结果。

### Q3：joint held-out source--frequency，论文主战场

目标为 $(s^\star,\omega^\star)$。每个 context pair $(s_i,\omega_i)$ 必须同时满足：

$$
s_i\neq s^\star,
\qquad
\omega_i\neq\omega^\star.
$$

因此模型既不能从同频率其他 source 做普通行补全，也不能从同 source 其他频率做普通谱插值。所有较小预算
使用同一冻结顺序的 prefix；target pair 永不进入 context。

正文主预算固定为 $K\in\{1,2,4,8\}$，其中 $K=4$ 为共同主 operating point，完整曲线进入正文。每个
physical system 是一个统计样本；先对该系统内所有 target queries 与 prompt orders 平均，再做跨系统统计。

## 4. 单一方法：Prompted Bayesian Neural Operator

### 4.1 Query trunk

第一版 trunk 输入无量纲 source、log-frequency 与 receiver coordinates，并输出 $M$ 个 complex basis values：

$$
(s,\log\omega,r)\longmapsto
\{\psi_{\phi,m}(s,\omega,r)\}_{m=1}^{M}.
$$

坐标先经 Fourier features；receiver 方向保留高容量 oscillatory representation，不施加已经被证伪的低
source/receiver rank。规则 receiver grid 上可用 spectral convolution，但它只实现函数 trunk，不改变
prompt posterior 的定义。

### 4.2 Analytic Bayesian prompt posterior

令 $\Phi_D$ 为所有 context coordinates 上的 basis evaluation，$y_D$ 为观测。使用对角 complex Gaussian
latent prior 与 observation noise 后：

$$
\Sigma_D^{-1}
=
\Lambda+\Phi_D^{*}\Sigma_{e}^{-1}\Phi_D,
$$

$$
m_D
=
\Sigma_D\Phi_D^{*}\Sigma_e^{-1}(y_D-\mu_D).
$$

目标 query 的 posterior mean 为：

$$
\widehat y_{q}
=
\mu_{\phi}(q)+\Phi_{\phi}(q)m_D.
$$

训练目标只有一个 coherent predictive objective：复高斯 posterior predictive NLL；开发阶段同时监控 raw
complex RelL2。reciprocity 只在数据确实共址时用结构化对称化实现，不增加 loss soup。

### 4.3 Flow 的边界

flow 只能作用在低维 $z_{\theta}$ posterior，不能直接生成整张高维场。只有 conditional-collision 审计显示
Gaussian posterior 的 best-of-8 或 coverage 存在至少 15% 可利用 gap，才启动 conditional flow；最终必须用
NLL、energy score 与 coverage 证明必要性，不能只报 best-of-many error。

## 5. 分阶段实验与停止规则

### M0：信息与容量审计

实现跨全部 source--frequency response table 的 global complex Bayesian subspace，并比较：train mean、最近
query、context-conditioned system retrieval、isotropic prior 与 latent-spectrum prior。

必须回答：

- full-table reconstruction error 随 system rank 如何变化；
- Q3 error 是否随 K 单调下降；
- 失败来自表示容量，还是 prompts 无法辨识 latent；
- Q3 在哪些域仍留下至少 10% neural headroom。

若 global subspace 在某域对 K 完全不敏感且 retrieval 也不改善，则不在该域盲目训练 PBNO，先缩为 in-band
bracketed query 或重新审计 response normalization。

### M1：最小 semilinear PBNO

仅实现 query trunk + analytic posterior。初始化可来自 M0 global SVD，但训练时通过 variable-budget Q2/Q3
episodes 优化 response functions。第一轮不加入 flow、PDE residual、额外 phase loss或 amortized encoder。

### M2：matched neural baselines 与正式统计

比较参数量/训练步数匹配的 DeepONet/CNP 或 masked response operator。三 seed、system-level paired bootstrap，
并报告参数、FLOP、峰值显存与推理延迟。

### M3：只由证据触发的 extension

- conditional ambiguity 成立：latent conditional flow；
- regular-grid trunk 明显优于 coordinate trunk：receiver FNO；
- reciprocity 覆盖足够且消融有收益：结构化 reciprocal basis；
- active oracle 显示 query selection headroom：再进入 Wave as Action。

## 6. Promotion gates

### Q2 保底 gate

half-budget 相对当前冻结 spectrum-prior：至少两个域 raw complex RelL2 再降低 5%，第三域退化不超过 2%。
若 Q2 已接近可约误差下限，允许以相同误差（正负 2%）减少至少 25% latent/storage/compute 作为通过。

### Q3 主 gate

相对 validation 选出的 global subspace、retrieval 与 matched neural baseline 中最强者：

- 至少两个域降低 raw complex RelL2 10%；
- system win rate 至少 70%，paired bootstrap 95% 下界大于 0；
- 同一 checkpoint 支持至少三个 prompt budgets；
- Deepwave/Ocean OOD 至少一域改善 5%，另一域退化不超过 3%。

### 论文性 gate

只有同时满足以下条件，才把 PBNO 定位为方法型 AI 论文：

1. Q3 而不只是 Q2 有跨域正结果；
2. learned query function 的消融优于离散 global PCA 特例；
3. prompt posterior 随 K 呈合理收缩，并有校准指标；
4. 参数/延迟与 matched baseline 公平；
5. 方法图、主表、预算曲线和 failure case 可由单命令复现。

## 7. 24 小时执行顺序

1. 冻结 Q3 episodes、global Bayesian subspace 与 retrieval baselines；
2. 在 OpenFWI、Deepwave、Ocean 做 M0 validation rank/prior selection；
3. 只在验证集有 headroom 的两个以上域实现 semilinear PBNO；
4. 单 seed development 通过后再做三 seed test/OOD；
5. 生成正文级表图、中文报告、配置/hash 与实验账本；
6. 未通过 gate 的组件写成明确负结果，不通过改变 test metric 补救。

## 8. 当前最重要的防偏航原则

不再尝试把 source、receiver 或频率分别强制压成小 tensor rank。已经成立的事实是：低秩主要位于跨系统的
response family；PBNO 学的是 **system latent × query-conditioned oscillatory response functions**。方法的
现代 AI 味来自 in-context operator identification、functional query representation 与 Bayesian prediction，
而不是组件数量。

## 9. 2026-09-25 M0/M1 执行结果

完整数字见 [PBNO M1 报告](../reports/PBNO_JOINT_QUERY_M1_ZH.md)。阶段判断如下：

- M0 strict Q3 existence gate 通过：三个域的 global Bayesian operator 均随 K 稳定下降；
- M1 neural transport 在 OpenFWI sparse budget 达到 11.7%，Deepwave 达到 7.1%，但没有满足两个域均
  10% 的 promotion gate；
- OpenFWI K=8 退化 3.0%，已经触及允许边界，下一版必须优先修复而非只刷 K=1；
- Ocean nonlinear residual 无收益，停止对其增加 neural capacity；
- posterior-covariance concat 消融失败，不进入主方法；
- Q2 必须保留 frequency-local head，Q3 使用 global head，形成由可观测 query mask 决定的 hierarchy；
- severe latent shift 能从 prompts 检出，但当前 Ocean abstention 仍是开发期 safety POC；
- flow existence gate 未通过，保持关闭。

后续开发已经进一步验证：普通 prompt-set updater 没有超过 latent transport；receiver-preserving residual
只在 OpenFWI 显示单域正信号，在 Deepwave 上因 response scale 失配而不稳定；Deepwave coordinate-RMS
版本又降低精度。因此当前冻结 PBNO 为科学/任务论文候选，不再继续堆 encoder。重启神经 residual 的前置
条件改为：先得到跨域一致的 complex likelihood/whitening，使 Deepwave 训练稳定且线性初始化不退化。
