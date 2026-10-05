# 下一阶段跨数据集 Response Query 定量指标提案

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 日期：2026-09-25
>
> 状态：供讨论与冻结；当前 POC 的 test 结果不回溯改 gate

## 1. 为什么不能只看一个 raw RelL2

OpenFWI、Deepwave 与 Ocean 的 source 数、receiver 数、传播复杂度和无条件误差差异很大。raw complex
RelL2 必须保留，因为它回答“预测本身准不准”；但它不能独自回答：

1. 相对该数据域的强方法，增益是否真实；
2. 用一半 probes 达到了多少 full-context 信息；
3. source 数不同的时候，测量效率能否公平比较；
4. 改善来自 amplitude 还是 phase；
5. IID 正结果在 OOD 系统上还保留多少。

下一阶段建议只设一个主指标族，外加少量诊断指标。不要为每个数据集换一套“最好看”的度量。

## 2. 统计单位与统一预算轴

所有 query error 先在同一个 physical system 内，对 target source、frequency 和冻结的 context orders 求平均，
再在 systems 之间统计。不能把数万个 query 当成独立样本。

不同数据域使用归一化预算：

$$
\rho = \frac{K}{S-1},
$$

其中 `S` 是 source 数，target source 不计入可观测集合。建议正文固定两个 operating points：

- sparse：`rho` 约为 0.25；
- half-budget：`rho` 约为 0.50。

当前离散预算对应：

| 数据域 | source 数 | sparse K | half K | full K |
|---|---:|---:|---:|---:|
| OpenFWI | 5 | 1 | 2 | 4 |
| Deepwave | 16 | 4 | 8 | 15 |
| Ocean | 9 | 2 | 4 | 8 |

The Well 当前没有可验证的 source coordinates，不进入跨系统 gate。

## 3. 建议放在正文的五个指标

### M1. Raw complex relative L2

对每个 system `e`：

$$
E_e(K) =
\frac{\|\widehat{Y}_e-Y_e\|_2}{\|Y_e\|_2}.
$$

报告 system-macro mean、median 和 bootstrap 95% interval。这仍是第一主指标；amplitude 或 phase 指标不能
替代它。

### M2. Strong-baseline normalized gain

在 validation 上从冻结基线池中选出同预算最强基线 `b`，test 上计算：

$$
G_d(K) = 1 - \frac{E_{d,\mathrm{ours}}(K)}{E_{d,b}(K)}.
$$

`G=0.10` 表示同预算误差降低 10%。统计时保留 ours 与 baseline 的逐 system 配对误差；每次 bootstrap
重采样 physical systems，再计算两边 macro mean 的比值。另行报告逐 system win rate，不直接平均容易被
极小分母放大的 per-system ratio。下一阶段的基线池至少包括：

1. train mean、nearest/kernel source interpolation；
2. reciprocity anchor interpolation；
3. 当前 complex response subspace；
4. context-conditioned system retrieval；
5. direct supervised complex ridge；
6. 参数量与训练预算匹配的 generic masked Transformer 或 set operator。

因此，当前相对 local baseline 的 25%--99% 增益是 POC 证据，不自动等于下一阶段相对 strong learned
baseline 的增益。

### M3. Reducible-gap closure

设 `E0` 为 no-context train-mean error，`Efull` 为冻结 strong reference 在全部非目标 sources 下的误差：

$$
C_d(K) =
\frac{E_{0,d}-E_d(K)}{E_{0,d}-E_{\mathrm{full},d}}.
$$

它回答“当前 K 条 probes 关闭了多少 full-context 可约减误差”。`C=0` 等于完全没利用 context，`C=1`
达到 full-context reference。若 `E0-Efull` 小于 `0.05 E0`，该域没有足够 headroom，指标不使用，也不能
通过极小分母制造大数。

这个指标比直接比较三个域的 raw error 更公平：Ocean 的绝对误差可以极低，OpenFWI 则更难，但二者都能
比较 context information 被利用了多少。

### M4. Normalized budget-error AUC

在共同预算轴 `rho` 上，对 `E(rho)/E0` 做梯形积分：

$$
A_d = \int_0^1 \frac{E_d(\rho)}{E_{0,d}} d\rho.
$$

实际只在离散预算点上线性连接。AUC 越小越好，它避免只挑一个最好看的 K。论文中同时报告相对强基线的
AUC reduction：

$$
G_d^{AUC} = 1 - \frac{A_{d,\mathrm{ours}}}{A_{d,b}}.
$$

### M5. Equivalent probe saving

给定 baseline 的 full-budget error，寻找 ours 达到同等或更低误差的最小 K：

$$
K_{eq} = \min\{K:E_{ours}(K) \leq E_b(S-1)\}.
$$

测量节省比例为：

$$
Q_d = 1 - \frac{K_{eq}}{S-1}.
$$

这是最适合摘要和应用叙事的量：例如“达到 full-sweep baseline 精度时少测 50% sources”。若没有任何 K
达到 baseline full-budget error，则明确报告未达到，而不是外插。

## 4. 跨数据集汇总方式

三个域不能把全部 systems 或 queries 混在一起做 micro-average。建议正文同时报告：

1. 每域的 `G_d`、`C_d`、AUC 和 probe saving；
2. 通过 gate 的域数量，例如 `2/3`；
3. worst-domain gain；
4. error ratio 的 domain-macro geometric mean：

$$
R_{geo} =
\exp\left(
\frac{1}{D}\sum_{d=1}^{D}
\log\frac{E_{d,ours}}{E_{d,b}}
\right).
$$

`1-Rgeo` 可作为一个跨域摘要数，但不能代替逐域表格。只有三个数据域时，不对“数据域总体”做虚假的
显著性检验；统计不确定性仍在每个域内以 physical systems 为单位。

## 5. 必须保留的诊断指标

这些指标解释 raw error，但不参与主排名：

- amplitude relative L2；
- target-amplitude-weighted phase coherence；
- target-amplitude-weighted circular phase MAE；
- co-located sub-array reciprocity residual；
- train response spectrum 的 effective rank 与 rank--error curve；
- OOD degradation factor `E_OOD/E_IID`。

特别关注一个已经出现的事实：Deepwave K=1 时，subspace 的 amplitude error 没有优于 train mean，但 phase
MAE 明显下降。这说明 raw complex 增益确实来自 context-conditioned phase/system identification。

## 6. 下一阶段建议冻结的 promotion gate

以下 gate 比当前 POC 更严格，适用于“是否把新模型提升为论文主方法”：

### Gate A：half-budget strong-baseline gain

- 至少 `2/3` 多系统域相对最强同预算基线降低至少 10%；
- 对应 paired-system bootstrap 95% 下界大于 0；
- system win rate 至少 70%；
- 未通过的域不能退化超过 3%；
- domain geometric error ratio `Rgeo` 不高于 0.90。

### Gate B：measurement efficiency

满足下面至少一项：

- half-budget error 不高于 strong baseline full-budget error 的 1.05 倍；
- 达到 baseline full-budget error 时，至少节省 25% source probes；
- reducible-gap closure 在至少两个域达到 75%。

### Gate C：完整预算曲线

- normalized AUC 相对 strong baseline 至少降低 8%，至少两个域成立；
- 不能只在单个 K 上获益；
- sparse、half 和 full 三段中不得出现超过 3% 的系统性退化。

### Gate D：OOD retention

- Deepwave geometry/contrast OOD 与 Ocean parameter OOD 都必须报告；
- 至少一个 OOD 域相对 strong baseline 改善 5% 以上，另一个不退化超过 3%；
- 同时报告 `E_OOD/E_IID`，防止 IID 很强但 OOD 绝对误差失控。

### Gate E：三 seed 与复杂度

- 三个训练 seed 同向；
- 参数量、训练步数、validation 次数与 generic neural baseline 匹配；
- 额外参数超过 baseline 两倍时，必须额外报告 accuracy--compute Pareto，而不能只比较误差。

## 7. 我建议的正文组织

### 主表

half-budget 下每域一行：raw RelL2、strong-baseline gain、gap closure、win rate、bootstrap interval、OOD。

### 主图

横轴统一为 `rho=K/(S-1)`，纵轴为 `E/E0`；三个 panel 分别显示三个数据域，方法与强基线使用相同颜色。

### 应用表

报告 equivalent probe saving 和达到固定 error threshold 所需 K，不把模型参数量混入物理精度主表。

### 归因表

只在附表报告 amplitude error、phase coherence、phase MAE、reciprocity 与 effective rank。

## 8. 下一步最小执行顺序

1. 冻结三个域的 system split、target/context episodes 与归一化预算轴；
2. 补齐 retrieval、direct ridge 与参数匹配 masked Transformer；
3. 先计算它们与当前 subspace/M1 的 validation headroom；
4. 只有 strong baseline 仍留下至少 10% headroom，才设计下一版 query representation；
5. 方法选择冻结后一次性运行 test 与 OOD，并生成 M1--M5；
6. 若 strong baseline 已吃掉主要增益，则把当前结论定位为“低维 response state 的科学发现”，不强行发明更复杂模型。

## 9. 当前推荐的核心数字

如果现在必须只选三个跨数据集指标，我会选：

1. **half-budget strong-baseline normalized gain `G_d`**：回答是否真正优于强方法；
2. **reducible-gap closure `C_d`**：回答少量 probes 利用了多少可用系统信息；
3. **equivalent probe saving `Q_d` 或 normalized AUC**：回答实际少测了多少，以及优势是否覆盖完整预算曲线。

raw complex RelL2 仍必须与这三个数字并列展示，而不是被归一化指标取代。

更精确地说，论文评价应当采用“一个底座、三个主量、两个约束”：

- 底座：每个域的 raw complex RelL2；
- 三个主量：`G_d`、`C_d`、`Q_d`；若预算曲线足够密，用 AUC 替代单点 `Q_d`；
- 两个约束：OOD retention 与 accuracy--compute Pareto。

amplitude/phase 分解只负责解释机制，不负责决定胜负。
