# PBNO / Wave as Prompt：大白话故事与实验说明

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 更新日期：2026-10-03
> 本文解释 PBNO 的基础。项目已经进一步推进到 target-aware interrogation；完整的新故事、M1 结果与未来
> 三条路线见 [当前故事、证据与未来路线图](CURRENT_STORY_AND_ROADMAP_ZH.md)。
>
> 原文更新日期：2026-09-26
> 用途：给第一次接触项目的 AI 研究者、合作者和未来 Agent 阅读。公式与完整数字以
> `reports/PBNO_M4_RISK_RESOLUTION_ZH.md` 为准。统一 composer 的最新边界见
> `reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md`：统一和严格 fallback 成立，但 learned correction 未通过。

## 1. 一句话在做什么

我们不是让 AI 把一张低频波场直接修成一张高频波场，而是：

> 给 AI 看一个未知物理系统的几次波动实验，让它现场理解这个系统，然后预测换一个 source、换一个
> frequency 做新实验时，会测到怎样的复数响应。

这个任务可以叫 **Wave as Prompt**。每次已做实验都是一个 physical prompt，新 source--frequency 组合是
一个尚未执行的 query。

## 2. 一个直观例子

假设面前有一块内部结构未知的材料。我们不知道它内部的波速、缺陷或几何，但可以：

1. 在位置 A 用频率 1 激励一次，记录一圈传感器的复数响应；
2. 在位置 B 用频率 2 再激励一次；
3. 把这些响应交给模型；
4. 问模型：如果在从未激励过的位置 C、用从未测过的频率 3，会得到什么响应？

海洋水声里，同一件事可以理解为：只做少量 source--frequency 探测，在不知道完整声速场的情况下，预测
新的发射位置和频率下的接收阵列响应。

形式上，一次实验是：

$$
q_i=(s_i,f_i),
$$

$$
q_i \longrightarrow y_i.
$$

$s_i$ 是 source，$f_i$ 是 frequency，$y_i$ 是所有 receivers 上的复数响应。给模型 K 个 prompts 后，要求
它回答新 query $q_*$：

$$
D_K=\{(q_1,y_1),\ldots,(q_K,y_K)\},
$$

$$
D_K+q_* \longrightarrow \widehat y_*.
$$

## 3. 和原始 APEX 的关系

原始工作的出发点仍然成立：

> 频率变化后，能量或幅值结构通常变化得相对平缓，但相位会因为传播距离、干涉、共振和多路径快速变化。

因此，直接把低频相位搬到高频通常不可靠。原始 APEX 用较明确的物理相位先验解决它，但容易依赖已知
source、近似传播路径或解析结构。

现在的方案换了一个更一般的角度：

- 不认为某一张低频场经过固定变换就等于高频场；
- 认为不同 source、frequency 下的响应都来自同一个隐藏物理系统；
- 不直接复制相位，而是用幅值和相位共同识别这个隐藏系统；
- 识别系统后，再针对新 query 重新产生复响应。

所以更准确的科学判断是：

> 相位很难直接迁移，但它仍是识别隐藏 response operator 的重要信息。

## 4. 方法第一步：学习一套“波动响应零件”

训练数据包含许多不同物理系统。每个系统都有一张完整 response table，覆盖 source、frequency 和 receiver。
我们在许多训练系统之间学习一个共享复数基底：

$$
x_j \approx \mu+Bz_j.
$$

可以把它理解成：

- $\mu$ 是平均响应；
- $B$ 是一套反复出现的“典型响应零件”；
- $z_j$ 表示第 j 个系统用了多少个每种零件。

此前的 functional-rank 审计表明，稳定低秩主要存在于“不同物理系统构成的 response manifold”，不一定
存在于 source、receiver 等每个单独坐标轴。因此这里不是强行对所有物理轴做张量低秩分解。

## 5. 方法第二步：从少量实验反推系统 token

测试时只看到 K 个 prompts，相当于只看到完整 response table 的很少一部分：

$$
y_O=\mu_O+B_Oz+e.
$$

模型利用训练系统中学到的 latent spectrum 作为先验，计算 $z$ 的 Bayesian posterior。大白话说：

> 模型根据几次实验，推断这个未知系统用了哪些“响应零件”，同时知道哪些判断已经确定、哪些仍不确定。

posterior mean 与 posterior uncertainty 一起构成 system token。它不需要在新系统上更新网络参数，K 也不必
固定。

最简单的 query prediction 是：

$$
\widehat y_*=\mu_*+B_*\widehat z.
$$

这就是 global Bayesian operator，也是整个方法的严格线性起点。

## 6. 两种互补的非线性升级

### 6.1 Neural latent transport

一个轻量网络根据 posterior token、uncertainty 和目标 query，修正 latent coefficient，再解码复响应。

直观上，它学习：

> 在这种 source--frequency query 下，哪些系统特征应该被非线性地加强或减弱？

它在 Deepwave 超声数据上最好。

### 6.2 Posterior-kernel operator memory

另一条路线保留训练系统形成的经验先验。对于每个训练系统，使用与测试系统完全相同的 prompt mask 重新计算
posterior token，然后找到在当前观测条件下最相似的训练系统，借用它们对目标 query 的响应残差。

它不是直接查找完整场，也不是普通 nearest neighbor。关键要求是：

> memory key 必须根据本次实际给出的 prompts 重新计算，不能使用预先固定的完整系统编码。

它在 OpenFWI 和 Ocean 上最好。

## 7. 为什么最后还需要一个小 adapter

三个域偏好的 prior 不同：Deepwave 偏 neural transport，OpenFWI/Ocean 偏 posterior kernel。我们又不希望
部署时手工告诉模型数据集名字。

第一版 hard router 尝试用 leave-one-prompt-out reconstruction error 选择分支，但失败了。原因是：

> 能重建一个已经测过的 prompt，不代表能预测 source 和 frequency 都没见过的新 query。

最后成功的是连续融合：

$$
\widehat u=(1-g(D_K))\widehat u_{kernel}+g(D_K)\widehat u_{neural}.
$$

$g$ 是一个 1,665 参数的小网络，只看 prompts、分支之间的 disagreement 和可用的 leave-one-out 统计，不看
domain ID 或 test truth。它学习的是两个有偏复数预测何时可以互相抵消误差，而不是硬猜哪个分支获胜。

这个 adapter 解决的是统一部署，不是论文最主要的科学创新。核心仍然是 Bayesian system token 和
posterior-conditioned operator prior。

## 8. 为什么 memory 不等于“把训练数据全存起来”

直接删掉一半训练 prototypes 会明显掉点，因为局部经验 prior 需要训练分布密度。成功方案保留训练系统身份，
但把每个完整 response table 压成共享 basis 上的 16 个复系数。

- OpenFWI value bank 变成原来的 0.65%；
- Ocean value bank 变成原来的 0.93%；
- 计入共享 basis 后，总 operator storage 仍减少约 79%/66%；
- 四个 prompt budgets 下 test error 退化不超过 1.48%。

因此更合适的叫法是 **spectral operator dictionary**，而不是 raw training-response memory。

## 9. 实验是怎么考模型的

所有数据都整理为：

$$
system \times source \times frequency \times receiver.
$$

一次 strict-Q3 test episode：

1. 从一个从未参与训练的 physical system 取 K 个 prompts；
2. K 取 1、2、4、8；
3. target source 从未出现在 prompts 中；
4. target frequency 也从未出现在 prompts 中；
5. 模型预测该 target query 的完整 receiver 复响应。

所有数据按 physical system 划分 train、validation、test，绝不把同一环境的不同 source/frequency views
拆到不同 split。否则模型可能只是在记住同一个 Green operator。

## 10. 为什么主指标仍是 raw complex relative L2

主指标是：

$$
E=\frac{\|\widehat u-u\|_2}{\|u\|_2}.
$$

它同时惩罚幅值、相位、real/imaginary part 和干涉条纹位置错误。幅值与 phase/coherence 指标只用于诊断，
不替代主结论。统计时先按 physical system 聚合，避免拥有更多 queries 的系统支配平均值。

## 11. 数据域和当前结果

| 数据域 | 代表场景 | K=2 统一 adapter | 相对 global baseline 改善 |
|---|---|---:|---:|
| OpenFWI | 地震/地下介质 | 0.14430 | 17.25% |
| Deepwave | 环形超声阵列 | 0.09137 | 6.23% |
| Ocean | 海洋多模态声传播 | 0.47704 | 21.85% |

三组独立 prompt seeds、三个域、K=1/2/4/8 共 12 个设置中，统一 adapter 12/12 都优于该设置下事后最好的
单分支。它相对最佳分支的额外收益约 0.2%--2.6%，说明 adapter 的主要价值是稳定统一，而不是制造全部增益。

第四个 structural lattice 域使用独立的阻尼 mass--spring frequency-domain solver。posterior kernel 在 K=1/2
分别把 error 从 0.51577/0.48300 降到 0.46261/0.44348，改善 10.31%/8.18%。它验证跨物理机制，但尚未加入
三域 adapter calibration。

## 12. 最准确的三项贡献

1. **新问题**：把未知 wave system 的少量复响应当作 prompts，回答尚未执行的 joint source--frequency
   query。
2. **新表示**：用复 response manifold 的谱先验，把任意数量 prompts 转成带 uncertainty 的 Bayesian
   system token。
3. **新 operator prior**：在线性 Bayesian operator 上组合 neural latent transport、posterior-conditioned
   empirical operator prior 和可压缩 spectral dictionary。

## 13. 目前不能声称什么

- 没有解决任意远距离频率延拓；
- 不是跨任意 PDE 的 wave foundation model；
- adapter 尚未证明对完全未参与 calibration 的新物理域 zero-shot 泛化；
- Ocean noisy likelihood 和较高 absolute error 仍未解决；
- flow matching 尚无超过 15% 的 conditional oracle gap，因此没有证据说明它现在是必要组件；
- structural lattice 是可复现 synthetic benchmark，不是真实实验数据。

## 14. 最后再用一句话总结

> 频率变化使相位难以直接搬运，所以我们不再从一张低频场“修出”高频场；我们让 AI 通过少量复数波动实验
> 识别隐藏 response operator，再预测下一次尚未执行的 source--frequency 实验。
