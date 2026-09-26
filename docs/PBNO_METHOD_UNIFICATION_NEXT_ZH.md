# 从 PBNO 组件栈到统一的现代 In-Context Operator

> 日期：2026-09-26  
> 状态：方法设计备忘录，不代表已经取得实验结果  
> 目标：在不牺牲现有正结果和物理可解释性的前提下，把 Bayesian posterior、neural transport、operator
> dictionary 与 query decoder 收进一个统一、现代、可扩展的学习架构。

## 0. 先给判断

当前 PBNO 是一个有科学内容的强 POC，但从方法论文的视觉和概念结构看，确实偏零碎：

1. global Bayesian posterior 负责稳定 identification；
2. neural latent transport 负责 Deepwave；
3. posterior kernel 负责 OpenFWI/Ocean；
4. spectral dictionary 负责压缩；
5. 最后再用一个小 adapter 融合两个分支。

每个部件都有实验理由，但读者可能会问：

> 这是一个统一模型，还是若干有效技巧的组合？

只把其中某个 MLP 换成 Transformer，或者再加入 FNO/flow，并不能回答这个问题。真正值得做的升级是：

> 用一个 posterior-initialized、retrieval-augmented、query-conditioned latent operator，把现有所有正分支写成
> 同一计算图的不同退化情形。

本文暂用 **Posterior-Conditioned Operator Transformer** 作为架构描述，不建议在实验通过前把它固定成论文名。

## 1. 统一模型应该保留什么

现有证据要求下一版保留四个东西。

### 1.1 解析 posterior 不能丢

它是目前最稳定的 system identification 起点，天然支持 variable K，并提供 uncertainty。完全换成黑盒 set
encoder 会失去严格 PCA--MAP 特例，也可能重走此前 set-encoder 不稳定的弯路。

### 1.2 经验 operator dictionary 不能退回 raw memory

OpenFWI/Ocean 的正结果说明局部经验先验有用；rank-16 compression 又说明 memory 应以低维 operator tokens
存在，而不是保存整张训练 response table。

### 1.3 Query 必须是一等公民

模型不应先生成整张固定网格场，再从中裁出答案。source、frequency、receiver coordinate 应直接作为 query
tokens，模型输出所问坐标上的函数值。

### 1.4 复波动物理不能只当双通道图片

模型至少应满足一个朴素规则：如果所有输入响应同时乘同一个 global phase，预测也应乘同样的 phase。无需把
论文写成群论工作，但应避免网络把任意 phase reference 当作物理差异。

## 2. 推荐的统一计算图

整体流程不是简单线性流水线，更直观的计算图是：

```text
response prompts ──> analytic posterior mean / uncertainty ──> latent operator tokens
                                                                  │
rank-16 spectral dictionary ───── retrieval cross-attention ──────┤
                                                                  ▼
target source / frequency ─────── query-conditioned token update ──> response decoder
                                                                              │
receiver coordinates ─────────────────────────────────────────────────────────┘
```

### 2.1 Functional prompt tokenizer

每个 prompt 不是一张图片，而是一项带坐标的复函数观测：

$$
(s_i,f_i,\{r_m,y_i(r_m)\}_{m=1}^{M}).
$$

第一步用共享 receiver-function encoder 把它编码为少量 prompt tokens。第一版直接复用现有 global spectral
basis；后续才考虑 Galerkin attention、局部 neural field 或 point-cloud operator encoder。

这样做的好处是：

- receiver 数量和采样位置可以变化；
- 未来从一维阵列扩展到二维面阵、三维场时，不必把通道数固定死；
- “函数上的 operator”而不是“图片 backbone”成为方法的基本单位。

### 2.2 Analytic posterior tokens

解析 Bayesian solve 给出 posterior mean $m$ 与 covariance $C$。不再把它们压成单个 flat vector，而是按
spectral direction 构成一组 latent tokens：

$$
t_l=(m_l,\log C_l,\lambda_l,e_l).
$$

$\lambda_l$ 是训练 response spectrum，$e_l$ 是第 l 个 basis/operator mode 的 embedding。uncertainty 直接
进入 token，让 attention 知道哪些 latent directions 已被 prompts 确认，哪些仍应依赖训练先验。

### 2.3 Spectral dictionary as retrieval tokens

每个训练系统目前已经被压成 16 个 complex coefficients。下一版把这些 coefficients 视为 operator memory
tokens，而不是在模型外部先做一次固定 kNN。

posterior tokens 向 dictionary 做 cross-attention：

$$
T' = T + A(T,M,C),
$$

其中 $T$ 是当前系统 tokens，$M$ 是压缩 operator dictionary。attention distance 由 posterior uncertainty
whiten：不确定的 latent direction 不应被当作可靠检索依据。

这一步把 posterior kernel 变成标准的 retrieval-augmented attention，而不是一个独立分支。

### 2.4 Neural transport as token update

现有 neural latent transport 改写为每层 attention 后的 token-wise nonlinear update：

$$
T''=T'+F(T',q).
$$

$F$ 是共享 feed-forward/operator block，目标 query $q$ 通过 conditioning 进入。这样 neural transport 和
memory prior 不再先各自预测完整场、最后在 output space 做凸组合，而是在同一 latent operator state 上迭代
更新。

### 2.5 Continuous query decoder

目标 source、frequency 和 receiver coordinate 形成 query tokens，向 system tokens 做 cross-attention：

$$
\widehat u(s_*,f_*,r_*)=D(T'',s_*,f_*,r_*).
$$

第一版 decoder 输出现有 spectral basis coefficients，保持计算便宜；第二版才升级为 complex neural field 或
Galerkin operator decoder，以支持 variable receivers、2D/3D coordinates 和跨分辨率查询。

## 3. 为什么它能真正统一当前方法

推荐架构不是把旧方法全部推倒。相反，旧方法可以成为它的清楚特例：

| 设置 | 得到的现有方法 |
|---|---|
| memory attention 和 nonlinear update 都置零 | global PCA--MAP / Bayesian operator |
| memory attention 置零，只保留 token update | neural latent transport |
| nonlinear update 置零，attention 固定为 kernel weights | posterior-kernel operator |
| 只允许一个全局 mixing token | 当前 1,665 参数 continuous adapter |

因此论文可以从“多个分支加 router”改写成：

> 一个 posterior-initialized in-context operator；解析 posterior 提供稳定初值，retrieval attention 注入局部
> empirical prior，neural token update 学习超出线性/核回归的 operator transport。

这比包装成 mixture of experts 更强，因为 MoE 仍然保留“先有多个完整专家、再路由”的零碎感。

## 4. 物理先验应该放在哪里

下一版不建议叠加多个 physics losses，而建议把物理放进表示和计算结构。

### 4.1 Global-phase consistency

所有 prompts 同时旋转相同 phase 时，latent token 与输出应以一致方式旋转。可以通过 complex-linear projection
和只依赖复内积的 attention score 实现，不需要在正文强调抽象群论。

### 4.2 Uncertainty-weighted assimilation

posterior covariance 决定 attention metric 和 residual step size：已被 prompts 精确识别的方向少改，不确定
方向更多依赖 dictionary/neural update。这比事后手写 uncertainty gate 更统一。

### 4.3 Source--receiver reciprocity 只作可用时的结构约束

当数据的 source/receiver 定义确实满足 reciprocity 时，可共享 coordinate embedding 或对称 decoder；不把它
设为跨所有数据域的强假设。

### 4.4 Frequency structure先从坐标编码进入

第一阶段使用连续 frequency encoding 与相对频率差。pole/resolvent 或 rational spectral layer 有物理吸引力，
但当前连续谱拟合曾经失败，不应在统一架构 POC 前重新成为必需组件。

## 5. 训练目标也应该统一

不建议再堆 reconstruction loss、router loss、amplitude loss、phase loss 和多个 regularizers。每个 episode 只做：

1. 从一个 physical system 采样 context prompts；
2. 采样与 context 不重叠的 target queries；
3. 最小化 target complex response 的一个 likelihood。

若先做 deterministic 版本，使用当前 system-balanced raw complex error。若输出 variance，则使用一个 proper
complex Gaussian NLL；它同时训练 mean 与 uncertainty，不再另加 calibration loss。

flow matching 只在同一条件下的 target residual 存在超过 15% best-of-N oracle gap 时，替换最后的
probabilistic decoder。它不是统一架构第一阶段的装饰组件。

## 6. 最小可行升级：不要一步到位

### M0：Latent composer POC

完全复用现有 datasets、splits、Bayesian basis、rank-16 dictionary 和 checkpoints。只训练一个 2--3 层、低于
1M 参数的 latent token composer：

- 输入 posterior mean/variance tokens；
- memory coefficients 作为 cross-attention keys/values；
- target source/frequency 作为 query condition；
- 输出 spectral residual coefficients；
- receiver decoder仍用现有 basis。

这一步不加入可学习 field encoder、不加入 flow、不加入额外 physics loss。

必须比较：

1. global Bayesian operator；
2. validation-selected best single branch；
3. 当前 continuous adapter；
4. 参数量匹配、没有 memory tokens 的 latent MLP/Transformer；
5. 固定 kernel attention，不训练 attention score 的退化版本。

建议 promotion gate：

- OpenFWI/Deepwave/Ocean 的 K=1/2/4/8 均不比当前 adapter 退化超过 1%；
- 至少两个域 K=2 相对当前 adapter 再降低 3%；
- 三个 prompt seeds 同向，paired bootstrap 下界为正；
- 参数少于 1M，latency 不超过当前 posterior kernel 的 1.5 倍；
- dictionary compression 保持，不恢复 full response memory。

若 M0 失败，应保留当前 PBNO，不通过增加层数或辅助 loss 强行救活。

### M1：Continuous functional decoder

只有 M0 通过后，才把固定 response basis decoder 替换为 coordinate-conditioned operator decoder，并测试：

- held-out receiver coordinates；
- receiver subsampling；
- 分辨率变化；
- 2D/3D receiver geometry。

这一步是把“现代 Transformer”升级为真正 function-space operator 的关键，不应与 M0 同时做。

### M2：Unseen-domain adaptation

冻结在 OpenFWI/Deepwave/Ocean 训练的 composer，在 structural lattice 或第五个新域上测试：

- 完全 zero-shot；
- 只校准 normalization；
- 少量 systems 做 parameter-efficient adaptation。

它决定能否使用 cross-physics 或 foundation-model 语言。没有这一阶段，正确说法仍是 multi-domain shared
adapter。

### M3：Probabilistic residual process

只有新的 ambiguity audit 通过，再比较：

- heteroscedastic complex Gaussian head；
- operator flow matching；
- conditional flow/diffusion decoder。

评价必须包含 mean error、NLL、coverage、CRPS 和 sample-oracle，不能只展示更丰富的 samples。

## 7. Functional tensor、FNO 和 flow 应该怎么放

### Functional tensor

适合用于压缩 dictionary、factorize query decoder 或共享不同坐标轴的 basis，不适合再次被当成“完整复场在
每个物理轴上都低秩”的核心假设。现有负结果已经否定后者。

### FNO / neural operator

FNO 的思想应该体现在 function-space encoder/decoder、离散化共享和连续坐标查询上，而不是简单把稀疏 prompts
栅格化后交给一个普通 FNO。当前输入是不规则 context set，Perceiver/Galerkin-style cross-attention 更自然。

### Flow matching

Operator flow matching 很适合未来建模 function-valued posterior，但只有任务确实存在条件多模态时才必要。当前
oracle 证据不足，过早加入会让 reviewer 认为只是追逐生成模型热点。

## 8. 和现代相关工作的关系

| 工作 | 可借鉴之处 | 我们仍然不同的地方 |
|---|---|---|
| [Neural Operator](https://arxiv.org/abs/2108.08481) | function-space map 与离散化共享 | 我们不读取完整 PDE coefficient，而从 response prompts 识别隐藏系统 |
| [GNOT](https://arxiv.org/abs/2302.14376) | heterogeneous inputs、irregular mesh、operator cross-attention | 我们有解析 Bayesian posterior 和复 operator dictionary |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | latent bottleneck、任意输入输出 query | 需加入 wave-specific complex/uncertainty structure |
| [Generalizable Neural Fields as Partially Observed Neural Processes](https://arxiv.org/abs/2309.06660) | 从部分观测形成 neural-process latent，再回答坐标 query | 我们关注隐藏 response operator 与 joint excitation query |
| [AROMA](https://arxiv.org/abs/2406.02176) | local neural fields、irregular grids、latent tokens | 它主要建模时间动力学，我们建模多 source--frequency resolvent responses |
| [CoDA-NO](https://arxiv.org/abs/2403.12553) | 把函数作为 attention tokens、跨多物理预训练 | 我们的 tokens 来自少量实验 posterior，而非已知完整输入函数 |
| [Zebra](https://arxiv.org/abs/2410.03437) | inference-time context adaptation 与生成式 PDE pretraining | 我们的系统隐藏且只有 sparse complex responses |
| [VICON](https://arxiv.org/abs/2411.16063) | 高维 patch/token 化的 in-context operator | wave prompts、复相位和 source--frequency query 是不同任务结构 |
| [Structure-Preserving Operator Learning](https://arxiv.org/abs/2410.01065) | 把连续空间结构放入 encoder/decoder | 可指导未来几何和边界保持，不必现在改成 FE 框架 |
| [Operator Flow Matching](https://arxiv.org/abs/2501.04126) | function-space stochastic process posterior | 只在 conditional ambiguity gate 通过后进入 |

## 9. 如果 M0 成功，论文方法故事会如何变得简洁

现在的写法是：

> Bayesian posterior + neural branch + kernel branch + compression + adapter。

升级后的写法可以变成：

> 我们提出一个 posterior-initialized retrieval-augmented in-context neural operator。解析 posterior 将 sparse
> complex prompts 转为 uncertainty-aware operator tokens；这些 tokens 从压缩 spectral dictionary 中检索并
> 经 query-conditioned nonlinear transport 更新；continuous decoder 直接回答新的 excitation query。

这条故事具有四个清楚的 AI 关键词，但不是为了包装硬贴：

- in-context learning；
- latent operator tokens；
- retrieval-augmented attention；
- continuous function query。

同时它保留了一个强物理中心：所有学习都围绕“从少量复响应识别隐藏 wave resolvent/operator”展开，而不是把
波场当成普通图片。

## 10. 最终建议

短期最值得实现的是 M0 latent composer，不是端到端大模型。它可以复用当前所有数据、posterior features、
dictionary 和 checkpoints，最快回答一个关键问题：

> 现有 kernel 与 neural 正信号，能否在 latent operator space 内被一个共享 attention mechanism 真正统一，
> 并显著超过现在只在 output space 做标量融合的 adapter？

如果答案是肯定的，我们才有充分理由继续做 continuous neural-field decoder、跨域预训练和 probabilistic
operator process；如果答案是否定的，当前 PBNO 仍然是一套诚实、有效但更偏 hybrid Bayesian-learning 的方法，
不应为了显得 modern 而增加复杂度。
