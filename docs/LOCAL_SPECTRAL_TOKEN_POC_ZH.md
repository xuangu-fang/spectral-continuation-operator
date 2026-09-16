# 局部复谱 Token POC：技术设计与冻结协议

日期：2026-09-16  
状态：实现前冻结的 development POC；不触碰 strict environments  
结果：已完成，method-level NO-GO；见
[`LOCAL_SPECTRAL_TOKEN_POC_ZH.md`](../reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md)

## 1. 一句话目标

学习一个局部、相位等变的复谱表示，使低频场在每个空间 patch 内形成可查询的 basis；
用未见频率查询验证表示，masked reconstruction 仅作为辅助学习信号。

本文档固定三层关系：

- **研究对象**：局部复谱 representation；
- **主要任务**：从 90–100 Hz 完整低频带查询未见频率复场；
- **辅助任务**：遮挡可见频率或局部 frequency–patch token 后重建。

空间随机 mask 不是主任务，高频查询处仍然没有任何目标场观测。

## 2. 为什么现在做局部 token

最新 oracle 诊断在 Ocean 110 Hz 上给出：

| 表示 | Complex RelL2 |
|---|---:|
| 全局 raw low-band span，rank 11 | 0.911 |
| 全局 gauged low-band span，rank 11 | 0.605 |
| raw 4×8 patch span，rank 11/patch | 0.196 |
| raw 8×16 patch span，rank 11/patch | 0.052 |
| gauged full-band basis，rank 11 | 0.077 |

因此首要问题不是继续扩大一个全局 latent，而是判断：能否从训练环境中学到局部 span 的
查询系数。patch oracle 使用目标真值拟合系数，当前 POC 则必须只使用低频场与查询频率预测系数。

## 3. 主任务与数据协议

输入与输出为：

$$
\big\{u(f_j): f_j=90,\ldots,100\big\},\; f_q
\quad\longrightarrow\quad
\widehat u(f_q).
$$

冻结协议：

- 128 个 development training environments；
- 32 个 development validation environments；
- 可见频率：90–100 Hz；
- 训练与 checkpoint 选择频率：101–106 Hz；
- 主要未见频率：107–112 Hz；
- horizon stress test：113–125 Hz；
- 主查询：110 Hz；
- 主指标：逐环境等权 raw complex relative L2；
- 机制指标：amplitude relative L2、phase coherence、circular phase error；
- strict environments 在模型与超参数冻结前不读取。

## 4. 最小方法：Local Span Query Operator

### 4.1 Patch 划分

把 $48\times384$ 场划分为固定 $P_h\times P_w$ token 网格。第一轮只比较：

- 4×8：较低容量；
- 8×16：oracle 更强但系数更多。

每个 token 对应一个空间 patch，并共享同一 encoder 和 query head。

### 4.2 相位不变的 token 特征

对每个 patch、每个可见频率计算并池化：

1. 幅值 $|u(f_j)|$；
2. 相邻频率的归一化复相关

$$
c_j=
\frac{2u(f_{j+1})u(f_j)^*}
{|u(f_{j+1})|^2+|u(f_j)|^2+\epsilon};
$$

3. 每个频率相对 100 Hz endpoint 的归一化复相关；
4. frequency–patch observation mask；
5. patch 坐标和可用的环境通道。

这些特征对所有输入同时乘以全局复相位 $e^{i\theta}$ 保持不变。

### 4.3 查询系数与解码

共享 token mixer 先产生一个与查询无关的局部谱状态。为避免非线性 query MLP 在训练频带外
任意延拓，查询系数被限制为有界二阶连续演化，其中归一化偏移为

$$
\delta=(f_q-100)/10
$$

token 状态输出每个可见 basis 的复 slope 与 curvature，并通过

$$
a_{p,j}(f_q)=e_{p,j}+\delta s_{p,j}+\frac{1}{2}\delta^2c_{p,j}
$$

得到查询系数，其中 $e_{p,j}$ 只在 100 Hz endpoint 上为 1，$s$ 与 $c$ 均由双曲正切限制。
将系数平滑上采样回原网格：

$$
\widehat u(x,f_q)
=
\sum_{j=1}^{11} a_j(x,f_q)u(x,f_j).
$$

该 decoder 直接对应 patch-span oracle，但系数由低频数据预测而不是使用目标真值拟合。
由于 token 特征相位不变、输出是输入复场的复线性组合，整个模型满足：

$$
\widehat u(e^{i\theta}u,f_q)
=e^{i\theta}\widehat u(u,f_q).
$$

第一版同时保留 raw 与 endpoint-gauged 两种坐标消融。gauged 版本先用 99–100 Hz 的复相位增量
延拓显式 carrier，token operator 只组合解调后的局部 residual，最后乘回查询 carrier；它仍保持
全局相位等变。第一版 token mixer 使用小型卷积 residual blocks，不使用 Transformer。只有局部表示产生正信号后，
才比较 attention 是否能更有效地关联远距离 patch。

### 4.4 容量披露

局部模型每个 query 产生 $2FP_hP_w$ 个实系数，明显多于单个全局 latent。论文和报告必须同时列出：

- 网络参数量；
- token 网格；
- 每个场的复系数数量 $FP_hP_w$；
- 与同参数量 image/global baseline 的结果。

不能把更细 token 的 oracle 误差下降直接写成更好的压缩率。

## 5. 辅助 mask 任务

主查询损失保持单一 raw complex relative L2。辅助项使用相同输出空间与相同误差，不增加幅值、
相位或频域 loss 堆叠。

### 5.1 Frequency mask

每个样本随机选择一个可见频率，将该频率所有 patch 隐藏，以剩余十个频率恢复完整被遮挡场。
它检验 token 是否学到带内谱联系。

### 5.2 Joint frequency–patch mask

随机选择一个可见目标频率，只隐藏其中一部分 patch；同时随机隐藏其他频率的少量 patch token。
loss 只在目标频率被隐藏的空间区域计算。该任务为跨样本 spatial prior 提供训练信号，但不改变
主任务的信息条件。

总目标为：

$$
L=L_{\mathrm{query}}+\lambda L_{\mathrm{mask}}.
$$

第一轮只取 $\lambda\in\{0,0.1,0.3\}$，不引入其他 loss。

## 6. 第一轮对照矩阵

先用 seed 42 做选择，禁止一次展开大量超参数搜索：

| ID | 网格 | 辅助 mask | 目的 |
|---|---|---|---|
| T0 | 4×8 | 无 | 局部 span query 最小基线 |
| T1 | 8×16 | 无 | 检验更细局部性 |
| T2 | 最佳网格 | frequency，$\lambda=0.1$ | 检验带内谱预训练信号 |
| T3 | 最佳网格 | joint，$\lambda=0.1$ | 检验空间 mask 是否帮助查询 |

T0/T1 首先各比较 raw 与 endpoint-gauged 坐标；后续 mask 实验只沿较优坐标继续。

若 T2/T3 没有超过 T0/T1，不增加 mask loss 权重或组件，而保留“辅助重建没有迁移到查询”的负结论。

现有冻结 baseline：

- hyper-pole：110 Hz `0.4473`；
- stable recurrence：110 Hz `0.4741`；
- carrier-relative query U-Net：110 Hz `0.6125`。

### 6.1 预注册的受限 fallback

若自由 local-span 系数在训练频率内可拟合、但频带外失稳，则只允许一个更受约束的 fallback：

- 从冻结的 hyper-pole checkpoint 精确初始化；
- 增加 8×16 局部 token，只输出 carrier slope 与 curvature 修正；
- 新 head 零初始化，因此 epoch 0 与原 checkpoint 完全相同；
- 第一轮冻结原 backbone，只训练新增 token branch；
- 不同时增加 residue、pole、loss 或 attention 组件。

该对照检验 local token 应放在“自由 basis 系数”还是“传播生成器”上。如果它仍不能改善 110 Hz，
本轮 POC 终止，不再通过扩大 token 网络搜索正结果。

## 7. Go/no-go

### 表示层 GO

满足以下两项：

1. 110 Hz RelL2 小于 0.40，并优于同协议 hyper-pole；
2. 107–112 Hz macro error 优于 hyper-pole，且 phase coherence 不下降。

理想目标为 110 Hz 小于 0.35，但它不是第一轮继续/停止的硬门槛。

### Mask transfer GO

最佳 masked variant 必须比相同网格、相同 query 模型的无 mask 版本改善至少 5%，否则不把
masked pretraining 放入主方法。

### No-go

出现任一情况即停止扩大该模型：

- 细 token 只改善 101–106 Hz，107–112 Hz 不改善；
- 查询误差仍高于 0.447；
- 输出接近 copy/插值，phase coherence 没有改善；
- 性能只能依靠 joint mask 中泄漏的目标频率可见 patch。

## 8. 正结果后的第二阶段

第一轮 GO 后才做：

1. seeds 42/43/44；
2. 冻结 tokenizer、只训练轻量 query probe；
3. 同参数量 global-token 与 image-token 对照；
4. 非整数频率 101.5–124.5 Hz；
5. 将同一 tokenizer 用于独立空间 random-mask completion；
6. 最后才比较 attention、state-space mixer 或 conditional generator。

这样可以分别回答“局部表示是否有效”“masked prior 是否迁移”“更强 mixer 是否必要”，避免把多个
组件同时加入后无法归因。
