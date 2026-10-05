# PBNO M2：效率正结果、receiver residual 边界与下一步判断

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](../docs/wave_as_prompt/README.md)。

> 日期：2026-09-25  
> 证据等级：Q2 efficiency 为三组 prompt-order seed 的 A 级结果；M2 neural 与 memory 为单 seed development evidence  
> 一句话结论：Q2 已通过“近等精度、更低解析推断代价”的替代 gate；Q3 的神经收益仍未达到两个域 10%，不再堆 receiver encoder。

## 1. 本轮为什么没有继续加大网络

上一阶段 PBNO 在严格 Q3 上已经证明：解析 Bayesian posterior 加一个小型 nonlinear latent transport，能在
OpenFWI K=1/2 改善 11.7%/9.9%，在 Deepwave 改善约 7%。剩余问题不是“网络能否拟合训练集”，而是
新增容量能否在未见 physical systems 上关闭可约误差。

本轮按顺序只检验三个假设：

1. Deepwave 的 receiver residual 失败是否只是复响应量纲约为 $10^{-7}$ 导致的数值问题；
2. response manifold 的 PCA 尾模态能否由前 128 个主坐标非线性预测；
3. 已成熟的 Q2 是否可在近等精度下减少 latent rank 与 posterior solve 代价。

## 2. Receiver residual：数值问题修好，但方法 headroom 不成立

### 2.1 修复

旧实现直接在 Bayesian normalization 中加入固定幅度残差。不同数据域的 response scale 差异很大，因此同一
残差上限在 OpenFWI 合理、在 Deepwave 却会爆炸。新实现只用训练集估计每个 response coordinate 的 RMS：

$$
\sigma_j = \sqrt{\frac{1}{N}\sum_n |y_{n,j}-\mu_j|^2}.
$$

context、baseline 与预测残差先变换到这个单位中，完成 correction 后再映回原始 complex response。最后一层
保持零初始化，因此 step 0 在任意数据量纲下严格等于 Bayesian baseline。新增单元测试覆盖了这一性质。

### 2.2 结果

尺度修复后的普通版本在 Deepwave K=1/2/4/8 为：

| K | scale-correct receiver residual |
|---:|---:|
| 1 | 0.11769 |
| 2 | 0.09081 |
| 4 | 0.06869 |
| 8 | 0.05354 |

进一步把训练预算偏向 K=1/2，并把 normalized residual bound 从 0.05 放宽到 0.20，得到：

| K | sparse-budget residual | 原 PBNO 附近 |
|---:|---:|---:|
| 1 | 0.11665 | 0.11747 |
| 2 | 0.09079 | 0.09077 |
| 4 | 0.06914 | 0.06893 |
| 8 | 0.05433 | 0.05389 |

预注册 K=1 通过线为 0.1138，实验没有通过。结论是：量纲失配解释了旧版不稳定，但不解释跨域增益不足；
receiver-local decoder 只在 OpenFWI 有单域正信号，停止继续扫描 residual bound、更多卷积层或 loss。

## 3. Principal-manifold decoder：漂亮但不泛化的负结果

我们检验了一个更现代的 nonlinear operator 假设：用完整 response table 得到前 128 个 oracle PCA
coefficients，再用 MLP 预测后 128 个 coefficients。这个审计对 inference 非常宽松，因为输入主坐标来自完整
目标系统，而不是 sparse prompts。

| 数据域 | rank-128 reconstruction | nonlinear tail prediction | rank-256 oracle |
|---|---:|---:|---:|
| Deepwave validation | 0.02939 | 0.03002 | 0.02262 |
| Deepwave test | 0.02983 | 0.03046 | 0.02304 |
| OpenFWI validation | 0.05526 | 0.05797 | 0.01860 |
| OpenFWI test | 0.05704 | 0.05983 | 0.01945 |

训练 reconstruction 会下降，但未见系统上反而退化。说明 tail energy 确实存在，却不是由前 128 个系数稳定
决定的函数。因为 oracle 输入都不成立，本轮不实现 episodic principal-manifold decoder，也不把更多参数浪费在
同一假设上。

## 4. Posterior-latent memory：单域机制正结果

另一个只读审计把训练 systems 当作 response memory。对 query system 和每个训练 system 都使用相同 prompts
求 posterior latent，再在 whitened posterior space 检索近邻并回答 target query。整个选择过程只读 prompts，
不读 target truth。

OpenFWI 上，validation 选择的 8-neighbor memory 达到：

| K | global Bayesian operator | latent memory | 原 PBNO |
|---:|---:|---:|---:|
| 1 | 0.19636 | **0.15290** | 0.1575 附近 |
| 2 | 0.17494 | **0.14914** | 0.1529 附近 |

Deepwave K=1 仅从 0.12660 降到 0.12409，K=2 不改善；与 PBNO 组合也仍约为 0.11754。这个结果说明
OpenFWI 的地质 response family 含有可复用的局部原型，而 Deepwave 更像连续变化的 global manifold。它是
值得保留的 retrieval-augmented operator 机制，但尚不是跨域主方法。

## 5. Q2 efficiency：正式通过替代 gate

Q2 的 frozen rank-128 spectrum prior 已很强。预注册协议允许：若误差已接近可约下限，可在误差变化不超过
2% 时减少至少 25% 的 latent、storage 或 compute。

validation 只选择一次配置，然后在三组 prompt-order seeds 上固定评测：

| 数据域与预算 | rank 128 reference | efficient operator | 误差变化 | 解析 solve 量变化 |
|---|---:|---:|---:|---:|
| OpenFWI K=2 | 0.129223 | **0.128800**，rank 96 | -0.33% | -57.8% |
| Deepwave K=8 | **0.043490** | 0.043936，rank 112 | +1.03% | -33.0% |

OpenFWI 同时减少 25% basis storage；Deepwave storage 减少 12.5%，但 rank-cubic posterior solve proxy
减少 33%。Deepwave OOD 从 rank-128 的 0.075519 变为 0.075270，没有以 OOD 为代价。

在 A100 上，不计一次性 SVD fitting、包含整套 batched prediction 与 error reduction 的实测延迟为：

| 数据域 | rank 128 | efficient rank | 端到端降低 |
|---|---:|---:|---:|
| OpenFWI | 4.09 ms | 3.59 ms | 12.4% |
| Deepwave | 10.50 ms | 9.24 ms | 12.0% |

端到端延迟低于理论 solve 降幅，是因为 gather、response decode 与 metric reduction 不随 rank 立方缩放。这个
结果通过 Q2 efficiency gate，但它是成熟解析 head 的部署证据，不替代 Q3 的方法创新 gate。

## 6. Conditional belief 与 flow gate

我们把 posterior-latent memory 的前 8 个候选看成一个最宽松的离散 conditional belief，并允许 oracle 读取
target truth 选择其中误差最小者。合法比较对象是同一组候选的可部署均值或 global posterior mean 中较强者：

| 数据域，K=1 | 最强 deterministic | best-of-8 oracle | oracle gap |
|---|---:|---:|---:|
| OpenFWI validation | 0.15215 | 0.13985 | 8.08% |
| OpenFWI test | 0.15290 | 0.14140 | 7.52% |
| Deepwave validation | 0.12508 | 0.12411 | 0.78% |
| Deepwave test | 0.12660 | 0.12599 | 0.48% |

两域都低于预注册 15% gate；而 PBNO 在 Deepwave 还强于这里的 best-of-8。OpenFWI 的 latent distance 与
候选 target spread 相关系数约为 0.62--0.64，说明局部原型距离具有 failure signal，但剩余分支收益不足以
证明 conditional flow 的必要性。因此 flow 保持关闭，不用生成式标签增加方法复杂度。

## 7. 当前论文判断

现在最诚实也最完整的故事是：

1. strict joint query 定义了真正的 wave-system prompting，而不是普通同频补全；
2. global Bayesian response operator 在三域成立，并在 Q2 展示可量化的 measurement/compute efficiency；
3. nonlinear latent transport 在两个非饱和域稳定有效，但只有 OpenFWI 达到两位数；
4. local memory 在结构重复明显的系统族有效，receiver residual 与 principal-tail mapping 都有清楚边界。

它已经有 AI 论文的科学骨架，但还不足以声称一个统一 neural operator 在两个域都显著超过强线性特例。下一步
不应继续堆 decoder；更值得做的是让 posterior prior 本身从单高斯升级为 validation-gated 的结构化 latent
belief，并先用 conditional-collision / mixture evidence 判断是否真的存在可利用的分支结构。

## 8. 复现入口

Q2 正式汇总：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_latent_prior_multiseed \
  --config configs/pbno_q2_efficiency_formal_20260925.json \
  --selection-root artifacts/pbno_q2_efficiency_selection \
  --output artifacts/pbno_q2_efficiency_formal/summary \
  --seeds 20260924,20260925,20260926 \
  --device cuda
```

机器可读结果：

- `artifacts/pbno_q2_efficiency_formal/summary/summary.json`
- `artifacts/pbno_q2_efficiency_formal/summary/main_table.csv`
- `artifacts/pbno_q2_efficiency_formal/summary/compute.json`
- `artifacts/pbno_m2_receiver_residual/deepwave_sparse_seed20260925/summary.json`
- `artifacts/pbno_latent_belief_audit/summary.json`
