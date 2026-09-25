# Wave as Prompt：面向 AI 顶会的方法主线与长期目标

> 日期：2026-09-25
>
> 状态：active long-term plan
>
> 论文目标：从少量复数 response prompts 中识别一个未见 wave system，并回答尚未执行的 excitation query。

> 2026-09-25 执行更新：独立 CP functional rank 在三个域均失败；shared-coordinate Tucker 只在 Ocean
> 通过。当前不再假设 raw complex response 在 source--receiver 轴上低秩，而把低秩限定在跨环境 latent
> operator family。24 小时执行优先级已在第 9 节更新。

## 1. 一句话论文故事

传统 neural operator 通常读取介质、几何或 PDE 参数，再预测波场。真实实验中，这些隐藏环境参数往往不可得，
但可以用少量 source excitations 对系统进行探测。本文研究：

> 能否把少量 wave responses 当作 physical prompts，在不更新模型权重、不显式重建隐藏介质的条件下，形成
> 可查询的 system representation，并预测未执行激励的复响应？

主任务保持清楚：**同频率、未观测 source response query**。frequency、receiver 和 joint query 是连续算子
能力的扩展检验，不在第一阶段混入主 claim。

## 2. 论文需要形成的三项贡献

### C1. Response-prompt learning problem

定义 system-held-out、variable-budget 的 wave response prompting protocol。训练和测试环境严格分离，环境只通过
少量复响应暴露；模型不能读取 velocity、geometry、material parameter 或其他隐藏真值。

### C2. Prompted Functional Tensor Operator

提出一个连续、复值、低秩且物理对称的 latent response operator。它把当前 discrete response subspace 推广为：

1. 连续 source--frequency--receiver 坐标上的 functional response bases；
2. 由 prompts 识别的低维 system state；
3. 能在新坐标上被 query 的 complex operator；
4. 由参数化保证、而不是由额外 loss 惩罚的 reciprocity。

### C3. Cross-application measurement efficiency

在 seismic、ultrasound 与 ocean acoustics 三个域上，用同一个问题定义和方法原则证明：少量 probes 可以识别未知
系统，并减少达到给定复响应精度所需的物理测量数。

## 3. 单一核心方法

暂定名为 **Prompted Functional Tensor Operator，PFTO**。

未知系统的复响应写成：

$$
H_{\theta}(s,\omega,r)
=
\mu_{\psi}(s,\omega,r)
+
\sum_{l=1}^{L} z_{\theta,l} B_{\psi,l}(s,\omega,r).
$$

其中：

- $z_{\theta}$ 是由当前系统 prompts 识别出的低维 system state；
- $B_{\psi,l}$ 是训练环境共享的连续 complex response basis；
- $s$、$\omega$、$r$ 分别是 source、frequency 与 receiver query coordinates。

每个 basis 不存成固定离散矩阵，而使用 functional tensor rank：

$$
B_{\psi,l}(s,\omega,r)
=
\sum_{h=1}^{R}
a_{l,h}(\omega)
\phi_{l,h}(s,\omega)
\phi_{l,h}(r,\omega).
$$

在 source 与 receiver 共址的部分，共享 coordinate function，并使用普通转置意义下的 complex-symmetric
parameterization，从结构上满足：

$$
H_{\theta}(s,\omega,r)=H_{\theta}(r,\omega,s).
$$

这里不是 Hermitian symmetry，不对交换后的值取复共轭。非共址区域使用 source/receiver 两套 coordinate
functions，但仍共享 system state。

### 3.1 Prompt inference

给定 $K$ 个观测：

$$
D_K=\{(q_i,y_i)\}_{i=1}^{K},
\qquad q_i=(s_i,\omega_i),
$$

第一版不额外训练一个庞大 encoder，而是在共享 functional bases 上做可微的 regularized latent fitting：

$$
z_{\theta}^{*}
=
\arg\min_z
\sum_{i=1}^{K}
\frac{\|M_i(\mu_{\psi}+B_{\psi}z)-y_i\|_2^2}
{\|y_i\|_2^2}
+
\lambda\|z\|_2^2.
$$

这个小问题有闭式线性求解，不要求测试时更新网络参数。训练时对 source rows 做 episodic masking，使 bases
直接为少量 prompts 下的 query error 优化，而不是只优化 full-table reconstruction。

只有当解析 fitting 出现明确优化瓶颈时，才比较 permutation-invariant amortized set encoder；最终论文只保留
验证集上更好的一个 inference mechanism，不把二者堆叠。

### 3.2 波动物理如何进入方法

物理先验优先通过表示与结构实现，不堆多个 loss：

1. **complex response**：始终联合预测实部和虚部，不把 amplitude 当替代目标；
2. **reciprocity**：在共址 source/receiver 子空间中由 complex-symmetric parameterization 精确保证；
3. **wavelength-aware coordinates**：坐标 Fourier features 使用数据域公开的 reference speed 和 frequency
   做无量纲化，不读取隐藏介质；
4. **functional query**：输出依赖物理坐标，而不是固定 output index；
5. **low response-state rank**：压缩的是跨环境共享的 response operator family，而不是假设单张振荡场平滑。

第一阶段只使用 query loss 和 ridge regularization。除非消融证明必要，不加入 amplitude loss、phase loss、PDE
residual、contrastive loss 或多项 consistency loss。

## 4. FNO、functional tensor 与 flow 的明确角色

### Functional tensor：主方法

它把系统 latent、source、frequency 和 receiver 的作用分开，同时保留连续坐标查询能力。当前 PCA
response subspace 是离散 full-table basis；PFTO 是其坐标连续、参数共享、可施加互易性的推广。

### Neural operator / FNO：必要对照，而非默认主干

至少加入以下参数匹配对照：

- receiver-grid FNO：把 $K$ 条 context responses 当输入函数，输出 target response function；
- DeepONet-style branch/trunk；
- masked set/Transformer operator。

如果 FNO 在固定规则网格上更强，可把其 spectral block 用作单条 response function encoder；但不因此放弃
coordinate decoder，也不让固定 grid FNO 定义整篇方法。

### Conditional flow matching：后置的 latent belief module

Flow 不直接生成高维复场，只允许建模：

$$
p(z_{\theta}\mid D_K).
$$

启动 flow 必须先满足以下 existence gate：在冻结 sparse-budget episodes 上，存在相近 context 但明显不同 target
的 system pairs，并且 latent K-sample oracle 相对 deterministic posterior mean 至少降低 10% query error。若 gate
不通过，论文保持 deterministic，不为了“生成式”标签加入 flow。

## 5. 分阶段执行计划

### M0. Functional-rank existence audit

目的：先验证 response family 是否能被坐标连续的 functional tensor 表达。

执行：

1. 复现冻结 discrete response subspace；
2. 比较 complex CP、symmetric CP 和小 Tucker core；
3. 绘制 parameter/storage--full-table-error Pareto；
4. 单独检查共址 reciprocity residual；
5. 在 globally held-out receiver coordinates 上检查 continuous interpolation。

通过条件：

- 至少 `2/3` 域在不超过 discrete basis 50% 存储量时，full-table error 不高于它的 `1.10` 倍；
- 第三个域不高于 `1.25` 倍；
- 共址区域 reciprocity relative residual 不高于 `1e-5`；
- 至少两个域的 held-out receiver-coordinate error 相对最佳普通插值降低 10%。

若纯 CP 失败，只允许升级一次为 shared-coordinate Tucker basis；若仍失败，不把 functional tensor 定为论文主
方法，转而使用 continuous basis bank，保留同一 prompt-to-latent formulation。

### M1. Episodic PFTO

目的：让 continuous bases 针对 sparse prompts 下的 target query 训练。

执行：

1. 同一 checkpoint 随机接收 variable K；
2. differentiable ridge solve 识别 $z_{\theta}$；
3. mask target source rows，直接优化 system-macro complex RelL2；
4. validation 选择 rank、ridge 和唯一一种 inference mechanism；
5. 先单 seed development，通过后再运行三 seed test。

核心通过条件：

- half-budget 相对当前 response subspace 和 validation 选出的最强同预算 baseline，在至少 `2/3` 域降低
  raw complex RelL2 至少 10%；
- 第三个域退化不超过 3%；
- 对应域的 paired-system bootstrap 95% 下界大于零，system win rate 至少 70%。

为避免只靠相对数字，half-budget 的目标 raw error 同时设为：

| 数据域 | 当前最好附近 | 论文目标 |
|---|---:|---:|
| OpenFWI，K=2 | 0.165 | 不高于 0.150 |
| Deepwave，K=8 | 0.058 | 不高于 0.055 |
| Ocean，K=4 | 0.0043 | 不高于 0.0050，且主要看 OOD |

这些 absolute targets 不替代相对强基线 gate；Ocean IID 已接近饱和，不能靠它单独宣称方法成功。

### M2. Measurement-efficiency 与 continuous-query 证据

目的：证明 functional operator 不只是换一种矩阵分解写法。

执行：

1. 在归一化预算 $\rho=K/(S-1)$ 上生成完整 error curves；
2. 计算达到 full-budget strong baseline 精度所需的最小 K；
3. 对 Deepwave ring、Ocean depth array 和 OpenFWI line receivers 做 receiver-grid thinning；
4. 测试训练中 globally held-out 的 receiver coordinates；
5. 若同频任务通过，再在至少两个域做 in-band frequency interpolation，暂不要求远频 extrapolation。

通过条件：

- 至少两个域 equivalent probe saving 不低于 25%，或者 half-budget error 不高于 full-budget strong baseline
  的 `1.05` 倍；
- normalized budget-error AUC 在至少两个域降低 8%；
- continuous receiver query 在至少两个域相对最佳 interpolation/FNO baseline 降低 10%；
- frequency extension 只有达到 10% matched gain 才进入正文，否则只记录为 scope boundary。

### M3. Flow existence gate

目的：判断 sparse prompts 下是否存在值得建模的 conditional ambiguity。

先做：context collision、target divergence、deterministic ensemble disagreement、latent sample oracle。

只有 latent K-sample oracle 改善至少 10%，才训练 conditional rectified flow，并报告 NLL、coverage、CRPS 和
posterior-mean raw RelL2。否则本阶段以 negative scientific result 结束，不能影响 PFTO 主线完成。

### M4. Formal cross-domain evaluation

冻结模型与超参数后一次性运行：

- OpenFWI 500 test systems；
- Deepwave 256 IID 与 128 geometry/contrast OOD systems；
- Ocean 128 IID 与 64 parameter OOD systems；
- 三个训练 seed；
- physical-system paired bootstrap；
- parameter、training time、inference time 和显存统计。

OOD gate：Deepwave 与 Ocean 至少一个相对 strong baseline 改善 5%，另一个退化不超过 3%。

## 6. 顶会级最终硬指标

### 必须同时满足

1. 三个数据域使用同一问题定义、同一代码接口和同一方法原则；
2. half-budget strong-baseline gain：至少 `2/3` 域不低于 10%，第三域不低于 -3%；
3. 至少两个域达到 25% probe saving，或达到等价的 half-vs-full gate；
4. 至少两个域证明 continuous-coordinate query 优于普通插值/固定网格方法；
5. Deepwave/Ocean OOD gate 通过；
6. 三 seed 同向，paired-system interval 和 win rate 达标；
7. 主模型不超过 2M 参数；若超过 matched baseline 两倍，必须在 accuracy--compute Pareto 上占优；
8. 论文主结论始终使用 raw complex RelL2，不用 amplitude/phase 指标替代。

### 顶会投稿前的 NO-GO 条件

以下任一成立时，不把当前版本包装为方法型顶会论文：

- PFTO 在三个域都不能稳定超过 discrete response subspace；
- functional tensor 只有压缩收益，没有 query accuracy、coordinate generalization 或 measurement-efficiency 收益；
- 正结果只来自 Ocean IID 的简单低维 Pekeris family；
- flow 没有 ambiguity evidence，只改善 best-of-many samples；
- 需要四项以上辅助 loss 才能得到增益；
- 只在一个 K、一个 seed 或 query-level micro-average 上成立。

## 7. 论文正文预期结构

### Figure 1：问题与方法

少量 response prompts → system latent → continuous excitation query；对比传统已知介质 forward operator。

### Figure 2：核心方法

functional complex bases、analytic prompt fitting、reciprocal coordinate sharing；突出一个 latent operator，而不是
模块流水线。

### Figure 3：跨域 measurement curves

三个 panel 使用统一归一化 probe budget，展示 raw complex error 与 full-budget reference。

### Table 1：half-budget 主结果

每域报告 raw error、strong-baseline gain、probe saving、paired interval、win rate 和 OOD。

### Table 2：方法与物理消融

discrete PCA、普通 CP/Tucker、无 reciprocity、无 wavelength coordinates、PFTO、matched FNO/DeepONet。

### Figure 4：可解释性

system latent geometry、functional rank spectrum、phase-error maps、代表性 complex response 与 failure cases。

## 8. 明确剪枝

第一轮不同时加入：raw-field flow、Neural ODE、hypernetwork、PDE residual、多分支 token、active acquisition、
medium reconstruction 或 inverse task。它们只有在对应 gate 揭示明确瓶颈时才能进入。

本阶段最重要的成功不是“用了多少现代模块”，而是建立一个简单统一的对象：

> **a wave system is a latent functional response operator inferred from physical prompts.**

## 9. 2026-09-25 起的 24 小时执行窗口

### 已获得、必须继承的证据

1. 对每个 discrete PCA response basis 单独做 source--receiver CP：三个域均未通过 50% storage / 1.10×
   error gate；
2. shared source/receiver Tucker：Ocean 用 44.9% storage 无损复现，OpenFWI 与 Deepwave 仍失败；
3. receiver 轴 Fourier/cosine 截断，即使先去掉 reference direct-path carrier，也不能在 OpenFWI/Deepwave
   保留 query accuracy；
4. Deepwave 原 rank=32 baseline 不是容量上限；rank=64 的 validation error 从约 0.0583 降到 0.0519；
5. 初步 learned latent-spectrum prior 在重新调强的 rank baseline 上仍显示两位数 headroom，必须用正式逐
   query metric 复核。

上述结果意味着 functional tensor 的合理轴分解是：

```text
environment latent rank × continuous query-to-response functions
```

而不是：

```text
small source rank × small receiver rank
```

receiver function 可以保留足够多的 oscillatory modes；论文贡献来自 prompt-conditioned system posterior、
measurement efficiency 与 continuous evaluation，而不是强行声称 raw field spatial compression。

### 0--4 小时：冻结 Probabilistic Response Operator

- scalar ridge 与 spectrum prior 都允许 rank 至少搜索到 128；
- validation 使用两组 frozen context orders；
- test/OOD 使用三组 orders；
- 主指标恢复为逐 query relative L2 后再做 physical-system macro average；
- 输出逐系统 CSV、paired bootstrap、win rate、budget curve 和 equivalent probe saving。

继续条件：half-budget 至少两个域相对重新调强的 scalar baseline 改善 10%，第三域不退化超过 3%。

### 4--10 小时：PFTO 最小实现

- 保留训练系统学习到的 response functions 与 latent spectrum prior；
- 将 discrete receiver basis 改写为全频带 Fourier/cosine functional evaluation，不再截断 modes；
- source/frequency coordinate 进入 query trunk；
- 共址子阵 reciprocity 由 output projection 或 complex-symmetric block 保证；
- 只比较 analytic MAP latent solve 与一个小型 amortized prompt encoder，不叠加二者。

继续条件：至少一个非饱和域相对 Probabilistic Response Operator 再改善 5%，另一域不退化超过 3%；若没有，
保留解析 operator 为主方法，不用更大网络制造复杂度。

### 10--15 小时：continuous receiver query

- Ocean 重新生成更密 receiver-depth grid；
- Deepwave 生成同系统的 128-point ring response 子集；
- 训练只使用原 48/64 receiver grid，测试 interleaved unseen coordinates；
- 对照 linear/cubic interpolation、phase-aware interpolation、FNO/DeepONet trunk。

进入正文条件：两个域至少一个相对最强插值降低 10%，另一个不退化超过 3%。若新仿真不能按时可靠匹配原
系统，则只保留 validation existence audit，不用伪配对数据替代。

### 15--18 小时：latent flow existence gate

- 审计 context collision 与 target divergence；
- 在相同 context 上比较 posterior mean、Gaussian samples 与 latent K-sample oracle；
- 只有 K-sample oracle 改善 10% 才训练 conditional rectified flow；
- flow 即使失败也不能阻止 deterministic PFTO 完成。

### 18--24 小时：正式统计与论文 artifacts

- 对通过 development gate 的模型运行三 seed；
- 完成 IID/OOD、budget AUC、probe saving、bootstrap 和 compute Pareto；
- 生成正文级主表、三 panel budget curve、方法示意图和 failure cases；
- 更新 experiment ledger、当前状态、报告、复现命令与 checkpoint/hash；
- 代码测试、Ruff、公式兼容检查、Git commit 与 push。

### 24 小时 promotion gate

1. half-budget：至少 `2/3` 域相对最强 scalar/subspace baseline 改善 10%，第三域不低于 -3%；
2. raw error：OpenFWI K=2 不高于 0.150，Deepwave K=8 不高于 0.055；
3. measurement efficiency：至少两个域达到 25% probe saving，或 half-budget 不高于 full-budget baseline
   的 1.05 倍；
4. OOD：Deepwave/Ocean 至少一个改善 5%，另一个不低于 -3%；
5. continuous query：至少一个新 receiver-grid 域达到 10% matched gain，第二域必须诚实报告；
6. 三 seed、逐 physical-system paired statistics；任何只在单个 context order 上出现的增益不 promotion；
7. 若 flow gate 不通过，明确保留为负结果，不影响 deterministic paper story。

## 2026-09-25 M0 执行结果

Probabilistic Response Operator 已按上述严格协议完成，详细数字见
[正式报告](../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md)。当前 gate 状态：

- half-budget strong-baseline gain：OpenFWI 25.9%、Deepwave 18.2%、Ocean 39.3%，通过；
- raw error：OpenFWI 0.1292、Deepwave 0.04349，通过；
- equivalent probe saving：OpenFWI 与 Ocean 均为 25%，通过；
- half-budget OOD：Deepwave 16.1%、Ocean 14.6%，相对改善通过，但 Ocean 绝对误差 1.032；
- 固定一次 validation selection、三组 prompt-order seed、system-level paired bootstrap，完成；
- continuous receiver query：Ocean interleaved holdout 相对 matched scalar+cubic 改善 10.4%，达到最低
  gate；Deepwave 的 true-response interpolation oracle 为 1.387，确认当前 1/2 环阵采样不可辨识；
- Deepwave dense sampling：32/64/88 observed receiver grids 的 oracle 为 1.387/0.476/0.407，方法始终贴近
  oracle；96-grid 离散圆已到唯一 cell 上限附近，冻结 continuous 分支；
- prompt-only calibration：Ocean K=4 OOD 从 fixed spectrum 的 1.032 降到 0.846，同时 IID 仍相对
  scalar 改善 36.1%；K=8 OOD 尚未完全修复；
- functional tensor 的小 source/receiver rank 假设已被否定，后续不再强行压缩振荡空间轴；
- episodic dense-basis learning 无显著增益，flow 尚无多模态 existence 证据，二者暂不 promotion。

方法主线据此收敛为：`complex response representation + learned latent prior + in-context posterior update`。
下一阶段先做 prompt-adaptive OOD calibration，再为 Deepwave 生成更密 receiver grid；只有 dense-grid
sampling oracle 合格才升级到 neural functional decoder。
