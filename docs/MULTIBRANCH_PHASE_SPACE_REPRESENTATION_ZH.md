# 多分支相空间波表征：科学命题、相关工作与首轮 POC

> 状态：2026-09-23 frozen research brief；MPS-0/1 已完成
>
> 结果：波长尺度 oracle 通过；fixed multiscale Gram 未通过 15% downstream gate
>
> 目的：记录被检验的表征命题与预注册边界，而不是把事后结果写回假设。
> 相关入口：[项目主页](../README.md) · [当前状态](CURRENT_STATUS_ZH.md) ·
> [实验账本](EXPERIMENT_LEDGER_ZH.md) ·
> [MPS-1 结果报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)

## 执行后摘要

- 受控 2/3 分支混合中，正确的 K 将 held-out complex RelL2 从 0.664/0.772 降到 0.313/0.326；
- 固定 33-pixel 大窗口 oracle 只改善约 5%--8%，但按约 1.5 个局部波长设置窗口后，partial/closed 的
  K=4 相对 K=1 改善为 26.3%/25.7%，oracle gate 通过；
- 由此实现的 two-scale complex packet Gram 在三 seed 64→128 boundary shift 上改善 5.1%/11.1%/11.7%
  （open/partial/closed），所有系统同向，但未达到预注册 15%；
- 最符合已观察证据的情形是后文“结果 B”：表示空间存在，但固定滤波器组与 69-channel raster 只关闭了
  部分 gap。完整方法、表格、图和后续 stop decision 见
  [MPS-1 报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)。

## 0. 一句话命题

> 波动场真正特殊的对象不是一张高频复数图像，而是一个物理系统如何传播、散射并相干叠加局部波成分。

因此，下一阶段不再把波场默认表示成 raster patch，也不直接跳到一个庞大的 Green 函数生成器。我们先
验证一个更小、可证伪的假设：

> **同一空间邻域中允许存在多个带方向、尺度、复系数和可信度的传播分支，是否比当前单步局部相位
> 增量更充分、更稳定地表示复杂波动系统？**

当前 5-channel token 是这个命题的单尺度、单分支、一阶近似。多分支相空间 token 若成立，应首先改善
强反射、多路径、网格变化和稀疏观测效率；若这些条件下没有改善，就不应继续包装成更大的模型。

## 1. 为什么波动场不是普通物理图像

### 1.1 相位参考可以任意改变，相位关系却决定物理结果

若整个复场同时旋转一个相位：

$$
u'(x)=u(x)e^{i\phi_0},
$$

许多系统属性并没有变化。一个合理的 encoder 不应因为参考相位改变而改变 system latent。另一方面，
相邻位置、不同频率或不同传播路径之间的相对相位决定相长和相消干涉，不能把 phase 全部丢掉。

当前表示使用局部复共轭乘积：

$$
q_\delta(x)=u(x+\delta)\overline{u(x)},
$$

并做带幅值置信度的 soft normalization：

$$
c_\delta(x)=
\frac{q_\delta(x)}{|q_\delta(x)|+\epsilon_a}.
$$

它对共同相位旋转不敏感，同时保留局部相位 advance。两数据集结果已经说明，这种局部化相位不是噪声，
而是隐藏系统辨识的重要信息。

### 1.2 波场在空间域很复杂，在位置、方向和尺度的联合空间中可能更简单

均匀介质中的单个局部平面波可以写成：

$$
u_m(x)=a_m(x)e^{i k_m(x)\cdot x}.
$$

在缓变区域，幅值 $a_m$ 变化较慢，局部波矢 $k_m$ 同时描述传播方向与局部波数。复杂场更接近多个
局部分量的相干和：

$$
u(x)\approx \sum_{m=1}^{M(x)} a_m(x)e^{i\phi_m(x)}.
$$

同一点可能同时有直达、反射、绕射或不同模态。此时 $q_\delta$ 只给出这些分量混合后的一个局部复
统计量，并不等于任何一条真实路径的波矢。

经典 curvelet 理论说明：对一类线性双曲方程，波传播算子在同时局部化位置、方向和尺度的表示中近似
稀疏；单个 curvelet 的中心近似沿 Hamiltonian flow 传播。这提供的是理论动机，不直接等于一个可学习
模型已经成立。

### 1.3 单张场只是传播算子的一个切片

在线性系统中，给定环境 $e$，频域响应可以写为：

$$
u(r,\omega)=\int G_e(r,s,\omega)q(s,\omega)\,ds.
$$

$q$ 是激励，$s$ 是 source，$r$ 是 receiver，$G_e$ 是 Green 或 response operator。一张 dense field 只是
固定 source、frequency 和 source waveform 后的一个切片。普通图像 encoder 描述“状态长什么样”；
更理想的波表征应该描述“这个系统如何把激励传到接收点”。

### 1.4 线性叠加、互易性和有限传播速度是可检验的结构

对线性波系统：

$$
G_e(\alpha q_1+\beta q_2)=
\alpha G_e(q_1)+\beta G_e(q_2).
$$

许多系统还在正确物理测度下满足 source-receiver reciprocity。它们不应只作为训练数据中的偶然相关性，
而应成为检验表示是否真的学习传播规律的 OOD 变换。我们的诊断已经发现：普通像素度量下 reciprocity
误差约为 0.84，加入正确权重后约为 $2.1\times10^{-6}$。这说明“在哪个内积中表示响应”本身就是物理
结构，而不是无关的 normalization 细节。

### 1.5 波长决定有效分辨率，频率不是普通 metadata

频率升高会改变局部波长、有效自由度和表示所需尺度。因此频率不应只作为拼接到 MLP 的标量；理想
token 应在 wavelength-normalized 的尺度上工作，并允许同一共享规则跨 grid spacing 和空间维数复用。

## 2. 三类结论必须严格分开

### 2.1 传统波传播与信号处理已经知道的事实

- 相对相位与局部波矢携带传播信息；绝对相位依赖参考系。
- windowed Fourier、Gabor、wavelet、curvelet 等表示联合局部化空间与频率/方向。
- 多路径意味着同一位置可能存在多个传播分支，单值 phase gradient 不再充分。
- 线性波动满足叠加；许多介质在正确变量与测度下满足互易。
- 波传播算子的频率响应可能出现 resonance、sharp phase rotation 和 mode conversion。

这些不能写成我们的新理论贡献。

### 2.2 本项目已有实验支持的增量事实

- raw real/imag 并不是当前任务下最有效的复场输入坐标。
- 单视图归一化 amplitude 提供稳定 system fingerprint，但单独使用不充分。
- gauge-invariant、confidence-weighted spatial phase advance 与 amplitude 强互补：在原始 synthetic 上将
  hidden-medium RelL2 从 0.3588 降到 0.0504，并在公开 WaveBench 上跨 1--4 个频率预算稳定改善
  7.6%--12.7%。
- 该表示的优势主要出现在低频率预算和中等噪声；它不是 universal clean-OOD winner。
- fixed symmetric/directed edge normalization 没有进一步改善，说明继续手工增加 edge channels 不是
  当前有证据的方向。
- partial/closed boundary、128-grid 和完整 unseen-query field generation 仍是明确失败区。

### 2.3 下一阶段才要验证的研究假设

- 当前失败是否来自“一个邻域只允许一个平均相位 advance”，而非局部相位思想本身。
- 多尺度、多方向、多个软分支是否能在复杂边界中分离可复用传播成分。
- 这种 token 是否比 raster channel 更容易跨网格、跨源布局和跨二维/三维复用。
- token 的复数系数能否通过 coherent summation 重构局部场，而不是只提高 downstream probe。
- 表示是否最终能组成可查询的系统传播算子。

在这些问题被实验验证前，本文只称其为 hypothesis，不称为新的通用波动模型。

## 3. 表示层级：从场值到传播算子

| 层级 | 表示对象 | 能表达什么 | 当前证据 |
|---|---|---|---|
| L0 | raw real/imag 或 amplitude/phase | 单张场的值 | 强 baseline，但跨 view 混淆明显 |
| L1 | 邻点复共轭乘积 | 单尺度平均局部传播 | 两数据集成立，当前主结果 |
| L2 | 多尺度、多方向、多分支 wave tokens | 共存的入射、反射、绕射和模式 | 下一阶段 POC |
| L3 | 环境的传播或散射 latent | 任意 source、receiver、frequency 查询 | 长期命题，尚未解决 |

这也解释了 HarmoCore 类函数张量表示的边界：它擅长紧凑、连续地表示一个场，但 per-frequency core
主要编码场状态，并没有显式编码多个传播分支或共享的 system operator。因此稀疏重建成功不能自动推出
跨 source 或跨频率生成成功。

## 4. 多分支相空间 token 的最小定义

对局部窗口 $W_x$，不立即压缩成一个 phase gradient，而输出至多 $K$ 个软 token：

$$
z_m=
(x_m, k_m, \sigma_m, \alpha_m, \gamma_m, p_m).
$$

各变量含义为：

- $x_m$：token 的空间中心；
- $k_m$：局部波矢，包含方向与波数；
- $\sigma_m$：分析尺度或有效波长；
- $\alpha_m$：复系数，保留相干叠加所需的幅值与相对相位；
- $\gamma_m$：分支可信度或局部 coherence；
- $p_m$：可选的 mode / polarization 标记，首轮二维声学 POC 暂不使用。

局部重建应保持复数相干和：

$$
\widehat u(x)=\sum_{m=1}^{K}\alpha_m\psi_m(x;x_m,k_m,\sigma_m).
$$

这里的 $\psi_m$ 可以先使用固定 complex windowed Fourier / steerable atoms。首轮不要求端到端学习所有
原子，也不要求 token 对应严格的几何射线。衍射、焦散和有限频率效应使“硬 ray token”过于强；我们
使用 soft branches，允许分支重叠和不确定。

### 4.1 当前 token 是什么特例

当 $K=1$、尺度固定、方向只通过 x/y 两条边估计，而且不显式局部重建时，上述表示退化为当前 phase
advance。新 POC 的必要性因此很清楚：只检验“允许多个局部传播分支”是否解决当前明确的 multipath
边界，而不是无目的增加容量。

### 4.2 为什么不直接使用全局 FFT

全局 FFT 具有频率分辨能力，但在非均匀介质和有边界散射时失去空间定位；同一个 Fourier mode 可以在
多个区域经历不同传播。多分支 token 需要同时保留位置、方向、尺度和复系数。

### 4.3 为什么不先做大 Transformer

如果固定或轻量可学习的局部分析都不能证明多分支信息存在，Transformer 只能增加优化与归因混淆。
首轮的顺序应是：

```text
controlled branch identifiability
        ↓
fixed complex wave-packet analysis
        ↓
soft top-K local branch tokens
        ↓
small matched set operator
        ↓
only then: learned propagation/message passing
```

## 5. 它应带来哪些独特泛化能力

| 泛化轴 | 对应物理性质 | 最小检验 |
|---|---|---|
| 全局相位旋转 | gauge invariance | latent drift 与输出 equivariance |
| 单波到波的叠加 | linear superposition | 未见复系数组合的响应误差 |
| 新 source 位置/形状 | response operator | source-held-out evaluation |
| 单路径到多路径 | 多分支局部结构 | partial/closed boundary |
| 新 grid spacing | wavelength-normalized scale | 64 到 128 matched test |
| 新频率 | dispersion 与尺度变化 | frequency-held-out readout |
| 二维到三维 | $k$ 是向量而非手工 channel | 通过二维 gate 后再做 existence test |
| source/receiver 交换 | reciprocity | 正确物理测度下的一致性 |

只有在这些结构性变换上改善，才能说明模型捕捉了 wave-specific representation，而不仅是一个更强的
image encoder。

## 6. 值得长期发展的任务

### 6.1 Few-measurement wave-system cloning

给定未知环境的少量 source-frequency-response views，构建可查询的传播 latent：

$$
Z_e=E\left(\{s_j,\omega_j,u_j\}_{j=1}^{N}\right),
$$

并回答新的 source、receiver、frequency 与 waveform query。跨频延拓、空间稀疏重建和 hidden-medium
recovery 都可视为这个大任务的不同切片。

### 6.2 Active wave interrogation

模型不再被动接受随机 0.5% 观测，而是在当前不确定性下选择下一个最有信息量的 source、frequency 或
receiver。主指标是达到给定可靠精度需要多少测量，而不是只报告某个固定 mask 的 RelL2。

### 6.3 环境变化检测与快速 latent update

先为一个系统建立传播 latent；环境局部改变后，只用少量新测量更新表示并定位变化。这直接对应无损
检测、结构健康监测、医学超声、地下监测和设备数字孪生。

### 6.4 Multipath decomposition

不只输出最终复场，还识别直达、反射、绕射或 mode-converted 分量。该任务既有应用价值，也是检验
token 是否具有传播语义、而非换一种方式压缩像素的直接标准。

### 6.5 控制与逆设计

若 latent 真正表示传播算子，就应支持阵列相位控制、聚焦、反射抑制和材料逆设计。控制性能可以成为
比同分布重建更严格的 representation probe。

## 7. 与相关工作的关系

以下文献用于确定理论来源和方法边界，不用于宣称这些方向无人研究。

1. **Curvelet Representation of Wave Propagators is Optimally Sparse** 证明一类双曲传播算子在
   curvelet 表示中近似稀疏且沿 Hamiltonian flow 有组织地移动。它支持 phase-space localization 的
   理论动机，但不解决从有限 response views 学习 system latent。
   [Candès and Demanet, 2004](https://arxiv.org/abs/math/0407210)
2. **Windowed Fourier Propagator** 使用 frequency-local propagation，并显式保持 superposition，从
   plane waves 泛化到更一般的复杂波态。它说明 wave-specific operator 可以从通用 FNO 中分化出来；
   我们更关注从少量观测辨识隐藏系统以及局部多分支表示。
   [WFP, 2026](https://arxiv.org/abs/2603.14289)
3. **Enforcing Reciprocity in Operator Learning for Seismic Wave Propagation** 将 source-receiver
   reciprocity 硬编码进 neural operator，并支持多源同时预测。它支持把结构性变换作为模型约束；我们
   额外观察到 reciprocity 依赖正确物理测度。
   [RENO, 2026](https://arxiv.org/abs/2602.11631)
4. **Hybrid Operator Learning of Wave Scattering Maps in High-Contrast Media** 将平滑背景传播与高对比
   散射 correction 分开，说明强散射不宜完全交给单一全局 Fourier block。
   [Hybrid scattering operator, 2026](https://arxiv.org/abs/2602.11197)
5. **Neural Operators with Localized Integral and Differential Kernels** 提供分辨率无关的局部 integral /
   differential kernel 设计。它是把 fixed x/y edge token 升级成 displacement-conditioned shared
   operator 的直接架构参考。
   [Localized kernels, 2024](https://arxiv.org/abs/2402.16845)
6. **Neural Operator: Learning Maps Between Function Spaces** 给出 discretization-invariant operator
   learning 的通用框架；我们的目标不是再证明 universal approximation，而是确定波动问题所需的特殊
   observation coordinates 与结构约束。
   [Neural Operator, 2021](https://arxiv.org/abs/2108.08481)
7. **Real vs. Complex Spectral Bases for Neural Operators** 强调表示基应与 Green operator 的相位与对称
   结构对齐。这与我们“不把 complex field 当普通双通道图片”的经验一致，但该工作研究的是 spectral
   basis 选择，不是 response-conditioned multi-branch tokens。
   [Green-function-aligned bases, 2026](https://arxiv.org/abs/2606.24851)

## 8. 24 小时首轮 POC：只验证一个因果链

### 8.1 冻结假设

> 当前 token 在强反射和多路径条件下失败，至少部分原因是它把同一邻域中的多个传播分支压成一个
> 单尺度平均相位 advance。显式的 soft multi-branch local decomposition 应提高系统辨识和跨条件稳定性。

### 8.2 MPS-0：分支可辨识性与信息 gate

先构造或复用可控的 1、2、3 个局部 plane-wave / curved-wave 混合，不训练 downstream 网络。

比较：

- 当前 single-step phase advance；
- local FFT / windowed Fourier 能量峰；
- 多尺度 complex steerable filter bank；
- 使用 target local patch 拟合的 top-K oracle atoms。

指标：

- held-out grid 上的局部 complex reconstruction error；
- branch direction 与 local wavenumber error；
- 全局相位旋转后的 token stability；
- 新复系数叠加下的 reconstruction / superposition error；
- 低幅值和两分支相消位置的 calibration。

**Stop rule：** 在真实 synthetic 的 partial/closed patches 上，top-K oracle 相对 K=1 不能稳定降低至少
15% local complex error，则停止“分支容量不足”归因，不进入 learned branch router。

### 8.3 MPS-1：固定前端的最小下游 POC

只增加一个组件：固定或弱可学习的 complex wave-packet bank 加 soft top-K pooling。下游继续使用当前
matched shared set operator；不加入 Transformer、PDE loss、生成 loss 或完整 query decoder。

对照：

- raw complex；
- amplitude-only；
- 当前 5-channel phase token；
- 单分支 $K=1$ wave-packet token；
- 多分支 $K=2$ 或 $K=4$ token。

主任务仍是 system readout / hidden-medium recovery，因为它已经有可靠 split 和强 baseline。优先使用最能
区分假设的 partial/closed boundary 与 128-grid stress test；IID 只作防退化检查。

**Promotion gate：** multi-branch 相对当前 phase token 在至少一个预先指定困难条件下降低 paired RelL2
至少 15%，在另一数据域和 IID 中不退化超过 2%。未过 gate 不增加 backbone。

### 8.4 MPS-2：只在 MPS-1 通过后学习 analysis/router

若固定前端已经显示多分支正信号，只允许学习下列一项：

- analysis filters 的小残差；或
- token routing / branch confidence。

不同时学习 filters、propagation graph 和 decoder。必须报告 branch collapse、slot permutation stability、
有效分支数与相位旋转一致性。

### 8.5 MPS-3：第二数据集和机制验证

在 WaveBench 上进行 matched comparison，但不为了追求平均数字而把所有 easy cases 混在一起。至少按
contrast、frequency budget 和局部 spectral multi-modality 分层报告。只有 synthetic 困难边界与 WaveBench
至少一赢一平，才保留为下一阶段主线。

## 9. 首轮明确不做什么

- 不同时实现 full Green operator、active sensing 和 3D generator；
- 不把 source location、解析 travel time 或 true medium 输入 token extractor；
- 不用 query target truth 选择分支；
- 不以 amplitude/phase 辅助指标代替逐系统 raw complex RelL2；
- 不把 analytic plane-wave success 写成现实数据成功；
- 不把更多 channels 本身当作贡献，参数量与训练预算必须 matched；
- 不在 oracle information gate 失败后用更大网络强行优化。

## 10. 可能的结论及其含义

### 结果 A：oracle 与 learned multi-branch 都明显成功

说明当前瓶颈确实包括局部分支混叠。下一阶段可以研究 token propagation graph，以及可查询 response
operator。

### 结果 B：oracle 成功，learned token 失败

说明表示空间存在，但分支提取或监督不可辨识。优先研究 assignment、self-supervision 和 acquisition，
而不是增加 downstream 容量。

### 结果 C：受控混合成功，真实多路径不成功

说明 plane-wave 假设过强，可能需要 curved wavefront、diffraction-aware atoms 或直接学习局部 integral
kernel。不得声称现实 multipath 已被表示。

### 结果 D：oracle 本身不优于当前 token

说明 partial/closed 失败更可能来自全局传播关系、boundary condition 或观测不足，而非局部分支容量。
冻结该方向，转向物理测度、global response graph 或更有信息量的 acquisition。

## 11. 长期愿景与当前阶段的边界

长期科学命题可以表述为：

> **从少量复杂波动响应中，学习一个紧凑、可查询的表示，描述物理系统如何传播和散射相干波。**

多分支相空间 token 是表示层的候选答案；few-measurement system cloning、active interrogation 和环境变化
更新是任务层的候选答案。本阶段回答了更窄的问题：多分支局部表示在 oracle 层面比单分支 phase
advance 更接近困难波场的充分统计量，但当前 fixed Gram 实现尚未达到主方法门槛。后续若重启，必须带来
可学习分析或显式分支 routing 等新的归因变量，不能继续枚举固定通道。
