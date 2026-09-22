# 复杂波动场的谱可预测性、可辨识性与校准

> **定位说明（2026-09-22）：** 本文是长期科学背景，不是当前实验进度。最新方向、数字和 gates 见
> [一页式当前状态](CURRENT_STATUS_ZH.md)；旧阶段的压缩结论见[历史归档索引](archive/INDEX_ZH.md)。

日期：2026-09-17

状态：科学 formulation 备忘录。本文不提出已经验证的新模型，也不把尚未实验的判断写成结论。
它试图用一套统一语言解释本项目迄今最重要的正结果和负结果，并定义未来真正值得研究的对象。

## 阅读说明：四类证据不要混在一起

本文面向有机器学习背景、但不一定学过信号处理或系统辨识的读者。为了避免把经典知识误写成
“我们的发现”，后文使用四种标签：

- **[经典理论]**：波动方程、信号处理或系统辨识已经给出的结论；
- **[项目事实]**：在当前数据、协议和对照实验中已经重复观察到的结果；
- **[交叉增量]**：不是新的数学定理，而是把经典理论转化为 AI 任务、表征原则或可检验指标后，
  由本项目实验支持的新认识；
- **[研究假说]**：由理论和现象共同提示、但尚未被充分验证的判断。

这里的“增量”是相对于本项目已经建立的证据链而言，不主张历史上首次发现。是否构成论文贡献，
还需要更广泛的文献检索和跨数据验证。

### 三分钟直觉版

可以把一个复杂波场想成许多传播路径的叠加。频率稍微改变，每条路径都会按自己的传播时延转动
相位。相邻频率足够近时，我们能看出它们“转得多快”；频带足够宽时，我们才可能把时延相近的
路径分开。困难来自三处：观测频率太稀会把相位圈数看错，频带太窄会把不同路径混在一起，而新的
共振或模态会让旧频带里根本没出现过的成分进入系统。

因此，低频到高频不是永远可预测或永远不可预测。更合理的问题是：现有观测还能稳定预测多远，
什么表示保留了真正与未来频率有关的信息，以及何时应停止外推并请求少量新观测做校准。

### 常用英文术语速查

| 术语 | 本文中的通俗含义 |
|---|---|
| complex field | 同时保存振幅和相位的复数波场，不是普通 RGB 图像 |
| frequency response | 系统对不同激励频率的稳态响应 |
| impulse response | 系统对瞬时激励的时域响应；它的傅里叶变换就是 frequency response |
| delay spread | 最早和最晚显著到达之间的时延跨度 |
| phase wrapping | 相位每转一圈就回到同一角度，因此相差整数圈看起来相同 |
| phase unwrapping | 利用连续性猜回相位转过的整数圈数 |
| group delay | 相位随频率变化的斜率所对应的有效传播时延 |
| coherence bandwidth | 在多大频率间隔内，两个响应仍高度相关；通常与 delay spread 成反比 |
| resolvent | 给定频率时，把激励映射到响应的“系统逆算子” |
| pole / residue | 共振所在的复频率位置 / 该共振如何贡献到观测场 |
| observability | 现有观测能否区分内部状态 |
| system identification | 从输入输出数据辨识系统状态或动力学参数 |
| analytic continuation | 从已知区间延伸解析函数到区间外；数学上可能唯一，数值上仍可能极不稳定 |
| spectral innovation | 查询频率出现、但可见频带没有充分暴露的新模态或新干涉结构 |
| predictive state | 只保留预测未来所需信息的状态，而不要求完整重建过去 |

## 0. 核心判断

复杂波动场的频率延拓不是普通图像到图像回归，也不应被简化为“幅值可迁移、相位不可迁移”。
更准确的判断是：

> 一个有限频带通常能够确定局部的谱传播状态，但只能在有限 horizon 内稳定预测。相位坐标、
> population prior 和物理结构可以延长这个 horizon；当未观测模态、共振或干涉分支成为主导时，
> 新信息必须通过 calibration 进入系统。

这里有三个必须分开的概念：

1. **解析唯一性**：理想的连续、无噪频率数据在数学上是否唯一决定其他频率；
2. **稳定可辨识性**：有限、离散、有噪观测是否能以有界条件数确定查询场；
3. **分布可预测性**：跨实例训练得到的先验，能否把原本不可稳定辨识的问题限制到较小的可行集合。

本项目真正观察到的不是“波场不能延拓”，而是稳定可辨识性存在强烈的频率距离、传播时延、
干涉零点和 modal innovation 依赖。

## 1. 从波动方程到频率响应算子 [经典理论]

考虑离散化后的线性波动系统：

$$
M_e\frac{d^2 v_e}{dt^2}
+C_e\frac{d v_e}{dt}
+K_e v_e
=B_e s(t),
$$

其中 $e$ 表示介质、几何、边界和 source configuration。对单频激励
$s(t)=\mathrm{Re}(s_0 e^{i\omega t})$，稳态复场满足：

$$
A_e(\omega)u_e(\omega)=b_e(\omega),
$$

$$
A_e(\omega)=K_e-\omega^2M_e+i\omega C_e.
$$

因此：

$$
u_e(\omega)=A_e(\omega)^{-1}b_e(\omega).
$$

$A_e(\omega)^{-1}$ 是频率参数化的 resolvent。直观地说，它是在某个频率上把外部激励转换成
系统响应的逆算子。对开放域 Helmholtz 或 scattering 问题，定义会加入
radiation boundary condition，但核心仍然是一个随复频率变化的 resolvent。远离 eigenfrequency、
resonance 和 cutoff 时，响应对频率通常是光滑或解析的；在适当延拓下，resolvent 是 meromorphic，
其 poles 对应共振，residues 对应 resonant states。散射 resolvent 的这一结构有严格数学基础，参见
[Zworski, 2017](https://doi.org/10.1007/s13373-017-0099-4)。

这带来第一个重要事实：

> 波场在频率轴上不是任意函数，它具有由 wave operator 决定的解析和极点结构。

但这并不意味着从有限低频样本到高频的数值延拓一定稳定。

## 2. 解析唯一性不等于稳定预测 [经典理论]

对 causal linear response，频率响应在复频率半平面具有解析性，并满足 Kramers--Kronig 类型的
全局约束。理想情况下，一个解析函数在实轴区间上的精确连续取值可以决定其解析延拓。

然而有限区间上的解析延拓是经典 ill-posed problem：极小的观测扰动可以在区间外被放大为巨大误差。
Dienstfrey 与 Greengard 对有限实轴区间上的 upper-half-plane analytic continuation 明确指出了这种
不适定性，并用 singular-value expansion 暴露可恢复与不可恢复方向，参见
[NIST / Inverse Problems, 2001](https://www.nist.gov/publications/analytic-continuation-singular-value-expansions-and-kramers-kronig-analysis)。

因此应区分：

$$
\mathrm{exact\ continuum\ uniqueness}
\quad\neq\quad
\mathrm{stable\ finite\ sample\ predictability}.
$$

这里的 AI 类比是：训练分布像一个强先验，它能缩小答案范围，却不能凭空补回观测中不存在、
且在训练分布内部仍有多种可能性的实例信息。神经网络也不能取消这个区别。跨样本训练可以提供先验，从而排除许多数学上可行但在数据分布中
不常见的延拓；它不能从观测中恢复在训练分布内仍然存在的不可区分分支。

一个很实用的形式化是定义与观测相容的可行集合。给定观测 $O_e$ 和容差 $\epsilon$：

$$
\mathcal H(O_e,\epsilon)
=
\left\{
e' : \|O_{e'}-O_e\|\leq\epsilon
\right\}.
$$

在查询频率 $\omega_q$，条件解集合的直径为：

$$
D(\omega_q\mid O_e,\epsilon)
=
\sup_{e_1,e_2\in\mathcal H(O_e,\epsilon)}
\|u_{e_1}(\omega_q)-u_{e_2}(\omega_q)\|.
$$

如果 $D$ 很大，那么任何 deterministic predictor 的 worst-case error 都不能普遍小于 $D/2$。
模型容量无法改变这个下界；新增有效观测或更强且正确的先验才可以。

## 3. 时域到达结构给出的第二种视角 [经典理论]

对固定 source 和 receiver，频率响应也可以写成 causal impulse response 的傅里叶变换：

$$
u_e(x,\omega)
=
\int_0^\infty h_e(x,\tau)e^{i\omega\tau}\,d\tau.
$$

若局部响应近似由 $P$ 条路径组成：

$$
h_e(x,\tau)
\approx
\sum_{p=1}^{P}a_{e,p}(x)\delta(\tau-\tau_{e,p}(x)),
$$

则：

$$
u_e(x,\omega)
\approx
\sum_{p=1}^{P}a_{e,p}(x)e^{i\omega\tau_{e,p}(x)}.
$$

这正是 harmonic carrier、Prony、matrix pencil 和有限指数和模型背后的信号结构。Prony 和
matrix pencil 可以粗略理解为：从一串复数频率样本中，反推出有多少个旋转指数分量，以及每个
分量的旋转速度和权重。有限指数和在理想
条件下可以由有限样本恢复；finite-rate-of-innovation sampling 给出了比普通 bandlimited sampling
更一般的可恢复性框架，参见
[Vetterli, Marziliano and Blu, 2002](https://www.ee.cuhk.edu.hk/~tblu/monsite/pdfs/vetterli0201.pdf)。
Matrix pencil 则是估计 damped/undamped exponentials 的经典方法，参见
[Hua and Sarkar, 1990](https://escholarship.org/uc/item/2n45r2z5)。

但复杂波动场与理想 $P$-path 模型之间有三个差异：

- 有效路径数随位置、频率和分辨率变化；
- continuous reverberation 或 scattering tail 不一定具有有限 $P$；
- 不同分量可在频率上出现 cutoff、resonance 和强烈 amplitude modulation。

因此，有限指数和更适合作为局部模型和 identifiability probe，而不是全频真理。

## 4. 频率采样有两个互补限制 [经典理论]

### 4.1 频率间隔控制无歧义时延范围

单路径的相邻频率相位增量为：

$$
\Delta\phi
=
\angle\left[
u(\omega+\Delta\omega)u(\omega)^*
\right]
\equiv
\Delta\omega\tau
\pmod{2\pi}.
$$

若只依赖主值相位，要无歧义恢复时延，需要近似满足：

$$
|\Delta\omega\tau|<\pi.
$$

使用 Hz 表示：

$$
\Delta f < \frac{1}{2\tau_{\max}}.
$$

均匀频率采样间隔 $\Delta f$ 还会使对应时域响应以 $1/\Delta f$ 为周期发生 aliasing。因此
$\Delta f$ 决定可无歧义表示的 delay aperture，而不是总带宽。

**[项目事实]** 本项目中 79/80 Hz 优于 74/77/80 Hz；“增加更多但相隔较远的频率”也没有帮助。
经典理论给出一个与该现象一致的解释：相位展开在频率采样过稀时本身就具有分支歧义，这不应被
简单归因为训练不足。这里验证的是理论机制与神经算子外推现象的一致性，而不是重新发现相位采样定理。

### 4.2 总带宽控制时延分辨率

设可见频带宽度为 $B$。有限带宽只能把时域到达结构分辨到约 $1/B$ 的尺度。两条到达时间差远小于
$1/B$ 的路径，在当前带宽中可能表现为几乎同一个有效 carrier，却在更远频率逐渐分离。

所以频率观测必须同时满足：

- 足够小的 $\Delta f$，避免长时延 aliasing；
- 足够大的 $B$，分辨相近到达或相近 poles。

样本数大致为 $m\approx B/\Delta f$。这意味着有效自由度会随 bandwidth--delay product 增加，
与 Slepian/Landau--Pollak 的 time--bandwidth concentration 理论一致。该理论说明有限时间与有限频带
中显著自由度的数量由 time--bandwidth product 控制，而不是由中心频率单独控制。一个现代综述入口见
[Simons, 2011](https://geoweb.princeton.edu/people/simons/PDF/reprints/GEM-2011.pdf)。

信号处理里常用 **coherence bandwidth（相干带宽）** 描述响应仍保持强相关的典型频率范围。
它通常与 delay spread 反向变化，但具体常数取决于相关阈值和系统定义。本文的 predictable horizon
与它有亲缘关系，却不完全相同：后者还包含 population prior、空间观测、模型偏差和校准信息。

由此得到一个比“高频更难”更精确的判断：

> 谱复杂度主要由观测带宽与有效 delay support 的乘积决定；中心频率只是通过 modal density、
> attenuation 和 wavelength 改变这两个对象。

## 5. 为什么干涉零点是困难区域 [经典理论 + 项目事实]

当 $u(\omega)\neq0$ 时，可以定义 complex logarithmic derivative：

$$
g(\omega)
=
\frac{d u(\omega)/d\omega}{u(\omega)}.
$$

其实部和虚部分别对应：

$$
\mathrm{Re}\,g(\omega)
=
\frac{d\log|u(\omega)|}{d\omega},
$$

$$
\mathrm{Im}\,g(\omega)
=
\frac{d\phi(\omega)}{d\omega}.
$$

第二项是局部 group-delay-like phase velocity。但当多路径 destructive interference 使
$|u(\omega)|$ 接近零时，$g(\omega)$ 的条件数急剧恶化；相位本身也变得没有稳定物理意义。
通俗地说，两个很强的波若刚好互相抵消，合成结果会接近零；这时极小噪声就能让合成向量的角度
发生巨大跳变，所以该处的“相位标签”本身不可靠。

这有三个直接推论：

1. phase loss 不应在低幅值位置与高能量位置等权；
2. 单一 per-pixel slope 在干涉节点附近必然不稳定；
3. 可靠性或 horizon 预测必须显式读取 amplitude、局部谱曲率和多模型分歧。

**[项目事实]** 我们观察到误差随传播距离、谱曲率和干涉节点增加。该结果支持上述经典条件数
解释，但当前证据还不足以声称三者构成完整的因果模型。

## 6. Modal/resolvent 视角下的有限 horizon [经典理论 + 项目事实]

在局部频带，频率响应可以写成近似 modal expansion：

$$
u_e(\omega)
\approx
\sum_{j=1}^{r}
\frac{r_{e,j}}{\lambda_{e,j}-\omega^2+i\omega\gamma_{e,j}}
+u_{e,\mathrm{smooth}}(\omega).
$$

这解释了为什么 rational interpolation、Loewner realization 和 rational Krylov methods 适合频率响应。
Loewner framework 可以直接从 frequency-response measurements 构造 reduced state-space realization，
其阶数与 Loewner matrix 的有效秩相关，参见
[Ionita and Antoulas, 2014](https://doi.org/10.1137/130914619)。针对 Helmholtz frequency response 的
自适应 rational model reduction 也利用了同样结构，参见
[Bonizzoni, Pradovera and Ruggeri, 2021](https://doi.org/10.48550/arXiv.2112.04302)。

但一个在可见频带内准确的低阶 realization 不一定在带外继续有效：

- 查询频率接近未见 pole 时，导数和条件数快速增大；
- mode cutoff 会改变有效状态维度；
- residues 的空间混合可能随频率旋转；
- 在开放或复杂散射系统中，有限窗口内看似低阶的响应可能只是高阶系统的局部投影。

**[项目事实]** 增加固定 pole 数、使用单一 Neural ODE、或假设全局共享 AR coefficients，都没有
稳定解决远频预测。理论上，频率是 PDE 参数，不是物理时间；一般不存在严格 semigroup：

$$
u(\omega+\delta)
\neq
T_{\delta}u(\omega)
$$

且通常也不满足：

$$
T_{\delta_1+\delta_2}
=
T_{\delta_1}T_{\delta_2}.
$$

把频率轴直接当时间轴做 autonomous rollout，是一个需要验证的局部近似，而不是物理定律。
Neural ODE 在这里失败，不一定说明它拟合能力弱；也可能是它在忠实推进一个不完整的状态。当查询
频带出现新的 active mode 时，再精确的旧状态动力学也无法生成从未进入状态的分量。

## 7. 四个层次的可辨识性 [经典理论 + 交叉 formulation]

### 7.1 Structural identifiability

在一个指定模型族中，连续无噪观测是否唯一决定未知参数或查询响应。这是 uniqueness 问题。
直观问题是：“如果测量无限精确，答案是否只有一个？”这是经典系统辨识概念。

### 7.2 Stable identifiability

有限离散观测和小噪声是否只导致有界查询误差。这由 observation map 的最小奇异值、参数相关性和
query sensitivity 决定。
直观问题是：“测量只有一点误差时，答案会不会差很多？”这是经典逆问题概念。

若 $z_e$ 是局部 latent physical state，线性化后：

$$
\delta O
\approx J_O\delta z,
$$

$$
\delta u_q
\approx J_q\delta z.
$$

带噪观测的局部 information matrix 可以写成：

$$
F_O
=
J_O^*\Sigma^{-1}J_O.
$$

预测不确定度由 $J_q F_O^{-1}J_q^*$ 控制。即使 $F_O$ 对某些状态方向可逆，这些方向如果在 query
处被 $J_q$ 强烈放大，远频预测仍会不稳定。

### 7.3 Distributional identifiability

训练分布提供 population prior $p(z)$，把可行状态限制到经验流形。神经网络的主要价值在这里：
它不是改变 PDE 的 uniqueness，而是减少在真实数据分布中需要区分的状态集合。
**[交叉增量]** 我们把这一层单独列出，是为了区分“物理上能否辨识”和“借助学习分布后能否预测”。
这是面向当前 AI 任务的概念拆分，不是新的辨识定理。

### 7.4 Calibration identifiability

测试时 observation $C_e$ 更新实例状态：

$$
p(z_e\mid O_e,C_e)
\propto
p(C_e\mid z_e,O_e)p(z_e\mid O_e).
$$

Calibration 的价值应定义为它对 query uncertainty 的减少，而不是 support reconstruction error：

$$
\Delta R_q(C_e)
=
R_q(O_e)-R_q(O_e,C_e).
$$

若不同环境的 calibration 可以互换而不影响结果，说明模型没有使用实例信息；本项目的
shuffled-instance control 明显恶化，支持 calibration 确实在辨识实例状态。
**[项目事实]** 正确实例的稀疏校准有稳定收益，而打乱实例配对会显著恶化；这排除了“校准输入只
提供一个通用频率偏置”的简单解释。

## 8. 可预测 horizon 是更基础的科学对象 [交叉 formulation]

对误差阈值 $\eta$，定义实例条件的稳定预测 horizon：

$$
H_{\eta}(O_e)
=
\sup\left\{
h\geq0:
R(\omega\mid O_e)\leq\eta,
\ \omega\in[\omega_0,\omega_0+h]
\right\}.
$$

$R$ 可以是 conditional Bayes risk、可行集合直径的上界，或经过 calibration 的经验误差上界。

这个定义有几个优点：

- 不把 110 Hz 或二倍频当成特殊物理对象；
- 把近程成功和远程失败放入同一条曲线；
- 允许不同环境有不同 horizon；
- 可以直接与 acquisition cost 和 uncertainty calibration 结合；
- 能定义 calibration 的作用为 horizon reset 或 horizon extension。

更进一步，可定义单位成本的 horizon gain：

$$
G(C_e)
=
\frac{H_{\eta}(O_e,C_e)-H_{\eta}(O_e)}{\mathrm{cost}(C_e)}.
$$

这比“使用了多少百分比传感器”更接近真正部署价值。

这里的 horizon 不是传统信号处理中唯一固定的物理常数，而是一个任务条件量：它同时取决于观测、
环境、训练先验、允许误差和模型。**[项目事实]** 我们的多个方法都表现出近程改善、随后与强基线
交叉的曲线，因此只报告某个单频点会掩盖最关键的科学现象。

![不同谱状态模型的频率误差曲线](../artifacts/spectral_generator_gate/continuous_state.png)

**图 1｜项目实证，不是理论示意。** 灰色区域为 calibration 频带，右侧是完全未见的查询频率。
三个面板分别给出复场误差、幅值误差和加权相位误差。Hyper-pole 在近程有优势，但随后与稳定
recurrence 交叉；所有方法都随频率距离恶化。这正是“有限且模型相关的预测 horizon”，而不是
“某模型拥有全局外推能力”。

## 9. 不应急于压成单个标量的控制变量 [研究假说]

目前至少有五个相互作用的无量纲或归一化量：

1. **phase-alias number**：$a=2\Delta f\tau_{\max}$；
2. **delay-resolution number**：$b=B\Delta\tau$；
3. **normalized query offset**：$q=(f_q-f_0)\tau_{\mathrm{eff}}$；
4. **local curvature number**：$c=(f_q-f_0)^2|d^2\phi/df^2|$；
5. **modal innovation count**：可见边界与 query 之间新增的有效 poles、cutoffs 或 modes 数。

可检验的初始假说是：

- $a<1$ 是局部 phase slope 可无歧义估计的必要近似条件；
- $b$ 太小会使相近路径不可分，导致远频分支歧义；
- $q$ 决定局部 slope error 的累积；
- $c$ 和 modal innovation 决定一阶 state 何时失效；
- error curve 更可能在这些归一化变量上跨几何 collapse，而不是在 raw Hz 上 collapse。

这组假说比预设一个“普适 6 Hz horizon”更科学，也更容易被跨域证伪。

## 10. 对 amplitude--phase 发现的更严格表述 [经典机制 + 项目归纳]

“幅值可迁移、相位不可迁移”是有启发性的第一版，但容易被反例攻击。更精确的三层结构是：

### 10.1 Shared envelope prior

路径包络、几何衰减和低空间频率结构通常比 complex interference pattern 更平滑，适合由
population model 提供先验。但观测幅值 $|u|$ 本身包含相干叠加，因此在干涉节点和 mode transition
附近并不稳定。

### 10.2 Locally identifiable phase transport

当频率采样足够密、局部 amplitude 不接近零时，相邻复场比值能够稳定估计局部 phase velocity。
因此 phase 不是“不能迁移”，而是需要正确 gauge 和 sampling density。

### 10.3 Spectral innovation

新模态、共振、cutoff 和多路径重新干涉不能总由旧 slope 唯一确定。它们构成需要 calibration 或
更强先验的 innovation。

前两层与多路径和相位采样理论一致；第三层是我们为解释 AI 外推失败而提出的工作性归纳，仍需用
显式 modal-event 数据验证。所以更稳健的项目主张是：

> Population learning 提供共享 envelope 与结构先验；局部频带辨识 phase transport；
> calibration 负责解析 phase branch 并观测新的 spectral innovation。

## 11. 为什么“表示能力”与“预测能力”必须分开 [项目事实 + 交叉增量]

设 decoder family 为 $u\approx D(z,f)$。很低的 reconstruction oracle 只说明对每个目标存在某个
$z_q$：

$$
\inf_z\|D(z,f_q)-u(f_q)\|\ \mathrm{is\ small}.
$$

真正任务要求的是从历史观测确定该坐标：

$$
z_q=P(O,f_q).
$$

**[项目事实]** Local patch-span oracle 为 0.052，而 learned local-span 为 0.619。前者回答“目标能否被
这套字典压缩”，后者回答“仅凭可见频带能否推断正确坐标”。这个巨大差距说明：压缩得好并不等于
能从过去预测未来。

![全局与局部低秩表征诊断](../artifacts/global_subspace/global_subspace_summary.png)

**图 2｜项目实证。** 左图显示低频子空间随查询频率快速失配；中图显示把表示局部化、增加每个场的
复系数可以显著降低 oracle error；右图则显示 full-band oracle 还能更低。关键不是“低秩不存在”，
而是存在一个很大的 **oracle--learned gap**：目标位于某个低维空间，不代表其坐标能由过去频带预测。

理想 representation 应当是 future-predictive sufficient statistic：

$$
p(U_Q\mid O)=p(U_Q\mid z),
\qquad z=T(O).
$$

同时它应尽量丢弃只帮助重建过去、却不帮助预测未来的信息。Past--future information bottleneck
正是选择性保留 past 中对 future 有用信息的理论框架，参见
[Creutzig, Globerson and Tishby, 2009](https://doi.org/10.1103/PhysRevE.79.041925)。
Predictive-state representation 则直接用未来可观测测试的预测来定义 state，参见
[Littman, Sutton and Singh, 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)。

**[交叉增量]** 这对 AI 表征学习的启示是：

- autoencoding loss 不是充分目标；
- in-band masked reconstruction 也不是 out-of-band predictability 的充分代理；
- latent dimension 应由可预测 future tests 决定，而不是只由当前场重建误差决定；
- 一个有用的 spectral state 应同时输出 future prediction 和自身有效 horizon。

Koopman 方法试图寻找线性推进的 observables，但有限维 invariant subspace 并非普遍存在；即使存在
近似低维表示，也必须检验它是否适合 future prediction，而不只是重建。相关限制见
[Brunton et al., 2016](https://doi.org/10.1371/journal.pone.0150171)。

## 12. Calibration 的三个不同角色 [交叉 formulation]

### 12.1 Gauge calibration

确定共同相位、局部 phase branch 或 carrier slope。它通常只需要很少观测，但作用 horizon 有限。

### 12.2 State calibration

估计已有 reduced model 的 poles、residues、modal coefficients 或 latent state。其有效性依赖 support
是否对这些状态方向可观测。

### 12.3 Innovation calibration

当模型维度或 active modal subspace 已经改变时，旧状态空间本身不完整。此时 calibration 不只是
更新旧参数，而是触发新状态、扩大 latent rank 或重新初始化局部 realization。

**[项目事实]** 现有 sparse AR calibrator 主要做前两类；fixed-8-mode paired-fidelity 中缺失高阶 modes，更接近第三类。
这解释了为什么前者有 12% 稳定正收益，而后者 dense AR(1) 也只能短暂改善一步。

![稀疏异频校准的预算与 horizon](../artifacts/sparse_spectral_calibration/sparse_spectral_calibration.png)

**图 3｜项目实证。** 左上显示增加传感器后收益很快饱和，说明瓶颈不只是空间采样数量；右上和左下
显示校准收益集中在紧邻 support 的频率，并随查询距离衰减；右下显示稀疏校准只能关闭约一半 dense
oracle gap。它支持“校准可更新局部实例状态”，同时反驳“一次校准即可恢复任意远频率”。

![110 Hz 的幅值与相位校准机制](../artifacts/sparse_spectral_calibration/sparse_calibration_mechanism.png)

**图 4｜项目实证。** 正确 sparse calibration 对幅值误差改善有限，却提高了 phase coherence；把其他
实例的校准观测错配进来后，幅值和相位都显著恶化。这是当前最直接的证据：有效信息不是一个通用
频率偏置，而是实例特定的 phase-sensitive state。

## 13. Multi-fidelity 的关键不是 LF error，而是 correctability [交叉假说]

设 low/high discrepancy 为：

$$
d_e(\omega)=u_e^H(\omega)-u_e^L(\omega).
$$

通常只比较 $\|d_e\|$ 和 solver cost。但 calibration task 真正关心的是：

1. $d_e$ 是否位于低维、跨频稳定的 discrepancy subspace；
2. calibration support 是否能观测这个 subspace；
3. query 处是否出现 support 中不存在的新 discrepancy modes。

因此两个 LF solver 即使 raw RelL2 相同，也可能具有完全不同的 calibratability。更合理的 fidelity
设计目标是：

$$
\min_{L}
\left[
\mathrm{cost}(L)
+\lambda\,R_Q(O^L,C^H)
\right],
$$

而不是单独最小化 LF-to-HF error。**[项目事实]** 固定 `nmodes=8` 在浅水实例中几乎没有节省，在
深水实例中又产生难以校正的缺失模态。**[研究假说]** 因而 fidelity knob 更可能需要按有效模态数
或能量比例归一化，而不能跨环境固定绝对截断数；这仍需多种 matched-cost LF 构造验证。

## 14. 对现有实验的统一解释 [项目事实 + 机制归因]

| 实验事实 | 统一解释 |
|---|---|
| 粗频带直接拼接不改善 | $\Delta f$ 超过无歧义 phase-slope 范围，增加的是 alias 后的信息 |
| 79/80 Hz carrier 强于 74/77/80 Hz | 更小的 terminal spacing 给出更可靠的局部 phase transport |
| Carrier M1 在 105/110 Hz 大幅胜过 direct | 正确 gauge 将高频振荡变成较平滑 residual，延长近程 horizon |
| M1 在训练频带内弱于 direct | 结构化 bias 牺牲带内自由拟合，换取带外稳定性 |
| Hyper-pole 只在 107--112 优于 recurrence | 更强 state 改善局部曲率，但其额外参数未稳定扩大远程 horizon |
| Local-span oracle 极低、learned model 很差 | 表示空间足够，但 query coordinates 不能由低频唯一辨识 |
| Masked-frequency pretraining 不帮助外推 | 带内 interpolation tests 不是带外 future-sufficient state |
| Correct calibration 有效、shuffled 失败 | support 在更新实例特定 phase state，而不是提供通用 bias |
| Paired-fidelity oracle 107 Hz 强、随后衰减 | calibration 修正局部 discrepancy state，但没有观测后续 missing-mode innovation |
| Fixed n8 浅水无成本、深水难校正 | fidelity gap 随真实 modal count 改变，absolute truncation 不具跨实例一致性 |

## 15. 传统理论与本项目增量的明确边界

传统理论已经很好地解释了“为什么这个问题可能困难”，但没有自动回答“在跨实例学习、空间场输出和
稀疏测试时校准同时存在时，什么任务最合理、什么表示真的有效”。两者的边界如下。

| 传统理论能够给出的结论 | 传统理论本身没有直接给出、本项目正在建立的增量 |
|---|---|
| 频率响应有解析、极点与模态结构 | 如何把这些结构转化成可学习、可跨实例的高维空间场表示 |
| 有限区间解析延拓通常数值不稳定 | 对一个已训练模型和具体实例，稳定预测 horizon 在哪里、如何估计 |
| 频率间隔约束无歧义时延范围，总带宽约束时延分辨率 | 细 terminal band 与 carrier representation 在神经算子外推中如何共同改善近程泛化 |
| 有限指数和与低阶 realization 在给定模型族内可辨识 | learned latent 是否真由可见频带确定，而不只是事后能重建目标 |
| observability 描述观测能否区分内部状态 | disjoint-frequency sparse calibration 是否更新实例状态，以及 shuffled-instance control 如何检验它 |
| 多保真模型存在成本—误差权衡 | 哪种低保真误差可被少量高保真观测校正，即 fidelity correctability |
| 干涉零点附近相位估计病态 | 面向全空间神经算子的幅相指标、风险预测和停止外推规则应如何设计 |

**最重要的边界声明：** “解析延拓不适定”“相位会 wrapping”“Prony 能估计有限指数和”都不是本项目
贡献。当前真正有增量的部分，是把这些分散理论组织成一个可证伪的 AI 任务，并通过正负实验发现：

1. **频率外推表现为实例相关的有限 horizon，而不是一个全局二元能力。** 多种结构模型都只在近程
   改善，之后与基线交叉；因此应该评价整条 error--horizon 曲线。
2. **一个表示可以很容易容纳未来场，却不能由过去稳定推断。** 0.052 的 local oracle 与 0.619 的
   learned result 构成直接证据，要求把 reconstruction capacity 和 predictive identifiability 分开。
3. **稀疏、异频校准确实携带实例信息，但主要修正局部 phase state，不能自动补回新模态。** 正确
   calibration 改善、shuffled control 恶化、收益随频率距离衰减，共同支持这一判断。
4. **低保真是否有用取决于 correctability，而不只是 raw error。** 固定八模态的实验表明，成本低、
   表面误差相近的先验，可能因缺失 query-sensitive modes 而难以校正。
5. **AI 表征的目标应从“重建整个可见频带”转向“保留对未来查询充分的信息，并预测自己的有效
   范围”。** 这是由项目证据支持的设计原则，目前还不是已完成的方法贡献。

下面再按证据强度列出当前结论。

### 15.1 理论上成立

- wave frequency response 具有 resolvent/meromorphic structure；
- causal frequency response 的有限带解析延拓一般不适定；
- uniform frequency spacing 限制无歧义 delay aperture；
- total bandwidth 限制 delay resolution；
- 干涉零点使 phase derivative 和 local carrier 估计病态；
- 低阶 realization 的存在不保证带外稳定性。

### 15.2 本项目已经稳定观察到

- 细频率 spacing 比粗 multi-frequency input 更有价值；
- carrier-relative representation 显著提高近程 extrapolation；
- predictability 随 query distance、传播距离和谱曲率下降；
- calibration 的主要收益来自实例特定 phase correction；
- reconstruction-friendly local latent 不一定是 prediction-friendly latent；
- 真实 LF/HF gap 中，一步 calibration 有强信号，但六步 open-loop 不成立。

### 15.3 尚待验证的研究假说

- 不同 wave domains 的 error--horizon curve 能否对 normalized delay variables 发生 collapse；
- learned uncertainty 能否在真实 error 爆发之前识别 modal innovation；
- sequential sparse calibration 能否以一半以下 HF acquisition cost 保持给定误差；
- predictive bottleneck 是否能学到比 reconstruction latent 更小、更稳定的 spectral state；
- fidelity correctability 是否比 LF raw error 更能预测最终 calibrated performance。

## 16. 更深的任务 formulation

### 16.1 Open-loop spectral prediction

给定过去频带 $O_e$，预测 disjoint future queries，并同时输出可靠性：

$$
(\widehat u_e(\omega_q),\widehat r_e(\omega_q))
=F_\theta(O_e,\omega_q).
$$

$\widehat r$ 不只是 variance，而应估计 raw field error 超过阈值的概率或上界。

### 16.2 Sequential spectral calibration

在第 $t$ 步，模型拥有 history $H_t$，选择下一次 calibration action：

$$
a_t=(\omega_t,\mathcal M_t),
$$

其中 $\mathcal M_t$ 是固定或可选 sensor set。获得观测后：

$$
H_{t+1}=H_t\cup\{a_t,y_t\}.
$$

目标是在成本预算内最小化整个 query band 的误差：

$$
\min_\pi
\mathbb E\left[
\sum_{\omega\in Q}L(\widehat u(\omega),u(\omega))
+\lambda\sum_t\mathrm{cost}(a_t)
\right].
$$

这把 calibration 从静态附加输入提升为 experimental design 或 active sensing 问题。

### 16.3 Spectral event prediction

除了预测场，还预测下一次 modal birth、cutoff、resonance proximity 或 horizon failure。这可能比要求
模型在不可辨识区域继续输出看似确定的场更科学。

### 16.4 Multi-fidelity extension

Cheap solver prior 是可选信息源，不是主任务定义：

$$
H_t=\{u^L(\omega),C_t^H\}.
$$

评价重点应是它减少了多少 HF calibration cost，而不是是否在某个频率提高几个百分点。

## 17. 对未来 representation 的约束

任何下一代方法至少应满足：

1. **Complex/gauge aware**：共同相位旋转不应改变内部物理判断；
2. **Local but coupled**：允许空间位置相关 state，又不能为每个 patch 分配完全自由的 future code；
3. **Multi-resolution in frequency**：同时读取短间隔 phase increments 和较宽带路径分辨信息；
4. **State plus innovation**：区分已有 state rollout 与新模态触发；
5. **Prediction-oriented**：latent 由 future tests 监督，而非只由 reconstruction 监督；
6. **Self-aware**：输出 horizon 或 error risk；
7. **Calibration-updatable**：新 support 更新低维 state，而不是 full-network fine-tuning。

一个可能的最小状态分解为：

$$
z_e=
\left(
z_e^{\mathrm{env}},
z_e^{\mathrm{envelope}},
z_e^{\mathrm{phase}},
z_e^{\mathrm{innovation}},
z_e^{\mathrm{risk}}
\right).
$$

这不是要求一次实现五个网络分支，而是明确不同信息的职责，避免再次用一个自由 U-Net 混合所有机制。

## 18. 对 flow 或生成模型的判断

解析延拓不适定并不自动等价于目标分布多模态。要证明生成模型必要，需要观察到：

- 在相同可见条件容差内，真实数据分布存在多个稳定分离的 future branches；
- deterministic conditional mean 会系统性抹去这些 branches；
- sampling 能改善 calibrated coverage、proper scoring rule 或 downstream decision，而不只是 best-of-N。

在此之前，更合理的是 deterministic predictor 加 calibrated uncertainty 或 set-valued prediction。
Flow matching 可以成为 posterior representation 的工具，但不能再作为故事出发点。

## 19. 下一步最小科学诊断

在写新模型前，建议只做以下四件事：

1. **Horizon atlas**：统计 error 首次超过阈值的位置，并对 propagation distance、delay spread、
   terminal spacing、visible bandwidth、spectral curvature 和 modal count 回归；
2. **Normalized collapse**：检验不同环境和数据集能否在 $(f_q-f_0)\tau_{\mathrm{eff}}$ 等归一化轴上
   形成统一 error curve；
3. **Sequential oracle**：比较每 1、2、4、6 Hz calibration 时的 error--acquisition-cost frontier；
4. **Correctability audit**：对多个 LF constructions，在匹配成本和 raw error 后比较 dense calibration
   能关闭多少 discrepancy gap。

只有 sequential oracle 明显通过，才值得训练 horizon-aware calibration operator。

## 20. 可能形成的长期研究定位

最有潜力的研究问题不再是：

> 能否从低频生成高频？

而是：

> 给定有限且有成本的频率观测，一个复杂 wave operator 在多远范围内仍然可辨识；什么表示能够逼近
> 这个稳定极限；模型何时应该停止外推并请求新的 calibration？

这一问题位于四个领域交叉处：

- 波动方程的 resolvent、resonance 和 modal analysis；
- 信号处理中的 delay--bandwidth、phase unwrapping 和 exponential estimation；
- 系统辨识中的 realization、observability 和 experimental design；
- AI 中的 predictive representation、operator learning 和 uncertainty-aware active acquisition。

即使最终没有立即形成一篇 AI 顶会论文，这个 formulation 仍然给出一个有价值的科学事实：

> 复杂波动场的频率可预测性不是二元属性，而是一种由 observation geometry、delay support、modal
> innovation、population prior 和 calibration budget 共同决定的有限资源。

## 参考理论入口

- [Zworski, Mathematical study of scattering resonances, 2017](https://doi.org/10.1007/s13373-017-0099-4)
- [Dienstfrey and Greengard, Analytic Continuation, Singular Value Expansions, and Kramers--Kronig Analysis, 2001](https://www.nist.gov/publications/analytic-continuation-singular-value-expansions-and-kramers-kronig-analysis)
- [Vetterli, Marziliano and Blu, Sampling Signals With Finite Rate of Innovation, 2002](https://www.ee.cuhk.edu.hk/~tblu/monsite/pdfs/vetterli0201.pdf)
- [Hua and Sarkar, Matrix Pencil Method, 1990](https://escholarship.org/uc/item/2n45r2z5)
- [Ionita and Antoulas, Data-Driven Parametrized Model Reduction in the Loewner Framework, 2014](https://doi.org/10.1137/130914619)
- [Bonizzoni, Pradovera and Ruggeri, Rational Model Reduction of Helmholtz Frequency Responses, 2021](https://doi.org/10.48550/arXiv.2112.04302)
- [Littman, Sutton and Singh, Predictive Representations of State, 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)
- [Creutzig, Globerson and Tishby, Past--Future Information Bottleneck, 2009](https://doi.org/10.1103/PhysRevE.79.041925)
- [Brunton et al., Koopman Invariant Subspaces and Finite Linear Representations, 2016](https://doi.org/10.1371/journal.pone.0150171)

## 项目内部证据

- [谱连续算子第一阶段报告](../reports/STAGE1_REPORT_ZH.md)
- [连续谱生成器 Gate 报告](../reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md)
- [稀疏谱校准 POC](../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)
- [全局隐空间与局部 token 诊断](../reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)
- [局部复谱 Token POC](../reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md)
- [独立多保真 Calibration Gate](../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)
