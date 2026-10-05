# 下一阶段 POC：Few-shot Spectral Calibration

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

日期：2026-09-16

状态：协议已冻结并完成 P0。独立 Ocean paired-fidelity dense oracle 未通过 task gate，因此 P1 暂停。
结果与归因见 [P0 Gate 报告](../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)。

## 1. POC 只回答什么

这一阶段不追求最终架构，也不尝试一次性补齐多个数据集。它只回答两个有先后关系的问题：

1. 在独立产生的 low/high fidelity gap 下，少量高保真频率扫描能否改善完全无观测频率的完整场？
2. 如果能，phase-aware spectral state 是否比同预算的通用适配机制更有效？

第一个问题不通过，就停止 multi-fidelity framing；第二个问题不通过，则保留任务但放弃当前表示。

## 2. 主任务

对未见物理实例 $e$，给定：

$$
P_e=\{u_e^L(x,f):f\in\mathcal F_L\},
$$

以及固定传感器阵列的高保真 support：

$$
C_e=\{x_j,f_i,u_e^H(x_j,f_i)},
\qquad x_j\in\mathcal M_e,
\qquad f_i\in\mathcal F_C,
$$

预测 disjoint query band：

$$
\widehat u_e^H(x,f_q),
\qquad f_q\in\mathcal F_Q,
\qquad \mathcal F_Q\cap\mathcal F_C=\varnothing.
$$

主 benchmark 暂时沿用易懂的连续频带：

- cheap prior band：90--100 Hz；
- high-fidelity calibration band：101--106 Hz；
- zero-observation near query：107--112 Hz；
- horizon：113--125 Hz。

这些数值只是一个标准切片。模型输入使用归一化频率和相对 band endpoint 的 offset，不允许依赖
硬编码绝对频率。

## 3. Fidelity 定义

### 3.1 强制条件

Low-fidelity fields 必须由一次独立的廉价求解产生，不能从 high-fidelity target 做下采样、低通滤波
或加噪得到。两个 solver 必须共享物理实例 ID、source 和输出坐标，并记录：

- 离散化、边界近似和截断参数；
- 每个频率的 wall-clock 与峰值内存；
- 与 high-fidelity 的 raw complex RelL2 和 phase coherence；
- shared grid 上的插值方式。

### 3.2 P0 推荐实现

首先使用现有 Ocean 环境 ID 和 high-fidelity archive，只补跑同实例的 cheap solver variant。优先顺序：

1. reduced modal truncation 或显式简化的传播模型；
2. 独立的粗离散 solver；
3. 若前两者不可用，再构建新的通用 2D Helmholtz paired-fidelity 数据。

不接受直接缩小 high-fidelity array 作为 low fidelity。Ocean P0 只用于快速验证任务；若通过，下一阶段
必须在带 variable source 的通用 Helmholtz 或 WaveBench-style 数据上复现，才能主张通用 wave
operator。

### 3.3 Query-frequency cheap solver baseline

若 cheap solver 可以在 107--125 Hz 运行，必须把它的 query-frequency 输出作为强 baseline。主方法
可以选择只输入 low-frequency prior，但不能通过隐藏一个可用的 cheap high-frequency prediction 制造
收益。

## 4. 数据划分

P0 development 沿用现有 128 个 training environments，并重新划分：

- 96 个拟合；
- 16 个模型选择；
- 16 个 development diagnosis。

现有 32 个 validation environments 已被多轮实验触碰，只用于 pipeline debugging。最终 POC gate
必须使用在协议和 checkpoint 冻结后才生成的 32 个新 environment IDs。split 必须按物理实例，而不
是按频率或空间 patch。

P0 暂时固定 source，以最小代价验证 task gate，并在报告中称为 field-query POC。只有 P0 通过才进入
P1 variable-source operator setting；固定 source 的结果不能单独支撑完整 operator claim。

## 5. P0：先做 training-free task audit

训练任何新网络前，依次计算：

1. low-fidelity prior 在 calibration 和 query bands 的误差；
2. 现有 pretrained hyper-pole 在新 fidelity 输入下的零校准误差；
3. 使用 dense high-fidelity 101--106 Hz 拟合 stable AR(1) 的 oracle；
4. 使用 dense calibration 拟合 2--4 component low-order rational state 的 oracle；
5. oracle error 随 calibration bandwidth 和 query distance 的变化。

P0 task gate：

- 最佳 dense calibration oracle 相对最佳 uncalibrated baseline，在 107--112 Hz raw complex RelL2
  至少改善 30%；
- oracle 收益至少覆盖 107--110 Hz 四个连续零观测频率；
- 正收益同时反映在 phase coherence，且不是由 amplitude normalization 单独造成；
- cheap solver 的累计成本相对 high-fidelity dense query sweep 至少有可报告的实质差异。

若 oracle 不能通过，停止训练，不通过增加网络容量挽救任务。

## 6. P1：最小方法阶梯

只有 P0 通过才启动 P1，并严格按以下顺序增加容量。

### 6.1 Episodic pretraining protocol

P1 不训练一个固定的 `90--100 → 107--112` 映射，而在 training environments 上构造多个
support/query episodes。推荐的三个训练窗口为：

| Episode | cheap prior | sparse high-fidelity support | supervised training query |
|---|---|---|---|
| A | 74--84 Hz | 85--90 Hz | 91--96 Hz |
| B | 80--90 Hz | 91--96 Hz | 97--102 Hz |
| C | 84--94 Hz | 95--100 Hz | 101--106 Hz |

严格测试 episode 为 90--100 Hz cheap prior、101--106 Hz sparse support 和 107--112 Hz query。
因此模型可以在较低窗口学习“如何根据 support 识别 operator”，但所有 107 Hz 及以上 high-fidelity
query labels 在 checkpoint 冻结前不可见。网络只接收相对 band endpoint 的归一化频率坐标，防止
记住绝对窗口编号。

训练 episode 允许读取对应 training-environment query labels；这是标准的跨任务 meta-training，不能
与 test-time label leakage 混淆。strict environment、strict query band 和 mask seeds 均在模型冻结后
打开。

### 6.2 M0：现有单状态 calibrator

复用现有 invariant set evidence 和 stable AR(1) decoder，只把 true dense LF input 替换为独立产生的
cheap prior。该模型是机制基线，不重新调大 backbone。

### 6.3 M1：通用 support encoder

使用同样的传感器集合、参数量和训练 episode，采用 permutation-invariant set encoder 加 direct
continuous query head。不显式规定 pole 或 phase transport，用来检验收益是否只是通用 conditional
generation。

### 6.4 M2：phase-aware multi-component state

只有 M0 相对 uncalibrated baseline 为正、但与 dense multi-component oracle 存在明显差距时启动。
M2 使用 2--4 个低维谱分量：

$$
\widehat u(x,f)=
\sum_{p=1}^{P} r_p(x)
\exp((f-f_c)[g_p(x)+i\tau_p(x)]),
$$

其中 $g_p(x)$ 有界，$P\in\{2,4\}$。频率轴不是时间轴，因此不强行规定每个分量随频率单调衰减；
resonance 附近允许局部增长。训练前同时比较 exponential atoms 与低阶 rational resolvent atoms 的
dense oracle，只实现 oracle error 更低且跨窗口更稳定的一种。support encoder 只预测相对于 population
prior 的有界状态修正，不为每个 query frequency 生成独立图像。本阶段仍只使用一个 complex field
loss，不添加独立幅值、相位、PDE residual 或 adversarial loss。

## 7. 必须比较的 baseline

所有方法使用相同 training instances、support observations 和优化更新数。

1. high-fidelity population prior，不使用 calibration；
2. cheap solver 的 query-frequency 输出；
3. 当前 LF-only hyper-pole；
4. 通用 latent-code test-time optimization，只更新与结构化状态相同数量的自由变量；
5. parameter-efficient adaptation，优先 last-layer 或 LoRA，不做 full retraining；
6. permutation-invariant generic support encoder；
7. structured spectral-state calibrator；
8. shuffled-instance 和 no-sensor controls；
9. dense calibration oracle，只表示信息上限。

同频 sparse reconstruction 可以作为应用上界，但必须明确它在 query frequency 使用了传感器，不能与
zero-observation 方法并列成同一部署条件。

## 8. 最小实验矩阵

方法选择阶段只使用：

- sensor count $K\in\{46,92,184\}$；
- calibration width $m\in\{2,6\}$；
- query offset $\Delta f\in\{1,2,4,6\}$；
- 一个训练 seed、两个 mask seeds。

只有候选方法通过 development gate，才扩展为三个模型 seeds、三个 mask seeds、完整 107--125 Hz
horizon 和新生成的 strict environments。噪声、主动选点、不规则频率和第二数据集全部后置，避免在
任务尚未成立时扩大实验矩阵。

## 9. 指标与成本核算

主指标：

- 每个物理实例等权的 raw complex RelL2；
- 107--112 Hz macro average；
- error 对 query distance 的曲线；
- paired bootstrap confidence interval。

机制指标：

- amplitude RelL2；
- weighted phase error；
- phase coherence；
- LF-only 到 dense-oracle 的 gap closure。

预算必须分开报告：

| 量 | 定义 |
|---|---|
| hardware budget | 物理传感器数 $K$ |
| spectral acquisition budget | 每个传感器的 calibration frequencies 数 $m$ |
| value budget | 复数观测总数 $K\times m$ |
| spatial coverage | $K/(H\times W)$ |
| simulation cost | low/high solver wall-clock 和内存 |
| adaptation cost | 每个新实例的时间、更新步数和 trainable parameters |

如果最终目标是应用价值，另报告至少一个不替代场误差的下游量：共振峰位置、band energy 或高能量
热点位置。

## 10. P1 方法 gate

在 $K=92,m=6$ 的主设置下，structured method 必须同时满足：

1. 107--112 Hz raw complex RelL2 相对最佳 uncalibrated baseline 改善至少 20%；
2. 相对 strongest generic adaptation baseline 改善至少 10%；
3. 关闭至少 60% 的 uncalibrated-to-dense-oracle gap；
4. 至少 75% strict instances 改善，paired 95% CI 不跨零；
5. shuffled-instance calibration 不产生同方向收益；
6. 收益至少持续到 $\Delta f=4$；
7. amplitude 不得以明显恶化换取表面上的 phase improvement。

若只通过第 1 项但不优于 generic adaptation，结论是“任务成立，当前物理表示不成立”；若只在 Ocean
成立，结论限制为单域 POC，不启动通用 AI 论文叙事。

## 11. P1 通过后的下一步

只有上述 gate 通过，才依次增加：

1. variable source/forcing，完成真正的 operator setting；
2. 第二个通用 wave dataset；
3. irregular continuous calibration/query frequencies；
4. 观测噪声和 sensor placement；
5. 跨 solver 或跨 wave domain 扩大 episodic pretraining。

这一顺序刻意把应用真实性和 task validity 放在架构复杂度之前。
