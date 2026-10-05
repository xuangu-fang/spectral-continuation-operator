# 稀疏谱校准 POC：结果、机制与停止判断

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](../docs/wave_as_prompt/README.md)。

日期：2026-09-16
结论：**机制级正结果，预注册方法 gate 未通过；不应把当前模型包装成论文级主方法。**

## 0. 后续 framing 更新

本报告中的数字和停止判断保持不变，但其任务定位已经更新。当前 POC 使用的完整 90--100 Hz 场来自
与目标相同的 high-fidelity archive，因此它证明的是“稀疏 calibration 能否跨频迁移”的机制，尚未
实例化 calibrated digital twin 所需的真实 low/high fidelity gap。

今后的核心任务是
[Few-shot Spectral Calibration](../docs/RESEARCH_MEMORY_ZH.md)：用 pretrained
population prior、独立产生的 cheap simulator prior 和少量 high-fidelity support，辨识新实例的
latent operator，再回答完全无观测的条件查询。频率外推是 benchmark，不是最终任务名称。

另需同时披露：主设置的 0.5% 表示每个频率 92 个空间点；六频 sweep 实际使用 92 个物理传感器、
552 个复数观测。以下旧 POC 仍按当时协议报告，不回溯改变定义。

## 1. 一句话结果

在一个新环境中，给定完整的 90--100 Hz 波场，再让固定的稀疏传感器阵列扫描
101--106 Hz，我们可以利用这些观测校准实例特定的复谱传播状态，并改善完全没有传感器观测的
107--112 Hz。每个校准频率使用 92 个传感器，即只覆盖 $92/18432=0.5\%$ 的空间点时，三模型
seed、三 mask seed、32 个 validation environments 上：

- raw complex RelL2 从 **0.4106 降至 0.3611**，相对改善 **12.05%**；
- 关闭 LF-only 到 dense-calibration oracle 误差差距的 **49.48%**；
- 110 Hz raw complex RelL2 从 **0.4473 降至 0.4044**；
- 110 Hz phase coherence 从 **0.8603 升至 0.8737**；
- 110 Hz amplitude RelL2 从 0.2646 变差到 0.2890，说明 complex-error 收益主要来自相位；
- 把校准观测换成另一个环境的观测后，110 Hz 误差恶化到 **0.5515**。

这证明“少量宽带观测可以辨识可跨频迁移的实例状态”不是假信号，但没有达到预注册的
20% near-band 改善和 110 Hz RelL2 小于 0.35 两个方法 gate。收益也主要集中在距离校准带较近的
107--112 Hz，不能声称任意远频外推。

## 2. 任务究竟与稀疏重建有什么不同

HarmoCore 研究的是同频率补全：在要输出的频率 $f_q$ 上已经观察到稀疏传感器值，再恢复同一个
$f_q$ 的完整空间场。这里研究的是实例级谱校准：

$$
[u(90{:}100),\;M u(101{:}106)]
\longrightarrow
u(107{:}125),
$$

其中 $M$ 是固定的稀疏空间阵列，所有评价频率 107--125 Hz 都没有任何传感器观测。稀疏观测的
作用不是补目标图像的已知像素，而是估计“这个新环境的谱传播状态”。因此，成功必须同时满足：

1. 收益出现在完全无观测的查询频率；
2. 使用错误环境的 calibration 值不能获得相同收益；
3. 收益随查询频率离开 calibration band 而出现可解释的 horizon，而不是被隐藏的同频插值。

这使该 POC 与我们已有的稀疏波场重建工作形成互补，而不是重复。

## 3. 数据协议与信息边界

使用 Ocean integer-dense development archive。空间网格为 $48\times384=18432$，共 160 个环境：

- 128 个 training environments，其中前 112 个拟合 calibrator，后 16 个只做模型选择；
- 32 个 validation environments，只在方法冻结后做完整预算评估；
- 完整可见带：90--100 Hz；
- 稀疏 calibration sweep：101--106 Hz；
- 完全无观测查询：107--125 Hz；
- near horizon：107--112 Hz；主查询：110 Hz。

每个环境在六个 calibration 频率使用相同空间位置。正式预算为每个频率
$K\in\{23,46,92,184,369\}$，对应 0.125%--2% 空间覆盖。不同预算使用 nested random masks，
大预算严格包含小预算。结果平均三个 base/calibrator seed 和三个 mask seed。

必须强调：这 32 个 validation environments 在此前项目中已作为 development evidence 使用，因而
这里只能称为 POC validation，不能冒充一次未触碰的最终 test。

## 4. 从失败的两频校准到六频谱状态

实验没有一开始堆叠完整模型，而是按可辨识性逐级推进。

### 4.1 测试时自由相位修正失败

首先只在 105/106 Hz 的稀疏观测上，对冻结 hyper-pole 的局部 slope token 做测试时优化。全局
$1\times1$ 修正近似无效；$2\times4$ 和 $4\times8$ 自由局部修正分别开始恶化。即便使用完整
105/106 Hz 场直接拟合自由 slope，收益也只有约 0.6%。这说明“把 calibration fit 得更好”并不等于
识别出可迁移的谱状态。

### 4.2 两频 dense oracle 暴露信息上限

在训练环境小样本上，用真实 dense 105/106 Hz 构造 endpoint carrier，107--112 Hz 曾出现约
36.6% 改善；但在 32 个 validation environments 上，其 oracle 只有
$0.4042\rightarrow0.3403$，上限约 15.8%，已经低于预注册 20% gate。因此继续优化两频 sparse
encoder 无法从信息上达到 gate。实际两频 amortized model 在 $K=92$ 时仅达到 0.3883，约 3.95%
改善。

### 4.3 六频 AR(1) oracle 通过信息上限检查

把 calibration 从两个切片扩展为 101--106 Hz 的短宽带 sweep 后，对每个空间位置拟合稳定复数
一步传播系数：

$$
a(x)=
\frac{\sum_{f=102}^{106} u(x,f)u(x,f-1)^*}
{(1+\lambda)\sum_{f=102}^{106}|u(x,f-1)|^2},
\qquad |a(x)|\leq 1,
$$

其中 $\lambda=0.01$。用真实 dense calibration band 得到的 oracle 在 107--112 Hz 为 **0.3106**，
相对 LF-only 的 0.4106 改善约 24.4%。只有在这个 oracle 证明信息上限足够后，才训练六频稀疏
calibrator。

## 5. 最终 POC 方法

### 5.1 冻结先验

冻结已有 hyper-pole 生成器。它从完整 90--100 Hz 得到 base spectral state，并产生 101--106 Hz
的 dense prior $\widetilde u_f(x)$。最终 calibrator 不重新训练这个生成器。

### 5.2 相位不变的稀疏证据

对于传感器位置，calibrator 不直接输入绝对 real/imag 值，而构造相对于 prior 的证据：

$$
c_f(x)=
\frac{2u_f(x)\widetilde u_f(x)^*}
{|u_f(x)|^2+|\widetilde u_f(x)|^2+\epsilon},
$$

$$
r_f(x)=\log
\frac{|u_f(x)|+\epsilon}{|\widetilde u_f(x)|+\epsilon}.
$$

$c_f$ 的实部、虚部和 $r_f$ 在共同全局相位旋转下不变。它们以三种尺度进入网络：原始稀疏
evidence、归一化的 $2\times4$ local set aggregation、全局 set average。网络还读取低频幅值、低频
相邻频率的复相关、prior 幅值和介质/坐标。这一设计把 calibration 视为无序传感器集合，而不是把
稀疏点简单当作一张有大量零值的图像。

### 5.3 只预测低维传播状态

约 4.93M 参数的轻量 U-Net 不直接预测 19 个未来频率的图像，而只修正两个复数空间状态：

- 106 Hz reference field $r(x)$；
- 稳定的一步 complex pole $a(x)$。

reference 用有界 log-amplitude 和 phase correction 参数化；pole 用有界 log-magnitude 和 angle
correction 参数化，并显式保证 $|a(x)|\leq1$。任意连续查询由同一个状态生成：

$$
\widehat u(x,f_q)=
r(x)\exp((f_q-106)
[\log|a(x)|+i\,\arg a(x)]).
$$

因此模型容量不能为每个 query frequency 记忆一张独立输出图。

### 5.4 不读取 future labels 的训练

训练只读取 101--106 Hz calibration fields。用它们拟合 target reference $u(106)$ 和上式的稳定
target pole，再在 offset 0--6 上比较两套状态各自的 analytic rollout：

$$
\mathcal L=
\frac{\|G(r_\theta,a_\theta,\Delta f)
-G(r_c,a_c,\Delta f)\|_2^2}
{\|G(r_c,a_c,\Delta f)\|_2^2}.
$$

这里的 target 是 calibration band 所定义的解析 AR rollout，而不是真实 107--112 Hz field。
真实 future labels 只在冻结评测时读取。

## 6. 正式结果

### 6.1 预算曲线，107--112 Hz macro raw complex RelL2

| 每频率传感器 $K$ | Sparse AR | 相对 LF-only 改善 | oracle gap closed |
|---:|---:|---:|---:|
| 23 | 0.3720 ± 0.0065 | 9.40% | 38.60% |
| 46 | 0.3657 ± 0.0044 | 10.92% | 44.83% |
| 92 | 0.3611 ± 0.0045 | 12.05% | 49.48% |
| 184 | **0.3596 ± 0.0025** | **12.41%** | **50.94%** |
| 369 | 0.3623 ± 0.0024 | 11.76% | 48.28% |

对照为 LF-only hyper-pole **0.4106 ± 0.0059**，dense six-slice AR oracle **0.3106**。误差在
$K=184$ 后饱和并略微反弹，说明瓶颈已不是传感器数量本身，而是从稀疏 evidence 恢复局部 pole
field 的表示与归纳偏置。

### 6.2 110 Hz 逐环境 paired 统计

在预注册主预算 $K=92$：

- 平均绝对改善：0.04294；相对改善：9.60%；
- 32 个环境中 62.5% 改善；
- 环境级 paired bootstrap 95% CI 为 $[0.0132,0.0812]$，不跨零；
- 但低于预注册“至少 80% 环境改善”的稳定性要求。

### 6.3 负对照

- no-sensor calibrator：107--112 Hz 为 0.5119，说明收益不是 calibrator 网络自身带来的；
- prior-only AR(1)：0.4107，与 LF-only 0.4106 基本相同；
- shuffled-instance，$K=92$：107--112 Hz 为 0.5243，110 Hz 为 0.5515；
- 正确 calibration 的 near-band phase coherence 为 0.8852，高于 LF-only 的 0.8735；
- shuffled calibration 的 phase coherence 降到 0.8078。

这些控制共同支持：模型使用的是当前环境的 calibration 信息，而且主要校准了跨频相位。

### 6.4 Horizon 边界

在完整 107--125 Hz 上，LF-only、dense oracle、$K=92$ 分别为 0.6517、0.6256、0.6321。远离
106 Hz 后，稳定 AR(1) 的模型偏差累积，稀疏校准相对于 LF-only 的优势趋于消失，部分远频甚至
转为负值。当前证据只能支持短程 spectral continuation，不能支持无限 horizon。

## 7. 预注册 gate

| Gate | 要求 | 观察值 | 结论 |
|---|---:|---:|---|
| 107--112 Hz 相对改善 | 至少 20% | 12.05% | 未通过 |
| 110 Hz raw RelL2 | 小于 0.35 | 0.4044 | 未通过 |
| 改善环境比例 | 至少 80% | 62.5% | 未通过 |
| paired CI | 不跨零 | $[0.0132,0.0812]$ | 通过 |
| phase signal | 改善 | 0.8603 → 0.8737 | 通过 |
| shuffled control | 不得同向改善 | 0.5515 | 通过 |

所以总判定是 **NO-GO for a standalone paper method，GO for the research hypothesis**。

## 8. 第一性原理归因

1. **问题不是没有可迁移状态。** Dense six-slice oracle 的 24.4% near-band 改善说明短宽带中存在
   足够的实例谱状态。
2. **两个频率无法稳定区分 phase slope、干涉零点和局部 modal mixing。** 两频 oracle 本身就不够，
   不是调参能补回的信息缺口。
3. **单一局部 AR pole 仍过于刚性。** 它把多模态干涉压成一个一阶状态，距离越远，轻微 phase
   bias 越快累积。
4. **当前 set-to-field encoder 只恢复了约一半 oracle gap。** $K$ 从 184 增加到 369 不再改善，说明
   主要瓶颈转为表示，而不是样本预算。
5. **幅值与相位仍未完全解耦。** 模型改善相位但牺牲幅值；这与核心发现一致，也暴露了下一版状态
   空间应把 slowly varying envelope 与 oscillatory phase transport 分开建模。

## 9. 下一步建议

后续复盘改变了优先级：不应立即扩大当前 U-Net、堆 loss 或直接训练 multi-pole model。下一阶段先
验证应用任务本身：用独立 cheap solver 产生 dense prior，用 sparse high-fidelity sweep 做 calibration，
再预测 disjoint query band。只有 dense calibration oracle 和最小 AR calibrator 都在真实 fidelity gap
下通过 task gate，才启动 2--4 component spectral state。

完整的最小实验矩阵、baseline、成本核算和停止规则见
[下一阶段 POC 设计](../docs/FEW_SHOT_SPECTRAL_CALIBRATION_POC_ZH.md)。sensor placement、噪声、
Transformer、diffusion 和第二数据集全部后置。

## 10. 证据入口

- 主图：[`sparse_spectral_calibration.png`](../artifacts/sparse_spectral_calibration/sparse_spectral_calibration.png)
- 幅相机制图：[`sparse_calibration_mechanism.png`](../artifacts/sparse_spectral_calibration/sparse_calibration_mechanism.png)
- 汇总数字：[`summary.json`](../artifacts/sparse_spectral_calibration/summary.json)
- 全部聚合记录：[`records.csv`](../artifacts/sparse_spectral_calibration/records.csv)
- 实验前协议：[`SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md`](../docs/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)
- 复现命令：[`REPRODUCIBILITY_ZH.md`](../docs/REPRODUCIBILITY_ZH.md)
