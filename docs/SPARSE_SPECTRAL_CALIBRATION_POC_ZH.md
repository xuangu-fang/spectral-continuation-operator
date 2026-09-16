# 稀疏谱校准 POC：预注册协议

日期：2026-09-16
状态：实验前冻结；本文件先于首次 validation sparse-calibration 结果生成。

## 1. 核心问题

本 POC 不研究“在目标频率观察少量空间点，再补全同一频率”。它研究：

> 对一个新的物理环境，完整低频带已经给定时，能否只在少数校准频率、少数空间位置测量
> 复波场，辨识该实例的谱传播状态，并将一次校准迁移到完全没有观测的更高频率？

给定

$$
U_L = \{u_s(x,f): f\in\mathcal F_L\},
$$

以及稀疏校准集合

$$
\mathcal C_s =
\{(x_m,f_c,u_s(x_m,f_c)):x_m\in\mathcal M_s, f_c\in\mathcal F_C\},
$$

预测

$$
\widehat u_s(x,f_q),
\qquad
f_q\in\mathcal F_Q,
\qquad
\mathcal F_Q\cap\mathcal F_C=\varnothing.
$$

校准点永远不位于被评价的查询频率，因此任何正结果都必须来自跨频状态辨识，而不能来自
同频空间插值。

## 2. 与已有 sparse reconstruction 的边界

HarmoCore 的主要问题是 $M_q u(f_q)\rightarrow u(f_q)$：每个目标频率都有同频传感器观测。
本 POC 的主要问题是

$$
[U_L,\,M_Cu(\mathcal F_C)]
\longrightarrow
u(\mathcal F_Q),
\qquad \mathcal F_C\cap\mathcal F_Q=\varnothing.
$$

稀疏观测在这里是实例级 spectral calibration，而不是待输出图像中的已知像素。

## 3. 冻结数据协议

数据使用现有 Ocean integer-dense development archive：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/
spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz
```

数据身份、环境划分和 active medium channels 沿用连续谱实验，不改变环境 split。

- 完整低频输入：90--100 Hz，共 11 个频率；
- 预训练监督：只沿用已有 checkpoint 的 101--106 Hz training-environment 标签；
- 主校准频率：105/106 Hz；
- 校准消融：106 Hz、101/102 Hz、101/106 Hz；
- 主要零观测查询：107--112 Hz；
- horizon 查询：113--125 Hz；
- 主查询：110 Hz；
- validation environments：现有 32 个 development validation environments；
- strict environments：本 POC 不读取。

所有校准超参数首先只在 training environments 上选择。冻结后才对 32 个 validation
environments 运行完整预算曲线。现有 validation 已被早期项目用于模型开发，因此这里的结果只作为
POC 证据，不伪装成最终投稿 test set。

## 4. 传感器协议

每个校准频率使用相同空间位置，以模拟一个固定传感器阵列的短频率 sweep。空间网格为
$48\times384=18432$ 点。主预算为每个校准切片：

| 比例 | 传感器数 $K$ | 两个频率的复数观测总数 |
|---:|---:|---:|
| 0.125% | 23 | 46 |
| 0.25% | 46 | 92 |
| 0.5% | 92 | 184 |
| 1% | 184 | 368 |
| 2% | 369 | 738 |

每个环境使用独立、可复现的 nested random mask；大预算严格包含小预算。正式统计组合三个
checkpoint seed 和至少三个 mask seed。另保留 uniform-grid mask 作为布局压力测试，但不用于选择
主方法。

## 5. M0：冻结生成器上的测试时相位状态校准

复用三个 `m4_balanced_train106_hyper_seed*` checkpoint。对低频输入，冻结模型产生
pole--residue 状态

$$
(a_s, r_s, p_s, \tau_s, \kappa_s)=E_\theta(U_L).
$$

第一版不修改 residue、pole 或幅值，只引入低分辨率、实例特定的相位生成元修正：

$$
\tau'_s(x)=\tau_s(x)+I_x(\delta\tau_s),
$$

$$
\kappa'_s(x)=\kappa_s(x)+I_x(\delta\kappa_s),
$$

其中 $I_x$ 是从 $g_h\times g_w$ token grid 到完整空间的双线性插值。修正以有界
`tanh` 参数化；其自由度由 token grid 和物理边界直接限制，不增加额外平滑 loss。

每个新环境只最小化校准传感器上的 complex relative squared error：

$$
\mathcal L_{cal}
=
\frac{
\sum_{f_c\in\mathcal F_C}
\|M_s(\widehat u_s(f_c)-u_s(f_c))\|_2^2
}{
\sum_{f_c\in\mathcal F_C}
\|M_su_s(f_c)\|_2^2
}.
$$

没有 amplitude loss、phase loss、query supervision 或 PDE residual。校准结束后，同一个修正状态
直接查询全部 107--125 Hz。

按以下顺序增加容量，并在 training environments 上停止：

1. 全局 slope：$1\times1$，只校准 $\delta\tau$；
2. 局部 slope：$2\times4$、$4\times8$；
3. 只有 slope 明确受限时，才加入共同 curvature；
4. 本阶段不校准自由 local-span 系数，也不修改 pole/residue。

## 6. 必须比较的基线与控制

1. **LF-only**：冻结 hyper-pole，不使用校准观测；
2. **global calibration**：相同观测、只有全局 slope 参数；
3. **local spectral calibration**：低维局部 slope，必要时 slope + curvature；
4. **shuffled-instance control**：使用另一环境的校准值，检验收益是否来自实例信息；
5. **same-frequency calibration upper comparator**：在每个 $f_q$ 使用同样 $K$ 个同频观测拟合相同
   token，明确区分“跨频校准”与更容易的同频修正；
6. **dense-calibration oracle**：只作信息上界，不作为可部署结果。

若 M0 通过，再实现 set encoder 将测试时优化 amortize 为一次前向；若 M0 不通过，不启动
set encoder、Transformer 或 diffusion。

## 7. 指标

主指标保持逐环境等权的 raw complex RelL2。辅助指标为 amplitude RelL2、phase coherence、
校准点外 RelL2，以及 oracle-gap closure：

$$
GapClosure(K)
=
\frac{E_{LF}-E_K}
{E_{LF}-E_{oracle}}.
$$

主要图表预注册为：

- 107--112 Hz macro error 对总复数观测预算；
- 110 Hz error 对预算；
- $K\times(f_q-\max\mathcal F_C)$ 的误差/相对改善热图；
- amplitude 与 phase 的分解曲线；
- 按预定义中位 paired improvement 选择的场级可视化。

## 8. Gate 与停止规则

在 105/106 Hz、每个切片 0.5% 随机传感器下，M0 至少满足：

1. 107--112 Hz macro raw RelL2 相对 LF-only 下降不低于 20%；
2. 110 Hz raw RelL2 低于 0.35；
3. 至少 80% validation environments 改善，paired bootstrap 95% CI 不跨零；
4. phase coherence 改善，且 amplitude error 不出现足以解释全部收益的变化；
5. shuffled-instance control 不产生同方向收益；
6. 收益至少延续到两个完全无观测查询频率，而非只出现在紧邻校准边界的 107 Hz。

理想结果是在 0.5% 预算下关闭至少 30% 的 LF-only 到 representation oracle gap。若随机校准只能
改善校准频率自身、不能迁移到无观测频率，则核心假设失败；不通过改变评价频率或只展示 best case
挽救。

## 9. 实验后附录：协议演化与最终判定

本节在 validation 评测完成后追加，不修改以上预注册内容。

### 9.1 为什么从 105/106 Hz 扩展到 101--106 Hz

预注册 M0 的测试时自由 slope token 没有通过 training-environment 筛选：全局修正近似无效，局部
修正随自由度增加而恶化。随后两频 dense endpoint-carrier oracle 在 32 个 validation environments
上只能把 107--112 Hz macro RelL2 从 0.4042 降至 0.3403，最大相对改善约 15.8%，低于 20%
gate。因此，两频 calibration 在信息上已不能满足原 gate。

在不读取任何 107 Hz 及以上 future label 训练模型的前提下，我们把 calibration measurement 改成
固定传感器阵列的 101--106 Hz 短宽带 sweep。六频 dense stable-AR oracle 在相同 validation
environments 上达到 0.3106，相对 LF-only 的三 seed 0.4106 有约 24.4% 改善，证明新的观测协议
具有足够的信息上限。只有此后才训练六频 amortized calibrator。该变化是由 oracle 可辨识性诊断
驱动的任务修正，必须与原始两频预注册结果同时披露。

### 9.2 冻结后的正式结果

最终方法使用六个 calibration 频率、每频率 0.5% 即 $K=92$ 个相同空间传感器，平均三个模型 seed、
三个 mask seed 和 32 个 validation environments：

- 107--112 Hz macro RelL2：0.4106 降至 0.3611，相对改善 12.05%；
- dense six-frequency AR oracle：0.3106，因而关闭 oracle gap 的 49.48%；
- 110 Hz RelL2：0.4473 降至 0.4044；
- 110 Hz phase coherence：0.8603 升至 0.8737；
- 110 Hz paired bootstrap 95% CI：$[0.0132,0.0812]$；
- 改善环境比例：62.5%；
- shuffled-instance control：110 Hz RelL2 为 0.5515。

### 9.3 判定

核心假设得到支持：calibration 信息确实迁移到了完全无观测的查询频率，并主要改善相位。但原始
方法 gate 未通过：near-band 改善低于 20%，110 Hz RelL2 未低于 0.35，改善环境比例也未达到
80%。完整结果与归因见
[稀疏谱校准 POC 报告](../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)。

## 10. 后续任务定位说明

本预注册协议保留为历史记录，不回溯修改。后续研究不再把 `90--100 Hz → 107--125 Hz` 的固定映射
视为最终任务，而将它作为 few-shot spectral calibration 的一个 benchmark。

当前协议中的完整 90--100 Hz 输入来自同一个 high-fidelity archive，因此还不能被称为 cheap
simulator prior；主设置的 0.5% 也只表示每个频率的空间覆盖，实际预算为 92 个固定传感器、六个
calibration frequencies、共 552 个复数观测。新的 multi-fidelity claim 必须建立在独立产生的
low/high fidelity pairs 上。

当前统一 framing 见[研究记忆](RESEARCH_MEMORY_ZH.md)，下一阶段实验设置见
[Few-shot Spectral Calibration POC](FEW_SHOT_SPECTRAL_CALIBRATION_POC_ZH.md)。
