# 当前状态：Wave-System Representation

> 更新日期：2026-09-23
> 用途：项目的一页式决策入口。数字细节见完整 POC 报告，旧路线见历史归档。

## 2026-09-23 M1 更新：第二数据集成立，但强效应 gate 未过

在公开 WaveBench time-harmonic acoustic 数据上，使用 8,000 / 500 / 500 个 system-held-out
train/validation/test split，完成了三个随机种子的 matched comparison。同一个 shared set operator 从任意
1--4 个 frequency responses 恢复隐藏 wavespeed：

| 表示 | 1 frequency | 2 frequencies | 3 frequencies | 4 frequencies |
|---|---:|---:|---:|---:|
| Raw complex | 0.3024 | 0.2422 | 0.2077 | 0.1811 |
| Amplitude only | 0.3807 | 0.2891 | 0.2398 | 0.2088 |
| Amplitude + phase advance | **0.2794** | **0.2140** | **0.1814** | **0.1584** |

相对 raw 的改善为 7.6%、11.6%、12.7%、12.5%，所有 seed 和 budget 同向。完整四频 paired bootstrap
的 95% 区间为 12.10%--12.93%。这使“phase-coordinate 有跨数据集实用性”从单域观察升级为 A 级事实；
但没有达到预注册的“两个 sparse budgets 至少 15%”强门槛，不能包装成显著架构突破。

两个更复杂的 fixed complex-edge tokens 均弱于当前 soft phase advance。下一步不再继续手工加通道，
而是检验 shared complex-edge graph/point operator。详见
[WaveBench M1 报告](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)。

## 2026-09-22 M0 更新

shared view encoder 与 permutation-invariant 4-slot aggregation 已完成三 seed 正式评估。最重要的结果不再是
固定 24-channel concat，而是一个参数量不随 view budget 增长的 set operator：

| 表示 | 24 views | 6 sources × 1 frequency | 20 dB |
|---|---:|---:|---:|
| Raw complex | 0.0593 | 0.2283 | 0.5288 |
| Amplitude only | 0.1226 | 0.5836 | 1.0807 |
| Amplitude + phase advance | **0.0533** | **0.1622** | **0.1124** |

新结论是：局部 phase advance 主要提高 frequency-poor 与 noisy regime 的观测效率。full-context 相对 raw
仅改善 10.1%，paired bootstrap 区间跨零；clean ellipse 和 128-grid boundary 上 raw 更强；10 dB 仍失败。
详见 [M0 正式报告](../reports/WAVE_SET_OPERATOR_M0_ZH.md)。

当前 instance-contrastive pretraining 不是贡献：冻结 probe 约 0.31，且 full-label 下从头训练为 0.0491，
优于预训练初始化的 0.0639。后续不再继续扫 pooling 或 auxiliary loss。

## 一句话问题

给定一个未知物理系统产生的少量复波动响应，能否学到一个跨 source、frequency、geometry 与
discretization 可复用的系统表示，并最终用它回答新的物理 query？

当前先解决“表示是否存在且是否保留系统信息”，再解决完整复场生成。这个顺序来自一个反复出现的
事实：模型可以较好识别 amplitude 与局部传播状态，却很难从有限频带唯一确定远处频率的完整相位。

## 当前表示

把复场写成 amplitude 与 phase 后，不直接编码绝对 phase，而使用：

1. 每个 response view 内归一化的 log-amplitude；
2. 固定物理距离上，水平方向的复相位 advance 的实部与虚部；
3. 固定物理距离上，竖直方向的复相位 advance 的实部与虚部。

共 5 个实通道。局部相位 advance 通过 `u(x + delta) * conjugate(u(x))` 获得，因此对全局相位旋转
不敏感，也不需要显式 phase unwrapping。低幅值位置用 soft amplitude gate 降低相位通道权重。

这个设计借用了经典的 local wavevector 与 coherence 思想。项目的增量不在于重新发明相位差，而在于
证明它可以作为 response-conditioned system representation，并建立跨 source-frequency 的学习任务、
匹配对照和 failure boundary。

## 已经观察到的强正结果

| Probe | Raw complex | Amplitude | Amplitude + phase advance |
|---|---:|---:|---:|
| Cross-view system retrieval Top-1 | 38.9% | 61.1% | **100.0%** |
| IID hidden-medium RelL2 | 0.3588 | 0.2478 | **0.0504** |
| OOD ellipse hidden-medium RelL2 | 0.4637 | 0.4407 | **0.0955** |

其中 retrieval 的 context 与 candidate views 在 source 和 frequency 上都不重叠；hidden-medium 的训练、
验证和测试按介质严格划分。三 seed 标准差和所有消融见
[完整报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

最重要的科学修正是：

> “幅值容易迁移、相位难迁移”只对了一半。真正不稳定的是绝对相位；局部、gauge-invariant 且按
> 置信度加权的 phase advance 反而是辨识传播系统的关键信息。

## 最重要的负结果

| 尝试 | 结果 | 科学含义 |
|---|---|---|
| source-frequency RBF / eikonal 坐标插值 | RelL2 约 1.15 | 普通坐标平滑不能表达复杂干涉与散射 |
| 少源 Nyström / 全局低秩 Green completion | RelL2 约 0.99 | 少数全局 source columns 不足以恢复复杂响应算子 |
| raw、carrier 或 oracle-medium U-Net 直接生成 missing block | RelL2 约 0.93--0.97 | 当前 decoder 与数据规模不足以生成高频完整复场 |
| unit phase token 加噪 | 20 dB 已明显崩溃 | 低幅值区的 phase 必须做 confidence weighting |
| 64→128 且 partial / closed boundary | 最好仍为 0.614 / 0.637 | 固定物理 stride 有效，但单尺度局部 token 无法处理强多路径 |

这些失败把当前问题定位得更清楚：瓶颈不只是模型容量，也不是简单增加 loss 可以解决；需要分别处理
context representation、multi-branch propagation 和 oscillatory decoding。

## 现在可以相信什么

- 复波场不应默认使用 real/imag 两通道图片表示。
- amplitude 提供相对稳定的系统 fingerprint，局部 phase advance 提供 amplitude 缺失的传播方向和
  局部波数信息。
- 一个轻量、可解释、无需解析 travel time 的 representation 已经显示出强系统辨识能力。
- representation sufficiency 与 field generation 是两个不同问题；前者的成功不能被写成后者已经解决。

## 现在不能声称什么

- 不能声称已经完成 unseen source-frequency 的高精度完整场生成。
- 不能把一个 simulator 上的 ellipse OOD 写成跨方程或跨物理域泛化。
- 不能声称在 10 dB、强混响边界或真实稀疏阵列上稳健。
- 不能把经典 phase-gradient 理论本身写成新的理论贡献。

## 下一阶段的三个决策 gate

### Gate 1：从二维 raster token 到 learned complex connection operator

把 node amplitude 与 gauge-invariant complex edge transport 作为 observation primitive，用 displacement-
conditioned shared kernel 聚合，而不是为 x/y/z/t 手工增加方向通道。先在当前 synthetic 和 WaveBench
matched 比较：至少一域改善、另一域不退化，才继续。

### Gate 2：3D existence，不先做完整生成

优先用已有 3D Helmholtz 数据做 system readout。方法必须复用同一个 node/edge 定义和参数共享规则；若
需要重新发明 3D 专用 token，就没有通过“维度无关”gate。

### Gate 3：连续 query decoder

借鉴 HarmoCore 的 continuous function bases，但不使用 per-frequency independent cores。先用真实
coefficient/system latent 训练 oracle decoder；若 raw complex RelL2 不能低于 0.30，就不把 learned
encoder 接上去。

## 当前判断

两个数据集已经足以说明局部复传播坐标是值得继续投入的科学信号，但仍不足以支撑“通用波动表征”的
论文主张。缺口不再是再加一个 dataset 数字，而是把二维手工 token 提升为真正 dimension-agnostic、
coordinate-continuous 的学习算子。若 learned connection operator 不能在两个二维域至少一赢一平，就应
冻结当前表征，不进入 3D 或完整生成。

## 入口

- [完整方法与实验报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)
- [WaveBench 独立验证与高维路线](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)
- [实验账本：正负结果、禁重复清单与未开发信号](EXPERIMENT_LEDGER_ZH.md)
- [下一阶段冻结协议](WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md)
- [科学备忘录](SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)
- [数据与复现档案](REPRODUCIBILITY_ZH.md)
- [历史探索索引](archive/INDEX_ZH.md)
