# 当前状态：Wave-System Representation

> 更新日期：2026-09-22  
> 用途：项目的一页式决策入口。数字细节见完整 POC 报告，旧路线见历史归档。

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

### Gate 1：公平的可变观测编码

建立 shared view encoder 与 permutation-invariant set aggregation。所有 source / frequency budget 共用
参数，性能应随有效观测量增加而稳定改善。

### Gate 2：跨条件稳定的局部传播表示

只增加多尺度固定物理 offset 与 confidence fusion。要求跨 64→128、partial / closed boundary 和 20 dB
噪声稳定优于 amplitude-only；10 dB 至少能可靠退回 amplitude fallback。

### Gate 3：先证明 decoder 可行

使用真实 coefficient field 单独训练 unseen-query decoder。若 raw complex RelL2 不能低于 0.30，就不把
learned encoder 接上去，也不扩大端到端生成模型。

## 当前判断

这个 POC 足以支持继续投入，但还不足以独立支撑“通用波动表征”的论文主张。下一阶段的价值不在于
继续刷当前 IID 数字，而在于验证该坐标系是否能通过公平 set encoding、强反射边界和 oracle decoder
三个结构性 gate。若其中两个失败，应停止当前 token 路线并转向显式 multi-branch / microlocal 表示。

## 入口

- [完整方法与实验报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)
- [下一阶段冻结协议](WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md)
- [科学备忘录](SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)
- [数据与复现档案](REPRODUCIBILITY_ZH.md)
- [历史探索索引](archive/INDEX_ZH.md)
