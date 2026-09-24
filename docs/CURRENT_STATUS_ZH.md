# 当前状态：Wave-System Representation

> 更新日期：2026-09-24
> 用途：项目的一页式决策入口。数字细节见完整 POC 报告，旧路线见历史归档。

## 2026-09-24 主线切换：In-Context Wave Response Query

当前核心问题不再是改进 medium readout，也不是从低频直接生成整张高频场，而是：

> 给定一个未见 system 的少量 source--frequency complex responses，能否在 inference time 识别其 response
> operator，并回答 source 和 frequency 都未见的新 query？

首轮固定 receiver array，定义 Q1 frequency-held-out、Q2 source-held-out、Q3 joint-held-out；Q3 为主任务。
使用现有 160×9×8 dense archive 抽取 receiver vectors，先完成 conditional identifiability、强
interpolation/low-rank/retrieval baselines、decoder existence 和 active-query oracle，再训练最小
DeepSets/CNP。只有 global bottleneck 被证实才进入 cross-attention。

Belief/flow 需要条件多模态证据，active policy 需要 oracle headroom 与 uncertainty calibration；RL、可变
receiver、world model、3D 和 branch router 均后置。冻结协议见
[In-Context Wave Response Query 计划](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)。

## 2026-09-23 最新阶段：多分支机制成立，fixed 实现未过方法 gate

单个 local phase advance 会把同一邻域内的多个相干传播分量压成一个统计量。MPS-0/1 已完成从
information oracle 到 matched downstream 的逐层检验：

| 层级 | 主结果 | 判断 |
|---|---|---|
| 受控 2/3 分支混合 | 正确 K 将 RelL2 0.664/0.772 降到 0.313/0.326 | 分支可辨识 |
| wavelength-scaled real-field oracle | partial/closed 改善 26.3%/25.7% | oracle gate 通过 |
| three-seed synthetic single-frequency readout | 改善 12%--18% | 稳定正信号 |
| 64→128 partial/closed boundary | 改善 11.1%/11.7%，30/30 system 同向 | **15% gate 未过** |
| WaveBench relative-phase Gram | 单 seed full-context 改善 3.54% | 机制结果，不是正式验证 |

最重要的新原则是：局部分析尺度应随波长变化；复分支表示只应 quotient 一个共同 global phase，同时保留
分支之间决定干涉的相对相位。当前 two-scale packet Gram 把这两个原则实现成固定 69-channel raster，
但只关闭了部分 oracle gap。因此停止继续枚举固定滤波器、半径和手工 edge；若重启，必须引入 learnable
analysis、显式 branch routing 或新的 multipath query。

完整方法、三 seed 统计和 stop decision 见
[MPS-1 报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)；原始命题与相关工作见
[多分支相空间波表征](MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md)。

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

## 当前 response-query 决策 gate

### Gate 1：Q3 information 与 baseline gate

Q3 target 必须随 context budget 变得更可辨识；learned query model 必须相对最强 interpolation/retrieval
baseline 改善至少 15%，不能只在 Q1 frequency interpolation 上成立。

### Gate 2：Belief gate

只有 context-near systems 对 target 存在分离 modes，且 deterministic model 出现均值化，才训练 mixture、
flow 或 diffusion。

### Gate 3：Active gate

只有 complete-table best-next-query oracle 相对最佳 fixed schedule 的 error-AUC 改善至少 15%，且 learned
uncertainty 可以校准，才训练 active acquisition；RL 最后进入。

## 当前判断

已有结果足以说明 complex response representation 能辨识 system，但尚未证明它能回答 counterfactual
probe。Response Query 是区分“system fingerprint”与“可查询 operator latent”的关键任务。当前先用固定
receiver 和 joint source-frequency query 建立最小正结果；不要用 foundation model、belief-space 或 active
agent 的长期 framing 掩盖最基本的 query error。

## 入口

- [完整方法与实验报告](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)
- [当前 Response Query 主线与冻结协议](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)
- [多分支相空间 MPS-1 报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)
- [WaveBench 独立验证与高维路线](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)
- [实验账本：正负结果、禁重复清单与未开发信号](EXPERIMENT_LEDGER_ZH.md)
- [下一阶段冻结协议](WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md)
- [科学备忘录](SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)
- [数据与复现档案](REPRODUCIBILITY_ZH.md)
- [历史探索索引](archive/INDEX_ZH.md)
