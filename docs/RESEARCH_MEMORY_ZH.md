# 研究记忆：Few-shot Spectral Calibration

更新日期：2026-09-16

本文是当前项目的统一研究记忆。历史预注册协议和实验报告保持原样，用于追踪证据；若历史文档的
任务表述与本文冲突，以本文的研究定位和声明边界为准，但历史数字不得被改写。

项目的波动方程与信号处理基础、英文术语解释、经典理论与项目增量的边界，以及现有实验的统一
归因，集中整理在
[谱可预测性、可辨识性与校准科学备忘录](SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)。
该备忘录用于解释科学问题；本文继续负责记录当前任务定义和声明边界。

## 1. 当前核心任务

项目的核心不再定义为一个固定的低频到高频图像映射，也不把 `90--100 Hz → 107--125 Hz` 本身
当作最终贡献。当前任务是：

> 对一个新的物理实例，给定预训练的 population-level operator、廉价但有偏的模拟先验，以及
> 少量测试时传感器观测，辨识该实例的隐式谱传播算子，并在完全没有观测的条件上查询完整场。

暂定英文名称：

**Few-shot Spectral Calibration of Neural Operators**

备选名称：

**Sparse Spectral Calibration of Pretrained Neural Operators**

“frequency extrapolation”是主要 benchmark；“few-shot calibration of a latent operator”才是 AI
任务。System identification 描述 calibration 所恢复的内部机制，不作为论文主任务名称。

## 2. 形式化

第 $e$ 个物理实例具有高保真响应算子

$$
G_e^H:(c_e,s,f,x)\longmapsto u_e^H(x;f,s),
$$

其中 $c_e$ 表示介质或几何，$s$ 表示 source、forcing 或 boundary excitation，$f$ 是频率，$x$ 是
空间查询位置。

测试时可获得三类信息：

1. 跨实例预训练得到的 population prior $G_\theta$；
2. 新实例上的廉价低保真先验

$$
P_e=\{u_e^L(x,f):f\in\mathcal F_L\};
$$

3. 固定稀疏阵列的短频带高保真 calibration sweep

$$
C_e=\{x_j,f_i,u_e^H(x_j,f_i):x_j\in\mathcal M_e,f_i\in\mathcal F_C\}.
$$

目标是预测

$$
\widehat u_e^H(x,f_q),
\qquad f_q\in\mathcal F_Q,
\qquad \mathcal F_Q\cap\mathcal F_C=\varnothing.
$$

评价频率没有任何高保真传感器观测，因此正结果必须来自实例级状态辨识，不能来自同频空间插值。

## 3. AI 问题的本质

这是一个 function-space few-shot learning 问题。一个 episode 对应一个新物理实例：

- population pretraining 学习跨实例共享的场结构和幅值先验；
- sparse support set 描述新实例的少量高保真响应；
- query set 位于 support 未覆盖的物理条件；
- 模型从 support 推断 latent operator，再回答连续 query。

它与普通稀疏重建的区别是：稀疏观测不在待输出频率；与普通 neural-operator fine-tuning 的区别是：
适配信息是一个不规则传感器集合，并要求跨条件迁移；与单纯 frequency extrapolation 的区别是：
最终目标是识别新实例的响应算子，而不是记住一个固定频带映射。

相关 AI 方向包括：

- [Poseidon](https://arxiv.org/abs/2405.19101)：PDE foundation model 与样本高效下游适配；
- [ICON](https://arxiv.org/abs/2304.07993)：无需权重更新的 in-context operator learning；
- [GEPS](https://arxiv.org/abs/2410.23889)：低维 context parameter adaptation；
- [Neural Operator Processes](https://arxiv.org/abs/2606.22946)：partial observations 下的条件算子学习。

稀疏适配本身不是新颖性。项目必须证明的独特部分是：**support 与 query 同时在空间和频率条件上
错开，而相位感知的谱状态比通用微调或直接条件生成更适合这种迁移。**

## 4. 幅值—相位发现如何进入故事

当前证据支持以下分工，而不是“幅值完全可迁移、相位完全不可迁移”的绝对说法：

- 幅值包络和大尺度空间结构更适合由 population prior 提供；
- 实例特定的 phase transport、局部干涉和 modal mixing 是主要 domain gap；
- calibration support 的核心作用是辨识 phase-sensitive latent state；
- continuous query decoder 将该状态传播到无观测频率。

因此，预训练和测试时 calibration 不是两个松散组件：前者承担可共享表示，后者承担不可直接跨实例
迁移的谱相位状态。

## 5. 应用原型

最合适的统一应用是 calibrated spectral digital twin：

1. 离线用多个物理实例预训练通用算子；
2. 部署到新实例时，廉价模拟器提供 dense low-fidelity prior；
3. 少量固定传感器完成一次短频率 sweep；
4. 模型在一次前向或极少量低维优化中识别实例状态；
5. 随后查询未测频率和未布置传感器位置的完整响应。

这一抽象可覆盖结构振动、声学、超声、电磁和地震波传播。Ocean 是数据来源之一，不是任务定义。

## 6. 对当前结果的正确解释

现有 Ocean 稀疏谱校准 POC 证明了一个机制：正确实例的 101--106 Hz 稀疏观测能够改善完全无观测
的 107--112 Hz，并主要改善相位；shuffled-instance calibration 明显失败。

但当前 POC 还不能声称完成 multi-fidelity operator adaptation，原因是：

1. 完整 90--100 Hz 输入来自与目标相同的高保真 archive，不是真实的 cheap simulator；
2. 所有频率来自同一 solver，没有实例化 low/high fidelity gap；
3. source 基本固定，因此更接近 environment-to-field surrogate，而不是真正包含 forcing 变化的算子；
4. 0.5% 只表示每个频率的空间覆盖。主设置实际为 92 个物理传感器、6 个 calibration frequencies、
   共 552 个复数观测；
5. 0.4 左右的 raw field error 只足以支持机制，不足以声称可部署 digital twin。

今后的文档不得把现有 POC 写成已经验证的 multi-fidelity 应用。

## 7. 强制报告规则

所有后续实验必须同时报告：

- 物理传感器数 $K$；
- calibration frequency 数 $m$；
- 总复数观测数 $K\times m$；
- 每个频率的空间覆盖率；
- low-fidelity 与 high-fidelity 数据如何独立产生；
- 模拟 wall-clock、适配时间和推理时间；
- support/query 是否严格频率不相交；
- environment、source 和 geometry 的 split 粒度；
- raw complex RelL2，以及只作机制诊断的 amplitude 和 phase 指标。

若 low-fidelity 只是 high-fidelity field 的下采样或滤波，不得称为独立 multi-fidelity evidence。

## 8. 当前不采用的说法

- 不称当前模型为 foundation model；
- 不声称任意远频率恢复；
- 不用 `0.5% observations` 隐藏六频扫描的总观测数；
- 不把 true dense low-frequency fields 描述为免费的测试输入；
- 不把固定 source 的 environment-to-field mapping 直接称为完整 wave operator；
- 不用幅值或 phase 指标替代 raw complex field error；
- 不在 task gate 通过前堆叠 diffusion、Transformer 或大规模 multi-pole architecture。

## 9. 当前研究决策

2026-09-16 已完成真实 fidelity gap 的第一轮 P0。Ocean LF/HF 由独立 Pekeris 求解产生，8-mode LF
在适用环境中约有 1.8--2.1 倍求解加速；但 dense 101--106 Hz calibration oracle 在 107--112 Hz
只将 raw complex RelL2 从 0.7057 降到 0.5950，改善 15.69%，且到 112 Hz 已变为负收益。因此 task
gate 未通过，当前不训练新的 calibrator，也不把单步 107 Hz 正结果改写成完整 continuation 成功。

项目仍采用两级 gate：

1. **Task gate**：独立产生的 cheap prior 加 sparse high-fidelity calibration，必须能改善完全无观测
   query；
2. **Representation gate**：只有 task gate 通过，才比较物理结构化 latent state 与通用 set encoder、
   latent optimization 和参数高效微调。

具体协议见 [Few-shot Spectral Calibration POC](FEW_SHOT_SPECTRAL_CALIBRATION_POC_ZH.md)，实证结论见
[独立多保真 P0 Gate 报告](../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)。若以后重启，先改变
fidelity 定义或校准的信息结构，并要求新的 dense oracle 在更大实例集通过，再进入表征学习。

## 10. Reliable-horizon Map 阶段结论

2026-09-17 完成了实例级、空间级和 post-reset reliability POC。需要长期保留的发现是：

1. 三 seed disagreement 在 Ocean 与 localized-multipath synthetic 的 fixed-frequency failure AUROC
   分别为 0.906 和 0.960，显著高于 distance-only 的 0.371 和 0.394；
2. 在同一环境、同一频率内部，4×8 patch risk 的 spatial AUROC 为 0.945 和 0.921；2×4 到 8×16
   三个尺度方向一致；
3. Ocean 一次 risk-triggered calibration 把 macro raw complex RelL2 从 0.6314 降到 0.5720，优于
   最佳固定触发的 0.5920，且 shuffled-instance calibration 退回 0.6312；
4. reset 后 failure risk 仍可预测，但第二、三次重复 reset 不再降低 Ocean raw error；
5. synthetic post-reset AUROC 约 0.982，但 sparse action 的 raw improvement 只有约 0.2%，dense action
   ceiling 也很低。

因此后续必须区分两个变量：

- failure risk：当前 prediction 是否即将失效；
- calibration value：候选观测和状态更新器能否真正修复该失效。

当前 Reliable-horizon Map 判为 GO，完整 repeated sequential calibration 判为 NO-GO。若继续 active
acquisition，应学习 value-of-calibration 或 actionable uncertainty，而不是继续扩大 failure-risk head
或直接训练 RL。完整证据见
[Reliable-horizon Map 最终报告](../reports/RELIABLE_HORIZON_MAP_FINAL_ZH.md)。

## 11. Actionable-horizon 阶段结论

2026-09-22 完成了第一阶段 value-of-calibration POC。任务是在购买观测前，从冻结谱状态、实例风险图、
空间风险图和候选 action 描述中预测该动作对 107--125 Hz 完整部署轨迹的真实 raw-error 收益。

必须长期保留的结果是：

1. 32 个 Ocean development environments 上，15 个 endpoints × 3 个等成本随机 masks 形成 45 个
   actions；每个 action 使用 92 个传感器扫描 6 个频率，共 552 个复数观测；
2. 三-mask oracle raw complex RelL2 为 0.5540，最佳 development fixed action 为 0.5920，相对 headroom
   只有 6.42%，未过 10% paper-level task gate；
3. 环境级四折 OOF 的线性 state--action value probe 达到 within-environment utility Spearman 0.624；
   shuffled-instance control 显著退化，说明 value signal 确实具有实例性；
4. value policy raw error 为 0.5846，虽然优于 OOF fold-specific fixed 0.6025，但差于已经冻结的简单
   risk-trigger 0.5720，且 paired 区间方向稳定地不利于 value policy；
5. 新增两个随机 masks 仅把单-mask oracle 从 0.5566 降到 0.5540。当前主要决策自由度是 endpoint，
   随机 mask 没有改变校准器的有效可控子空间。

因此当前停止增加 selector 容量。Actionable value 是一个真实、可部分学习的科学量，但现有 action space
和 AR state update 的 controllability 不足以支持复杂 policy 贡献。若重启，先用 oracle audit 检验风险定向
传感器、不同频率窗口、不同 source/boundary excitation 或更丰富观测类型；只有新 action space 先产生
至少 10% fixed-to-oracle headroom，才训练 action-conditioned representation。详细协议和结果见
[Actionable Horizon POC](ACTIONABLE_HORIZON_POC_ZH.md)与
[第一阶段报告](../reports/ACTIONABLE_HORIZON_POC_ZH.md)。
