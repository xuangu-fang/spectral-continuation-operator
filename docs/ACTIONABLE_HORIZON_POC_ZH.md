# Actionable Horizon：校准动作价值图 POC

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

更新日期：2026-09-22

## 1. 一句话问题

Reliable-horizon map 回答“当前预测何时、何处会失败”。本阶段进一步回答：

> 在尚未购买新观测之前，能否判断一次候选 calibration action 是否真的会降低后续未观测频率上的完整复数场误差，并选择最有价值的动作？

这不是把 failure risk 换一个名字。高风险只表示“可能出错”，action value 还取决于传感器是否观察到
可辨识的状态、校准器是否有足够表达能力，以及剩余查询区间是否足够长。

## 2. 清晰的问题设定

对新物理环境 $e$，冻结的谱算子已经给出 107--125 Hz 的完整场预测。部署状态 $s_e$ 只包含：

- 90--100 Hz 可见频带和冻结谱表示；
- 三个独立 seed 的预测及其 disagreement；
- 在查询真值未知时得到的实例级和空间级 failure-risk map；
- 已经执行过的 calibration history；第一阶段 history 为空。

一个候选动作写为

$$
a=(f_c,M,K,W),
$$

其中 $f_c$ 是短 sweep 的终点，$M$ 是固定传感器布局，$K$ 是每频率传感器数，$W$ 是 sweep
包含的频率数。执行动作后可以读取 $f_c-W+1$ 到 $f_c$ 上 $M$ 位置的复数场，并更新谱状态。

主 POC 固定 $K=92$、$W=6$，只选择 sweep endpoint 和三种等成本随机 sensor-mask realizations：

$$
f_c\in\{106,107,\ldots,120\},\qquad M\in\{4201,4301,4401\}.
$$

因此每个环境有 45 个 matched-cost actions，每次动作的 gross cost 都是 552 个复数观测。mask seed
在不同环境中生成不同的随机坐标，并不代表跨环境共享同一绝对硬件位置。固定成本
可以先隔离“动作选择”本身，避免用更多传感器自然换来更低误差。

### 2.1 主价值定义

令 $E_e^0(f)$ 为不校准时的 raw complex RelL2，$E_e^a(f)$ 为执行动作后的误差。动作只能影响
$f\geq f_c$ 的预测，因此部署轨迹为

$$
T_e^a(f)=
\begin{cases}
E_e^0(f), & f<f_c,\\
E_e^a(f), & f\geq f_c.
\end{cases}
$$

主 utility 是整个固定部署区间上的误差下降：

$$
V_e(a)=\frac{1}{19}\sum_{f=107}^{125}\left(E_e^0(f)-T_e^a(f)\right).
$$

这个定义同时惩罚“触发太晚”和“修复无效”。它不会因为较晚 action 只在更短区间上评分而获得不公平
优势。$V_e(a)\leq0$ 定义为 harmful calibration。

同时报告局部固定窗口 utility，作为机制诊断：从动作后的首个 query 开始，只看接下来 6 个频率；
它区分“校准器短期可修复，但部署收益被触发时机稀释”的情况。

### 2.2 两种 horizon

- **Reliable horizon**：未校准预测保持在误差阈值以内的最远频率；
- **Actionable horizon**：至少存在一个可用动作，使期望 utility 为正且 harmful risk 低于阈值的最远状态；
- **Recoverable horizon**：使用真实反事实误差计算的 oracle actionable horizon，只用于评价，不作为输入。

第一项是“会不会坏”，后两项是“坏了还能不能用当前工具修”。

## 3. 无泄漏协议

动作执行前允许使用：冻结预测、seed disagreement、频率距离、预测场自身的幅值/粗糙度/谱变化、
OOF failure probability，以及候选传感器在 OOF spatial-risk map 上的覆盖统计。

严格禁止使用：

- 107--125 Hz query target、真实 query error 或由它们计算的曲率；
- 候选动作尚未购买的 calibration values；
- 执行动作后才出现的 calibrated-state disagreement；
- 同一环境的其他 action label 进入训练折。

所有切分以物理环境为单位。一个环境的 45 个 action 必须整体位于同一折；不能把 action rows 随机
拆分。标准化和 ridge 超参数选择只使用训练环境。

## 4. 分阶段计划

### M0A：已有 replay 的回顾性 task audit

复用已经存在的单阵列反事实 tensor，测量：最佳固定 endpoint、逐实例 oracle endpoint、动作最优点分布、
harm rate 和 fixed-to-oracle gap。这个数据在本文档前已经产生，因此明确标为 retrospective mechanism
audit，不伪装成盲预注册实验。

进入学习 POC 的最低机制 gate：

- oracle 相对最佳固定动作至少改善 5%；
- absolute macro RelL2 gap 至少 0.02；
- 至少 50% 环境的最优 endpoint 不等于最佳固定 endpoint。

这只说明存在可学习的选择空间，不是论文级成功标准。

### M0B：冻结后的扩展 action table

保持 checkpoint、环境、query、$K$、$W$ 不变，新生成两个 mask seed，得到 45 个 matched-cost actions。
报告 action heterogeneity、oracle headroom、harm rate、mask 与 endpoint 各自贡献。

论文级 task gate：扩展 action space 中，逐实例 oracle 相对“训练环境选择的最佳固定 action”改善至少
10%，或者关闭 no-calibration 到 oracle 总 gap 的优势足以稳定地区分 learned policy 与 fixed policy。
若未通过，不训练复杂 action encoder。

### M1：轻量 value probe

只在机制 gate 通过后训练低容量 ridge value regressor。输入由状态特征、动作描述和二者的简单交互组成，
输出环境内相对 utility score；另一个相同容量、预测绝对 utility 的 ridge gate 决定是否 no action。
四折 environment-level OOF prediction 后，每个环境选择相对 score 最大的动作。

对照包括：

1. no calibration；
2. best fixed action，只用训练折选择；
3. random matched-cost action；
4. earliest failure-risk trigger；
5. maximum-risk 或 maximum-disagreement action；
6. per-instance counterfactual oracle。

不在第一阶段加入 RL、Transformer、bandit、额外神经 loss 或新的 calibration backbone。

## 5. 强制指标与 gate

主指标始终是 untouched full-field 上的 raw complex RelL2。必须同时报告：

- selected-policy macro raw complex RelL2；
- 相对 best-fixed 的 paired improvement 与 bootstrap 区间；
- action regret，即 oracle utility 减去所选 action utility；
- oracle-gap closure；
- harmful-calibration rate；
- within-environment utility Spearman；
- endpoint/mask 选择分布；
- error--cost frontier；第一阶段所有非空 action 成本相同。

M1 机制通过标准：OOF learned policy 必须优于 fold-specific best fixed，并关闭至少 30% 的
fixed-to-oracle gap；harmful rate 不高于 best fixed；shuffled feature/action control 必须失效。

论文级标准更严格：需要新的严格环境、至少第二个具有 action headroom 的数据域，以及相对 best fixed
具有统计稳定且实际显著的完整场误差收益。当前 32 个 development environments 只能证明机制。

## 6. 停止线与后续分叉

- 若 expanded oracle 仍只比 best fixed 好不到 10%，瓶颈是 action space 或 calibrator controllability，
  不是 policy capacity；停止加复杂 selector。
- 若 oracle headroom 足够而 ridge 失败，再尝试 action-conditioned set encoder 或低秩双线性 state--action
  表示。
- 若 ridge 能排序但收益被校准器上限卡住，下一阶段研究“可消除 disagreement”的表示分解，而不是继续
  优化 failure-risk AUROC。
- synthetic 现有 action ceiling 极低，只作为 negative control；不要求学习策略在无 controllability 的
  域上制造正收益。

## 7. 本阶段可回答与不可回答的问题

可以回答：不同环境中相同成本的 calibration 时机和阵列是否具有不同价值；这些差异能否从 target-free
谱表示中提前预测；risk 是否等同于 action value。

不能回答：任意波动方程上的通用主动学习、长期多步最优策略、真实硬件噪声下的部署价值，以及无限远频率
延拓。多步 sequential calibration 只有在单步 value POC 通过后才重新开放。
