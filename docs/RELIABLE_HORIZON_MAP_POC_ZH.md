# Reliable-horizon Map 与 Sequential Calibration POC

日期：2026-09-17

状态：实验前协议。本文冻结第一阶段问题、数据使用、指标和停止线；后续结果不得反向修改 gate。

## 1. 核心问题

现有实验已经说明，复杂波场频率外推通常只有有限有效范围，而且这个范围随实例、空间位置、模型和
误差容忍度变化。下一阶段不先追求更远的无条件外推，而是回答：

> 给定可见频带、冻结模型的候选预测以及少量 calibration 信息，模型能否在不知道查询真值时，
> 可靠判断哪些未来频率和空间区域仍可使用；当预测即将失效时，能否用最少的新观测延长有效范围？

这里的 `map` 有两个层次：

1. **实例—频率风险图**：每个新环境、每个查询频率有一个失败概率或误差上界；
2. **空间风险图**：在同一频率内，对空间 patch 给出局部失败风险。

第一层是主 POC；只有第一层通过 gate，才训练第二层和 sequential policy。

## 2. 为什么先做可靠性，而不是继续堆预测器

现有正负结果共同指向三个事实：

- structured carrier/pole 方法在近程改善明显，但收益会随频率距离衰减并与强基线交叉；
- 同一个频率上，不同环境的误差跨度很大，固定的全局 horizon 不可靠；
- calibration 能修正实例特定相位状态，但不能保证补回未来新模态。

所以当前更可证伪的 AI 问题不是“能否永远预测”，而是“能否知道何时不该继续预测”。这也把
uncertainty estimation、selective prediction、operator learning 和 active acquisition 连接起来。

## 3. 正式定义

第 $e$ 个实例在查询频率 $f$ 的真实复场误差为：

$$
E_e(f)=
\frac{\|\widehat u_e(f)-u_e(f)\|_2}{\|u_e(f)\|_2}.
$$

给定容忍度 $\eta$，失败标签为：

$$
Y_e(f;\eta)=1[E_e(f)>\eta].
$$

真实可靠 horizon 定义为从可见边界 $f_0$ 开始、所有连续查询仍未越界的最大距离：

$$
H_e(\eta)=
\max\left\{
h: E_e(f)\leq\eta,
\quad f_0<f\leq f_0+h
\right\}.
$$

模型不直接看 $u_e(f)$，只允许读取：

- 可见频带复场及环境输入；
- 冻结预测器在查询频率的预测；
- 多个冻结预测器或 seed 之间的 disagreement；
- 已发生 calibration 的观测残差；
- query offset、幅值、谱斜率、谱曲率和空间位置等 target-free 特征。

输出为失败概率 $\widehat p_e(f)$、误差上界 $\widehat U_e(f)$，以及从这些量导出的
$\widehat H_e(\eta)$。

## 4. M0 数据与无泄漏协议

### 4.1 Ocean

- 主协议：90--100 Hz 可见，101--106 Hz 为已有 development calibration，107--125 Hz 为查询；
- 基础预测器：冻结的三 seed hyper-pole 与 stable recurrence；
- 当前 32 个 validation environments 只用于开发期交叉验证；
- 已登记但未打开的 strict environments 在方法、阈值和 gate 冻结前不得生成或读取场标签。

### 4.2 合成域

- 主协议：可见频带沿用现有 endpoint-pole 实验，19--24 Hz 为查询；
- 基础预测器：三 seed pole 与 recurrence；
- 合成域用于检验方法是否只记住 Ocean 的传播距离或环境参数。

### 4.3 切分

风险校准和风险评估必须按物理实例切分，不能把同一环境的不同频率分到两侧。开发阶段使用固定
environment-hash folds：一部分环境拟合轻量 risk calibrator，另一部分环境评估，轮换报告均值与
fold 间范围。最终 strict evaluation 只运行一次。

### 4.4 禁止的信息

- 查询频率真值及其任何派生幅值、相位、曲率；
- 使用未来 query error 选择 risk feature 或超参数；
- 将同一环境的其他查询频率标签泄漏进测试 fold；
- 在 sequential evaluation 中读取尚未购买的频率或传感器。

## 5. M0：先建立真实 horizon atlas

现有逐环境 metric vectors 足以先回答四个问题：

1. 不同 $\eta$ 下，首次越界频率是否具有足够实例差异；
2. 单个环境的 error curve 是否大体单调，还是频繁越界后恢复；
3. seed 平均后 horizon 是否稳定；
4. Ocean 与合成域是否存在共同的归一化 error--distance 形态。

预注册容忍度为 raw complex RelL2 $\eta\in\{0.4,0.5,0.6\}$，其中 0.5 为主阈值，另外两个只做
sensitivity analysis。若某数据域在一个阈值下超过 90% 样本集中于同一 horizon bin，则该阈值不用于
分类指标，但仍保留连续误差和 horizon regression 指标。

## 6. M1 方法阶梯

### M1.0 无学习基线

- 仅使用 query distance；
- 可见频带最后两点的幅值、相位增量和曲率；
- frozen model 的递推稳定因子或 pole magnitude；
- 三 seed prediction disagreement；
- hyper-pole 与 stable-recurrence 的跨架构 disagreement。

这些基线回答：horizon 是否只由距离决定，以及 disagreement 是否已包含足够风险信息。

M1 的主部署预测定义为三个独立 seed 复场预测的逐点复均值；风险标签是这个 ensemble-mean field 的
raw complex RelL2 是否越界。单 seed 结果作为敏感性分析。这样 disagreement 与被保护的预测来自
同一组冻结模型，并且在查看 risk 结果之前固定预测规则。

### M1.1 轻量标量校准

在独立 calibration fold 上使用 logistic regression、isotonic regression 或 conformal residual
quantile，把 training-free score 映射为失败概率或误差上界。这里不训练新的场生成器。

### M1.2 轻量 learned risk head

只有 M1.1 仍有明显可提升空间时，训练小型 permutation-invariant risk head。输入由冻结场模型产生，
主模型权重不更新。输出需要对 query distance 保持弱单调约束，减少远处风险反而降低的异常行为。

### M1.3 空间风险图

将场划分为固定 patch，对每个 patch 预测能量归一化复场误差是否越界。局部图只有在实例—频率级
gate 通过后启动，因为全局风险都无法排序时，增加空间分辨率只会放大噪声。

## 7. 主指标

### 7.1 风险排序

- failure AUROC 与 AUPRC；
- risk score 和真实连续误差之间的 Spearman correlation；
- top-risk 20% 是否覆盖大多数 catastrophic failures。

### 7.2 概率校准

- Brier score；
- expected calibration error；
- reliability diagram；
- 在目标覆盖率 50%、70%、90% 下的实际失败率。

### 7.3 Selective prediction

模型只接受预测风险最低的一部分 query：

- risk--coverage curve；
- area under risk--coverage curve；
- coverage at guaranteed error budget；
- 与 distance-only、ensemble variance 和 oracle ranking 对比。

### 7.4 Horizon

- horizon MAE，单位为 Hz 和 query steps；
- unsafe overreach rate：预测 horizon 超过真实 horizon 的比例；
- conservative gap：真实 horizon 减预测 horizon；
- 不同 $\eta$ 下结论是否一致。

相比对称 MAE，unsafe overreach 更重要：过早请求校准只增加成本，过晚请求会输出不可靠场。

## 8. M1 Go/No-Go gate

在两个数据域分别报告，不允许只汇总平均。进入 sequential calibration 至少满足：

1. 主阈值 $\eta=0.5$ 的 failure AUROC 不低于 0.75；
2. 连续误差 Spearman correlation 不低于 0.50；
3. 相比 distance-only baseline，至少关闭 50% 的 oracle risk-ranking gap；
4. 经 calibration 后，名义 90% safe coverage 的实际覆盖不低于 85%；
5. unsafe overreach rate 不高于 15%；
6. 至少一个 target-free score 在 Ocean 和合成域方向一致。

若只在一个域通过，则结论降级为 domain-specific reliability，不进入通用方法叙事。

**2026-09-17 指标修正记录：** 原协议把第三项写成 raw risk--coverage AUC 相对改善 15%。M1 pilot
后发现，即使直接用真实误差做 oracle 排序，Ocean 和 synthetic 上这一指标的理论上限也分别只有
约 13.4% 和 10.5%，因此原 gate 数学上不可达到。这里将其改为“关闭 oracle 可达 gap 的比例”，
同时永久保留并报告原始相对改善值。此修正在任何 sequential calibration 实验之前完成，但发生在
M1 development pilot 之后，不能声称是完全结果盲的预注册。

## 9. M2 Sequential calibration

### 9.1 最小部署过程

从可见边界开始按频率向前查询。模型持续输出场和风险。当下一频率的上界超过阈值时，系统在该频率
购买一次固定稀疏阵列观测，更新实例状态，再继续外推。评价频率若被购买，则从未观测 query 指标中
移除，并计入 acquisition cost。

### 9.2 两阶段复杂度

第一阶段只选择 **何时、在哪个频率校准**，传感器阵列固定。通过后再研究 risk map 指导的空间
sensor selection。这样可以区分“horizon 判断有效”和“空间主动采样有效”。

### 9.3 Baselines

- no further calibration；
- 固定每 1、2、4 或 6 个频率步校准；
- 相同预算下随机选择 calibration frequency；
- distance-only trigger；
- ensemble-disagreement trigger；
- oracle trigger，作为不可部署上界。

### 9.4 主结果

主图为 error--acquisition-cost frontier，而不是单一终点误差。另报告：

- 达到指定全频带误差所需的 complex observations；
- calibration 次数与频率位置；
- 未观测 query coverage；
- 每次 calibration 带来的 horizon extension；
- 策略在不同实例上的最坏四分位表现。

## 10. M2 Go/No-Go gate

Risk-guided policy 需要在两个域中满足：

1. 匹配 acquisition cost 时，macro raw complex RelL2 相比最佳固定 schedule 改善至少 10%；或
2. 匹配 error budget 时，complex observations 至少减少 20%；
3. 优势不能只来自把困难 query 全部购买并从评价中删除，必须同时报告 query coverage；
4. 最坏四分位环境不劣于最佳固定 schedule 超过 5%；
5. shuffled-instance calibration 必须显著弱于正确实例 calibration。

未通过时，保留 Reliable-horizon Map 作为独立科学结果，不继续训练复杂 RL policy。

## 11. 当前最小实现顺序

1. 从现有三 seed metric vectors 生成 horizon atlas 和标签稳定性报告；
2. 扩展冻结 evaluator，保存 target-free disagreement、幅值和谱曲率特征；
3. 实现 distance-only、ensemble disagreement、isotonic/conformal baseline；
4. 在 development folds 上完成可靠性图和 selective-risk 曲线；
5. 通过 M1 gate 后，实现 fixed-sensor sequential replay；
6. 最后才考虑 spatial risk head 和主动 sensor selection。

## 12. 预期贡献边界

POC 成功时可以支持：

> 复杂波动场频率延拓存在可学习、实例相关的可靠 horizon；一个经过校准的预测状态能够在误差真正
> 爆发前识别风险，并用少量顺序观测延长可用频带。

它不能自动支持：

- 对任意远频率的可靠恢复；
- 严格的 worst-case 数学保证；
- 任意波动方程或任意传感器配置上的通用性；
- 生成模型必要性；
- 未经 strict environments 验证的部署声明。
