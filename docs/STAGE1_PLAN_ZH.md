# 谱连续算子：第一阶段研究计划

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

## 1. 核心问题

给定同一环境中少量低频复波场及可选的已知介质信息，预测一个未见查询频率的复波场：

$$
[u(x,ω_j)]_{j=1}^{m}, c(x), ω_q
→ u_θ(x,ω_q)
$$

其中查询频率 $ω_q$ 高于所有已观测频率。

整数倍频是标准评测切片，不是模型定义。第一阶段不预设复杂生成模型，也不预设路径
token 一定有效，而是先回答两个可证伪问题：

1. 从单频增加到稀疏低频带，是否显著降低高频场的 raw complex error？
2. 在同样输入、骨干、参数量和训练预算下，传播感知表征是否优于直接复场回归？

## 2. 第一性原理假设

局部波场可近似写成多个传播分量的叠加：

$$
u(x,ω) ≈ Σ_{p=1}^{P} a_p(x,ω)e^{iωτ_p(x)}.
$$

相对稳定的对象更可能是路径包络 $a_p$ 和旅行时间 $τ_p$，而不是干涉后的观测幅值
$|u|$。单频只能观察到相位在 $2π$ 周期下的等价类，多个低频点可能提高旅行时间的
可辨识性。因此 APEX-H 的 harmonic carrier 被视为单主路径、单频极限，而不是最终模型。

## 3. 数据与无泄漏协议

### WaveBench

- 公开 isotropic acoustic 数据，频率为 10、15、20、40 Hz；
- 使用旧项目已经冻结的 split seed 42；
- 第一阶段只使用 development train/validation IDs，不读取 500 个旧 held-out fields；
- 主比较为 20 Hz 单频与 10/15/20 Hz 稀疏低频带预测 40 Hz；
- 由于只有四个离散频率，它主要回答“多频是否提高可辨识性”，不单独支撑连续查询结论。

### Ocean Pekeris

- 复用既有环境生成分布，但生成更密的频率 ladder；
- environment ID 在频率间严格配对，train/validation 按环境切分；
- 输入低频带、训练查询频率和外推查询频率在配置中预注册；
- 既有 96 个 strict environments 不用于第一阶段调试。

一个关键设计量是频率采样间隔。若主传播延迟为 $τ$，相邻频率的相位差为
$2πΔfτ$；仅依赖主值相位时，避免混叠大致要求

$$
Δf < 1/(2τ_{max}).
$$

当前 Ocean 最大传播距离约 750 m、最低水中声速约 1470 m/s，对应
$τ_{max}≈0.51$ s，因此临界间隔约 0.98 Hz。旧 50/100/200 Hz ladder
并不能辨识谱相位斜率；第一阶段额外构造 78/79/80 Hz 的 1 Hz terminal context band，
专门区分“谱连续假设失败”和“频率采样已经 alias”这两种解释。主条件采用最小的
79/80 Hz terminal band；78/79/80 Hz 三点线性拟合作为消融，因为多路径引起的局部
谱曲率意味着“更多频率点”不必然优于最近端的两点导数。

## 4. 最小模型阶梯

M0：matched direct query operator。把可见复场、介质、频率 mask 和查询频率共同输入
同一 U-Net，直接输出查询复场。

M1：carrier-relative query operator。根据多个可见频率估计局部相位随频率的增量，构造
查询频率 carrier，网络只预测 carrier-relative 复残差。仍然只优化最终复场重建损失。

M2 只有在 M1 和多频可辨识性均为正时启动：学习少量延迟/包络 latent components，
通过显式 $e^{iω_qτ}$ 解码。第一阶段不会直接堆到 M2。

## 5. 基线与公平性

- medium-only；
- nearest observed field / amplitude-preserving analytic continuation；
- single-frequency direct U-Net；
- multi-frequency direct query U-Net；
- 参数量匹配的 carrier-relative query U-Net；
- 在 Ocean 上增加频率插值作为 sanity check，频率外推才是主任务。

网络比较必须具有相同的样本 ID、可见条件、参数量级、优化器、更新数和最终 complex
loss。不能把额外低频场当成免费信息：同时报告低频求解数量与估计 solver cost。

## 6. 指标与第一阶段判据

主指标始终为逐物理环境等权的 raw complex relative L2。诊断指标包括 amplitude
relative L2、幅值加权 circular phase error、coherence 和误差随查询频率的曲线。

第一阶段的 go 条件：

1. 两个域中稀疏低频带相对单频输入的 raw error 均有稳定正收益；
2. Ocean 未见高频查询上，传播感知模型相对 matched direct 模型至少有一个可复现的
   实质收益，并且不是以幅值或相位另一项显著恶化为代价；
3. 收益在至少三个随机种子或物理样本 paired bootstrap 下方向稳定。

若第一条失败，说明增加频谱信息没有解决主要可辨识性瓶颈；若第一条成立但第二条失败，
则保留 spectral continuation 任务，放弃当前 carrier 参数化，下一阶段转向 learned
path-token representation。生成式模型只有在重复条件或受控隐藏变量证明条件分布歧义后
才进入计划。
