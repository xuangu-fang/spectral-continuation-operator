# APEX/APEX-H 冻结证据与归因

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

本文档只整理截至 2026-09-14 已完成的结果，不把旧结果重新解释为谱连续实验。旧仓库
revision、原始 summary、dataset/checkpoint hash 将由 provenance manifest 绑定。

## 1. 可信正结果

### 稀缺高频标注下的 APEX-H R1

| 数据域与任务 | Direct raw RelL2 | R1 raw RelL2 | 相对下降 | 证据级别 |
|---|---:|---:|---:|---|
| WaveBench 20→40 Hz，K=1 | 0.86688 | 0.80205 | 7.48% | 一次性 held-out，三 seed |
| Ocean 100→200 Hz，K=1 | 0.80793 | 0.73190 | 9.41% | 预注册 strict，三 seed |

WaveBench 的 paired absolute-improvement 95% CI 为 [0.06210, 0.06753]，98.6% 的
500 个物理样本改善；Ocean 为 [0.06892, 0.08371]，96/96 个环境改善。幅值和相位
分解指标方向一致。这说明局部相位增量与 carrier-relative residual 在数据稀缺区间具有
稳定 inductive-bias 价值。

### 数据量效应

R1 的优势随目标高频标签数量增加而缩小：WaveBench 从 K=1 的约 7% 降到充分监督时
基本持平；Ocean 也在 K=100 后由 Direct 追平。这归因于 sample efficiency，而不能
归因为更强的充分监督算子。

### 非二倍频迁移

旧 development 配置在 Ocean n=2、3、4 上观察到 transferred R1 相对 scratch R1
分别约 17.67%、20.34%、15.09% 的改善。它排除“只能平方”的狭义解释，但由于配置与
主实验不同，不能与 locked 数值混成统一曲线，也不能声称任意倍率已经验证。

## 2. 已被否定或未被支持的解释

- Fully supervised source-operator：WaveBench 上 harmonic/direct 只比 raw 好约 4.37%，
  R1 更差；Ocean injective lift 的 source error 也差于 raw。因此尚不能声称 APEX-H
  本身是普遍更强的 fully supervised operator。
- Selective gate：WaveBench/Ocean K=1 的三参数 gate 为 0.8285/0.7249，未稳定优于
  R1；小型 local adapter 也没有改善 R1。pixel oracle 的 0.7130/0.6033 只是不可部署
  上限，不能作为方法结果。
- 更深网络、identity bridge、受限 adapter 和更换等价 loss 均未产生稳定提升。
- 现有结果没有证明 flow matching 必要，也没有证明条件高频分布是多模态的。

## 3. 对新课题的因果启示

最保守的归因是：R1 对单频相位外推提供了有用但不完整的结构，其约 8%–10% 上限可能
来自多路径干涉与单频 phase aliasing，而不只是优化不足。Direct/R1 的 pixel oracle 有
明显 headroom，但低频幅值和预测 disagreement 学不到跨环境 gate，也提示缺失的是更有
辨识力的条件信息或传播表示。

因此谱连续第一阶段首先改变信息条件：从一个低频场扩展到稀疏低频带，再判断是否需要
更复杂的 latent propagation model。旧 locked/strict 数据从此只作为冻结证据，不用于
新课题调参。
