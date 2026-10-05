# PBNO M4：论文剩余风险的分阶段解决计划

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 日期：2026-09-26
> 状态：已执行；正式结论见 `reports/PBNO_M4_RISK_RESOLUTION_ZH.md`

## 0. 执行结果

- M4-A 通过：不是删除 instances，而是用 rank-16 spectral coefficients 压缩每个 operator value；
  value-bank storage 降低超过 99%，误差退化小于 1.5%。
- M4-B 通过：hard LOO 与 shared risk head 未过，但 1,665 参数的连续 prior adapter 在三 prompt seeds、
  三域、K=1/2/4/8 的 12/12 设置中均优于最佳单分支；OpenFWI/Ocean K=2 相对 global baseline 改善
  17.25%/21.85%。NO-GO 只适用于 hard routing，不适用于连续 composition。
- M4-C 通过：独立 structural lattice 域在 K=1/2 改善 10.31%/8.18%，bootstrap 下界为正。
- conditional flow gate 没有新增超过 15% 的证据，保持关闭。

## 1. 目标不是再加 backbone

M3 已证明 prompt-conditioned posterior kernel 在 OpenFWI 与 Ocean 的 strict joint source--frequency query
上有效，而 Deepwave 更适合 neural latent transport。当前剩余问题不是“模型是否还能更深”，而是三件更接近
论文可信度的问题：分支能否由 prompts 自己选择、经验 memory 是否能压缩、结论能否离开现有三个数据处理链。

## 2. M4-A：operator dictionary 压缩

训练系统的完整 response operator 先投影到 train-only global latent，再用 greedy farthest-point sampling 或
固定随机子集选择 prototypes。测试时仍根据当前 prompt mask，重新计算 query 与每个 prototype 的 Bayesian
posterior token；因此 prototype selection 不读取测试 prompts，neighbor retrieval 不读取目标 response。

冻结 gate：OpenFWI 与 Ocean 至少保留 50% 或更少的 train operators，test raw complex RelL2 相对完整 bank
退化不超过 2%。若失败，报告 12.5%--100% 的 accuracy--storage Pareto，不改变标准。

## 3. M4-B：prompt-only prior routing

候选分支保持 M3 不变：global Bayesian mean、posterior-kernel operator，以及已有 neural latent transport。
路由信号只能来自 prompts：leave-one-prompt-out response error、posterior latent energy、nearest-prototype
distance 与 attention entropy。禁止读取 domain label、目标 response 或 test-set aggregate error。

第一版只实现可解释的 leave-one-prompt-out selector；K=1 没有可留出的 prompt，使用 validation 冻结的
confidence rule。只有该规则未达 gate、且诊断显示连续置信度有信息时，才学习一个极小 calibration head。

冻结 gate：三域均不比各自 M3 validation-selected branch 退化超过 2%；至少两个域 K=2 相对 strongest
baseline 保持 10% 以上改善。

## 4. M4-C：第四验证域

优先级依次为：具有真实 source/receiver metadata 的公开 multistatic 数据；独立 elastic/structural wave
solver；现实噪声 protocol。新域必须保持 `system × source × frequency × receiver` 接口、system-held-out
split 和 strict Q3，不得因方法表现而更换主指标。

最低 gate：相对 validation-selected strongest baseline 改善至少 8%，paired-system bootstrap 95% 下界大于
零。若不通过，必须给 posterior oracle、effective rank 与 prompt identifiability 审计，再决定调整 query
跨度或停止。

## 5. 执行顺序和停止规则

1. 先做压缩，因为它只改变 bank，不需重训；
2. 再做 prompt-only routing，因为它复用 M3 全部分支；
3. 最后生成或接入第四域，先跑线性/global 与 kernel，不预设 neural 一定更强；
4. conditional flow 只有在第四域出现至少 15% conditional sample-oracle gap 时才允许启动；
5. 所有正式数字使用三组 prompt-order seeds、逐 physical system paired bootstrap、raw complex RelL2。

输出包括正文级主表、storage--accuracy 图、route confusion/失败案例、第四域数据 provenance、完整复现命令，
并同步更新 README、当前状态、实验账本与代理入口。
