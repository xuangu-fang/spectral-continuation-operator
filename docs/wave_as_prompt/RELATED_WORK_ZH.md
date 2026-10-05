# 已有工作如何约束我们的新意与基线

> 检索核对：2026-10-05。以下是定位与方法参照，不声称穷尽文献，也不声称本仓库已全部复现。
> 优先记录正式发表来源；只有 arXiv 状态可确认的条目明确写作预印本。

| 工作 / 状态 | 与哪条路线有关 | 已经做了什么 | 我们不能直接声称什么 |
|---|---|---|---|
| [HarmoCore](https://arxiv.org/abs/2609.00679)，2026 arXiv 预印本、用户既有工作 | A | 连续 Functional Tucker 复场表示、core diffusion prior、稀疏观测后验重构 | 不能把“稀疏波场 + 连续低秩 latent + 生成”说成新问题 |
| [In-context operator learning with data prompts](https://pmc.ncbi.nlm.nih.gov/articles/PMC10523630/)，PNAS 2023 | B | 用输入输出 demonstrations，在推理时回答新的算子问题 | 不能把 operator prompts 或免权重更新本身当独有创新 |
| [Few-Shot Audio-Visual Learning of Environment Acoustics](https://proceedings.neurips.cc/paper_files/paper/2022/file/113ae3a9762ca2168f860a8501d6ae25-Paper-Conference.pdf)，NeurIPS 2022 | B | 用稀疏视听信息预测环境中新的声学响应 | 不能声称“几次声学示范预测新位置/实验”从未研究；其视觉信息与我们的输入不同 |
| [Neural Inverse Operators](https://proceedings.mlr.press/v202/molinaro23a.html)，ICML 2023 | A/B/C 的方法参照 | 将 PDE 逆问题建模为 operator-to-function 学习，结合 DeepONet/FNO | 不能将其恢复介质的分数直接与我们的响应误差比较 |
| [On Learned Operator Correction in Inverse Problems](https://epubs.siam.org/doi/10.1137/20M1338460)，SIAM J. Imaging Sciences 2021 | C | 学习近似算子的修正，研究模型偏差对逆问题的影响 | 不能把 learned correction / 物理模型加神经修正当作新范式 |
| [Neural Operator: Learning Maps Between Function Spaces](https://www.jmlr.org/papers/v24/21-1524.html)，JMLR 2023 | A/B 的 baseline 家族 | 学习函数空间间映射的算子架构 | 连续坐标接口或使用 FNO 本身不构成新贡献 |

## 可能真正有增量的三句话（待实验，而非结论）

- A：**让表示专门保留稀疏测量能确定的相位信息，而不是只追求重构训练图像。**
- B：**从少量示范推断环境的生成规律，让紧凑状态生成不必低秩的新源响应。**
- C：**学习可由少量测量辨认的模型偏差，让校准跨环境复用。**

每句都需要具体结构、强基线、机制消融和失败边界支撑。
尤其 A 与 HarmoCore 的差异，必须落实在表示/推断及跨环境证据上，不是换 slogan。

## 怎样公平使用这些基线

文献上的模型只有在输入、输出、训练监督、观测数和计算预算匹配后，才成为公平对照。
比如视觉+声学模型不能直接拿掉图像后引用原论文成绩；已知介质的 FNO 也不能不说明额外信息。
先用本仓库已实现的经典与学习组件建立协议，再决定哪些外部方法值得投入复现。
对“尚未适配”的方法，正文中应标为计划，不写成已验证基线。

## 文献边界

这份核对服务于研究定位，不足以支持“首个”或“目前无人研究”的强新颖性声明。
投稿前需要围绕最终方法重新做系统检索。
