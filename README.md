# 高精度复波场的连续表征与查询

当前问题：能否学习一种保留相干相位的连续表示，让复杂波场的重建与频率查询不再受固定“图像像素”束缚？

海洋水声是主要应用，异质介质声学用于检验一般性。现在先解决表示精度，再研究共享预测与连续谱；不推进主动选点、RL 或盲远频率外推。

## 从这里开始

| 想了解什么 | 阅读入口 |
|---|---|
| 最新结果、方法细节、正负结论与图 | [连续复场表征进度报告](reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md) |
| 当前故事和下一阶段的取舍 | [当前主线与路线图](docs/CURRENT_STORY_AND_ROADMAP_ZH.md) |
| 本次24小时目标、阶段和门槛 | [执行协议](docs/COHERENT_CONTINUOUS_REPRESENTATION_24H_ZH.md) |
| 数据怎么生成、哪些能复用 | [数据来源](docs/COHERENT_DATA_PROVENANCE_ZH.md) |
| Agent接手、避免重复实验 | [AGENTS](AGENTS.md)、[当前状态](docs/CURRENT_STATUS_ZH.md)、[实验账本](docs/EXPERIMENT_LEDGER_ZH.md) |

## 已经知道什么

- 高精度空间表示可行；逐场拟合、共享编码、未观测频率生成必须分开评价。
- 异质介质中，学习局部相位坐标与包络度量的连续核出现稳定开发信号；三seed局部误差约1.07%–1.20%。但现场优化同一核族可到0.72%，learning目前更像快速推断初值，尚非最优精度方法。极密采样下三次插值更强。
- 均匀水层Ocean中，加入圆柱扩散补偿的经典波基比当前神经模型更准。不能宣称learning普遍必要。
- 空间chart很准，不代表跨频插值更好；传播模态分解在Ocean连续谱查询中有较强正信号，但它目前是经典机制对照，不是AI方法创新。

全部新结果仍为开发阶段，尚未通过正式跨域方法gate。旧Ocean封存测试集没有打开。完整数字、对照、失败和证据范围见报告。

## 当前不做什么

不为了现代AI标签叠加flow、Transformer、群论或多种loss；不把源区排除后重新定义主指标；不把新频率有空间观测的重建写成无观测频率生成；不把开发环境重新命名为test。

## 历史研究如何查

历史版本和代码保留，不表示仍在执行：

- [本轮前完整README时间线](docs/archive/README_BEFORE_COHERENT_20261003.md)
- [历史报告索引](reports/README.md)
- [历史正负结果与重启条件](docs/EXPERIMENT_LEDGER_ZH.md)
- [归档导航](docs/archive/INDEX_ZH.md)

仓库名 `spectral-continuation-operator` 是历史名称；不等于当前只研究低频到高频外推。
