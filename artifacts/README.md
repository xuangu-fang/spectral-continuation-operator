# Artifact 索引与保存策略

## 当前正文级结果

`wave_operator_representation/` 是当前唯一 active artifact collection：

- `summary/wave_representation_main.png`：检索、介质恢复、source budget 与 stress-test 主图；
- `summary/wave_representation_examples.png`：隐藏介质恢复定性结果；
- `summary/summary.json`：机器可读的主结果、三 seed 聚合与 failure boundary；
- 子目录中的 JSON / CSV / PDF：matched ablation 和绘图源数据。

完整解释见
[`reports/WAVEFIELD_REPRESENTATION_POC_ZH.md`](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

## 历史冻结 collections

其余目录记录早期 spectral continuation、continuous state、sparse calibration、reliable/actionable
horizon 与 paired multi-fidelity 实验。它们不再构成当前主页故事，只作为负结果、研究决策和复现证据
保留。对应结论见 [`docs/archive/INDEX_ZH.md`](../docs/archive/INDEX_ZH.md)。

## 保存规则

大型 checkpoint、逐样本数组与可再生中间结果留在 attached storage，并由 `.gitignore` 排除。Git 只
保存配置、manifest、汇总 JSON / CSV 和论文级 PNG / PDF。每个正式汇总应记录父数据与 checkpoint
hash；任何手工挑选的可视化都必须能追溯到机器可读 summary。
