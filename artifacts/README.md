# 实验产物：保留证据，不覆盖历史

当前方向：[三条 Wave as Prompt 路线](../docs/wave_as_prompt/README.md)。
本目录中已有的 acquisition、horizon、query、coherent 等产物都是带具体协议的历史结果，
不是自动运行入口，不能仅凭目录名认定是当前最佳模型。

## 资产角色

- `cross_domain_response_query/data/`：B 可复用的 Ocean/Deepwave/OpenFWI 等数组及 manifest。
- `wave_as_prompt_pfto/`、`pbno_*/`：强统计/学习基线、存储与泛化诊断，含重要负结果。
- `coherent_*/`、`learned_wave_*/`：连续复场、物理生成状态和相位机制诊断。
- 其他早期谱模型、calibration、horizon、active 分支：历史复现；不因保留而继续开发。

大型 NPZ/PT/checkpoint 保持本地且由 gitignore 排除；GitHub 保留既有可公开摘要和图。
原始数据许可不因本项目有副本而改变。

**不改老目录、不覆盖旧预测、不把已使用 test 称为未打开确认。**
新路线的输出需新目录、对应新配置和信息白名单。
[数据地图](../data/README.md) · [代码入口](../src/spectral_continuation/README.md) ·
[完整历史文件清单](../docs/archive/INVENTORY_20261005.md)。
