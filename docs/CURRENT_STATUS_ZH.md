# 当前研究状态

> 2026-10-03；目标进行中，窗口至2026-10-04 14:36 UTC。不是最终验收。

## 当前入口

- [故事与路线图](CURRENT_STORY_AND_ROADMAP_ZH.md)
- [执行协议与gate](COHERENT_CONTINUOUS_REPRESENTATION_24H_ZH.md)
- [滚动进度报告：方法、数字、图、失败](../reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md)
- [数据与相位约定](COHERENT_DATA_PROVENANCE_ZH.md)

## 已执行

- R0：Ocean/异质Helmholtz表示、SIREN/Fourier INR、局部波字典与函数SVD控制。
- R1：参考载波、局部chart谱插值、经典模态谱查询。
- R2：共享局部波字典、相位—包络连续核、载波/度量/曲率/全局字典消融；异质核三seed。
- 独立Pekeris细坐标真值、Helmholtz257→513数值收敛与细坐标查询。
- 保留全部强基线推翻早期结论的记录。没有打开旧封存test。

## 尚未完成

- 同核族的逐实例context-only优化已更准（约0.00725 vs神经直接预测0.01196）；attention为0.01317、两步端到端调整为0.00783，均未胜出。局部插值暂收口为强基线，不继续容量扫描；
- 高精度共享表示的正式新系统测试、公平成本与完整频率scaling；
- 学习模型对强经典方法的第二域稳定优势；
- 真正连续谱查询中的可学习核心，而非只靠经典DMD。

不要将上述未完成项写成已有贡献。新正式集尚未生成，当前所有新数字是development。

## 重要限制

均匀Ocean的经典物理基更准；异质介质极密采样时三次插值更准；新频率有空间观测只算同频重建。旧的相位不可辨识性结论只针对其信息条件，不能用于否定新的充分监督表示。

历史数字与详细关闭条件见[实验账本](EXPERIMENT_LEDGER_ZH.md)和[本轮前状态档案](archive/STATUS_BEFORE_COHERENT_20261003.md)。
