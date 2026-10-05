# 配置入口：旧协议保留，新路线尚未启动训练

当前任务是 [A 补全 / B 新实验 / C 校准](../docs/wave_as_prompt/README.md)。
本目录已有 JSON 是各历史实验的冻结记录，**不是下一轮应顺序运行的任务列表**。
目录中的旧 active、horizon、pending、M0 等名字不代表当前优先级。

## 怎么复用

1. 从[问题卡](../docs/wave_as_prompt/README.md)确定输入输出和评价条件。
2. 在[机器目录](../docs/wave_as_prompt/research_catalog.json)与[历史清单](../docs/archive/INVENTORY_20261005.md)
   找到相关数据/协议；核对原报告，不只看配置名字。
3. 若修改 source/frequency、split、观测预算、metric 或先验信息，新建配置和输出目录，
   不覆盖旧配置，不把历史分数作为新协议结果。
4. 在生成/读取独立确认真值前冻结配置；所有已使用系统继续标记为 DEV 或历史 test。

可优先参考的旧配置：

- [跨域数据](cross_domain_response_query_20260924.json)：B 的数据来源与轴。
- [WaveBench](wavebench_representation_m1_20260923.json)：读取/split 参考；不是相位重构实验。
- [安全 Helmholtz 切片](coherent_helmholtz_multiscale_sparse_dev_20261004.json)：公共输入白名单。
- [独立多保真 pilot](ocean_multifidelity_pilot_20260916.json)：C 的旧负结果，不是现成训练集。

本次不创建伪装成“已冻结新实验”的 A/B/C 训练配置。
