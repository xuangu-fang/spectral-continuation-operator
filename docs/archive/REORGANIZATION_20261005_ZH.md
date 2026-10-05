# 2026-10-05 重整记录

## 此次完成什么

研究入口从开放式“通用表征/谱外推/主动查询”收敛为三个 Wave as Prompt 任务：
同频空间补全、同频新源预测、测量校准近似模拟。

- 根 README 缩为58行；三张问题卡各自给出一句话、简单公式、学习假设、数据、基线、下一步和停止条件。
- 研究记忆整理14条正负证据与4条科学 insight，所有关键数字保留出处与信息条件。
- 数据地图列出11组资产及 A/B/C 适配边界；机器目录索引25个代码组件、14条证据。
- 加入只读检索/检查工具：不导入训练模块、不读数组、不访问 GPU。
- 12份旧入口保存全文快照；94份旧技术文档/报告加历史提示，正文保持不变（末尾空白除外）。
- 原报告、图、配置和 checkpoint 路径均可继续复现；不因重新命名故事而改变旧实验身份。

## 刻意没有做什么

没有移动或删除原始数据、checkpoint、实验模型模块、配置和 artifact；
没有启动训练、生成数据、读取封存真值，也没有宣称新路线已达到某个 gate。
源代码重组采用**分类导航与可检索组件目录**，而不是移动模块破坏 import/checkpoint。
已有研究实现的数值行为未改动。

没有把 A/B/C 预定成三篇论文，也没有把“Prompt”当作新颖性证明。
若后续要发不同文章，必须有不同机制与独立证据；相同实验换标题不算新工作。

## 关键纠正

1. 旧源码、配置、artifact 入口还在推荐 active query，现已全部换成新路线。
2. WaveBench 的旧正信号是介质识别，不是相位场重构。
3. OpenFWI 本机介质与波形不配对；The Well 的 source/environment 身份不可靠。
4. Helmholtz 512训练介质≠512场样本；三频四线切片也≠同频 A 协议。
5. 独立 LF/HF 小 pilot 的受限 AR oracle 失败不是普遍信息论上限。
6. HarmoCore 等已有工作已覆盖相近任务；新意要落实到学习机制和精度/泛化证据。

## 检查结果

- 32项相关单元测试通过：新目录工具、既有公共输入提取、独立多保真协议。
- 新工具 Ruff 检查与格式检查通过。
- 当前文档、归档快照、历史索引的本地 Markdown 目标存在性检查通过；未逐一验证旧网页锚点。
- 40个科学公式 SVG 与文档数学标记检查通过；既有公式没有回退为不兼容宏。
- 11组数据资产列出的本机路径均存在；只检查存在性，不等于重新验证数组内容与物理精度。
- 94份历史正文逐份对比重整前 Git 版本，无实质内容改写。

检查命令：

```bash
python3 scripts/research_catalog.py check --local-data
.venv/bin/python -m pytest tests/test_research_catalog.py \
  tests/test_prepare_helmholtz_sparse_dev.py tests/test_paired_fidelity_protocol.py -q
.venv/bin/ruff check scripts/research_catalog.py tests/test_research_catalog.py
.venv/bin/ruff format --check scripts/research_catalog.py tests/test_research_catalog.py
.venv/bin/python scripts/render_spectral_science_math.py --check
```

重整前 Git commit：`233b2fc323934896627412b54e7784ef4d588310`。
[原文本 hash / 入口快照](20261005_before_three_routes/MANIFEST.md) ·
[完整文件导航](INVENTORY_20261005.md) · [返回当前方向](../wave_as_prompt/README.md)。
