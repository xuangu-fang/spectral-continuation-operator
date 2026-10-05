# Wave as Prompt：用少量测量，重建、预测和校准波场

**我们想做的事：让模型看懂少量波的测量，准确回答尚未测到的复数响应，尤其是相位。**

仓库名暂保留 `spectral-continuation-operator`，以兼容旧代码与 checkpoint。
2026-10-05 起，研究主线收敛为下面三条路线，不再泛泛扩展“通用波场表征”。

| 路线 | 一句话任务 | Prompt 是什么 | 当前基础 |
|---|---|---|---|
| **[A · 补全波场](docs/wave_as_prompt/RECONSTRUCTION_ZH.md)** | 测几个位置，把同一次实验的相位场补准。 | 同源、同频的稀疏测量 | 有完整复场和表征线索；新协议尚未验证 |
| **[B · 预测新实验](docs/wave_as_prompt/NEW_EXPERIMENT_ZH.md)** | 看几个声源的响应，预测换一个声源会发生什么。 | 同环境的示范实验 | 有多域数据与强统计基线；学习方法仍有缺口 |
| **[C · 校准近似模型](docs/wave_as_prompt/CALIBRATION_ZH.md)** | 用少量实测，把不够准的模拟结果校正。 | 测量与近似模型的差异 | 有小型配对试验及负结果；成熟配对数据仍缺 |

共同表达很简单：

```text
少量测量 + 要问的位置/实验 + 可选近似模型 → 未测响应
```

三条路线共享数据、复数读出和评价工具，但解决不同问题。**“Wave as Prompt”是统一叙述，不是算法创新本身，也不是三篇论文的保证。**

## 从哪里读

1. **先看故事与节奏：**[当前主线与路线图](docs/CURRENT_STORY_AND_ROADMAP_ZH.md)。
2. **再看经验：**[最值得保留的正负发现与科学 insight](docs/EVIDENCE_TO_OCEAN_WAVE_PROMPT_ZH.md)。
3. **准备动手：**[数据能做什么、不能做什么](data/README.md)；[代码复用地图](src/spectral_continuation/README.md)。
4. **核对成果：**[当前状态](docs/CURRENT_STATUS_ZH.md)；[历史证据入口](reports/README.md)。

不需要先读数百份旧文档。完整档案仍在：[历史索引](docs/archive/INDEX_ZH.md)。
更深的科学讨论按需看[可预测性、可辨识性与校准备忘录](docs/SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)。

## 已有发现，不夸大成新路线成绩

- **表征变量确实可能影响相位精度。** 一个受控波导族中，同样的训练介质标签和低频观测，
  声速表征的高频复场误差约 0.005，而匹配的另一种物理变量表征约 0.081。
  换到另一介质族后优势只有约 3%–6%；这不是普适结论，更不是神经方法已成功。
- **少量响应能帮助预测其他实验。** 海洋、Deepwave、OpenFWI 已有跨系统的新源响应基线；
  但强 PCA/贝叶斯先验很难被已有神经修正稳定超过。
- **压缩得好、能从少量测量推断出来、最终相位准确，是三件不同的事。**
  旧实验已多次说明：增加容量、复杂 loss 或生成模型，并不会自动解决后两件事。

完整数字、适用范围和反例见上面的证据文档。旧论文方向、旧门槛和已使用测试集不自动继承为新结论。

## 当前边界

- A 为主要推进路线；B 复用现成多源数据做小规模并行验证；C 先建立可信的配对模型偏差。
- 首阶段不加入主动选测量、RL、远频外推、跨物理统一 foundation model。
- 相位须有一致的声源/时钟参考；绝对相位不可靠的真实场景必须另定义协议。
- 本次是整理与规划，**没有启动新训练、生成新数据或宣称三条路线已通过论文级验证**。

给下一位 Agent：[AGENTS.md](AGENTS.md)。快速只读检索：

```bash
python3 scripts/research_catalog.py list --route A
python3 scripts/research_catalog.py check --local-data
```

大型数据与 checkpoint 保留本地，GitHub 保存代码、配置、摘要与图；不上传原始数据。
