# 从复波动响应中学习可查询的 Wave-System Representation

> 仓库名 `spectral-continuation-operator` 是历史名称。当前研究主线已经从“直接做远频率延拓”推进到一个
> 更基础的问题：**能否从少量复波动 response views 中识别可复用、可查询的隐藏波动系统表示？**

我们的核心发现很简单：完整复场不适合直接当作普通图片处理。绝对相位会随 source、frequency 和
全局相位基准快速变化，但**归一化幅值与局部复相位增量**可以把“局部能量”和“局部传播”分开表示，
从而更稳定地识别产生这些响应的物理系统。

## 当前主线：In-Context Wave Response Query

当前不再把 frequency 只当作网络的 condition channel，也不以重建某一张 field 为最终目标。基本任务是：

> 给定一个未见波动系统的少量 source--frequency probe responses，在不更新模型权重的条件下，回答新的
> source--frequency response query。

首轮固定 receiver array，将一个 probe 写成 $q=(s,\omega)$，response 是该 receiver array 上的复数向量。
主实验严格区分 frequency-held-out、source-held-out 和二者同时未见的 joint query；使用现有
160-system × 9-source × 8-frequency dense archive 生成可复现 episodes。先做 identifiability、强
interpolation/low-rank/retrieval baselines 与 active-oracle audit，再训练最小 DeepSets/CNP query model。

Generative belief、flow matching、active policy、RL、可变 receiver、3D 与 multibranch router 都有明确前置
gate，不进入第一阶段。完整 formulation、24 小时执行顺序与 stop rules 见
[In-Context Wave Response Query 计划](docs/IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)。

```text
Wave as Query → Wave as Prompt → Wave as Belief → Wave as Action
```

## 上一阶段：多分支相空间 POC

新的 oracle 审计证明：复杂波场局部确实存在单个 phase advance 无法表达的多分支信息，但尺度必须按
波长定义。约 1.5-wavelength 的局部窗口中，K=4 相对 K=1 在 partial/closed boundary 上将 held-out
complex error 降低 26.3%/25.7%；固定 33-pixel 窗口却只有约 6%，说明“局部”的物理单位是波长而非
像素。

由此设计的 two-scale complex packet Gram quotient 一个共同 global phase，同时保留分支之间的相对相位。
在三 seed synthetic 任务中，它在单频观测预算下相对现有五通道 token 改善 12%--18%；在 64→128
boundary shift 上，open/partial/closed 的逐系统改善为 5.1%/11.1%/11.7%，30/30 个系统同向。由于困难
条件仍未达到预注册 15% gate，当前结论是**oracle 机制成立、fixed 69-channel 实现只关闭部分 gap**，不把
它提升为论文主方法。

完整公式、尺度诊断、三 seed 表格、负结果与 stop decision 见
[多分支相空间 MPS-1 报告](reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)。

![多分支相空间 POC](artifacts/multibranch_phase_space_mps1/summary/multibranch_phase_space_summary.png)

## 独立复现：公开 WaveBench

频率感知表示已在第二个、独立的公开数据集 WaveBench 上完成三随机种子 matched comparison。同一个
shared set operator 接收任意 1--4 个频率响应并恢复产生它们的 wavespeed field；所有表示使用相同网络、
split 和训练预算。

| 观测频率数 | Raw complex | Amplitude only | Amplitude + phase advance | 相对 raw 改善 |
|---:|---:|---:|---:|---:|
| 1 | 0.3024 | 0.3807 | **0.2794** | **7.6%** |
| 2 | 0.2422 | 0.2891 | **0.2140** | **11.6%** |
| 3 | 0.2077 | 0.2398 | **0.1814** | **12.7%** |
| 4 | 0.1811 | 0.2088 | **0.1584** | **12.5%** |

完整四频的逐系统 paired bootstrap 改善为 12.52%，95% 区间为 12.10%--12.93%；普通 physical RelL2
从 2.98% 降到 2.61%。结果跨全部 seed 和 budget 同向，但没有通过预注册的 sparse-budget 15% 强效应
gate。因此它建立的是**可复现的跨数据集 measurement-efficiency prior**，不是已经完成的通用新架构。
完整归因见 [WaveBench M1 报告](reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)。

![WaveBench 独立验证](artifacts/wavebench_representation_m1/summary/wavebench_representation_budget.png)

## 上一阶段：可变观测 Wave Set Operator

固定通道 POC 已升级为 shared 4-slot set operator：同一 checkpoint 接收任意顺序和数量的
source-frequency views，参数量不随预算增长。三 seed 结果显示，full-context 下 phase token 与 strong raw
接近（0.0533 vs 0.0593）；真正稳定的优势出现在每源只有 1--2 个频率时，相对 raw 降低 24--30%。在
clean-trained 20 dB 测试上为 0.112，raw 为 0.529。

这个结果同时收缩了 claim：phase token 不是 universal OOD winner。clean ellipse 和 128-grid boundary
上 raw 更强，10 dB 仍失败；instance-contrastive pretraining 也没有提高 full-label 最优精度。完整结果见
[Wave Set Operator M0 报告](reports/WAVE_SET_OPERATOR_M0_ZH.md)。

![可变观测预算结果](artifacts/wave_set_m0/summary/wave_set_budget_scaling.png)

![波动场表征主结果](artifacts/wave_operator_representation/summary/wave_representation_main.png)

## 当前最强结果

主实验在 160 个二维变速声学系统上进行，按介质严格划分 train / validation / test。检索任务的两组
response views 在 source 和 frequency 上都不重叠；OOD 测试使用训练中未见过的 ellipse geometry。

| 表示 | 跨视图系统检索 Top-1 | IID 介质恢复 RelL2 | OOD ellipse RelL2 |
|---|---:|---:|---:|
| Raw complex | 38.9% ± 2.0% | 0.3588 ± 0.0120 | 0.4637 ± 0.0279 |
| Amplitude only | 61.1% ± 3.9% | 0.2478 ± 0.0071 | 0.4407 ± 0.0283 |
| Amplitude + local phase advance | **100.0% ± 0.0%** | **0.0504 ± 0.0008** | **0.0955 ± 0.0020** |

相对 raw complex，结构化 token 将 IID 和 OOD 恢复误差分别降低 **86.0%** 和 **79.4%**。这支持的
不是“相位不可迁移”，而是更精确的判断：

> **绝对相位难迁移；经过 gauge-invariant 局部化后，相位 advance 是很强的可迁移传播表征，并与
> amplitude 互补。**

![隐藏介质恢复示例](artifacts/wave_operator_representation/summary/wave_representation_examples.png)

## 表示是什么

对每个复场 `u`，当前最小表示只有 5 个实通道：

```text
log-normalized amplitude,
real/imag horizontal phase advance,
real/imag vertical phase advance
```

相位增量直接由相邻位置的复共轭乘积计算，不做 phase unwrapping，不需要已知 source location、解析
travel time 或 eikonal solver。低幅值区域使用 amplitude gate 抑制不可信相位；跨分辨率时，增量的
offset 按固定物理距离而不是固定像素数定义。

## 当前结论边界

这是一项有力的 representation POC，**还不是完整的通用波动生成器**。

- 已成立：在当前二维 acoustic 数据上，该表示能跨 source、frequency 和未见 geometry 识别系统并恢复
  隐藏空间结构；中高 SNR 下局部 phase 明显补充 amplitude。
- 关键负结果：RBF 坐标插值、简单 eikonal demodulation 和少源 Nyström completion 的 complex error
  接近或超过 1；直接预测未见 source-frequency 的完整复场仍约为 0.93--0.97。
- 尚未通过：10 dB 噪声、partial / closed 强反射边界、真实部分观测、第二 PDE 数据域，以及可用精度的
  unseen-query field generation。
- 因而当前不主张“通用波动基础模型”或“任意远频率恢复”。

## 已检验的新问题：从单分支 phase advance 到多分支相空间 token

现有 token 可以看成局部传播关系的一阶、单尺度、单分支近似。同一位置存在直达、反射和绕射分量时，
一个相邻复共轭乘积会把多个传播方向压成单个统计量。MPS-0/1 已验证：多分支信息的 oracle headroom
存在，但 fixed packet Gram 的三 seed downstream 增益仍低于 promotion gate。后续不再无条件增加固定
filter、半径或手工 phase channels。

完整科学命题、传统理论与本项目新增证据的边界、相关论文以及 24 小时最小 POC 见
[多分支相空间波表征](docs/MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md)，执行结果见
[MPS-1 报告](reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)。

## 后续研究边界

1. **主攻 response query**：先证明少量 complex responses 能在 inference time 支持 joint source-frequency
   query；不再以 medium recovery 或 dense field reconstruction 替代该任务。
2. **停止 fixed front-end enumeration**：当前 69-channel Gram 是机制 POC，不再扫更多半径、方向和 edge；
   多分支只在 Q3/Q4 multipath failure 明确后以 compact learnable tokens 重启。
3. **Belief 和 Active 均后置**：只有条件多模态与 active oracle headroom 得到证据后，才进入 flow 或 policy。

稀疏系数/场重构继续作为历史参考，不切换为当前主任务。

## 推荐阅读顺序

1. [当前状态：一页式结论与决策](docs/CURRENT_STATUS_ZH.md) — 五分钟了解现在做到哪里、相信什么、
   下一步为何这样做；
2. [In-Context Wave Response Query 计划](docs/IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md) — 当前 active
   formulation、Q1/Q2/Q3、oracle audits、模型顺序和 gates；
3. [多分支相空间 MPS-1 报告](reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md) — 最新已完成 oracle、方法、
   三 seed 正负结果与 gate；
4. [多分支相空间波表征](docs/MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md) — 冻结科学命题、相关工作与
   原始 POC 协议；
5. [WaveBench M1 报告](reports/WAVEBENCH_REPRESENTATION_M1_ZH.md) — 第二数据集正式结果、HarmoCore
   归因与 2D/3D/时空表示路线；
6. [实验账本](docs/EXPERIMENT_LEDGER_ZH.md) — 逐条路线的正负证据、不要重复的实验、重启条件，以及
   仍未充分开发的正信号；
7. [波动场表征学习 POC 完整报告](reports/WAVEFIELD_REPRESENTATION_POC_ZH.md) — 方法、协议、全部消融、
   stress tests 与声明边界；
8. [旧表征阶段冻结协议](docs/WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md) — M0--M4 与旧 gates；
9. [科学备忘录](docs/SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md) — 波动方程、信号处理、
   可预测性和可辨识性的通俗解释；
10. [历史探索归档索引](docs/archive/INDEX_ZH.md) — 旧的谱延拓、校准、horizon 和局部 token 路线，仅保留
   关键正负结论与原始证据入口。

具体命令、数据 hash 和旧实验参数统一放在[复现档案](docs/REPRODUCIBILITY_ZH.md)，不再占据主页。
按目录浏览时可使用 [docs 导航](docs/README.md) 和 [reports 导航](reports/README.md)。

## 当前代码入口

```text
src/spectral_continuation/wave_operator_data.py
src/spectral_continuation/audit_wave_operator_representation.py
src/spectral_continuation/train_wave_operator_identifier.py
src/spectral_continuation/train_wave_operator_contrastive.py
src/spectral_continuation/train_wave_medium_probe.py
src/spectral_continuation/evaluate_wave_medium_ood.py
src/spectral_continuation/summarize_wave_representation_poc.py
src/spectral_continuation/train_wavebench_identifier.py
src/spectral_continuation/summarize_wavebench_representation.py
src/spectral_continuation/plot_wavebench_examples.py
src/spectral_continuation/audit_multibranch_phase_space.py
src/spectral_continuation/audit_wavelength_scaled_phase_space.py
src/spectral_continuation/summarize_multibranch_phase_space.py
```

代码按 `active / supporting / archived` 的完整分类见
[源码索引](src/spectral_continuation/README.md)，配置分类见 [configs/README.md](configs/README.md)，artifact
分类见 [artifacts/README.md](artifacts/README.md)。历史文件暂不物理移动，以免破坏 checkpoint、manifest
和旧报告中的复现路径；它们已从当前接口中明确冻结。

后续研究 Agent 应先遵守根目录 [`AGENTS.md`](AGENTS.md) 中的证据语言、禁重复清单和实验归档流程。

## 最小复核

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

本仓库独立于 `APEX_repro_fsk`。旧仓库只作为原始论文、历史代码和数据来源；新方向的协议、实验与
结论均以本仓库为准。
