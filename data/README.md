# 数据地图：有哪些资产，哪些任务能做？

> 更新：2026-10-05。A=同频空间补全，B=同频新声源预测，C=测量校准近似模拟。
> 本轮只核对路径/元数据，不读取封存标签，不生成新数据。路径以本机为准，大数组不上传 GitHub。

## 先用这一张表选数据

“可用”表示数据轴支持，不代表新路线模型已验证。“需生成”表示当前配对/监督不够。

| 数据 | 真正存了什么 | A | B | C | 最大限制 |
|---|---|---|---|---|---|
| D01 Ocean dense | 160 系统、52 频率、完整二维复场 | 可重切同频观测 | 文件无多源轴 | 需配对近似模型 | 理想水平不变海洋；已有 32 DEV 反复使用 |
| D02 Ocean multistatic | 512 系统、9 源×4 频×48 接收 | 仅阵列补全 | 可用 | 需生成 | 环境只有 3 个变化参数 |
| D03 Deepwave ring | 1024 系统、16 源×5 FFT 频×64 接收 | 仅阵列补全 | 可用 | 需生成 | 不是完整内部场，频率来自有限时域脉冲 |
| D04 OpenFWI CurveFault-A | 1500 系统、5 源×7 FFT 频×70 接收 | 仅阵列补全 | 可用 | 暂不可直接做 | 本机 velocity 与波形不匹配 |
| D05 Structural lattice | 1024 系统、8 源×6 频×32 接收 | 仅阵列补全 | 可用 | 需生成 | synthetic mass–spring，不是真实结构监测 |
| D06 The Well staircase | 多频完整场、解析相位时间重复 | 单场诊断 | 当前不支持 | 当前不支持 | source/environment 身份不可靠，不能冒称跨系统 |
| D07 WaveBench | 50k 对齐系统、4 频复场和声速图 | 适合公开验证 | 多源轴未确认 | 需生成 | 只有 10/15/20/40 Hz，不是连续谱真值 |
| D08 旧 synthetic operator | 160 系统、9 源×8 频×64² 完整复场 | 可用 | 可用 | 需生成 | toy 与数值精度需审计，旧 test 已使用 |
| D09 距离相关单向波导 | 完整场、训练介质、传播状态等多个子集 | 机制诊断/重切 | 有少量换源诊断 | 可生成受控偏差 | 同一近似 PDE 的多介质族，不是多个物理域 |
| D10 二维 Helmholtz | Gaussian 完整场；多尺度 512 介质+8 DEV 响应 | 可重切；训练场需补 | 新源需生成 | 配对需生成 | 不得把 512 介质标签当 512 响应样本 |
| D11 旧独立 LF/HF pilot | 3 环境、8 模态/全模态独立计算 | 非首选 | 不支持 | 仅小型诊断 | 远频校准失败，不足以训练/验证新方法 |

**首批推荐：A 用 D01+D07，B 用 D03+D04 并以 D02 补充海洋应用，C 先从求解器准备可信配对。**
这些是可行性建议，不是保证数据适合最终论文；相位参考、网格精度与监督许可须先核验。

## 统一注意事项

- 数据集、物理系统、频率、source、receiver、时间步不是同一级独立样本。
- 同系统重新生成更密接收器、换频率或换 mask，仍不是新的独立环境。
- 固定 split 的旧 test/OOD 多数已有历史使用；对新候选不能自动算未见确认。
- train-only 的幅值归一化与 FFT 源谱处理要显式记录；不得用完整测试场的尺度泄漏目标。
- 时间/频率符号、Hz 与角频率、source 幅值/相位均需适配。
  不能不加核对就将所有数组称为同一个 Green's function。
- 本机数据存在不代表允许公开再分发；公开数据的 license 与上游引用要单独核对。
- manifest 为维护者追踪来源，不自动是模型输入。含 seed、真介质参数的 metadata 必须去掉。
- 本目录只放说明；[机器目录](../docs/wave_as_prompt/research_catalog.json)保存路径、限制与可复用代码引用。

## D01 · Ocean dense：完整二维场

路径：
`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz`

数组 `[160,52,48,384,2]`，74–125 Hz 每 1 Hz 一个切片；128 train + 32 已用 DEV。
额外预留的严格确认批次不在这个文件中，不因本轮整理而解封。
空间约 range 50–750 m、depth 1–78 m；水平不变理想海洋，固定源。
早期 metadata 中海底 attenuation 参数没有进入求解器，不应学习其不存在的效果。

细坐标 `coherent_ocean_fine_queries_20261003/` 与 fractional 子集是少量旧 DEV 的直接求解，
不是新增独立环境。相位约定 exp(-iωt)，出射 exp(+ikr r)。
[生成与数值说明](../docs/COHERENT_DATA_PROVENANCE_ZH.md)。

## D02–D05 · 已统一的 source–frequency–receiver 数据

目录：`artifacts/cross_domain_response_query/data/`，均为
`[system, source, frequency, receiver]` 复数数组，有 split、坐标、system ID 与 manifest。

| 文件 | split: train / val / test / OOD | 物理与频率 |
|---|---|---|
| `ocean_multistatic_v1.npz` | 256 / 64 / 128 / 64 | PekerisModeSolver，水深120 m、距离4 km；80/90/100/110 Hz |
| `deepwave_ring_v1.npz` | 512 / 128 / 256 / 128 | Deepwave 0.0.26；96²网格、1 mm；约92.6–194.4 kHz |
| `openfwi_curvefault_a_v1.npz` | 500 / 500 / 500 / 无 | 原 `[500,5,1000,70]` 分片，dt=1 ms，8/10/12/15/18/20/22 Hz |
| `structural_lattice_v1.npz` | 512 / 128 / 256 / 128 | 12×12 mass–spring；6 个角频率，不直接当 Hz |

Deepwave 128/176 receiver 版本位于 `artifacts/wave_as_prompt_pfto/data/`，与64版同系统。
256 nominal receivers 在该离散圆上只有188个唯一 cell，不能用重复位置伪造更密采样。
有限时长 FFT 的频率响应依赖源谱与窗口；不能不经处理当成任意稳态频率真值。

OpenFWI 原文件在 `/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a/raw`。
source 对齐 receiver indices `[0,17,34,52,69]`，已用 train 互易关系核验。
本机 velocity 文件不匹配，禁止喂入模型、作为标签或做近似环境配对。
Structural 的原始阻尼符号约定需与 acoustic 适配，原存档不要覆盖。
[完整来源、hash 与生成命令](../docs/CROSS_DOMAIN_DATA_PROVENANCE_ZH.md)。

## D06 · The Well：保留诊断，不包装成多环境成功

原目录：
`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/helmholtz_staircase_train`

16 频率文件，每个 `[26,50,1024,256]`。时间轴是解析相位重复，不是50个独立场。
当前文件缺 source coordinate；26条 trajectory 的场峰/残差均落同一 cell，
不能可靠识别 source/environment 的变化。adapter 仅读 t=0 并统一复数约定。
可以做单场表示检查，不能支撑 B 或独立系统泛化。

## D07 · WaveBench：公开完整复场

原目录：
`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/wavebench_full_remotezip_20260912`

50k 对齐系统，10/15/20/40 Hz，复场与介质声速；FFCV beton 读取器在
[train_wavebench_identifier.py](../src/spectral_continuation/train_wavebench_identifier.py)。
历史实际使用 8k/500/500、split seed42，三实验种子20260923–25。
旧结果是反推介质，不是 A 的相位重构，需新建 observation/query mask。
[旧报告](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)；
[上游 DOI](https://doi.org/10.5281/zenodo.8015144)。

## D08 · 旧多源完整场 toy

`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz`

160系统、9源、8频、64²复场，112/24/24；另有24个 ellipse OOD 与30个 legacy128-grid系统。
数据与生成代码可复用，但不能把不同网格的插值当新真值，数值收敛和系统身份要再核对。
[代码](../src/spectral_continuation/wave_operator_data.py)。

## D09 · 距离相关波导：机制实验资产

共同根目录：`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/`

| 子目录 | 内容/角色 |
|---|---|
| `coherent_waveguide_development_20261003` | 32 train + 8 DEV；60–140 Hz，完整场 `[9,129,2049]` |
| `coherent_waveguide_training_cache_20261003` | 512 训练介质/模态轨迹缓存；不能直接当512张完整密集场 |
| `coherent_waveguide_generator_query_transfer_20261003` | 旧8 DEV换源/新查询坐标，不是新系统 |
| `coherent_waveguide_frozen_system_audit_20261003` | 64系统两种源，共128文件；旧方法冻结后确认，已评测 |
| `coherent_waveguide_operator_action_20261004` | 旧8 DEV、40源作用矩阵，生成机制/输出秩诊断 |
| `coherent_waveguide_spectral_medium_20261004` | 16×16余弦介质族压力测试；多批使用身份须读原报告 |
| `coherent_waveguide_learned_speed_manifold_confirm_20261004` | 16个taper7.5新系统，曾是事前冻结确认，现在已使用 |

这个 PDE 是**单向传播**，忽略后向散射，顶部压力释放、底部刚性反射，没有可穿透海底。
它不是完整 Helmholtz 或真实海洋；Gaussian、cosine、不同 taper 不是三个独立物理域。
[方程、坐标与边界](../docs/RANGE_DEPENDENT_WAVEGUIDE_DATA_ZH.md)。

## D10 · Helmholtz：完整场与安全稀疏输入分开

同一根目录下：

- `coherent_helmholtz_development_20261003`：40系统、32train/8DEV，Gaussian介质；部分旧介质族只有很低真实自由度。
- `coherent_helmholtz_multiscale_pilot_20261004`：512份训练介质，8 DEV响应、2数值审计。
  多尺度纹理+inclusions；257-grid求解，Dirichlet+sponge，不是PML。
- `coherent_helmholtz_multiscale_sparse_dev_20261004`：8个
  `dev_00_four_lines.npz` 至 `dev_07_four_lines.npz`，每个3频×65点×4线=780复观测。
  仅公共输入，不含真值/seed；目标为11.3/17Hz，**不是同频 A**。
- `coherent_helmholtz_multiscale_sparse_dev_unsafe_seed_metadata_20261004`：
  **禁止用作模型输入**，历史泄漏样本留作追踪，不删改掩盖。

完整 pilot 的 NPZ spec 含真介质信息，不能直接喂给模型。安全白名单见
[提取代码](../src/spectral_continuation/prepare_helmholtz_sparse_dev.py)、
[数据协议](../docs/HELMHOLTZ_SPARSE_CONTEXT_ZH.md)。
257→513数值差约0.002量级；129-grid高频可达约0.0285，不能忽略数值误差刷高精度。

## D11 · 小型独立配对 pilot

[配置](../configs/ocean_multifidelity_pilot_20260916.json)：3环境，1train/2validation，
101–112 Hz、32×128接收网格；近似8模态与高保真全模态分别求解。
数据规模不足以形成新的学习结论；浅海截断可能不省计算，深海可能丢掉重要模态。
[solver 审计](../reports/OCEAN_MULTIFIDELITY_SOLVER_AUDIT_ZH.md)与
[负结果](../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)是 C 的风险提示，不是推荐直接续跑的 benchmark。

## 检索与核查

```bash
python3 scripts/research_catalog.py list --route B
python3 scripts/research_catalog.py check --local-data
```

只核对路径存在，不加载数组、更不证明 schema/相位/数值精度全部正确。
旧配置/manifest 的来源追踪看[历史文件清单](../docs/archive/INVENTORY_20261005.md)。
