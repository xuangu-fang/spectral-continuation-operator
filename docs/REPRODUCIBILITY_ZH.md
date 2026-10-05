# 技术复现档案

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 本文保存全部阶段的具体命令、路径与数据 hash，因此有意保持详细。当前科学结论请先读
> [一页式状态](CURRENT_STATUS_ZH.md)；最新 wave-system representation 的命令在
> [第 15 节](#15-wave-system-representation-poc)，其余章节均为历史实验复现档案。

所有命令从仓库根目录运行。旧阶段的研究判断与主要负结果已压缩到
[历史探索索引](archive/INDEX_ZH.md)。

## 0. 当前 PBNO M3

```bash
.venv/bin/python -m spectral_continuation.posterior_memory_response_operator \
  --config configs/pbno_posterior_memory_formal_20260925.json \
  --output artifacts/pbno_posterior_kernel_formal_v3 \
  --device cuda

.venv/bin/python -m spectral_continuation.summarize_pbno_m3 \
  --kernel-root artifacts/pbno_posterior_kernel_formal_v3 \
  --latent-root artifacts/pbno_m1_semilinear \
  --output artifacts/pbno_m3_summary

.venv/bin/python -m spectral_continuation.benchmark_posterior_kernel_operator \
  --config configs/pbno_posterior_memory_formal_20260925.json \
  --summary artifacts/pbno_posterior_kernel_formal_v3/summary.json \
  --output artifacts/pbno_m3_summary/compute.json \
  --device cuda
```

正式数字、CNP/belief checkpoint 命令、机器结果和 claim boundary 见
[PBNO M3 报告](../reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md)。

## 1. 环境

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

正式运行使用 Python 3.11、PyTorch 2.8.0+cu128 和一张 NVIDIA A100 80 GB。项目限制
`torch<2.9`，以免当前机器误解析到需要更新驱动的 CUDA 13 wheel。代码不要求
该型号 GPU；CPU 可运行但训练较慢。

## 2. 数据身份

Ocean fine-band 数据：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/
spectral_continuation_ocean_fineband_pilot_20260915/paired_fields.npz
```

SHA256：

```text
603890808c6cbf541583203730f81abbf61f69ca779647441065db36d782323d
```

加载器只保留 Pekeris 实际使用的四个介质通道，并把被排除通道写入每个 run summary。

WaveBench 使用旧项目冻结的 development split：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/runs/
apex_amp_phase_probe/wavebench_full_protocol_split42_20260912
```

held-out ID hash 为
`a6acce080129527137b9179758ac6f26867162571e35bc6eaa44e3fd0726ea5d`；新仓库的 reader
不会读取这些 held-out field bytes。

## 3. 无训练可辨识性探针

以下示例复核 79/80 Hz 对 125 Hz 的结果；其余查询只需替换 `--target`：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.audit_baselines \
  --domain ocean \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_fineband_pilot_20260915/paired_fields.npz \
  --frequencies 74,77,78,79,80,125 \
  --visible 80 --visible 74,77,80 --visible 79,80 --visible 78,79,80 \
  --target 125 \
  --output artifacts/stage1_fineband_analytic/target_125.json
```

绘图：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.plot_analytic_horizon \
  --input-dir artifacts/stage1_fineband_analytic \
  --output artifacts/stage1_fineband_analytic/phase_identifiability_horizon
```

## 4. Query operator

一个正式 M1 run 的完整命令如下：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_query_operator \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_fineband_pilot_20260915/paired_fields.npz \
  --output-dir experiments/runs/fine_query_formal_known_seed42_band_carrier \
  --visible 79,80 \
  --train-queries 90,100 \
  --extrapolation-queries 125,150,175,200 \
  --parameterization carrier_relative \
  --seed 42 --width 16 --epochs 80 --batch-size 16 --eval-batch-size 32
```

2×2 因子为：

- `--visible 80` 或 `--visible 79,80`；
- `--parameterization direct` 或 `carrier_relative`；
- known-medium 或额外加入 `--field-only`；
- seed 42、43、44。

其他预算固定不变。每个 checkpoint 只根据 90/100 Hz validation macro RelL2 选择；训练脚本
在选择冻结后才构造和评估外推 loader。

汇总：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_query \
  --run-root experiments/runs \
  --run-glob 'fine_query_formal_*/summary.json' \
  --output-dir artifacts/stage1_query_formal
```

## 5. 稠密 horizon 一次性评测

生成协议在
[`stage1_ocean_dense_horizon_20260915.json`](../configs/stage1_ocean_dense_horizon_20260915.json)
中，commit `cd66dd5` 早于数据生成和新标签读取。生成命令：

```bash
/home/ubuntu/project/ocean-acoustic-agent/.venv/bin/python \
  /home/ubuntu/project/APEX_repro_fsk/apex_modified/tools/build_ocean_octave_dataset.py \
  --output-dir /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_dense_horizon_20260915 \
  --ocean-repo /home/ubuntu/project/ocean-acoustic-agent \
  --frequencies 74,75,76,77,78,79,80,90,100,105,110,115,120,125,150,175,200 \
  --solver pekeris_mode --n-train 128 --n-val 32 --n-strict 32 --seed 20260912 \
  --range-start-m 50 --range-end-m 750 --range-steps 384 \
  --depth-start-m 1 --depth-end-m 78 --depth-steps 48 --receiver-depth-m 30
```

冻结 checkpoint 的一次性评测：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_horizon \
  --run-root experiments/runs \
  --run-glob 'fine_query_formal_*/summary.json' \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_dense_horizon_20260915/paired_fields.npz \
  --queries 105,110,115,120,125,150,175,200 \
  --output-root experiments/dense_evaluation
```

评测器会先验证环境 ID、split、四个 active medium channels 以及所有共享频率 fields 均
bitwise identical；任何一项不一致都会在打开新结果前报错。

完成后的 dense archive SHA256 为
`412e437d5c5cbf1b286f84af61e38aab3f896de25029b5a97a3f49a9bfa2383b`，manifest SHA256 为
`29a70954da0440abfb4690781f67d2cc4d8c27a4e37963975382f70f06f1100f`。本次 identity guard
通过，2,720 个 simulations 无失败。

## 6. 震荡复数场表征 POC

Ocean stable-state run 的命令模板如下；`SEED` 分别取 42、43、44：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_representation_poc \
  --domain ocean --model stable_recurrence \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_dense_horizon_20260915/paired_fields.npz \
  --output-dir experiments/runs/representation_poc_ocean_recurrence_history_seedSEED \
  --visible 74,75,76,77,78,79,80 --train-queries 90,100 \
  --test-queries 105,110,115,120,125 \
  --order 3 --state-width 64 --blocks 4 --stability-radius 1.0 \
  --history-weight 1.0 --epochs 60 --batch-size 8 --eval-batch-size 8 --seed SEED
```

Matched U-Net 只需将模型和模型尺寸参数替换为：

```text
--model image_unet --width 8
```

合成有限秩 existence probe：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_prony_poc \
  --domain synthetic \
  --output-dir experiments/runs/representation_poc_synthetic_prony_order5_fp64_matched \
  --visible 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --selection-queries 16,18 --test-queries 20,22,24 \
  --order 5 --ridge 1e-5 --batch-size 8 \
  --synthetic-train 512 --synthetic-validation 128
```

最终汇总：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_representation_poc \
  --run-root experiments/runs --output-dir artifacts/representation_poc
```

POC 使用已打开的 development labels 做架构归因，不能被称为新的 held-out test。

## 7. Artifact 边界

Git 提交 compact JSON/CSV、图和 provenance。大型 `best.pt`、逐样本 NPZ 与数据集位于本机
attached storage，并由 summary 中的 SHA256 绑定。旧 APEX 仓库只作为只读证据/生成器来源，
新实验代码与文档均在本仓库维护。

## 8. 连续 pole–residue 主实验

### 8.1 数据身份

整数频率 archive：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/
spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz
SHA256 edb5d0a4a340a60ab9c9ce947105577c0ef3c2d701d7bba632f6c4f3a195ba51
```

它包含 74–125 Hz 每个整数频率，shape 为 `[160,52,48,384,2]`。与上一节 archive 的
环境 ID、split、active medium channels 和 14 个共享频率 tensor 均 bitwise identical。

半整数 query archive：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/
spectral_continuation_ocean_fractional_101p5_124p5_20260915/paired_fields.npz
SHA256 1575e3a1903798dcf3ae9a28c8f535f55d871c8d0e649c5eb6614e8b995bf5eb
```

完整生成参数、manifest hashes、严格集策略和冻结顺序记录在
[`ocean_continuous_poles_stage_20260915.json`](../configs/ocean_continuous_poles_stage_20260915.json)。
模型配置在 commit `06f2cb1` 冻结，先于整数 archive 的首次读取；三 seed 开发证据在 commit
`83e9518` 固化。

### 8.2 主模型

以下命令中 `SEED` 取 42、43、44：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_representation_poc \
  --domain ocean --model carrier_poles --dataset /path/to/integer/paired_fields.npz \
  --output-dir experiments/runs/m3_ocean_dense_poles_seedSEED \
  --visible 74,75,76,77,78,79,80 \
  --train-queries 81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 \
  --test-queries 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --poles 4 --state-width 128 --blocks 4 --grouped-training \
  --pole-growth stable --max-growth-per-hz 0.4 \
  --max-residual-angle-per-hz 0.25 --pole-anchor residue_sum \
  --carrier-estimator endpoint_increment --residue-scale 2 \
  --history-weight 0 --relative-loss-power 1 \
  --epochs 100 --batch-size 4 --eval-batch-size 4 \
  --lr 5e-4 --weight-decay 1e-6 --seed SEED
```

matched recurrence 将模型参数替换为：

```text
--model stable_recurrence --order 3 --state-width 128 --blocks 4
--stability-radius 1 --history-weight 1 --grouped-training
```

matched image baseline 替换为：

```text
--model image_unet --width 16 --batch-size 16
```

两者仍使用 `--relative-loss-power 1`、相同 queries、轮数和随机种子。

### 8.3 冻结后的半整数评测

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_representation_checkpoint \
  --run-summary experiments/runs/m3_ocean_dense_poles_seed42/summary.json \
  --source-dataset /path/to/integer/paired_fields.npz \
  --query-dataset /path/to/fractional/paired_fields.npz \
  --queries 101.5,105.5,110.5,115.5,120.5,124.5 \
  --output-dir experiments/dense_evaluation/m3_ocean_fractional_poles_seed42
```

正文图由 `spectral_continuation.summarize_continuous_state` 和
`spectral_continuation.summarize_fractional_queries` 生成。提交到 Git 的 JSON/CSV 保存三 seed
逐频率指标；大型 checkpoints 和逐样本 vectors 仍只通过 SHA256 引用。

## 9. 连续谱生成器 gate

完整协议见
[`ocean_generator_gate_20260916.json`](../configs/ocean_generator_gate_20260916.json)。主模型命令中
`SEED` 取 42、43、44：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_representation_poc \
  --domain ocean --model carrier_poles --dataset /path/to/integer/paired_fields.npz \
  --output-dir experiments/runs/m4_balanced_train106_hyper_seedSEED \
  --visible 90,91,92,93,94,95,96,97,98,99,100 \
  --train-queries 101,102,103,104,105,106 \
  --test-queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --poles 4 --width 8 --blocks 4 --grouped-training \
  --pole-growth stable --max-growth-per-hz 0.4 \
  --max-residual-angle-per-hz 0.25 --pole-anchor residue_sum \
  --carrier-estimator endpoint_increment --residue-scale 2 \
  --global-latent-channels 64 --hyper-carrier-correction \
  --max-hyper-slope-per-hz 0.08 --max-hyper-curvature-per-hz2 0.005 \
  --history-weight 0 --relative-loss-power 1 \
  --epochs 100 --batch-size 4 --eval-batch-size 4 \
  --lr 5e-4 --weight-decay 1e-6 --seed SEED
```

matched recurrence 使用完全相同的频率、训练轮数和 seed，并替换为：

```text
--model stable_recurrence --order 3 --state-width 128 --blocks 4
--stability-radius 1 --history-weight 1
```

carrier-relative query U-Net：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_query_operator \
  --dataset /path/to/integer/paired_fields.npz \
  --output-dir experiments/runs/m4_balanced_train106_carrier_unet_seedSEED \
  --visible 90,91,92,93,94,95,96,97,98,99,100 \
  --train-queries 101,102,103,104,105,106 \
  --extrapolation-queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --parameterization carrier_relative --max-slots 11 --width 16 \
  --epochs 100 --batch-size 16 --eval-batch-size 16 \
  --lr 5e-4 --weight-decay 1e-6 --seed SEED
```

整数结果、paired statistics 和场级图：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_continuous_state \
  --run 'Hyper pole=experiments/runs/m4_balanced_train106_hyper_seed*/summary.json' \
  --run 'Stable recurrence=experiments/runs/m4_balanced_train106_recurrence_seed*/summary.json' \
  --run 'Carrier U-Net=experiments/runs/m4_balanced_train106_carrier_unet_seed*/summary.json' \
  --output-dir artifacts/spectral_generator_gate
```

分数频率先用 `evaluate_representation_checkpoint` 逐 seed 评测，再运行：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_fractional_evaluations \
  --evaluation 'Hyper pole=experiments/evaluations/m4_balanced_fractional_hyper_seed*/summary.json' \
  --output-dir artifacts/spectral_generator_gate
```

当前阶段仍使用 development environments。任何后续 strict 结果必须在本配置和 checkpoint hashes
冻结后另行生成，且不能回流调参。

## 10. 全局与局部复数子空间诊断

Ocean 主诊断：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.diagnose_global_subspace \
  --domain ocean --dataset /path/to/integer/paired_fields.npz \
  --output-dir artifacts/global_subspace/ocean_90_100 \
  --visible 90,91,92,93,94,95,96,97,98,99,100 \
  --queries 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --ranks 1 2 4 8 11 --polynomial-rank 6 --polynomial-degree 3 \
  --patch-grid 4 8 --patch-query 110 --max-environments 32
```

可控三路径数据由脚本确定性生成，不依赖外部 archive：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.diagnose_global_subspace \
  --domain synthetic --output-dir artifacts/global_subspace/synthetic_1_14 \
  --visible 1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --queries 15,16,17,18,19,20,21,22,23,24 \
  --ranks 1 2 3 4 6 8 12 14 --polynomial-rank 6 --polynomial-degree 3 \
  --patch-grid 4 8 --patch-query 20 --max-environments 32 \
  --synthetic-train 64 --synthetic-validation 32
```

Ocean 的 2×4 与 8×16 patch 扫描只需把 `--queries` 改为 `110`，并分别设置
`--patch-grid 2 4` 与 `--patch-grid 8 16`。汇总图命令：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_global_subspace \
  --ocean artifacts/global_subspace/ocean_90_100 \
  --synthetic artifacts/global_subspace/synthetic_1_14 \
  --grid-2x4 artifacts/global_subspace/ocean_grid_2x4 \
  --grid-8x16 artifacts/global_subspace/ocean_grid_8x16 \
  --output-dir artifacts/global_subspace
```

所有 projection 都按 validation environment 独立计算；`low_band_span`、`patch_span` 和
`full_band_oracle` 使用目标真值求投影系数，只能解释 representation floor，不是预测结果。

## 11. 局部复谱 token POC

自由 local-span 主实验：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_representation_poc \
  --domain ocean --model local_span --dataset /path/to/integer/paired_fields.npz \
  --output-dir experiments/runs/token_poc_gauged_quadratic_8x16_seed42 \
  --visible 90,91,92,93,94,95,96,97,98,99,100 \
  --train-queries 101,102,103,104,105,106 \
  --test-queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --token-grid 8 16 --state-width 64 --blocks 4 \
  --token-weight-bound 2 --token-carrier endpoint \
  --grouped-training --relative-loss-power 1 --epochs 50 \
  --batch-size 4 --eval-batch-size 4 --lr 5e-4 --weight-decay 1e-6 --seed 42
```

frequency-mask 版本增加：

```text
--mask-pretraining frequency --mask-weight 0.1
```

受限 local-carrier-token fallback 从冻结 hyper-pole checkpoint 初始化。除新增 token branch 外，
所有参数均冻结：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_representation_poc \
  --domain ocean --model carrier_poles --dataset /path/to/integer/paired_fields.npz \
  --output-dir experiments/runs/token_poc_hyper_localcarrier_8x16_frozen_seed42 \
  --visible 90,91,92,93,94,95,96,97,98,99,100 \
  --train-queries 101,102,103,104,105,106 \
  --test-queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --poles 4 --state-width 128 --blocks 4 --grouped-training \
  --pole-growth stable --max-growth-per-hz 0.4 \
  --max-residual-angle-per-hz 0.25 --pole-anchor residue_sum \
  --carrier-estimator endpoint_increment --residue-scale 2 \
  --global-latent-channels 64 --hyper-carrier-correction \
  --max-hyper-slope-per-hz 0.08 --max-hyper-curvature-per-hz2 0.005 \
  --local-carrier-token-grid 8 16 --local-carrier-token-width 32 \
  --max-local-token-slope-per-hz 0.08 --max-local-token-curvature-per-hz2 0.005 \
  --initial-checkpoint experiments/runs/m4_balanced_train106_hyper_seed42/best.pt \
  --allow-partial-initial-checkpoint --freeze-initial-backbone \
  --history-weight 0 --relative-loss-power 1 --epochs 50 \
  --batch-size 4 --eval-batch-size 4 --lr 5e-4 --weight-decay 1e-6 --seed 42
```

汇总图与 paired bootstrap：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_token_poc \
  --run 'Hyper-pole=experiments/runs/m4_balanced_train106_hyper_seed42/summary.json' \
  --run 'Free local span=experiments/runs/token_poc_gauged_quadratic_8x16_seed42/summary.json' \
  --run 'Span + freq mask=experiments/runs/token_poc_gauged_quadratic_8x16_frequencymask_seed42/summary.json' \
  --run 'Local carrier tokens=experiments/runs/token_poc_hyper_localcarrier_8x16_frozen_seed42/summary.json' \
  --baseline-label 'Hyper-pole' --candidate-label 'Local carrier tokens' \
  --output-dir artifacts/token_poc
```

## 12. 稀疏谱校准 POC

实验前协议见
[`SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md`](SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)，结果解释见
[`SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md`](../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)。
整数频率 archive 使用第 8.1 节给出的路径和 SHA256。

六频稳定谱状态 calibrator 的命令如下，`SEED` 取 42、43、44，并与同 seed 的冻结 base checkpoint
配对：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_ar_calibrator \
  --base-run-summary experiments/runs/m4_balanced_train106_hyper_seedSEED/summary.json \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz \
  --output-dir experiments/runs/sparse_ar1_calibrator_seedSEED \
  --calibration-frequencies 101,102,103,104,105,106 \
  --fit-environments 112 --sensor-budget 92 --sensor-token-grid 2 4 \
  --width 32 --rollout-offsets 0,1,2,3,4,5,6 \
  --epochs 120 --batch-size 4 --learning-rate 3e-4 \
  --weight-decay 1e-6 --seed SEED
```

训练只读取 90--100 Hz visible fields 和 101--106 Hz calibration fields。offset 1--6 的 target 是由
calibration fields 拟合出的稳定 AR 状态的解析 rollout，不读取真实 107--112 Hz labels。三次训练
选中的 epoch 分别为 109、96、106，checkpoint SHA256 分别为：

```text
seed42 695a351e1c8529f344368846dc8033f93c9017e560dbdb1aad45e50995104ef1
seed43 b8f9d52954123ced8a2fd339f9dee5879a4cf62078561ab24dd1a485409872e2
seed44 2be4530e3ee2d9f2c16d7c25e49493132cce2ab8e2437dbe5273533ce7fecf3d
```

冻结评测命令中 `SEED` 与 checkpoint 对应；三个 run 使用相同的三个 mask seeds：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_ar_calibration \
  --calibrator-summary experiments/runs/sparse_ar1_calibrator_seedSEED/summary.json \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz \
  --output-dir experiments/sparse_calibration/ar1_seedSEED_validation \
  --split val \
  --queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --sensor-budgets 23,46,92,184,369 \
  --mask-seeds 4201,4301,4401
```

最终聚合和作图：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_sparse_calibration \
  --run-dirs \
    experiments/sparse_calibration/ar1_seed42_validation \
    experiments/sparse_calibration/ar1_seed43_validation \
    experiments/sparse_calibration/ar1_seed44_validation \
  --output-dir artifacts/sparse_spectral_calibration
```

聚合 artifact 保留全部逐方法、预算、mask seed 和查询频率的均值记录；逐环境 vectors 和大型
checkpoints 不提交 Git，但由上述 checkpoint hash 与每个 evaluation summary 绑定。

## 13. Reliable-horizon Map、空间风险与 Sequential Calibration

实验前协议见 [`RELIABLE_HORIZON_MAP_POC_ZH.md`](RELIABLE_HORIZON_MAP_POC_ZH.md)，最终解释和
gate 审计见
[`RELIABLE_HORIZON_MAP_FINAL_ZH.md`](../reports/RELIABLE_HORIZON_MAP_FINAL_ZH.md)。大型逐实例 NPZ
由 `.gitignore` 排除，但以下命令可从冻结 checkpoint 重建；JSON、PNG 和 PDF 汇总已提交。

### 13.1 Ocean rolling reset replay

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_sequential_calibration \
  --calibrator-summary experiments/runs/sparse_ar1_calibrator_seed42/summary.json \
  --calibrator-summary experiments/runs/sparse_ar1_calibrator_seed43/summary.json \
  --calibrator-summary experiments/runs/sparse_ar1_calibrator_seed44/summary.json \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz \
  --output-dir artifacts/reliable_horizon_map/sequential_replay_ocean_m2b_20260917 \
  --split val \
  --queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --endpoints 106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --window 6 --sensor-budget 92 --mask-seed 4201 --device cuda:0
```

该命令除逐 endpoint error 外，还保存 calibrated seed 的 complex、amplitude 和 phase disagreement。
这些 feature 只依赖冻结预测与已经购买的 sparse sweep，不读取未来 query target。

### 13.2 Reset 后 risk 与 repeated frontier

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_calibrated_state_risk \
  --sequential-errors artifacts/reliable_horizon_map/sequential_replay_ocean_m2b_20260917/sequential_errors.npz \
  --output-dir artifacts/reliable_horizon_map/calibrated_state_risk_ocean_20260917 \
  --error-threshold 0.5 --safe-probability 0.1 --folds 4 --ridge 0.1

PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_repeated_calibration \
  --sequential-errors artifacts/reliable_horizon_map/sequential_replay_ocean_m2b_20260917/sequential_errors.npz \
  --base-risk artifacts/reliable_horizon_map/risk_ocean_hyper_20260917/risk_predictions.npz \
  --calibrated-risk artifacts/reliable_horizon_map/calibrated_state_risk_ocean_20260917/calibrated_state_risk.npz \
  --output-dir artifacts/reliable_horizon_map/repeated_calibration_ocean_20260917 \
  --maximum-actions 3 --sensor-budget 92 --window 6 --random-replicates 2000 \
  --failure-probability 0.1 --calibrated-failure-probability 0.1 \
  --reference-frequency 106 --error-threshold 0.5 \
  --calibrated-risk-method disagreement_only
```

固定策略穷举至多三次 action 的所有 endpoint 组合。报告同时给出 gross cost 和允许复用重叠
sensor-frequency 对后的 unique cost；所有 query 仍按完整空间场评价。

### 13.3 Synthetic 无训练 sparse ensemble calibration

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_ensemble_mixture_calibration \
  --run-summary experiments/runs/m2_synthetic_endpoint_poles_seed42/summary.json \
  --run-summary experiments/runs/m2_synthetic_endpoint_poles_seed43/summary.json \
  --run-summary experiments/runs/m2_synthetic_endpoint_poles_seed44/summary.json \
  --output-dir artifacts/reliable_horizon_map/sequential_replay_synthetic_mixture_ridge1.0_20260917 \
  --queries 19,19.5,20,21,22,23,24 --endpoints 18,19,20,21,22,23,24 \
  --window 4 --sensor-budget 10 --mask-seed 4201 --ridge 1.0 \
  --split val --device cuda:0
```

后续 `evaluate_calibrated_state_risk` 和 `evaluate_repeated_calibration` 命令与 Ocean 相同，只需替换
artifact 路径，并使用 `--sensor-budget 10 --window 4 --reference-frequency 18`。该 action 的权重满足
sum-to-one 约束，只用 rolling calibration window 拟合，不读取未来标签。

### 13.4 两域空间风险图

Ocean 主网格：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_spatial_horizon_risk \
  --run-summary experiments/runs/m4_balanced_train106_hyper_seed42/summary.json \
  --run-summary experiments/runs/m4_balanced_train106_hyper_seed43/summary.json \
  --run-summary experiments/runs/m4_balanced_train106_hyper_seed44/summary.json \
  --output-dir artifacts/reliable_horizon_map/spatial_risk_ocean_20260917 \
  --queries 107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125 \
  --reference-frequency 106 --patch-grid 4 8 --error-threshold 0.5 \
  --safe-probability 0.1 --energy-floor-fraction 0.1 --folds 4 --ridge 0.1 \
  --split val --batch-size 4 --device cuda:0
```

Synthetic 主网格：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_spatial_horizon_risk \
  --run-summary experiments/runs/m2_synthetic_endpoint_poles_seed42/summary.json \
  --run-summary experiments/runs/m2_synthetic_endpoint_poles_seed43/summary.json \
  --run-summary experiments/runs/m2_synthetic_endpoint_poles_seed44/summary.json \
  --output-dir artifacts/reliable_horizon_map/spatial_risk_synthetic_20260917 \
  --queries 19,19.5,20,21,22,23,24 --reference-frequency 18 \
  --patch-grid 4 8 --error-threshold 0.5 --safe-probability 0.1 \
  --energy-floor-fraction 0.1 --folds 4 --ridge 0.1 \
  --split val --batch-size 8 --device cuda:0
```

尺度敏感性只需将 `--patch-grid 4 8` 分别替换为 `2 4` 和 `8 16`。

### 13.5 最终总图

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_reliable_horizon_final \
  --ocean-risk artifacts/reliable_horizon_map/risk_ocean_hyper_20260917/risk_summary.json \
  --synthetic-risk artifacts/reliable_horizon_map/risk_synthetic_poles_20260917/risk_summary.json \
  --ocean-calibrated-risk artifacts/reliable_horizon_map/calibrated_state_risk_ocean_20260917/calibrated_state_risk_summary.json \
  --synthetic-calibrated-risk artifacts/reliable_horizon_map/calibrated_state_risk_synthetic_20260917/calibrated_state_risk_summary.json \
  --ocean-repeated artifacts/reliable_horizon_map/repeated_calibration_ocean_20260917/repeated_calibration_summary.json \
  --synthetic-repeated artifacts/reliable_horizon_map/repeated_calibration_synthetic_20260917/repeated_calibration_summary.json \
  --ocean-spatial artifacts/reliable_horizon_map/spatial_risk_ocean_2x4_20260917/spatial_risk_summary.json \
  --ocean-spatial artifacts/reliable_horizon_map/spatial_risk_ocean_20260917/spatial_risk_summary.json \
  --ocean-spatial artifacts/reliable_horizon_map/spatial_risk_ocean_8x16_20260917/spatial_risk_summary.json \
  --synthetic-spatial artifacts/reliable_horizon_map/spatial_risk_synthetic_2x4_20260917/spatial_risk_summary.json \
  --synthetic-spatial artifacts/reliable_horizon_map/spatial_risk_synthetic_20260917/spatial_risk_summary.json \
  --synthetic-spatial artifacts/reliable_horizon_map/spatial_risk_synthetic_8x16_20260917/spatial_risk_summary.json \
  --output-dir artifacts/reliable_horizon_map/final_summary_20260917
```

## 14. Actionable Horizon / Value-of-Calibration Map

冻结协议见 [`ACTIONABLE_HORIZON_POC_ZH.md`](ACTIONABLE_HORIZON_POC_ZH.md)，结果解释见
[`ACTIONABLE_HORIZON_POC_ZH.md`](../reports/ACTIONABLE_HORIZON_POC_ZH.md)。已有 mask 4201 replay 直接复用
13.1 的产物。新增两个等成本 mask replay 时，保持其他参数完全相同，只分别替换：

```bash
--output-dir artifacts/actionable_horizon/replay_ocean_mask4301_20260922 --mask-seed 4301
--output-dir artifacts/actionable_horizon/replay_ocean_mask4401_20260922 --mask-seed 4401
```

三个 replay 完成后运行环境级 OOF value probe：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.evaluate_actionable_horizon \
  --sequential-errors artifacts/reliable_horizon_map/sequential_replay_ocean_m2b_20260917/sequential_errors.npz \
  --sequential-errors artifacts/actionable_horizon/replay_ocean_mask4301_20260922/sequential_errors.npz \
  --sequential-errors artifacts/actionable_horizon/replay_ocean_mask4401_20260922/sequential_errors.npz \
  --mask-ids 4201 4301 4401 \
  --features artifacts/reliable_horizon_map/features_ocean_hyper_20260917/horizon_features.npz \
  --risk-predictions artifacts/reliable_horizon_map/risk_ocean_hyper_20260917/risk_predictions.npz \
  --spatial-risk-predictions artifacts/reliable_horizon_map/spatial_risk_ocean_20260917/spatial_risk_predictions.npz \
  --output-dir artifacts/actionable_horizon/value_probe_ocean_20260922 \
  --risk-method disagreement_only --spatial-risk-method local_combo \
  --primary-model state_action --ridge-grid 0.01 0.1 1 10 \
  --failure-probability 0.1 --primary-mask-id 4201 \
  --reference-frequency 106 --error-threshold 0.5 --shuffled-replicates 20000 \
  --local-window 6 --sensor-budget 92 --window 6 --field-shape 48 384 --seed 20260922
```

`actionable_horizon_predictions.npz` 保存逐环境逐 action 的 utility、特征和 OOF prediction，由
`.gitignore` 排除；JSON、PNG 和 PDF 汇总提交 Git。mask 4301/4401 的 replay NPZ 也不提交，但其
summary 保留数据、checkpoint hash、预算和 mask seed。

## 15. Wave-system representation POC

数据路径与 SHA-256 见 [`data/README.md`](../data/README.md)。以下命令均在仓库根目录执行。

生成主数据与 ellipse OOD：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.wave_operator_data \
  --config configs/wave_operator_representation_poc_20260922.json \
  --output /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz

PYTHONPATH=src .venv/bin/python -m spectral_continuation.wave_operator_data \
  --config configs/wave_operator_ood_ellipses_20260922.json \
  --output /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_ood_ellipses_24.npz
```

M0 与 matched field baseline：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.audit_wave_operator_representation \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/wave_operator_representation/m0

PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_wave_operator_identifier \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/wave_operator_representation/m1 \
  --representation raw --steps 3000 --seed 20260922
```

稳定 cross-view representation 与 hidden-medium probe：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_wave_operator_contrastive \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/wave_operator_representation/m2_ablation_seed20260922 \
  --representation amplitude_phase_gradient --steps 2500 \
  --batch-size 24 --views-per-set 6 --learning-rate 0.001 --seed 20260922

PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_wave_medium_probe \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/wave_operator_representation/m3_ablation_seed20260922 \
  --representation amplitude_phase_gradient --steps 3000 --seed 20260922
```

噪声训练版本将最后一条命令替换为
`--representation amplitude_soft_phase_gradient --train-snr-db 20 --noise-seed 9173`。
正式结果使用 seed 20260922、20260923、20260924。最终汇总：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_wave_representation_poc
```

checkpoint 与 prediction NPZ 受 `.gitignore` 排除；JSON、CSV、PNG、PDF 和配置进入版本管理。

## 16. Variable-budget Wave Set Operator M0

正式协议见 `configs/wave_set_operator_m0_formal_20260922.json`。先生成只用于携带 representation、scale 和
architecture schema 的 initializer；正式 probe 使用 `--reset-encoder`，不会读取其训练权重：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_wave_set_encoder \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output experiments/wave_set_m0_formal/initializers/phase_slots \
  --representation amplitude_soft_phase_gradient --steps 1 --evaluate-every 1 \
  --batch-size 24 --aggregation slots --slots 4 --width 24 --dimension 128 --seed 20260922

PYTHONPATH=src .venv/bin/python -m spectral_continuation.probe_wave_set_encoder \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --encoder-checkpoint experiments/wave_set_m0_formal/initializers/phase_slots/set_encoder_amplitude_soft_phase_gradient_seed20260922.pt \
  --output experiments/wave_set_m0_formal/amplitude_soft_phase_gradient_seed20260922 \
  --train-systems 112 --finetune --reset-encoder --steps 2500 \
  --batch-size 16 --full-context-probability 0.5 --training-view-domain low --seed 20260922
```

将 representation 替换为 `raw` 和 `amplitude`，并运行 seeds 20260922、20260923、20260924。正式汇总：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_wave_set_m0 \
  --root experiments/wave_set_m0_formal --output artifacts/wave_set_m0/summary
```

shift evaluation 使用 `spectral_continuation.evaluate_wave_set_probe`，分别输入 ellipse、IID noise 与 legacy
128-grid 数据；三 seed JSON 完成后运行：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_wave_set_robustness \
  --root experiments/wave_set_m0_robustness/formal --output artifacts/wave_set_m0/summary
```

## 17. WaveBench representation M1

冻结协议见 `configs/wavebench_representation_m1_20260923.json`。公开数据原始路径为：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/
wavebench_full_remotezip_20260912
```

其中包含 `isotropic_10.beton`、`isotropic_15.beton`、`isotropic_20.beton`、`isotropic_40.beton`。reader
直接解析固定 WaveBench FFCV v2 layout，不要求安装 FFCV。以下为单个正式 run；依次替换 representation 和
seed。三种正式 representation 为 `raw`、`amplitude`、`amplitude_soft_phase_gradient`，seeds 为
20260923、20260924、20260925。

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_wavebench_identifier \
  --data-root /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/wavebench_full_remotezip_20260912 \
  --output artifacts/wavebench_representation_m1/formal_seed20260925 \
  --representation amplitude_soft_phase_gradient \
  --train-systems 8000 --validation-systems 500 --test-systems 500 \
  --steps 3000 --evaluate-every 300 --batch-size 12 --eval-batch-size 32 \
  --width 24 --dimension 128 --slots 4 --learning-rate 0.001 \
  --full-context-probability 0.5 --split-seed 42 --seed 20260925
```

三 seed 汇总与 paired bootstrap：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_wavebench_representation \
  --roots artifacts/wavebench_representation_m1/development_seed20260923 \
          artifacts/wavebench_representation_m1/formal_seed20260924 \
          artifacts/wavebench_representation_m1/formal_seed20260925 \
  --output artifacts/wavebench_representation_m1/summary
```

固定测试样本可视化：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.plot_wavebench_examples \
  --data-root /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/wavebench_full_remotezip_20260912 \
  --raw-checkpoint artifacts/wavebench_representation_m1/development_seed20260923/raw_seed20260923.pt \
  --phase-checkpoint artifacts/wavebench_representation_m1/development_seed20260923/amplitude_soft_phase_gradient_seed20260923.pt \
  --output artifacts/wavebench_representation_m1/summary/wavebench_examples.png
```

`*.pt` checkpoint 与 `*_records.csv` 逐系统文件不提交 Git；三 seed JSON、聚合 CSV/JSON 和 PNG/PDF
保留。开发型 `amplitude_edge_connection` 与 `amplitude_transport_connection` 只用于冻结负结果，不计入
正式三 seed 比较。

## 18. 多分支相空间 MPS-0/1

冻结配置为：

```text
configs/multibranch_phase_space_mps0_20260923.json
configs/multibranch_phase_space_mps0b_20260923.json
configs/multibranch_phase_space_mps1_20260923.json
```

受控混合、固定窗口和真实 patch oracle：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.audit_multibranch_phase_space \
  --config configs/multibranch_phase_space_mps0_20260923.json --device cuda

PYTHONPATH=src .venv/bin/python -m spectral_continuation.audit_wavelength_scaled_phase_space \
  --config configs/multibranch_phase_space_mps0b_20260923.json --device cuda
```

MPS-1 synthetic control 从同一个既有五通道 checkpoint 继续 800 steps；candidate 复制原五通道第一层权重，
新增的 64 个输入通道权重置零。以下给出 seed 20260922 的 matched pair，另外两个 seed 只替换 `--seed`
和 output 路径：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.probe_wave_set_encoder \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --encoder-checkpoint experiments/wave_set_m0_formal/amplitude_soft_phase_gradient_seed20260922/set_probe_amplitude_soft_phase_gradient_enc20260922_labels112_seed20260922_finetune.pt \
  --output artifacts/multibranch_phase_space_mps1/synthetic_continuation_control_seed20260922 \
  --train-systems 112 --finetune --residual-initialize \
  --full-context-probability 0.5 --training-view-domain low \
  --steps 800 --batch-size 16 --learning-rate 0.0002 --seed 20260922

PYTHONPATH=src .venv/bin/python -m spectral_continuation.probe_wave_set_encoder \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --encoder-checkpoint experiments/wave_set_m0_formal/amplitude_soft_phase_gradient_seed20260922/set_probe_amplitude_soft_phase_gradient_enc20260922_labels112_seed20260922_finetune.pt \
  --output artifacts/multibranch_phase_space_mps1/synthetic_residual_multiscale_gram_seed20260922 \
  --representation-override amplitude_phase_multiscale_packet_gram \
  --train-systems 112 --finetune --residual-initialize \
  --full-context-probability 0.5 --training-view-domain low \
  --steps 800 --batch-size 16 --learning-rate 0.0002 --seed 20260922
```

对 128-grid boundary 数据分别运行 `evaluate_wave_set_probe`，checkpoint 使用上述 control/candidate 的
`.pt` 文件，输出放入 `artifacts/multibranch_phase_space_mps1/boundary_eval/`。数据为：

```text
/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/
operator_id_legacy_geometry_128_30.npz
```

WaveBench mechanism ablation 复用完整 3,000-step phase checkpoint，并 matched 追加 800 steps：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_wavebench_identifier \
  --data-root /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/wavebench_full_remotezip_20260912 \
  --output artifacts/multibranch_phase_space_mps1/residual_wave_packet_gram_seed20260923 \
  --representation amplitude_phase_wave_packet_gram \
  --initial-checkpoint artifacts/wavebench_representation_m1/development_seed20260923/amplitude_soft_phase_gradient_seed20260923.pt \
  --train-systems 8000 --validation-systems 500 --test-systems 500 \
  --steps 800 --evaluate-every 100 --batch-size 12 --eval-batch-size 32 \
  --width 24 --dimension 128 --slots 4 --learning-rate 0.0002 \
  --full-context-probability 0.5 --split-seed 42 --seed 20260923
```

最终三 seed 配对统计与正文图：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.summarize_multibranch_phase_space
```

冻结输出在 `artifacts/multibranch_phase_space_mps1/summary/`。汇总脚本先在每个 physical system 上平均
三个 seed，再对 system 做 paired bootstrap；不得把 90 个 seed-system pairs 当作独立样本。checkpoint、
prediction NPZ 与逐系统中间 CSV 不进 Git，最终 summary JSON/CSV 和 PNG/PDF 进入版本管理。

## In-Context Response Query RQ-1（2026-09-24）

冻结配置为 `configs/in_context_response_query_rq1_formal_20260924.json`。先运行非神经审计：

```bash
uv run python -m spectral_continuation.audit_response_query \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/in_context_response_query_rq0/audit_source_array_structured \
  --receivers 9 --receiver-mode source_array --episodes-per-system 8 --seed 20260924
```

正式模型每个 seed 先训练标准化 residual，再用最终 physical metric 微调；`SEED` 为 20260924、20260925、
20260926：

```bash
uv run python -m spectral_continuation.train_response_query \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/in_context_response_query_rq0/formal_seedSEED_stage1 \
  --receivers 9 --receiver-mode source_array \
  --reciprocity-residual --calibrated-reciprocity \
  --model transformer --hidden 64 --transformer-layers 2 --transformer-heads 4 \
  --steps 4000 --batch-size 256 --evaluate-every 500 --regimes q2 --seed SEED

uv run python -m spectral_continuation.train_response_query \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/in_context_response_query_rq0/formal_seedSEED_stage2 \
  --receivers 9 --receiver-mode source_array \
  --reciprocity-residual --calibrated-reciprocity \
  --model transformer --hidden 64 --transformer-layers 2 --transformer-heads 4 \
  --steps 2000 --batch-size 256 --evaluate-every 250 --regimes q2 --seed SEED \
  --learning-rate 0.0001 --relative-loss-weight 0.5 \
  --initial-checkpoint artifacts/in_context_response_query_rq0/formal_seedSEED_stage1/best.pt
```

test 只在结构和 loss 冻结后运行：

```bash
uv run python -m spectral_continuation.evaluate_response_query_checkpoint \
  --checkpoint artifacts/in_context_response_query_rq0/formal_seedSEED_stage2/best.pt \
  --dataset /mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz \
  --output artifacts/in_context_response_query_rq0/formal_test_seedSEED \
  --split test --episodes-per-system 8
```

三 seed 汇总使用 `spectral_continuation.summarize_response_query`。20260924 的 stage-2 目录保留了开发期名称
`calibrated_reciprocity_transformer_small_q2_relft_seed20260924`；对应 checkpoint hash 已在 RQ-1 报告记录。

## Cross-Domain Response Query（2026-09-24）

冻结配置：`configs/cross_domain_response_query_20260924.json`。四个 adapter 命令与数据限制见
[`CROSS_DOMAIN_DATA_PROVENANCE_ZH.md`](CROSS_DOMAIN_DATA_PROVENANCE_ZH.md)。正式 M0 审计：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.cross_domain_query_audit \
  --dataset artifacts/cross_domain_response_query/data/openfwi_curvefault_a_v1.npz \
  --output artifacts/cross_domain_response_query/openfwi_m0 \
  --domain openfwi_curvefault_a --budgets 1,2,4 --repeats 3

PYTHONPATH=src .venv/bin/python -m spectral_continuation.cross_domain_query_audit \
  --dataset artifacts/cross_domain_response_query/data/deepwave_ring_v1.npz \
  --output artifacts/cross_domain_response_query/deepwave_m0 \
  --domain deepwave_ring --budgets 1,2,4,8,15 --repeats 3

PYTHONPATH=src .venv/bin/python -m spectral_continuation.cross_domain_query_audit \
  --dataset artifacts/cross_domain_response_query/data/ocean_multistatic_v1.npz \
  --output artifacts/cross_domain_response_query/ocean_m0 \
  --domain ocean_multistatic --budgets 1,2,4,8 --repeats 3
```

M1 每域运行三个 seed；以下为 20260924，其余只改 `--seed` 和 output：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_cross_domain_query \
  --dataset artifacts/cross_domain_response_query/data/openfwi_curvefault_a_v1.npz \
  --output artifacts/cross_domain_response_query/openfwi_m1_seed20260924 \
  --rank-by-budget 1:4,2:4,4:16 --selection-budget 2 \
  --receiver-topology line --steps 2500 --batch-size 64 --seed 20260924

PYTHONPATH=src .venv/bin/python -m spectral_continuation.train_cross_domain_query \
  --dataset artifacts/cross_domain_response_query/data/deepwave_ring_v1.npz \
  --output artifacts/cross_domain_response_query/deepwave_m1_seed20260924 \
  --rank-by-budget 1:4,2:16,4:32,8:32,15:32 --selection-budget 8 \
  --receiver-topology periodic --steps 2500 --batch-size 64 --seed 20260924
```

正文图由以下命令从冻结 summary 重建：

```bash
PYTHONPATH=src .venv/bin/python -m spectral_continuation.plot_cross_domain_query \
  --artifact-root artifacts/cross_domain_response_query
```

NPZ、checkpoint 与逐 system CSV 不进 Git；archive manifest、每域 summary JSON、三 seed 汇总与正文 PNG 进入
版本管理。
