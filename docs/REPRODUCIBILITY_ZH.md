# 第一阶段复现说明

本文只记录如何复核报告数字。研究判断与结果请先读
[第一阶段主报告](../reports/STAGE1_REPORT_ZH.md)。所有命令从仓库根目录运行。

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
