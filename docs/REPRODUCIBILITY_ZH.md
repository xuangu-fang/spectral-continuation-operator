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

正式运行使用 Python 3.11、PyTorch 2.11.0+cu128 和一张 NVIDIA A100 80 GB。代码不要求
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
