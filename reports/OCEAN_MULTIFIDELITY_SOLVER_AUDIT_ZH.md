# Ocean 独立多保真 solver 审计与 tiny formal pilot

日期：2026-09-16

## 结论先行

本轮得到的是一个**分层结论**，而不是笼统的 GO：

1. **solver / 数据工程：有条件 GO。** `PekerisModeSolver(nmodes=8)` 会独立求解截断后的简正模态场，不读取、不降采样、也不插值 HF 场。在全模态数显著大于 8 的环境中，它提供了可测量的计算节省，并且正式 LF/HF 双归档通过了配对身份与来源校验。
2. **当前任务与学习信号：NO-GO。** 在两个未参与训练的 validation 环境上，用 101--106 Hz 的 dense HF calibration 和 107--112 Hz 的 cheap query 做无训练 AR(1) 校正，复数 RelL2 只从 `0.7057` 降到 `0.5950`，宏观改善为 `15.69%`；改善随外推距离迅速衰减，112 Hz 甚至劣于直接使用 cheap field。因此没有通过当前 POC gate，**不建议按固定 `nmodes=8` 直接扩大数据规模或启动重模型训练**。

这个结果仍然有价值：它排除了“LF 是 HF 下采样”的数据泄漏，证明了 ocean 数据可构造真正独立的多保真观测；同时也说明，只有便宜 solver 并不足以自动产生强而稳定的跨频 continuation 信号。

## 1. 审计问题与判据

本轮只回答三个问题：

- `ocean-acoustic-agent` 是否存在相对于 dense HF **真正独立**的低成本求解器？
- 在同一环境、同一频率、同一接收网格上，它的时间、内存分配与复场误差是否形成合理 trade-off？
- 这种独立 LF 在严格隔离 query HF target 后，是否足以支持一个有明显收益的 frequency-continuation POC？

微基准预注册门槛为四个环境的中位数同时满足：

| 指标 | 门槛 |
|---|---:|
| wall-time speedup | >= 1.5x |
| Julia allocation reduction | >= 1.2x |
| complex RelL2 | <= 0.8 |
| phase coherence | >= 0.65 |

注意：这里的 memory 是 Julia `@timed` 返回的累计 heap allocation volume，不是瞬时 peak memory；进程级 peak RSS 仅作为环境记录，不能归因给单次 solve。

## 2. Solver 源码审计

### 2.1 可用的独立 LF

`UnderwaterAcoustics.jl 0.8.1` 中的 `PekerisModeSolver(env; ngrid=0, nmodes=0)` 提供两个原生近似旋钮：

- `nmodes`：限制实际计算并叠加的最大模态数；`0` 表示不限制。它是本轮采用的 reduced-order LF，而不是 HF 后处理。
- `ngrid`：改变 fluid-bottom 模态根搜索网格。较小网格可能漏模态，但误差不够稳定。

源码中先根据 `nmodes` 截断待求根/待叠加的模态集合，再构造声场。因此 `nmodes=8` 与 `nmodes=0` 是两次独立传播求解，满足本轮的来源约束。

`PekerisRayTracer(max_bounces)` 也是独立的物理简化，但在当前小网格上比 mode solver 更慢且误差更大。本轮也检查了 ocean agent 的 AcousticsToolbox 路径：Bellhop 暴露 `nbeams`，Kraken 本身有模态/网格精度空间，但现有 adapter 没有形成同等清晰、可冻结的 paired-fidelity 接口，因此没有在本轮最小 POC 中采用。

### 2.2 适用边界

Pekeris 模型本身是等声速、距离无关水体与流体半空间海底的理想化问题；本轮设置也没有 bottom absorption。它不能单独支撑“对复杂海洋环境普遍成立”的论文表述。它的合适定位是：

- 一个可审计、可复现的独立多保真 pilot；
- 检验跨频校正机制是否存在强信号的最小环境；
- 后续转向 Kraken/Bellhop 或通用 Helmholtz 离散保真度之前的筛选器。

## 3. HF lineage 与复现

审计脚本用原始 dense ocean shard 的环境和 110 Hz 场重新运行 `PekerisModeSolver(nmodes=0, ngrid=0)`：

| 检查项 | 结果 |
|---|---:|
| archived `complex64` bitwise equality | true |
| maximum float32 absolute difference | 0 |
| fresh double vs archived float32 complex RelL2 | `2.55e-8` |
| phase coherence | `~1.0` |

因此正式 pilot 的 HF solver 与既有 ocean dense 数据 lineage 对齐，不是另一个未经核验的“高保真”定义。

完整审计结果：`artifacts/ocean_multifidelity_solver_audit/summary.json`。

## 4. 同实例、同频率微基准

测试频率为 110 Hz，接收网格为 `32 x 128`；每个 solver 先预热 JIT，再在四个环境上重复 5 次。时间取 fresh model construction + complex-field solve 的中位数。

| LF candidate | speedup | allocation reduction | complex RelL2 | phase coherence | gate |
|---|---:|---:|---:|---:|---|
| mode n=4 | 3.40x | 2.22x | 0.856 | 0.509 | FAIL：质量 |
| **mode n=8** | **1.78x** | **1.54x** | **0.665** | **0.738** | median PASS |
| mode n=12 | 1.25x | 1.21x | 0.276 | 0.916 | FAIL：速度 |
| coarse root grid 11 | 2.72x | 2.02x | 0.723 | 0.642 | FAIL：phase / 不稳定 |
| ray, 1 bounce | 0.74x | 0.38x | 0.872 | 0.383 | FAIL |
| ray, 3 bounces | 0.31x | 0.17x | 1.013 | 0.253 | FAIL |

这里必须保留一个重要反例：90 m 环境在 110 Hz 只有 7 个全模态，因此 `n=8` 实际仍计算全部 7 个模态。LF/HF 复场完全相同，speedup 只有 `1.0007x`，allocation reduction 为 `1.0x`。固定 `n=8` 的中位数通过不能被解读为对所有环境都便宜。

在 260 m 环境中，全模态数为 26，`n=8` 才形成明确的 reduced-order solve：speedup `3.19x`、allocation reduction `2.38x`，但 RelL2 增至 `0.832`、phase coherence 降至 `0.530`。这显示成本与保真度的取舍会随有效模态数明显变化。

## 5. 正式 tiny paired-fidelity pilot

生成器先完成全部 LF pass，之后才启动 HF pass；二者在相同 native receiver grid 上分别求解。LF pass 不接收任何 HF field 或 archive。

| 项目 | 设置 |
|---|---|
| 频率 | 101, 102, ..., 112 Hz |
| 环境 | 3 个：100 m train；160 m、220 m validation |
| 场张量 | `[3, 12, 32, 128, 2]`，末维为 real/imag |
| LF | `PekerisModeSolver(nmodes=8, ngrid=0)` |
| HF | `PekerisModeSolver(nmodes=0, ngrid=0)` |
| schema | `spectral-continuation-paired-fidelity-v1` |
| paired identity validation | PASS |

归档：

- `artifacts/ocean_multifidelity_formal_pilot/low.npz`
- `artifacts/ocean_multifidelity_formal_pilot/high.npz`
- `artifacts/ocean_multifidelity_formal_pilot/manifest.json`

两个 archive 的 environment/source/split/frequency/grid/parameter identity 完全一致，但文件路径、内容哈希与 fidelity provenance 不同。validation 两个环境、107--112 Hz 上，单次记录的中位 speedup 为 `2.12x`，allocation reduction 为 `1.77x`；cheap-vs-HF RelL2 为 `0.706`，phase coherence 为 `0.676`。

100 m train 环境是退化反例：101--112 Hz 的全模态数均为 8，所以 `n=8` 与 HF 完全相同且没有成本收益。该环境没有进入后面的 validation oracle；它仍说明未来数据域不能只按水深/频率采样，而必须显式约束 HF mode count 与 LF truncation ratio。

旧的 `artifacts/ocean_multifidelity_pilot/paired_multifidelity_fields.npz` 是早期单频、非正式 schema 的试产物，已被上述双 archive 取代，**不得用于后续训练或论文结果**。

## 6. Training-free oracle：当前任务信号没有过关

为了在启动模型训练前判断 paired fidelity 是否值得扩大，本轮在两个 validation 环境上冻结如下访问协议：

- 可见：101--106 Hz 的 HF calibration fields；
- 可见：107--112 Hz 的 LF query fields；
- 不可见：预测构造时的 107--112 Hz HF query targets；
- query HF 只在预测完成后用于评价。

最强的简洁基线是 dense calibration complex AR(1)：用 calibration band 拟合每个空间点的复数一步频率演化，再从 106 Hz 向前递推。结果为：

| 方法 | macro complex RelL2 | amplitude RelL2 | phase coherence |
|---|---:|---:|---:|
| cheap query 直接作为预测 | 0.7057 | 0.5103 | 0.6761 |
| dense-calibration AR(1) | **0.5950** | **0.3824** | **0.7088** |

宏观 RelL2 改善为 `15.69%`，低于当前要求的显著收益。逐频改善进一步显示明显 horizon decay：

| query | 相对 cheap-query 的 RelL2 改善 |
|---:|---:|
| 107 Hz | 45.88% |
| 108 Hz | 27.36% |
| 109 Hz | 16.06% |
| 110 Hz | 7.97% |
| 111 Hz | 1.28% |
| 112 Hz | -4.43% |

预注册的局部门槛要求 107--110 Hz 每个 query 都至少改善 30%，实际只有 107 Hz 达标。更复杂的 damped velocity linear state 也没有改善宏观结果。因此这不是“需要堆更多 loss 就能修好”的证据，而是当前固定模态截断 + 长 horizon formulation 的信号本身偏弱。

完整 oracle：`artifacts/ocean_paired_fidelity_oracle_20260916/summary.json`。

## 7. 下一步建议与停止线

### 7.1 现在不做

- 不把固定 `nmodes=8` pilot 扩展成大数据集。
- 不因为 107 Hz 的近程结果好，就只汇报单步 query 或更换指标掩盖 horizon decay。
- 不在当前信号强度下启动大 backbone 或叠加多项 loss。

### 7.2 若继续 ocean paired-fidelity，最小下一步

优先审计**按相对模态比例定义的 LF**，例如

```text
nmodes_LF = ceil(alpha * estimated_full_mode_count)
```

并预先限定 `full_mode_count > nmodes_LF`。这样可以避免浅水域出现 LF=HF，也能让 fidelity gap 在不同环境间更一致。它是新的 solver 定义，必须重新做本报告第 3--6 节的 lineage、成本、质量和 oracle gate，不能沿用当前结论。

另一个保守选择是只在全模态数明显大于 8 的深水/较高频域做 fixed-n pilot，但这会缩窄应用范围，论文中必须诚实陈述 selection rule。

### 7.3 更通用的 fallback

如果 modal fidelity 仍不能通过 oracle gate，建议转向通用 Helmholtz 离散保真度：

- LF：独立 coarse mesh / lower-order discretization solve；
- HF：收敛的 fine mesh / higher-order solve；
- LF 完成后才允许 prolong 到共同评价网格；
- 严禁从 HF 解降采样生成 LF。

这个定义可覆盖非 Pekeris 几何、空间变系数与边界条件，也更接近 AI4Physics 社区熟悉的 multi-fidelity setting。代价是需要新的数值收敛与计算成本审计。

## 8. 复现入口

```bash
PYTHONPATH=src /home/ubuntu/project/ocean-acoustic-agent/.venv/bin/python \
  -m spectral_continuation.audit_ocean_multifidelity_solver \
  --config configs/ocean_multifidelity_pilot_20260916.json \
  --output artifacts/ocean_multifidelity_solver_audit/summary.json

PYTHONPATH=src /home/ubuntu/project/ocean-acoustic-agent/.venv/bin/python \
  -m spectral_continuation.generate_ocean_multifidelity_pilot \
  --config configs/ocean_multifidelity_pilot_20260916.json \
  --output-dir artifacts/ocean_multifidelity_formal_pilot
```

生成器默认拒绝覆盖已有 output directory。正式 archive 的完整哈希、paired identity 哈希和每个 solve 的 component count 均记录在 manifest 中。
