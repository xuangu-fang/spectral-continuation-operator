# Cross-domain response-query 数据与复用说明

> 冻结日期：2026-09-26
>
> 统一 schema：`cross-domain-response-v1`
> 配置入口：`configs/cross_domain_response_query_20260924.json`

## 1. 统一轴与文件格式

所有域都被只读适配成：

```text
responses              complex64 [system, source, frequency, receiver]
sources                float32   [source, coordinate_dim]
receivers              float32   [receiver, coordinate_dim]
frequencies            float32   [frequency]
source_receiver_index  int64     [source]
split                   unicode   [system]
system_ids              unicode   [system]
valid_response_mask    bool      [source, receiver]
provenance_json         JSON scalar
```

其中 `source_receiver_index[s]` 表示与第 `s` 个 source 共址的 receiver；未知时为 `-1`。原始数据从未被覆盖。
本地大型 NPZ 和 checkpoint 由 `.gitignore` 排除，manifest、汇总 JSON、图和生成代码进入版本管理。

## 2. OpenFWI CurveFault-A

- 原始路径：`/mnt/data/xuangu-fang/ai-physical-dynamics/datasets/openfwi_curvefault_a/raw`；
- 使用 `seis2_1_0.npy`、`seis2_1_1.npy`、`seis2_1_2.npy`，分别固定为 train、validation、test；
- 每个 shard 为 500 个系统，原始波形 shape 为 `[500, 5, 1000, 70]`；
- 采样间隔为 1 ms，对时间轴做无窗 RFFT，保留 8、10、12、15、18、20、22 Hz；
- source 对齐 receiver index 为 `[0, 17, 34, 52, 69]`。该映射只用 train shard 的互易误差确定；正确映射的误差约为
  `6e-7`，错误的等间隔猜测约为 `0.78`；
- 输出：`artifacts/cross_domain_response_query/data/openfwi_curvefault_a_v1.npz`；
- manifest 保存三个 700 MB 原文件的 SHA256。

重要限制：本机 velocity 文件与 seismic family 不匹配，因此本任务完全不读 velocity，不做介质条件泄漏。

## 3. Deepwave ring ultrasound

- 求解器：Deepwave 0.0.26 constant-density scalar time-domain finite difference；
- 网格：`96 x 96`，间距 1 mm，时间步长 `1.5e-7 s`，720 步；
- 阵列：64 个环形 receiver，其中 16 个同时作为 source；阵列位于 PML 内边界之外；
- 激励：150 kHz Ricker pulse；FFT 实际频率约为 92.6、120.4、148.1、175.9、194.4 kHz；
- IID 系统：随机平滑背景加 1--4 个椭圆 inclusion；
- OOD 系统：五瓣非椭圆 inclusion 与更宽 contrast；
- split：512/128/256/128 个 train/validation/test/OOD systems；
- 输出：`artifacts/cross_domain_response_query/data/deepwave_ring_v1.npz`。

为 continuous receiver query 另生成同一随机种子、同一 systems/sources/frequencies 的 128 与 176 receiver
版本，SHA256 和 shape 分别记录在：

- `artifacts/wave_as_prompt_pfto/data/deepwave_ring_128_v1.manifest.json`；
- `artifacts/wave_as_prompt_pfto/data/deepwave_ring_176_v1.manifest.json`。

大型 NPZ 只在本地保留。96-grid、半径 `0.32 * grid` 的离散圆在 256 nominal receivers 时只有 188 个唯一
cell，adapter 现会在仿真前拒绝重复 receiver cells，不能用重复位置伪造更密采样。

最初 220--340 kHz 的草案会低于每波长 6 个网格点，已在正式生成前废弃。正式频带在最低速度处满足离散精度警戒线。

## 4. Ocean multistatic acoustics

- 求解器：UnderwaterAcoustics.jl 0.8.1 的 `PekerisModeSolver`，`nmodes=0, ngrid=0`；
- Julia 依赖由仓库根目录 `juliapkg.json` 固定；
- range-invariant 两层海洋，水深 120 m，传播距离 4 km；
- 48 个 receiver depth，9 个共深度 source，4 个频率：80、90、100、110 Hz；
- 系统参数：水体声速、海底声速、海底密度；IID 与 OOD 参数区间写在 adapter 中；
- split：256/64/128/64 个 train/validation/test/OOD environments；
- 512 个系统正式生成耗时约 8.3 s；
- 输出：`artifacts/cross_domain_response_query/data/ocean_multistatic_v1.npz`。

这是受控 application replication，不冒充真实海试数据。它的作用是验证同一 query formulation 在模态波导中的可用性。

## 5. Structural lattice waves

- 方程：`(K - omega^2 M + i omega C) u = f`，其中 `C = alpha M + beta K`；
- 求解器：仓库内 batched dense complex128 direct solve，不依赖前三域的 acoustic solver；
- 网格：12×12 heterogeneous mass--spring lattice；8 个 perimeter sources，32 receivers，6 个 angular
  frequencies；
- system variation：平滑随机 mass/stiffness、局部软缺陷与各向异性；
- OOD：更强软缺陷、更大 mass contrast 和训练区间外 anisotropy；
- split：512/128/256/128 train/validation/test/OOD systems；
- 输出：`artifacts/cross_domain_response_query/data/structural_lattice_v1.npz`；
- SHA256：`5dbe1359ba6a8a2f53add0a581c16c8f6e3892298d296b16c83ae95222d50819`；
- manifest：`artifacts/cross_domain_response_query/data/structural_lattice_v1.manifest.json`。

它是跨 solver、跨 application 的受控 synthetic structural-wave benchmark，不冒充真实结构健康监测数据。
生成配置为 `configs/structural_lattice_wave_q3_20260926.json`，物理矩阵的对称性和正定性有单元测试覆盖。

## 6. The Well Helmholtz staircase

- 原始路径：`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wavefield_lr/raw/helmholtz_staircase_train`；
- 16 个频率文件，每个文件约 2.75 GB，shape 为 `[26, 50, 1024, 256]`；
- 时间轴严格由解析相位重复，adapter 只读 `t=0`，并共轭到仓库统一的负指数约定；
- 官方 card 说明整个集合由频率与 source position 组合生成，但 HDF5 没有保存 source coordinate；
- 在最低频用场峰和离散 Helmholtz residual 反推时，26 条 trajectory 都落在同一 cell。因此当前文件不能可靠构造
  source--receiver 共址矩阵；`source_receiver_index` 被诚实标为 `-1`；
- 输出：`artifacts/cross_domain_response_query/data/the_well_staircase_v1.npz`，证据角色仅为 public semantic sanity audit。

不能把 The Well 当前结果计作跨系统成功，也不能把 25.8% 的伪“互易误差”解释为物理不互易。真正原因是缺少 source metadata。

## 7. 复现命令

```bash
.venv/bin/python -m spectral_continuation.cross_domain_response_data \
  --config configs/cross_domain_response_query_20260924.json \
  --domain openfwi \
  --output artifacts/cross_domain_response_query/data/openfwi_curvefault_a_v1.npz

.venv/bin/python -m spectral_continuation.cross_domain_response_data \
  --config configs/cross_domain_response_query_20260924.json \
  --domain deepwave \
  --output artifacts/cross_domain_response_query/data/deepwave_ring_v1.npz

.venv/bin/python -m spectral_continuation.cross_domain_response_data \
  --config configs/cross_domain_response_query_20260924.json \
  --domain ocean \
  --output artifacts/cross_domain_response_query/data/ocean_multistatic_v1.npz

.venv/bin/python -m spectral_continuation.cross_domain_response_data \
  --config configs/cross_domain_response_query_20260924.json \
  --domain the_well \
  --output artifacts/cross_domain_response_query/data/the_well_staircase_v1.npz

.venv/bin/python -m spectral_continuation.generate_structural_lattice_wave_data \
  --config configs/structural_lattice_wave_q3_20260926.json \
  --output artifacts/cross_domain_response_query/data/structural_lattice_v1.npz \
  --device cuda
```

每次生成都会运行 schema、shape、有限值、ID 唯一性与 source--receiver index 校验，并原子写入 NPZ。
