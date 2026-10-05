# 跨域 Wave Response Query：24 小时执行协议

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 冻结日期：2026-09-24
>
> 状态：completed；三个多系统域通过预注册 gate
> 目标：在 Ocean、Deepwave ultrasound、OpenFWI 与 The Well 上建立同一 response-query 接口；至少两个
> 多系统域通过 measurement-efficiency gate。The Well 只作为公开固定系统 sanity check。

## 1. 统一问题

每个物理系统由 response cube 表示：

$$
Y_e(s,f,r)\in\mathbb C,
$$

其中 $e$ 是 environment，$s$ 是 source，$f$ 是频率，$r$ 是固定 receiver array。给定不包含目标 source
的同频 context：

$$
D_K=\{(s_i,f,Y_e(s_i,f,:))\}_{i=1}^{K},
$$

预测：

$$
\widehat Y_e(s_\star,f,:).
$$

第一阶段只冻结 Q2 held-out-source。Q3 仅在数据提供足够密频率且 Q2 通过后，以 `reciprocal anchors +
per-link spectral state` 单独启动；不能把 Q2 写成 joint-query 成功。

## 2. 统一 archive schema

所有 adapter 输出不可变 NPZ：

```text
responses              complex64 [environment, source, frequency, receiver]
sources                float32   [source, coordinate_dim]
receivers              float32   [receiver, coordinate_dim]
frequencies            float32   [frequency]
source_receiver_index  int64     [source], -1 表示不共址
split                   unicode   [environment]
system_ids              unicode   [environment]
provenance_json         scalar unicode
```

原始数据不覆盖；adapter 写 SHA256、输入路径、处理参数、轴含义、单位、split 和代码版本。normalization 只读
train systems。

## 3. 数据域与不可混淆的证据角色

### 3.1 OpenFWI CurveFault-A：公开、多系统主域

- 本机三个 shard，各 500 systems；原始 shape 为 `[500,5,1000,70]`；
- shard 0/1/2 固定为 train/validation/test；先前任务读过 shard 2，故称 frozen replication test，不称
  从未开启的 blind test；
- 对时间轴做 RFFT，首轮固定 8、10、12、15、18、20、22 Hz；
- source 对齐的 receiver indices 冻结为 `[0,17,34,52,69]`；这个映射在 train shard 上由 reciprocity
  审计确认，不能按 validation/test 重选；
- target 是完整 70-receiver response。互易性只填 context-source 对应的少数 anchors，不能直接给整条线。

### 3.2 Deepwave ring ultrasound：受控、多系统主域

- 独立 time-domain scalar-wave solver；96 或 128 grid；
- 16 个环形共址 transmit/receive points；每个 shot 使用 broadband pulse；
- smooth background 加 1--4 个随机 inclusions；按 phantom seed 切分；
- 一次传播返回全部 receiver traces，RFFT 形成 dense source--frequency--receiver cube；
- geometry OOD 使用非椭圆或相连 inclusions；不把 OOD 用于模型选择。

### 3.3 Ocean acoustics：应用 replication 域

- 使用 `ocean-acoustic-agent`/UnderwaterAcoustics 独立生成多 source-depth、receiver-depth complex matrix；
- 先审计 exact reciprocity 是否使 full-budget baseline 接近解析极限；
- 若 K-full baseline 小于 0.10，不强迫 half-budget 击败解析答案，采用第 5 节条件式 gate。

### 3.4 The Well Helmholtz staircase：公开固定系统 sanity check

- 16 frequencies、32 source positions；本机 train 文件含 26 sources；
- 在 source coordinates 处采样 full fields，形成 multistatic matrix；
- 只有一个固定 staircase geometry，不计作 system-held-out 泛化，不参与“两个数据域通过”的计数；
- 用于 reciprocity、source interpolation、frequency interpolation和阵列规模适配审计。

## 4. 方法阶梯

所有域先跑相同顺序，前一级不过信息 gate 就停止昂贵训练。

### M0：无训练物理与信息审计

1. FFT energy / usable frequencies；
2. reciprocity residual 与 source--receiver alignment；
3. train mean、nearest source、complex coordinate interpolation；
4. reciprocity-anchor fill；
5. training-system PCA completion；
6. K-budget error 与 oracle headroom。

### M1：最小跨域 response operator

保留解析 reciprocity anchors，学习其余 receiver response：

```text
context complex responses
    -> shared 1D receiver encoder
    -> permutation-invariant/query-attentive source aggregation
target source + frequency + receiver coordinate + routed baseline/mask
    -> coordinate-aware residual decoder
```

同一实现支持 receiver line、ring 和一般 ordered array。首轮不加入 flow、PDE residual、多分支 packet 或
active policy。

### M2：只允许单变量迭代

按以下归因顺序，每次只改变一项：

1. normalization / physical relative loss；
2. receiver-local feature 保留，而非 global token；
3. periodic versus non-periodic receiver convolution；
4. query-to-context attention；
5. amplitude + local receiver phase advance 表示。

若三轮仍无稳定收益，记录为 information、representation、optimization 或 dataset-size failure，不继续堆
loss。

## 5. 预注册量化 gate

主指标始终是逐 environment 等权 raw complex relative L2；amplitude error、weighted phase coherence 与
circular phase error只用于归因。

### 5.1 同预算 gate

在最大 half-source budget 上：

- 相对最强同预算非神经 baseline 至少降低 20%；
- paired-system bootstrap 95% improvement 下界大于 0；
- 至少 75% test systems 改善；
- 三 seed 同向。

### 5.2 半预算测量效率 gate

- OpenFWI：ours K=2 不高于 full-coverage physics baseline K=4 的 1.10 倍，理想情况直接更低；
- Deepwave 16-source：ours K=8 不高于 baseline K=15 的 1.10 倍；
- Ocean：采用对应的 half/full budget。

如果 full-budget reciprocity baseline 本身低于 0.10，改为：ours half-budget 相对同预算 baseline 至少降低
20%，且与 full-budget baseline 的绝对差不超过 0.05。这个分支由训练前审计触发，不能看 learned test 后
选择。

### 5.3 绝对精度目标

- OpenFWI raw complex RelL2 不高于 0.40；
- Deepwave 不高于 0.35；
- Ocean 不高于 0.45；
- The Well 只报告，不设跨系统 promotion gate。

至少两个多系统域同时通过同预算 gate 与各自绝对目标，才称跨域 POC 成立；半预算 gate 单独标记。

## 6. Stop rules 与防止事后选择

- frequency bins、source alignment、receiver ordering仅由物理元数据或 train split确定；
- test 不参与 architecture、频带、loss weight 或 step 数选择；
- 每个域最多三轮有明确归因的结构迭代；
- 若 PCA/oracle half-budget 仍明显差于目标，停止训练并报告 information failure；
- The Well 的单 geometry 成功不能替代任一多系统域；
- 不用 amplitude-only 指标替代 raw complex error，也不通过删掉困难 systems 达成 gate。

## 7. 交付物

```text
configs/cross_domain_response_query_20260924.json
src/spectral_continuation/cross_domain_response_data.py
src/spectral_continuation/audit_cross_domain_response.py
src/spectral_continuation/train_cross_domain_response.py
artifacts/cross_domain_response_query/<domain>/...
reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md
```

每个数据 adapter 保存 manifest、SHA256、axis/schema validation；每个实验保存逐 system CSV、summary JSON、
配置、checkpoint hash 和正文图。最终更新 README、CURRENT_STATUS、EXPERIMENT_LEDGER、REPRODUCIBILITY
和 AGENTS handoff。
