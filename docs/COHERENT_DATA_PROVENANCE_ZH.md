# 连续复场表征的数据来源与复用说明

> 2026-10-03；对应 [24 小时执行协议](COHERENT_CONTINUOUS_REPRESENTATION_24H_ZH.md)。
> 本轮所有已打开的新数据均为 development；尚未生成或打开正式 test。
> 阶段一3组新数据共46个NPZ的SHA256、shape、频率，以及本地checkpoint清单见
> `artifacts/coherent_stage1_inventory_20261003/inventory.json`。

## Ocean：保留历史身份，补直接坐标求值

原始文件：
`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/spectral_continuation_ocean_integer_dense_74_125_20260915/paired_fields.npz`。

完整复场 `[160,52,48,384,2]`，74–125 Hz 整数频率，深度1–78 m、距离50–750 m。
原始128训练、32开发验证；32 strict_reserved 环境没有在文件内，本轮继续封存。
原始 SHA256 见 `configs/ocean_continuous_poles_stage_20260915.json`。
早期随机 seabed attenuation 参数未进入 Pekeris solver，不作为有效环境输入。

本轮局部编码器只读复样本、相对坐标与用于波长定标的 water speed/frequency；不读水深、海底声速、
密度、真实 modes 或 query 场值。用于比较的方向锥基线利用已知源在 range=0 这一几何。

新增直接求值：
`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/coherent_ocean_fine_queries_20261003/`。

- `system128_fine.npz`、`system129_fine.npz`：此前已打开的两个开发环境，不是新 test。
- `[F,95,767]` complex64；95/110/125/150/200 Hz；坐标轴独立保存。
- 全 modes PekerisModeSolver 直接在指定坐标计算，不用空间插值造真值。
- 先复算旧110 Hz粗网格，复场相对差约 `2.5e-8`，确认求解器、源、相位约定与旧档案一致。
- 时间约定 `exp(-i omega t)`，出射径向项 `exp(+i k_r r)`。
- 更高频测试若提供该频率空间样本，仅能称“表征跨频迁移”，不能称“无目标频率观测的频率生成”。

复现：`python -m spectral_continuation.generate_coherent_ocean --config
configs/coherent_ocean_fine_queries_20261003.json --output <dataset-directory>`。

新增非整数频率直接真值位于同一datasets根目录的
`coherent_ocean_fractional_queries_20261003/`：开发系统128–131，每个文件为
`[7,95,767]`，频率92.5/97.5/102.5/107.5/112.5/117.5/122.5 Hz。配置
`configs/coherent_ocean_fractional_queries_20261003.json`，仍由Pekeris直接求解，并通过旧110 Hz
复算身份检查。该数据是在阶段一inventory之后生成的，不包含于旧46文件清单；不能把旧清单当作
本轮新增数据的哈希证明。它用于零目标频率观测的谱查询，不用于空间context抽样。

## 独立二维异质 Helmholtz teacher

目录：`/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/coherent_helmholtz_development_20261003/`。

- 连续定义的4个平滑 Gaussian 低速体，背景声速1，单位正方形。
- 频率6–14，步长1；实值 Gaussian 空间源，宽度0.025。
- 求解 `Laplacian(u) + k(x)^2 u = -source`；`k=2*pi*f/c(x)*(1+i*loss(x))`。
- 边缘宽度0.16的三次 sponge 加小背景损耗；这是吸收层，不是完美匹配层/PML。
- 4阶、sine-compatible Dirichlet 离散，complex128 sparse LU；存储 complex64 场。
- 主物理区域在两轴 `[0.18,0.82]`，预先排除人工吸收层，保留源附近区域；源区/远场需另分层报告。
- 每个文件包含完整复场、连续环境参数、速度图、坐标、频率和求解残差，便于再生成更密网格。
- 32 train：seed 20261100–20261131；8 development validation：20261132–20261139。
- 拟训练频率6/8/10/12；开发查询频率7/9/11/13/14。空间表示实验在查询频率仍有 context，不是盲频率外推。

生成代码 `src/spectral_continuation/coherent_helmholtz_data.py`；配置
`configs/coherent_helmholtz_development_20261003.json`。40个系统已全部生成，完成情况可核对
目录 `audit.json` 的system记录与配置总数。

## 网格精度审计：已发现并修正一项风险

两个独立审计环境 seed 20261003/20261004，不混入上述训练/验证。
129→257 在14频率的内部区域复数差为0.02781/0.01908，第一例没有通过0.02门槛。
因此没有降低频率或改指标，而是将全部teacher改为257。
257→513在同频率的误差为0.001781/0.001234，通过0.005审计门槛；线性系统残差约 `1e-13`。
网格间差并非严格误差界，只是这两个case的收敛证据；不能自动推广为全数据误差保证。

额外检查整个开发集最低声速的seed 20261115（c_min约0.5856）：257→513最高频差0.002335，
也通过0.005门槛。该例是按数值采样风险选择，而非按模型误差挑选；详见
`artifacts/coherent_helmholtz_convergence_hardest_20261003/`。

独立细坐标开发数据位于 `coherent_helmholtz_fine_queries_20261003/`：两个已使用的开发系统
20261132/20261133，257和513两套网格，7/11/14频率。最高频网格差0.00128/0.00146。
空间连续查询评估以257-grid样本为输入、513-grid直接PDE解为目标，不用插值造真值；因此误差包含
求解器网格差，低于这一量级时不应夸大物理精度。

第二批非整数频率数据：`coherent_helmholtz_fractional_queries_20261003/`，仍为开发系统132/133
（完整seed为20261132/20261133），每个系统257与513两份文件，频率6.5/7.5/.../13.5。
对应配置`configs/coherent_helmholtz_fractional_queries_20261003.json`；两环境最高网格差
0.001131/0.001274。细网格真值不是插值；模型只允许插值其已知频率包络。

三组配对结构控制，均已生成两个相同geometry seed的257/513双网格，频率6/8/10/12/14：

- `coherent_structural_narrow_source_20261003/`：源宽度0.025→0.0125，介质不变；
- `coherent_structural_high_contrast_20261003/`：源不变，低速体contrast乘2；
- `coherent_structural_narrow_high_contrast_20261003/`：两因素同时改变。

contrast按预先声明的最低声速0.45统一缩放，最低值在513审计网格上检查；不按模型误差筛选环境。
最终contrast/source参数与有效缩放保存在每个NPZ的spec中。用冻结旧模型做配对敏感性分析，不能
宣称它们是独立随机抽样的正式OOD test。生成、收敛、评价配置均为`configs/coherent_structural_*`。
阶段一/二旧inventory不含全部新增文件；后续阶段三清单将记录这批数据，不覆盖旧时间点的快照。

阶段三清单已经记录上述8组共66个NPZ及当前代码/模型快照：
`artifacts/coherent_stage3_inventory_20261003/inventory.json`。这是捕获时的post-run快照，不证明之后
新增/运行中的artifact状态不变。原设置8/12 Hz的513-grid配对控制已经完成
`coherent_structural_reference_20261003/`，不在66文件清单中。

后续新增的距离相关一向波导、数值审计与谱查询pilot，见
[独立数据说明](RANGE_DEPENDENT_WAVEGUIDE_DATA_ZH.md)。它们尚未纳入阶段三清单；完成后另建快照，
不覆盖旧inventory。不要把该近似模型称为完整Helmholtz或可穿透海底Ocean数据。

`artifacts/coherent_source_space_*_cache_20261003/`中的NPZ是模型训练加速的二次型统计量，不是新场真值。
H/g/Q由已知物理与候选基得到；只有t/e/target_norm2依赖监督目标，不能作为编码器或物理系数求解的
输入。pilot旧metadata把Q也写为target-dependent，是过宽的文字标注，公式/代码实际不依赖目标；完整
cache已更正文案。全部预计算成本保存在summary.json，不能只报告之后的小矩阵GPU训练用时。

审计 artifacts：`artifacts/coherent_helmholtz_convergence_20261003/` 与
`artifacts/coherent_helmholtz_convergence_fine_20261003/`。

阶段四快照已生成：`artifacts/coherent_stage4_inventory_20261003/inventory.json`，12组共122个NPZ，
含完整40环境新波导、两组pilot/收敛及原设置配对细网格。仍为捕获时post-run快照；其后源空间容量
审计等新增artifact不在该时间点快照中，不能修改旧文件伪装成当时已有。

## 训练规模控制（20:22 UTC启动，512环境已生成）

新增路径`coherent_waveguide_training_cache_20261003/`位于同一datasets根目录。它不是新的完整
Helmholtz数据，也不是把图像插值成标签；是上述单向Galerkin模型用256个传播单元直接求出的
`[9,129,32]`复模态轨迹。生成前在3个固定seed、70/110/140 Hz用2049×129直接读出坐标比较256/512
单元，须通过0.001收敛gate。配置`configs/coherent_waveguide_training_cache_20261003.json`。

已生成512训练环境：原32训练seed20261300–331与480个新seed20261500–20261979。明确排除原8个
开发环境20261332–339；没有新增formal test。模型验证仍使用原2049距离的独立细teacher，而非新
129坐标缓存。缓存主要节省标签存储；原始完整环境spec与相位约定全部保留，但范围相关SSP不作为
observation-only模型输入。

目前512个文件已全部生成，数值审计最大差为0.00004214，低于0.001门槛；完成清单见目录的
`generation.json`。旧阶段四122文件inventory不含它们，不可据此宣称新数据已有hash证明。
启动条件是FiLM模型在原32系统的末步train已到0.02666，
不是为了掩盖先前0.105的容量失败而直接扩数据。

阶段五快照`artifacts/coherent_stage5_inventory_20261003/inventory.json`已记录13组、634个原始/训练
缓存NPZ的SHA256，并记录数据目录JSON元信息和当时108个artifact目录。它是post-run快照；随后
生成器识别/学习实验不在该捕获时点，后续需另存快照，不能覆盖旧文件。

## 冻结生成器的新query直接真值（16个NPZ已完成）

`coherent_waveguide_generator_query_transfer_20261003/`：原8开发系统20261332–339，每个保留原源
与源深度增加15m（预定[10,90]m边界）两种请求。频率72.5/87.5/107.5/122.5/137.5及150/170/200Hz；
前五为带内未见频率，后三为额外高频压力测试，分别报告。每文件[8,257,1025]复场；距离坐标
`(j+0.37)/1025*2000`、深度0.3–99.7m均为新坐标。由1024单元/32模态直接传播与基函数求值生成，
不是旧图像插值。两独立审计seed20262001/20262002的1024/32对2048/48，在107.5/200Hz、两种源
下最大复场差7.254e-6，低于预设0.001。完整配置/收敛/完成清单在数据目录，生成配置已版本管理。

这些是旧开发环境的配对能力测试，不是新独立系统或完整海洋Helmholtz。编码仍只读原60/100/140Hz
观测，不读取新源或新频率的场；请求源的Gaussian参数是公开query信息，重新计算初值，不用目标场
提取源。旧阶段五634文件快照不包含这16文件，后续另存阶段六快照。

## 高频/窄源的固定latent、可变解码分辨率真值

以下数据都是同一批开发环境的新query，不是新的独立训练/测试环境：

- `coherent_waveguide_adaptive_modes_truth_20261003/`：332/333两个系统（完整seed202613xx），
  原宽源，100/140/200/300/400Hz，2个NPZ。直接64模态、1024距离单元、256点Gauss积分，
  相对96模态/2048单元在独立20262001/2上最大差5.869e-6。
- `coherent_waveguide_narrow_source_modes_truth_20261003/`：相同两环境，仅查询源宽度改为2m，
  140/200/300/400Hz，2个NPZ；同数值设置，最大审计差5.676e-6。
- `coherent_waveguide_narrow_source_modes_expansion_20261003/`：完成其余334–339六开发环境，
  同窄源/频率/数值协议，6个NPZ，未按模型表现筛选。

每个场为385个深度×2049个新距离坐标，距离`(j+.37)/2049*2000`，深度0.3–99.7m；不插值旧场。
encoder仍只读原60/100/140Hz宽源场，生成器隐变量冻结，32/64模态对所有方法公平更换。原两系统
小诊断与完整八系统结果分存，不能把二者混为独立样本。源Gaussian参数属于公开query，介质真值
只允许teacher与明确oracle使用。全部NPZ保留原/改变后spec、相位约定和源变体；目录JSON记录配置、
收敛与完成清单。旧阶段五快照不包含这些10个文件。

阶段六快照`artifacts/coherent_stage6_inventory_20261003/inventory.json`已记录17组、660个NPZ及129个
artifact目录，包含上述新增真值。仍是post-run捕获，个别当时正在优化的artifact只代表捕获时已有
文件，不代表完成状态，也不证明此前运行的源码完全一致。后续多源loss实验不在该快照内。

## 新建的冻结候选独立系统审计（已生成，评估中）

`coherent_waveguide_frozen_system_audit_20261003/`包含`original/`与`narrow/`各64份NPZ，共128份。
系统20262300–20262363未参与原训练/开发。生成前已保存`freeze_manifest.json`，含三seed候选checkpoint
和基线超参数hash；完整生成记录含每个NPZ的SHA256。额外数值审计seed20262400/1/2与训练/审计系统
均不交叉，1024/2048单元、32/48（原源）或64/96（窄源）对比最大差**1.004e-5**。

原源文件是2049距离×129深度、60–140Hz九频率；窄源是2049新距离×385新深度、140/200/300/400Hz。
绝对相位、源/背景/边界信息条件与旧协议一致。该新集合是冻结候选的独立IID审计，不应加入训练或
开发调参，也不是第二个物理域。当前模型评估未完成；生成真值不等于新泛化证据已经成立。
旧阶段六660文件清单未包含这些128份，后续另存快照并单独标记它们的独立证据身份。

阶段七快照`artifacts/coherent_stage7_inventory_20261003/inventory.json`现已记录19组、788个NPZ、
145个artifact目录，独立64系统的两种query数据与开发数据分别标注。它是post-run快照；捕获时
seed43审计和新输入机制实验仍在执行，其artifact文件清单不表示完整结果，不回写成运行前证明。

## 旧 FDTD 数据的失败如何使用（历史）

`wave_operator/operator_id_64_160.npz` 的粗网格 R0 留点拟合失败被完整保留。
部分高频在 stride2 观测下存在严重采样风险，局部拟合还会出现样本数少于基函数数的病态问题；不能
把高误差直接解释为“异质波场无法表征”，也不能仅凭这一诊断断言全部失败都来自 aliasing。
该组结果不作为新方法的正式第二域；新 Helmholtz teacher 用于隔离数值精度、采样与表征限制。
