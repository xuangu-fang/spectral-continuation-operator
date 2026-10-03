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
新增/运行中的artifact状态不变。原设置8/12 Hz的513-grid配对控制正在另建
`coherent_structural_reference_20261003/`，不在66文件清单中。

审计 artifacts：`artifacts/coherent_helmholtz_convergence_20261003/` 与
`artifacts/coherent_helmholtz_convergence_fine_20261003/`。

## 旧 FDTD 数据的失败如何使用

`wave_operator/operator_id_64_160.npz` 的粗网格 R0 留点拟合失败被完整保留。
部分高频在 stride2 观测下存在严重采样风险，局部拟合还会出现样本数少于基函数数的病态问题；不能
把高误差直接解释为“异质波场无法表征”，也不能仅凭这一诊断断言全部失败都来自 aliasing。
该组结果不作为新方法的正式第二域；新 Helmholtz teacher 用于隔离数值精度、采样与表征限制。
