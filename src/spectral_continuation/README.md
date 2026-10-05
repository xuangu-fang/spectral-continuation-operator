# 代码地图：复用组件，不默认续跑旧实验

> 2026-10-05。现有 Python 模块保留原路径，保证旧 checkpoint、import 和复现命令不失效。
> 这次重构改变组织与入口，不改模型数值行为，也不声称已经实现三条新路线。

任务定义：[A 补全](../../docs/wave_as_prompt/RECONSTRUCTION_ZH.md) /
[B 新实验](../../docs/wave_as_prompt/NEW_EXPERIMENT_ZH.md) /
[C 校准](../../docs/wave_as_prompt/CALIBRATION_ZH.md)。

## 1. 优先复用的组件

| 用途 | 代码入口 | 复用边界 |
|---|---|---|
| source–frequency–receiver 统一数据 | [cross_domain_response_data.py](cross_domain_response_data.py) | B / 阵列 A；保留 source/receiver/FFT 来源 |
| 多源完整场 toy | [wave_operator_data.py](wave_operator_data.py) | A/B；先核对数值精度和系统 split |
| WaveBench 读取 | [train_wavebench_identifier.py](train_wavebench_identifier.py) | 读取器可复用；旧 trainer 是介质识别，不是 A |
| Ocean 直接查询真值 | [generate_coherent_ocean.py](generate_coherent_ocean.py) | 用新输出目录；不要覆盖旧场 |
| 单向波导生成 | [coherent_waveguide_data.py](coherent_waveguide_data.py) | 受控机制，不冒充完整海洋 |
| Helmholtz 数据 | [coherent_helmholtz_data.py](coherent_helmholtz_data.py)、[multiscale pilot](coherent_helmholtz_multiscale_pilot.py) | 介质标签与场监督分开，真值参数需剥离 |
| 安全公共输入提取 | [prepare_helmholtz_sparse_dev.py](prepare_helmholtz_sparse_dev.py) | 示例信息白名单；旧三频切片不是 A 同频协议 |
| 度量/统计 | [metrics.py](metrics.py)、[paired_environment_stats.py](paired_environment_stats.py) | 检查单位、相位权重和按系统汇总；新协议不要悄悄改变指标 |

## 2. B 的强基线组件

| 模块 | 角色 | 是否等于新 B 已评测 |
|---|---|---|
| [cross_domain_query_audit.py](cross_domain_query_audit.py) | source 插值、互易性、复数 response subspace | 否，沿用时锁定同频协议 |
| [probabilistic_response_operator.py](probabilistic_response_operator.py) | 学得各方向先验方差的解析 MAP | 否，但必须保留其强统计对照 |
| [posterior_memory_response_operator.py](posterior_memory_response_operator.py) | posterior-kernel / memory 检索 | 否，需与目标协议匹配 |
| [train_matched_q3_cnp.py](train_matched_q3_cnp.py) | 通用 CNP 学习基线 | 旧 Q3 同时换源/频率；需适配，不直接套结果 |
| [train_episodic_response_basis.py](train_episodic_response_basis.py) | 旧可训练 basis 对照 | 已有负/弱结果，不再当未经探索的新想法 |

## 3. A/C 的表示和物理诊断

- [coherent_representation.py](coherent_representation.py)、[learned_wave_chart.py](learned_wave_chart.py)、
  [learned_coherent_kernel.py](learned_coherent_kernel.py)：坐标/振荡表示的既有候选与容量对照。
  单场拟合成功不代表 sparse prompt encoder 成功。
- [learned_speed_manifold.py](learned_speed_manifold.py)、
  [audit_learned_speed_manifold_confirm.py](audit_learned_speed_manifold_confirm.py)：
  已验证的受限学习变量机制；PCA+经典推断，不包装成现代神经算子。
- [learned_waveguide_generator.py](learned_waveguide_generator.py)、
  [audit_helmholtz_generator_identification.py](audit_helmholtz_generator_identification.py)：
  保留完整轨迹 GN / PDE 逆解作为强对照。
- [sparse_calibration.py](sparse_calibration.py)、[paired_fidelity_protocol.py](paired_fidelity_protocol.py)、
  [ocean_multifidelity_solver.py](ocean_multifidelity_solver.py)：
  C 的旧协议、求解器与信息审计；需要新的偏差/同频切片，不能直接重命名旧结果。

## 4. 历史分支

未列为复用组件的训练/评测脚本默认是**保留复现，不是当前开发入口**。
包括 acquisition/RL、horizon、pole/ODE、通用 token、多分支 packet、posterior composer、
neural-field 容量扫描等。不要根据文件名中的 current、M0 或 pending 自动重启。

完整逐文件归类：[历史清单](../../docs/archive/INVENTORY_20261005.md)；
旧详尽代码说明：[归档版本](../../docs/archive/20261005_before_three_routes/src__spectral_continuation__README.md)。

## 5. 最小维护接口

```bash
python3 scripts/research_catalog.py list --route A
python3 scripts/research_catalog.py list --kind components --route B
python3 scripts/research_catalog.py check --local-data
```

命令从仓库根目录执行。只读目录，不导入研究训练模块，不访问 GPU，不读取数据内容。
新代码先围绕一张问题卡建立独立协议；不要为“整洁”移动几百个模块导致旧 checkpoint 失效。
