# Research handoff for future agents

本文件约束本仓库后续研究 Agent。开始编码或训练前，必须先读：

1. `README.md`；
2. `docs/CURRENT_STATUS_ZH.md`；
3. `docs/EXPERIMENT_LEDGER_ZH.md`；
4. 当前任务若涉及新模型，再读 `docs/WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md`。

不要通过从文件名猜测进度来选择实验。`docs/`、`reports/` 和 `src/` 中有大量已结束阶段，完整技术路径
保留是为了复现，不表示它们仍是 active roadmap。

## 当前主线

研究问题是：从少量 complex response views 中学习可复用的 wave-system representation。当前最强证据
来自 5-channel token：归一化 amplitude，加两个空间方向上固定物理距离的复相位 advance。

三 seed 主结果：

- 跨 source、frequency 不重叠的 system retrieval：100.0% Top-1；
- IID hidden-medium contrast RelL2：0.0504，raw complex 为 0.3588；
- 未见 ellipse geometry：0.0955，raw complex 为 0.4637。

这些结果证明 representation sufficiency，不证明 unseen-query complex field generation 已解决。当前
missing-block field error 仍约 0.93--0.97；10 dB 与 partial/closed 强反射边界也未通过。

## 当前执行顺序

除非用户明确改变方向，否则按以下顺序推进，不能把组件一次堆满：

1. shared view encoder + permutation-invariant set aggregation；
2. multi-scale、fixed-physical-offset phase advance 与 confidence fallback；
3. oracle-medium unseen-query decoder gate；
4. encoder--decoder cascade；
5. 真正部分观测和第二独立物理域。

`docs/EXPERIMENT_LEDGER_ZH.md` 中的 underexplored signals 是候选支线，不得全部并入第一阶段。

## 禁止无条件重复的路线

- 在相同 74--100 Hz 信息条件下继续增加 pole 数、局部曲率、Neural ODE 或 hypernetwork 容量；
- 把若干相距很远的低频 raw fields 直接拼接，期待模型自动解开 phase alias；
- 把极低的 patch-span oracle 当成可部署方法；oracle 使用了 query truth 来拟合系数；
- 在相同 90--106 Hz 协议上扩大 free local coefficients、frequency-mask loss 或 Transformer；
- 在当前 AR calibrator 上继续搜索随机 sensor masks、value head、RL policy 或重复 reset；
- 在 dense oracle 没有先通过时训练昂贵的 sparse multi-fidelity calibrator；
- 因使用 flow/diffusion 就声称任务多模态。必须先有同一条件对应多个真实可行 target 的证据；
- 用 amplitude、phase coherence 或挑选近频区间替代 raw complex RelL2 来制造主要性能结论。

上述路线只有在实验账本写明的“重启条件”满足时才能重新打开。

## 不要过度解释已有失败

- 普通 U-Net 使用 oracle medium 仍失败，只否定当前数据规模和 image decoder；不否定 differentiable
  solver、microlocal decoder 或其他 wave-aware propagator。
- 少源 Nyström 约 0.99，只否定当前预算下的全局 column completion；不否定局部或层次 Green 表示。
- raw budget 曲线随 source 数增加而变差，主要受固定 channel concatenation 混淆；不能写成“更多
  观测有害”。
- ellipse OOD 仍属于同一 simulator，不是跨 PDE、跨 solver 或真实数据泛化。
- reliable-horizon 的高 AUROC 证明 failure 可预测，不证明既定 calibration action 可修复 failure。

## 每个新实验的最低流程

1. 在配置或实验说明中写出单一 hypothesis、matched baseline、主指标和 stop rule；
2. 先做 information/oracle gate，再训练复杂模型；
3. 按 physical system 切分，不能把同一 Green operator 的 source/frequency views 跨 split；
4. 主指标使用逐系统等权 raw complex relative L2；representation probe 可额外报告 retrieval、IoU、
   correlation，但不能代替最终 query error；
5. 同时报告 amplitude 与 phase/coherence，用于解释而非改写主结论；
6. 对噪声、网格变化和 boundary 分开报告，禁止只汇总平均值；
7. 正式结果至少三 seed，并保留 paired environment statistics；
8. 结果完成后更新 `docs/EXPERIMENT_LEDGER_ZH.md`，包括失败归因、重启条件和 artifact 路径。

## 证据语言

- `成立`：matched、multi-seed、system-held-out 证据支持；
- `机制级正结果`：方向可信，但没有通过预注册方法 gate；
- `oracle/existence result`：仅说明表示或信息上限存在；
- `development result`：用于选方法，不得写成 untouched test；
- `NO-GO`：只对报告中冻结的信息、任务和方法族生效，不自动外推到所有变体。

## 文件与版本规则

- 当前 active 代码清单见 `src/spectral_continuation/README.md`；其余模块默认冻结；
- 不回写产生过正式结果的 dated config；为新实验新建配置；
- 大 checkpoint 和逐样本数组不提交 Git；提交配置、hash、summary JSON/CSV 和正文图；
- 不移动历史模块或 artifact 路径，旧 manifest 与报告依赖这些路径；
- 不把新的 exploratory run 混入 `artifacts/wave_operator_representation/summary/`；只有冻结汇总才能进入。
