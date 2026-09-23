# Research handoff for future agents

本文件约束本仓库后续研究 Agent。开始编码或训练前，必须先读：

1. `README.md`；
2. `docs/CURRENT_STATUS_ZH.md`；
3. `docs/EXPERIMENT_LEDGER_ZH.md`；
4. 当前任务若涉及新模型，再读 `docs/WAVEFIELD_REPRESENTATION_NEXT_STAGE_ZH.md`。
5. 当前任务若涉及相空间、多路径或 wave token，再读
   `docs/MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md`。

不要通过从文件名猜测进度来选择实验。`docs/`、`reports/` 和 `src/` 中有大量已结束阶段，完整技术路径
保留是为了复现，不表示它们仍是 active roadmap。

## 当前主线

研究问题是：从少量 complex response views 中学习可变观测预算的 wave-system representation。当前 M0
方法是 5-channel token 加 shared 4-slot set operator；所有预算共用约 0.98M 参数。

新的三 seed set-operator 结果：full-context RelL2 为 0.0533，raw 为 0.0593；每源只有 1--2 个频率时
相对 raw 降低 24--30%，20 dB 为 0.112 vs raw 0.529。10 dB 仍失败；clean ellipse 与 128-grid boundary
上 raw 更强。instance-contrastive 预训练冻结 probe 约 0.31，且不改善 full-label 最优点，不是当前贡献。

2026-09-23 的公开 WaveBench 独立验证使用 8,000/500/500 system split 和三 seed。phase-coordinate 在
1--4 frequency budgets 相对 raw 分别降低 7.6%/11.6%/12.7%/12.5%，所有 seed 同向；full-context
paired bootstrap 95% 区间为 12.10%--12.93%。该结果确立跨数据集实用性，但未过预注册 15% 强效应
gate。两个更复杂的 fixed complex-edge token 都更差，不再枚举手工 normalization。

这些结果支持 measurement-efficiency representation，不证明 unseen-query complex field generation 已解决。

## 当前执行顺序

除非用户明确改变方向，否则按以下顺序推进，不能把组件一次堆满：

1. 冻结二维 baseline：single-scale soft phase token + shared slot set operator；
2. 多分支方向必须先过 branch-identifiability/oracle gate，再做 fixed wave-packet + soft top-K matched POC；
   不得从大 Transformer 或完整生成器开始；
3. 只推进 dimension-agnostic complex-connection graph/point operator，先在 synthetic 与 WaveBench matched
   比较；至少一域改善、另一域不退化才进入下一步；
4. 通过后才做 3D system-readout existence gate；
5. 若继续 field generation，先过 continuous oracle-medium unseen-query decoder gate；
6. encoder--decoder cascade 与真实部分观测。

`docs/EXPERIMENT_LEDGER_ZH.md` 中的 underexplored signals 是候选支线，不得全部并入第一阶段。

## 禁止无条件重复的路线

- 在相同 74--100 Hz 信息条件下继续增加 pole 数、局部曲率、Neural ODE 或 hypernetwork 容量；
- 把若干相距很远的低频 raw fields 直接拼接，期待模型自动解开 phase alias；
- 把极低的 patch-span oracle 当成可部署方法；oracle 使用了 query truth 来拟合系数；
- 在相同 90--106 Hz 协议上扩大 free local coefficients、frequency-mask loss 或 Transformer；
- 为 set encoder 继续扫 moment、更多 slot、slot-stack、metadata FiLM、coordinate basis 或 masked-view
  prediction；这些 matched POC 均未改善冻结 transfer；
- 把当前 instance-contrastive checkpoint 当成核心贡献；从头监督训练最终更好（0.0491 vs 0.0639）；
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
