# 实验账本：有效事实、失败边界与可重启条件

> 历史证据 / 技术参考（2026-10-05 归档标记）：保留原实验、数值和复现路径。
> 文内“当前/下一步/待完成”描述属于当时阶段，不是现行任务；请先读[三条 Wave as Prompt 路线](wave_as_prompt/README.md)。

> 更新日期：2026-10-04（连续复波场表征、独立64与新32环境审计；旧表保留历史）
> 目的：让新的研究者或 Agent 在十分钟内知道哪些问题已经做过、结果究竟否定了什么，以及哪些正信号
> 仍未充分开发。本文是决策账本，不替代原始报告和机器可读 artifact。

## 1. 如何读取证据

**2026-10-04新增的容量—辨识边界（已用系统，事后开发；不是新确认）：** 同1024训练介质标签，余弦新16在800Hz的q-PCA真介质欧氏投影oracle从rank255的raw0.0778/0.0757降到rank768的0.00309/0.00311，说明额外3倍代码可弥补线性q容量；但同低频5+5步无持续先验的rank768部署raw反而0.123/0.123。高斯代码MAP强度1e-6在前8开发、后8固定强度条件性复核后合并为0.03185/0.03353，仍高于冻结声速255的0.00538/0.00505。首8的0/1/3/5步结果单调改善，不能把无先验失败粗暴归因为“晚期过拟合”；MAP强度之外的先验/优化尚未排除。**不要只引用q768无先验作为强基线，也不要把真介质oracle放进部署排行榜。** 图、数值/存储曲线与重启条件见[容量审计](Q_CAPACITY_ORACLE_AUDIT_ZH.md)、[报告第23节](../reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md)。下一主线应在新二维回散射/Ocean条件下预设强q-MAP与物理逆解，而非继续在旧余弦集扫超参。

**新增第二PDE数据M0通过、但完整场问题NO-GO：** 二维多尺度纹理＋局部包体Helmholtz的512训练介质标签、8开发复场系统、2独立网格审计系统已生成；257→513在14/17的raw最坏0.001984。完整65×65低频复场的经典谱反演初版raw0.145/0.129被定位为**错误的旧`q≥1`裁剪**（新介质约40%像素合法q<1），不能当学习余量；统一物理范围`[0.5,6.25]`修正后查询11.3/17 raw仅0.000123/0.0000684，129/257输入也约6e-5。两版结果/源SHA保留，见[完整M0数据和事故记录](HELMHOLTZ_MULTISCALE_M0_PILOT_ZH.md)、[报告第24节](../reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md)。下一次如果仍拿完整低频场做第二域，神经方法几乎没有正当提升空间；必须先定义有限接收线真实需求，并比较同信息PDE-MAP/正则化q。此项**没有**第二域学得表征或独立方法测试。

**有限观测输入已准备，方法比较未开始：** 同8个二维开发系统的6/10/14响应已沿四条×65点切成每环境780复数的公开NPZ，保留源/损耗/边界，不包含隐藏介质、完整低频场、目标场或seed；本机路径、冻结几何和白名单详见[四线输入记录](HELMHOLTZ_SPARSE_CONTEXT_ZH.md)。初版临时文件误含seed，已隔离并禁止使用。下一步的前置检查是同信息PDE-MAP及噪声/可辨识性，不是把8个旧系统直接拿来宣称跨PDE学习成功。

**四线信息—容量控制，仍非方法成绩：** 同512训练介质标签构造局部声速先验，在训练均值/公开源处的129网格PDE切线及假定1%复噪声下，rank64/256四线查询11.3 raw预期0.00382/0.01307；但额外读取真实介质的8已用开发环境rank64/256投影oracle为0.0623/0.00819，17频率同方向。低维的模型内后验小不代表实际高精度，必须先做真正的四线非线性编码与同信息强PDE逆解。图、公式、两信息条件的全部限制见[双审计](HELMHOLTZ_SPARSE_INFO_AUDIT_ZH.md)。

**局部线性化不能跨整个介质分布；129网格不够做0.01级17查询：** 在已用8环境、额外读取真实介质差值的事后控制中，训练均值处一阶物理展开的11.3/17 raw平均0.238/0.251，中位0.104/0.138；真实介质129网格直接查询与257教师之差0.00705/0.02849。它把“局部Jacobian足够”与“真实非线性逆解可达”再分开，且要求后续物理/学习方法在257或同等数值收敛网格上评价。源、完整48行和范围见[同一双审计的C节](HELMHOLTZ_SPARSE_INFO_AUDIT_ZH.md)。

最新执行分支见 [连续表征滚动报告](../reports/COHERENT_CONTINUOUS_REPRESENTATION_PROGRESS_ZH.md)。
除新建64环境和随后**另行冻结的32环境高频审计**外，当前级别均为C/开发证据：高精度逐场表示通过；Ocean learning优势被更强扩散补偿基线推翻；异质介质
共享编码仍需强插值比较；局部空间chart直接跨频插值无收益；经典模态跨频有正信号但不是AI创新。
不得把这些结果与下面历史已冻结的多seed事实混用。

**最新决策事实（05:41 UTC）：** 同128环境实码、16→32介质深度项的训练统计先验，在另行冻结的
32个新IID同族系统上，使800Hz宽/新窄源raw0.01196/0.01218→0.00497/0.00543，
系统配对改善58.5%/55.4%，各31/32胜出；但全局基和解码成本增长，仍是PCA+GN+物理读出，
不是深学习跨域贡献。旧8开发上64项再增益17%–20%；边界系数解析闭合虽准确，最终复场
真实边界斜率oracle仅改善15%–16%，没过预定30%门槛，停止小网络M1。更决定性地，若额外
知道旧合成介质恰为15参数Gaussian族，24训练初值+5次经典GN可把旧8系统800Hz宽/窄源
raw降至0.000266/0.000254。24初值是看到初版2例陷局部最优后的post-hoc开发控制，不是
同信息/同成本或新32预注册方法；它证明现数据族的低维性会削弱普适表示结论。新的16×16
相关随机谱介质族已在目标真值生成前登记并执行，仍属同一个单向PDE。最新报告第16–20节
给出图、SHA与正负归因。不要重跑固定项数/弱优化器扫描，也不要把“精确生成族已知”的收益
记到AI方法名下。

**最新优先读（2026-10-04）：** 全程传播Jacobian + 128维训练PCA先验的Gauss–Newton，
原8开发环境1/5步0.000734/0.000702，显著胜过神经和旧L-BFGS。64独立IID系统的后补对照
1/5步0.000824/0.000677；新窄源400Hz0.005930/0.005273。原冻结神经三seed已全部完成：
带内直接0.008560、20步0.004105；400Hz直接0.04524、20步0.03023。原协议和后补强对照
分别保留，不能宣称当前神经有最强精度/校正优势，也不能把PCA当成不需要数据学习。
神经初值在PCA子空间外的残差阻碍受限更新，先投影回训练子空间后恢复到经典相近精度。

观测复残差输入控制失败：249520实参数、同初值、同3000步，创新输入0.006907、零特征续训
0.006855。停止增加输入/宽度/帧变换扫描。只反驳本次特征机制，不证明复响应没有信息。

少剖面诊断也有重要反证：17剖面GN5仍0.000730；5剖面只有更新正则时50步0.07774，
但加入整个白化坐标的Gaussian先验惩罚，在三个预设强度下变成0.00879/0.00762/0.01410。
所以不能把“几乎拟合观测但query差”直接称不可辨识；先验失配/优化仍是实质原因。均为旧8开发
环境，完整结果见profile-budget文档；不再扫先验强度，当前回到连续表示与解码精度边界。

下文逐阶段文字是历史时间线，其中“待补/进行中”只描述当时状态；被上述强控制取代的优势不得复用。

01:15 UTC表示机制更新（均开发诊断，不是新正式test）：

- 连续range读出16项、256段+5步：旧query0.000226、窄源400Hz0.005106；oracle32项0.000835，
  定位深度带宽瓶颈。同128维训练先验扩为32深度项后为0.000127/0.001552，先验容量oracle
  0.000184/0.001335。固定每环境代码不等于免费增加全局basis和求解成本。不继续rank容量扫描。
- 40源子空间审计：新代码平均raw0.001700，最坏复组合的系统均值0.005189，最大0.01534；
  8主+3数值审计NPZ保留。40源不是40独立环境。近等奇异值使固定输出rank8/16/32的最优误差
  0.8944/0.7746/0.4472；传统事实的有限模型验证，不攻击所有FNO/张量/算子PCA。
- 二维Helmholtz压缩：最强未压缩0.0000984，分离rank4/8为0.000393/0.000205，PCA128
  0.001713。真介质是4个可分Gaussian，rank4 oracle优势是数据结构。停止此域无机制的AE扩张。
  共享参数、训练介质标签与同离散teacher边界明确，见第二域审计。

本轮重启只接受明确的查询误差/计算瓶颈，不因现代AI标签再试flow、token或宽度。新问题应把
输入可辨识、表示容量、编码精度和读出分辨率分开；不能用之前弱优化器的差距支持神经必要性。

最新补充（23:23 UTC）：128维训练介质Gaussian/PCA先验容量oracle0.000810，接近1024系数
真介质0.000782。约束在该先验内的非线性反演20/100步约0.007668/0.00585，并未胜过Gaussian
初值后自由优化；不能据此归因为线性表示不足。暂停非线性latent/flow的无证据升级。
两开发环境、原15个物理参数内实际成对扰动支持局部query误差放大：110Hz约1.56–1.86倍，400Hz
约8.33–12.25倍。实际context差仅约1e-4，受边界限幅；不是1%→10%的实测、不是非唯一性定理。
保留首次JSON布尔序列化失败和完整重跑。下一单机制测试只增加已有观测的复残差，不改信息条件、
生成器容量或loss；详见`GENERATOR_COHERENT_INNOVATION_POC_ZH.md`。64新环境不参与此实验选型。

2026-10-03阶段一追加：相位—包络核的3seed局部误差为0.0107–0.0120，但context-only经典优化更好
（0.00725）；attention与两步端到端调整未胜出。源区之外收益更大，但主ROI不删源区。局部插值分支
暂停容量扫描；重启需要新的信息条件、可验证的误差机制或误差—成本优势，不是更换现代模块名称。

阶段二追加：Ocean模态谱插值的主要误差被定位到新模态出现边界，不是已有模态相位。使用目标真值
补齐模态是O级归因；可部署零锚点/截止修正只带来不稳定的小幅改善。独立非整数频率与细坐标真值
已生成，旧实现的区间外置零错误不能升级为信息论不可辨识结论。

已知介质/源/PDE的新协议下，相位预处理的小型基空间配合物理最小残差投影，在4个异质开发系统上
3/5个观测频率对应误差0.03243/0.01551。是C级经典正对照，不是已成功的learning方法。
PPRKS已有相关核心思想，必须据此界定新颖性；下一步检验首达走时、更复杂介质和可学习表示增量。

阶段二学习增量：首达走时、全局相位系数、1409参数相位MLP均未改善8环境约0.03335；双精度复跑
相同。固定18列径向包络0.02379，神经包络0.02308（约3%），两参数指数控制无改善。6列有限源复
Green参考在4环境已到0.02316。相位/包络网络尚未通过方法gate；下一步先查物理遗漏与复杂介质边界，
不直接扫网络规模。当前没有“学习普遍胜过经典物理表示”的证据。

独立细网格追加：未平滑相位载波重投影约0.397不是表示容量下界。沿用粗系数约0.029，目标拟合
oracle约0.020；假PDE残差集中在源附近。源尺度平滑恢复约0.029，保留源复响应后约0.020，同为6列。
这是经典正则性修正；不能宣传为AI改进，更不能把旧离散失稳写成物理不可辨识性。下一步只做与
该机制对应的平滑神经特征/包络与配对物理条件检验，见[机制备忘](OPERATOR_COMPATIBLE_WAVE_REPRESENTATION_ZH.md)。

阶段三追加：完整平滑特征/因子后，18列神经包络在8环境为0.02273（固定0.02371），独立513-grid
为0.02050（固定0.02160、6列源参考0.02118）；复包络1508参数也仅到0.02259。源参考载波与神经包络
结合只改善约1%。均未达到20%学习gate。源宽度×contrast配对控制中，contrast加倍时6列源参考
0.07069优于18列神经0.08540。停止同类点式MLP的盲规模扫描；下一步检验次级源/传播分解，不能
重新包装成已成功的学习基空间。新经典诊断必须计入背景求解成本。

阶段四否证：六列次级源/背景传播在独立细网格约0.01179，但共享编码从0.01455仅到0.01446，加入
PDE梯度信息也只到0.01416。更强的同PDE右预条件GMRES，在四环境×7/11/13Hz，从同场空间初值
出发四次背景逆即0.001987、六次0.0003467，而匹配次级源0.013523。因此暂停源空间网络扩张；
其逐实例oracle0.00682不构成方法优势，不能再跳过经典迭代比较。作用次数不等于公平端到端延迟。
重启须提出摊销成本、表示压缩或跨查询能力的新假设，并保留该强对照。

新距离相关一向波导40开发系统已生成，不是完整Helmholtz海洋。跨频模态的符号规范会造成虚假
谱难度：两个pilot用固定表面导数规范后误差显著下降，但仍需模态幅值/相位/耦合归因。此修正是
经典实现问题，不是新的物理不可辨识性，也不是AI贡献。详见报告第9节及独立数据说明。

阶段五：8环境波导的完整SSP局部模态/已知源残差谱插值为0.09241/0.08327；只用背景的模态法
0.21258。两个信息条件不同。完整SSP经典稀疏矩阵传播32/64cells为0.00282/0.000649，比所有
谱插值更强且无需context；这是低维求解器，不是学到的latent。相同频率的R0空间codec则只需
65/129剖面就到0.00156/0.000372，故下一步需预测紧凑包络/耦合，不是证明它能存下场值。
任何新学习方法必须说明完整介质是否可见；可见就保留强求解对照，不可见就将它标作额外信息上限。

结构性局部核容量试验：在同一2048 patch、64context、200步context边际似然下，1/2/4相位分量
分别0.007247/0.008415/0.010201。新增分量不改善可部署推断；按预注册条件不进行共享训练/分量数
扫描。这不是多路径表示容量不够的证明。`coherent_kernel_mixture_capacity_20261003`保存结果。

阶段五学习负结果：32系统的背景模态包络FNO，三完整context、不提供完整SSP。开发最佳仍为第0步
0.212574，末步0.310725；同输入RBF回归0.211940。补末步诊断的相同重跑得到train0.105016。
未过训练容量gate，不启动已准备的512环境新cache。重启需先查条件化或耦合后一个模态含多种传播
相位的结构问题，不将“有FNO”当作已经现代化成功。均为开发，不影响旧test封存。

条件化后续：FiLM + 相同4k步 train0.05229；同架构12k步train0.026657、dev0.302861，最优仍初值。
只有在这个训练容量gate通过后（20:22 UTC）才启动512环境cache的数值审计/生成。它不是泛化成功，
也不授权跳过同输入RBF回归及完整SSP经典上限。复数参数的实自由度须双计。
多载波系数控制亦失败：稳定局部4/8分支在ridge1e-4下约0.230/0.240 vs单载波0.213；全局衰减
形式甚至灾难性失稳，不能上升为物理不可预测性。幅相/空间展开控制也未给出超过约3%的改善。

规模控制已完成：相同FiLM/12k步/原8开发系统，32/128/512训练环境最佳dev分别
0.212574/0.201853/0.175268；同输入核回归0.211940/0.200983/0.161893。末步训练
0.02670/0.05013/0.09123。512训练缓存全部生成，数值审计最大差4.214e-5。数据多样性有益但
神经仍不如强基线，未过原定.10/20%gate；停止单纯width/step/data扩张。下一重启条件为新的共享
生成规律/物理坐标假设及其识别审计，不把更多数据后的微小优势写成论文成功。

共同生成器识别控制已完成：相同3频率×65剖面+背景，实慢度系数局部反演+相干矩阵传播为
0.01601，五次导数/距离正则为0.01136，远强于包络FiLM0.1753。257/2049剖面则0.00180/0.000789，
真介质投影oracle0.000782。这是经典物理收益，不是learning贡献；也推翻“高包络回归误差=信息
不足”的笼统归因。神经修正必须对照0.01136，并明确训练期额外真实介质监督。正在做，不是已成功。

生成器共享学习后续：系数监督神经0.008142（240304实参数），同监督核0.008874；单一复场loss微调
到0.006954，但context-only非线性反演5/10/20/50/100步更强地达到0.007527/0.006668/0.005983/
0.004680/0.003503。不宣称神经绝对精度领先。下一假设只限定为摊销精度—时间—存储，须公平实测，
并冻结后检查新频率/源/坐标。经典初值直接复场训练3000步0.012349，不等计算，不能把差距全归因于
额外介质标签。不能以胜过核的21.6%掩盖更强非线性控制，完整两域gate仍未通过。

阶段六补齐：生成器field微调三seed .006954/.007153/.007894，均值.007333，对核只稳定约17.4%，
未过20%gate。无介质标签11k等更新控制.010589；物理Gaussian/PCA prior20选项最好.009872。
端到端编码+6query warm pilot .1710秒（共享GPU），对经典5步约4.8×而非通过5×；充分迭代更准。
冻结新频率/新源/连续坐标可查询。2D Helmholtz同信息正弦谱反演65²已.0000989（相对同257离散
teacher），不在该无噪稠密协议继续盲训神经。不能因波导正信号忽略第二域的经典反证。

高频分辨率控制：宽源32→64mode没有收益；窄2m源在全8开发环境400Hz、冻结latent下，神经
三seed从约.123降到约.0369，oracle从.1168降到.00558。共享1024实生成器系数与可变物理读出
分离有实证，但该结构也适用于所有经典方法。64mode下核.04296、经典100步.04058，神经增量
并不巨大。正在把经典优化延至预定2000步，排除尚未收敛后才能讨论旧声源未激发方向的可辨识性。
只扩展已有开发环境的query，不是新的正式test；全部两环境先导与八环境结果分别保留。

优化控制随后完成：500/2000步原query .001669/.001634，窄源400Hz .02180/.02114，已经比神经更强。
不可把旧100步的劣势升级为未激发方向不可辨识性的结论。剩余学习候选是更快/更紧凑的共同算子推断。
系数监督模型对窄源400Hz三seed .03831/.03620/.03791，与场微调接近。下一单机制实验仅比较
同投影teacher的固定声源vs随机声源作用loss，明确训练期介质监督，原query选型，不接触高频标签。

该作用loss试验随后失败：随机源旧query .007660 vs固定 .006926，新窄源400Hz .03791 vs .03585。
停止三seed/源分布扫描，不按新query重新选模型。一阶局部平方根生成器反演控制（避免旧二阶省略项）
四组合最好.01069，未解决主要误差，不继续局部GN扫参。神经warmstart+相同20步物理校正则有较清楚
seed42信号：.003444 vs核warmstart .004642，端到端均约3秒。三seed与Gaussian priorwarmstart待补，
没有因此宣布独立环境或两域学习gate通过。

补强后：20步三seed均值.003506，Gaussian prior初始化同20步.004153，强于核.004642；神经只改善
15.6%，未过20%。不再单拿核初始化作为最强参照。冻结候选后启动64新IID系统独立审计，生成前保存
模型/配置hash；新系统不用于调参。结果尚未产生，不能由生成脚本/配置存在宣称已有独立泛化证据。

本仓库的结果分为四级：

| 级别 | 含义 | 可以如何使用 |
|---|---|---|
| A | system-held-out、matched baseline、三 seed 或强 paired 统计 | 当前主要事实 |
| B | development set 上的多 seed / 多环境稳定结果 | 决定是否继续，不作最终泛化 claim |
| C | 单 seed、小 pilot 或协议探索 | 生成假说，不作为方法贡献 |
| O | 使用 query truth 的 oracle / existence probe | 判断信息或表示上限，不可当部署结果 |

“NO-GO”始终带作用域。例如，free local coefficients 的 NO-GO 不等于 local representation 的
NO-GO；普通 U-Net decoder 的失败也不等于波动物理 decoder 不可能。

## 2. 当前事实总表

| ID | 结论 | 证据 | 状态 |
|---|---|---|---|
| F1 | coarse multi-frequency raw concatenation 不会自然带来跨频收益 | WaveBench 与 Ocean 单频/多频 matched comparison | 冻结负结果 |
| F2 | 频率采样必须相对传播时延足够密，才能辨识 phase slope | 79/80 Hz carrier 比 3 Hz 间隔显著更好 | 有效设计原则 |
| F3 | 结构化稳定复谱状态比 matched image U-Net 更适合频率查询 | Ocean 五频 macro 下降 34.5%，共同相位等变误差约 6.5e-5 | B 级正结果 |
| F4 | 增加生成器容量不能突破长 horizon phase identifiability | pole、ODE、curve、hyper variants 均失败 | 冻结负结果 |
| F5 | 局部低频 span 能表示目标，但自由系数不能由当前低频条件稳定确定 | 110 Hz patch oracle 0.052，learned local span 0.6186 | O 正 / 方法负 |
| F6 | 稀疏宽带 calibration 含实例状态信号，但当前方法只关闭约一半 oracle gap | near-band 0.4106→0.3611，12.05% | 机制级正结果 |
| F7 | failure risk 可跨实例、频率和空间稳定预测 | fixed-frequency AUROC 0.906/0.960；spatial AUROC >0.92 | 强正结果 |
| F8 | failure predictability 不等于 action controllability | repeated reset 无增益；随机 mask oracle headroom 仅 6.42% | 冻结负结果 |
| F9 | 物理测度决定 Green reciprocity | 普通 metric error 0.84，加权后 2.1e-6 | 强诊断，未深挖 |
| F10 | amplitude + local spatial phase advance 是当前最强 system representation | retrieval 100%；IID 0.0504；ellipse 0.0955 | A 级主结果 |
| F11 | 当前 representation 尚未解决完整 complex response query | raw/carrier/oracle-medium decoder 约 0.93--0.97 | 架构级失败 |
| F12 | shared slot set operator 将 phase token 的优势定位到低频率预算与 20 dB | 1--2 frequency 相对 raw 降 24--30%；20 dB 0.112 vs 0.529 | A 级主结果 |
| F13 | phase token 不是 universal clean-OOD winner | ellipse 0.100 vs raw 0.073；128-grid overall 0.394 vs 0.288 | A 级边界 |
| F14 | instance-contrastive 预训练不改善 full-label 上限 | pretrained 0.0639；scratch 0.0491；frozen 约 0.31 | 冻结负结果 |
| F15 | phase-coordinate 的观测效率在公开 WaveBench 上可复现，但效应为中等 | 1--4 frequencies 相对 raw 降 7.6%--12.7%；三 seed 全同向 | A 级正结果，15% gate 未过 |
| F16 | 更复杂的 fixed complex-edge normalization 不自动带来更好表示 | symmetric/directed edge 在全部 budgets 弱于 soft phase token | development 负结果 |
| F17 | strict joint source--frequency prompts 能识别跨 query 共享的 system state | 三域 global Bayesian operator 随 K 稳定下降 | A 级新事实 |
| F18 | nonlinear latent transport 的收益集中在 sparse/mid budget | OpenFWI 最高 11.7%，Deepwave 最高 7.1%；高预算饱和 | A 级中等正结果 |
| F19 | global 与 frequency-local state 是互补层级 | global PBNO 解 Q3；逐频率 head 在 Deepwave/Ocean Q2 更强 | 方法设计原则 |
| F20 | posterior latent energy 可暴露灾难性 prior shift | Ocean OOD fixed 8--15，prompt-only gate 约 1.0 | C 级 safety POC |
| F21 | posterior memory 的实例身份重要，但完整 response value 可被 spectral dictionary 极强压缩 | rank-16 value bank 为原始 0.65%/0.93%，误差退化小于 1.5% | A 级效率正结果 |
| F22 | prompt-conditioned empirical operator prior 跨到结构波方程仍有效 | structural lattice K=1/2 改善 10.31%/8.18%，bootstrap 下界为正 | A 级跨 solver 正结果 |
| F23 | in-prompt reconstruction risk 不能可靠替代 unseen-query extrapolation risk | hard LOO 在 Ocean K=2 退化约 10%；shared risk head 仍未全域过 2% gate | 冻结路由负结果 |
| F24 | posterior kernel 在 noisy prompts 下可比 global posterior 更稳，但 safety fallback 不等于精确预测 | OpenFWI/structural 10--20 dB 改善 9%--21%；Ocean 回退约 1.0 | A 级 robustness 边界 |
| F25 | 解析 posterior 与经验 priors 可统一成严格 fallback 的 latent operator，但小 calibration split 上的 token correction 不跨 system 泛化 | 121,603 参数统一算子复现强 adapter；learned correction 在 Ocean selection +8%，unseen test 却退化 3.7%--7.1% | 统一正结果 / refinement NO-GO |

## 3. 路线级复盘

### E01：APEX-H 单频谐波迁移

- **原假设**：carrier-relative / harmonic prior 能提高低频到高频迁移的样本效率。
- **可信正结果**：K=1 时 WaveBench 与 Ocean 分别比 direct 降低 7.48% 和 9.41%，样本级方向稳定。
- **决定性边界**：标签增多后优势消失；fully supervised operator 没有稳定更强；flow matching 的必要性
  没有证据。
- **正确归因**：这是低数据区间的 inductive bias，不是完整的通用频率算子。
- **不要重复**：更深网络、等价 loss、local adapter、selective gate 已无稳定收益。
- **仅在何时重启**：新的表示在 zero/few-shot 与充分监督之间显示明确、可解释的 scaling law。
- **证据**：[APEX 冻结证据](APEX_LEGACY_EVIDENCE_ZH.md)。

### E02：粗频带直接拼接

- **原假设**：增加多个低频场自然改善高频预测。
- **结果**：WaveBench 10/15/20→40 与 Ocean 50/100→200 均不优于强单频 direct；粗频 carrier 更差。
- **归因**：相邻频率间隔超过由传播时延决定的 phase-alias threshold，更多 aliased phase 不是更多可用信息。
- **不要重复**：在没有先做 delay / alias audit 时任意选几个频率拼接。
- **重启条件**：频率采样依据最大有效时延设计，或输入表示显式处理多分支 phase ambiguity。
- **证据**：[Stage-1 报告](../reports/STAGE1_REPORT_ZH.md)。

### E03：细频带 phase carrier

- **原假设**：相邻频率复相关能估计可迁移 phase slope。
- **正结果**：无训练 79/80→90 Hz 相对单频 carrier 降低 57.1%；known-medium 冻结模型在新生成的
  105/110 Hz 上相对 direct 降低 59.0%/50.7%；125 Hz 仍降低约 30%。
- **负结果**：训练区间内不如 direct，150 Hz 后 raw error 超过 1，收益随 horizon 衰减。
- **保留结论**：局部 spectral phase advance 是有效信息坐标，但只是 local continuation law。
- **尚未充分开发**：它尚未与当前 spatial phase-advance token 在同一个 set encoder 中统一比较。
- **证据**：[Stage-1 报告](../reports/STAGE1_REPORT_ZH.md)。

### E04：稳定复谱状态与连续 generator

- **原假设**：显式 gauge invariance、稳定 propagation state 比 image bottleneck 更适合 oscillatory fields。
- **正结果**：matched Ocean 105--125 Hz macro 从 image U-Net 1.184 降至 0.776；解析三路径 Prony 在
  20--24 Hz 达到 0.043 且跨分辨率保持。
- **后续最好结果**：校准协议下 hyper-pole 的 110 Hz 为 0.4473，优于 recurrence 0.4741 和 query
  U-Net 0.6125。
- **决定性边界**：113 Hz 左右与 recurrence 交叉；107--125 macro 反而更差。增加 pole、局部曲率、
  affine components 或 Neural ODE 都没有突破。
- **不要重复**：保持同一观测带却继续扫容量或积分器。
- **重启条件**：新增可辨识信息，例如更合适的频率采样、不同 excitation、部分 query observations，
  或明确的 multi-branch state supervision。
- **证据**：[振荡场 POC](OSCILLATORY_REPRESENTATION_POC_ZH.md)、
  [Generator gate](../reports/SPECTRAL_GENERATOR_GATE_REPORT_ZH.md)。

### E05：全局低秩、局部 span 与 masked frequency

- **原假设**：局部低频 basis 可表示高频，masked learning 可预测其系数。
- **oracle 正结果**：Ocean 110 Hz 的 8×16 raw patch-span oracle 为 0.052；三路径场为 0.0015。gauged
  带内 masked-frequency 解析恢复可到 0.089。
- **方法负结果**：learned free local-span 为 0.6186，frequency-mask 辅助变差到 0.6479；受限 local
  carrier token 只带来 0.97% 改善。
- **正确归因**：表示空间存在，但 query 对应的坐标不可由当前条件唯一确定；masked interpolation 不等于
  extrapolation。
- **不要重复**：更细 patch、更宽 token、相同协议上的 Transformer 或更多 mask loss。
- **重启条件**：context 新增能区分系统/路径的 latent，或训练分布提供同一局部 basis 的可辨识动力学。
- **证据**：[全局与局部诊断](../reports/GLOBAL_LATENT_DIAGNOSTIC_ZH.md)、
  [Local-token 结果](../reports/LOCAL_SPECTRAL_TOKEN_POC_ZH.md)。

### E06：稀疏宽带 spectral calibration

- **原假设**：固定稀疏阵列的短 sweep 能校准实例相关 propagation state，并改善完全无观测的后续频率。
- **正结果**：每频率 0.5% 传感器时，107--112 Hz 从 0.4106 降至 0.3611；shuffled-instance 恶化；
  收益主要来自 phase。
- **负结果**：只改善 12.05%，110 Hz 为 0.4044，改善环境比例 62.5%，均未过 gate；K>184 饱和。
- **信息审计**：两频 dense oracle 上限仅 15.8%；六频 dense oracle 可改善 24.4%。
- **不要重复**：两频自由 phase correction、继续增加随机传感器、直接训练多 pole sparse model。
- **重启条件**：先证明新观测设计的 dense oracle 有至少 30% headroom，且跨足够多系统稳定。
- **证据**：[Sparse calibration 报告](../reports/SPARSE_SPECTRAL_CALIBRATION_POC_ZH.md)。

### E07：独立 Ocean low/high fidelity

- **原假设**：8-mode cheap solver + HF calibration 可以形成真实 multi-fidelity task。
- **工程正结果**：LF/HF 独立求解与 lineage 成立；适用环境中约 1.8--2.1 倍加速。
- **任务负结果**：仅含两个 validation environments 的 tiny pilot 中，dense HF calibration 对
  107--112 Hz 只改善 15.69%；107 Hz 单步改善 45.88%，到 112 Hz 已为负。这是停止昂贵扩展的
  pilot gate，不是总体效应量估计。
- **归因**：固定 8-mode 在浅水可能等于 HF，在深水又删去过多独立模态；fidelity knob 不连续。
- **不要重复**：直接把旧 sparse calibrator 迁移到这组 LF/HF 数据。
- **重启条件**：连续、跨环境稳定的 fidelity knob，或改成 receding-horizon tracking，并先过 dense oracle。
- **证据**：[Paired-fidelity gate](../reports/PAIRED_FIDELITY_CALIBRATION_GATE_ZH.md)。

### E08：Reliable-horizon map

- **原假设**：不看 target truth，也能预测延拓何时、何处失效。
- **强正结果**：Ocean/Synthetic fixed-frequency AUROC 为 0.906/0.960；4×8 同场 spatial AUROC 为
  0.945/0.921；不同 patch 尺度和 reset 后均保持。
- **边界**：这是 error ranking / selective prediction，不等于 error reduction。
- **不要重复**：继续堆更复杂 risk head；简单 disagreement 已解释大部分收益。
- **尚未充分开发**：作为当前 wave-system encoder 的 confidence/readout，检验表征是否知道自己的局部
  phase token 在何处失效。
- **证据**：[Reliable-horizon 最终报告](../reports/RELIABLE_HORIZON_MAP_FINAL_ZH.md)。

### E09：Actionable horizon 与随机 sensor masks

- **原假设**：学习 action value 可以选择更好的 calibration endpoint 和 sensor mask。
- **机制正结果**：within-environment utility Spearman 0.624，shuffled-instance 显著更差。
- **决定性负结果**：learned value 0.5846 差于 risk-trigger 0.5720；三个随机 masks 的 oracle 相对最佳
  fixed 只有 6.42% headroom，新增 masks 仅改善 0.48%。
- **归因**：当前 actions 的信息子空间几乎相同，问题不在 selector capacity。
- **不要重复**：更多随机 masks、Transformer value head、RL、重复同类 reset。
- **重启条件**：actions 必须改变信息类型，例如 excitation、频率窗口、观测模态或几何，并先过 oracle gate。
- **证据**：[Actionable-horizon 报告](../reports/ACTIONABLE_HORIZON_POC_ZH.md)。

### E10：Wave-system representation

- **原假设**：amplitude 与固定物理间距的 local phase advance 比 raw complex pixels 更能表示隐藏系统。
- **主正结果**：retrieval 100%；IID medium 0.0504；ellipse OOD 0.0955；均显著优于 raw 和
  amplitude-only。
- **关键消融**：phase-only 0.108，amplitude-only 约 0.238，组合 0.050；加入绝对 normalized phase
  反而退化到 0.064。
- **边界**：10 dB 失败；partial/closed boundary 为 0.614/0.637；输入仍是 dense internal fields；
  retrieval 可能已饱和，不能单独承担后续方法比较。
- **架构负结果**：普通 missing-block decoder 即使输入 true medium 也约 0.965。这只否定当前 image
  decoder，不是信息论 no-go。
- **当前状态**：唯一 active 主线。
- **证据**：[完整 POC](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md)。

### E11：Green reciprocity 与物理 metric

- **诊断**：普通像素内积下 reciprocity error 约 0.84；按 source 处的 `c(s)^2` 加权后为
  `2.1e-6`。只用低频 cross responses 可把相对权重估到约 `5e-7` log RMSE。
- **意义**：响应算子的对称性依赖物理测度；默认欧氏 token aggregation 可能把可恢复结构隐藏掉。
- **尚未做**：没有把 learned measure 用于 encoder normalization、reciprocity pretraining 或
  source-selection generalization。
- **注意**：它没有救活少源 Nyström，不能写成 Green completion 已解决。
- **证据**：[Wave-system POC，第 5.2 节](../reports/WAVEFIELD_REPRESENTATION_POC_ZH.md#52-一个额外的物理诊断互易性依赖正确-metric)。

### E12：Variable-budget wave set operator

- **假设**：物理 phase token 能在参数量固定的 shared set operator 中提高观测效率，而不是只帮助固定通道
  U-Net。
- **正式结果**：24-view 为 0.0533，raw 为 0.0593；每源 1--2 frequencies 时相对 raw 稳定降低
  24--30%；20 dB 为 0.112，raw 为 0.529。
- **强负对照**：amplitude-only 为 0.1226，说明幅值稳定但不充分；strong raw 在信息充分时几乎追平。
- **边界**：10 dB 为 0.797；ellipse 与 128-grid partial/closed 均弱于 raw。
- **预训练归因**：instance contrastive 只加速早期训练，冻结可读性差，最终不如 scratch。
- **已停止架构**：moment、slot-stack、masked-view prediction、metadata FiLM、coordinate basis 与
  13-channel multi-scale token 均未过 frozen/cross-group gate。
- **正确 claim**：measurement efficiency 与 moderate-noise inductive bias，不是 universal OOD、foundation
  model 或 unseen-field generator。
- **重启条件**：新 acquisition 信息、可证明 identifiability 的自监督目标，或第二物理域上重复出现相同
  budget scaling。
- **证据**：[M0 报告](../reports/WAVE_SET_OPERATOR_M0_ZH.md)。

### E13：WaveBench 独立验证与 complex-connection 表示

- **假设**：amplitude + local phase advance 的优势来自一般的复传播坐标，而非原 simulator 或固定
  source layout；同一个表示应在公开独立数据上提高频率观测效率。
- **冻结协议**：WaveBench isotropic 10/15/20/40 Hz；按 50,000 个物理系统固定切分，8,000 / 500 /
  500 train/validation/test；同一 checkpoint 随机接受 1--4 个频率；raw、amplitude、phase-coordinate
  使用完全相同 encoder/decoder；三 seed，枚举每个预算的所有频率组合。
- **正式正结果**：1/2/3/4-frequency contrast RelL2 从 raw 的 0.3024/0.2422/0.2077/0.1811 降到
  0.2794/0.2140/0.1814/0.1584；相对降低 7.6%/11.6%/12.7%/12.5%。完整四频 paired bootstrap
  95% 区间为 12.10%--12.93%。
- **强负对照**：amplitude-only 为 0.3807/0.2891/0.2398/0.2088，说明稳定 envelope 不足以辨识
  system，local phase transport 提供独立信息。
- **未过 gate**：预注册要求至少两个 sparse budgets 平均改善 15%，本轮未达到；只能声称稳定中等效应，
  不能声称碾压 strong raw baseline。
- **方法负结果**：symmetric Hermitian edge 与 directed amplitude-transport edge 都略差于当前 soft
  phase token。不要继续枚举 fixed normalization 或把更多方向硬塞进 channel axis。
- **HarmoCore 归因**：Functional Tucker 的 continuous coordinate basis 和 3D scaling 值得继承；但
  per-frequency independent core 没有显式 system latent 或 frequency dynamics，不应用其 sparse
  reconstruction 成功替代跨频 system representation 问题。
- **重启/推进条件**：下一方法只测试 dimension-agnostic shared edge kernel；两个二维域至少一域稳定
  改善、另一域不退化，才进入 3D。3D 先做 system readout，不先做生成。
- **证据**：[WaveBench M1 报告](../reports/WAVEBENCH_REPRESENTATION_M1_ZH.md)。

### E14：多分支相空间 oracle 与 fixed Gram POC

- **单一假设**：partial/closed boundary 的失败部分来自单个 phase advance 把同一局部邻域中的多个相干
  传播分支压成一个统计量。
- **oracle 正结果**：受控 2/3 分支混合在正确 K 下，held-out complex RelL2 从 0.664/0.772 降到
  0.313/0.326；约 1.5-wavelength 的真实场窗口中，K=4 相对 K=1 在 partial/closed 上改善
  26.3%/25.7%，95% 区间均严格为正。
- **关键尺度发现**：固定 33-pixel 窗口只改善约 6%；9/17/25/33 pixels 的增益单调减小。局部
  plane-wave sparsity 必须按 wavelength，而不是 raster pixels 定标。
- **方法设计**：8 orientations × 2 radial scales 的固定 complex packet bank；归一化 packet energy 加
  complex Gram edges。Gram quotient 一个共同 global phase，同时保留 branch-to-branch relative phase。
- **WaveBench 机制结果**：relative-phase Gram 的单 seed full-context paired 改善为 3.54%，优于逐 packet
  独立 phase quotient 的 2.27%；只是 development evidence。
- **三 seed正结果**：synthetic 的 1-frequency budgets 改善 12.1%--17.7%；full context 仅改善 1.9%。
  64→128 boundary shift 中 open/partial/closed 的逐系统改善为 5.1%/11.1%/11.7%，所有 30 个物理系统
  同向。
- **未过 gate**：partial/closed 均未达到预注册 15%。不能用某个 seed 的 15%--16% 或 oracle 的 26%
  替代正式方法结果；fixed multiscale Gram 不提升为主方法。
- **失败归因**：oracle 用目标 patch 自适应选 atoms 与系数；fixed filters 不适应局部 wavespeed/curved
  wavefront，69-channel raster 没有显式 branch routing，下游 loss 仅间接监督局部分解。
- **不要重复**：更多固定半径、方向、bandwidth、Gram edge 或从头训练大通道前端。
- **重启条件**：引入受 wavelength 约束的 learnable analysis、显式 sparse branch routing，或一个直接
  需要 multipath 的 query；仍需通过同一三 seed 15%/2% gate。
- **证据**：[MPS-1 完整报告](../reports/MULTIBRANCH_PHASE_SPACE_MPS1_ZH.md)；机器汇总
  `artifacts/multibranch_phase_space_mps1/summary/summary.json`。

### E15：In-Context Wave Response Query（RQ-1 已完成）

- **核心问题**：从 unseen physical system 的少量 source--frequency complex receiver vectors 中，在不更新
  权重时回答 joint-held-out source--frequency query。
- **为什么现在推进**：hidden-medium recovery 已证明 responses 包含 system identity，但 dense missing-field
  generation 约 0.93--0.97；固定 receiver response query 将 system inference 与 dense oscillatory image
  decoding分开。
- **首轮剪枝**：receiver array 固定；Q3 joint query 为主；不加入 flow、RL、variable receivers、3D 或
  branch router。
- **先验审计**：conditional ambiguity、nearest query、coordinate interpolation、low-rank completion、
  context-conditioned retrieval、true-medium decoder、active oracle。
- **最小模型**：DeepSets/CNP；只有 global bottleneck 明确才做 query-to-context cross-attention。
- **主 gate**：Q3 相对最强非神经 baseline 改善至少 15%，K=1 到 K=8 稳定下降；所有统计以 physical
  system 为推断单位。
- **belief gate**：只有 context-near systems 对 target 出现分离 modes 且 deterministic mean 明显平均化，
  才训练 mixture/flow。
- **active gate**：best-next-query oracle 的 error-AUC 相对最佳 fixed schedule 至少改善 15%，且 uncertainty
  校准成立，才训练 policy；RL 最后进入。
- **协议**：[Response Query 主线](IN_CONTEXT_WAVE_RESPONSE_QUERY_PLAN_ZH.md)；配置
  `configs/in_context_response_query_rq0_20260924.json`。
- **Cartesian-array 负结果**：Q3 K=8 的 train mean/PCA 约 0.939；DeepSets 0.936、cross-attention
  0.927、spatial set operator 0.934、masked Transformer 最佳约 0.911。true-medium CNN/FNO decoder 最佳
  约 0.908/0.851。该 setting 未过 gate，不再只换 backbone。
- **任务修正**：采用 9 个 source/receiver co-located transducers，使 Green reciprocity 成为可用的信息路由；
  Q2 target source 不见、target frequency 已见。
- **最终方法**：log-gain-calibrated reciprocity routing + 2-layer、121,938 参数 masked residual
  Transformer；标准化训练后用 physical raw RelL2 小步微调。
- **IID 正结果**：三 seed test K=8 为 0.498 ± 0.014，calibrated reciprocity 为 0.624，相对提升
  20.1%，system bootstrap 95% 区间 [12.7%, 28.7%]。
- **geometry OOD**：ellipse K=8 为 0.427 ± 0.002 vs 0.558，相对提升 23.5%。
- **active oracle 负结果**：192 个 test episodes 上 best-next-source greedy oracle 的 error-AUC 为 0.603，
  nearest-first 为 0.642，只改善 6.0%，未过 15% gate；当前 Q2 不训练 policy/RL。
- **边界**：只证明 Q2 和 co-located array；Q3 joint query、任意 receivers、第二 PDE/真实数据仍未证明。
- **正式报告与配置**：[RQ-1 报告](../reports/IN_CONTEXT_RESPONSE_QUERY_RQ1_ZH.md)；
  `configs/in_context_response_query_rq1_formal_20260924.json`。

### E16：Cross-Domain Response Query（M0 通过，M1 已完成首轮）

- **核心假设**：完整复响应表在跨环境方向存在低维结构；少量 source probes 可以在 inference time 识别
  当前环境的 response-subspace coordinate，再回答 held-out source。
- **统一接口**：`Y[system, source, frequency, receiver]`；同频 Q2；system-held-out split；raw complex RelL2；
  context source 与 target 严格不重叠。
- **数据域**：OpenFWI 1500 systems、Deepwave 1024 systems、Ocean 512 environments；The Well 只作 fixed
  public semantic audit。
- **M0 方法**：对每个频率在 train systems 上学习 complex SVD response basis；query time 用 context rows
  做 ridge completion。rank/ridge 仅由 validation 选择。
- **OpenFWI 正结果**：K=1/2/4 为 0.213/0.176/0.120；同预算 baseline 为 0.547；相对改善
  61.3%/68.5%/78.7%，逐系统胜率 99.0%/99.6%/100%。
- **Deepwave 正结果**：K=1/2/4/8/15 为 0.145/0.115/0.080/0.059/0.053；相对 baseline 改善
  24.6%--72.6%。OOD K=8 为 0.094。
- **Ocean 正结果**：互易误差为 0，但 anchor IDW 在 K=4/8 仍为 0.886/0.564；subspace 为
  0.0043/0.00053。参数 OOD K=4/8 为 0.325/0.199。
- **measurement efficiency**：half-budget 对 full-budget local baseline 的误差比，OpenFWI/Deepwave/Ocean
  分别为 0.322/0.302/0.0076。三个域都通过预注册 gate。
- **M1 方法**：M0 routed baseline + 1D receiver-local encoder + source-set attention + coordinate-aware、
  zero-initialized residual decoder，约 293k 参数。
- **M1 三 seed**：OpenFWI 相对 M0 再改善 2.4%--8.7%；Deepwave 只改善 0.04%--3.5%，且 K=15 的
  paired interval 跨零。因此不把网络 residual 写成主贡献，不继续无条件增加 Transformer/loss。
- **The Well 负发现**：HDF5 不含 source coordinates，场峰与离散 PDE residual 都把 26 trajectories 定位到
  同一 cell。当前 `source_receiver_index=-1`，不能把伪互易矩阵当成物理证据。
- **不要重复**：在同一 Q2 上继续堆大 backbone；只报告 Ocean IID 极小误差；把 reciprocal anchors 说成
  完整答案；用 The Well trajectory index 假装 source geometry。
- **下一步强对照**：参数匹配的 generic masked Transformer、context-conditioned system retrieval、direct
  supervised ridge；只有排除 weak-baseline 解释后，才升级为论文主方法。
- **证据**：[跨域 POC 报告](../reports/CROSS_DOMAIN_RESPONSE_QUERY_POC_ZH.md)；
  [数据 provenance](CROSS_DOMAIN_DATA_PROVENANCE_ZH.md)；机器汇总
  `artifacts/cross_domain_response_query/*/audit_summary.json`。

### E17：Wave as Prompt latent-spectrum prior（M0 正式通过）

- **单一假设**：sparse prompt identification 的误差主要来自不同 system-latent directions 的可辨识性不均；
  train-system response spectrum 可以提供比 scalar ridge 更正确的方向相关先验。
- **方法**：保留同一 complex PCA response basis，把 $\lambda I$ 替换为由训练 latent variance 确定的
  diagonal Gaussian precision；闭式 MAP 更新，无 target leakage、无额外网络 backbone。
- **强基线修正**：baseline rank 搜索扩展到 128。Deepwave 原 rank-32 并非容量上限；rank-64 已把
  validation error 从约 0.0583 降至 0.0519。
- **三 seed half-budget**：OpenFWI 0.1292 vs 0.1745（25.9%）；Deepwave 0.04349 vs 0.05313
  （18.2%）；Ocean 0.001018 vs 0.001678（39.3%）。固定一次 validation selection 后跨三组
  prompt orders 评测，system win rate 为 98.8%/100%/100%。
- **probe saving**：OpenFWI K=3 达到 K=4 scalar；Ocean K=6 达到 K=8 scalar，均节省 25%；Deepwave
  没有达到该 gate。
- **OOD**：half-budget Deepwave/Ocean 相对 scalar 改善 16.1%/14.6%，但 Ocean 绝对 RelL2 仍为 1.032；
  Ocean K=1/K=8 分别明显退化，禁止 universal-OOD claim。
- **functional-rank 负结果**：独立 source--receiver CP 在三域失败；低存储 shared Tucker 只在 Ocean
  保真。receiver Fourier/DCT 即使 carrier demodulation 后也不能保留 OpenFWI/Deepwave query accuracy。
  正确低秩轴是 physical systems，不是振荡 source/receiver coordinates。
- **episodic basis 负结果**：Deepwave 无改善，OpenFWI 约 1.4%；不继续堆 dense basis loss。
- **flow 决策**：没有条件多模态/K-sample oracle 证据，暂不训练 conditional flow。
- **continuous query**：Ocean 全局 interleaved receiver holdout 为 0.007414 vs scalar+cubic 0.008275，
  改善 10.4%；Deepwave 为 1.3874 vs 1.3878，且 true-response interpolation oracle 已达 1.3868，归因为
  receiver undersampling/aliasing，不继续堆 coordinate MLP。
- **Deepwave sampling curve**：相同 systems/sources/frequencies 下，32/64/88 observed ring points 的 oracle
  为 1.387/0.476/0.407，operator 为 1.387/0.479/0.410；96-grid 离散圆在 256 nominal receivers 时发生
  cell duplication，因此冻结该 continuous 分支。
- **prompt calibration**：leave-one-prompt-source-out 二选一不读取 target；Ocean K=4 OOD 从 fixed prior
  1.032 降至 0.846，相对再改善 18.0%，IID 轻微回退 5.3%；K=8 OOD 仍退化 11.2%，只作 robustness
  extension。
- **继续条件**：prompt-only OOD calibration；Deepwave 只有在更密 receiver-grid sampling oracle 合格后才
  重启 neural functional decoder。
- **证据**：[M0 报告](../reports/WAVE_AS_PROMPT_LATENT_PRIOR_M0_ZH.md)；机器汇总
  `artifacts/wave_as_prompt_pfto/m0_latent_prior_multiseed/summary.json`。

### E18：PBNO strict joint source--frequency query

- **新任务**：Q3 的 target source 与 target frequency 均不得出现在 prompts；排除同频 source completion
  与同 source 频率插值两条捷径。
- **M0 强正结果**：global complex Bayesian operator 在 OpenFWI/Deepwave/Ocean 的 K=1→8 分别从
  0.194→0.149、0.126→0.055、0.792→0.401；三域均随 budget 改善。
- **M1 方法**：global SVD response functions + analytic Bayesian prompt posterior + zero-initialized
  query-functional residual + whitened nonlinear latent transport；343,809 参数，一个 checkpoint 支持
  K=1/2/4/8。
- **三 seed 正结果**：相对 validation-selected retrieval/global baseline，OpenFWI K=1/2/4 改善
  11.7%/9.9%/5.3%，Deepwave 改善 7.1%/7.0%/5.1%；paired bootstrap 下界均为正。
- **决定性边界**：OpenFWI K=8 退化 3.0%；Ocean neural residual 无增益；因此没有通过“两个域均超过
  10%”方法 gate。
- **层级发现**：global state 负责 Q3；frequency-local state 在 Deepwave/Ocean Q2 更强。后续应采用由
  query mask 决定的 hierarchical router，不强迫一个 latent head 兼顾所有 regimes。
- **不要重复**：加深 coordinate MLP、把 posterior covariance diagonal 直接拼进 latent MLP、在 Ocean
  IID 上继续加容量。两者已经完成，均弱于简单 latent-mean transport或线性初始化。
- **OOD**：Deepwave K=8 neural transport 改善 4.85%；Ocean fixed prior 会灾难性失效。prompt latent
  energy 能检测 severe shift，但当前 abstention threshold 是开发期 POC，不作为正式 OOD claim。
- **flow 决策**：仍无 conditional multimodality 证据；不得为了生成式标签启动 flow。
- **M2 负结果**：普通 prompt-set updater 参数翻倍但没有新增益；Deepwave coordinate-RMS head 弱于 raw
  head；receiver-preserving residual 仅在 OpenFWI 单 seed 达到 K=1/2 的 12.6%/10.6%，Deepwave 因
  response-scale 失配不稳定并回退到初始化。
- **唯一建议下一步**：先建立跨域一致的 complex likelihood/whitening，再重启 receiver-local residual；
  在此之前不再堆 set encoder。OpenFWI K=2/4 与 Deepwave K=1/2 至少两域过 10%，且 OpenFWI K=8
  不退化，才 promotion。
- **证据**：[PBNO M1 报告](../reports/PBNO_JOINT_QUERY_M1_ZH.md)；机器主表
  `artifacts/pbno_m1_summary/main_table.csv`。

### E19：PBNO M2 efficiency 与 neural boundary

- **尺度修复**：receiver residual 改为仅由 train systems 估计的 coordinate-RMS 单位；零初始化在任意
  数据量纲下严格退化为 Bayesian baseline，Deepwave 数值爆炸已消失。
- **最终 receiver 结果**：偏向 K=1/2 训练、residual bound 0.20 后，Deepwave K=1/2/4/8 为
  0.11665/0.09079/0.06914/0.05433；K=1 未过预注册 0.1138，停止继续扫网络与 bound。
- **nonlinear tail NO-GO**：即使输入来自完整目标 table 的 oracle rank-128 coefficients，MLP 预测
  rank-129--256 tail 后，Deepwave/OpenFWI validation 与 test 都比 rank-128 reconstruction 更差；不启动
  episodic principal-manifold decoder。
- **latent memory 正信号**：在 posterior latent 空间做 context-only retrieval，OpenFWI Q3 K=1/2 为
  0.15290/0.14914，优于原 PBNO；Deepwave 只改善约 2% 或无改善，因此是单域机制结果，不 promotion。
- **Q2 efficiency A 级正结果**：固定配置、三 prompt-order seeds 下，OpenFWI rank 96 为 0.12880，
  相对 rank 128 的 0.12922 略优，同时少 25% storage、少 57.8% rank-cubic solve；Deepwave rank 112
  为 0.04394，相对 0.04349 退化 1.03%，rank-cubic solve 少 33.0%，OOD 0.07527 还略优于 0.07552。
- **边界**：A100 端到端 batched latency 只降低约 12%，因为 gather、decode 与 metric reduction 不随 rank
  立方下降；论文必须同时报告 FLOP proxy 与实测 latency。
- **是否继续**：Q2 保底 gate 已通过；Q3 仍未达到两个域 10%。下一次方法升级只允许改变 latent belief
  本身，并先通过 conditional branch / mixture evidence，不再堆 receiver encoder。后续 best-of-8 审计已经
  完成：OpenFWI 相对 strongest deterministic 的 gap 为 7.5%--8.1%，Deepwave 小于 1%，均未过 15%，
  所以 conditional flow 保持关闭。
- **证据**：[PBNO M2 报告](../reports/PBNO_M2_EFFICIENCY_AND_BOUNDARIES_ZH.md)；机器结果
  `artifacts/pbno_q2_efficiency_formal/summary/summary.json`。

### E20：PBNO M3 posterior-kernel operator 与 stable posterior

- **单一假设**：单高斯 global prior 没有利用训练 response family 的局部 operator prototypes；用相同 prompt
  mask 得到的 posterior tokens 做经验贝叶斯核注意力，可以在不读取 target truth 的情况下改进 strict Q3。
- **协议**：OpenFWI/Deepwave/Ocean 原 system splits；$K=1,2,4,8$；所有 kernel dimension、neighbor count、
  temperature 与 blend 只由 validation systems 选择；三组 test prompt-order seeds；主指标为逐 system 等权
  raw complex RelL2。
- **数值修复**：Ocean rank-128 latent system 在 complex64 下随 event batching 改变结果；统一改为
  complex128 solve 并增加 batching-invariance test。旧 Ocean Q3 数字不得继续用于比较。
- **主结果**：K=2 的 OpenFWI/Ocean 为 0.14707/0.48723，相对 strongest validation-selected baseline
  0.17015/0.60843 改善 13.56%/19.92%，system win rate 93.6%/77.3%，bootstrap 下界均大于 10%。
  Deepwave 由 neural latent transport 得到 0.09077 vs 0.09744，改善 6.85%。
- **OOD**：belief transport 在 Deepwave K=8 三 seed 为 0.09079 vs 0.09664，改善 6.05%、128/128 systems
  同向；Ocean gated kernel 各预算改善 1.3%--4.4%，没有退化。预注册 OOD gate 通过。
- **强神经基线**：293,250 参数 generic CNP 在 OpenFWI 从 checkpoint 继续训练、累计约 20k steps 后 K=2
  仍为 0.23247；Ocean 6k-step 为 0.9826。主分母仍使用更强的 global/retrieval baseline。
- **效率**：vectorized A100 latency 为 11.85--25.53 ms/完整 test population，约为线性 head 的
  3.13--3.96 倍；response bank 3.38--20 MiB。逐 event prototype 慢 48--91 倍，已淘汰。
- **负结果**：用完整训练 table 预计算 full-latent keys 没有在任何域/预算被 validation 选中；prompt-mask
  posterior shrinkage 不能忽略。Deepwave posterior kernel 只有约 0%--2.4%，仍需 neural transport。
- **兼容性**：Q2 efficiency 由 local head 保留；Ocean continuous receiver query 由 query-mask router 精确
  调用冻结 functional head，保持 0.007414，不能宣称 memory branch 原生连续插值。
- **判断**：strict-Q3 两域 10% gate、Q2 efficiency gate、OOD gate 均已通过，成为论文候选；flow gate
  仍失败。下一步只优先做 memory compression、统一 prior adapter 或真实测量域，不重启普通 encoder 堆叠。
- **证据**：[PBNO M3 报告](../reports/PBNO_M3_POSTERIOR_KERNEL_OPERATOR_ZH.md)；机器结果
  `artifacts/pbno_m3_summary/` 与 `artifacts/pbno_posterior_kernel_formal_v3/summary.json`。

### E21：PBNO M4 memory compression、structural waves 与统一 prior adapter

- **压缩假设**：局部 empirical prior 需要保留训练实例密度，但不一定需要保存每个实例的完整 response
  table。
- **prototype 负结果**：random/FPS/k-means 直接减半 instances 时，OpenFWI 退化约 3%--4%，Ocean 约
  5%--14%；不要继续只扫 prototype selection。
- **spectral dictionary 正结果**：保留所有 instances，把每个 value 存为 shared response basis 上的前 16 个
  complex coefficients；key 仍按实际 prompt mask 重算 posterior。OpenFWI/Ocean value-bank storage 为
  原始 0.65%/0.93%，四预算 test 退化不超过 1.48%/1.19%。
- **重要区分**：这不是 M3 已失败的 unconditional full-latent key。失败版本绕过 prompt shrinkage；成功
  版本仅压缩 value，并保留 prompt-conditioned key。
- **延迟边界**：K=2 A100 latency 与 full bank 基本相同；只通过 storage gate，不宣称 latency 改善。
- **第四域**：1024-system heterogeneous damped structural lattice，独立 complex128 frequency-domain direct
  solver。K=1/2 为 0.46261/0.44348 vs global 0.51577/0.48300，改善 10.31%/8.18%，bootstrap 下界
  9.64%/7.51%。
- **OOD 边界**：相对 global 改善 13%--27%，但绝对 error 仍约 0.87--0.98；不能写成强结构 OOD 已解决。
- **hard router NO-GO**：hard LOO 在 Ocean 把 prompt reconstruction 与 Q3 extrapolation 混淆；1,443 参数
  shared risk head 三 seed 后，Deepwave K=2 仍比最佳分支差约 2.1%--2.4%。不要继续扫 selection threshold。
- **连续 adapter 正结果**：在 validation cross-query episodes 上学习两个复 residual 的连续混合，而不是预测
  hard branch label。单个 1,665 参数网络不读 domain ID/test truth；三组独立 prompt seeds 下，三个域和
  K=1/2/4/8 的 12/12 设置均优于最佳单分支。OpenFWI/Deepwave/Ocean K=2 相对 global baseline 改善
  17.25%/6.23%/21.85%。这是已有 calibration domains 上的统一部署，不是 unseen-domain zero-shot 证据。
- **噪声**：OpenFWI/structural 在 20/10 dB 下相对 global 改善 9%--21%；Ocean gate 安全回退约 1.0，
  说明需要 noise-aware complex likelihood。
- **证据**：[M4 正式报告](../reports/PBNO_M4_RISK_RESOLUTION_ZH.md)；正文图与主表
  `artifacts/pbno_m4_summary/`。

### E22：Posterior operator composer M0

- **假设**：用解析 posterior 初始化 latent tokens，让它们读取 rank-16 operator dictionary，并由
  source--frequency query 与 target-free risk features 条件化，可以把 M4 的多个分支统一，并进一步学习
  mode-wise composition。
- **严格特例**：learned gate/residual 零初始化；step 0 精确恢复现有连续 adapter，而非随机近似。
- **规模与协议**：121,603 参数；OpenFWI/Deepwave/Ocean；K=1/2/4/8；calibration systems 按 75/25 拟合与
  selection；正式 test 为完全未见 systems 和三组 prompt seeds。
- **统一正结果**：共享模型不读 domain ID，正式 K=2 baseline 为 Deepwave 0.09220、OpenFWI 0.14668、
  Ocean 0.48116；解析 prior、dictionary、neural proposal 与 risk adapter 已进入同一可执行计算图。
- **关键负结果**：risk features 进入 query token 后，selection 上 Ocean K=2/K=4 改善 8.2%/8.8%，但
  unseen test 退化 5.2%/7.1%；Ocean 全预算 bootstrap 区间均为负。OpenFWI/Deepwave 基本持平。
- **效率负结果**：A100、OpenFWI K=2、500 systems 整批 latent update 为 6.383 ms，analytic kernel 为
  0.607 ms，即 10.51 倍；1.5 倍 latency gate 未通过。
- **归因**：小 calibration selection 允许模型记住 system residual pattern；现有 target-free 证据不足以判断
  correction 在新 system 上的符号。三域平衡、analytic anchor 与 adapter initialization 都不能修复。
- **决定**：冻结 zero-update composer；learned token correction 为 NO-GO；M1 continuous decoder 与 flow
  不启动。不得再按 calibration selection 数字声称 Transformer 增益。
- **重启条件**：更多独立 calibration systems、nested system cross-fitting、新真实 multistatic 域，或
  mode-wise correction 的 target-free oracle headroom 超过 10%。
- **证据**：[M0 报告](../reports/PBNO_POSTERIOR_COMPOSER_M0_ZH.md)；冻结机器结果与图在
  `artifacts/pbno_posterior_composer_m0_summary/`。

### E23：Heterogeneous functional operator M0

- **单一假设**：coordinate-aware complex set encoder 与 nonlinear neural field 可以替换固定 PCA chart，
  同时保留 arbitrary receiver masks 下的 posterior evidence accumulation。
- **数据与 split**：OpenFWI 500/500/500、Deepwave 512/128/256、Ocean 256/64/128；physical-system split；
  strict-Q3 K=2；每个 prompt 独立 10%/25%/50%/100% receiver masks。
- **matched baseline**：M4 rank-16 posterior memory / spectral dictionary；所有 residual 模型 zero-update
  时回到该基线。
- **oracle**：10% receivers 的 compressed full-dictionary headroom 为 OpenFWI 16.2%、Deepwave 21.0%、
  Ocean 49.3%；Ocean 在 full receivers 仍为 37.1%。oracle 使用 target truth，只是 existence result。
- **functional evidence**：64,226 参数；OpenFWI 约 +0.1%，Deepwave +1.2%--2.4%，Ocean 选择 step 0。
- **reranking**：127,522 参数 end-to-end residual reranker 最多约 1%--2%；learned mix 为负。改成直接
  listwise response-metric 且扩到 16 个 train layouts 后，OpenFWI test +0.6%--1.2%，Deepwave 回退，
  Ocean 仅最稀疏 setting +0.43%。
- **pairwise metric 负结果**：Ocean 10% receivers 的 analytic/raw-L2/coherence/amplitude candidate error 为
  0.698/0.804/0.825/0.911，target oracle 为 0.360；普通 prompt similarity 无法预测 target-query similarity。
- **neural-field 分解**：212,354 参数 nonlinear decoder 的 train full-table reconstruction 为 0.229；unseen
  full-response latent fit 为 0.368，但 K=2 sparse optimization 为 1.09--1.27，328,736 参数 amortized encoder
  test 约 1.035。capacity 成立不代表 sparse identifiability。
- **结论作用域**：NO-GO 只针对当前三域、strict-Q3 K=2 和无约束 nonlinear latent；不否定有显式物理
  observation model、更多信息 prompts 或真实新域下的 functional representation。
- **决定**：不继续扫 width/depth/heads/loss/latent dimension；保留线性 posterior chart 作为可观测坐标，
  learning 只放在 operator prior 与受控 chart transport。
- **重启条件**：更强 conditional prompts、物理可辨识 latent、新真实 multistatic 域，或不读 target truth 的
  target-free oracle 相对 M4 仍有至少 15% headroom。
- **证据**：[正式报告](../reports/HETEROGENEOUS_FUNCTIONAL_OPERATOR_M0_ZH.md)；冻结汇总与主图在
  `artifacts/heterogeneous_functional_operator_m0_summary/`。

## 4. 值得从正反馈继续挖的方向

以下按“与当前主线的连接强度 / 最小验证成本”排序。它们不是要求同时实现的组件清单。

### P0：多分支相空间 wave tokens（首轮已完成）

**当前状态。** wavelength-scaled oracle 已通过，fixed multiscale Gram 的三 seed downstream gate 未过。
见 E14。本条不再是可以原样执行的计划。

**若重启。** 只允许加入 learnable wavelength-constrained analysis、显式 sparse routing 或直接的
multipath query；不再枚举 fixed filter channels。

**进入下一阶段的门槛不变。** 在一个预先指定困难条件相对当前 token 降低至少 15% paired system
RelL2，另一二维域和 IID 不退化超过 2%。

**完整协议。** [多分支相空间波表征](MULTIBRANCH_PHASE_SPACE_REPRESENTATION_ZH.md)。

### P1：统一 spatial 与 spectral phase advance

**为什么现在值得做。** 当前最强 token 使用同频空间邻点的复相关；旧 Stage-1 最强机制结果来自同点
相邻频率的复相关。两者本质上都在估计局部 complex phase transport，但过去被分在两条路线中。

**最小假设。** 对同一 source 的邻近频率 views，增加 spectral phase advance 与 amplitude ratio，可以
帮助 set encoder 区分 local wavevector、delay 和 boundary-induced multipath；尤其可能改善 closed
boundary 与低 source budget，而不要求解析 source location。

**最小 POC。** 先完成 shared set encoder；随后只增加 spectral increment channels，保持 backbone、
参数量、views 和训练预算一致。比较 amplitude-only、spatial-only、spectral-only、spatial+spectral。

**成功门槛。** 不以已饱和的 retrieval 为唯一指标；至少在 coefficient recovery、20 dB、closed
boundary 三项中的两项稳定优于 spatial-only，并在可变 frequency budget 下方向一致。

**为何不是重走旧路。** 旧 carrier 直接把单一 slope 外推成目标相位；这里把 phase increment 当作
system-identification evidence，由 set encoder 聚合，不强制单路径 continuation law。

### P2：学习物理测度，而不只是学习 feature

**为什么值得做。** reciprocity metric 从 0.84 到 `2.1e-6` 是本项目最干净、但尚未进入模型的物理
诊断之一；而且权重可仅由低频 response 自监督恢复。

**最小假设。** 让 encoder 先估计 response space 的正测度，再在该测度下做 view normalization、
reciprocity consistency 或 attention，可能提高跨 source layout 和跨 resolution 的泛化。

**最小 POC。** 不加 PDE loss；比较 Euclidean aggregation、oracle metric 和 learned metric 三者，先看
cross-layout retrieval 与 medium recovery。learned metric 必须只读 context views。

**成功门槛。** learned metric 接近 oracle，并在新 source layout 或 boundary 上产生至少 10% paired
error reduction；若只让 reciprocity residual 更小但 downstream 不变，则作为科学诊断保留，不进入主方法。

### P3：用 system latent 重新审视 local-span oracle gap

**为什么过去不该继续，现在可能不同。** 旧 free local-span predictor 失败，是因为低频条件没有提供足够
system identity；当前 token 却能以 0.0504 恢复 hidden medium。这个新 latent 可能提供当时缺失的全局
条件，使局部系数不再完全自由。

**前置 gate。** 必须先让 true-medium oracle decoder 低于 0.30。若真实介质都不能驱动合格 decoder，
learned latent 不值得接入。

**最小 POC。** 不预测 1408 个自由复系数。让 system latent 只调制低秩、多分支或局部 propagation
decoder，并与 true-medium、predicted-medium、unconstrained latent 三种条件 matched 比较。

**成功门槛。** predicted-system latent 相对 raw-context decoder 至少改善 30%，且能把 oracle-medium 与
learned-medium 的 gap 解释清楚。

### P4：把 Reliable-horizon 变成 representation confidence probe

**为什么值得保留。** 空间 disagreement 的 AUROC 在两个域和多个尺度上都很强，说明模型能识别局部
不可置信区域。这可以回答 phase token 在低 amplitude、多路径或 OOD boundary 下何时可靠。

**最小 POC。** 对 set encoder 做小 ensemble 或轻量多头，预测 medium error / query error 的 patch map；
比较 amplitude gate、token coherence 与 learned disagreement。

**限制。** 它是安全性与诊断支线，不应代替降低 field error 的主任务，也不应重新进入 calibration policy。

### P5：将独立多保真改成 tracking，而不是 open-loop extrapolation

**保留理由。** 真实 LF/HF gap 在第一步 107 Hz 有 45.88% oracle 改善，但六步 macro 失败。这是很清楚
的 short-memory system tracking 信号。

**为什么暂缓。** 它偏向部署与 sequential sensing，不如 wave-system representation 通用；当前不应与
主线混做。

**重启方式。** 使用连续 fidelity knob 和 receding-horizon observation，先比较固定成本 tracking oracle；
只有跨至少 30 个系统通过后才训练 representation model。

### P6：解析 Prony 只作为 identifiability unit test

合成三路径的 0.043 证明结构正确时 compact spectral state 可以非常准。它适合成为模型必须通过的
unit test：恢复阶数、跨分辨率与 phase equivariance。由于真实场逐像素 Prony 病态，它不适合作为论文
主数据或现实 baseline 的替代。

## 5. 推荐的研究优先级

### 立即推进

1. 冻结 E16 的三个多系统域 split、episodes 与 test，不再按 test 调参；
2. 补齐 retrieval、direct complex ridge 和参数匹配 generic query network，重新定义 strongest baseline；
3. 在共同的归一化 probe 预算上报告 raw error、strong-baseline gain、gap closure 与 budget AUC；
4. 只有至少两个域仍保留 10% 以上 strong-baseline headroom，才升级 query representation；
5. Q3 的 reciprocity routing + per-link spectral state 放在 Q2 强基线审计之后；
6. best-next-source oracle 已完成且未过 gate；active 只有扩展 action space 后才重启。

完整指标定义与 promotion gate 见
[跨数据集指标提案](CROSS_DOMAIN_QUERY_METRICS_NEXT_ZH.md)。

当前方法升级冻结为 PFTO：先检查 functional tensor rank，再做 episodic differentiable latent fitting，最后才判断
是否需要 amortized encoder 或 conditional latent flow。详细 M0--M4 与 NO-GO 条件见
[Wave as Prompt 顶会计划](WAVE_AS_PROMPT_TOPCONFERENCE_PLAN_ZH.md)。

### 小成本并行诊断

- P2 learned physical metric 的 downstream usefulness；
- 把现有 reliable-horizon feature 作为 representation confidence 的只读 probe。

### 等前置条件后再做

- generative belief、active acquisition 与 variable receiver query；
- compact multibranch tokens，仅在 Q3/Q4 multipath failure 明确后；
- 第二 PDE / 外部数据域；
- 部分边界阵列与真实 source phase/gain mismatch。

### 暂停

- free local coefficients、generic masked Transformer、更多 pole/ODE/hypernetwork；
- 当前 AR action 上的 sensor-mask / RL 搜索；
- 没有重复条件证据的 flow/diffusion；
- 为了数字好看缩短 horizon、换主指标或只报告 amplitude。

## 6. 新实验完成后的更新模板

每次正式实验在本文件追加：

```text
ID / 日期 / commit:
单一假设:
数据身份与 system split:
允许读取的信息:
matched baseline:
预先 gate:
主结果与 paired uncertainty:
最强负对照:
结论作用域:
失败归因（信息 / 表示 / 优化 / 数据 / 指标）:
是否继续:
若停止，什么新条件下才可重启:
summary / figure / checkpoint hash:
```

没有填写“结论作用域”和“重启条件”的实验，不视为完成归档。

## E20 / 2026-09-27 / Uncertainty-Conditioned Proximal Operator Prior M0

**单一假设。** 保留可观测 Bayesian posterior chart，让小型复数网络只学习 operator prior，并将 correction
经过 posterior precision 闭式投影，能否在不破坏 prompt consistency 的前提下相对 M4 再改善至少 5%。

**数据与协议。** OpenFWI、Deepwave、Ocean；完整 system split；strict-Q3 K=2；每个 prompt 独立
10%/25%/50%/100% receiver masks；三个 formal prompt seeds；主指标为逐系统 raw complex RelL2。

**方法与 matched baseline。** 29,761 参数 bias-free complex network + radial activation + uncertainty gate；
解析 precision projection与闭式 prompt safeguard；zero correction / zero blend 精确回退完整 receivers 下的 M4
adapter 或 masked setting 下的 rank-16 posterior memory。

**主结果。** 最佳为 Deepwave 25% receivers：0.11035 到 0.10712，改善 2.93%，bootstrap 95% 区间
[2.63%, 3.23%]，系统胜率 89.5%。OpenFWI 约 0.14%，Ocean 最高 0.55%；完整 receivers 下三域低于
0.2%。12 个正式点没有退化，但没有一个达到 5%。

**负对照。** 独立 learned prior predictor 三域均差于 M4；single-domain chart 不能把增益提高到 5%；latent
denoising pretraining 把 Deepwave 10% 从 2.45% 降到 2.13%；proximal multiplier 0.1/1/10 不改变结论。

**归因。** 数据一致性可闭式保证，但 unseen-query residual 的主要部分在稀疏 prompts 下仍不可辨识；Ocean
49% target-truth candidate oracle 不是一个平滑、target-free operator-prior gap。Deepwave 中等 receiver
budget 的稳定 2%--3% 是机制级正信号，不是主方法贡献。

**决策。** M0 NO-GO；M1 query decomposition 不启动。禁止继续扫网络容量、denoising noise、steps 或
proximal strength。

**重启条件。** 新信息条件改变 conditional identifiability；target-free correction-sign oracle 在 held-out
systems 超过 10%；真实跨设备数据证明 fixed chart 失效；或出现可跨域对齐的 operator coordinates。

**Artifacts。** `reports/PBNO_UNCERTAINTY_PROXIMAL_PRIOR_M0_ZH.md`；
`artifacts/pbno_uncertainty_proximal_prior_m0_summary/summary.json`、`m0_mask_results.csv`、主图 PNG/PDF；
配置 `configs/pbno_uncertainty_proximal_prior_m0_20260927.json`。

## E21 / 2026-09-27 / Target-Aware Wave Interrogation M0

**单一假设。** 固定 sparse prompts 下的主要瓶颈已经是 conditional identifiability；把 source 与 frequency
作为 inference-time actions，并用 learned wave-system operator belief 选择对指定 target response 最有信息的
probe，能否在不增加测量预算的情况下显著降低 strict-Q3 error。

**数据与协议。** OpenFWI、Deepwave、Ocean；原 system split；strict-Q3；相同第一条随机 probe；后续
source--frequency action 不得包含 target source 或 target frequency；nested K=1/2/4/8；三个 prompt seeds；
逐 physical system 等权 raw complex RelL2、paired bootstrap、win rate 与 log2-budget AUC。

**方法与 matched baseline。** PBNO complex response basis、Bayesian posterior、rank-16 dictionary、neural
transport 和 continuous adapter 均冻结。planner 最小化 target rows 的 posterior predictive variance；random
schedule 使用相同 K 和同一第一条 probe。另审计 train-population fixed schedule、target-truth per-system oracle、
query-conditioned positive posterior metric、bounded-residual neural policy 与 noisy likelihood calibration。

**主结果。** K=2 random 到 target-aware：Deepwave 0.09220 到 0.06715，改善 27.17%，95% 区间
[26.64%, 27.70%]；Ocean 0.48116 到 0.38365，改善 20.27%，区间 [18.77%, 21.74%]；OpenFWI
0.14668 到 0.14289，改善 2.58%，区间 [2.21%, 3.01%]。三域系统胜率为 100%/100%/85.2%。
K=1/2/4/8 AUC 分别改善 21.14%/12.35%/3.52%。Structural Lattice K=2 改善 7.60%，只作跨物理机制证据。

**最强负对照。** train-population fixed schedule 比解析 planner 仅好 0.06%/0.24%/0.56%；其与
per-system target-truth oracle 的剩余差距为 9.3%/14.1%/10.7%，均未过 15% adaptivity gate。learned policy
在 Deepwave 回退，Ocean 比解析 planner 只额外改善 0.37%；learned metric 最高改善 2.55%。

**归因与结论作用域。** 强结果来自 target-conditioned experiment design，而非 response-conditioned active
reasoning。learning 的可靠角色是形成可查询的 complex operator belief；解析 uncertainty 把 representation
与 physical action 接成闭环。当前可 claim target-aware sequential design，不能 claim adaptive agent、RL
优势或 classical variance criterion 本身的新颖性。

**决策。** POC 通过，成为当前主线。停止扩大 metric/policy；M1 必须先做 untouched systems、uniform/
nearest/D-opt/population acquisition baselines 和真实成本轴。唯一优先 learning 升级为参数少于 500k 的
continuous acquisition utility operator。

**重启 response-adaptive policy 的条件。** 新设定下 population schedule 与 per-system oracle 的差距至少
15%，且 action utility 确实随已观测 response 改变；例如 nonlinear/multimodal belief、未知 sensor noise、
partial failures 或 task-level inverse ambiguity。

**Artifacts。** `reports/ACTIVE_WAVE_INTERROGATION_M0_ZH.md`；
`artifacts/sequential_active_wave_query_m0_summary/summary.json`、`sequential_results.csv`、主图 PNG/PDF；
配置 `configs/sequential_active_wave_query_m0_20260927.json`；下一阶段冻结计划
`docs/ACTIVE_WAVE_INTERROGATION_NEXT_PLAN_ZH.md`。

## E22 / 2026-09-28 / Untouched Interrogation + Continuous Utility M1

**单一假设。** M0 target-aware effect 能否在完全新 systems 上保持；一个小型 learning-based set utility
能否学习 posterior query geometry，并在不读取 test truth 的条件下近似 exhaustive acquisition。

**数据与协议。** 新生成 Deepwave/Ocean 各 256 IID + 128 OOD systems，seed base 20261027；冻结 M0
PBNO、adapter、query rule 后一次性打开。相同首 probe、strict-Q3、nested K=1/2/4/8、三 seeds。强基线为
random、nearest、farthest、D-opt、exact target risk、validation-only population schedule 和 K=2 truth oracle。

**主结果。** Untouched IID K=2 exact 相对 random：Deepwave 0.09353→0.06802（27.27%），Ocean
0.50662→0.40346（20.36%）；AUC 改善 21.15%/12.47%。Nearest 是 Deepwave 强 heuristic；exact 相对
最强 geometry heuristic 的 AUC 只再改善 3.28%/4.84%。Population 与 exact 基本等价。

**Learning 结果。** 160,801 参数 continuous set utility 在未见 source--frequency pairs 上 held-out regret
0.71%/1.12%。纯 utility AUC gap 2.006%/0.271%，Deepwave 擦线失败；top-4 learned proposal + analytic
certification 将 gap 降至 0.29%/0.16%，两域通过 2%。

**关键负结果。** 整块留出 source locations 与 frequency coordinates 后 regret 为 10.23%/18.43%；5×
speed 未过，hybrid 也不比 batched exact 快。Ocean OOD absolute error 数百至数千，K=2 改善区间跨零。
Deepwave OOD 仍改善 23%--28%。Population-to-truth-oracle gap 为 12.48%/15.10%，没有跨域 policy gate。

**归因与结论作用域。** learned utility 能组合泛化未见 action pairs，并能为解析 belief 提供近无损 proposal；
它尚未学出任意连续 source/frequency geometry。Acquisition 改善 belief 内的信息效率，不能修复 OOD belief
失配。

**决策。** M1 POC 通过，接口冻结为 learned PBNO belief + continuous proposal + analytic top-4
certification。停止 width/depth/shortlist sweeps。下一次重启只允许更密连续 physics teacher、真实 multistatic
data，或 Ocean nonlinear-belief oracle-first 设定。

**Artifacts。** `reports/ACTIVE_WAVE_INTERROGATION_M1_ZH.md`；
`artifacts/active_wave_interrogation_m1_summary/summary.json`、`paper_table.csv`、主图 PNG/PDF；配置
`configs/acquisition_baselines_m1_untouched_20260928.json` 与
`configs/continuous_acquisition_utility_m2_coordinate_holdout_20260928.json`。

## E23 / 2026-10-04 / 连续复波场相位表示：相关谱介质容量与固定代码频率查询

**为什么做。** 原一向波导介质只含3个二维Gaussian/15个隐藏参数，且知道精确合成族的经典多初值反演可在旧8系统800Hz做到raw约0.00026，比128维PCA代码更准。原漂亮的“紧凑相位表示”不足以外推到复杂介质。

**固定数据与方法。** 相同单向PDE，介质改为16×16相关余弦谱/RMS15m/s；训练512介质标签，首32系统检验原128维，另32全新系统20265000–031在真值生成前冻结rank128/256直接先验+相同60/100/140Hz宽源×65完整剖面、5+5次GN、400/800Hz宽/未见2m窄源、完整ROI未配准raw/phase门槛。新32数值审计最坏5.93e-5。不是第二物理域，也不是真实Ocean。

**主正结果。** 原128维在新32的800Hz宽/窄raw0.51131/0.49965；rank256仅由低频context推断降至0.01475/0.01532、相位RMSE0.01123/0.01170rad，双源32/32受益，配对改善97.11%/96.93%，最差系统raw0.02263/0.02439，全部事前gate通过。固定前8新系统、**确认过门槛后才执行**的200–1000Hz查询在1000Hz宽/窄raw0.02081/0.02016、phase0.01615/0.01588rad，预登记门槛通过；数值审计7.22e-5。该子集不是又一批独立测试。

**主负结果与归因。** 原128维训练介质方差解释率99.8637%，真测试介质的欧氏PCA投影800Hz仍raw0.3906/0.3859；rank256真介质投影降至0.00809/0.00797。少数背景的局部响应Jacobian选128维基失败（0.3796/0.3846），一阶解析余弦先验在旧8开发上也失败（0.2058/0.2276），但精确非线性已知谱族逆解**尚未测试**。旧8同谱形状RMS15→8.1689m/s，原128维raw0.53366/0.54360→0.30562/0.30982：强度解释部分恶化，但不单独解释失败。当前256维接近合成器255个随机谱系数，非高度压缩latent；1+1次GN旧8已到raw约0.027/0.028，不能仅以摊销加速包装一个未证实必要的网络。统计PCA先验+经典物理反演是已验证方法，深神经/跨PDE/真实海洋增量未通过gate。

**复用与禁忌。** 读`docs/SPECTRAL_MEDIUM_STRESS_POC_ZH.md`及报告第21节，图`reports/figures/spectral_rank256_confirm.png`与`spectral_frequency_curve_confirm.png`。原始谱介质64+64+16+40份NPZ分别在本地同名数据目录；训练标签、每系统代码、结果与预真值manifest在`artifacts/coherent_waveguide_spectral_*_20261004/`，大NPZ Git ignored。不要把原32后选rank256的旧8开发写成独立，不能把确认前的128失败和确认后的256成功表述为同容量神经优势；不能把1000Hz同8系统再算新的独立集合。阶段十post-run inventory记录哈希，但不是预冻结证据。

**后补方法边界（同日，非原独立候选）。** 旧8开发中，65→17条干净context的800Hz宽/窄raw仍0.0181/0.0188；0.1%复模态噪声0.0243/0.0249；1%噪声原方案0.172/0.178。固定其他条件、用本已公开的源解析初态而非带噪首剖面，1%下降至0.0845/0.0889，但最差系统约0.180/0.168；训练均值介质初始化反而约0.49，不要隐去失败。首次稀疏代码因缺观测索引在结果前崩溃，修正版v2结果与失败目录分开。额外知道真16×16余弦速度生成族的**精确非线性**经典反演，在旧8开发800Hz raw0.00750/0.00798；固定后在已用32系统raw0.00942/0.01033、相位0.00703/0.00765，比PCA256均值进一步好36.1%/32.6%，宽/窄分别28/32、25/32胜出。它不读测试系数/训练标签但有真合成族额外信息，不能作同信息排名；说明一阶解析基失败不等于训练PCA必要。见[阶段决策短文](PHASE_REPRESENTATION_DECISION_20261004_ZH.md)。

**E23后补噪声条件复核（同已用32，不增加独立系统数）。** 旧8开发固定的已知源初态改动，在已用32系统新注入的1%复模态噪声下，800Hz宽/窄raw由0.15054/0.16094降至0.08619/0.08805，相位0.06791/0.06901rad；32/32、31/32获益，最差raw0.12740/0.14945，预设四类gate全部通过。其余噪声定义、rank256、5+5GN均不变。它说明噪声先前主要经源初态污染放大，但带噪残差仍约无噪0.015的六倍；不可宣称真实传感器鲁棒。另有[相关尺度变化协议](SPECTRAL_TAPER_SHIFT_POC_ZH.md)用于新的同PDE分布外边界，不是第二域。

**E24新16环境相关尺度分布外负结果。** 在运行前提交配置/代码`15f9aa7`、预先固定taper5训练→taper10测试、RMS15不变、seed20267000–015及四类gate。800Hz宽/窄源部署raw0.4155/0.4350、phase0.3743/0.4041rad，0/16系统raw≤0.10，三个效果gate失败；独立数值审计2.39e-4通过。额外读取真测试介质的欧氏PCA256投影oracle raw0.2925/0.2789，真实32项物理oracle约0.0028。前者不是真实响应最优容量下界，不能下“任意256维表示不可能”结论；[事后响应优化oracle](SPECTRAL_TAPER_RESPONSE_ORACLE_ZH.md)已另预登记，用目标800Hz宽源数据，绝不作为部署成绩。旧同分布32环境仍独立有效，但本负结果使跨介质统计泛化主张失败；不是新物理域。

**E25事后单目标响应oracle。** 在原已看过OOD16系统，额外读取真介质初始化与目标800Hz宽源场的257条完整剖面，用同训练PCA256子空间固定5步GN，宽源raw0.2925→0.07536，未用于目标拟合的窄源0.2789→0.22321。于是欧氏投影不等于最优响应代码，但也没有观察到准确共享源表示。方法额外目标信息极强，不可部署、不可混入E24。为排除窄源未被监督，另冻结[双目标源共同oracle](SPECTRAL_TAPER_JOINT_ORACLE_ZH.md)作为最后一个机制诊断；无论结果如何停止同批系统的oracle搜索。

**E26最后的双目标源oracle和匹配统计重训。** 双目标源泄漏oracle固定5步，800Hz宽/窄raw0.08293/0.08303、phase0.06232/0.06265rad；仍未到0.05，且第5步全部非零，不是全局最优下界。目标taper10新512介质训练标签（与测试16 seed分离）重拟合rank256后，同低频输入部署800Hz raw0.12705/0.13668、phase0.10172/0.10970；比旧taper5先验改善69.43%/68.58%，**未过**事后各≥70%及raw/phase≤0.05门槛。新先验真介质欧氏投影oracle仍0.06728/0.06516，虽训练方差保留99.99675%。两个控制的信息条件不同：双源oracle读取目标真值和真介质，匹配先验训练期读取新分布512介质标签；均不能称原OOD部署成功。按预协议停止本16系统的rank/步数/oracle追逐。详见[共同oracle](SPECTRAL_TAPER_JOINT_ORACLE_ZH.md)与[匹配先验](SPECTRAL_TAPER_MATCHED_PRIOR_ZH.md)。

**E27精确已知函数族的最强结构物理基线（同已用16系统）。** 在额外知道真实16×16余弦声速函数族及taper10、不给测试系数/目标场或训练介质标签时，沿用旧8开发初始化岭1e-7与5+5GN，同低频输入800Hz宽/窄raw**0.003880/0.004129**、phase0.003021/0.003217rad，最差raw0.005807/0.006825；事后预设gate全部通过。400Hz也0.001745/0.001906。删除测试spec中的真实谱系数重复1+1步，估计坐标和beta逐元素差0；正结果不是测试系数泄漏。它的信息条件**比PCA学习先验更强**，不能作公平胜负榜，也不是第二个独立队列；但足以反驳“taper10低频在当前理想方程下根本没有高频信息”。主要困难是未知生成机制下如何获得这样相位保真的、跨介质统计可适配表示。详见[物理上限协议](SPECTRAL_TAPER_EXACT_FAMILY_ZH.md)。

**E28最小学习变量机制（旧16开发）。** `e7768fc`先登记同1024介质网格标签的声速PCA255＋正确`c^{-2}`映射，对照同标签q-PCA255，两者同60/100/140Hz×65密集复剖面及5+5GN。已用taper10的16环境800Hz宽/窄raw0.008592/0.009278、相位0.006615/0.007292rad，q对照0.115636/0.127069，预设开发三gate全过。两源各16/16系统配对改善，但**开发旧环境不是独立测试**；原始声速共享基是q的5倍存储，在线慢约1秒/环境，且rank255几乎等于合成自由度。详见[开发协议与结果](LEARNED_SPEED_MANIFOLD_POC_ZH.md)。

**E29冻结新16确认（同族统计插值，不是第二PDE）。** `5fe88a5`先冻结新taper7.5系统20268000–015、审计20268100–101、模型先验SHA、双初始化强q-PCA、评价ROI和五门槛，再生成真值。新800Hz宽/窄raw**0.005375/0.005052**、phase**0.004197/0.003915rad**，q-PCA raw0.080678/0.081965，改善93.34%/93.84%，各16/16系统raw≤0.05，最差0.01769/0.01624；数值审计最大9.13e-5，全部前置gate通过。真值32NPZ及192评价行和SHA保留。只证明同已知16×16余弦合成族taper5/10训练之间的插值；不证明未知物理族/真实海洋/稀疏带噪/深神经必要性。见[确认记录](LEARNED_SPEED_MANIFOLD_CONFIRM_ZH.md)、[配对图](../reports/figures/spectral_speed_manifold_confirm.png)。

**E30共享基存储审计（不是新增预测候选）。** `8d5cd0c`预登记深度rank16 SVD，只压缩冻结声速基，不读取新目标/重训：先验相对重建2.56e-15，预选旧开发固定代码生成器相对差3.91e-15，均过1e-10门槛。共享float64数值2621440→264704，为q基524288的50.5%；目前仍展开基进行在线物理反演，推断时间**未**改善。这个rank16是合成器16深度余弦的结构反映，不是通用3D波场发现。见[存储审计](LEARNED_SPEED_BASIS_STORAGE_AUDIT_ZH.md)。

**E31第二介质统计族开发：显著增量失败。** `092b224`先登记旧Gaussian训练512与旧开发8、rank128同标签声速/q-PCA、相同低频5+5GN、raw≤.005/phase≤.005rad/相对强q基线双源各≥20%门槛。800Hz声速raw.003219/.003435，q强基线.003415/.003528，仅改善5.74%/2.64%，逐系统各4/8胜；绝对精度门槛通过，**增量门槛失败**。不生成新Gaussian确认、不在这批旧开发系统继续调rank/岭/步数。这个族只含15连续参数、q-PCA已非常准，故不推翻余弦族前置新16正结果，但阻止“声速变量普适大幅领先”故事。它仍是同一个一向PDE。详见[完整负结果](GAUSSIAN_SPEED_VARIABLE_DEV_ZH.md)。

**E32固定代码高频读出诊断：绝对精度正、适配收益负。** `a7363b7`在新1000/1200Hz目标前固定已用taper7.5确认集首8、四种256/512段×128/192模态读出、完整ROI与门槛。1200Hz原256×128双源raw.006741/.006537、phase.005435/.005189rad；512×192 raw.006526/.006527，仅改善3.19%/.15%，预设各≥20%失败；数值审计最坏1.935e-4通过。只说明**该频带这8系统**不靠加密读出解决主要误差；不重新拟合、不提供1200Hz q同频对照，不能计为新独立系统或跨域方法胜利。见[详细记录](SPEED_RESOLUTION_SCALING_ZH.md)。

**E33两统计族真介质投影oracle：机制模式通过，非部署/非容量下界。** `f5e29b5`事前固定只读既有真值的欧氏投影检查。余弦新16的800Hz宽/窄真介质速度投影响应raw.002391/.002404，q投影.077785/.075747；Gaussian旧8速度.002448/.003065，q.002484/.002884。四项机制检查通过，训练q方差率余弦99.997452%/Gaussian99.9999842%，测试q系数相对投影残差.006644/.000469。两族差异支持“非线性变量变换在特定介质族产生相位敏感残差”假说，但两轴都读取测试真介质，欧氏投影不是最佳响应代码的严格下界；不能当部署或公平跨族排行榜。见[完整归因](VARIABLE_PROJECTION_ATTRIBUTION_ZH.md)和[图](../reports/figures/spectral_variable_projection_attribution.png)。
