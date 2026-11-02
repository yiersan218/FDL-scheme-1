# 补全驱动的联邦多视图聚类

当前算法标记为 `completion-v1`。目标是在客户端视图不完整时，通过跨视图潜在补全改善聚类；完整数据是掩码全真的特例。本文描述系统级算法，客户端公式与实现细节见 [模型架构.md](D:/Project/Python/研究生毕设/FDL-scheme-1/模型架构.md)。

实现按职责分开：[multiview.py](D:/Project/Python/研究生毕设/FDL-scheme-1/system/flcore/trainmodel/multiview.py) 定义模型与前向，[completion.py](D:/Project/Python/研究生毕设/FDL-scheme-1/system/flcore/trainmodel/completion.py) 定义补全模块，[losses.py](D:/Project/Python/研究生毕设/FDL-scheme-1/system/flcore/trainmodel/losses.py) 定义 DEC 目标与总损失。客户端和服务端文件名保持不变；本次模块拆分不改变下述数学定义。

## 1. 问题定义与统一数据协议

客户端 \(k\) 持有互不重叠的样本集 \(\mathcal S_k\)，每个样本有 \(V\) 个视图及观测掩码 \(m_{iv}\)。当前实验生成器选取 \(\operatorname{round}(\rho N_k)\) 个样本，每个样本恰好缺一个视图；固定掩码贯穿训练、初始化、压缩和评估。

所有缺失率均在客户端仅用可见行归一化，\(\rho=0\) 也不例外。原始隐藏值不参与统计或监督。当前仍为从完整 MAT 数据模拟整视图缺失的单进程联邦实验，不自动识别 NaN、零值或外部 mask 文件。

## 2. 观测表示与补全预测

每视图独立自编码器产生 \(z_i^{(v)}=f_v(x_i^{(v)})\) 并重构输入。可见编码归一化后，以 \(a_v=\operatorname{softmax}(w)_v\) 在该样本的可见集合内融合：

\[
h_i^{\mathrm{obs}}
=\operatorname{norm}\left(
\frac{\sum_vm_{iv}a_v\operatorname{norm}(z_i^{(v)})}
{\sum_vm_{iv}a_v}\right).
\]

预测目标视图 \(v\) 时，Query 和锚点 Key 均排除 \(v\)，默认 Value 取本地完整锚点的真实目标视图编码；同 ID 锚点不得作为邻居。预测由有界近恒等适配器与受限注意力修正相加得到：

\[
\hat z_i^{(v)}
=\operatorname{norm}\left[
A_vq_{iv}^{\mathrm{src}}
+\eta_{\mathrm{att}}\kappa_tg_{iv}
\operatorname{norm}(\mathrm{MHA}_{iv})\right].
\]

锚点和教师编码停止梯度，空锚点库回退到映射预测；无支持预测不进入完成融合。预测字段始终与观测编码、原始掩码分开。

## 3. 聚类概率与梯度边界

各视图维护独立 Student-t 原型。真实观测概率等权平均为

\[
Q_i^{\mathrm{obs}}=\frac{\sum_vm_{iv}q_i^{(v)}}{\sum_vm_{iv}}.
\]

缺失预测经停止梯度的原型产生 \(q_i^{(v),\mathrm{imp}}\)。用 detached 的视图权重、支持度与有效标记在缺失集合内归一化，得到 \(Q_i^{\mathrm{imp}}\)。正式概率为

\[
Q_i=(1-\bar\beta_i)Q_i^{\mathrm{obs}}
+\bar\beta_iQ_i^{\mathrm{imp}},\qquad
\bar\beta_i=\operatorname{sg}(\beta_i),\quad
0\le\beta_i\le\mathrm{fusion\_weight}\le0.25 .
\]

完整样本、无支持预测和关闭补全概率消融均取 \(\beta_i=0\)。预测分支可更新补全器和源编码器，但不直接通过伪观测梯度更新原型；观测分支正常更新原型。原型聚合的硬、软计数也只来自真实观测。

原 A2 方案仅使用 \(Q^{\mathrm{obs}}\)。新模型保留逐视图中心组织，但正式预测路径已变化，因此结果必须以新版本标记保存。

## 4. 四项统一损失

\[
\mathcal L=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}
+\gamma_t\lambda_{\mathrm{clu}}\mathcal L_{\mathrm{KL}}
+\gamma_t\lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}} .
\]

### 4.1 观测重构

对每个非空观测视图，按观测样本及特征数平均重构平方误差，再等权平均视图。真实缺失值没有重构监督，零占位不作为真值。

### 4.2 留一视图补全 MSE

令 \(\mathcal S_v\) 为目标 \(v\) 真实可见、有其他源视图且预测/教师有效的行，\(\mathcal V_S=\{v:|\mathcal S_v|>0\}\)。正式默认目标为

\[
\mathcal L_{\mathrm{comp}}
=\frac1{|\mathcal V_S|}
\sum_{v\in\mathcal V_S}
\frac1{|\mathcal S_v|d}
\sum_{i\in\mathcal S_v}
\|\hat z_i^{(v)}-\operatorname{sg}(\operatorname{norm}(z_i^{(v)}))\|_2^2 .
\]

完整与缺失数据都使用此定义。教师是当前前向的观测编码 detach；它不同于下面按整轮固定的 DEC 教师。没有监督时返回图内零，不切回旧 MSE。一旦补全已配置，训练损失必须收到补全上下文；无上下文前向仅用于观测教师和真实计数。

`contrastive`、`paired` 是显式损失消融，不能由缺失率暗中切换。显式 `completion.enabled=false` 的观测基线保留旧融合一致性，作为不同方法参与对照。

### 4.3 固定观测教师的 DEC

每轮开始使用客户端全部样本的 \(Q^{\mathrm{obs}}\)：

\[
f_j=\sum_{i\in\mathcal S_k}Q_{ij}^{\mathrm{obs}},\qquad
\hat p_{ij}=\operatorname{sg}\left[
\frac{(Q_{ij}^{\mathrm{obs}})^2/f_j}
{\sum_l(Q_{il}^{\mathrm{obs}})^2/f_l}\right].
\]

本轮所有 batch、epoch 共用 \(\hat P\)，学生使用当前完成概率：

\[
\mathcal L_{\mathrm{KL}}
=\frac1B\sum_{i,j}\hat p_{ij}\log\frac{\hat p_{ij}}{Q_{ij}} .
\]

### 4.4 弱簇均衡

\[
\bar q_j=\frac1B\sum_iQ_{ij},\qquad
\mathcal L_{\mathrm{bal}}=\sum_j\bar q_j\log(K\bar q_j).
\]

它是抑制退化的正则，不保证硬簇等量，不应被解释为各客户端分布均匀的先验事实。正式 `view_semantic=0`，不叠加第五项。

## 5. 两阶段调度

前 `center_init_round` 轮使用重构和补全；完成这些轮次后、下一轮训练前，由所有客户端摘要初始化中心。初始化后的预训练逐步增加 DEC/均衡项：

\[
\gamma_t=
\begin{cases}
0,&t\le T_{\mathrm{init}},\\
(t-T_{\mathrm{init}})/\max(T_{\mathrm{pre}}-T_{\mathrm{init}},1),
&T_{\mathrm{init}}<t\le T_{\mathrm{pre}},\\
1,&t>T_{\mathrm{pre}}.
\end{cases}
\]

第 `pretrain_rounds+1` 轮进入 `clustering`。历史只记录 `pretraining/clustering`，中心初始化不是第三个训练阶段。学习率由初始化前预训练、初始化后预训练、正式聚类三段参数控制；AdamW 每轮重建。

## 6. 初始化和联邦聚合

初始化以本地观测融合或完成融合做共同 KMeans，逐视图中心均值仅取真实观测非零编码，硬计数取真实观测分组。客户端上传中心、计数与覆盖量。

服务端选择覆盖量最大的参考视图，求共同簇排列并应用到全部视图。共享普通参数按参与客户端样本数平均；原型按对齐后的逐视图软计数平均，再应用 `center_momentum`。中心统计不包含预测视图。

`parameter_scope=federated` 时补全参数共享；`local` 时每个客户端跨轮保存自己的参数，服务端公共初始化不参与这些参数的聚合。

## 7. 推理、最佳轮和状态恢复

正式评估由服务端调用 `client.predict(global_model, round_number)`。客户端内部遍历已归一化的数据，使用同一真实 mask，载入全局共享参数及对应 local 私有参数，重新构建本地锚点，以完成概率 \(Q\) 的 argmax 预测。接口返回 CPU 上的 assignments、labels 与 assignment_beta，服务端据此汇总指标；这些返回值属于仿真评估，不改变训练上传协议。

仅正式聚类阶段按 `selection_metric` 选最佳轮。最佳快照包含全局参数、对应轮号以及各客户端 local 私有参数，恢复时必须成套。逐轮历史保留真实末轮；恢复最佳状态后的指标不能冒充真实末轮。

当前输出仍以 history/summary JSON 为主；`output.save_model` 是兼容字段，不意味着生成模型权重文件。

## 8. 压缩与缺失率解耦

压缩是通信选择，不是缺失学习的前提。`none`、Stage＋EF 等应在全部缺失率采用同一设置，比较方法时保持实际字节预算一致。通信实现位于 `system/flcore/communication/`：`codec.py` 负责参数范围、编解码、字节和状态恢复，`compression.py` 负责候选评分、Stage 与 EF。

Stage 的任务损失和分配关系使用当前完成模型；真实观测 mask、固定 DEC 目标、参考锚点与教师在候选间保持一致。local 参数、锚点、ID、EF 残差不上传。编码载荷字节不等于实测网络流量。

## 9. 配置语义

三个顶层 null 占位仍支持。当前默认补全开启、MSE 监督、完成概率开启、federated 作用域；缺失率为 0，压缩为 none。

`loss_weights.completion` 是正式第二项权重；旧 `consistency` 可经加载器迁移。七份当前配置使用 completion=1.0 作为尚未调优的起点，其他超参数也不能继承旧实验的最优性结论。

## 10. 实验记录与证据

当前仅保留 [results](D:/Project/Python/研究生毕设/FDL-scheme-1/results) 中的 175 次实验：七数据集，缺失率 0、0.1、0.3、0.5、0.7，种子 42、2024、2025、2026、2027。方案为 completed，无通信压缩，统一观测初始化。原始记录、计划和源码快照保留在该目录，其他历史性能目录和报告已按用户 2026-11-02 的要求删除。

[统计报告](D:/Project/Python/研究生毕设/FDL-scheme-1/results/statistics.md) 按数据集和缺失率分别给出 ACC、NMI、ARI 的五次均值与样本方差（ddof=1），同一最佳轮与真实末轮分表。该批未包含关闭补全对照，不能单独用来证明补全收益。

统一入口为 `system/run_incomplete.py`，比较 `baseline/observed_completion/completed` 三组，默认只运行 completed。所有率统一客户端观测归一化与通信设置，三组都关闭完成初始化；压缩时使用关闭补全的基线模型导出的相同绝对字节预算。计划、源码、配置不一致时要求新的结果目录，命令见 [README](D:/Project/Python/研究生毕设/FDL-scheme-1/README.md)。

后续配对验证应固定预处理、mask、种子、预算，同时报告最佳轮、真实末轮、稳定性及通信载荷。新结果必须携带 `algorithm_version=completion-v1`，变更实验设置时另选目录，保留本批结果。
