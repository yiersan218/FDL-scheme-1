# 项目上下文：补全驱动的联邦多视图聚类

当前算法版本为 `completion-v1`。用户已于 2026-11-01 授权把缺失视图补全改为正式模型核心，统一完整/缺失预处理与损失，并让受支持补全概率参与聚类。下述规则替代旧版本“A2-only、补全默认关闭、缺失强制 Stage＋EF”的约束。

## 模块职责

- 保留 `clients/clientcluster.py` 与 `servers/servercluster.py` 文件名；客户端负责本地训练和预测，服务端负责调度与聚合。
- `trainmodel/multiview.py` 只组织模型、原型及前向，`completion.py` 保留潜在补全；`target_distribution`、`clustering_objective` 统一放在 `trainmodel/losses.py`。
- 通信代码位于 `flcore/communication/`：`codec.py` 管理共享参数、编解码、字节计量和状态恢复，`compression.py` 管理候选评分、Stage 与 EF，`__init__.py` 为包入口；调用方按职责导入具体模块，不恢复旧 `flcore/compression.py` 单文件。
- 服务端评估调用 `client.predict(global_model, round_number)`，由客户端完成私有参数恢复、锚点与批次处理；不再跨角色调用客户端私有评估辅助方法。
- predict 返回 CPU assignments、labels、assignment_beta，属于单进程仿真评估接口，不新增训练上传字段。校准和上传包装保持客户端内部职责。
- 布局拆分不改变算法版本或默认配置；新计划按实际源码模块清单记录指纹，旧指纹不匹配时使用新目录，不能改写旧计划。

## 当前计算合同

- 客户端以无标签随机索引持有互不重叠、数量尽量均衡的样本，当前为单进程联邦仿真。
- 当前数据集为 ALOI_100、flower17、LandUse_21、Mfeat、NUSWIDE、Scene-15、animal；不恢复 HW/AWA。
- 所有运行先加载未归一化 MAT，再按客户端真实可见行归一化。所有率都构造固定 mask，rate=0 是全真 mask。
- 当前掩码函数保留：选择 round(rate*N_client) 行，每行隐藏恰好一个视图，至少保留一个可见视图。真实 NaN/外部掩码输入尚未接入，不能宣传为已实现。
- 真实标签仅用于指标、最佳轮选择与类别数校验，不进入划分、缺失、损失、补全检索、初始化和聚合。
- 正式补全默认 enabled=true、alignment_mode=mse、use_completed_assignments=true、parameter_scope=federated；local 是明确的参数作用域选择。
- 正式四项为观测重构、留一视图补全 MSE、DEC KL、弱均衡。MSE 先按目标视图有效监督样本与特征维度平均，再平均非空目标视图。
- Query/Key 排除目标视图，同 ID 锚点排除，教师和锚点编码 detach。真实缺失位置无重构或补全真值监督。没有有效监督返回图内零。
- 当前前向观测编码 detach 是补全教师；整轮固定的观测 Qobs 生成 DEC 教师 P。两类教师生命周期不得混写。
- 正式训练、评估均使用补全上下文。enabled=true 而损失缺少上下文须报错，不能静默切换为旧一致性。无上下文 forward 仅供观测教师/统计工具使用。
- contrastive/paired 保留为显式损失消融；enabled=false 是观测基线，允许保留旧融合 MSE，不属于正式补全模型的率依赖分支。

## 原型与概率边界

- 每视图独立 Student-t 原型，默认 alpha=1、distance_scale=1，view_semantic=0。余弦聚类未恢复。
- Qobs 是真实可见视图分配等权平均。Qimp 只来自有支持的缺失预测，使用停止梯度的原型中心。
- Q=(1-beta)*Qobs+beta*Qimp。概率融合支持权重与 beta 均 detach；beta<=fusion_weight<=0.25。无缺失/无支持/关闭use_completed_assignments 时 beta=0。
- 预测支路可更新补全器与源编码器，不直接更新中心。真实观测分配/硬软计数/中心均值保持独立，不能把估计视图算作新增真实观测。
- completed_embedding 的有界融合可用于共同 KMeans 初始化伪标签；这允许间接改变分组，但均值成员仍来自观测编码。
- 所有视图维持相同簇语义排列：共同本地伪标签与服务端同一 Hungarian 排列继续保留。

## 两阶段与状态

- 前 center_init_round 轮只训练观测重构和补全；完成这些轮后、下一轮训练前初始化中心。
- 中心初始化后的 pretraining 逐步增加 DEC 与均衡权重；pretrain_rounds+1 起为 clustering，使用完整权重。
- phase 只有 pretraining/clustering，初始化不是第三个训练阶段。
- 每轮按本客户端全部样本的 Qobs 构建固定 DEC 目标缓存，所有本地 epoch/batch 复用。
- AdamW 每轮重建，普通表示与原型可有独立学习率倍数；不保存 optimizer state。
- local 补全参数在客户端 CPU detached 快照跨轮保存；不上传、不参与 FedAvg、共享 EF 或下行字节统计。
- 锚点库为临时缓存，每个本地 epoch及正式评估重建，不属于模型 state_dict。
- 最佳轮必须成套保存/恢复全局状态、各客户端私有补全状态与评估轮号；仅 clustering 阶段参与最佳轮选择。
- 共享普通参数按样本数 FedAvg，原型按真实软计数加权并可应用 center_momentum。

## 配置与结果保护

- 当前配置统一为 config/*.json，三项 missing/completion/compression 的顶层 null 使用当前默认值。
- missing 默认 enabled=true、rate=0.0；旧 enabled=false 表示有效缺失率0，预处理仍统一。用户未指定缺失率时不擅自设置新率。
- loss_weights.completion 为正式字段，旧 consistency 仅加载/override 迁移兼容；不得允许冲突字段静默覆盖。
- 七份 root 配置 completion=1.0 是未调优起点，不宣称新算法已有最优超参数。
- 压缩独立，默认 none；不得恢复“有缺失必须Stage＋EF”的耦合。
- 新运行标记 algorithm_version=completion-v1，独立输出；不得用当前算法伪装历史算法的复现。
- 用户于 2026-11-02 明确要求删除本次实验以外的历史性能记录，该授权替代此前保留旧档案的规则。当前仅保留 results/：七数据集、五档缺失率、五种子，共 175 次 completed 无压缩实验，统一观测初始化。
- results/ 中原始运行、计划、源码快照、统计与日志作为本批完整记录保留；后续变更配置或源码时使用新目录，不改写本批证据。statistics.md 分别报告最佳轮和真实末轮的均值与样本方差（ddof=1）。
- 历史 A2、missing2 运行与核验脚本及其专用测试已按用户要求移除，不恢复这些历史入口；新矩阵实验使用 system/run_incomplete.py。保留当前模型仍采用的逐视图 Student-t 原型、补全消融、压缩策略及历史结果写入保护。
- 旧性能目录、根性能统计报告和先前功能验证输出已清理，不恢复历史方案入口。本批仅包含 completed，不能据此单独断言补全优于关闭补全或压缩带来收益，相关结论仍需配对实验。

## 维护与验证

- 保留中文模块、类、函数文档字符串，复杂接口说明 Args/Returns/Raises；注释解释边界而非复述语法。
- 用户要求模型架构.md保持“公式介绍＋自然语言解释”风格，更新公式时同步验证分母、集合、mask、detach 和默认值。
- 算法变化同步更新 README、model.md、模型架构.md、dataset/detals.md及模型优化文档。
- 核心逻辑改动必须运行完整单元测试及跨阶段 GPU 冒烟；当前环境 C:\Users\29101\.conda\envs\torch_251_118_39\python.exe。
- 测试应覆盖真实隐藏值隔离、统一全真mask、补全MSE/教师detach、预测梯度与原型梯度分离、概率归一化、无支持回退、local状态配套恢复及通信排除边界。
- 数据、原始表示、锚点和样本ID不进训练上传包。载荷字节仅为编码大小，不当作实测网络流量。
- 压缩配对使用相同实际字节预算，所有率保持同一压缩选择；new version 与旧版本性能记录不得混用。
- 当前 save_model 是兼容元数据，输出以 history/summary JSON 为主，不应声称已落盘权重。
