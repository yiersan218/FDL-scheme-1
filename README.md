# 缺失联邦多视图聚类

当前算法版本为 `completion-v1`，核心方法是**通过补全缺失视图改善联邦聚类**。所有缺失率统一采用客户端观测值归一化、固定观测掩码和同一组四项损失；完整样本就是全真掩码的特例。补全模块默认启用，有支持的缺失预测以受限权重参与聚类。

当前已完成无通信压缩的七数据集、五档缺失率、五种子实验，共 175 次运行，统计见 [results/statistics.md](D:/Project/Python/研究生毕设/FDL-scheme-1/results/statistics.md)。这批实验尚未包含关闭补全或压缩策略的配对对照，因此不能单独证明补全或压缩带来提升。按用户 2026-11-02 的清理要求，项目只保留本批性能记录。

## 文档入口

| 文档 | 介绍内容 |
|---|---|
| [模型架构.md](D:/Project/Python/研究生毕设/FDL-scheme-1/模型架构.md) | 客户端架构、补全与聚类公式、四项损失、训练步骤和梯度边界 |
| [model.md](D:/Project/Python/研究生毕设/FDL-scheme-1/model.md) | 数据流、客户端与服务端职责、初始化、聚合和评估 |
| [二次优化方案.md](D:/Project/Python/研究生毕设/FDL-scheme-1/模型优化/二次优化方案.md) | 当前方案与配对验证设计 |
| [缺失补全.md](D:/Project/Python/研究生毕设/FDL-scheme-1/模型优化/缺失补全.md) | 掩码、观测归一化、数据边界 |
| [缺失补全优化2.md](D:/Project/Python/研究生毕设/FDL-scheme-1/模型优化/缺失补全优化2.md) | 正式补全、消融、回退与历史证据边界 |
| [模型压缩.md](D:/Project/Python/研究生毕设/FDL-scheme-1/模型优化/模型压缩.md) | 独立通信选项 Stage＋误差反馈及预算口径 |
| [dataset/detals.md](D:/Project/Python/研究生毕设/FDL-scheme-1/dataset/detals.md) | 七个数据集与加载规则 |

## 代码职责与入口

| 文件或目录 | 职责 |
|---|---|
| `system/flcore/clients/clientcluster.py` | 本地训练、校准上下文、上传包装和公开 `predict(global_model, round_number)` 评估接口 |
| `system/flcore/servers/servercluster.py` | 轮次调度、初始化、聚合、指标汇总及最佳状态恢复 |
| `system/flcore/trainmodel/multiview.py` | 自编码器、逐视图原型、模型前向和补全概率融合 |
| `system/flcore/trainmodel/completion.py` | 潜在补全、锚点检索与补全自监督 |
| `system/flcore/trainmodel/losses.py` | `target_distribution` 与 `clustering_objective`，集中定义 DEC 目标和统一损失 |
| `system/flcore/communication/codec.py` | 共享参数范围、稀疏编解码、字节计量与共享状态恢复 |
| `system/flcore/communication/compression.py` | 候选评分、Stage 选点与误差反馈 |
| `system/flcore/communication/__init__.py` | 通信包入口；调用方按职责从 codec 或 compression 导入 |

本次布局调整保持 `completion-v1` 的数学定义和默认配置。服务端通过客户端公开预测接口组织本地仿真评估；返回的 CPU 预测与标签用于指标计算，训练上传载荷不变。

## 当前模型

各视图分别使用 MLP 自编码器，观测编码经 L2 归一化后形成源共识。补全器预测目标视图时排除该目标，使用视图适配器与本地完整锚点注意力，学习预测其潜在表示。目标必须真实可见且至少另有一个源视图才可提供自监督；真实缺失数据不提供重构或补全真值。

每视图维护独立 Student-t 原型。真实可见视图的聚类概率等权平均得到 \(Q^{\mathrm{obs}}\)，受支持缺失预测产生 \(Q^{\mathrm{imp}}\)，最终使用

\[
Q=(1-\beta)Q^{\mathrm{obs}}+\beta Q^{\mathrm{imp}},
\qquad 0\le\beta\le\mathrm{fusion\_weight}\le0.25 .
\]

支持权重、\(\beta\) 和补全概率分支使用的原型中心均停止梯度。聚类损失可以更新补全器及源编码器，预测不会作为新增真实观测更新原型计数。无缺失或无支持时 \(\beta=0\)，回到观测分配。

正式目标为

\[
\mathcal L=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+\lambda_{\mathrm{comp}}\mathcal L_{\mathrm{comp}}
+\gamma_t\lambda_{\mathrm{clu}}\mathcal L_{\mathrm{DEC}}
+\gamma_t\lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}} .
\]

- 观测重构：仅对可见输入计算 MSE。
- 补全自监督：留一目标预测与停止梯度的当前观测编码计算归一化潜在 MSE；先平均每视图有效行与特征维度，再平均有效视图。
- DEC：每轮开始用本客户端全部样本的 \(Q^{\mathrm{obs}}\) 构建固定目标，本轮训练完成概率 \(Q\)。
- 弱均衡：约束批次平均完成概率，权重需结合数据分布验证。

前 `center_init_round` 轮训练重构和补全，随后初始化中心；预训练余下轮逐步增加 DEC 和均衡权重，`pretrain_rounds+1` 起进入正式聚类。服务端按样本数聚合共享普通参数，按真实观测软计数聚合对齐后的逐视图原型。评估也为各客户端构建补全上下文；local 模式使用该客户端私有参数，最佳状态配套保存全局参数、私有参数与轮号。

## 数据与环境

| 数据集 | 样本数 | 视图数 | 视图维度 | 簇数 |
|---|---:|---:|---|---:|
| ALOI_100 | 10800 | 4 | 77, 13, 64, 125 | 100 |
| flower17 | 1360 | 7 | 1360 × 7 | 17 |
| LandUse_21 | 2100 | 3 | 20, 59, 40 | 21 |
| Mfeat | 2000 | 6 | 216, 76, 64, 6, 240, 47 | 10 |
| NUSWIDE | 5000 | 5 | 65, 226, 145, 74, 129 | 5 |
| Scene-15 | 4485 | 3 | 20, 59, 40 | 15 |
| animal | 10158 | 2 | 4096, 4096 | 50 |

当前是单进程联邦仿真，客户端持有互不重叠的无标签随机样本分片。标签仅用于指标、最佳轮选择和类别数校验。训练上传包不含原始样本、逐样本表示或锚点，但这不等同于物理隔离或形式化隐私保证。

项目环境为 Python 3.9、PyTorch 2.5.1、SciPy、scikit-learn、Matplotlib。现有解释器：

```text
C:\Users\29101\.conda\envs\torch_251_118_39\python.exe
```

## 配置与开启缺失

七份配置位于 `config/<dataset>.json`。保留的顶层占位表示采用当前默认值：

| JSON 字段 | `null` 加载后的行为 |
|---|---|
| `missing` | enabled=true、rate=0.0；完整数据，全真 mask |
| `completion` | enabled=true、alignment_mode=mse、parameter_scope=federated |
| `compression` | method=none，不压缩 |

因此直接运行根配置仍是完整输入，但会训练补全自监督。需要缺失实验时，把对应占位改为：

```json
"missing": {
  "enabled": true,
  "rate": 0.3
}
```

`rate=0.3` 表示约 30% 的本地样本各缺失一个视图，不是 30% 的视图单元格。掩码在客户端划分后生成、跨轮固定，每个样本至少保留一个视图。所有缺失率均按客户端真实可见值归一化；旧 `missing.enabled=false` 仅表示有效缺失率为零。

| 常用字段 | 当前含义 |
|---|---|
| `loss_weights.completion` | 正式补全系数，七份配置均从 1.0 开始，尚未调优 |
| `completion.use_completed_assignments` | 是否让有支持的缺失概率参与最终聚类，默认 true |
| `completion.use_completed_initialization` | 是否让完成融合影响共同 KMeans 伪标签，普通运行默认 true |
| `completion.parameter_scope` | federated 随共享模型聚合；local 仅在客户端跨轮保存补全参数 |
| `completion.anchor_size/topk` | 默认 256/16，分别控制锚点库上限和邻居数 |
| `completion.alignment_mode` | 正式为 mse；contrastive、paired 为显式消融 |
| `completion.enabled` | false 为观测基线；只在此基线保留旧观测融合 MSE |
| `compression.method/error_feedback` | 独立通信选项，缺失不会强制启用压缩 |

旧 `loss_weights.consistency` 可在加载或命令行覆盖时迁移为 `completion`，正式配置使用新名称。`null` 约定只适用于三个顶层配置节，内部参数需填写具体类型。

## 运行命令

以下命令在项目根目录执行。单数据集完整输入：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config Scene-15
```

同一正式方法的缺失率 0.3 实验，另存结果：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config Scene-15 `
  --override missing.rate=0.3 `
  --override output.directory=results-补全聚类-r03
```

单次运行默认写到 `results-补全聚类/<dataset>/`。改变配置时应使用独立目录；单数据集入口不提供矩阵入口的计划复用校验。运行七个数据集：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all.py --results-dir results-补全聚类
```

统一实验入口 `system/run_incomplete.py` 支持三组对照：

| variant | 补全自监督 | 补全概率参与 Q |
|---|---|---|
| `baseline` | 关闭补全，使用观测基线 MSE | 否 |
| `observed_completion` | 开启 | 否 |
| `completed` | 开启 | 是 |

三组都关闭 `use_completed_initialization`，统一使用观测初始化，避免把初始化变化混入概率融合增益。普通根配置仍保留完成初始化开关，因此矩阵不是对默认初始化设置的逐字复刻。三组共享数据划分、固定 mask、归一化、公共模型参数和通信选择；baseline 与另两组损失目标不同，不能直接比较总损失数值。

先生成可核查的计划和按实际模块清单保存的源码快照：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_incomplete.py `
  --datasets Scene-15 --rates 0 0.3 --seeds 42 `
  --variants baseline observed_completion completed --compression none `
  --results-dir results-补全聚类-对照 --plan-only
```

检查后移除 `--plan-only` 即按同一计划训练。省略筛选参数时默认七个数据集、`rates=0/0.1/0.3/0.5/0.7`、`seeds=42`、仅 `completed`、不压缩，写入 `results-补全聚类-矩阵`。用 `--seeds 42 2024 2025` 可指定多个种子；`--device cpu` 可切换 CPU；共同模型或损失参数用 `--override` 设置。

压缩选项为 `none/stage_ef/topk_ef`。后两者使用关闭补全的基线共享模型导出的同一绝对 pre/post 字节预算（比例默认 0.5），新增补全参数不会使候选得到更多预算。所有缺失率保持同一通信方式。修改压缩或其他计划参数需使用新的 `--results-dir`。

矩阵入口保存 `plan.json`、`source_snapshot/`、每任务 `history.json/summary.json`、总 `summary.json` 与 `report.md`。同目录只复用完全匹配的配置、源码和完整运行记录；不匹配时拒绝混写。模块拆分后源码清单与指纹变化，即使算法版本不变，也需要新的结果目录。报告同时记录真实末轮与聚类阶段按标签指标选择的最佳轮；后者属于标签辅助评估。

独立压缩比较入口：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_compression.py `
  --results-dir results-补全聚类-压缩 --budget-ratio 0.5
```

## 检查与可视化

运行单元测试：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' -m unittest discover -s tests -v
```

可视化已有结果：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' visualization/main.py --config Scene-15 `
  --skip-training --results-directory results-补全聚类 `
  --plot-directory visualization/output-completion-v1
```

可视化展示逐轮指标、损失和两阶段边界。`output.save_model` 当前为兼容元数据，不会把最佳权重写成磁盘检查点；保存的主要结果为配置、历史和摘要 JSON。

此前完成过单元测试、跨阶段 CUDA 功能验证及布局调整前后的数值一致性检查；这些功能检查不属于性能实验，其旧输出已随历史记录清理。

## 保留的实验记录与当前限制

项目当前仅保留 [results](D:/Project/Python/研究生毕设/FDL-scheme-1/results) 中的本批性能记录：七个数据集，缺失率 0、0.1、0.3、0.5、0.7，种子 42、2024、2025、2026、2027，共 175 次完整运行。通信压缩为 none，补全方案为 completed，统一使用观测初始化；网络规模和训练轮数沿用各数据集配置。

[statistics.md](D:/Project/Python/研究生毕设/FDL-scheme-1/results/statistics.md) 分别列出同一最佳轮和真实末轮的 ACC、NMI、ARI 五次均值与样本方差（ddof=1，分母为 4）；[statistics.json](D:/Project/Python/研究生毕设/FDL-scheme-1/results/statistics.json) 另保留总体方差。单次原始记录、计划、训练源码快照、统计源码快照及执行日志均位于本批目录内。后续改变实验设置时使用新的输出目录，避免与已完成记录混写。

旧 A2、missing2 性能目录、旧性能统计报告及先前功能验证输出已按用户要求删除；对应历史运行与核验入口也已移除。本文中的其他输出目录名称是未来运行示例，不表示仍保留旧结果。

当前通过完整 MAT 模拟样本视图缺失，尚未接入真实 NaN 或外部 mask 输入。注意力仍依赖本地真实完整锚点；空库回退为映射预测，有效支持不足时不混入补全概率。只有一个可见视图的样本无法提供留一目标自监督。支持度不是补全准确概率，受限融合也不保证性能；仍需多种子配对实验验证补全消融、损失权重及通信交互。
