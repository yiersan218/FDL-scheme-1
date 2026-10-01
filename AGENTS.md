# 项目上下文：两阶段联邦多视图聚类

本项目当前的有效实验入口是配置驱动的横向联邦多视图聚类。仓库中仍保留部分早期联邦分类文件，但本任务不得让它们进入聚类训练路径。

## 约束

- 客户端持有互不重叠且数量尽量相等的样本子集；旧实验保留全部视图，可选缺失实验在划分后固定遮蔽视图，并保证每个样本至少有一个可见视图。
- 客户端划分只使用随机样本索引，不使用标签分层。
- 标签字段 `Y`、`labels` 或 `gt` 只用于 ACC、NMI、ARI 评估，不得进入训练损失、划分、中心初始化或聚合。
- 服务端只接收模型参数、客户端聚类中心和簇计数摘要，不接收原始样本或单样本表示。
- 当前数据集为 `ALOI_100、flower17、LandUse_21、Mfeat、NUSWIDE、Scene-15、animal`，不要恢复 HW 或 AWA。
- 七个正式配置的重构、一致性、聚类和均衡损失权重都必须大于 0。

## 两阶段流程

1. 第 1 轮至 `pretrain_rounds` 统一属于 `pretraining`。
2. 前 `center_init_round` 轮仅使用重构和跨视图一致性损失。
3. 完成对应轮次后，客户端用本地融合表示生成 KMeans 中心和计数摘要，服务端据此初始化全局中心。
4. 中心初始化后仍属于预训练阶段，DEC KL 和均衡损失权重线性增加到 1。
5. 第 `pretrain_rounds + 1` 轮起进入 `clustering`，使用完整损失权重继续联合优化。
6. 每个通信轮开始时基于本地全部样本固定一次 DEC 目标分布；缺失实验只使用可见视图及本地补全表示，本轮所有 batch 和本地 epoch 共用。
7. 普通参数按样本数 FedAvg；聚类中心先 Hungarian 对齐，再按软簇计数聚合，并可使用 `center_momentum`。
8. 仅在 `clustering` 阶段按配置指标选择最佳检查点。

历史记录只能使用 `pretraining` 和 `clustering` 两种 phase。中心初始化是阶段一内部事件，不得重新命名为独立阶段。

## 核心文件

```text
config/*.json                              六个主数据集配置
config/backup/animal.json                  animal 配置
config/init/*.json                         六个主数据集统一初始参数
system/config.py                           配置加载、验证和覆盖
system/main.py                             单数据集入口
system/run_all.py                          七数据集批量入口
system/flcore/trainmodel/multiview.py       多视图模型和损失
system/flcore/clients/clientcluster.py      客户端本地训练与中心摘要
system/flcore/servers/servercluster.py      两阶段调度、聚合、评估和保存
system/flcore/compression.py                紧凑模型更新编码、解码与校准评分
system/run_compression.py                   固定字节预算候选搜索和压缩报告
system/run_missing.py                       缺失率实验入口和结果汇总
system/utils/mat_data.py                   MAT 加载和无标签客户端划分
visualization/plot_history.py              两阶段曲线绘制
tests/test_clustering.py                    核心回归测试
tests/test_visualization.py                 可视化回归测试
tests/test_missing.py                       掩码与隐藏值隔离回归测试
```

## 配置语义

- `rounds`：总通信轮数。
- `pretrain_rounds`：阶段一结束轮次。
- `center_init_round`：阶段一内部中心初始化时点。
- `learning_rate`：中心初始化前的预训练学习率。
- `pretraining_end_learning_rate`：中心初始化后的预训练学习率。
- `clustering_learning_rate`：阶段二学习率。
- `pretraining_local_epochs`、`clustering_local_epochs`：两阶段本地 epoch。
- `cluster_head_learning_rate_multiplier`：聚类头学习率倍数。
- `center_momentum`：服务端中心融合动量，范围 `[0, 1)`。

禁止重新引入 `representation_warmup_rounds`、`joint_learning_rate`、`warmup_local_epochs` 或 `joint_local_epochs`。配置加载器应对这些旧键直接报错。

## 环境与命令

默认环境为 `C:\Users\29101\.conda\envs\torch_251_118_39`。

```powershell
# 单数据集
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config ALOI_100

# 七数据集正式两阶段运行
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all.py --results-dir results-2阶段

# 测试
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' -m unittest discover -s tests -v
```

## 输出与验证

- 正式结果：`results-2阶段/<dataset>/{summary.json,history.json}`。
- 总表：`results-2阶段/summary.md`。
- 调参记录：`results-2阶段/tuning_summary.md`。
- 候选实验：`results-2阶段/tuning/`，用于记录参数选择依据并随正式结果提交。
- 稳定性要求：最佳轮到末轮的 `|ΔACC|`、`|ΔNMI|`、`|ΔARI|` 均不超过 0.01。
- 改动共享训练逻辑后至少运行完整单元测试和一个跨阶段 GPU 冒烟测试。
- 报告指标必须同时注明 seed、最佳轮和总轮数，不得声称理论或全局最优。

## 维护要求

- 算法、配置、数据格式、运行命令或结果语义变化时，直接更新 README、model.md、dataset/detals.md 和本文档的现有内容。
- 不将 MAT 数据、模型检查点和缓存作为源码提交；当前两阶段运行结果与调参记录需要一并提交。
- 压缩实验必须比较相同**实际上行模型载荷字节数**下的候选；压缩载荷不得包含标签、原始样本或单样本表示。普通参数可稀疏上传，聚类中心与簇计数继续按原规则聚合。
- 压缩正式结果写入 `results-压缩/<dataset>/`，候选保留在 `results-压缩/tuning/`；根目录 `summary.md` 只汇总正式 Stage＋反馈结果，`tuning_summary.md` 只比较它与同预算 Top-k_ef。载荷字节不能写成实测网络流量或延迟。
- 当前压缩正式结果在七个数据集统一使用 `stage_ef`（Stage＋客户端误差反馈）。重新运行 `system/run_compression.py` 默认复用或运行 `stage_ef`、`topk_ef`，按用户指定方案发布，不按七集平均 NMI 自动选法。旧自动统一选优和逐数据集选优需分别显式指定 `--selection uniform`、`--selection per-dataset`，并使用独立结果目录避免覆盖正式结果。
- 20% 上行预算的三种子探索报告位于 `results-压缩/tuning/历史报告/Stage优化实验.md`，原始 JSON 保留在 `results-压缩/tuning/`。当前 50% 单种子结果中 Stage＋反馈的七集平均 NMI 低于 Top-k_ef，不得将局部改善写成已验证的整体创新收益。
- 缺失实验最终只使用本地完整锚点 `attention` 与 `stage_ef`，按 `模型优化/缺失补全.md` 的固定样本×视图掩码协议执行。隐藏原始值不得参与归一化、训练、中心初始化或压缩校准；本地锚点不上传。结果只保留在 `results-缺失/attention_stage_ef/` 并汇总至根目录，注明实际缺失率、最佳轮、总轮数、字节与稳定性。不得把论文已有注意力机制称作本项目原创，也不得用单种子结果声称普遍最优。
