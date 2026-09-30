# Federated Multi-View Clustering

本项目实现配置驱动的横向联邦多视图聚类。每个客户端持有互不重叠的样本子集，每个样本保留全部视图；真实标签只用于计算 ACC、NMI 和 ARI，不参与训练、客户端划分、中心初始化或聚合。

## 模型结构

- 每个视图使用独立的 MLP 自编码器提取表示并重构输入。
- 各视图表示经过 L2 归一化后，由可学习 softmax 权重进行融合。
- 聚类头维护可训练中心，使用 Student-t 分布计算软分配。
- 训练损失由重构损失、跨视图一致性损失、DEC KL 损失和簇均衡损失组成。
- 服务端对普通参数执行样本数加权 FedAvg，对聚类中心先进行 Hungarian 对齐，再按软簇计数聚合。
- 可选上行压缩将客户端普通参数更新编码成稀疏索引与数值；聚类中心及软簇计数保持原有聚合方式。

## 两阶段训练

### 阶段一：聚类感知预训练

第 1 轮至 `pretrain_rounds` 均记录为 `pretraining`。

1. 前 `center_init_round` 轮优化重构和跨视图一致性，建立稳定表示。
2. 完成指定轮数后，各客户端对本地融合表示执行 KMeans，只上传中心和簇计数；服务端据此初始化全局中心。
3. 中心初始化后，在同一个预训练阶段内加入 DEC KL 和均衡损失，其权重线性增加到 1。
4. 每个通信轮开始时根据完整本地数据生成一次固定 DEC 目标分布，本轮所有 batch 共用。

中心初始化是预训练阶段内部事件，不构成额外训练阶段。

### 阶段二：正式聚类

从第 `pretrain_rounds + 1` 轮开始，历史记录为 `clustering`。模型使用完整损失权重和独立的正式聚类学习率继续联合优化，并仅在该阶段按 NMI 选择最佳检查点。

## 数据集

| 数据集 | 样本数 | 视图数 | 视图维度 | 聚类数 | 标签字段 |
|---|---:|---:|---|---:|---|
| ALOI_100 | 10800 | 4 | 77, 13, 64, 125 | 100 | `Y` |
| flower17 | 1360 | 7 | 1360 × 7 | 17 | `Y` |
| LandUse_21 | 2100 | 3 | 20, 59, 40 | 21 | `Y` |
| Mfeat | 2000 | 6 | 216, 76, 64, 6, 240, 47 | 10 | `Y` |
| NUSWIDE | 5000 | 5 | 65, 226, 145, 74, 129 | 5 | `labels` |
| Scene-15 | 4485 | 3 | 20, 59, 40 | 15 | `Y` |
| animal | 10158 | 2 | 4096, 4096 | 50 | `gt` |

六个主数据集配置位于 `config/`，animal 配置位于 `config/backup/animal.json`。MAT 加载器接受 `X/Y`、`data/labels` 和 `X/gt`，并自动识别 `D_v × N` 视图矩阵。

## 环境

默认环境：

```text
C:\Users\29101\.conda\envs\torch_251_118_39
Python 3.9.23
PyTorch 2.5.1 + CUDA
SciPy 1.13.1
scikit-learn 1.6.1
Matplotlib
```

## 关键配置

- `rounds`：总通信轮数。
- `pretrain_rounds`：阶段一结束轮次。
- `center_init_round`：完成多少轮预训练后初始化聚类中心。
- `learning_rate`：中心初始化前的预训练学习率。
- `pretraining_end_learning_rate`：中心初始化后的预训练学习率。
- `clustering_learning_rate`：阶段二学习率。
- `pretraining_local_epochs`、`clustering_local_epochs`：两个阶段的客户端本地 epoch。
- `cluster_head_learning_rate_multiplier`：聚类头相对当前阶段基础学习率的倍数。
- `center_momentum`：服务端聚类中心融合动量。

未显式配置阶段 epoch 时继承 `local_epochs`；未显式配置预训练末端学习率时默认为基础学习率的 0.1 倍，未显式配置正式聚类学习率时继承基础学习率。命令行覆盖基础参数时会同步更新仍处于继承状态的参数。

## 运行命令

单个主数据集：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config ALOI_100
```

animal：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config config/backup/animal.json
```

运行全部七个数据集并写入两阶段结果目录：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all.py --results-dir results-2阶段
```

固定上行字节预算运行 Stage＋误差反馈与 Top-k_ef，并生成正式结果和同预算对比：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_compression.py --results-dir results-压缩 --budget-ratio 0.5
```

默认以 `stage_ef`（阶段感知整包评分＋客户端误差反馈）作为七个数据集统一正式方案，并与相同实际编码上行预算的 `topk_ef` 对照；已有匹配结果直接复用。这里的正式选定是研究方案选择，不是按平均 NMI 事后选优。旧的自动统一选优或逐数据集选优仍可分别用 `--selection uniform`、`--selection per-dataset` 复现，但会覆盖正式结果，应使用独立的 `--results-dir`。其他候选包括 `none`、`topk`、`paper`、`stage`、`stage_task`。单数据集可用 `system/main.py --config Mfeat --override compression.method="stage" --override compression.error_feedback=true --override compression.budget_ratio=0.5`。方案和边界见 [`模型优化/模型压缩.md`](模型优化/模型压缩.md)。

临时覆盖参数：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py `
  --config Scene-15 `
  --override training.rounds=5 `
  --override training.pretrain_rounds=3 `
  --override training.center_init_round=2
```

测试：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' -m unittest discover -s tests -v
```

训练并绘制指标与损失曲线：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' visualization/main.py --config Scene-15
```

可视化图只标记预训练到正式聚类的阶段边界。

## 结果

- 两阶段正式结果：`results-2阶段/<dataset>/{summary.json,history.json}`。
- 汇总表：`results-2阶段/summary.md`。
- 参数选择记录：`results-2阶段/tuning_summary.md`。
- 候选实验：`results-2阶段/tuning/`。
- 压缩正式结果：`results-压缩/<dataset>/{summary.json,history.json}`；[`summary.md`](results-压缩/summary.md) 按两阶段结果样式汇总当前 Stage＋误差反馈性能，[`tuning_summary.md`](results-压缩/tuning_summary.md) 仅对比同预算 Top-k_ef。
- 候选原始结果与早期探索报告保留在 `results-压缩/tuning/`；传输量是单进程模拟中的模型载荷字节，不是实测网络速率。

所有正式结果均固定 `seed=42`，选择指标为 NMI。稳定性要求为最佳轮到末轮的 `|ΔACC|`、`|ΔNMI|`、`|ΔARI|` 均不超过 0.01。当前七个数据集均满足该要求；正式论文仍需独立种子或验证集确认。

当前七个正式配置中的重构、一致性、聚类和均衡损失权重均大于 0。NUSWIDE 的一致性权重经调参设为 0.02，Scene-15 和 animal 的均衡权重设为 0.05。
