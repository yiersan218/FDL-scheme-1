# Federated Multi-View Clustering

本项目实现配置驱动的横向联邦多视图聚类。每个客户端持有互不重叠的样本子集；原始数据包含全部视图，可选实验在客户端划分后固定遮蔽部分视图，并保证每个样本至少保留一个视图。真实标签只用于计算 ACC、NMI 和 ARI，不参与训练、客户端划分、缺失掩码、中心初始化或聚合。

## 模型结构

- 每个视图使用独立的 MLP 自编码器提取表示并重构输入。
- 各视图表示经过 L2 归一化后，由可学习 softmax 权重进行融合。
- A2 为每个视图维护独立的 Student-t 原型，并只平均真实观测视图的软分配。
- 训练损失由重构损失、跨视图一致性损失、DEC KL 损失和簇均衡损失组成。
- 服务端对普通参数执行样本数加权 FedAvg。每视图原型选择覆盖率最高的参考视图，只求一次 Hungarian 排列并同步到该客户端全部视图，再按各视图软簇计数聚合。
- 可选上行压缩将客户端普通参数更新编码成稀疏索引与数值；聚类中心及软簇计数保持原有聚合方式。

## 两阶段训练

### 阶段一：聚类感知预训练

第 1 轮至 `pretrain_rounds` 均记录为 `pretraining`。

1. 前 `center_init_round` 轮优化重构和跨视图一致性，建立稳定表示。
2. 完成指定轮数后，各客户端对本地融合表示执行 KMeans，只上传中心和簇计数；服务端据此初始化全局中心。
3. 中心初始化后，在同一个预训练阶段内加入 DEC KL 和均衡损失，其权重线性增加到 1。
4. 每个通信轮开始时根据本地全部样本生成一次固定 DEC 目标分布；缺失实验只使用真实可见视图，本轮所有 batch 共用。

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
- `model.prototype_mode`：固定为 `per_view`，即每个视图维护独立原型。
- `model.per_view_head_type`：A2 候选消融支持 `student_t` 或 `cosine`；最终冻结性能规则使用 Student-t `s=1`，不增加单视图语义损失。
- `missing.enabled/rate`：开启固定缺失视图实验并设置不完整样本率。A2 不补全隐藏表示，只融合真实可见视图。

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
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_all.py --results-dir results-A2-两阶段
```

固定上行字节预算运行 Stage＋误差反馈与 Top-k_ef，并生成正式结果和同预算对比：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/run_compression.py --results-dir results-A2-压缩 --budget-ratio 0.5
```

默认以 `stage_ef`（阶段感知整包评分＋客户端误差反馈）作为七个数据集统一正式方案，并与相同实际编码上行预算的 `topk_ef` 对照；已有匹配结果直接复用。这里的正式选定是研究方案选择，不是按平均 NMI 事后选优。旧的自动统一选优或逐数据集选优仍可分别用 `--selection uniform`、`--selection per-dataset` 复现，但会覆盖正式结果，应使用独立的 `--results-dir`。其他候选包括 `none`、`topk`、`paper`、`stage`、`stage_task`。单数据集可用 `system/main.py --config Mfeat --override compression.method="stage" --override compression.error_feedback=true --override compression.budget_ratio=0.5`。方案和边界见 [`模型优化/模型压缩.md`](模型优化/模型压缩.md)。

缺失视图最终方案为固定掩码下的 A2 可见视图融合。默认在七个数据集上运行完整控制组 `r=0` 和 0.1、0.3、0.5、0.7 四档不完整样本率；`r=0` 不压缩，正缺失率使用 `stage_ef` 与当前模型稠密上行载荷 50% 的预算：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' -u system/run_a2_matrix.py --results-dir results-性能统计
```

用 `--datasets Scene-15 --rates 0.3 --seeds 42` 可选择单项。掩码在客户端划分后生成，每个不完整样本恰好缺失一个视图；本地归一化仅用可见值。由于 `r=0` 与正缺失率的压缩设置不同，五档结果不能解释为只改变缺失率的单因素消融。方法、公式、标签隔离和结论边界见 [`缺失补全.md`](模型优化/缺失补全.md)。

A2 每视图原型是当前唯一保留的二次优化方案。完整视图 Mfeat 示例：

```powershell
& 'C:\Users\29101\.conda\envs\torch_251_118_39\python.exe' system/main.py --config results-优化2/protocol-v2/a2/configs/Mfeat__full__selected.json
```

缺失率 0.3 的可运行配置使用同目录的 `*__missing_0p3__selected.json`。A2 的表示融合、聚类分配、中心和计数都只使用真实观测行，不生成隐藏视图表示。方案、论文依据和验证边界见 [`二次优化方案.md`](模型优化/二次优化方案.md)。

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

- 当前 A2 权威性能记录位于 `results-性能统计/A2/` 与 `results-优化2/protocol-v2/a2/`。
- `results-2阶段/`、`results-压缩/`、`results-缺失/` 与 `results-优化1/` 是切换到 A2-only 前保留的历史实验，不作为当前 A2 性能证据，也不应由当前入口原地覆盖。
- 新的七集批量或压缩实验应使用独立目录，例如 `results-A2-两阶段/`、`results-A2-压缩/`；传输量是单进程模拟中的模型载荷字节，不是实测网络速率。
- A2 缺失视图实验写入 `results-性能统计/A2/rate-<r>/seed-<seed>/<dataset>/`；每个任务保留配置、掩码统计与逐轮历史。缺失实验使用可见值本地归一化，不能直接与旧预处理结果比较。
- A2 多缺失率正式结果：[`性能统计.md`](性能统计.md)。七个数据集在缺失率 `0/0.1/0.3/0.5/0.7`、`seed=42` 下的原始记录位于 `results-性能统计/A2/`；完整视图与缺失率 0.3 的三种子验证记录位于 `results-优化2/protocol-v2/a2/validation/`。

当前 A2 矩阵固定 `seed=42`，选择指标为 NMI。稳定性要求为最佳轮到末轮的 `|ΔACC|`、`|ΔNMI|`、`|ΔARI|` 均不超过 0.01；正式论文仍需更多独立种子或验证集确认。

当前七个正式配置中的重构、一致性、聚类和均衡损失权重均大于 0。NUSWIDE 的一致性权重经调参设为 0.02，Scene-15 和 animal 的均衡权重设为 0.05。
