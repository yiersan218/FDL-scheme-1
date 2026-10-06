# A2 候选筛选记录

四个候选均属于 A2 每视图原型架构，使用四个代表数据集、完整视图与缺失率 0.3、`seed=42` 统一筛选。

| 排名 | A2规则 | 稳定单元 | 宏末轮 NMI | 宏最佳轮 NMI |
|---:|---|---:|---:|---:|
| 1 | Student-t(s=1), semantic=0 | 8/8 | 0.719555 | 0.720294 |
| 2 | Student-t(s=1), semantic=0.1 | 8/8 | 0.719517 | 0.720257 |
| 3 | Student-t(s=2), semantic=0.1 | 8/8 | 0.718606 | 0.719254 |
| 4 | cosine(tau=0.1), semantic=0.1 | 7/8 | 0.701602 | 0.705049 |

冻结第一名 `Student-t(s=1), semantic=0`。筛选原始记录位于 [`tuning/search/`](tuning/search/)，正式三种子记录位于 [`validation/`](validation/)。
