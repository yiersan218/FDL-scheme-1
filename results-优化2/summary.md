# A2 最终结果

当前二次优化仅保留 A2 每视图原型方案。A2 为每个视图维护独立的 Student-t 原型，只融合真实观测视图的软分配；冻结规则为 `Student-t(s=1), view_semantic=0`。

七数据集三种子在完整视图与缺失率 0.3 下的逐数据集 ACC、NMI、ARI 及宏平均见 [`protocol-v2/a2/summary.md`](protocol-v2/a2/summary.md)。原始结果与哈希见 [`protocol-v2/a2/validation_summary.json`](protocol-v2/a2/validation_summary.json)。
