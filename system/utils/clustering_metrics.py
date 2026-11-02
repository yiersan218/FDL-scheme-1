"""计算聚类评估指标；标签匹配仅用于评估，不反馈到训练目标。"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_accuracy(labels_true, labels_pred):
    """用 Hungarian 最优标签匹配计算不受簇编号排列影响的 ACC。

    Args:
        labels_true: 非空的一维真实标签数组，元素为非负整数。
        labels_pred: 与真实标签等长的非负整数簇编号数组。

    Returns:
        最优一一匹配后的正确样本比例，类型为 float。
    """
    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)
    size = max(labels_true.max(initial=0), labels_pred.max(initial=0)) + 1
    confusion = np.zeros((size, size), dtype=np.int64)
    np.add.at(confusion, (labels_pred, labels_true), 1)
    # 最小化转换后的代价，等价于最大化匹配簇中的样本数。
    row, col = linear_sum_assignment(confusion.max() - confusion)
    return float(confusion[row, col].sum() / len(labels_true))


def evaluate_clustering(labels_true, labels_pred):
    """返回包含 acc、nmi、ari 三项 Python 浮点数的指标字典。"""
    return {
        "acc": clustering_accuracy(labels_true, labels_pred),
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
    }

