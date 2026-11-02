"""将两阶段训练历史绘制为聚类指标曲线及总损失、四项损失分量曲线。"""

import json
from pathlib import Path

import matplotlib

# 使用非交互后端，保证无图形界面的训练服务器也能保存图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = ("acc", "nmi", "ari")
LOSS_KEYS = ("loss", "reconstruction", "consistency", "clustering", "balance")
CURRENT_LOSS_KEYS = ("loss", "reconstruction", "completion", "clustering", "balance")
COLORS = {
    "acc": "#2563eb",
    "nmi": "#059669",
    "ari": "#dc2626",
    "loss": "#111827",
    "reconstruction": "#2563eb",
    "consistency": "#7c3aed",
    "completion": "#7c3aed",
    "clustering": "#ea580c",
    "balance": "#0891b2",
}


def _load_json(path):
    """以 UTF-8 编码读取 JSON 文件并返回解析后的对象。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_history(history):
    """检查历史非空且各轮包含 round 和 train；格式不符时抛出 ValueError。"""
    if not isinstance(history, list) or not history:
        raise ValueError("history.json must contain a non-empty list of round records")
    for index, record in enumerate(history, start=1):
        if not isinstance(record, dict) or "round" not in record or "train" not in record:
            raise ValueError(f"Invalid history record at position {index}")


PHASE_BOUNDARY_STYLES = (
    ("clustering", "Clustering stage starts", "#6b7280", "--"),
)


def _phase_boundaries(history):
    """返回预训练到聚类阶段的分界位置及线条样式，不另标中心初始化。"""
    boundaries = []
    for phase, label, color, linestyle in PHASE_BOUNDARY_STYLES:
        for record in history:
            if record.get("phase") == phase:
                # 分界线放在相邻通信轮之间，避免与某轮的指标点重叠。
                boundaries.append(
                    (float(record["round"]) - 0.5, label, color, linestyle)
                )
                break
    return sorted(boundaries, key=lambda item: item[0])


def _decorate_axis(ax, history, best_round=None):
    """为坐标轴添加训练阶段边界、可选最佳轮标记及网格。"""
    for boundary, label, color, linestyle in _phase_boundaries(history):
        ax.axvline(
            boundary,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    if best_round is not None:
        ax.axvline(float(best_round), color="#f59e0b", linestyle=":", linewidth=1.5, label="Best round")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Communication round")


def _plot_metrics(history, output_path, dataset, best_round, dpi):
    """仅绘制实际记录的 ACC/NMI/ARI 点，保存后关闭图对象。"""
    figure, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    for key in METRIC_KEYS:
        rounds, values = [], []
        for record in history:
            value = record.get("clustering", {}).get(key)
            if value is not None:
                rounds.append(record["round"])
                values.append(value)
        if rounds:
            ax.plot(rounds, values, marker="o", markersize=4, linewidth=2, color=COLORS[key], label=key.upper())
    _decorate_axis(ax, history, best_round)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{dataset} clustering metrics")
    ax.legend(loc="best")
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_losses(history, output_path, dataset, best_round, dpi):
    """绘制总损失和四项分量，空缺值以 NaN 断开曲线，末格放置图例。"""
    rounds = np.asarray([record["round"] for record in history], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True, sharex=True)
    # 旧历史仍按一致性显示；新协议用显式补全损失，避免标签掩盖训练目标变化。
    loss_keys = CURRENT_LOSS_KEYS if any("completion" in row["train"] for row in history) else LOSS_KEYS
    for ax, key in zip(axes.flat, loss_keys):
        values = np.asarray([record["train"].get(key, np.nan) for record in history], dtype=float)
        ax.plot(rounds, values, marker="o", markersize=3.5, linewidth=1.8, color=COLORS[key])
        _decorate_axis(ax, history, best_round)
        ax.set_title("Total loss" if key == "loss" else f"{key.capitalize()} loss")
        ax.set_ylabel("Loss")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        legend_ax.legend(handles, labels, loc="center", frameon=False)
    figure.suptitle(f"{dataset} training losses", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def visualize_history(history_path, output_directory, summary_path=None, image_format="png", dpi=160):
    """根据已有运行记录保存聚类指标图和训练损失图。

    Args:
        history_path: 包含逐轮训练数据的 history.json 路径。
        output_directory: 输出目录；不存在时自动创建。
        summary_path: 可选摘要路径，用于取得数据集名称和最佳轮。
        image_format: 文件扩展名对应的图片格式，例如 png、pdf 或 svg。
        dpi: 图片保存分辨率。

    Returns:
        字典，其 metrics 和 losses 项分别为两张图的 Path 对象。

    Raises:
        ValueError: 训练历史为空或缺少必要字段。
    """
    history_path = Path(history_path)
    history = _load_json(history_path)
    _validate_history(history)

    summary = {}
    if summary_path is not None and Path(summary_path).exists():
        summary = _load_json(summary_path)
    dataset = summary.get("dataset", history_path.parent.name)
    best_round = summary.get("best_round")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    suffix = image_format.lower().lstrip(".")
    metric_path = output_directory / f"clustering_metrics.{suffix}"
    loss_path = output_directory / f"training_losses.{suffix}"
    _plot_metrics(history, metric_path, dataset, best_round, dpi)
    _plot_losses(history, loss_path, dataset, best_round, dpi)
    return {"metrics": metric_path, "losses": loss_path}
