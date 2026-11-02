"""加载、校验和覆盖两阶段联邦多视图聚类配置。"""

import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
ALGORITHM_VERSION = "completion-v1"
# 完整视图是统一缺失模型的零率特例；补全是正式模型，关闭只用于消融。
COMPLETION_DEFAULTS = {
    "enabled": True,
    "anchor_size": 256,
    "topk": 16,
    "heads": 4,
    "adapter_delta": 0.1,
    "attention_strength": 0.3,
    "fusion_weight": 0.25,
    "temperature": 0.2,
    "truncation": 9,
    "use_attention": True,
    "use_completed_initialization": True,
    "use_completed_assignments": True,
    "gate": True,
    "value_mode": "per_view",
    "alignment_mode": "mse",
    "parameter_scope": "federated",
    "seed": 0,
}


class ExperimentConfig(dict):
    """记录配置值及仍继承基础训练参数的阶段字段。"""

    def __init__(self, *args, inherited_training_keys=None, **kwargs):
        """初始化配置字典，并独立保存需要随基础参数更新的字段集合。"""
        super().__init__(*args, **kwargs)
        self.inherited_training_keys = set(inherited_training_keys or ())

    def __deepcopy__(self, memo):
        """深拷贝配置及继承关系，避免候选实验之间共享可变状态。"""
        copied = type(self)(
            copy.deepcopy(dict(self), memo),
            inherited_training_keys=copy.deepcopy(self.inherited_training_keys, memo),
        )
        memo[id(self)] = copied
        return copied


def load_config(config_name_or_path):
    """加载单个数据集配置，补齐默认值并校验实验约束。

    Args:
        config_name_or_path: 数据集名称，或相对于配置目录/项目根目录的 JSON 路径。
            也可传入绝对路径。

    Returns:
        ExperimentConfig: 已补齐默认值的独立配置，包含绝对配置路径及继承字段。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置不满足两阶段训练、每视图原型或补全模块的约束。
    """
    path = Path(config_name_or_path)
    if not path.suffix:
        path = DEFAULT_CONFIG_DIR / f"{path.name}.json"
    elif not path.is_absolute():
        candidate = DEFAULT_CONFIG_DIR / path
        path = candidate if candidate.exists() else PROJECT_ROOT / path

    if not path.exists():
        available = ", ".join(p.stem for p in sorted(DEFAULT_CONFIG_DIR.glob("*.json")))
        raise FileNotFoundError(f"Config not found: {path}. Available configs: {available}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if config.get("algorithm_version", ALGORITHM_VERSION) != ALGORITHM_VERSION:
        raise ValueError("Config declares a different algorithm_version; archived protocols are read-only")
    config = ExperimentConfig(copy.deepcopy(config))
    model = config.get("model", {})
    model.setdefault("cluster_head_type", "student_t")
    model.setdefault("student_t_distance_scale", 1.0)
    # 每视图独立原型保留；完成分配属于新算法版本，不能冒充历史 A2 结果。
    model.setdefault("prototype_mode", "per_view")
    model.setdefault("per_view_head_type", model["cluster_head_type"])
    training = config.get("training", {})
    training.setdefault("center_init_round", 1)
    # 仅为缺省字段建立继承关系，避免命令行覆盖基础值时覆盖用户独立指定的阶段值。
    if "local_epochs" in training:
        for key in ("pretraining_local_epochs", "clustering_local_epochs"):
            if key not in training:
                training[key] = training["local_epochs"]
                config.inherited_training_keys.add(key)
    if "learning_rate" in training:
        if "pretraining_end_learning_rate" not in training:
            training["pretraining_end_learning_rate"] = (
                float(training["learning_rate"]) * 0.1
            )
            config.inherited_training_keys.add("pretraining_end_learning_rate")
        if "clustering_learning_rate" not in training:
            training["clustering_learning_rate"] = float(training["learning_rate"])
            config.inherited_training_keys.add("clustering_learning_rate")
    training.setdefault("cluster_head_learning_rate_multiplier", 1.0)
    training.setdefault("center_momentum", 0.0)
    # JSON 中这三个可选配置节允许用 null 占位，加载时按未配置处理。
    for section in ("compression", "missing", "completion"):
        if config.get(section) is None:
            config[section] = {}
    config.setdefault("compression", {})
    compression = config["compression"]
    compression.setdefault("method", "none")
    compression.setdefault("budget_ratio", 0.5)
    compression.setdefault("calibration_size", 64)
    compression.setdefault("candidates", 6)
    compression.setdefault("view_weight", 0.1)
    compression.setdefault("pair_weight", 0.1)
    compression.setdefault("error_feedback", False)
    compression.setdefault("budget_bytes", None)
    compression.setdefault("budget_bytes_pair", None)
    config.setdefault("missing", {})
    missing = config["missing"]
    missing.setdefault("enabled", True)
    missing.setdefault("rate", 0.0)
    config.setdefault("completion", {})
    for key, value in COMPLETION_DEFAULTS.items():
        config["completion"].setdefault(key, value)
    config.setdefault("research", {})
    config["research"].setdefault("enabled", False)
    _normalize_loss_weights(config)
    config["algorithm_version"] = ALGORITHM_VERSION
    config["config_path"] = str(path.resolve())
    _validate_config(config)
    return config


def _validate_config(config):
    """校验必需字段、取值范围及统一缺失补全模型的组合约束。"""
    required_sections = {"dataset", "model", "training", "loss_weights", "output"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    dataset = config["dataset"]
    for key in ("name", "file", "num_clusters", "num_clients"):
        if key not in dataset:
            raise ValueError(f"Missing dataset.{key}")
    model = config["model"]
    if model.get("cluster_head_type") != "student_t":
        raise ValueError("model.cluster_head_type only supports student_t")
    if model.get("prototype_mode", "per_view") != "per_view":
        raise ValueError("model.prototype_mode only supports per_view")
    if model.get("per_view_head_type", model["cluster_head_type"]) != "student_t":
        raise ValueError("model.per_view_head_type only supports student_t")
    # 历史 Student-t 配置中的温度字段只作为原记录元数据保留，不参与模型计算。
    for key in ("student_t_alpha", "student_t_distance_scale"):
        value = float(model.get(key, 1.0))
        if not 0 < value < float("inf"):
            raise ValueError(f"model.{key} must be finite and positive")
    training = config["training"]
    for key in ("rounds", "pretrain_rounds", "local_epochs", "batch_size", "learning_rate"):
        if key not in training:
            raise ValueError(f"Missing training.{key}")
    if training["pretrain_rounds"] >= training["rounds"]:
        raise ValueError("training.pretrain_rounds must be smaller than training.rounds")
    legacy_keys = {
        "representation_warmup_rounds",
        "warmup_local_epochs",
        "joint_local_epochs",
        "joint_learning_rate",
    }.intersection(training)
    if legacy_keys:
        raise ValueError(
            "Three-stage training keys are no longer supported: "
            f"{sorted(legacy_keys)}"
        )
    center_init_round = int(training["center_init_round"])
    if not 0 <= center_init_round <= int(training["pretrain_rounds"]):
        raise ValueError(
            "training.center_init_round must be in [0, training.pretrain_rounds]"
        )
    if not 0 < float(training["join_ratio"]) <= 1:
        raise ValueError("training.join_ratio must be in (0, 1]")
    if training.get("selection_metric", "nmi") not in {"acc", "nmi", "ari"}:
        raise ValueError("training.selection_metric must be acc, nmi, or ari")
    if float(training["clustering_learning_rate"]) <= 0:
        raise ValueError("training.clustering_learning_rate must be positive")
    if float(training["pretraining_end_learning_rate"]) <= 0:
        raise ValueError("training.pretraining_end_learning_rate must be positive")
    for key in (
        "local_epochs",
        "pretraining_local_epochs",
        "clustering_local_epochs",
    ):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if float(training["cluster_head_learning_rate_multiplier"]) <= 0:
        raise ValueError("training.cluster_head_learning_rate_multiplier must be positive")
    if not 0 <= float(training["center_momentum"]) < 1:
        raise ValueError("training.center_momentum must be in [0, 1)")

    loss_weights = config["loss_weights"]
    required_losses = {"reconstruction", "completion", "clustering", "balance"}
    missing_losses = required_losses.difference(loss_weights)
    if missing_losses:
        raise ValueError(f"Missing loss weights: {sorted(missing_losses)}")
    for name in required_losses:
        if not 0 <= float(loss_weights[name]) < float("inf"):
            raise ValueError(f"loss_weights.{name} must be finite and nonnegative")
    if "view_semantic" in loss_weights:
        value = float(loss_weights["view_semantic"])
        if not 0 <= value < float("inf"):
            raise ValueError("loss_weights.view_semantic must be finite and nonnegative")
    compression = config.get("compression", {})
    if compression.get("method", "none") not in {"none", "topk", "paper", "stage"}:
        raise ValueError("compression.method must be none, topk, paper, or stage")
    if not 0 < float(compression.get("budget_ratio", 0.5)) <= 1:
        raise ValueError("compression.budget_ratio must be in (0, 1]")
    if int(compression.get("calibration_size", 64)) <= 0:
        raise ValueError("compression.calibration_size must be positive")
    if int(compression.get("candidates", 6)) <= 0:
        raise ValueError("compression.candidates must be positive")
    for name in ("view_weight", "pair_weight"):
        if float(compression.get(name, 0.1)) < 0:
            raise ValueError(f"compression.{name} must be nonnegative")
    if not isinstance(compression.get("error_feedback", False), bool):
        raise ValueError("compression.error_feedback must be boolean")
    absolute_budget = compression.get("budget_bytes")
    if absolute_budget is not None and (
        isinstance(absolute_budget, bool)
        or not isinstance(absolute_budget, int)
        or absolute_budget <= 0
    ):
        raise ValueError("compression.budget_bytes must be a positive integer or null")
    budget_pair = compression.get("budget_bytes_pair")
    if budget_pair is not None:
        if not isinstance(budget_pair, dict) or set(budget_pair) != {
            "pre_center", "post_center"
        }:
            raise ValueError("compression.budget_bytes_pair requires pre_center and post_center")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in budget_pair.values()):
            raise ValueError("compression.budget_bytes_pair values must be positive integers")
    research = config.get("research", {})
    if not isinstance(research.get("enabled", False), bool):
        raise ValueError("research.enabled must be boolean")
    completion = config.get("completion", {})
    for key in ("enabled", "use_attention", "use_completed_initialization", "use_completed_assignments", "gate"):
        if not isinstance(completion.get(key, COMPLETION_DEFAULTS[key]), bool):
            raise ValueError(f"completion.{key} must be boolean")
    for key in ("anchor_size", "topk", "heads", "truncation"):
        value = completion.get(key, COMPLETION_DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"completion.{key} must be a positive integer")
    for key, upper in {"adapter_delta": 0.1, "attention_strength": 0.3, "fusion_weight": 0.25}.items():
        value = float(completion.get(key, COMPLETION_DEFAULTS[key]))
        if not 0 <= value <= upper:
            raise ValueError(f"completion.{key} must be in [0, {upper}]")
    if not 0 < float(completion.get("temperature", 0.2)) < float("inf"):
        raise ValueError("completion.temperature must be finite and positive")
    if completion.get("value_mode", "per_view") not in {"per_view", "consensus"}:
        raise ValueError("completion.value_mode must be per_view or consensus")
    if completion.get("alignment_mode", "mse") not in {"mse", "contrastive", "paired"}:
        raise ValueError("completion.alignment_mode must be mse, contrastive, or paired")
    if completion.get("parameter_scope", "federated") not in {"federated", "local"}:
        raise ValueError("completion.parameter_scope must be federated or local")
    if completion.get("enabled", False):
        if int(model["embedding_dim"]) % int(completion.get("heads", 4)):
            raise ValueError("model.embedding_dim must be divisible by completion.heads")
    # 正式补全与消融保持相同 Student-t 原型尺度和四项顶层目标。
    if research.get("enabled", False) or completion.get("enabled", False):
        if (model.get("per_view_head_type", model["cluster_head_type"]) != "student_t"
                or model["cluster_head_type"] != "student_t"
                or float(model.get("student_t_distance_scale", 1.0)) != 1.0
                or float(loss_weights.get("view_semantic", 0.0)) != 0.0):
            raise ValueError("Completion retains Student-t s=1 and view_semantic=0")
    missing = config.get("missing", {})
    if not isinstance(missing.get("enabled", False), bool):
        raise ValueError("missing.enabled must be boolean")
    if not 0 <= float(missing.get("rate", 0.0)) < 1:
        raise ValueError("missing.rate must be in [0, 1)")
    # 缺失率、补全与通信方法独立配置；零率和正率都走观测掩码路径。


def _normalize_loss_weights(config):
    """把历史 consistency 权重迁移为 completion；拒绝两个键给出不同值。"""
    weights = config.get("loss_weights", {})
    if "consistency" in weights:
        legacy = weights.pop("consistency")
        if "completion" in weights and weights["completion"] != legacy:
            raise ValueError("Conflicting loss_weights.consistency and loss_weights.completion")
        weights.setdefault("completion", legacy)


def resolve_project_path(path_value):
    """保留绝对路径；将相对路径解析到项目根目录下。"""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_writable_output(path_value):
    """解析输出路径，阻止新算法覆盖历史记录、冻结快照和源文件目录。

    resolve 会规范化 .. 并解析符号链接或目录联接；已有报告的硬链接也会被识别。
    本函数不创建任何文件或目录，调用方必须在首次写入之前检查。
    """
    target = resolve_project_path(path_value).resolve()
    root = PROJECT_ROOT.resolve()
    protected_dirs = [root / name for name in (
        "results-性能统计", "results-优化2", "results-缺失2",
        "config", "dataset", "system", "tests", "visualization", "模型优化",
    )]
    protected_files = [root / name for name in ("性能统计.md", "性能统计2.md")]
    if target == root:
        raise ValueError("Use a dedicated output directory for completion-v1")
    for protected in protected_dirs:
        protected = protected.resolve()
        if target == protected or protected in target.parents:
            raise ValueError(f"Protected historical/source output path: {target}")
    for protected in protected_files:
        if target == protected.resolve() or (target.exists() and protected.exists() and target.samefile(protected)):
            raise ValueError(f"Protected historical report: {target}")
    # 硬链接不会被 resolve 展开；仅对多链接文件核对历史 inode，避免普通写入扫描整个归档。
    if target.is_file() and target.stat().st_nlink > 1:
        identity = (target.stat().st_dev, target.stat().st_ino)
        for directory in protected_dirs:
            for original in directory.rglob("*"):
                if original.is_file() and (original.stat().st_dev, original.stat().st_ino) == identity:
                    raise ValueError(f"Protected historical/source hard-link alias: {target}")
    return target


def apply_overrides(config, overrides):
    """在配置副本上应用点分路径覆盖，并更新仍继承基础值的阶段参数。

    Args:
        config: 已加载的配置，允许携带阶段参数继承信息。
        overrides: key=value 字符串序列；值优先按 JSON 解析，否则保留为字符串。

    Returns:
        ExperimentConfig: 通过校验的新配置，原配置保持不变。

    Raises:
        KeyError: 覆盖路径或字段不存在。
        ValueError: 覆盖表达式无效，或覆盖后的配置违反约束。
    """
    result = copy.deepcopy(config)
    _normalize_loss_weights(result)
    parsed_overrides = []
    loss_alias_values = {}
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Invalid override '{expression}', expected key=value")
        dotted_key, raw_value = expression.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if dotted_key in {"loss_weights.consistency", "loss_weights.completion"}:
            loss_alias_values[dotted_key] = value
            dotted_key = "loss_weights.completion"
        parsed_overrides.append((dotted_key, value))

    if len(loss_alias_values) == 2 and loss_alias_values["loss_weights.consistency"] != loss_alias_values["loss_weights.completion"]:
        raise ValueError("Conflicting completion weight overrides through legacy and current names")

    overridden_keys = {item[0] for item in parsed_overrides}
    for dotted_key, value in parsed_overrides:
        target = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"Unknown override key: {dotted_key}")
            target = target[part]
        leaf = parts[-1]
        if leaf not in target:
            raise KeyError(f"Unknown override key: {dotted_key}")
        target[leaf] = value

    # 显式覆盖的阶段参数优先，仍继承默认值的字段才跟随基础参数联动。
    inherited_training_keys = getattr(result, "inherited_training_keys", set())
    training = result["training"]
    if "training.local_epochs" in overridden_keys:
        for key in ("pretraining_local_epochs", "clustering_local_epochs"):
            if key in inherited_training_keys and f"training.{key}" not in overridden_keys:
                training[key] = training["local_epochs"]
    if "training.learning_rate" in overridden_keys:
        if (
            "pretraining_end_learning_rate" in inherited_training_keys
            and "training.pretraining_end_learning_rate" not in overridden_keys
        ):
            training["pretraining_end_learning_rate"] = (
                float(training["learning_rate"]) * 0.1
            )
        if (
            "clustering_learning_rate" in inherited_training_keys
            and "training.clustering_learning_rate" not in overridden_keys
        ):
            training["clustering_learning_rate"] = float(training["learning_rate"])

    for dotted_key in overridden_keys:
        if dotted_key.startswith("training."):
            inherited_training_keys.discard(dotted_key[len("training."):])
    _normalize_loss_weights(result)
    if result.get("algorithm_version", ALGORITHM_VERSION) != ALGORITHM_VERSION:
        raise ValueError("Current code only executes algorithm_version=completion-v1")
    result["algorithm_version"] = ALGORITHM_VERSION
    _validate_config(result)
    return result
