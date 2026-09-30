import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
INIT_CONFIG_DIR = DEFAULT_CONFIG_DIR / "init"
BACKUP_CONFIG_DIR = DEFAULT_CONFIG_DIR / "backup"


class ExperimentConfig(dict):
    """Configuration retaining which two-stage values came from defaults."""

    def __init__(self, *args, inherited_training_keys=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inherited_training_keys = set(inherited_training_keys or ())

    def __deepcopy__(self, memo):
        copied = type(self)(
            copy.deepcopy(dict(self), memo),
            inherited_training_keys=copy.deepcopy(self.inherited_training_keys, memo),
        )
        memo[id(self)] = copied
        return copied


def load_config(config_name_or_path):
    """Load and validate one dataset experiment configuration."""
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

    config = ExperimentConfig(copy.deepcopy(config))
    training = config.get("training", {})
    training.setdefault("center_init_round", 1)
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
    config.setdefault("compression", {})
    compression = config["compression"]
    compression.setdefault("method", "none")
    compression.setdefault("budget_ratio", 0.5)
    compression.setdefault("calibration_size", 64)
    compression.setdefault("candidates", 6)
    compression.setdefault("view_weight", 0.1)
    compression.setdefault("pair_weight", 0.1)
    compression.setdefault("error_feedback", False)
    config["config_path"] = str(path.resolve())
    _validate_config(config)
    return config


def _validate_config(config):
    required_sections = {"dataset", "model", "training", "loss_weights", "output"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    dataset = config["dataset"]
    for key in ("name", "file", "num_clusters", "num_clients"):
        if key not in dataset:
            raise ValueError(f"Missing dataset.{key}")
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
    required_losses = {"reconstruction", "consistency", "clustering", "balance"}
    missing_losses = required_losses.difference(loss_weights)
    if missing_losses:
        raise ValueError(f"Missing loss weights: {sorted(missing_losses)}")
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


def resolve_project_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_overrides(config, overrides):
    """Apply dotted-key overrides and refresh inherited two-stage values."""
    result = copy.deepcopy(config)
    parsed_overrides = []
    for expression in overrides or []:
        if "=" not in expression:
            raise ValueError(f"Invalid override '{expression}', expected key=value")
        dotted_key, raw_value = expression.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        parsed_overrides.append((dotted_key, value))

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
    _validate_config(result)
    return result
