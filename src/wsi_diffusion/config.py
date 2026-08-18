"""Configuration loading and validation for the official WSI-Diffusion code."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator, Mapping, MutableMapping
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml


TISSUE_CLASSES = (
    "neoplastic",
    "dead",
    "inflammatory",
    "non_neoplastic_epithelial",
    "connective",
    "no_label",
)


class Config(MutableMapping):
    """Recursively attribute-accessible mapping backed by plain Python values."""

    def __init__(self, values: Optional[Mapping[str, Any]] = None) -> None:
        object.__setattr__(self, "_values", {})
        for key, value in (values or {}).items():
            self._values[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    @classmethod
    def _unwrap(cls, value: Any) -> Any:
        if isinstance(value, Config):
            return {key: cls._unwrap(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._unwrap(item) for item in value]
        return value

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = self._wrap(value)

    def __delitem__(self, key: str) -> None:
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self._values[name] = self._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        return self._unwrap(self)

    def clone(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))


def _deep_merge(
    base: Dict[str, Any],
    override: Mapping[str, Any],
    reject_unknown: bool = False,
    prefix: str = "",
) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if reject_unknown and key not in merged:
            suggestion = get_close_matches(key, list(merged), n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            raise KeyError(f"Unknown configuration key {dotted!r}{hint}")
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value, reject_unknown, dotted)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path, active: Optional[Set[Path]] = None) -> Dict[str, Any]:
    path = path.resolve()
    chain = set() if active is None else active
    if path in chain:
        raise ValueError(f"Cyclic configuration inheritance at {path}")
    chain.add(path)
    with path.open("r", encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    if not isinstance(current, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")
    base_value = current.pop("_base_", None)
    if base_value is None:
        chain.remove(path)
        return current
    base_paths = [base_value] if isinstance(base_value, str) else list(base_value)
    merged: Dict[str, Any] = {}
    for base_path in base_paths:
        merged = _deep_merge(merged, _read_yaml(path.parent / str(base_path), chain))
    chain.remove(path)
    return _deep_merge(merged, current, reject_unknown=True)


def _apply_override(values: Dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, received {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    allow_new = dotted_key.startswith("+")
    dotted_key = dotted_key[1:] if allow_new else dotted_key
    if not dotted_key:
        raise ValueError("Override key cannot be empty")
    cursor = values
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor:
            if not allow_new:
                suggestion = get_close_matches(part, list(cursor), n=1)
                hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
                raise KeyError(f"Unknown override component {part!r}{hint}")
            cursor[part] = {}
        if not isinstance(cursor[part], dict):
            raise TypeError(f"Cannot descend through non-mapping key {part!r}")
        cursor = cursor[part]
    leaf = parts[-1]
    if leaf not in cursor and not allow_new:
        suggestion = get_close_matches(leaf, list(cursor), n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(f"Unknown override key {dotted_key!r}{hint}")
    cursor[leaf] = yaml.safe_load(raw_value)


def _positive(errors: List[str], value: Any, name: str, allow_zero: bool = False) -> None:
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{name} must be finite")
    elif number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        errors.append(f"{name} must be {qualifier}")


def validate_config(config: Config) -> None:
    """Fail early on unsupported or internally inconsistent experiments."""

    errors: List[str] = []
    if tuple(config.data.tissue_classes) != TISSUE_CLASSES:
        errors.append("data.tissue_classes must preserve the canonical six-class order")
    if int(config.data.num_tissue_classes) != len(TISSUE_CLASSES):
        errors.append("data.num_tissue_classes must equal 6")
    for key in ("wsi_dim", "patch_dim"):
        _positive(errors, config.data[key], f"data.{key}")
    _positive(errors, config.data.feature_cache_size, "data.feature_cache_size", True)
    for key in ("dataset", "manifest", "split_file"):
        if not str(config.data[key]).strip():
            errors.append(f"data.{key} cannot be empty")
    if int(config.data.min_slides_for_diffusion) < 2:
        errors.append("data.min_slides_for_diffusion must be at least 2")
    patch_cap = config.data.max_patches_per_real_slide
    if patch_cap is not None:
        _positive(errors, patch_cap, "data.max_patches_per_real_slide")
    if str(config.data.event_column_semantics) not in {"observed", "censored"}:
        errors.append("data.event_column_semantics must be observed or censored")

    for key in ("patch_size", "stride", "thumbnail_max_side"):
        _positive(errors, config.preprocessing[key], f"preprocessing.{key}")
    _positive(errors, config.preprocessing.level, "preprocessing.level", True)
    if int(config.preprocessing.stride) < int(config.preprocessing.patch_size):
        errors.append("preprocessing.stride must be >= patch_size for non-overlap")
    for key in ("minimum_tissue_fraction", "otsu_saturation_threshold"):
        if not 0.0 <= float(config.preprocessing[key]) <= 1.0:
            errors.append(f"preprocessing.{key} must lie in [0, 1]")
    _positive(errors, config.preprocessing.encoder.batch_size, "preprocessing.encoder.batch_size")
    hovernet_mapping = config.preprocessing.hovernet.pannuke_label_order.to_dict()
    if set(hovernet_mapping) != set(TISSUE_CLASSES[:-1]):
        errors.append("HoVer-Net mapping must define every non-empty canonical tissue class")
    if len(set(int(value) for value in hovernet_mapping.values())) != len(hovernet_mapping):
        errors.append("HoVer-Net source type IDs must be unique")

    if int(config.split.n_folds) < 2:
        errors.append("split.n_folds must be at least 2")
    if not 0.0 < float(config.split.validation_fraction) < 1.0:
        errors.append("split.validation_fraction must lie in (0, 1)")
    _positive(errors, config.split.stratify_time_bins, "split.stratify_time_bins")
    if str(config.normalization.method) != "standard":
        errors.append("normalization.method currently supports only standard")
    if str(config.normalization.fit_on) != "train_only":
        errors.append("normalization.fit_on must be train_only")
    _positive(errors, config.normalization.epsilon, "normalization.epsilon")

    diffusion = config.diffusion
    if str(diffusion.prediction_type) != "epsilon":
        errors.append("diffusion.prediction_type must be epsilon")
    if str(diffusion.schedule.name) not in {"linear", "cosine"}:
        errors.append("diffusion.schedule.name must be linear or cosine")
    if not 0.0 < float(diffusion.schedule.beta_start) <= float(diffusion.schedule.beta_end) < 1.0:
        errors.append("diffusion beta endpoints must satisfy 0 < start <= end < 1")
    for key in ("time_embedding_dim", "hidden_dim", "condition_dim", "num_residual_blocks"):
        _positive(errors, diffusion[key], f"diffusion.{key}")
    if not 0.0 <= float(diffusion.dropout) < 1.0:
        errors.append("diffusion.dropout must lie in [0, 1)")
    for level in ("wsi_level", "patch_level"):
        _positive(errors, diffusion[level].timesteps, f"diffusion.{level}.timesteps")
    if str(diffusion.wsi_level.distribution_parameterization) != "direct":
        errors.append("WSI tissue distribution parameterization must be direct")
    if str(diffusion.wsi_level.condition_pooling) != "mean":
        errors.append("WSI condition pooling must be mean")
    if str(diffusion.patch_level.condition_fusion) != "hadamard":
        errors.append("patch condition fusion must be hadamard")
    _positive(
        errors,
        diffusion.wsi_level.consistency_loss_weight,
        "WSI consistency loss weight",
        True,
    )
    _positive(
        errors,
        diffusion.wsi_level.distribution_loss_weight,
        "WSI distribution loss weight",
        True,
    )

    if str(config.training.optimizer) not in {"adam", "adamw"}:
        errors.append("training.optimizer must be adam or adamw")
    for stage in ("wsi_diffuser", "patch_diffuser", "survival"):
        settings = config.training[stage]
        for key in ("epochs", "batch_size", "learning_rate", "patience"):
            _positive(errors, settings[key], f"training.{stage}.{key}")
        _positive(errors, settings.weight_decay, f"training.{stage}.weight_decay", True)
        if not 0 <= int(settings.warmup_epochs) < int(settings.epochs):
            errors.append(f"training.{stage}.warmup_epochs must lie in [0, epochs)")
    _positive(errors, config.training.patch_diffuser.samples_per_epoch, "patch samples_per_epoch")
    _positive(errors, config.training.patch_diffuser.slide_local_block_size, "patch block size")
    _positive(errors, config.training.gradient_clip_norm, "training.gradient_clip_norm")
    _positive(errors, config.training.checkpoint_every, "training.checkpoint_every")
    _positive(errors, config.training.num_workers, "training.num_workers", True)
    if not 0.0 <= float(config.training.ema_decay) < 1.0:
        errors.append("training.ema_decay must lie in [0, 1)")
    if str(config.training.survival.cox_risk_set) not in {"batch", "full_dataset"}:
        errors.append("cox_risk_set must be batch or full_dataset")
    if str(config.training.survival.cox_ties) not in {"breslow", "efron"}:
        errors.append("cox_ties must be breslow or efron")
    if str(config.training.survival.cox_reduction) not in {"mean", "sum"}:
        errors.append("cox_reduction must be mean or sum")

    if str(config.generation.strategy) not in {"wholly", "partially", "unequally"}:
        errors.append("generation.strategy must be wholly, partially, or unequally")
    if str(config.generation.class_sampling) not in {"generated", "uniform", "empirical_global"}:
        errors.append("generation.class_sampling is unsupported")
    _positive(errors, config.generation.generated_slides_per_patient, "generated slides", True)
    _positive(errors, config.generation.patches_per_generated_slide, "generated patches")
    _positive(errors, config.generation.distribution_temperature, "distribution temperature")
    _positive(errors, config.generation.min_patches_per_nonzero_class, "minimum class patches", True)
    _positive(errors, config.generation.partially_threshold, "partial threshold", True)
    _positive(errors, config.generation.seed_offset, "generation.seed_offset", True)
    if (
        int(config.generation.min_patches_per_nonzero_class)
        * int(config.data.num_tissue_classes)
        > int(config.generation.patches_per_generated_slide)
    ):
        errors.append("minimum class allocations exceed the generated patch budget")
    target = config.generation.target_total_slides
    if target not in (None, "cohort_max"):
        _positive(errors, target, "generation.target_total_slides")

    for key in ("cmil_hidden_dim", "cmil_output_dim"):
        _positive(errors, config.survival_model[key], f"survival_model.{key}")
    if str(config.survival_model.wsi_pooling) != "mean" or str(config.survival_model.patient_pooling) != "mean":
        errors.append("survival pooling currently supports mean/mean")
    if not config.survival_model.patient_hidden_dims:
        errors.append("survival_model.patient_hidden_dims cannot be empty")
    elif any(int(width) < 1 for width in config.survival_model.patient_hidden_dims):
        errors.append("survival_model.patient_hidden_dims must all be positive")
    for key in ("cmil_dropout", "dropout"):
        if not 0.0 <= float(config.survival_model[key]) < 1.0:
            errors.append(f"survival_model.{key} must lie in [0, 1)")
    if str(config.evaluation.primary_metric) != "c_index":
        errors.append("evaluation.primary_metric must be c_index")
    _positive(errors, config.evaluation.bootstrap_samples, "evaluation.bootstrap_samples")
    if not 0.0 < float(config.evaluation.bootstrap_confidence) < 1.0:
        errors.append("evaluation.bootstrap_confidence must lie in (0, 1)")
    if str(config.evaluation.stage5.definition) not in {"fixed_year_accuracy", "quantile_accuracy"}:
        errors.append("unsupported STAGE-5 definition")
    boundaries = [float(value) for value in config.evaluation.stage5.boundaries_days]
    if len(boundaries) != 4 or any(
        not math.isfinite(value) or value <= 0 for value in boundaries
    ) or boundaries != sorted(set(boundaries)):
        errors.append("fixed STAGE-5 requires four increasing positive boundaries")
    if int(config.evaluation.stage5.num_bins) != 5:
        errors.append("STAGE-5 num_bins must equal 5")
    if str(config.evaluation.stage5.censored_policy) not in {
        "exclude_ambiguous",
        "include",
    }:
        errors.append("unsupported STAGE-5 censored_policy")
    if str(config.experiment.precision) not in {"fp32", "amp"}:
        errors.append("experiment.precision must be fp32 or amp")
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))


def load_config(path: Union[str, Path], overrides: Optional[List[str]] = None) -> Config:
    config_path = Path(path).expanduser().resolve()
    values = _read_yaml(config_path)
    for expression in overrides or []:
        _apply_override(values, expression)
    root = config_path.parent
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "pyproject.toml").is_file():
            root = candidate
            break
    values["runtime"] = {"config_path": str(config_path), "project_root": str(root)}
    config = Config(values)
    validate_config(config)
    return config


def dump_config(config: Config, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload.pop("runtime", None)
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
