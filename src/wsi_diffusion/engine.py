"""Training, checkpointing, experiment I/O, and evaluation utilities.

The official implementation keeps the complete runtime layer in this module so
that training state, fold evaluation, and experiment completion semantics are
defined in one place.  A stage is complete only after its explicit completion
marker has been written and every declared output still exists.  Checkpoints
contain enough state for an exact continuation, including all random generators.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wsi_diffusion.config import Config
from wsi_diffusion.data import PatientBags
from wsi_diffusion.models import (
    HierarchicalCMILSurvival,
    negative_cox_partial_log_likelihood,
)


# ---------------------------------------------------------------------------
# Standard, atomic I/O
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Return a recursively JSON-compatible value.

    Metrics without a valid estimate (for example, a C-index with no comparable
    patient pairs) are represented by standard JSON ``null`` rather than the
    non-standard JavaScript tokens ``NaN`` or ``Infinity``.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    """Write indented UTF-8 JSON with an atomic same-directory replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(
                _json_safe(payload),
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Atomically save a trusted PyTorch artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a trusted project artifact across PyTorch 1.9--current releases."""

    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch 1.9 does not expose ``weights_only``.
        return torch.load(Path(path), map_location=map_location)


def torch_load_cpu(path: str | Path) -> Any:
    return torch_load(path, map_location="cpu")


def torch_load_restricted_cpu(path: str | Path) -> Any:
    """Prefer PyTorch's restricted tensor loader when the runtime supports it."""

    source = Path(path)
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with the paper environment (PyTorch 1.9).
        warnings.warn(
            "This PyTorch version cannot restrict pickle loading; only open "
            f"trusted feature files: {source}",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(source, map_location="cpu")


# ---------------------------------------------------------------------------
# Devices, determinism, logging, layout, and provenance
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def amp_enabled(precision: str, device: torch.device) -> bool:
    if precision not in {"fp32", "amp"}:
        raise ValueError("precision must be 'fp32' or 'amp'")
    return precision == "amp" and device.type == "cuda"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch, including deterministic CUDA kernels.

    The cuBLAS workspace variable is installed before the first CUDA query.  If
    CUDA was initialized by caller code earlier, a restart warning explains why
    PyTorch 1.9 may still reject a deterministic matrix multiplication.
    """

    if deterministic:
        had_workspace_setting = "CUBLAS_WORKSPACE_CONFIG" in os.environ
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            warnings.warn(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be "
                "':4096:8' or ':16:8' in the PyTorch 1.9 environment.",
                RuntimeWarning,
                stacklevel=2,
            )
        cuda_was_initialized = bool(
            getattr(torch.cuda, "is_initialized", lambda: False)()
        )
        if cuda_was_initialized and not had_workspace_setting:
            warnings.warn(
                "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG was set. "
                "Restart the process and call seed_everything before creating CUDA "
                "tensors when using deterministic PyTorch 1.9 training.",
                RuntimeWarning,
                stacklevel=2,
            )
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
        except TypeError:  # PyTorch 1.9 has no ``warn_only`` keyword.
            torch.use_deterministic_algorithms(deterministic)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_logging(
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> logging.Logger:
    logger = logging.getLogger("wsi_diffusion")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if output_dir is not None:
        log_path = Path(output_dir) / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def project_path(config: Config, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config.runtime.project_root) / path).resolve()


def fold_directory(config: Config, fold: int) -> Path:
    if fold < 0:
        raise ValueError("fold must be non-negative")
    return project_path(config, config.experiment.output_dir) / f"fold_{fold}"


def checkpoint_path(
    config: Config,
    fold: int,
    stage: str,
    prefer_best: bool = True,
) -> Path:
    directory = fold_directory(config, fold) / "checkpoints" / stage
    best = directory / "best.pt"
    last = directory / "last.pt"
    if prefer_best and best.exists():
        return best
    return last if last.exists() else best


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_provenance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    commit = _git_value(root, "rev-parse", "HEAD")
    status = _git_value(root, "status", "--porcelain") if commit is not None else None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "source_control_available": commit is not None,
    }


def write_provenance(
    project_root: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    destination = Path(output_dir) / "provenance.json"
    if overwrite or not destination.exists():
        atomic_json_dump(collect_provenance(project_root), destination)
    return destination


# ---------------------------------------------------------------------------
# Explicit stage-completion markers
# ---------------------------------------------------------------------------


def stage_marker_path(directory: str | Path, stage: str) -> Path:
    if not stage or any(part in stage for part in ("/", "\\", "..")):
        raise ValueError(f"Invalid stage name: {stage!r}")
    return Path(directory) / "stages" / f"{stage}.complete.json"


def mark_stage_complete(
    directory: str | Path,
    stage: str,
    outputs: Sequence[str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a completion marker only after every declared output exists."""

    resolved_outputs = [Path(path).expanduser().resolve() for path in outputs]
    if not resolved_outputs:
        raise ValueError("A completed stage must declare at least one output")
    missing = [str(path) for path in resolved_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot mark stage {stage!r} complete; missing outputs: {missing}"
        )
    marker = stage_marker_path(directory, stage)
    atomic_json_dump(
        {
            "schema_version": 1,
            "stage": stage,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "outputs": [str(path) for path in resolved_outputs],
            "metadata": dict(metadata or {}),
        },
        marker,
    )
    return marker


def stage_is_complete(
    directory: str | Path,
    stage: str,
    required_outputs: Sequence[str | Path] | None = None,
) -> bool:
    """Return true only for a valid marker whose required outputs still exist."""

    marker = stage_marker_path(directory, stage)
    if not marker.is_file():
        return False
    try:
        with marker.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return False
    if payload.get("stage") != stage:
        return False
    declared = payload.get("outputs")
    if not isinstance(declared, list) or not declared or not all(
        isinstance(path, str) for path in declared
    ):
        return False
    declared_paths = [Path(path) for path in declared]
    requested_paths = (
        [Path(path).expanduser().resolve() for path in required_outputs]
        if required_outputs is not None
        else declared_paths
    )
    return (
        bool(requested_paths)
        and set(requested_paths).issubset(set(declared_paths))
        and all(path.exists() for path in declared_paths)
        and all(path.exists() for path in requested_paths)
    )


def remove_stage_marker(directory: str | Path, stage: str) -> bool:
    """Remove one marker when explicitly restarting a stage."""

    marker = stage_marker_path(directory, stage)
    if not marker.exists():
        return False
    marker.unlink()
    return True


# ---------------------------------------------------------------------------
# Optimizers, moving averages, early stopping, and checkpoints
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer {name!r}")


def build_epoch_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_epochs < 1 or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("Require total_epochs >= 1 and 0 <= warmup_epochs < total_epochs")

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return max(1.0e-8, (epoch + 1) / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class ExponentialMovingAverage:
    """Parameter EMA used for stable diffusion validation and sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = decay
        self.averaged_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.averaged_parameters:
                average = self.averaged_parameters[name]
                if average.device != parameter.device or average.dtype != parameter.dtype:
                    average = average.to(device=parameter.device, dtype=parameter.dtype)
                    self.averaged_parameters[name] = average
                average.lerp_(parameter.detach(), 1 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "average": self.averaged_parameters}

    def load_state_dict(self, state: Mapping[str, Any], model: nn.Module | None = None) -> None:
        self.decay = float(state["decay"])
        raw_average = state.get("average")
        if not isinstance(raw_average, Mapping):
            raise ValueError("EMA checkpoint has no averaged parameter mapping")
        parameters = dict(model.named_parameters()) if model is not None else {}
        self.averaged_parameters = {}
        for name, value in raw_average.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"EMA value for {name!r} is not a tensor")
            parameter = parameters.get(str(name))
            self.averaged_parameters[str(name)] = (
                value.detach().clone().to(parameter)
                if parameter is not None
                else value.detach().clone()
            )

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        original: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.averaged_parameters:
                    original[name] = parameter.detach().clone()
                    parameter.copy_(self.averaged_parameters[name].to(parameter))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in original:
                        parameter.copy_(original[name])


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    minimum_delta: float = 0.0
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if self.minimum_delta < 0:
            raise ValueError("minimum_delta cannot be negative")

    def update(self, value: float) -> tuple[bool, bool]:
        """Update selection state; a non-finite metric is always a bad epoch."""

        if not math.isfinite(value):
            self.bad_epochs += 1
            return False, self.bad_epochs >= self.patience
        improved = self.best is None or (
            value < self.best - self.minimum_delta
            if self.mode == "min"
            else value > self.best + self.minimum_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("mode", self.mode) != self.mode:
            raise ValueError("Early-stopping mode differs from the checkpoint")
        self.best = None if state.get("best") is None else float(state["best"])
        self.bad_epochs = int(state.get("bad_epochs", 0))


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy checkpoint metadata into a plain string-keyed mapping."""

    return {str(key): value for key, value in metadata.items()}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    ema: ExponentialMovingAverage | None,
    epoch: int,
    best_metric: float | None,
    history: list[dict[str, float]],
    metadata: Mapping[str, Any],
    early_stopping: EarlyStopping | None = None,
) -> None:
    """Save complete continuation state and descriptive run metadata."""

    atomic_torch_save(
        {
            "schema_version": 2,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
            "epoch": int(epoch),
            "best_metric": (
                float(best_metric)
                if best_metric is not None and math.isfinite(float(best_metric))
                else None
            ),
            "history": history,
            "early_stopping": early_stopping.state_dict() if early_stopping else None,
            "rng": rng_state(),
            "metadata": _checkpoint_metadata(metadata),
        },
        path,
    )


def _validate_checkpoint_metadata(
    path: str | Path,
    metadata: Any,
    expected_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError(f"Checkpoint {path} has no metadata mapping")
    for key, expected in (expected_metadata or {}).items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Checkpoint {path} metadata mismatch for {key!r}: "
                f"expected {expected!r}, got {metadata.get(key)!r}"
            )
    return metadata


def load_model_checkpoint(
    path: str | Path,
    model: nn.Module,
    use_ema: bool = True,
    strict: bool = True,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Not a valid model checkpoint: {path}")
    _validate_checkpoint_metadata(path, payload.get("metadata"), expected_metadata)
    model.load_state_dict(payload["model"], strict=strict)
    if use_ema and payload.get("ema") is not None:
        average = payload["ema"].get("average")
        if not isinstance(average, Mapping):
            raise ValueError(f"Checkpoint {path} contains invalid EMA state")
        state = model.state_dict()
        for name, value in average.items():
            if name in state:
                state[name] = value.to(state[name])
        model.load_state_dict(state, strict=strict)
    return payload


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_metric: float | None
    history: list[dict[str, float]]


def _optimizer_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _optimizer_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_optimizer_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_value_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_parameters(
    optimizer: torch.optim.Optimizer,
) -> None:
    """Move CPU-loaded momentum tensors beside their associated parameters."""

    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor) or not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            state[key] = _optimizer_value_to_device(value, parameter.device)


def restore_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    ema: ExponentialMovingAverage | None,
    early_stopping: EarlyStopping,
    expected_metadata: Mapping[str, Any] | None = None,
) -> ResumeState:
    """Restore all train state from ``last.pt`` and return the next epoch."""

    payload = torch_load_cpu(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Not a valid training checkpoint: {path}")
    _validate_checkpoint_metadata(path, payload.get("metadata"), expected_metadata)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    _move_optimizer_state_to_parameters(optimizer)
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"], model=model)
    stopping_state = payload.get("early_stopping")
    if isinstance(stopping_state, Mapping):
        early_stopping.load_state_dict(stopping_state)
    if isinstance(payload.get("rng"), Mapping):
        restore_rng_state(payload["rng"])
    raw_history = payload.get("history", [])
    history = list(raw_history) if isinstance(raw_history, list) else []
    best = payload.get("best_metric")
    return ResumeState(
        start_epoch=int(payload.get("epoch", -1)) + 1,
        best_metric=float(best) if best is not None and math.isfinite(float(best)) else None,
        history=history,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _batch_size(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.size(0))
    raise ValueError("A training batch must contain at least one batched tensor")


LossFunction = Callable[[nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]]


class DiffusionTrainer:
    """Unified mixed-precision trainer for WSI- and patch-level diffusers."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        patience: int,
        optimizer_name: str,
        gradient_clip_norm: float,
        ema_decay: float,
        amp: bool,
        checkpoint_dir: str | Path,
        checkpoint_every: int,
        logger: logging.Logger,
        metadata: Mapping[str, Any],
        resume: bool = False,
    ) -> None:
        if epochs < 1 or checkpoint_every < 1:
            raise ValueError("epochs and checkpoint_every must be positive")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.logger = logger
        self.metadata = _checkpoint_metadata(metadata)
        self.optimizer = build_optimizer(
            self.model, optimizer_name, learning_rate, weight_decay
        )
        self.scheduler = build_epoch_scheduler(
            self.optimizer, epochs, warmup_epochs
        )
        self.ema = ExponentialMovingAverage(self.model, ema_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.amp = bool(amp)
        self.early_stopping = EarlyStopping(patience, mode="min")
        self.resume = resume

    def _epoch(
        self,
        loader: DataLoader[Any],
        loss_function: LossFunction,
        training: bool,
    ) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        examples = 0
        batches = 0
        for raw_batch in loader:
            if not isinstance(raw_batch, Mapping):
                raise TypeError("Diffusion DataLoader batches must be mappings")
            moved = _move_batch(raw_batch, self.device)
            batch_size = _batch_size(moved)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=self.amp):
                    losses = loss_function(self.model, moved)
                if "loss" not in losses:
                    raise KeyError("loss_function must return a mapping containing 'loss'")
                if losses["loss"].ndim != 0:
                    raise ValueError("loss_function['loss'] must be a scalar tensor")
                if not bool(torch.isfinite(losses["loss"]).all()):
                    phase = "training" if training else "validation"
                    raise FloatingPointError(f"Non-finite {phase} diffusion loss")
                if training:
                    self.scaler.scale(losses["loss"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.ema.update(self.model)
            for name, value in losses.items():
                if value.ndim != 0:
                    raise ValueError(f"Loss component {name!r} must be scalar")
                scalar = float(value.detach())
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"Non-finite loss component {name!r}")
                totals[name] = totals.get(name, 0.0) + scalar * batch_size
            examples += batch_size
            batches += 1
        if batches == 0 or examples == 0:
            raise ValueError("Diffusion DataLoader produced no examples")
        return {name: value / examples for name, value in totals.items()}

    def _save(
        self,
        path: str | Path,
        epoch: int,
        history: list[dict[str, float]],
    ) -> None:
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
            epoch=epoch,
            best_metric=self.early_stopping.best,
            history=history,
            metadata=self.metadata,
            early_stopping=self.early_stopping,
        )

    def _resume_state(self) -> ResumeState:
        last = self.checkpoint_dir / "last.pt"
        if not self.resume or not last.is_file():
            return ResumeState(0, None, [])
        state = restore_training_checkpoint(
            path=last,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
            early_stopping=self.early_stopping,
            expected_metadata=self.metadata,
        )
        self.logger.info("resumed diffusion training from epoch %d", state.start_epoch)
        return state

    def _ensure_inference_checkpoint(self) -> None:
        """Provide a deterministic fallback when every validation loss was invalid."""

        best = self.checkpoint_dir / "best.pt"
        last = self.checkpoint_dir / "last.pt"
        if (best.exists() and self.early_stopping.best is not None) or not last.exists():
            return
        payload = torch_load_cpu(last)
        payload["selection"] = "last_epoch_fallback_no_finite_validation_metric"
        atomic_torch_save(payload, best)
        self.logger.warning(
            "no finite validation metric was available; best.pt uses the final epoch"
        )

    def fit(
        self,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        loss_function: LossFunction,
    ) -> list[dict[str, float]]:
        state = self._resume_state()
        history = state.history
        if self.early_stopping.bad_epochs >= self.early_stopping.patience:
            self.logger.info("restored run had already reached early stopping")
            self._ensure_inference_checkpoint()
            return history
        for epoch in range(state.start_epoch, self.epochs):
            dataset = train_loader.dataset
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            train_metrics = self._epoch(train_loader, loss_function, training=True)
            with self.ema.average_parameters(self.model):
                validation_metrics = self._epoch(
                    validation_loader, loss_function, training=False
                )
            validation_loss = float(validation_metrics["loss"])
            improved, should_stop = self.early_stopping.update(validation_loss)
            record = {
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            self.logger.info(
                "epoch=%d train_loss=%.6f validation_loss=%.6f",
                epoch,
                train_metrics["loss"],
                validation_loss,
            )
            if improved:
                self._save(self.checkpoint_dir / "best.pt", epoch, history)

            # Scheduler state and last.pt both describe the start of the next epoch.
            self.scheduler.step()
            self._save(self.checkpoint_dir / "last.pt", epoch, history)
            if (epoch + 1) % self.checkpoint_every == 0:
                self._save(
                    self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", epoch, history
                )
            if should_stop:
                self.logger.info("early stopping after epoch %d", epoch)
                break
        self._ensure_inference_checkpoint()
        return history


def move_patients(
    batch: Sequence[PatientBags], device: torch.device
) -> list[PatientBags]:
    return [
        PatientBags(
            patient_id=patient.patient_id,
            real_slides=[
                slide.to(device, non_blocking=True) for slide in patient.real_slides
            ],
            generated_slides=[
                slide.to(device, non_blocking=True)
                for slide in patient.generated_slides
            ],
            time=patient.time,
            event=patient.event,
        )
        for patient in batch
    ]


class SurvivalTrainer:
    """Hierarchical C-MIL/Cox trainer with exact or batch-local risk sets."""

    def __init__(
        self,
        model: HierarchicalCMILSurvival,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        patience: int,
        optimizer_name: str,
        gradient_clip_norm: float,
        amp: bool,
        risk_set_mode: str,
        cox_ties: str,
        cox_reduction: str,
        checkpoint_dir: str | Path,
        logger: logging.Logger,
        metadata: Mapping[str, Any],
        checkpoint_every: int = 1,
        resume: bool = False,
    ) -> None:
        if risk_set_mode not in {"batch", "full_dataset"}:
            raise ValueError("risk_set_mode must be 'batch' or 'full_dataset'")
        if epochs < 1 or checkpoint_every < 1:
            raise ValueError("epochs and checkpoint_every must be positive")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.optimizer = build_optimizer(
            self.model, optimizer_name, learning_rate, weight_decay
        )
        self.scheduler = build_epoch_scheduler(
            self.optimizer, epochs, warmup_epochs
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.amp = bool(amp)
        self.risk_set_mode = risk_set_mode
        self.cox_ties = cox_ties
        self.cox_reduction = cox_reduction
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = checkpoint_every
        self.early_stopping = EarlyStopping(patience, mode="max")
        self.logger = logger
        self.metadata = _checkpoint_metadata(metadata)
        self.resume = resume

    def _all_risks(
        self,
        loader: DataLoader[Any],
        gradients: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[int]]:
        risks: list[torch.Tensor] = []
        times: list[float] = []
        events: list[int] = []
        patient_ids: list[str] = []
        original_counts: list[int] = []
        with torch.set_grad_enabled(gradients):
            for raw_batch in loader:
                batch = move_patients(raw_batch, self.device)
                if not batch:
                    continue
                with torch.cuda.amp.autocast(enabled=self.amp):
                    batch_risks = self.model(batch)
                if not bool(torch.isfinite(batch_risks).all()):
                    raise FloatingPointError("Survival model produced non-finite risk")
                risks.append(batch_risks)
                times.extend(float(patient.time) for patient in batch)
                events.extend(int(patient.event) for patient in batch)
                patient_ids.extend(patient.patient_id for patient in batch)
                original_counts.extend(len(patient.real_slides) for patient in batch)
        if not risks:
            raise ValueError("Survival DataLoader produced no patients")
        return (
            torch.cat(risks),
            torch.tensor(times, dtype=torch.float64, device=self.device),
            torch.tensor(events, dtype=torch.bool, device=self.device),
            patient_ids,
            original_counts,
        )

    def _train_full_dataset(self, loader: DataLoader[Any]) -> float:
        """Exact fold-level Cox gradient via deterministic two-pass replay.

        The first pass computes detached risks and dL/drisk for the complete
        training fold.  The second pass restores each microbatch RNG snapshot
        and injects that scalar gradient into a fresh forward graph, preserving
        exact risk sets without retaining every WSI graph simultaneously.
        """

        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        detached_risks: list[torch.Tensor] = []
        times: list[float] = []
        events: list[int] = []
        expected_ids: list[list[str]] = []
        rng_snapshots: list[tuple[torch.Tensor, list[torch.Tensor] | None]] = []
        with torch.no_grad():
            for raw_batch in loader:
                rng_snapshots.append(
                    (
                        torch.get_rng_state(),
                        (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                    )
                )
                batch = move_patients(raw_batch, self.device)
                expected_ids.append([patient.patient_id for patient in batch])
                with torch.cuda.amp.autocast(enabled=self.amp):
                    batch_risks = self.model(batch)
                if not bool(torch.isfinite(batch_risks).all()):
                    raise FloatingPointError("Survival model produced non-finite risk")
                detached_risks.append(batch_risks.detach().float().cpu())
                times.extend(float(patient.time) for patient in batch)
                events.extend(int(patient.event) for patient in batch)
        if not detached_risks:
            raise ValueError("Survival training loader produced no patients")
        if not any(events):
            raise ValueError("Full-fold Cox training requires an observed event")

        risk_leaf = torch.cat(detached_risks).requires_grad_(True)
        time_tensor = torch.tensor(times, dtype=torch.float64)
        event_tensor = torch.tensor(events, dtype=torch.bool)
        loss = negative_cox_partial_log_likelihood(
            risk_leaf,
            time_tensor,
            event_tensor,
            ties=self.cox_ties,
            reduction=self.cox_reduction,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite full-fold Cox loss")
        risk_gradient = torch.autograd.grad(loss, risk_leaf)[0].detach()

        offset = 0
        for batch_index, raw_batch in enumerate(loader):
            cpu_state, cuda_state = rng_snapshots[batch_index]
            torch.set_rng_state(cpu_state)
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            batch = move_patients(raw_batch, self.device)
            actual_ids = [patient.patient_id for patient in batch]
            if actual_ids != expected_ids[batch_index]:
                raise RuntimeError(
                    "full_dataset Cox replay order changed; disable DataLoader shuffling"
                )
            with torch.cuda.amp.autocast(enabled=self.amp):
                replayed_risk = self.model(batch)
                gradient = risk_gradient[offset : offset + len(batch)].to(replayed_risk)
                surrogate = torch.sum(replayed_risk * gradient)
            self.scaler.scale(surrogate).backward()
            offset += len(batch)
        if offset != risk_gradient.numel():
            raise AssertionError("Two-pass Cox replay did not cover every patient")
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return float(loss.detach())

    def _train_batches(self, loader: DataLoader[Any]) -> float:
        weighted_loss = 0.0
        examples = 0
        skipped_no_event = 0
        for raw_batch in loader:
            batch = move_patients(raw_batch, self.device)
            if not batch:
                continue
            events = torch.tensor(
                [patient.event for patient in batch], device=self.device
            ).bool()
            if not events.any():
                skipped_no_event += 1
                continue
            times = torch.tensor(
                [patient.time for patient in batch],
                dtype=torch.float64,
                device=self.device,
            )
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                risks = self.model(batch)
                loss = negative_cox_partial_log_likelihood(
                    risks,
                    times,
                    events,
                    ties=self.cox_ties,
                    reduction=self.cox_reduction,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite batch Cox loss")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            weighted_loss += float(loss.detach()) * len(batch)
            examples += len(batch)
        if examples == 0:
            raise ValueError(
                "No Cox batch contained an observed event; use larger batches or "
                "risk_set_mode='full_dataset'"
            )
        if skipped_no_event:
            self.logger.debug("skipped %d Cox batches without events", skipped_no_event)
        return weighted_loss / examples

    @torch.no_grad()
    def predict(self, loader: DataLoader[Any]) -> dict[str, Any]:
        self.model.eval()
        risks, times, events, patient_ids, counts = self._all_risks(
            loader, gradients=False
        )
        time_array = times.cpu().numpy()
        event_array = events.cpu().numpy().astype(int)
        risk_array = risks.float().cpu().numpy()
        result = concordance_index(time_array, event_array, risk_array)
        return {
            "patient_ids": patient_ids,
            "time": time_array,
            "event": event_array,
            "risk": risk_array,
            "original_slide_count": np.asarray(counts, dtype=int),
            "c_index": result.c_index,
            "comparable_pairs": result.comparable,
        }

    def _save(
        self,
        path: str | Path,
        epoch: int,
        history: list[dict[str, float]],
    ) -> None:
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=None,
            epoch=epoch,
            best_metric=self.early_stopping.best,
            history=history,
            metadata=self.metadata,
            early_stopping=self.early_stopping,
        )

    def _resume_state(self) -> ResumeState:
        last = self.checkpoint_dir / "last.pt"
        if not self.resume or not last.is_file():
            return ResumeState(0, None, [])
        state = restore_training_checkpoint(
            path=last,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=None,
            early_stopping=self.early_stopping,
            expected_metadata=self.metadata,
        )
        self.logger.info("resumed survival training from epoch %d", state.start_epoch)
        return state

    def _ensure_inference_checkpoint(self) -> None:
        best = self.checkpoint_dir / "best.pt"
        last = self.checkpoint_dir / "last.pt"
        if (best.exists() and self.early_stopping.best is not None) or not last.exists():
            return
        payload = torch_load_cpu(last)
        payload["selection"] = "last_epoch_fallback_no_finite_validation_c_index"
        atomic_torch_save(payload, best)
        self.logger.warning(
            "validation had no comparable pairs; best.pt uses the final epoch"
        )

    def fit(
        self,
        train_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
    ) -> list[dict[str, float]]:
        state = self._resume_state()
        history = state.history
        if self.early_stopping.bad_epochs >= self.early_stopping.patience:
            self.logger.info("restored run had already reached early stopping")
            self._ensure_inference_checkpoint()
            return history
        for epoch in range(state.start_epoch, self.epochs):
            self.model.train()
            train_loss = (
                self._train_full_dataset(train_loader)
                if self.risk_set_mode == "full_dataset"
                else self._train_batches(train_loader)
            )
            validation = self.predict(validation_loader)
            validation_c = float(validation["c_index"])
            improved, should_stop = self.early_stopping.update(validation_c)
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_c_index": validation_c,
                "validation_comparable_pairs": float(
                    validation["comparable_pairs"]
                ),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            self.logger.info(
                "epoch=%d train_cox=%.6f validation_c_index=%s pairs=%d",
                epoch,
                train_loss,
                f"{validation_c:.6f}" if math.isfinite(validation_c) else "undefined",
                int(validation["comparable_pairs"]),
            )
            if improved:
                self._save(self.checkpoint_dir / "best.pt", epoch, history)

            self.scheduler.step()
            self._save(self.checkpoint_dir / "last.pt", epoch, history)
            if (epoch + 1) % self.checkpoint_every == 0:
                self._save(
                    self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", epoch, history
                )
            if should_stop:
                self.logger.info("early stopping after epoch %d", epoch)
                break
        self._ensure_inference_checkpoint()
        return history


# ---------------------------------------------------------------------------
# Survival metrics and fold-level evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcordanceResult:
    c_index: float
    comparable: int
    concordant: int
    discordant: int
    tied_risk: int

    def to_dict(self) -> dict[str, float | int | None]:
        return _json_safe(asdict(self))


def _survival_arrays(
    time: np.ndarray | Sequence[float],
    event: np.ndarray | Sequence[int],
    risk: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(time, dtype=float).ravel()
    raw_events = np.asarray(event).ravel()
    risks = np.asarray(risk, dtype=float).ravel()
    if not (times.size == raw_events.size == risks.size):
        raise ValueError("time, event, and risk must have equal lengths")
    if not np.isfinite(times).all() or not np.isfinite(risks).all():
        raise ValueError("time and risk must contain only finite values")
    try:
        numeric_events = raw_events.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("event must contain binary 0/1 values") from error
    if not np.isfinite(numeric_events).all() or not np.isin(
        numeric_events, [0.0, 1.0]
    ).all():
        raise ValueError("event must contain binary 0/1 values")
    return times, numeric_events.astype(bool), risks


def concordance_index(
    time: np.ndarray | Sequence[float],
    event: np.ndarray | Sequence[int],
    risk: np.ndarray | Sequence[float],
) -> ConcordanceResult:
    """Dependency-free Harrell C-index where larger risk means earlier event."""

    times, events, risks = _survival_arrays(time, event, risk)
    concordant = 0
    discordant = 0
    tied = 0
    comparable = 0
    for left in range(times.size):
        for right in range(left + 1, times.size):
            if times[left] == times[right]:
                continue
            earlier, later = (
                (left, right)
                if times[left] < times[right]
                else (right, left)
            )
            if not events[earlier]:
                continue
            comparable += 1
            if risks[earlier] > risks[later]:
                concordant += 1
            elif risks[earlier] < risks[later]:
                discordant += 1
            else:
                tied += 1
    value = (
        (concordant + 0.5 * tied) / comparable
        if comparable
        else float("nan")
    )
    return ConcordanceResult(value, comparable, concordant, discordant, tied)


class BreslowBaseline:
    """Breslow baseline hazard for turning Cox risk into survival estimates."""

    def __init__(self) -> None:
        self.event_times: np.ndarray | None = None
        self.cumulative_hazard: np.ndarray | None = None
        self._risk_shift: float | None = None

    def fit(
        self,
        time: np.ndarray | Sequence[float],
        event: np.ndarray | Sequence[int],
        log_risk: np.ndarray | Sequence[float],
    ) -> "BreslowBaseline":
        times, events, risks = _survival_arrays(time, event, log_risk)
        if not times.size:
            raise ValueError("Breslow fitting requires at least one patient")
        if not events.any():
            raise ValueError("Breslow fitting requires at least one observed event")
        event_times = np.unique(times[events])
        risk_shift = float(risks.max())
        exp_risk = np.exp(risks - risk_shift)
        increments: list[float] = []
        for event_time in event_times:
            deaths = int(np.sum((times == event_time) & events))
            denominator = float(exp_risk[times >= event_time].sum())
            if denominator <= 0 or not math.isfinite(denominator):
                raise FloatingPointError("Invalid Breslow risk-set denominator")
            increments.append(deaths / denominator)
        self.event_times = event_times
        self.cumulative_hazard = np.cumsum(increments)
        self._risk_shift = risk_shift
        return self

    def _fitted(self) -> tuple[np.ndarray, np.ndarray, float]:
        if (
            self.event_times is None
            or self.cumulative_hazard is None
            or self._risk_shift is None
        ):
            raise RuntimeError("Call fit before requesting a survival estimate")
        return self.event_times, self.cumulative_hazard, self._risk_shift

    def median_survival_time(
        self, log_risk: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        event_times, cumulative_hazard, risk_shift = self._fitted()
        risks = np.asarray(log_risk, dtype=float).ravel()
        if not np.isfinite(risks).all():
            raise ValueError("log_risk must contain only finite values")
        target_hazard = np.log(2.0) / np.exp(risks - risk_shift)
        medians = np.full(risks.size, np.inf, dtype=float)
        for index, target in enumerate(target_hazard):
            crossing = np.flatnonzero(cumulative_hazard >= target)
            if crossing.size:
                medians[index] = event_times[crossing[0]]
        return medians

    def survival_probability(
        self, log_risk: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        _, cumulative_hazard, risk_shift = self._fitted()
        risks = np.asarray(log_risk, dtype=float).ravel()
        if not np.isfinite(risks).all():
            raise ValueError("log_risk must contain only finite values")
        return np.exp(-np.outer(np.exp(risks - risk_shift), cumulative_hazard))

    def state_dict(self) -> dict[str, Any]:
        event_times, cumulative_hazard, risk_shift = self._fitted()
        return {
            "event_times": event_times.tolist(),
            "cumulative_hazard": cumulative_hazard.tolist(),
            "risk_shift": risk_shift,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.event_times = np.asarray(state["event_times"], dtype=float)
        self.cumulative_hazard = np.asarray(
            state["cumulative_hazard"], dtype=float
        )
        self._risk_shift = float(state["risk_shift"])
        if self.event_times.size != self.cumulative_hazard.size:
            raise ValueError("Breslow state arrays must have equal lengths")


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    successful_samples: int

    def to_dict(self) -> dict[str, float | int | None]:
        return _json_safe(asdict(self))


def bootstrap_metric(
    arrays: tuple[np.ndarray, ...],
    metric: Callable[..., float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 2025,
) -> BootstrapInterval:
    """Compute a patient-level percentile-bootstrap confidence interval."""

    converted = tuple(np.asarray(array) for array in arrays)
    if not converted or len({len(array) for array in converted}) != 1:
        raise ValueError("All bootstrap arrays must have the same length")
    size = len(converted[0])
    if size == 0:
        raise ValueError("Bootstrap arrays cannot be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = generator.integers(0, size, size=size)
        try:
            value = float(metric(*(array[indices] for array in converted)))
        except (ValueError, FloatingPointError):
            continue
        if math.isfinite(value):
            values.append(value)
    try:
        estimate = float(metric(*converted))
    except (ValueError, FloatingPointError):
        estimate = float("nan")
    if not values:
        return BootstrapInterval(estimate, float("nan"), float("nan"), 0)
    alpha = (1 - confidence) / 2
    return BootstrapInterval(
        estimate=estimate,
        lower=float(np.quantile(values, alpha)),
        upper=float(np.quantile(values, 1 - alpha)),
        successful_samples=len(values),
    )


@dataclass(frozen=True)
class Stage5Result:
    accuracy: float
    evaluated: int
    excluded_censored: int
    time_boundaries: list[float]
    risk_boundaries: list[float]
    definition: str = (
        "training-event time quintiles and reverse training-risk quintiles"
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _exclude_censored(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"exclude", "excluded", "observed_only", "true"}:
        return True
    if normalized in {"include", "included", "all", "false"}:
        return False
    raise ValueError(
        "censored policy must be a boolean or one of exclude/include/observed_only/all"
    )


class QuantileStageMetric:
    """Five-stage accuracy using thresholds fitted on the training fold only."""

    def __init__(
        self,
        num_bins: int = 5,
        exclude_censored: bool | str = True,
    ) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least two")
        self.num_bins = int(num_bins)
        self.exclude_censored = _exclude_censored(exclude_censored)
        self.time_boundaries: np.ndarray | None = None
        self.risk_boundaries: np.ndarray | None = None

    def fit(
        self,
        train_time: np.ndarray,
        train_event: np.ndarray,
        train_risk: np.ndarray,
    ) -> "QuantileStageMetric":
        times, events, risks = _survival_arrays(
            train_time, train_event, train_risk
        )
        observed_times = times[events]
        if observed_times.size < self.num_bins:
            raise ValueError(
                f"At least {self.num_bins} observed training events are required"
            )
        quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]
        self.time_boundaries = np.quantile(observed_times, quantiles)
        self.risk_boundaries = np.quantile(risks, quantiles)
        return self

    def _fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.time_boundaries is None or self.risk_boundaries is None:
            raise RuntimeError("Call fit before score")
        return self.time_boundaries, self.risk_boundaries

    def score(
        self,
        time: np.ndarray,
        event: np.ndarray,
        risk: np.ndarray,
    ) -> Stage5Result:
        time_boundaries, risk_boundaries = self._fitted()
        times, events, risks = _survival_arrays(time, event, risk)
        eligible = events if self.exclude_censored else np.ones_like(events)
        true_stage = np.digitize(
            times[eligible], time_boundaries, right=True
        )
        predicted_stage = self.num_bins - 1 - np.digitize(
            risks[eligible], risk_boundaries, right=True
        )
        accuracy = (
            float(np.mean(true_stage == predicted_stage))
            if eligible.any()
            else float("nan")
        )
        return Stage5Result(
            accuracy=accuracy,
            evaluated=int(eligible.sum()),
            excluded_censored=int((~eligible).sum()),
            time_boundaries=time_boundaries.tolist(),
            risk_boundaries=risk_boundaries.tolist(),
        )

    def state_dict(self) -> dict[str, Any]:
        time_boundaries, risk_boundaries = self._fitted()
        return {
            "num_bins": self.num_bins,
            "exclude_censored": self.exclude_censored,
            "time_boundaries": time_boundaries.tolist(),
            "risk_boundaries": risk_boundaries.tolist(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["num_bins"]) != self.num_bins:
            raise ValueError("Stage metric bin count differs from saved state")
        self.exclude_censored = bool(state["exclude_censored"])
        self.time_boundaries = np.asarray(state["time_boundaries"], dtype=float)
        self.risk_boundaries = np.asarray(state["risk_boundaries"], dtype=float)


def fixed_time_stage_accuracy(
    observed_time: np.ndarray,
    event: np.ndarray,
    predicted_time: np.ndarray,
    boundaries: Sequence[float] = (365, 730, 1095, 1460),
    exclude_censored: bool | str = True,
) -> Stage5Result:
    """Evaluate five survival stages separated by fixed time boundaries."""

    times = np.asarray(observed_time, dtype=float).ravel()
    raw_events = np.asarray(event).ravel()
    predictions = np.asarray(predicted_time, dtype=float).ravel()
    if not (times.size == raw_events.size == predictions.size):
        raise ValueError("observed_time, event, and predicted_time need equal lengths")
    if not np.isfinite(times).all() or np.isnan(predictions).any():
        raise ValueError("Times cannot contain NaN; observed times must be finite")
    try:
        numeric_events = raw_events.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("event must contain binary 0/1 values") from error
    if not np.isin(numeric_events, [0.0, 1.0]).all():
        raise ValueError("event must contain binary 0/1 values")
    events = numeric_events.astype(bool)
    boundary_array = np.asarray(boundaries, dtype=float).ravel()
    if (
        not np.isfinite(boundary_array).all()
        or np.any(np.diff(boundary_array) <= 0)
    ):
        raise ValueError("boundaries must be finite and strictly increasing")
    exclude = _exclude_censored(exclude_censored)
    eligible = events if exclude else np.ones_like(events)
    true_stage = np.digitize(times[eligible], boundary_array, right=False)
    predicted_stage = np.digitize(
        predictions[eligible], boundary_array, right=False
    )
    accuracy = (
        float(np.mean(true_stage == predicted_stage))
        if eligible.any()
        else float("nan")
    )
    return Stage5Result(
        accuracy=accuracy,
        evaluated=int(eligible.sum()),
        excluded_censored=int((~eligible).sum()),
        time_boundaries=boundary_array.tolist(),
        risk_boundaries=[],
        definition="fixed survival-time intervals with Cox-Breslow median prediction",
    )


def slide_count_bias_metrics(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    slide_count: np.ndarray,
    group_bounds: Mapping[str, Sequence[int | None]] | None = None,
) -> dict[str, float | int | None]:
    """Report C-index for low-, medium-, and high-WSI-count strata."""

    times, events, risks = _survival_arrays(time, event, risk)
    counts = np.asarray(slide_count, dtype=int).ravel()
    if len(counts) != len(times):
        raise ValueError("slide_count must have one value per patient")
    if (counts < 0).any():
        raise ValueError("slide_count cannot be negative")
    bounds = group_bounds or {
        "low": (0, 2),
        "medium": (3, 3),
        "high": (4, None),
    }
    output: dict[str, float | int | None] = {}
    finite_scores: list[float] = []
    for name, limits in bounds.items():
        if len(limits) != 2:
            raise ValueError(f"Bias group {name!r} must have [minimum, maximum]")
        minimum, maximum = limits
        mask = counts >= int(minimum or 0)
        if maximum is not None:
            mask &= counts <= int(maximum)
        result = concordance_index(times[mask], events[mask], risks[mask])
        score = result.c_index
        output[f"{name}_patients"] = int(mask.sum())
        output[f"{name}_comparable_pairs"] = result.comparable
        output[f"{name}_c_index"] = score if math.isfinite(score) else None
        if math.isfinite(score):
            finite_scores.append(score)
    output["gap"] = (
        max(finite_scores) - min(finite_scores)
        if len(finite_scores) >= 2
        else None
    )
    return output


def aggregate_metric_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Aggregate per-fold scalars as mean and sample standard deviation."""

    if not paths:
        raise ValueError("At least one fold metric file is required")
    payloads: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or "fold" not in payload:
            raise ValueError(f"Metric file has no fold mapping: {path}")
        payloads.append(payload)
    fold_ids = [int(payload["fold"]) for payload in payloads]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError(f"Cannot aggregate duplicate fold IDs: {fold_ids}")
    scalar_keys = sorted(
        {
            key
            for payload in payloads
            for key, value in payload.items()
            if (
                key in {"c_index", "stage5_accuracy", "gap"}
                or key.endswith("_c_index")
                or key.endswith("_accuracy")
            )
            and not isinstance(value, bool)
            and (value is None or isinstance(value, (int, float)))
        }
    )
    output: dict[str, Any] = {
        "folds": len(payloads),
        "fold_ids": sorted(fold_ids),
        "per_fold": payloads,
    }
    for key in scalar_keys:
        values = np.asarray(
            [
                float(payload[key])
                for payload in payloads
                if key in payload
                and payload[key] is not None
                and math.isfinite(float(payload[key]))
            ],
            dtype=float,
        )
        output[key] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": (
                float(values.std(ddof=1))
                if len(values) > 1
                else (0.0 if len(values) else None)
            ),
            "valid_folds": int(len(values)),
        }
    return output


def write_aggregate(
    paths: Sequence[str | Path], output_path: str | Path
) -> dict[str, Any]:
    aggregate = aggregate_metric_files(paths)
    atomic_json_dump(aggregate, output_path)
    return aggregate


def evaluate_survival_predictions(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    slide_count: np.ndarray | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    seed: int = 2025,
    group_bounds: Mapping[str, Sequence[int | None]] | None = None,
) -> dict[str, Any]:
    """Build the common C-index, confidence-interval, and bias report."""

    times, events, risks = _survival_arrays(time, event, risk)
    concordance = concordance_index(times, events, risks)
    interval = bootstrap_metric(
        (times, events, risks),
        lambda sampled_time, sampled_event, sampled_risk: concordance_index(
            sampled_time, sampled_event, sampled_risk
        ).c_index,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=seed,
    )
    report: dict[str, Any] = {
        **concordance.to_dict(),
        "c_index_ci_lower": (
            interval.lower if math.isfinite(interval.lower) else None
        ),
        "c_index_ci_upper": (
            interval.upper if math.isfinite(interval.upper) else None
        ),
        "c_index_bootstrap_successful_samples": interval.successful_samples,
    }
    if slide_count is not None:
        report.update(
            slide_count_bias_metrics(
                times,
                events,
                risks,
                np.asarray(slide_count),
                group_bounds=group_bounds,
            )
        )
    return report


__all__ = [
    "BootstrapInterval",
    "BreslowBaseline",
    "ConcordanceResult",
    "DiffusionTrainer",
    "EarlyStopping",
    "ExponentialMovingAverage",
    "QuantileStageMetric",
    "ResumeState",
    "Stage5Result",
    "SurvivalTrainer",
    "aggregate_metric_files",
    "amp_enabled",
    "atomic_json_dump",
    "atomic_torch_save",
    "bootstrap_metric",
    "build_epoch_scheduler",
    "build_optimizer",
    "checkpoint_path",
    "collect_provenance",
    "concordance_index",
    "configure_logging",
    "evaluate_survival_predictions",
    "fixed_time_stage_accuracy",
    "fold_directory",
    "load_model_checkpoint",
    "make_generator",
    "mark_stage_complete",
    "move_patients",
    "project_path",
    "remove_stage_marker",
    "resolve_device",
    "restore_rng_state",
    "restore_training_checkpoint",
    "rng_state",
    "save_checkpoint",
    "seed_everything",
    "seed_worker",
    "slide_count_bias_metrics",
    "stage_is_complete",
    "stage_marker_path",
    "torch_load",
    "torch_load_cpu",
    "torch_load_restricted_cpu",
    "write_aggregate",
    "write_provenance",
]
