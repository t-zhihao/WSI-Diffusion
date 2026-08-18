"""Data contracts, cohort handling, feature storage, and WSI preprocessing.

The official implementation keeps the complete data boundary in this module so
that every stage uses the same manifest semantics, feature schema, fold checks,
and preprocessing conventions.  Raw slides are never required at training time:
``preprocess_slide`` converts them into compact, versioned feature bundles.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared constants and typed records
# ---------------------------------------------------------------------------

TISSUE_CLASSES: tuple[str, ...] = (
    "neoplastic",
    "dead",
    "inflammatory",
    "non_neoplastic_epithelial",
    "connective",
    "no_label",
)
TISSUE_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(TISSUE_CLASSES)
}
INDEX_TO_TISSUE: dict[int, str] = {
    index: name for name, index in TISSUE_TO_INDEX.items()
}
REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "dataset",
    "patient_id",
    "slide_id",
    "feature_path",
    "time",
    "event",
)
SUPPORTED_SLIDE_EXTENSIONS: tuple[str, ...] = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
)
FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SlideRecord:
    """One manifest row after path and endpoint normalization."""

    dataset: str
    patient_id: str
    slide_id: str
    feature_path: Path
    time: float
    event: int
    wsi_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientRecord:
    """Patient-level clinical label and all associated slides."""

    patient_id: str
    slides: tuple[SlideRecord, ...]
    time: float
    event: int


@dataclass
class FeatureBundle:
    """Canonical feature artifact exchanged by all model stages."""

    wsi_feature: torch.Tensor
    patch_features: torch.Tensor
    tissue_labels: torch.Tensor
    coordinates: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        wsi_dim: int | None = None,
        patch_dim: int | None = None,
    ) -> None:
        if self.wsi_feature.ndim != 1:
            raise ValueError(
                f"wsi_feature must be [D], got {tuple(self.wsi_feature.shape)}"
            )
        if self.patch_features.ndim != 2:
            raise ValueError(
                "patch_features must be [N, D], got "
                f"{tuple(self.patch_features.shape)}"
            )
        if self.patch_features.shape[0] < 1:
            raise ValueError("patch_features must contain at least one tissue patch")
        if self.tissue_labels.ndim != 1:
            raise ValueError(
                f"tissue_labels must be [N], got {tuple(self.tissue_labels.shape)}"
            )
        if self.patch_features.shape[0] != self.tissue_labels.shape[0]:
            raise ValueError("patch_features and tissue_labels have different patch counts")
        if self.coordinates is not None:
            if self.coordinates.ndim != 2 or self.coordinates.shape != (
                self.patch_features.shape[0],
                2,
            ):
                raise ValueError("coordinates must have shape [N, 2]")
        if wsi_dim is not None and self.wsi_feature.shape[0] != wsi_dim:
            raise ValueError(
                f"expected WSI dimension {wsi_dim}, got {self.wsi_feature.shape[0]}"
            )
        if patch_dim is not None and self.patch_features.shape[1] != patch_dim:
            raise ValueError(
                f"expected patch dimension {patch_dim}, got {self.patch_features.shape[1]}"
            )


@dataclass(frozen=True)
class SurvivalPrediction:
    patient_id: str
    time: float
    event: int
    risk: float
    original_slide_count: int
    generated_slide_count: int
    fold: int


@dataclass
class PatientBags:
    """Variable-length real and generated slide bags for one patient."""

    patient_id: str
    real_slides: list[torch.Tensor]
    generated_slides: list[torch.Tensor]
    time: float
    event: int


# ---------------------------------------------------------------------------
# Small atomic I/O helpers used by feature and split artifacts
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    """Write valid UTF-8 JSON and atomically replace the destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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
    """Write a PyTorch artifact through a same-directory temporary file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(handle)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def torch_load_cpu(path: str | Path) -> Any:
    """Load a trusted project artifact on CPU across supported PyTorch versions."""

    source = Path(path)
    try:
        return torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(source, map_location="cpu")


def torch_load_restricted_cpu(path: str | Path) -> Any:
    """Use PyTorch's restricted tensor loader when the installed version has it."""

    source = Path(path)
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        warnings.warn(
            "This PyTorch version predates restricted tensor loading. Only open "
            f"feature files produced by a trusted pipeline: {source}",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(source, map_location="cpu")


# ---------------------------------------------------------------------------
# Manifest and patient-level split contracts
# ---------------------------------------------------------------------------

def _resolve_optional_path(value: object, root: Path) -> Path | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def read_manifest(
    path: str | Path,
    project_root: str | Path | None = None,
    event_semantics: str = "observed",
    expected_dataset: str | None = None,
) -> list[SlideRecord]:
    """Read a one-row-per-slide CSV and normalize to ``event=1`` observed.

    ``event_semantics='censored'`` interprets an input value of one as censored
    and flips it exactly once at this data boundary.
    """

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else manifest_path.parent
    )
    frame = pd.read_csv(manifest_path, dtype={"patient_id": str, "slide_id": str})
    missing = sorted(set(REQUIRED_MANIFEST_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Manifest contains no slides")
    for identifier in ("patient_id", "slide_id"):
        if frame[identifier].isna().any():
            raise ValueError(f"{identifier} must be non-empty for every manifest row")
        frame[identifier] = frame[identifier].astype(str).str.strip()
        if (frame[identifier] == "").any():
            raise ValueError(f"{identifier} must be non-empty for every manifest row")
    if frame["slide_id"].duplicated().any():
        duplicates = frame.loc[frame["slide_id"].duplicated(), "slide_id"].tolist()
        raise ValueError(
            f"slide_id values must be unique; duplicates include {duplicates[:5]}"
        )
    if event_semantics not in {"observed", "censored"}:
        raise ValueError("event_semantics must be 'observed' or 'censored'")
    if frame["dataset"].isna().any() or (
        frame["dataset"].astype(str).str.strip() == ""
    ).any():
        raise ValueError("dataset must be non-empty for every manifest row")
    datasets = {str(value).strip() for value in frame["dataset"]}
    if len(datasets) != 1:
        raise ValueError(
            f"Manifest must contain exactly one dataset/cohort, got {sorted(datasets)}"
        )
    manifest_dataset = next(iter(datasets))
    if expected_dataset is not None:
        expected = str(expected_dataset).strip()
        if not expected:
            raise ValueError("expected_dataset must be non-empty when provided")
        if manifest_dataset != expected:
            raise ValueError(
                f"Manifest dataset {manifest_dataset!r} does not match "
                f"expected dataset {expected!r}"
            )

    records: list[SlideRecord] = []
    feature_owners: dict[Path, tuple[str, str]] = {}
    wsi_owners: dict[Path, tuple[str, str]] = {}
    reserved = set(REQUIRED_MANIFEST_COLUMNS).union({"wsi_path"})
    for row in frame.to_dict(orient="records"):
        slide_id = str(row["slide_id"])
        patient_id = str(row["patient_id"])
        try:
            numeric_event = float(row["event"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"event must be binary for slide {slide_id}") from error
        if not math.isfinite(numeric_event) or numeric_event not in {0.0, 1.0}:
            raise ValueError(f"event must be binary for slide {slide_id}")
        raw_event = int(numeric_event)
        event = raw_event if event_semantics == "observed" else 1 - raw_event
        try:
            time = float(row["time"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"time must be numeric for slide {slide_id}") from error
        if not math.isfinite(time) or time <= 0:
            raise ValueError(f"time must be finite and positive for slide {slide_id}")
        feature_path = _resolve_optional_path(row["feature_path"], root)
        if feature_path is None:
            raise ValueError(f"feature_path is empty for slide {slide_id}")
        owner = (patient_id, slide_id)
        previous_feature_owner = feature_owners.get(feature_path)
        if previous_feature_owner is not None and previous_feature_owner != owner:
            raise ValueError(
                "Resolved feature_path is reused by different manifest rows: "
                f"{feature_path} belongs to {previous_feature_owner} and {owner}"
            )
        feature_owners[feature_path] = owner
        wsi_path = _resolve_optional_path(row.get("wsi_path"), root)
        if wsi_path is not None:
            previous_wsi_owner = wsi_owners.get(wsi_path)
            if previous_wsi_owner is not None and previous_wsi_owner != owner:
                raise ValueError(
                    "Resolved wsi_path is reused by different manifest rows: "
                    f"{wsi_path} belongs to {previous_wsi_owner} and {owner}"
                )
            if wsi_path.suffix.lower() not in SUPPORTED_SLIDE_EXTENSIONS:
                warnings.warn(
                    f"Unrecognized WSI extension for {slide_id}: {wsi_path.suffix}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            wsi_owners[wsi_path] = owner
        records.append(
            SlideRecord(
                dataset=manifest_dataset,
                patient_id=patient_id,
                slide_id=slide_id,
                feature_path=feature_path,
                wsi_path=wsi_path,
                time=time,
                event=event,
                extra={key: value for key, value in row.items() if key not in reserved},
            )
        )
    group_patients(records)
    return records


def group_patients(records: Sequence[SlideRecord]) -> list[PatientRecord]:
    grouped: dict[str, list[SlideRecord]] = defaultdict(list)
    for record in records:
        grouped[record.patient_id].append(record)
    patients: list[PatientRecord] = []
    for patient_id, slides in sorted(grouped.items()):
        labels = {(slide.time, slide.event) for slide in slides}
        if len(labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has inconsistent survival labels "
                f"across slides: {labels}"
            )
        time, event = next(iter(labels))
        patients.append(
            PatientRecord(
                patient_id=patient_id,
                slides=tuple(sorted(slides, key=lambda item: item.slide_id)),
                time=time,
                event=event,
            )
        )
    return patients


def validate_feature_paths(records: Iterable[SlideRecord]) -> list[str]:
    """Return all missing feature paths without mutating the manifest."""

    return [str(record.feature_path) for record in records if not record.feature_path.is_file()]


def _strata(patients: Sequence[PatientRecord], time_bins: int) -> np.ndarray:
    times = np.asarray([patient.time for patient in patients], dtype=float)
    events = np.asarray([patient.event for patient in patients], dtype=int)
    quantiles = np.unique(np.quantile(times, np.linspace(0, 1, time_bins + 1)))
    bins = np.digitize(times, quantiles[1:-1], right=True)
    return np.asarray(
        [f"{event}:{time_bin}" for event, time_bin in zip(events, bins)]
    )


def _partition_ids(fold: Mapping[str, Any], partition: str) -> list[str]:
    if partition not in fold:
        raise ValueError(f"Fold {fold.get('fold')} is missing partition {partition!r}")
    value = fold[partition]
    if not isinstance(value, list):
        raise ValueError(
            f"Fold {fold.get('fold')} partition {partition!r} must be a list"
        )
    if any(not isinstance(patient_id, str) or not patient_id.strip() for patient_id in value):
        raise ValueError(
            f"Fold {fold.get('fold')} partition {partition!r} contains "
            "a non-string or empty ID"
        )
    counts = Counter(value)
    duplicates = sorted(
        patient_id for patient_id, count in counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Fold {fold.get('fold')} partition {partition!r} contains duplicate "
            f"patient IDs: {duplicates[:5]}"
        )
    return value


def assert_patient_disjoint(fold: Mapping[str, Any]) -> None:
    if not isinstance(fold, Mapping):
        raise ValueError("Each fold must be a mapping")
    train = set(_partition_ids(fold, "train"))
    validation = set(_partition_ids(fold, "validation"))
    test = set(_partition_ids(fold, "test"))
    if train & validation or train & test or validation & test:
        raise ValueError(f"Patient leakage detected in fold {fold.get('fold')}")


def validate_split_payload(
    payload: Mapping[str, Any],
    expected_patient_ids: Iterable[str] | None = None,
) -> None:
    """Validate disjoint folds and the once-only outer-test invariant."""

    if not isinstance(payload, Mapping):
        raise ValueError("Split payload must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported split schema_version {payload.get('schema_version')!r}; expected 1"
        )
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Split payload must contain a non-empty folds list")
    declared_n_folds = payload.get("n_folds")
    if declared_n_folds is not None:
        if isinstance(declared_n_folds, bool) or not isinstance(declared_n_folds, int):
            raise ValueError("Split payload n_folds must be an integer")
        if declared_n_folds != len(folds):
            raise ValueError(
                f"Split payload declares n_folds={declared_n_folds}, but contains "
                f"{len(folds)} folds"
            )

    expected_list = None if expected_patient_ids is None else list(expected_patient_ids)
    if expected_list is not None:
        expected_counts = Counter(expected_list)
        expected_duplicates = sorted(
            patient_id
            for patient_id, count in expected_counts.items()
            if count > 1
        )
        if expected_duplicates:
            raise ValueError(
                "Expected patient universe contains duplicate IDs: "
                f"{expected_duplicates[:5]}"
            )
        expected_universe: set[str] | None = set(expected_list)
    else:
        expected_universe = None

    fold_numbers: set[int] = set()
    outer_test_counts: Counter[str] = Counter()
    for position, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise ValueError(f"Fold at position {position} must be a mapping")
        fold_number = fold.get("fold")
        if isinstance(fold_number, bool) or not isinstance(fold_number, int):
            raise ValueError(f"Fold at position {position} has an invalid fold identifier")
        if fold_number in fold_numbers:
            raise ValueError(f"Split payload contains duplicate fold identifier {fold_number}")
        fold_numbers.add(fold_number)
        assert_patient_disjoint(fold)
        train = set(_partition_ids(fold, "train"))
        validation = set(_partition_ids(fold, "validation"))
        test_ids = _partition_ids(fold, "test")
        universe = train | validation | set(test_ids)
        if not universe:
            raise ValueError(f"Fold {fold_number} contains no patients")
        if not train or not validation or not test_ids:
            raise ValueError(
                f"Fold {fold_number} must have non-empty train, validation, and test sets"
            )
        if expected_universe is None:
            expected_universe = universe
        if universe != expected_universe:
            missing = sorted(expected_universe.difference(universe))
            unexpected = sorted(universe.difference(expected_universe))
            raise ValueError(
                f"Fold {fold_number} patient universe differs from the cohort; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        outer_test_counts.update(test_ids)

    assert expected_universe is not None
    missing_test = sorted(expected_universe.difference(outer_test_counts))
    repeated_test = sorted(
        patient_id
        for patient_id, count in outer_test_counts.items()
        if patient_id in expected_universe and count != 1
    )
    unexpected_test = sorted(set(outer_test_counts).difference(expected_universe))
    if missing_test or repeated_test or unexpected_test:
        raise ValueError(
            "Across outer folds, every cohort patient must occur in test exactly once; "
            f"missing={missing_test[:5]}, repeated={repeated_test[:5]}, "
            f"unexpected={unexpected_test[:5]}"
        )


def make_patient_splits(
    patients: Sequence[PatientRecord],
    n_folds: int = 5,
    validation_fraction: float = 0.10,
    time_bins: int = 5,
    seed: int = 2025,
) -> dict[str, Any]:
    """Build patient-disjoint outer CV and fold-local validation splits."""

    # Split construction is the only data operation that needs scikit-learn;
    # importing locally keeps feature loading and WSI preprocessing lightweight.
    from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if len(patients) < n_folds:
        raise ValueError(
            f"n_folds={n_folds} exceeds the {len(patients)} available patients"
        )
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if time_bins < 1:
        raise ValueError("time_bins must be positive")
    patient_id_list = [patient.patient_id for patient in patients]
    duplicate_ids = sorted(
        patient_id
        for patient_id, count in Counter(patient_id_list).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            f"Patient records contain duplicate patient_id values: {duplicate_ids[:5]}"
        )
    patient_ids = np.asarray(patient_id_list)
    labels = _strata(patients, time_bins)
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    can_stratify = bool(counts) and min(counts.values()) >= n_folds
    if can_stratify:
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        outer_splits = outer.split(patient_ids, labels)
    else:
        outer = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        outer_splits = outer.split(patient_ids)

    folds: list[dict[str, Any]] = []
    for fold_number, (train_val_index, test_index) in enumerate(outer_splits):
        train_val_ids = patient_ids[train_val_index]
        train_val_labels = labels[train_val_index]
        if len(train_val_ids) < 2:
            raise ValueError("Each outer training set needs at least two patients")
        validation_size = max(1, round(len(train_val_ids) * validation_fraction))
        validation_size = min(validation_size, len(train_val_ids) - 1)
        inner_counts = [
            int(np.sum(train_val_labels == label))
            for label in np.unique(train_val_labels)
        ]
        inner_stratified = (
            min(inner_counts, default=0) >= 2
            and validation_size >= len(inner_counts)
            and len(train_val_ids) - validation_size >= len(inner_counts)
        )
        if inner_stratified:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=validation_size,
                random_state=seed + fold_number,
            )
            train_subindex, validation_subindex = next(
                splitter.split(train_val_ids, train_val_labels)
            )
        else:
            generator = np.random.default_rng(seed + fold_number)
            order = generator.permutation(len(train_val_ids))
            validation_subindex = order[:validation_size]
            train_subindex = order[validation_size:]
        fold_payload = {
            "fold": fold_number,
            "train": sorted(train_val_ids[train_subindex].tolist()),
            "validation": sorted(train_val_ids[validation_subindex].tolist()),
            "test": sorted(patient_ids[test_index].tolist()),
        }
        assert_patient_disjoint(fold_payload)
        folds.append(fold_payload)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "n_folds": n_folds,
        "validation_fraction": validation_fraction,
        "time_bins": time_bins,
        "stratification": "event_x_time_quantile" if can_stratify else "fallback_kfold",
        "folds": folds,
    }
    validate_split_payload(payload, expected_patient_ids=patient_id_list)
    return payload


def save_splits(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_split_payload(payload)
    atomic_json_dump(dict(payload), path)


def load_fold(
    path: str | Path,
    fold: int,
    expected_n_folds: int | None = None,
    expected_dataset: str | None = None,
    expected_seed: int | None = None,
    expected_validation_fraction: float | None = None,
    expected_time_bins: int | None = None,
) -> dict[str, Any]:
    """Load one fold while checking all configuration-critical metadata."""

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    validate_split_payload(payload)
    if expected_n_folds is not None and int(payload.get("n_folds", -1)) != expected_n_folds:
        raise ValueError(
            f"Split declares n_folds={payload.get('n_folds')}, expected {expected_n_folds}"
        )
    if expected_dataset is not None and payload.get("dataset") != expected_dataset:
        raise ValueError(
            f"Split dataset {payload.get('dataset')!r} does not match "
            f"{expected_dataset!r}"
        )
    if expected_seed is not None and int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(f"Split seed {payload.get('seed')} does not match {expected_seed}")
    if expected_validation_fraction is not None and not math.isclose(
        float(payload.get("validation_fraction", -1)),
        expected_validation_fraction,
        rel_tol=0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Split validation_fraction does not match the current config")
    if expected_time_bins is not None and int(payload.get("time_bins", -1)) != expected_time_bins:
        raise ValueError("Split time_bins does not match the current config")
    matches = [item for item in payload["folds"] if int(item["fold"]) == fold]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one fold {fold} in {path}")
    assert_patient_disjoint(matches[0])
    return dict(matches[0])


def partition_patients(
    patients: Sequence[PatientRecord],
    fold: Mapping[str, Any],
) -> dict[str, list[PatientRecord]]:
    """Map a validated fold to records, rejecting omissions and unknown IDs."""

    assert_patient_disjoint(fold)
    patient_counts = Counter(patient.patient_id for patient in patients)
    duplicate_patients = sorted(
        patient_id for patient_id, count in patient_counts.items() if count > 1
    )
    if duplicate_patients:
        raise ValueError(
            f"Manifest patients contain duplicate IDs: {duplicate_patients[:5]}"
        )
    by_id = {patient.patient_id: patient for patient in patients}
    partition_ids = {
        name: list(_partition_ids(fold, name))
        for name in ("train", "validation", "test")
    }
    fold_universe = set().union(*(set(ids) for ids in partition_ids.values()))
    manifest_universe = set(by_id)
    missing_from_fold = sorted(manifest_universe.difference(fold_universe))
    unknown_to_manifest = sorted(fold_universe.difference(manifest_universe))
    if missing_from_fold or unknown_to_manifest:
        raise ValueError(
            "Fold partitions must contain every manifest patient exactly once; "
            f"missing={missing_from_fold[:5]}, unknown={unknown_to_manifest[:5]}"
        )
    return {
        name: [by_id[patient_id] for patient_id in ids]
        for name, ids in partition_ids.items()
    }


# ---------------------------------------------------------------------------
# Feature bundle storage and train-fold normalization
# ---------------------------------------------------------------------------

class FeatureBundleCache:
    """Small per-process LRU cache for random patch sampling."""

    def __init__(
        self,
        capacity: int,
        wsi_dim: int | None,
        patch_dim: int | None,
        num_classes: int | None = None,
    ) -> None:
        self.capacity = max(0, int(capacity))
        self.wsi_dim = wsi_dim
        self.patch_dim = patch_dim
        self.num_classes = num_classes
        self._values: OrderedDict[Path, FeatureBundle] = OrderedDict()

    def get(self, path: str | Path) -> FeatureBundle:
        key = Path(path).resolve()
        if key in self._values:
            value = self._values.pop(key)
            self._values[key] = value
            return value
        value = load_feature_bundle(
            key,
            self.wsi_dim,
            self.patch_dim,
            self.num_classes,
        )
        if self.capacity:
            self._values[key] = value
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)
        return value

    def clear(self) -> None:
        self._values.clear()


def _tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype).contiguous()
    return torch.as_tensor(value, dtype=dtype).contiguous()


def _tissue_label_tensor(value: Any, source: Path) -> torch.Tensor:
    raw = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if raw.dtype == torch.bool or raw.is_complex():
        raise ValueError(f"tissue_labels in {source} must contain integer class indices")
    if raw.is_floating_point():
        if not bool(torch.isfinite(raw).all()) or not bool(torch.equal(raw, raw.round())):
            raise ValueError(
                f"tissue_labels in {source} must contain finite integer values"
            )
    return raw.to(dtype=torch.long).contiguous()


def _schema_scalar(value: Any, source: Path) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Feature schema_version in {source} must be a scalar integer"
            )
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"Feature schema_version in {source} must be a scalar integer"
            )
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Feature schema_version in {source} must be a scalar integer")
    return int(value)


def _validated_metadata(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    raw_metadata = payload.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        raise TypeError(f"Feature metadata in {source} must be a mapping")
    nested_version = metadata.get("schema_version")
    top_level_version = payload.get("schema_version")
    if nested_version is None and top_level_version is None:
        raise ValueError(f"Feature file {source} is missing schema_version")
    nested = None if nested_version is None else _schema_scalar(nested_version, source)
    top_level = (
        None if top_level_version is None else _schema_scalar(top_level_version, source)
    )
    if nested is not None and top_level is not None and nested != top_level:
        raise ValueError(
            f"Feature file {source} has conflicting schema versions {nested} and {top_level}"
        )
    version = nested if nested is not None else top_level
    if version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported feature schema_version {version} in {source}; "
            f"expected {FEATURE_SCHEMA_VERSION}"
        )
    metadata["schema_version"] = FEATURE_SCHEMA_VERSION
    return metadata


def _validate_finite_features(bundle: FeatureBundle, source: Path) -> None:
    for name, values in (
        ("wsi_feature", bundle.wsi_feature),
        ("patch_features", bundle.patch_features),
    ):
        if not bool(torch.isfinite(values).all()):
            invalid = int((~torch.isfinite(values)).sum().item())
            raise ValueError(f"{name} in {source} contains {invalid} non-finite values")


def validate_tissue_labels(labels: torch.Tensor, num_classes: int) -> None:
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if labels.numel() == 0:
        raise ValueError("A slide must contain at least one patch label")
    minimum, maximum = int(labels.min()), int(labels.max())
    if minimum < 0 or maximum >= num_classes:
        raise ValueError(
            f"Tissue labels must lie in [0, {num_classes - 1}], "
            f"got [{minimum}, {maximum}]"
        )


def load_feature_bundle(
    path: str | Path,
    wsi_dim: int | None = None,
    patch_dim: int | None = None,
    num_classes: int | None = None,
) -> FeatureBundle:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Feature bundle does not exist: {source}")
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            payload: Mapping[str, Any] = {key: archive[key] for key in archive.files}
    else:
        payload = torch_load_restricted_cpu(source)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Feature file must contain a mapping: {source}")
    required = {"wsi_feature", "patch_features", "tissue_labels"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Feature file {source} is missing keys {missing}")
    metadata = _validated_metadata(payload, source)
    coordinates = payload.get("coordinates")
    bundle = FeatureBundle(
        wsi_feature=_tensor(payload["wsi_feature"], torch.float32),
        patch_features=_tensor(payload["patch_features"], torch.float32),
        tissue_labels=_tissue_label_tensor(payload["tissue_labels"], source),
        coordinates=(
            None if coordinates is None else _tensor(coordinates, torch.long)
        ),
        metadata=metadata,
    )
    bundle.validate(wsi_dim=wsi_dim, patch_dim=patch_dim)
    _validate_finite_features(bundle, source)
    if num_classes is not None:
        validate_tissue_labels(bundle.tissue_labels, num_classes)
    return bundle


def save_feature_bundle(bundle: FeatureBundle, path: str | Path) -> None:
    bundle.validate()
    destination = Path(path)
    _validate_finite_features(bundle, destination)
    tissue_labels = _tissue_label_tensor(bundle.tissue_labels, destination)
    metadata = {**bundle.metadata, "schema_version": FEATURE_SCHEMA_VERSION}
    atomic_torch_save(
        {
            "wsi_feature": bundle.wsi_feature.detach().cpu().float().contiguous(),
            "patch_features": bundle.patch_features.detach().cpu().float().contiguous(),
            "tissue_labels": tissue_labels,
            "coordinates": (
                None
                if bundle.coordinates is None
                else bundle.coordinates.detach().cpu().long().contiguous()
            ),
            "metadata": metadata,
        },
        destination,
    )


class RunningMoments:
    """Numerically stable streaming mean and sample standard deviation."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        matrix = values.detach().cpu().double().reshape(-1, self.mean.numel())
        if matrix.shape[0] == 0:
            return
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("Cannot fit normalization on non-finite features")
        batch_count = matrix.shape[0]
        batch_mean = matrix.mean(dim=0)
        batch_m2 = ((matrix - batch_mean) ** 2).sum(dim=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta.square() * self.count * batch_count / total
        self.count = total

    def finalize(self, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.count < 2:
            raise ValueError("At least two feature vectors are needed to fit standardization")
        variance = self.m2 / (self.count - 1)
        return self.mean.float(), variance.clamp_min(epsilon**2).sqrt().float()


@dataclass
class FeatureNormalizer:
    wsi_mean: torch.Tensor
    wsi_std: torch.Tensor
    patch_mean: torch.Tensor
    patch_std: torch.Tensor
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in ("wsi_mean", "wsi_std", "patch_mean", "patch_std"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if self.wsi_mean.shape != self.wsi_std.shape:
            raise ValueError("wsi_mean and wsi_std shapes differ")
        if self.patch_mean.shape != self.patch_std.shape:
            raise ValueError("patch_mean and patch_std shapes differ")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def normalize_wsi(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.wsi_mean.to(values)) / self.wsi_std.to(values).clamp_min(
            self.epsilon
        )

    def denormalize_wsi(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.wsi_std.to(values) + self.wsi_mean.to(values)

    def normalize_patch(self, values: torch.Tensor) -> torch.Tensor:
        return (
            values - self.patch_mean.to(values)
        ) / self.patch_std.to(values).clamp_min(self.epsilon)

    def denormalize_patch(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.patch_std.to(values) + self.patch_mean.to(values)

    def save(self, path: str | Path) -> None:
        atomic_torch_save(
            {
                "wsi_mean": self.wsi_mean.detach().cpu(),
                "wsi_std": self.wsi_std.detach().cpu(),
                "patch_mean": self.patch_mean.detach().cpu(),
                "patch_std": self.patch_std.detach().cpu(),
                "epsilon": float(self.epsilon),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        payload = torch_load_cpu(path)
        if not isinstance(payload, Mapping):
            raise TypeError("Normalizer artifact must contain a mapping")
        required = {"wsi_mean", "wsi_std", "patch_mean", "patch_std", "epsilon"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Normalizer artifact is missing fields: {missing}")
        return cls(**{key: payload[key] for key in required})


def fit_normalizer(
    records: Sequence[SlideRecord],
    wsi_dim: int,
    patch_dim: int,
    epsilon: float = 1.0e-6,
) -> FeatureNormalizer:
    """Fit only from the records passed by the caller (normally the train fold)."""

    if not records:
        raise ValueError("Cannot fit a normalizer without training records")
    wsi_moments = RunningMoments(wsi_dim)
    patch_moments = RunningMoments(patch_dim)
    for record in records:
        bundle = load_feature_bundle(
            record.feature_path,
            wsi_dim=wsi_dim,
            patch_dim=patch_dim,
        )
        wsi_moments.update(bundle.wsi_feature)
        patch_moments.update(bundle.patch_features)
    wsi_mean, wsi_std = wsi_moments.finalize(epsilon)
    patch_mean, patch_std = patch_moments.finalize(epsilon)
    return FeatureNormalizer(wsi_mean, wsi_std, patch_mean, patch_std, epsilon)


# ---------------------------------------------------------------------------
# DDPM and survival datasets
# ---------------------------------------------------------------------------

def tissue_distribution(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    validate_tissue_labels(labels, num_classes)
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    return counts / counts.sum().clamp_min(1)


class LeaveOneSlideOutDataset(Dataset):
    """One target WSI per item; all other patient WSIs form the condition."""

    def __init__(
        self,
        patients: Sequence[PatientRecord],
        wsi_dim: int,
        patch_dim: int,
        num_classes: int,
        normalizer: FeatureNormalizer | None = None,
        minimum_slides: int = 2,
        cache_size: int = 8,
    ) -> None:
        if minimum_slides < 2:
            raise ValueError("minimum_slides must be at least two")
        self.patients = list(patients)
        self.wsi_dim = wsi_dim
        self.patch_dim = patch_dim
        self.num_classes = num_classes
        self.normalizer = normalizer
        self.cache = FeatureBundleCache(cache_size, wsi_dim, patch_dim, num_classes)
        self.examples = [
            (patient_index, target_index)
            for patient_index, patient in enumerate(self.patients)
            if len(patient.slides) >= minimum_slides
            for target_index in range(len(patient.slides))
        ]
        if not self.examples:
            raise ValueError("No patient has enough slides for leave-one-slide-out diffusion")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        patient_index, target_index = self.examples[index]
        patient = self.patients[patient_index]
        target_record = patient.slides[target_index]
        target = self.cache.get(target_record.feature_path)
        other_features = torch.stack(
            [
                self.cache.get(slide.feature_path).wsi_feature
                for slide_index, slide in enumerate(patient.slides)
                if slide_index != target_index
            ]
        )
        target_wsi = target.wsi_feature
        condition = other_features.mean(dim=0)
        if self.normalizer is not None:
            target_wsi = self.normalizer.normalize_wsi(target_wsi)
            condition = self.normalizer.normalize_wsi(condition)
        return {
            "target_wsi": target_wsi,
            "target_distribution": tissue_distribution(
                target.tissue_labels,
                self.num_classes,
            ),
            "condition": condition,
            "patient_id": patient.patient_id,
            "slide_id": target_record.slide_id,
        }


class PatchDiffusionDataset(Dataset):
    """Deterministically sample real patches and WSI/class conditions."""

    def __init__(
        self,
        patients: Sequence[PatientRecord],
        wsi_dim: int,
        patch_dim: int,
        num_classes: int,
        samples_per_epoch: int,
        normalizer: FeatureNormalizer | None = None,
        seed: int = 2025,
        cache_size: int = 8,
        slide_local_block_size: int = 512,
    ) -> None:
        self.slides = [slide for patient in patients for slide in patient.slides]
        if not self.slides:
            raise ValueError("PatchDiffusionDataset received no slides")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.wsi_dim = wsi_dim
        self.patch_dim = patch_dim
        self.num_classes = num_classes
        self.samples_per_epoch = samples_per_epoch
        self.normalizer = normalizer
        self.seed = seed
        self.cache = FeatureBundleCache(cache_size, wsi_dim, patch_dim, num_classes)
        self.slide_local_block_size = max(1, slide_local_block_size)
        self.epoch = 0
        self._slide_order = list(range(len(self.slides)))
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch
        generator = torch.Generator().manual_seed(self.seed + epoch * 1_000_003)
        self._slide_order = torch.randperm(
            len(self.slides),
            generator=generator,
        ).tolist()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * self.samples_per_epoch + index
        )
        block = index // self.slide_local_block_size
        slide_index = self._slide_order[block % len(self._slide_order)]
        record = self.slides[slide_index]
        bundle = self.cache.get(record.feature_path)
        patch_index = int(
            torch.randint(bundle.patch_features.shape[0], (1,), generator=generator)
        )
        patch = bundle.patch_features[patch_index]
        consistency = bundle.wsi_feature
        if self.normalizer is not None:
            patch = self.normalizer.normalize_patch(patch)
            consistency = self.normalizer.normalize_wsi(consistency)
        tissue_class = bundle.tissue_labels[patch_index].long()
        validate_tissue_labels(tissue_class.reshape(1), self.num_classes)
        return {
            "patch": patch,
            "consistency": consistency,
            "tissue_class": tissue_class,
            "slide_id": record.slide_id,
        }


class SurvivalDataset(Dataset):
    """Patient-level variable-length bags with optional generated slides."""

    def __init__(
        self,
        patients: Sequence[PatientRecord],
        wsi_dim: int,
        patch_dim: int,
        normalizer: FeatureNormalizer | None = None,
        generated_index: str | Path | None = None,
        max_patches_per_slide: int | None = None,
        expected_fold: int | None = None,
        partition: str | None = None,
    ) -> None:
        self.patients = list(patients)
        self.wsi_dim = wsi_dim
        self.patch_dim = patch_dim
        self.normalizer = normalizer
        if max_patches_per_slide is not None and max_patches_per_slide < 1:
            raise ValueError("max_patches_per_slide must be positive when provided")
        self.max_patches = max_patches_per_slide
        self.expected_fold = expected_fold
        self.generated: dict[str, list[str]] = {}
        if generated_index is not None:
            index_path = Path(generated_index).resolve()
            if not index_path.is_file():
                raise FileNotFoundError(f"Generated index does not exist: {index_path}")
            with index_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("Generated index must use schema_version=1")
            if expected_fold is not None and int(payload.get("fold", -1)) != expected_fold:
                raise ValueError("Generated index belongs to a different fold")
            patient_index = payload.get("patients")
            partition_index = payload.get("partitions")
            if not isinstance(patient_index, dict) or not isinstance(partition_index, dict):
                raise ValueError(
                    "Generated index must contain patients and partitions mappings"
                )
            if partition is None or partition not in partition_index:
                raise ValueError("A valid partition is required with a generated index")
            patient_ids = {patient.patient_id for patient in self.patients}
            unknown = sorted(patient_ids.difference(patient_index))
            if unknown:
                raise ValueError(f"Generated index is missing patients: {unknown[:5]}")
            selected: dict[str, list[str]] = {}
            selected_paths: list[str] = []
            for patient_id in sorted(patient_ids):
                paths = patient_index[patient_id]
                if not isinstance(paths, list) or not all(
                    isinstance(path, str) and path.strip() for path in paths
                ):
                    raise ValueError(
                        f"Generated paths for {patient_id} must be a list of strings"
                    )
                resolved = [
                    str(
                        (
                            Path(path)
                            if Path(path).is_absolute()
                            else index_path.parent / path
                        ).resolve()
                    )
                    for path in paths
                ]
                selected[patient_id] = resolved
                selected_paths.extend(resolved)
            declared_paths = partition_index[partition]
            if not isinstance(declared_paths, list) or not all(
                isinstance(path, str) and path.strip() for path in declared_paths
            ):
                raise ValueError(
                    f"Generated partition {partition} must be a list of paths"
                )
            resolved_declared = {
                str(
                    (
                        Path(path)
                        if Path(path).is_absolute()
                        else index_path.parent / path
                    ).resolve()
                )
                for path in declared_paths
            }
            if len(selected_paths) != len(set(selected_paths)):
                raise ValueError("A generated feature path is assigned more than once")
            if set(selected_paths) != resolved_declared:
                raise ValueError(
                    f"Generated patient mapping does not match partition {partition!r}"
                )
            self.generated = selected

    def __len__(self) -> int:
        return len(self.patients)

    def _patch_bag(self, bundle: FeatureBundle) -> torch.Tensor:
        patches = bundle.patch_features
        if self.max_patches is not None and patches.shape[0] > self.max_patches:
            indices = torch.linspace(
                0,
                patches.shape[0] - 1,
                self.max_patches,
            ).long()
            patches = patches[indices]
        return self.normalizer.normalize_patch(patches) if self.normalizer else patches

    def _generated_bundle(self, path: str, patient_id: str) -> FeatureBundle:
        bundle = load_feature_bundle(path, self.wsi_dim, self.patch_dim)
        metadata = bundle.metadata
        if metadata.get("synthetic") is not True:
            raise ValueError(f"Generated bundle is not marked synthetic: {path}")
        if str(metadata.get("patient_id")) != patient_id:
            raise ValueError(f"Generated bundle patient mismatch for {path}")
        if self.expected_fold is not None and int(metadata.get("fold", -1)) != self.expected_fold:
            raise ValueError(f"Generated bundle fold mismatch for {path}")
        return bundle

    def __getitem__(self, index: int) -> PatientBags:
        patient = self.patients[index]
        real_slides = [
            self._patch_bag(
                load_feature_bundle(
                    slide.feature_path,
                    self.wsi_dim,
                    self.patch_dim,
                )
            )
            for slide in patient.slides
        ]
        generated_slides = [
            self._patch_bag(self._generated_bundle(path, patient.patient_id))
            for path in self.generated.get(patient.patient_id, [])
        ]
        return PatientBags(
            patient_id=patient.patient_id,
            real_slides=real_slides,
            generated_slides=generated_slides,
            time=patient.time,
            event=patient.event,
        )


def patient_bag_collate(batch: list[PatientBags]) -> list[PatientBags]:
    return batch


# ---------------------------------------------------------------------------
# Tissue-conditioned synthetic allocation
# ---------------------------------------------------------------------------

def project_to_simplex(values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(values / temperature, dim=-1)


def largest_remainder_counts(
    probabilities: torch.Tensor,
    total: int,
    minimum_per_nonzero_class: int = 0,
) -> torch.Tensor:
    """Allocate exact integer counts while minimizing rounding error."""

    if probabilities.ndim != 1 or probabilities.numel() < 1:
        raise ValueError("probabilities must be a non-empty 1-D tensor")
    if total < 0:
        raise ValueError("total must be non-negative")
    if minimum_per_nonzero_class < 0:
        raise ValueError("minimum_per_nonzero_class cannot be negative")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    probs = probabilities.float().clamp_min(0)
    if float(probs.sum()) == 0:
        probs.fill_(1 / probs.numel())
    else:
        probs /= probs.sum()
    counts = torch.zeros_like(probs, dtype=torch.long)
    if minimum_per_nonzero_class > 0:
        active = probs > 0
        required = int(active.sum()) * minimum_per_nonzero_class
        if required > total:
            raise ValueError("minimum allocation exceeds requested total")
        counts[active] = minimum_per_nonzero_class
    remaining = total - int(counts.sum())
    expected = probs * remaining
    floor = expected.floor().long()
    counts += floor
    remainder = total - int(counts.sum())
    if remainder:
        fractions = expected - floor
        ordered = sorted(
            range(fractions.numel()),
            key=lambda index: (-float(fractions[index].detach().cpu()), index),
        )
        indices = torch.tensor(
            ordered[:remainder],
            dtype=torch.long,
            device=counts.device,
        )
        counts[indices] += 1
    if int(counts.sum()) != total:
        raise AssertionError("allocation failed to preserve total")
    return counts


def expand_class_counts(
    counts: torch.Tensor,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if counts.ndim != 1 or counts.numel() < 1:
        raise ValueError("counts must be a non-empty 1-D tensor")
    if bool((counts < 0).any()):
        raise ValueError("counts cannot be negative")
    classes = torch.repeat_interleave(torch.arange(counts.numel()), counts.cpu())
    if shuffle and classes.numel() > 1:
        classes = classes[torch.randperm(classes.numel(), generator=generator)]
    return classes.long()


# ---------------------------------------------------------------------------
# Raw WSI masking and patch extraction
# ---------------------------------------------------------------------------

def otsu_threshold(grayscale: np.ndarray) -> int:
    values = np.asarray(grayscale, dtype=np.uint8).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot threshold an empty image")
    histogram = np.bincount(values, minlength=256).astype(np.float64)
    probability = histogram / values.size
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256))
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1 - cumulative_probability)
    between_class = np.zeros(256, dtype=np.float64)
    valid = denominator > 0
    between_class[valid] = (
        global_mean * cumulative_probability[valid] - cumulative_mean[valid]
    ) ** 2 / denominator[valid]
    return int(np.argmax(between_class))


def rgb_saturation(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("rgb must have shape [H, W, 3]")
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    return np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )


def make_tissue_mask(
    thumbnail: Image.Image | np.ndarray,
    minimum_saturation: float = 0.05,
) -> np.ndarray:
    if not 0 <= minimum_saturation <= 1:
        raise ValueError("minimum_saturation must lie in [0, 1]")
    rgb = np.asarray(
        thumbnail.convert("RGB") if isinstance(thumbnail, Image.Image) else thumbnail
    )
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("thumbnail must have RGB shape [H, W, 3]")
    grayscale = np.round(
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.uint8)
    threshold = otsu_threshold(grayscale)
    return (grayscale < threshold) & (rgb_saturation(rgb) >= minimum_saturation)


def tissue_fraction(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    crop = mask[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    return float(crop.mean()) if crop.size else 0.0


@dataclass(frozen=True)
class PatchCoordinate:
    x: int
    y: int
    level: int
    size: int
    tissue_fraction: float


def open_slide(path: str | Path) -> Any:
    try:
        import openslide
    except ImportError as error:
        raise ImportError(
            "openslide-python is required for raw WSI preprocessing; "
            "install the project's WSI dependencies"
        ) from error
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"WSI does not exist: {source}")
    return openslide.OpenSlide(str(source))


def thumbnail_with_scale(slide: Any, maximum_side: int) -> tuple[Image.Image, float, float]:
    if maximum_side < 1:
        raise ValueError("maximum_side must be positive")
    width, height = slide.dimensions
    if width < 1 or height < 1:
        raise ValueError("Slide dimensions must be positive")
    scale = min(1.0, maximum_side / max(width, height))
    thumbnail = slide.get_thumbnail(
        (max(1, round(width * scale)), max(1, round(height * scale)))
    )
    scale_x = thumbnail.width / width
    scale_y = thumbnail.height / height
    return thumbnail.convert("RGB"), scale_x, scale_y


def enumerate_tissue_patches(
    slide: Any,
    patch_size: int,
    stride: int,
    level: int,
    thumbnail_max_side: int,
    minimum_tissue_fraction: float,
    minimum_saturation: float,
) -> list[PatchCoordinate]:
    if patch_size < 1 or stride < 1:
        raise ValueError("patch_size and stride must be positive")
    if stride < patch_size:
        raise ValueError("Non-overlapping extraction requires stride >= patch_size")
    if level < 0 or level >= len(slide.level_dimensions):
        raise ValueError(f"Invalid slide level {level}")
    if not 0 <= minimum_tissue_fraction <= 1:
        raise ValueError("minimum_tissue_fraction must lie in [0, 1]")
    thumbnail, scale_x, scale_y = thumbnail_with_scale(slide, thumbnail_max_side)
    mask = make_tissue_mask(thumbnail, minimum_saturation)
    level_width, level_height = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])
    coordinates: list[PatchCoordinate] = []
    for level_y in range(0, max(level_height - patch_size + 1, 0), stride):
        for level_x in range(0, max(level_width - patch_size + 1, 0), stride):
            x = round(level_x * downsample)
            y = round(level_y * downsample)
            x0, y0 = round(x * scale_x), round(y * scale_y)
            x1 = round((x + patch_size * downsample) * scale_x)
            y1 = round((y + patch_size * downsample) * scale_y)
            fraction = tissue_fraction(mask, x0, y0, x1, y1)
            if fraction >= minimum_tissue_fraction:
                coordinates.append(
                    PatchCoordinate(x, y, level, patch_size, fraction)
                )
    return coordinates


def read_patch(slide: Any, coordinate: PatchCoordinate) -> Image.Image:
    return slide.read_region(
        (coordinate.x, coordinate.y),
        coordinate.level,
        (coordinate.size, coordinate.size),
    ).convert("RGB")


def iter_patch_batches(
    slide: Any,
    coordinates: Sequence[PatchCoordinate],
    batch_size: int,
) -> Iterator[tuple[list[Image.Image], np.ndarray]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        yield (
            [read_patch(slide, coordinate) for coordinate in batch_coordinates],
            np.asarray(
                [[item.x, item.y] for item in batch_coordinates],
                dtype=np.int64,
            ),
        )


# ---------------------------------------------------------------------------
# HIPT-compatible representation encoder adapters
# ---------------------------------------------------------------------------

class RepresentationEncoder(ABC):
    @abstractmethod
    def encode_patches(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Return a ``[B, D_patch]`` tensor."""

    @abstractmethod
    def encode_wsi(
        self,
        patch_features: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Return one ``[D_wsi]`` tensor."""


def imagenet_tensor(image: Image.Image, size: int = 256) -> torch.Tensor:
    if size < 1:
        raise ValueError("size must be positive")
    resized = image.convert("RGB").resize(
        (size, size),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return (tensor - mean) / std


class TorchModuleEncoder(RepresentationEncoder):
    """Adapter for HIPT patch/WSI modules or API-compatible encoders."""

    def __init__(
        self,
        patch_model: nn.Module,
        wsi_model: nn.Module | None,
        device: torch.device,
        image_size: int = 256,
    ) -> None:
        self.patch_model = patch_model.eval().to(device)
        self.wsi_model = None if wsi_model is None else wsi_model.eval().to(device)
        self.device = device
        self.image_size = image_size

    @staticmethod
    def _unwrap(output: Any, name: str) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            if not output:
                raise ValueError(f"{name} returned an empty sequence")
            output = output[0]
        if isinstance(output, Mapping):
            output = output.get("features", output.get("x"))
        if not isinstance(output, torch.Tensor):
            raise ValueError(f"{name} must return tensor features")
        return output

    @torch.no_grad()
    def encode_patches(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("encode_patches received an empty batch")
        inputs = torch.stack(
            [imagenet_tensor(image, self.image_size) for image in images]
        ).to(self.device)
        output = self._unwrap(self.patch_model(inputs), "patch_model")
        if output.ndim != 2 or output.shape[0] != len(images):
            raise ValueError(
                "patch_model must return [B, D] features matching the input batch"
            )
        return output.detach().cpu().float()

    @torch.no_grad()
    def encode_wsi(
        self,
        patch_features: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if patch_features.ndim != 2:
            raise ValueError("patch_features must have shape [N, D]")
        if coordinates.shape != (patch_features.shape[0], 2):
            raise ValueError("coordinates must have shape [N, 2]")
        if self.wsi_model is None:
            return patch_features.mean(dim=0)
        output = self._unwrap(
            self.wsi_model(
                patch_features.unsqueeze(0).to(self.device),
                coordinates.unsqueeze(0).to(self.device),
            ),
            "wsi_model",
        )
        if output.ndim == 1:
            vector = output
        elif output.ndim == 2 and output.shape[0] == 1:
            vector = output[0]
        else:
            raise ValueError(
                "wsi_model must return [D] or [1, D]; got "
                f"{tuple(output.shape)}"
            )
        return vector.detach().cpu().float()


class TimmMeanEncoder(TorchModuleEncoder):
    """Convenience patch-backbone adapter with mean WSI pooling."""

    @classmethod
    def create(
        cls,
        model_name: str,
        device: torch.device,
        checkpoint: str | Path | None = None,
        image_size: int = 256,
    ) -> "TimmMeanEncoder":
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "Install the project's WSI dependencies to use timm encoders"
            ) from error
        model = timm.create_model(
            model_name,
            pretrained=checkpoint is None,
            num_classes=0,
        )
        if checkpoint is not None:
            state = torch_load_cpu(checkpoint)
            if not isinstance(state, Mapping):
                raise TypeError("Encoder checkpoint must contain a state mapping")
            model.load_state_dict(state.get("model", state))
        return cls(model, None, device, image_size)


# ---------------------------------------------------------------------------
# HoVer-Net parsing, spatial index, and patch majority vote
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NucleusPrediction:
    x: float
    y: float
    nucleus_type: int
    probability: float


def load_hovernet_json(path: str | Path) -> list[NucleusPrediction]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, Mapping):
        nuclei: Any = payload.get("nuc", payload.get("nuclei", payload))
    else:
        nuclei = payload
    entries = nuclei.values() if isinstance(nuclei, Mapping) else nuclei
    if not isinstance(entries, Iterable):
        raise ValueError("HoVer-Net JSON must contain an iterable nucleus collection")
    predictions: list[NucleusPrediction] = []
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        centroid = item.get("centroid", item.get("center"))
        if centroid is None or not isinstance(centroid, Sequence) or len(centroid) != 2:
            continue
        prediction = NucleusPrediction(
            x=float(centroid[0]),
            y=float(centroid[1]),
            nucleus_type=int(item.get("type", 0)),
            probability=float(item.get("type_prob", item.get("probability", 1.0))),
        )
        if not all(
            math.isfinite(value)
            for value in (prediction.x, prediction.y, prediction.probability)
        ):
            raise ValueError("HoVer-Net centroids and probabilities must be finite")
        predictions.append(prediction)
    return predictions


def majority_vote_patch(
    nuclei: Iterable[NucleusPrediction],
    x: int,
    y: int,
    size_level0: int,
    hovernet_type_to_tissue: Mapping[int, int],
    empty_class: int = TISSUE_TO_INDEX["no_label"],
) -> int:
    if size_level0 < 1:
        raise ValueError("size_level0 must be positive")
    counts: dict[int, int] = {}
    probability_sums: dict[int, float] = {}
    for nucleus in nuclei:
        if x <= nucleus.x < x + size_level0 and y <= nucleus.y < y + size_level0:
            tissue_class = hovernet_type_to_tissue.get(
                nucleus.nucleus_type,
                empty_class,
            )
            counts[tissue_class] = counts.get(tissue_class, 0) + 1
            probability_sums[tissue_class] = (
                probability_sums.get(tissue_class, 0.0) + nucleus.probability
            )
    if not counts:
        return empty_class
    return max(
        counts,
        key=lambda key: (counts[key], probability_sums[key], -key),
    )


def label_patch_coordinates(
    nuclei: Sequence[NucleusPrediction],
    coordinates: torch.Tensor,
    patch_size: int,
    level_downsample: float,
    hovernet_type_to_tissue: Mapping[int, int],
    empty_class: int = TISSUE_TO_INDEX["no_label"],
) -> torch.Tensor:
    """Assign tissue classes through an exact uniform-grid spatial join."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [N, 2]")
    if patch_size < 1 or not math.isfinite(level_downsample) or level_downsample <= 0:
        raise ValueError("patch_size and level_downsample must be positive")
    if empty_class not in range(len(TISSUE_CLASSES)):
        raise ValueError("empty_class is outside the configured tissue vocabulary")
    for source_type, target_class in hovernet_type_to_tissue.items():
        if not isinstance(source_type, int) or target_class not in range(len(TISSUE_CLASSES)):
            raise ValueError("HoVer-Net mapping contains an invalid class identifier")
    size_level0 = round(patch_size * level_downsample)
    spatial: dict[tuple[int, int], list[NucleusPrediction]] = defaultdict(list)
    for nucleus in nuclei:
        cell = (
            math.floor(nucleus.x / size_level0),
            math.floor(nucleus.y / size_level0),
        )
        spatial[cell].append(nucleus)

    def candidates(x_value: int, y_value: int) -> list[NucleusPrediction]:
        minimum_x = math.floor(x_value / size_level0)
        maximum_x = math.floor((x_value + size_level0 - 1.0e-9) / size_level0)
        minimum_y = math.floor(y_value / size_level0)
        maximum_y = math.floor((y_value + size_level0 - 1.0e-9) / size_level0)
        return [
            nucleus
            for cell_x in range(minimum_x, maximum_x + 1)
            for cell_y in range(minimum_y, maximum_y + 1)
            for nucleus in spatial.get((cell_x, cell_y), ())
        ]

    labels = [
        majority_vote_patch(
            candidates(int(coordinate[0]), int(coordinate[1])),
            int(coordinate[0]),
            int(coordinate[1]),
            size_level0,
            hovernet_type_to_tissue,
            empty_class,
        )
        for coordinate in coordinates
    ]
    return torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# End-to-end preprocessing
# ---------------------------------------------------------------------------

def preprocess_slide(
    record: SlideRecord,
    encoder: RepresentationEncoder,
    output_path: str | Path,
    hovernet_json: str | Path,
    patch_size: int = 256,
    stride: int = 256,
    level: int = 0,
    thumbnail_max_side: int = 2048,
    minimum_tissue_fraction: float = 0.5,
    minimum_saturation: float = 0.05,
    encoder_batch_size: int = 256,
    hovernet_type_to_tissue: Mapping[int, int] | None = None,
    patch_encoder_path: str | Path | None = None,
    wsi_encoder_path: str | Path | None = None,
    expected_wsi_dim: int | None = None,
    expected_patch_dim: int | None = None,
) -> FeatureBundle:
    """Convert one WSI and its HoVer-Net output into a feature bundle."""

    if record.wsi_path is None:
        raise ValueError(f"Manifest has no wsi_path for {record.slide_id}")
    hovernet_path = Path(hovernet_json).resolve()
    if not hovernet_path.is_file():
        raise FileNotFoundError(f"HoVer-Net JSON does not exist: {hovernet_path}")
    mapping = dict(
        hovernet_type_to_tissue
        or {1: 0, 2: 2, 3: 4, 4: 1, 5: 3}
    )
    slide = open_slide(record.wsi_path)
    try:
        coordinates = enumerate_tissue_patches(
            slide,
            patch_size,
            stride,
            level,
            thumbnail_max_side,
            minimum_tissue_fraction,
            minimum_saturation,
        )
        if not coordinates:
            raise ValueError(f"No tissue patches passed QC for {record.slide_id}")
        feature_batches: list[torch.Tensor] = []
        coordinate_batches: list[torch.Tensor] = []
        for images, coordinate_array in iter_patch_batches(
            slide,
            coordinates,
            encoder_batch_size,
        ):
            features = encoder.encode_patches(images)
            if features.shape[0] != len(images):
                raise ValueError("Encoder output batch does not match input patches")
            feature_batches.append(features)
            coordinate_batches.append(torch.from_numpy(coordinate_array))
        patch_features = torch.cat(feature_batches)
        coordinate_tensor = torch.cat(coordinate_batches).long()
        wsi_feature = encoder.encode_wsi(patch_features, coordinate_tensor)
        nuclei = load_hovernet_json(hovernet_path)
        tissue_labels = label_patch_coordinates(
            nuclei,
            coordinate_tensor,
            patch_size,
            float(slide.level_downsamples[level]),
            mapping,
            TISSUE_TO_INDEX["no_label"],
        )
        patch_encoder = (
            None if patch_encoder_path is None else str(Path(patch_encoder_path).resolve())
        )
        wsi_encoder = (
            None if wsi_encoder_path is None else str(Path(wsi_encoder_path).resolve())
        )
        wsi_stat = record.wsi_path.stat()
        bundle = FeatureBundle(
            wsi_feature=wsi_feature,
            patch_features=patch_features,
            tissue_labels=tissue_labels,
            coordinates=coordinate_tensor,
            metadata={
                "patient_id": record.patient_id,
                "slide_id": record.slide_id,
                "source_wsi": str(record.wsi_path),
                "source_wsi_size": int(wsi_stat.st_size),
                "source_wsi_mtime_ns": int(wsi_stat.st_mtime_ns),
                "patch_size": patch_size,
                "stride": stride,
                "level": level,
                "level_downsample": float(slide.level_downsamples[level]),
                "minimum_tissue_fraction": minimum_tissue_fraction,
                "minimum_saturation": minimum_saturation,
                "hovernet_json": str(hovernet_path),
                "hovernet_type_to_tissue": {
                    str(key): int(value) for key, value in sorted(mapping.items())
                },
                "tissue_classes": list(TISSUE_CLASSES),
                "encoder_class": type(encoder).__name__,
                "patch_encoder": patch_encoder,
                "wsi_encoder": wsi_encoder,
            },
        )
        bundle.validate(wsi_dim=expected_wsi_dim, patch_dim=expected_patch_dim)
        validate_tissue_labels(bundle.tissue_labels, len(TISSUE_CLASSES))
        save_feature_bundle(bundle, output_path)
        return bundle
    finally:
        slide.close()


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "INDEX_TO_TISSUE",
    "REQUIRED_MANIFEST_COLUMNS",
    "SUPPORTED_SLIDE_EXTENSIONS",
    "TISSUE_CLASSES",
    "TISSUE_TO_INDEX",
    "FeatureBundle",
    "FeatureBundleCache",
    "FeatureNormalizer",
    "LeaveOneSlideOutDataset",
    "NucleusPrediction",
    "PatchCoordinate",
    "PatchDiffusionDataset",
    "PatientBags",
    "PatientRecord",
    "RepresentationEncoder",
    "RunningMoments",
    "SlideRecord",
    "SurvivalDataset",
    "SurvivalPrediction",
    "TimmMeanEncoder",
    "TorchModuleEncoder",
    "assert_patient_disjoint",
    "atomic_json_dump",
    "atomic_torch_save",
    "enumerate_tissue_patches",
    "expand_class_counts",
    "fit_normalizer",
    "group_patients",
    "imagenet_tensor",
    "iter_patch_batches",
    "label_patch_coordinates",
    "largest_remainder_counts",
    "load_feature_bundle",
    "load_fold",
    "load_hovernet_json",
    "majority_vote_patch",
    "make_patient_splits",
    "make_tissue_mask",
    "open_slide",
    "otsu_threshold",
    "partition_patients",
    "patient_bag_collate",
    "preprocess_slide",
    "project_to_simplex",
    "read_manifest",
    "read_patch",
    "rgb_saturation",
    "save_feature_bundle",
    "save_splits",
    "thumbnail_with_scale",
    "tissue_distribution",
    "tissue_fraction",
    "torch_load_cpu",
    "torch_load_restricted_cpu",
    "validate_feature_paths",
    "validate_split_payload",
    "validate_tissue_labels",
]
