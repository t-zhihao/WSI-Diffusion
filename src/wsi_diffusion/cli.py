"""Single command-line entry point for the official WSI-Diffusion pipeline."""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from wsi_diffusion.config import Config, dump_config, load_config
from wsi_diffusion.data import (
    FeatureBundle,
    FeatureNormalizer,
    LeaveOneSlideOutDataset,
    PatchDiffusionDataset,
    PatientRecord,
    SlideRecord,
    SurvivalDataset,
    TorchModuleEncoder,
    fit_normalizer,
    group_patients,
    load_feature_bundle,
    load_fold,
    make_patient_splits,
    partition_patients,
    patient_bag_collate,
    preprocess_slide,
    read_manifest,
    save_feature_bundle,
    save_splits,
    validate_feature_paths,
)
from wsi_diffusion.engine import (
    BreslowBaseline,
    DiffusionTrainer,
    QuantileStageMetric,
    SurvivalTrainer,
    amp_enabled,
    atomic_json_dump,
    bootstrap_metric,
    concordance_index,
    configure_logging,
    fixed_time_stage_accuracy,
    fold_directory,
    load_model_checkpoint,
    make_generator,
    mark_stage_complete,
    project_path,
    resolve_device,
    seed_everything,
    seed_worker,
    slide_count_bias_metrics,
    stage_is_complete,
    write_aggregate,
    write_provenance,
)
from wsi_diffusion.models import (
    build_hierarchical_generator,
    build_patch_diffusion,
    build_survival_model,
    build_wsi_diffusion,
    number_to_generate,
)


@dataclass
class FoldContext:
    config: Config
    fold: int
    records: List[SlideRecord]
    partitions: Dict[str, List[PatientRecord]]
    directory: Path
    logger: Any


def _resolved_config_payload(config: Config) -> Dict[str, Any]:
    payload = config.to_dict()
    payload.pop("runtime", None)
    return payload


def _has_comparable_pair(patients: Sequence[PatientRecord]) -> bool:
    for left, first in enumerate(patients):
        for second in patients[left + 1 :]:
            if first.time == second.time:
                continue
            earlier = first if first.time < second.time else second
            if earlier.event == 1:
                return True
    return False


def _validate_partition_requirements(
    config: Config,
    partitions: Mapping[str, List[PatientRecord]],
) -> None:
    if not any(patient.event == 1 for patient in partitions["train"]):
        raise ValueError("The training partition has no observed event for Cox loss")
    if not _has_comparable_pair(partitions["validation"]):
        raise ValueError("The validation partition has no comparable survival pair")
    if bool(config.generation.enabled) and not bool(config.generation.skip_wsi_level):
        minimum = int(config.data.min_slides_for_diffusion)
        for name in ("train", "validation"):
            eligible = [patient for patient in partitions[name] if len(patient.slides) >= minimum]
            if not eligible:
                raise ValueError(
                    f"{name} has no patient with at least {minimum} slides for WSI diffusion"
                )


def prepare_fold(
    config_path: str,
    fold: int,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> FoldContext:
    config = load_config(config_path, overrides)
    if fold not in range(int(config.split.n_folds)):
        raise ValueError(f"fold must be in [0, {int(config.split.n_folds) - 1}]")
    seed_everything(int(config.experiment.seed) + fold, bool(config.experiment.deterministic))
    directory = fold_directory(config, fold)
    directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(directory, verbose)
    records = read_manifest(
        project_path(config, config.data.manifest),
        config.runtime.project_root,
        str(config.data.event_column_semantics),
        expected_dataset=str(config.data.dataset),
    )
    split = load_fold(
        project_path(config, config.data.split_file),
        fold,
        expected_n_folds=int(config.split.n_folds),
        expected_dataset=str(config.data.dataset),
        expected_seed=int(config.experiment.seed),
        expected_validation_fraction=float(config.split.validation_fraction),
        expected_time_bins=int(config.split.stratify_time_bins),
    )
    partitions = partition_patients(group_patients(records), split)
    _validate_partition_requirements(config, partitions)

    resolved_path = directory / "resolved_config.yaml"
    current = _resolved_config_payload(config)
    if resolved_path.is_file() and (directory / "stages").exists():
        with resolved_path.open("r", encoding="utf-8") as stream:
            previous = yaml.safe_load(stream)
        if previous != current:
            raise RuntimeError(
                f"Existing completed stages in {directory} use a different configuration. "
                "Choose a new experiment.output_dir or remove the old stage outputs."
            )
    dump_config(config, resolved_path)
    write_provenance(config.runtime.project_root, directory)
    return FoldContext(config, fold, records, partitions, directory, logger)


def _records(patients: Sequence[PatientRecord]) -> List[SlideRecord]:
    return [slide for patient in patients for slide in patient.slides]


def get_or_fit_normalizer(context: FoldContext) -> Optional[FeatureNormalizer]:
    if not bool(context.config.normalization.enabled):
        return None
    path = context.directory / "normalizer.pt"
    if path.is_file():
        normalizer = FeatureNormalizer.load(path)
        if normalizer.wsi_mean.numel() != int(context.config.data.wsi_dim):
            raise ValueError("Saved normalizer WSI dimension differs from the configuration")
        if normalizer.patch_mean.numel() != int(context.config.data.patch_dim):
            raise ValueError("Saved normalizer patch dimension differs from the configuration")
        return normalizer
    normalizer = fit_normalizer(
        _records(context.partitions["train"]),
        int(context.config.data.wsi_dim),
        int(context.config.data.patch_dim),
        float(context.config.normalization.epsilon),
    )
    normalizer.save(path)
    return normalizer


def validate_data(config_path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    config = load_config(config_path, overrides)
    records = read_manifest(
        project_path(config, config.data.manifest),
        config.runtime.project_root,
        str(config.data.event_column_semantics),
        expected_dataset=str(config.data.dataset),
    )
    missing = validate_feature_paths(records)
    if missing:
        raise FileNotFoundError(f"{len(missing)} feature files are missing; first={missing[:5]}")
    patch_counts: List[int] = []
    tissue_counts = torch.zeros(int(config.data.num_tissue_classes), dtype=torch.long)
    for record in records:
        bundle = load_feature_bundle(
            record.feature_path,
            int(config.data.wsi_dim),
            int(config.data.patch_dim),
            int(config.data.num_tissue_classes),
        )
        patch_counts.append(int(bundle.patch_features.shape[0]))
        tissue_counts += torch.bincount(
            bundle.tissue_labels,
            minlength=int(config.data.num_tissue_classes),
        )
    patients = group_patients(records)
    slides = [len(patient.slides) for patient in patients]
    return {
        "dataset": str(config.data.dataset),
        "patients": len(patients),
        "slides": len(records),
        "events": sum(patient.event for patient in patients),
        "slides_per_patient_min": min(slides),
        "slides_per_patient_max": max(slides),
        "patches_per_slide_min": min(patch_counts),
        "patches_per_slide_max": max(patch_counts),
        "tissue_counts": tissue_counts.tolist(),
    }


def make_splits(
    config_path: str,
    overrides: Optional[List[str]] = None,
    force: bool = False,
) -> Path:
    config = load_config(config_path, overrides)
    records = read_manifest(
        project_path(config, config.data.manifest),
        config.runtime.project_root,
        str(config.data.event_column_semantics),
        expected_dataset=str(config.data.dataset),
    )
    patients = group_patients(records)
    payload = make_patient_splits(
        patients,
        int(config.split.n_folds),
        float(config.split.validation_fraction),
        int(config.split.stratify_time_bins),
        int(config.experiment.seed),
    )
    payload["dataset"] = str(config.data.dataset)
    for fold in payload["folds"]:
        _validate_partition_requirements(config, partition_patients(patients, fold))
    destination = project_path(config, config.data.split_file)
    if destination.is_file() and not force:
        with destination.open("r", encoding="utf-8") as stream:
            if json.load(stream) == payload:
                return destination
        raise FileExistsError(
            f"A different split already exists at {destination}; pass --force explicitly"
        )
    save_splits(payload, destination)
    return destination


def preprocess(
    config_path: str,
    patch_model_path: str,
    wsi_model_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> None:
    config = load_config(config_path, overrides)
    device = resolve_device(str(config.experiment.device))
    patch_model = torch.jit.load(patch_model_path, map_location=device)
    wsi_model = (
        None if wsi_model_path is None else torch.jit.load(wsi_model_path, map_location=device)
    )
    encoder = TorchModuleEncoder(
        patch_model,
        wsi_model,
        device,
        int(config.preprocessing.patch_size),
    )
    records = read_manifest(
        project_path(config, config.data.manifest),
        config.runtime.project_root,
        str(config.data.event_column_semantics),
        expected_dataset=str(config.data.dataset),
    )
    prediction_root = project_path(config, config.preprocessing.hovernet.prediction_root)
    class_to_index = {name: index for index, name in enumerate(config.data.tissue_classes)}
    mapping = {
        int(hovernet_id): int(class_to_index[name])
        for name, hovernet_id in config.preprocessing.hovernet.pannuke_label_order.items()
    }
    for record in records:
        if record.feature_path.is_file():
            continue
        preprocess_slide(
            record=record,
            encoder=encoder,
            output_path=record.feature_path,
            hovernet_json=prediction_root / f"{record.slide_id}.json",
            patch_size=int(config.preprocessing.patch_size),
            stride=int(config.preprocessing.stride),
            level=int(config.preprocessing.level),
            thumbnail_max_side=int(config.preprocessing.thumbnail_max_side),
            minimum_tissue_fraction=float(config.preprocessing.minimum_tissue_fraction),
            minimum_saturation=float(config.preprocessing.otsu_saturation_threshold),
            encoder_batch_size=int(config.preprocessing.encoder.batch_size),
            hovernet_type_to_tissue=mapping,
            patch_encoder_path=patch_model_path,
            wsi_encoder_path=wsi_model_path,
            expected_wsi_dim=int(config.data.wsi_dim),
            expected_patch_dim=int(config.data.patch_dim),
        )


def _loader_options(config: Config, batch_size: int) -> Dict[str, Any]:
    return {
        "batch_size": batch_size,
        "num_workers": int(config.training.num_workers),
        "pin_memory": bool(config.training.pin_memory),
        "worker_init_fn": seed_worker,
    }


def train_wsi(
    config_path: str,
    fold: int,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    context = prepare_fold(config_path, fold, overrides, verbose)
    config = context.config
    normalizer = get_or_fit_normalizer(context)
    arguments = {
        "wsi_dim": int(config.data.wsi_dim),
        "patch_dim": int(config.data.patch_dim),
        "num_classes": int(config.data.num_tissue_classes),
        "normalizer": normalizer,
        "minimum_slides": int(config.data.min_slides_for_diffusion),
        "cache_size": int(config.data.feature_cache_size),
    }
    train_dataset = LeaveOneSlideOutDataset(context.partitions["train"], **arguments)
    validation_dataset = LeaveOneSlideOutDataset(context.partitions["validation"], **arguments)
    settings = config.training.wsi_diffuser
    loader_options = _loader_options(config, int(settings.batch_size))
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    device = resolve_device(str(config.experiment.device))
    trainer = DiffusionTrainer(
        model=build_wsi_diffusion(config),
        device=device,
        learning_rate=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
        epochs=int(settings.epochs),
        warmup_epochs=int(settings.warmup_epochs),
        patience=int(settings.patience),
        optimizer_name=str(config.training.optimizer),
        gradient_clip_norm=float(config.training.gradient_clip_norm),
        ema_decay=float(config.training.ema_decay),
        amp=amp_enabled(str(config.experiment.precision), device),
        checkpoint_dir=context.directory / "checkpoints" / "wsi_diffuser",
        checkpoint_every=int(config.training.checkpoint_every),
        logger=context.logger,
        metadata={"stage": "wsi_diffuser", "fold": fold, "dataset": str(config.data.dataset)},
        resume=bool(config.experiment.resume),
    )

    def loss_function(module: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return module.training_loss(
            batch["target_wsi"], batch["target_distribution"], batch["condition"]
        )

    history = trainer.fit(train_loader, validation_loader, loss_function)
    atomic_json_dump(history, context.directory / "history_wsi_diffuser.json")


def _balanced_patch_validation(number_slides: int, training_samples: int) -> Tuple[int, int]:
    if number_slides < 1:
        raise ValueError("Patch validation requires at least one slide")
    desired = min(10000, max(1000, training_samples // 10))
    block_size = max(1, math.ceil(desired / number_slides))
    return number_slides * block_size, block_size


def train_patch(
    config_path: str,
    fold: int,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    context = prepare_fold(config_path, fold, overrides, verbose)
    config = context.config
    normalizer = get_or_fit_normalizer(context)
    settings = config.training.patch_diffuser
    common = {
        "wsi_dim": int(config.data.wsi_dim),
        "patch_dim": int(config.data.patch_dim),
        "num_classes": int(config.data.num_tissue_classes),
        "normalizer": normalizer,
        "cache_size": int(config.data.feature_cache_size),
    }
    train_dataset = PatchDiffusionDataset(
        context.partitions["train"],
        samples_per_epoch=int(settings.samples_per_epoch),
        seed=int(config.experiment.seed) + fold,
        slide_local_block_size=int(settings.slide_local_block_size),
        **common,
    )
    number_validation_slides = sum(len(p.slides) for p in context.partitions["validation"])
    validation_samples, validation_block = _balanced_patch_validation(
        number_validation_slides, int(settings.samples_per_epoch)
    )
    validation_dataset = PatchDiffusionDataset(
        context.partitions["validation"],
        samples_per_epoch=validation_samples,
        seed=int(config.experiment.seed) + fold + 50000,
        slide_local_block_size=validation_block,
        **common,
    )
    options = _loader_options(config, int(settings.batch_size))
    train_loader = DataLoader(train_dataset, shuffle=False, **options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **options)
    device = resolve_device(str(config.experiment.device))
    trainer = DiffusionTrainer(
        model=build_patch_diffusion(config),
        device=device,
        learning_rate=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
        epochs=int(settings.epochs),
        warmup_epochs=int(settings.warmup_epochs),
        patience=int(settings.patience),
        optimizer_name=str(config.training.optimizer),
        gradient_clip_norm=float(config.training.gradient_clip_norm),
        ema_decay=float(config.training.ema_decay),
        amp=amp_enabled(str(config.experiment.precision), device),
        checkpoint_dir=context.directory / "checkpoints" / "patch_diffuser",
        checkpoint_every=int(config.training.checkpoint_every),
        logger=context.logger,
        metadata={"stage": "patch_diffuser", "fold": fold, "dataset": str(config.data.dataset)},
        resume=bool(config.experiment.resume),
    )

    def loss_function(module: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return module.training_loss(batch["patch"], batch["consistency"], batch["tissue_class"])

    history = trainer.fit(train_loader, validation_loader, loss_function)
    atomic_json_dump(history, context.directory / "history_patch_diffuser.json")


def _portable_patient_name(patient_id: str) -> str:
    """Encode an arbitrary patient identifier into a collision-free path component."""

    encoded = base64.urlsafe_b64encode(patient_id.encode("utf-8")).decode("ascii").rstrip("=")
    return "patient_" + (encoded or "empty")


def _generated_paths(index_path: Path) -> List[Path]:
    if not index_path.is_file():
        return []
    try:
        with index_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        patients = payload["patients"]
        paths = [value for values in patients.values() for value in values]
        return [
            (Path(value) if Path(value).is_absolute() else index_path.parent / value).resolve()
            for value in paths
        ]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return []


def generate(
    config_path: str,
    fold: int,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    context = prepare_fold(config_path, fold, overrides, verbose)
    config = context.config
    normalizer = get_or_fit_normalizer(context)
    device = resolve_device(str(config.experiment.device))
    hierarchy = build_hierarchical_generator(config)
    expected_common = {"fold": fold, "dataset": str(config.data.dataset)}
    if not bool(config.generation.skip_wsi_level):
        load_model_checkpoint(
            context.directory / "checkpoints" / "wsi_diffuser" / "best.pt",
            hierarchy.wsi_diffusion,
            expected_metadata={"stage": "wsi_diffuser", **expected_common},
        )
    load_model_checkpoint(
        context.directory / "checkpoints" / "patch_diffuser" / "best.pt",
        hierarchy.patch_diffusion,
        expected_metadata={"stage": "patch_diffuser", **expected_common},
    )
    hierarchy.to(device).eval()

    if str(config.generation.class_sampling) == "empirical_global":
        counts = torch.zeros(int(config.data.num_tissue_classes), dtype=torch.float64)
        for patient in context.partitions["train"]:
            for slide in patient.slides:
                bundle = load_feature_bundle(
                    slide.feature_path,
                    int(config.data.wsi_dim),
                    int(config.data.patch_dim),
                    int(config.data.num_tissue_classes),
                )
                counts += torch.bincount(
                    bundle.tissue_labels,
                    minlength=int(config.data.num_tissue_classes),
                )
        hierarchy.set_fallback_distribution(counts.float())

    generated_root = context.directory / "generated"
    index: Dict[str, Any] = {
        "schema_version": 1,
        "fold": fold,
        "dataset": str(config.data.dataset),
        "strategy": str(config.generation.strategy),
        "patients": {},
        "partitions": {},
    }
    cohort_max = max(
        len(patient.slides)
        for patients in context.partitions.values()
        for patient in patients
    )
    configured_target = config.generation.target_total_slides
    target_total = cohort_max if configured_target in (None, "cohort_max") else int(configured_target)
    if str(config.generation.strategy) == "unequally" and target_total < cohort_max:
        raise ValueError(
            f"unequally requires target_total_slides >= cohort maximum {cohort_max}"
        )
    index["target_total_slides"] = target_total
    index["cohort_max_original_slides"] = cohort_max
    seed = int(config.experiment.seed) + int(config.generation.seed_offset) + fold * 10000
    generator = make_generator(seed, device)

    for partition, patients in context.partitions.items():
        partition_paths: List[str] = []
        index["partitions"][partition] = partition_paths
        for patient in patients:
            patient_paths: List[str] = []
            index["patients"][patient.patient_id] = patient_paths
            count = number_to_generate(
                original_count=len(patient.slides),
                strategy=str(config.generation.strategy),
                fixed_count=int(config.generation.generated_slides_per_patient),
                partial_threshold=int(config.generation.partially_threshold),
                target_total=target_total,
            )
            if count == 0:
                continue
            observed = torch.stack(
                [
                    load_feature_bundle(
                        slide.feature_path,
                        int(config.data.wsi_dim),
                        int(config.data.patch_dim),
                        int(config.data.num_tissue_classes),
                    ).wsi_feature
                    for slide in patient.slides
                ]
            )
            if normalizer is not None:
                observed = normalizer.normalize_wsi(observed)
            slides = hierarchy.generate_patient(observed.to(device), count, generator)
            patient_directory = generated_root / partition / _portable_patient_name(patient.patient_id)
            for generated_index, slide in enumerate(slides):
                path = patient_directory / f"generated_{generated_index:03d}.pt"
                wsi_feature = slide.consistency.detach().cpu()
                patch_features = slide.patch_features.detach().cpu()
                if normalizer is not None:
                    wsi_feature = normalizer.denormalize_wsi(wsi_feature)
                    patch_features = normalizer.denormalize_patch(patch_features)
                save_feature_bundle(
                    FeatureBundle(
                        wsi_feature=wsi_feature,
                        patch_features=patch_features,
                        tissue_labels=slide.tissue_labels.detach().cpu(),
                        metadata={
                            "synthetic": True,
                            "patient_id": patient.patient_id,
                            "fold": fold,
                            "partition": partition,
                            "generated_index": generated_index,
                            "tissue_distribution": slide.distribution.detach().cpu().tolist(),
                            "seed_base": seed,
                        },
                    ),
                    path,
                )
                relative = path.resolve().relative_to(generated_root.resolve()).as_posix()
                patient_paths.append(relative)
                partition_paths.append(relative)
            context.logger.info("generated patient=%s slides=%d", patient.patient_id, count)
    atomic_json_dump(index, generated_root / "index.json")


def _save_predictions(payload: Mapping[str, Any], path: Path, fold: int, dataset: str) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": payload["patient_ids"],
            "time": payload["time"],
            "event": payload["event"],
            "risk": payload["risk"],
            "original_slide_count": payload["original_slide_count"],
            "fold": fold,
            "dataset": dataset,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def train_survival(
    config_path: str,
    fold: int,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> None:
    context = prepare_fold(config_path, fold, overrides, verbose)
    config = context.config
    normalizer = get_or_fit_normalizer(context)
    generated_index: Optional[Path] = context.directory / "generated" / "index.json"
    if bool(config.generation.enabled) and not generated_index.is_file():
        raise FileNotFoundError(f"Generation is enabled but {generated_index} is missing")
    if not bool(config.generation.enabled):
        generated_index = None
    maximum = config.data.max_patches_per_real_slide
    maximum_patches = None if maximum is None else int(maximum)
    datasets = {
        name: SurvivalDataset(
            patients=patients,
            wsi_dim=int(config.data.wsi_dim),
            patch_dim=int(config.data.patch_dim),
            normalizer=normalizer,
            generated_index=generated_index,
            max_patches_per_slide=maximum_patches,
            expected_fold=fold,
            partition=name,
        )
        for name, patients in context.partitions.items()
    }
    settings = config.training.survival
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(settings.batch_size),
            shuffle=(name == "train" and str(settings.cox_risk_set) == "batch"),
            num_workers=int(config.training.num_workers),
            pin_memory=bool(config.training.pin_memory),
            collate_fn=patient_bag_collate,
            worker_init_fn=seed_worker,
        )
        for name, dataset in datasets.items()
    }
    device = resolve_device(str(config.experiment.device))
    model = build_survival_model(config)
    trainer = SurvivalTrainer(
        model=model,
        device=device,
        learning_rate=float(settings.learning_rate),
        weight_decay=float(settings.weight_decay),
        epochs=int(settings.epochs),
        warmup_epochs=int(settings.warmup_epochs),
        patience=int(settings.patience),
        optimizer_name=str(config.training.optimizer),
        gradient_clip_norm=float(config.training.gradient_clip_norm),
        amp=amp_enabled(str(config.experiment.precision), device),
        risk_set_mode=str(settings.cox_risk_set),
        cox_ties=str(settings.cox_ties),
        cox_reduction=str(settings.cox_reduction),
        checkpoint_dir=context.directory / "checkpoints" / "survival",
        checkpoint_every=int(config.training.checkpoint_every),
        logger=context.logger,
        metadata={"stage": "survival", "fold": fold, "dataset": str(config.data.dataset)},
        resume=bool(config.experiment.resume),
    )
    history = trainer.fit(loaders["train"], loaders["validation"])
    atomic_json_dump(history, context.directory / "history_survival.json")
    load_model_checkpoint(
        context.directory / "checkpoints" / "survival" / "best.pt",
        model,
        use_ema=False,
        expected_metadata={
            "stage": "survival",
            "fold": fold,
            "dataset": str(config.data.dataset),
        },
    )
    for name, loader in loaders.items():
        _save_predictions(
            trainer.predict(loader),
            context.directory / "predictions" / f"{name}.csv",
            fold,
            str(config.data.dataset),
        )


def _validate_predictions(frame: pd.DataFrame, context: FoldContext, partition: str) -> None:
    required = {
        "patient_id",
        "time",
        "event",
        "risk",
        "original_slide_count",
        "fold",
        "dataset",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")
    if frame["patient_id"].duplicated().any():
        raise ValueError(f"Prediction CSV contains duplicate {partition} patients")
    if not np.isfinite(frame[["time", "risk"]].to_numpy(float)).all():
        raise ValueError("Prediction CSV contains non-finite time or risk values")
    expected = {patient.patient_id: patient for patient in context.partitions[partition]}
    if set(frame["patient_id"].astype(str)) != set(expected):
        raise ValueError(f"Prediction patients do not match the {partition} partition")
    if set(frame["fold"].astype(int)) != {context.fold}:
        raise ValueError("Prediction CSV belongs to a different fold")
    if set(frame["dataset"].astype(str)) != {str(context.config.data.dataset)}:
        raise ValueError("Prediction CSV belongs to a different dataset")
    indexed = frame.assign(patient_id=frame["patient_id"].astype(str)).set_index("patient_id")
    for patient_id, patient in expected.items():
        row = indexed.loc[patient_id]
        if not np.isclose(float(row["time"]), patient.time, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"Survival time mismatch for {patient_id}")
        if int(row["event"]) != patient.event:
            raise ValueError(f"Event indicator mismatch for {patient_id}")
        if int(row["original_slide_count"]) != len(patient.slides):
            raise ValueError(f"Slide count mismatch for {patient_id}")


def evaluate(
    config_path: str,
    fold: int,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    context = prepare_fold(config_path, fold, overrides, verbose)
    prediction_dir = context.directory / "predictions"
    train = pd.read_csv(prediction_dir / "train.csv")
    test = pd.read_csv(prediction_dir / "test.csv")
    _validate_predictions(train, context, "train")
    _validate_predictions(test, context, "test")
    train_time = train["time"].to_numpy(float)
    train_event = train["event"].to_numpy(int)
    train_risk = train["risk"].to_numpy(float)
    test_time = test["time"].to_numpy(float)
    test_event = test["event"].to_numpy(int)
    test_risk = test["risk"].to_numpy(float)
    c_result = concordance_index(test_time, test_event, test_risk)

    stage_definition = str(context.config.evaluation.stage5.definition)
    exclude_censored = (
        str(context.config.evaluation.stage5.censored_policy) == "exclude_ambiguous"
    )
    if stage_definition == "fixed_year_accuracy":
        predicted_time = BreslowBaseline().fit(
            train_time, train_event, train_risk
        ).median_survival_time(test_risk)
        stage = fixed_time_stage_accuracy(
            test_time,
            test_event,
            predicted_time,
            list(context.config.evaluation.stage5.boundaries_days),
            exclude_censored=exclude_censored,
        )
    elif stage_definition == "quantile_accuracy":
        metric = QuantileStageMetric(
            int(context.config.evaluation.stage5.num_bins),
            exclude_censored=exclude_censored,
        ).fit(train_time, train_event, train_risk)
        stage = metric.score(test_time, test_event, test_risk)
    else:
        raise ValueError(f"Unknown STAGE-5 definition {stage_definition!r}")

    interval = bootstrap_metric(
        (test_time, test_event, test_risk),
        lambda time, event, risk: concordance_index(time, event, risk).c_index,
        int(context.config.evaluation.bootstrap_samples),
        float(context.config.evaluation.bootstrap_confidence),
        int(context.config.experiment.seed) + fold,
    )
    bias = slide_count_bias_metrics(
        test_time,
        test_event,
        test_risk,
        test["original_slide_count"].to_numpy(int),
        context.config.evaluation.bias_groups.to_dict(),
    )
    metrics: Dict[str, Any] = {
        "fold": fold,
        "dataset": str(context.config.data.dataset),
        "c_index": c_result.c_index,
        "c_index_comparable_pairs": c_result.comparable,
        "c_index_ci_lower": interval.lower,
        "c_index_ci_upper": interval.upper,
        "c_index_bootstrap_successful_samples": interval.successful_samples,
        "stage5_accuracy": stage.accuracy,
        "stage5_definition": stage.definition,
        "stage5_evaluated": stage.evaluated,
        "stage5_excluded_censored": stage.excluded_censored,
        "stage5_time_boundaries": stage.time_boundaries,
        **bias,
    }
    atomic_json_dump(metrics, context.directory / "metrics.json")
    context.logger.info(
        "fold=%d c_index=%.6f stage5=%.6f", fold, c_result.c_index, stage.accuracy
    )
    return metrics


def _generation_artifacts(index_path: Path) -> List[Path]:
    """Return the complete generated bundle inventory, or an empty list if invalid."""

    if not index_path.is_file():
        return []
    try:
        with index_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("patients"), dict):
            return []
        values = [path for paths in payload["patients"].values() for path in paths]
        if not all(isinstance(path, str) and path for path in values):
            return []
        resolved = [
            (Path(path) if Path(path).is_absolute() else index_path.parent / path).resolve()
            for path in values
        ]
        return [index_path.resolve(), *resolved]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return []


def _stage_outputs(config: Config, directory: Path, stage: str) -> List[Path]:
    normalizer = [directory / "normalizer.pt"] if bool(config.normalization.enabled) else []
    if stage == "train_wsi":
        return [
            directory / "checkpoints" / "wsi_diffuser" / "best.pt",
            directory / "checkpoints" / "wsi_diffuser" / "last.pt",
            directory / "history_wsi_diffuser.json",
            *normalizer,
        ]
    if stage == "train_patch":
        return [
            directory / "checkpoints" / "patch_diffuser" / "best.pt",
            directory / "checkpoints" / "patch_diffuser" / "last.pt",
            directory / "history_patch_diffuser.json",
            *normalizer,
        ]
    if stage == "generate":
        return _generation_artifacts(directory / "generated" / "index.json")
    if stage == "train_survival":
        return [
            directory / "checkpoints" / "survival" / "best.pt",
            directory / "checkpoints" / "survival" / "last.pt",
            directory / "history_survival.json",
            directory / "predictions" / "train.csv",
            directory / "predictions" / "validation.csv",
            directory / "predictions" / "test.csv",
            *normalizer,
        ]
    if stage == "evaluate":
        return [directory / "metrics.json"]
    raise KeyError(f"Unknown pipeline stage {stage!r}")


def _selected_folds(config: Config, folds: Optional[Sequence[int]]) -> List[int]:
    expected = list(range(int(config.split.n_folds)))
    selected = expected if folds is None else list(folds)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Fold selection must be non-empty and contain unique IDs")
    invalid = sorted(set(selected).difference(expected))
    if invalid:
        raise ValueError(f"Invalid folds {invalid}; configured folds are {expected}")
    return selected


def run_pipeline(
    config_path: str,
    folds: Optional[List[int]] = None,
    overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> Path:
    config = load_config(config_path, overrides)
    selected = _selected_folds(config, folds)
    split_path = project_path(config, config.data.split_file)
    if not split_path.is_file():
        make_splits(config_path, overrides)

    stages: List[Tuple[str, Callable[..., Any]]] = []
    if bool(config.generation.enabled):
        if not bool(config.generation.skip_wsi_level):
            stages.append(("train_wsi", train_wsi))
        stages.extend([("train_patch", train_patch), ("generate", generate)])
    stages.extend([("train_survival", train_survival), ("evaluate", evaluate)])

    for fold in selected:
        context = prepare_fold(config_path, fold, overrides, verbose)
        upstream_ran = False
        for stage_name, stage_function in stages:
            outputs = _stage_outputs(config, context.directory, stage_name)
            completed = bool(outputs) and stage_is_complete(
                context.directory, stage_name, outputs
            )
            if bool(config.experiment.resume) and not upstream_ran and completed:
                context.logger.info("skipping completed stage %s", stage_name)
                continue
            stage_function(config_path, fold, overrides, verbose)
            outputs = _stage_outputs(config, context.directory, stage_name)
            if not outputs:
                raise RuntimeError(f"Stage {stage_name} did not produce a valid output inventory")
            mark_stage_complete(
                context.directory,
                stage_name,
                outputs,
                {"fold": fold, "dataset": str(config.data.dataset)},
            )
            upstream_ran = True

    output_root = project_path(config, config.experiment.output_dir)
    expected = list(range(int(config.split.n_folds)))
    aggregate_name = (
        "cross_validation_metrics.json"
        if selected == expected
        else "cross_validation_metrics_folds_" + "-".join(map(str, selected)) + ".json"
    )
    destination = output_root / aggregate_name
    write_aggregate([fold_directory(config, fold) / "metrics.json" for fold in selected], destination)
    return destination


def _load_experiment_book(path: str) -> Tuple[Path, Dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Experiment file must be a schema_version=1 mapping")
    root = source.parent
    for candidate in [source.parent, *source.parents]:
        if (candidate / "pyproject.toml").is_file():
            root = candidate
            break
    return root, payload


def _matrix_variants(specification: Mapping[str, Any]) -> List[Tuple[str, List[str]]]:
    if "variants" in specification:
        variants = specification["variants"]
        if not isinstance(variants, dict) or not variants:
            raise ValueError("Matrix variants must be a non-empty mapping")
        return [(str(name), [str(value) for value in values]) for name, values in variants.items()]
    grid = specification.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("A matrix needs variants or a non-empty grid")
    keys = list(grid)
    if any(not isinstance(grid[key], list) or not grid[key] for key in keys):
        raise ValueError("Every grid parameter needs a non-empty value list")
    result: List[Tuple[str, List[str]]] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        pieces = [f"{key.rsplit('.', 1)[-1]}-{value}" for key, value in zip(keys, values)]
        overrides = [f"{key}={value}" for key, value in zip(keys, values)]
        slide_key = "generation.generated_slides_per_patient"
        if slide_key in keys:
            slide_count = int(values[keys.index(slide_key)])
            overrides.append(f"generation.enabled={'false' if slide_count == 0 else 'true'}")
        result.append(("__".join(pieces), overrides))
    return result


def run_matrix(
    config_path: str,
    experiments_path: str,
    name: str,
    folds: Optional[List[int]] = None,
    user_overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> Path:
    root, book = _load_experiment_book(experiments_path)
    matrices = book.get("matrices")
    datasets = book.get("datasets")
    if not isinstance(matrices, dict) or name not in matrices:
        raise KeyError(f"Unknown experiment matrix {name!r}")
    if not isinstance(datasets, dict):
        raise ValueError("Experiment file has no datasets mapping")
    specification = matrices[name]
    cohort_names = specification.get("datasets")
    if not isinstance(cohort_names, list) or not cohort_names:
        raise ValueError("Matrix datasets must be a non-empty list")
    variants = _matrix_variants(specification)
    records: List[Dict[str, Any]] = []
    for cohort_name in cohort_names:
        if cohort_name not in datasets:
            raise KeyError(f"Unknown dataset {cohort_name!r} in matrix {name}")
        cohort = datasets[cohort_name]
        required = {"dataset", "manifest", "split_file"}
        if not isinstance(cohort, dict) or not required.issubset(cohort):
            raise ValueError(f"Dataset {cohort_name!r} is missing {sorted(required)}")
        for variant_name, scientific in variants:
            output = f"outputs/paper_experiments/{name}/{cohort_name}/{variant_name}"
            overrides = [
                f"data.dataset={cohort['dataset']}",
                f"data.manifest={cohort['manifest']}",
                f"data.split_file={cohort['split_file']}",
                f"experiment.name={name}_{cohort_name}_{variant_name}",
                f"experiment.output_dir={output}",
                *scientific,
                *(user_overrides or []),
            ]
            run_pipeline(config_path, folds, overrides, verbose)
            records.append(
                {
                    "dataset": cohort_name,
                    "variant": variant_name,
                    "scientific_overrides": scientific,
                    "output_dir": str((root / output).resolve()),
                }
            )
    destination = root / "outputs" / "paper_experiments" / name / "matrix_index.json"
    atomic_json_dump(
        {
            "schema_version": 1,
            "name": name,
            "experiments_file": str(Path(experiments_path).resolve()),
            "runs": records,
        },
        destination,
    )
    return destination


def run_sweep(
    config_path: str,
    kind: str,
    folds: Optional[List[int]] = None,
    user_overrides: Optional[List[str]] = None,
    verbose: bool = False,
) -> Path:
    base = load_config(config_path, user_overrides)
    jobs: List[Tuple[str, List[str]]] = []
    if kind == "timesteps":
        jobs = [(f"t{step}", [f"diffusion.patch_level.timesteps={step}"]) for step in (10, 20, 30, 40, 50)]
    elif kind == "generation-grid":
        for slides, patches in itertools.product(
            base.evaluation.sensitivity_grid.generated_slides,
            base.evaluation.sensitivity_grid.generated_patches,
        ):
            jobs.append(
                (
                    f"g{slides}_m{patches}",
                    [
                        f"generation.enabled={'false' if int(slides) == 0 else 'true'}",
                        f"generation.generated_slides_per_patient={int(slides)}",
                        f"generation.patches_per_generated_slide={int(patches)}",
                    ],
                )
            )
    elif kind == "strategies":
        jobs = [(strategy, [f"generation.strategy={strategy}"]) for strategy in ("wholly", "partially", "unequally")]
    else:
        raise ValueError("kind must be timesteps, generation-grid, or strategies")
    records = []
    for job_name, scientific in jobs:
        output = f"outputs/{base.experiment.name}_sensitivity/{kind}/{job_name}"
        overrides = [
            *(user_overrides or []),
            f"experiment.name={base.experiment.name}_{kind}_{job_name}",
            f"experiment.output_dir={output}",
            *scientific,
        ]
        run_pipeline(config_path, folds, overrides, verbose)
        records.append({"name": job_name, "overrides": scientific, "output_dir": output})
    destination = project_path(
        base, f"outputs/{base.experiment.name}_sensitivity/{kind}/sweep_index.json"
    )
    atomic_json_dump({"kind": kind, "runs": records}, destination)
    return destination


def case_study(
    config_path: str,
    fold: int,
    patient_id: str,
    output: str,
    figure: Optional[str] = None,
    max_points_per_slide: int = 2000,
    overrides: Optional[List[str]] = None,
) -> None:
    from sklearn.manifold import TSNE

    context = prepare_fold(config_path, fold, overrides)
    patients = {patient.patient_id: patient for patient in group_patients(context.records)}
    if patient_id not in patients:
        raise KeyError(f"Unknown patient {patient_id!r}")
    index_path = context.directory / "generated" / "index.json"
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != 1 or int(index.get("fold", -1)) != fold:
        raise ValueError("Generated index has an incompatible schema or fold")
    if index.get("dataset") != str(context.config.data.dataset):
        raise ValueError("Generated index belongs to a different dataset")

    bags: List[Tuple[str, np.ndarray]] = []
    for slide in patients[patient_id].slides:
        features = load_feature_bundle(
            slide.feature_path,
            num_classes=int(context.config.data.num_tissue_classes),
        ).patch_features.numpy()
        bags.append((slide.slide_id, features))
    for generated_index, value in enumerate(index["patients"].get(patient_id, [])):
        path = (Path(value) if Path(value).is_absolute() else index_path.parent / value).resolve()
        bundle = load_feature_bundle(path, num_classes=int(context.config.data.num_tissue_classes))
        if bundle.metadata.get("synthetic") is not True:
            raise ValueError(f"Generated bundle is not marked synthetic: {path}")
        if bundle.metadata.get("patient_id") != patient_id or int(bundle.metadata.get("fold", -1)) != fold:
            raise ValueError(f"Generated bundle metadata mismatch: {path}")
        bags.append((f"generated_{generated_index}", bundle.patch_features.numpy()))
    sampled: List[Tuple[str, np.ndarray]] = []
    for label, values in bags:
        if values.shape[0] > max_points_per_slide:
            indices = np.linspace(0, values.shape[0] - 1, max_points_per_slide, dtype=int)
            values = values[indices]
        sampled.append((label, values))
    values = np.concatenate([features for _, features in sampled])
    labels = np.concatenate([np.repeat(label, len(features)) for label, features in sampled])
    if len(values) < 3:
        raise ValueError("The case study needs at least three patch vectors")
    perplexity = min(30.0, max(2.0, (len(values) - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=int(context.config.experiment.seed),
    ).fit_transform(values)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": embedding[:, 0], "y": embedding[:, 1], "source": labels}).to_csv(
        destination, index=False
    )
    if figure is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError("Install the plots extra to render a case-study figure") from error
        figure_path = Path(figure)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure_object, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
        palette = plt.get_cmap("tab20")
        for source_index, source in enumerate(dict.fromkeys(labels.tolist())):
            mask = labels == source
            generated = str(source).startswith("generated_")
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=14 if generated else 10,
                alpha=0.75,
                marker="X" if generated else "o",
                color=palette(source_index % 20),
                label=str(source),
                linewidths=0,
            )
        axis.set(title=f"Patient {patient_id}: real vs generated patches", xlabel="t-SNE 1", ylabel="t-SNE 2")
        axis.legend(frameon=False, markerscale=1.4, fontsize=8)
        figure_object.savefig(figure_path, dpi=300)
        plt.close(figure_object)


def _add_config_arguments(
    parser: argparse.ArgumentParser,
    one_fold: bool = False,
    many_folds: bool = False,
    verbose: bool = True,
) -> None:
    parser.add_argument("--config", required=True, help="Path to the main YAML configuration")
    if one_fold:
        parser.add_argument("--fold", type=int, required=True, help="Outer fold index")
    if many_folds:
        parser.add_argument(
            "--folds",
            type=int,
            nargs="+",
            help="Outer fold indices; omitted means all configured folds",
        )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Strict dotted configuration override; repeat as needed",
    )
    if verbose:
        parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsi-diffusion",
        description="Official hierarchical WSI diffusion and survival-prediction pipeline",
    )
    parser.add_argument("--version", action="version", version="WSI-Diffusion 1.0.0")
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate", help="Validate manifest and features")
    _add_config_arguments(validate_parser, verbose=False)

    split_parser = commands.add_parser("split", help="Create patient-disjoint folds")
    _add_config_arguments(split_parser, verbose=False)
    split_parser.add_argument("--force", action="store_true", help="Overwrite a different split")

    preprocess_parser = commands.add_parser("preprocess", help="Extract WSI feature bundles")
    _add_config_arguments(preprocess_parser, verbose=False)
    preprocess_parser.add_argument("--patch-model", required=True, help="TorchScript patch encoder")
    preprocess_parser.add_argument("--wsi-model", help="Optional TorchScript WSI encoder")

    for command, help_text in (
        ("train-wsi", "Train the joint WSI-level diffuser"),
        ("train-patch", "Train the tissue-conditioned patch diffuser"),
        ("generate", "Generate synthetic WSI patch bags"),
        ("train-survival", "Train the hierarchical C-MIL Cox model"),
        ("evaluate", "Evaluate one held-out fold"),
    ):
        stage_parser = commands.add_parser(command, help=help_text)
        _add_config_arguments(stage_parser, one_fold=True)

    run_parser = commands.add_parser("run", help="Run the complete cross-validation pipeline")
    _add_config_arguments(run_parser, many_folds=True)

    matrix_parser = commands.add_parser("matrix", help="Run a named paper experiment matrix")
    _add_config_arguments(matrix_parser, many_folds=True)
    matrix_parser.add_argument("--experiments", required=True, help="Paper experiment YAML")
    matrix_parser.add_argument("--name", required=True, help="Matrix name in the experiment file")

    sweep_parser = commands.add_parser("sweep", help="Run one sensitivity family")
    _add_config_arguments(sweep_parser, many_folds=True)
    sweep_parser.add_argument(
        "--kind",
        required=True,
        choices=["timesteps", "generation-grid", "strategies"],
    )

    case_parser = commands.add_parser("case-study", help="Joint real/generated t-SNE")
    _add_config_arguments(case_parser, one_fold=True, verbose=False)
    case_parser.add_argument("--patient-id", required=True)
    case_parser.add_argument("--output", required=True, help="Output coordinate CSV")
    case_parser.add_argument("--figure", help="Optional PNG/PDF output")
    case_parser.add_argument("--max-points-per-slide", type=int, default=2000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate_data(args.config, args.overrides), indent=2, ensure_ascii=False))
    elif args.command == "split":
        print(make_splits(args.config, args.overrides, args.force))
    elif args.command == "preprocess":
        preprocess(args.config, args.patch_model, args.wsi_model, args.overrides)
    elif args.command == "train-wsi":
        train_wsi(args.config, args.fold, args.overrides, args.verbose)
    elif args.command == "train-patch":
        train_patch(args.config, args.fold, args.overrides, args.verbose)
    elif args.command == "generate":
        generate(args.config, args.fold, args.overrides, args.verbose)
    elif args.command == "train-survival":
        train_survival(args.config, args.fold, args.overrides, args.verbose)
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate(args.config, args.fold, args.overrides, args.verbose),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "run":
        print(run_pipeline(args.config, args.folds, args.overrides, args.verbose))
    elif args.command == "matrix":
        print(
            run_matrix(
                args.config,
                args.experiments,
                args.name,
                args.folds,
                args.overrides,
                args.verbose,
            )
        )
    elif args.command == "sweep":
        print(run_sweep(args.config, args.kind, args.folds, args.overrides, args.verbose))
    elif args.command == "case-study":
        if args.max_points_per_slide < 1:
            raise ValueError("--max-points-per-slide must be positive")
        case_study(
            args.config,
            args.fold,
            args.patient_id,
            args.output,
            args.figure,
            args.max_points_per_slide,
            args.overrides,
        )
    else:
        raise AssertionError(f"Unhandled command {args.command!r}")
    return 0


__all__ = [
    "FoldContext",
    "build_parser",
    "case_study",
    "evaluate",
    "generate",
    "get_or_fit_normalizer",
    "main",
    "make_splits",
    "prepare_fold",
    "preprocess",
    "run_matrix",
    "run_pipeline",
    "run_sweep",
    "train_patch",
    "train_survival",
    "train_wsi",
    "validate_data",
]


if __name__ == "__main__":
    raise SystemExit(main())
