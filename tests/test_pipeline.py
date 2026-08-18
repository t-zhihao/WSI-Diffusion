"""Configuration, metric and orchestration tests for the compact public API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsi_diffusion import cli
from wsi_diffusion.config import load_config
from wsi_diffusion.engine import (
    EarlyStopping,
    aggregate_metric_files,
    atomic_json_dump,
    concordance_index,
    mark_stage_complete,
    stage_is_complete,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_default_config_and_strict_override() -> None:
    config = load_config(
        ROOT / "configs" / "default.yaml",
        ["diffusion.patch_level.timesteps=10"],
    )
    assert config.diffusion.patch_level.timesteps == 10
    assert config.diffusion.wsi_level.timesteps == 30
    assert config.data.num_tissue_classes == 6
    with pytest.raises(KeyError, match="Unknown override"):
        load_config(ROOT / "configs" / "default.yaml", ["diffusion.timestpes=10"])


def test_single_command_parser_exposes_all_stages() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["train-patch", "--config", "config.yaml", "--fold", "2"])
    assert args.command == "train-patch"
    assert args.fold == 2
    matrix = parser.parse_args(
        [
            "matrix",
            "--config",
            "config.yaml",
            "--experiments",
            "experiments.yaml",
            "--name",
            "table4_ablation",
        ]
    )
    assert matrix.name == "table4_ablation"


def test_stage_marker_requires_declared_outputs(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    marker = mark_stage_complete(tmp_path, "train", [artifact], {"fold": 0})
    assert marker.is_file()
    assert stage_is_complete(tmp_path, "train", [artifact])
    artifact.unlink()
    assert not stage_is_complete(tmp_path, "train", [artifact])
    with pytest.raises(FileNotFoundError):
        mark_stage_complete(tmp_path, "missing", [artifact])


def test_json_is_standard_and_non_finite_values_become_null(tmp_path: Path) -> None:
    destination = tmp_path / "metrics.json"
    atomic_json_dump({"finite": 1.0, "missing": float("nan")}, destination)
    with destination.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload == {"finite": 1.0, "missing": None}
    assert "NaN" not in destination.read_text(encoding="utf-8")


def test_c_index_and_early_stopping_reject_non_finite_selection() -> None:
    result = concordance_index(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1, 1, 0]),
        np.asarray([3.0, 2.0, 1.0]),
    )
    assert result.c_index == pytest.approx(1.0)
    with pytest.raises(ValueError, match="finite"):
        concordance_index(
            np.asarray([1.0, 2.0]),
            np.asarray([1, 0]),
            np.asarray([float("nan"), 0.0]),
        )
    stopper = EarlyStopping(patience=2, mode="max")
    improved, stopped = stopper.update(float("nan"))
    assert not improved and not stopped and stopper.best is None
    improved, _ = stopper.update(0.7)
    assert improved and stopper.best == pytest.approx(0.7)


def test_fold_aggregation_tracks_ids_and_only_aggregates_metrics(tmp_path: Path) -> None:
    paths = []
    for fold, score, count in ((0, 0.6, 12), (1, 0.8, 21)):
        path = tmp_path / f"fold_{fold}.json"
        atomic_json_dump(
            {
                "fold": fold,
                "dataset": "TEST",
                "c_index": score,
                "c_index_comparable_pairs": count,
                "stage5_accuracy": score - 0.1,
            },
            path,
        )
        paths.append(path)
    aggregate = aggregate_metric_files(paths)
    assert aggregate["fold_ids"] == [0, 1]
    assert aggregate["c_index"]["mean"] == pytest.approx(0.7)
    assert "c_index_comparable_pairs" not in aggregate
    with pytest.raises(ValueError, match="duplicate fold"):
        aggregate_metric_files([paths[0], paths[0]])


def test_matrix_grid_expansion_disables_zero_slide_generation() -> None:
    variants = cli._matrix_variants(
        {
            "grid": {
                "generation.generated_slides_per_patient": [0, 3],
                "generation.patches_per_generated_slide": [100],
            }
        }
    )
    assert len(variants) == 2
    assert "generation.enabled=false" in variants[0][1]
    assert "generation.enabled=true" in variants[1][1]
