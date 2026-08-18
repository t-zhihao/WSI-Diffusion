"""Scientific data-contract tests for the compact official implementation."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

import wsi_diffusion.data as data


def _write_manifest(tmp_path, rows: list[dict[str, object]]):
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _bundle(wsi: list[float], patches: list[list[float]]) -> data.FeatureBundle:
    patch_tensor = torch.tensor(patches, dtype=torch.float32)
    return data.FeatureBundle(
        wsi_feature=torch.tensor(wsi, dtype=torch.float32),
        patch_features=patch_tensor,
        tissue_labels=torch.arange(patch_tensor.shape[0], dtype=torch.long) % 2,
        coordinates=torch.arange(patch_tensor.shape[0] * 2).reshape(-1, 2),
        metadata={"encoder": "unit-test"},
    )


def test_manifest_normalizes_event_semantics_once_and_groups_patients(tmp_path):
    rows = [
        {
            "dataset": "COHORT",
            "patient_id": "P01",
            "slide_id": "S01",
            "feature_path": "features/S01.pt",
            "time": 12.5,
            "event": 1,
        },
        {
            "dataset": "COHORT",
            "patient_id": "P01",
            "slide_id": "S02",
            "feature_path": "features/S02.pt",
            "time": 12.5,
            "event": 1,
        },
    ]
    manifest = _write_manifest(tmp_path, rows)

    observed = data.read_manifest(manifest, event_semantics="observed")
    censored = data.read_manifest(manifest, event_semantics="censored")

    assert [record.event for record in observed] == [1, 1]
    assert [record.event for record in censored] == [0, 0]
    assert observed[0].feature_path == (tmp_path / "features/S01.pt").resolve()
    patient = data.group_patients(censored)[0]
    assert patient.patient_id == "P01"
    assert patient.event == 0
    assert [slide.slide_id for slide in patient.slides] == ["S01", "S02"]


def test_manifest_rejects_patient_label_conflicts_and_split_leakage(tmp_path):
    rows = [
        {
            "dataset": "COHORT",
            "patient_id": "P01",
            "slide_id": "S01",
            "feature_path": "S01.pt",
            "time": 10,
            "event": 1,
        },
        {
            "dataset": "COHORT",
            "patient_id": "P01",
            "slide_id": "S02",
            "feature_path": "S02.pt",
            "time": 11,
            "event": 1,
        },
    ]
    with pytest.raises(ValueError, match="inconsistent survival labels"):
        data.read_manifest(_write_manifest(tmp_path, rows))

    leaking_fold = {
        "fold": 0,
        "train": ["P01", "P02"],
        "validation": ["P02"],
        "test": ["P03"],
    }
    with pytest.raises(ValueError, match="leakage"):
        data.assert_patient_disjoint(leaking_fold)


def test_feature_bundle_schema_round_trip_and_rejects_unversioned_artifact(tmp_path):
    destination = tmp_path / "slide.pt"
    original = _bundle([1.0, 2.0], [[3.0, 4.0], [5.0, 6.0]])
    data.save_feature_bundle(original, destination)

    loaded = data.load_feature_bundle(
        destination,
        wsi_dim=2,
        patch_dim=2,
        num_classes=2,
    )
    assert loaded.metadata["schema_version"] == data.FEATURE_SCHEMA_VERSION
    assert loaded.wsi_feature.dtype == torch.float32
    assert loaded.tissue_labels.dtype == torch.long
    torch.testing.assert_close(loaded.patch_features, original.patch_features)
    torch.testing.assert_close(loaded.coordinates, original.coordinates)

    invalid = tmp_path / "unversioned.pt"
    data.atomic_torch_save(
        {
            "wsi_feature": torch.zeros(2),
            "patch_features": torch.zeros(1, 2),
            "tissue_labels": torch.zeros(1, dtype=torch.long),
        },
        invalid,
    )
    with pytest.raises(ValueError, match="missing schema_version"):
        data.load_feature_bundle(invalid, wsi_dim=2, patch_dim=2)


def test_normalizer_is_fit_from_supplied_training_records_only(tmp_path):
    train_values = [
        ([1.0, 3.0], [[1.0, 2.0], [3.0, 4.0]]),
        ([3.0, 7.0], [[5.0, 6.0], [7.0, 8.0]]),
    ]
    records: list[data.SlideRecord] = []
    for index, (wsi, patches) in enumerate(train_values):
        path = tmp_path / f"train-{index}.pt"
        data.save_feature_bundle(_bundle(wsi, patches), path)
        records.append(
            data.SlideRecord(
                dataset="COHORT",
                patient_id=f"P{index}",
                slide_id=f"S{index}",
                feature_path=path,
                time=float(index + 1),
                event=index % 2,
            )
        )

    # This deliberately extreme held-out feature is never passed to fit_normalizer.
    data.save_feature_bundle(
        _bundle([10_000.0, 20_000.0], [[30_000.0, 40_000.0]]),
        tmp_path / "held-out.pt",
    )
    normalizer = data.fit_normalizer(records, wsi_dim=2, patch_dim=2)

    torch.testing.assert_close(normalizer.wsi_mean, torch.tensor([2.0, 5.0]))
    torch.testing.assert_close(
        normalizer.wsi_std,
        torch.tensor([math.sqrt(2.0), math.sqrt(8.0)]),
    )
    torch.testing.assert_close(normalizer.patch_mean, torch.tensor([4.0, 5.0]))
    expected_patch_std = torch.full((2,), math.sqrt(20.0 / 3.0))
    torch.testing.assert_close(normalizer.patch_std, expected_patch_std)
    values = torch.tensor([[2.5, 6.0], [1.0, 3.0]])
    torch.testing.assert_close(
        normalizer.denormalize_wsi(normalizer.normalize_wsi(values)),
        values,
    )


def test_largest_remainder_tissue_allocation_preserves_exact_quota():
    probabilities = torch.tensor([0.5, 0.3, 0.2, 0.0])
    counts = data.largest_remainder_counts(probabilities, total=11)
    assert counts.tolist() == [6, 3, 2, 0]
    assert int(counts.sum()) == 11

    guarded = data.largest_remainder_counts(
        torch.tensor([0.5, 0.5, 0.0]),
        total=5,
        minimum_per_nonzero_class=1,
    )
    assert guarded.tolist() == [3, 2, 0]
    labels = data.expand_class_counts(guarded, shuffle=False)
    assert labels.tolist() == [0, 0, 0, 1, 1]


def test_hovernet_spatial_join_votes_across_grid_cell_boundaries():
    nuclei = [
        data.NucleusPrediction(x=9.0, y=4.0, nucleus_type=1, probability=0.9),
        data.NucleusPrediction(x=12.0, y=4.0, nucleus_type=2, probability=0.5),
        data.NucleusPrediction(x=16.0, y=7.0, nucleus_type=2, probability=0.6),
        # Outside the first patch and therefore irrelevant to its majority.
        data.NucleusPrediction(x=18.0, y=4.0, nucleus_type=1, probability=1.0),
    ]
    coordinates = torch.tensor([[8, 0], [100, 100]], dtype=torch.long)
    labels = data.label_patch_coordinates(
        nuclei=nuclei,
        coordinates=coordinates,
        patch_size=10,
        level_downsample=1.0,
        hovernet_type_to_tissue={1: 0, 2: 4},
    )

    assert labels.tolist() == [4, data.TISSUE_TO_INDEX["no_label"]]


def test_hovernet_vote_breaks_count_ties_by_probability_then_class_index():
    nuclei = [
        data.NucleusPrediction(1.0, 1.0, nucleus_type=1, probability=0.4),
        data.NucleusPrediction(2.0, 2.0, nucleus_type=2, probability=0.8),
    ]
    assert data.majority_vote_patch(nuclei, 0, 0, 10, {1: 3, 2: 2}) == 2
    equal_probability = [
        data.NucleusPrediction(1.0, 1.0, nucleus_type=1, probability=0.5),
        data.NucleusPrediction(2.0, 2.0, nucleus_type=2, probability=0.5),
    ]
    assert data.majority_vote_patch(equal_probability, 0, 0, 10, {1: 3, 2: 2}) == 2
