"""Mathematical and shape tests for the compact official model API."""

from __future__ import annotations

import math

import pytest
import torch

import wsi_diffusion.data as data
import wsi_diffusion.models as models


def _tiny_hierarchy(timesteps: int = 2):
    wsi_denoiser = models.JointWSIDenoiser(
        wsi_dim=4,
        num_classes=3,
        hidden_dim=8,
        condition_dim=6,
        time_dim=6,
        num_blocks=1,
        dropout=0.0,
    )
    wsi = models.WSILevelDiffusion(
        denoiser=wsi_denoiser,
        consistency_schedule=models.DiffusionSchedule(timesteps, name="linear"),
    )
    conditioner = models.WSIClassConditioner(4, 3, 6)
    patch = models.PatchLevelDiffusion(
        conditioner=conditioner,
        denoiser=models.PatchDenoiser(
            patch_dim=5,
            condition_dim=6,
            hidden_dim=8,
            time_dim=6,
            num_blocks=1,
            dropout=0.0,
        ),
        schedule=models.DiffusionSchedule(timesteps, name="linear"),
    )
    return wsi, patch


def test_schedules_are_valid_and_forward_equation_is_invertible():
    linear = models.make_beta_schedule("linear", 5, 1.0e-4, 2.0e-2)
    cosine = models.make_beta_schedule("cosine", 5)
    assert linear.shape == cosine.shape == (5,)
    torch.testing.assert_close(linear[[0, -1]], torch.tensor([1.0e-4, 2.0e-2]))
    assert bool(((cosine > 0) & (cosine < 1)).all())

    schedule = models.DiffusionSchedule(5, name="cosine")
    assert bool((schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]).all())
    clean = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    noise = torch.tensor([[0.2, 0.4], [-1.0, 0.3]])
    timestep = torch.tensor([0, 4])
    noisy = schedule.q_sample(clean, timestep, noise)
    expected = (
        schedule.extract(schedule.sqrt_alpha_bars, timestep, clean) * clean
        + schedule.extract(schedule.sqrt_one_minus_alpha_bars, timestep, clean)
        * noise
    )
    torch.testing.assert_close(noisy, expected)
    torch.testing.assert_close(schedule.predict_x0(noisy, timestep, noise), clean)


def test_ddpm_step_at_zero_returns_exact_reconstructed_sample():
    schedule = models.DiffusionSchedule(4, name="linear")
    noisy = torch.tensor([[0.5, -1.0], [2.0, 3.0]])
    predicted_noise = torch.tensor([[0.1, 0.2], [-0.4, 0.8]])
    timestep = torch.zeros(2, dtype=torch.long)
    sample, predicted_x0 = models.ddpm_step(
        noisy,
        predicted_noise,
        timestep,
        schedule,
    )
    torch.testing.assert_close(sample, predicted_x0, rtol=0, atol=0)
    torch.testing.assert_close(
        predicted_x0,
        schedule.predict_x0(noisy, timestep, predicted_noise),
    )


def test_joint_and_patch_diffusion_losses_and_samples_have_expected_shapes():
    wsi, patch = _tiny_hierarchy()
    generator = torch.Generator().manual_seed(7)
    consistency = torch.randn(3, 4, generator=generator)
    distribution = torch.tensor(
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.0, 0.4, 0.6]]
    )
    condition = torch.randn(3, 4, generator=generator)

    wsi_loss = wsi.training_loss(
        consistency,
        distribution,
        condition,
        generator,
    )
    assert wsi_loss["loss"].ndim == 0
    assert set(wsi_loss) == {"loss", "consistency_loss", "distribution_loss"}
    generated = wsi.sample(condition, generator)
    assert generated.consistency.shape == (3, 4)
    assert generated.distribution.shape == (3, 3)
    assert bool((generated.distribution >= 0).all())
    torch.testing.assert_close(generated.distribution.sum(dim=1), torch.ones(3))

    patches = torch.randn(3, 5, generator=generator)
    classes = torch.tensor([0, 1, 2])
    patch_loss = patch.training_loss(patches, consistency, classes, generator)
    assert patch_loss["loss"].ndim == 0
    assert patch.sample(consistency, classes, generator).shape == (3, 5)


class _RecordingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wsi_dim = 3
        self.num_classes = 2
        self.inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, noisy_z, noisy_p, condition, timestep):
        self.inputs.append((noisy_z.detach().clone(), noisy_p.detach().clone()))
        return torch.zeros_like(noisy_z), torch.zeros_like(noisy_p)


def test_disabled_wsi_branch_is_identically_zero_in_training_and_sampling():
    denoiser = _RecordingDenoiser()
    diffusion = models.WSILevelDiffusion(
        denoiser=denoiser,
        consistency_schedule=models.DiffusionSchedule(2, name="linear"),
        consistency_weight=0.0,
        distribution_weight=1.0,
        use_consistency=False,
        use_distribution=True,
    )
    condition = torch.randn(2, 3)
    diffusion.training_loss(
        consistency=torch.randn(2, 3),
        distribution=torch.tensor([[0.7, 0.3], [0.4, 0.6]]),
        condition=condition,
        generator=torch.Generator().manual_seed(2),
    )
    training_input = denoiser.inputs[-1][0]
    diffusion.sample(condition, torch.Generator().manual_seed(3))
    sampling_inputs = [pair[0] for pair in denoiser.inputs[1:]]

    assert torch.count_nonzero(training_input) == 0
    assert sampling_inputs
    assert all(torch.count_nonzero(value) == 0 for value in sampling_inputs)


def test_hierarchy_rejects_mismatched_ablation_wiring():
    wsi, patch = _tiny_hierarchy(timesteps=1)
    patch.conditioner.use_distribution = False
    with pytest.raises(ValueError, match="distribution ablation must match"):
        models.HierarchicalWSIGenerator(wsi, patch, patches_per_slide=5)


class _Dims:
    def __init__(self, wsi_dim: int, num_classes: int) -> None:
        self.wsi_dim = wsi_dim
        self.num_classes = num_classes


class _FixedWSI(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.denoiser = _Dims(wsi_dim=3, num_classes=3)
        self.use_consistency = True
        self.use_distribution = True

    def sample(self, condition, generator=None):
        batch = condition.shape[0]
        return models.WSILevelOutput(
            consistency=torch.ones(batch, 3, device=condition.device),
            distribution=torch.tensor(
                [[0.5, 0.3, 0.2]],
                device=condition.device,
                dtype=condition.dtype,
            ).expand(batch, -1),
            distribution_latent=torch.zeros(
                batch,
                3,
                device=condition.device,
                dtype=condition.dtype,
            ),
        )


class _Condition:
    def __init__(self) -> None:
        self.wsi_dim = 3
        self.num_classes = 3
        self.use_consistency = True
        self.use_distribution = True


class _LabelEchoPatch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conditioner = _Condition()

    def sample(self, consistency, tissue_class, generator=None):
        return torch.stack(
            [tissue_class.float(), consistency[:, 0], consistency[:, 1], consistency[:, 2]],
            dim=1,
        )


def test_hierarchical_generation_realizes_the_exact_tissue_quota():
    generator = models.HierarchicalWSIGenerator(
        wsi_diffusion=_FixedWSI(),
        patch_diffusion=_LabelEchoPatch(),
        patches_per_slide=7,
    )
    result = generator.generate_one(torch.randn(2, 3))

    assert result.patch_features.shape == (7, 4)
    assert result.distribution.tolist() == pytest.approx([0.5, 0.3, 0.2])
    assert torch.bincount(result.tissue_labels, minlength=3).tolist() == [4, 2, 1]
    torch.testing.assert_close(
        result.patch_features[:, 0],
        result.tissue_labels.float(),
    )


def test_cmil_is_permutation_invariant_and_cox_matches_closed_form():
    model = models.HierarchicalCMILSurvival(
        patch_dim=3,
        cmil_hidden_dim=5,
        cmil_output_dim=4,
        cmil_dropout=0.0,
        patient_hidden_dims=(3,),
        dropout=0.0,
    ).eval()
    slide_a = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    slide_b = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    patient = data.PatientBags("P1", [slide_a, slide_b], [], 1.0, 1)
    permuted = data.PatientBags(
        "P1",
        [slide_b.flip(0), slide_a.flip(0)],
        [],
        1.0,
        1,
    )
    risk, representation = model.forward_with_representations([patient, permuted])
    assert risk.shape == (2,)
    assert representation.shape == (2, 4)
    torch.testing.assert_close(risk[0], risk[1])
    torch.testing.assert_close(representation[0], representation[1])

    log_two = math.log(2.0)
    cox_risk = torch.tensor([log_two, 0.0], requires_grad=True)
    loss = models.negative_cox_partial_log_likelihood(
        cox_risk,
        time=torch.tensor([1.0, 2.0]),
        event=torch.tensor([1, 1]),
        ties="breslow",
    )
    assert float(loss) == pytest.approx(0.5 * math.log(1.5))
    loss.backward()
    assert cox_risk.grad is not None
    assert bool(torch.isfinite(cox_risk.grad).all())


def test_cox_rejects_a_fold_without_observed_events():
    with pytest.raises(ValueError, match="without an observed event"):
        models.negative_cox_partial_log_likelihood(
            torch.tensor([0.2, -0.1]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 0]),
        )
