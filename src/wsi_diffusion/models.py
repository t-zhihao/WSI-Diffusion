"""Official WSI-Diffusion model, generation, and survival components.

This module intentionally keeps the mathematical core in one place: the two
conditional DDPM levels, hierarchical feature generation, C-MIL survival model,
and Cox objective.  The compact layout makes the GitHub release easy to audit
without hiding the implementation behind framework-specific abstractions.

Notation follows the paper:

* ``z`` is the WSI-level consistency representation;
* ``p`` is the six-class tissue distribution;
* ``x`` is a patch representation;
* the denoisers predict epsilon at every diffusion step; and
* a larger survival risk denotes an earlier event.

The implementation uses APIs available in PyTorch 1.9.1 and later.  In
particular, stable sorting is implemented in Python where needed and the cosine
schedule avoids overlapping-memory in-place operations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.nn import functional as F

from wsi_diffusion.data import expand_class_counts, largest_remainder_counts

if TYPE_CHECKING:
    from wsi_diffusion.config import Config
    from wsi_diffusion.data import PatientBags


# ---------------------------------------------------------------------------
# Diffusion schedules and reverse process
# ---------------------------------------------------------------------------


def make_beta_schedule(
    name: str,
    timesteps: int,
    beta_start: float = 1.0e-4,
    beta_end: float = 2.0e-2,
) -> torch.Tensor:
    """Construct a validated DDPM beta schedule.

    Coefficients are calculated in float64 and returned as float32 buffers.  The
    cosine formula is the improved-DDPM schedule with ``s=0.008``.  Its
    normalization is out-of-place because a view such as ``alpha_bar[0]`` may
    alias the left-hand side on modern PyTorch versions.
    """

    timesteps = int(timesteps)
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    if not 0.0 < float(beta_start) < 1.0:
        raise ValueError("beta_start must lie in (0, 1)")
    if not 0.0 < float(beta_end) < 1.0:
        raise ValueError("beta_end must lie in (0, 1)")
    if float(beta_start) > float(beta_end):
        raise ValueError("beta_start cannot exceed beta_end")

    normalized_name = str(name).lower()
    if normalized_name == "linear":
        betas = torch.linspace(
            float(beta_start), float(beta_end), timesteps, dtype=torch.float64
        )
    elif normalized_name == "cosine":
        offset = 0.008
        points = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        angles = ((points / timesteps + offset) / (1.0 + offset)) * math.pi / 2.0
        alpha_bar_unnormalized = torch.cos(angles).pow(2)
        # clone() makes the non-aliasing intent explicit on PyTorch 1.9--2.x.
        normalizer = alpha_bar_unnormalized[0].clone()
        alpha_bar = alpha_bar_unnormalized / normalizer
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(min=1.0e-8, max=0.999)
    else:
        raise ValueError("schedule name must be 'linear' or 'cosine'")

    if not bool(torch.isfinite(betas).all()):
        raise ValueError("beta schedule contains a non-finite coefficient")
    if bool(((betas <= 0) | (betas >= 1)).any()):
        raise ValueError("every beta must lie in (0, 1)")
    return betas.float()


class DiffusionSchedule(nn.Module):
    """Precomputed forward and posterior DDPM coefficients.

    All quantities are buffers, so moving a model between CPU and CUDA also
    moves its schedule.  The posterior at step zero is represented exactly:
    its variance is zero and its mean coefficients are ``(1, 0)``.
    """

    def __init__(
        self,
        timesteps: int,
        name: str = "cosine",
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        betas = make_beta_schedule(name, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat(
            [torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0
        )
        denominator = (1.0 - alpha_bars).clamp_min(torch.finfo(alpha_bars.dtype).eps)

        posterior_variance_raw = betas * (1.0 - previous_alpha_bars) / denominator
        posterior_variance = torch.cat(
            [torch.zeros_like(posterior_variance_raw[:1]), posterior_variance_raw[1:]],
            dim=0,
        )
        posterior_mean_coef1_raw = (
            betas * previous_alpha_bars.sqrt() / denominator
        )
        posterior_mean_coef2_raw = (
            (1.0 - previous_alpha_bars) * alphas.sqrt() / denominator
        )
        posterior_mean_coef1 = torch.cat(
            [torch.ones_like(posterior_mean_coef1_raw[:1]), posterior_mean_coef1_raw[1:]],
            dim=0,
        )
        posterior_mean_coef2 = torch.cat(
            [torch.zeros_like(posterior_mean_coef2_raw[:1]), posterior_mean_coef2_raw[1:]],
            dim=0,
        )

        self.timesteps = int(timesteps)
        self.name = str(name).lower()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("previous_alpha_bars", previous_alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer("sqrt_recip_alpha_bars", (1.0 / alpha_bars).sqrt())
        self.register_buffer(
            "sqrt_recipm1_alpha_bars", (1.0 / alpha_bars - 1.0).sqrt()
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @staticmethod
    def extract(
        values: torch.Tensor,
        timestep: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gather a scalar coefficient per batch row and broadcast to target."""

        if timestep.ndim != 1:
            raise ValueError("timestep must be a one-dimensional batch tensor")
        if timestep.shape[0] != target.shape[0]:
            raise ValueError("timestep and target batch sizes differ")
        selected = values.gather(0, timestep.long())
        shape = (timestep.shape[0],) + (1,) * (target.ndim - 1)
        return selected.reshape(shape).to(device=target.device, dtype=target.dtype)

    def q_sample(
        self,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``q(x_t | x_0)`` in one operation."""

        if clean.shape != noise.shape:
            raise ValueError("clean and noise tensors must have identical shapes")
        return (
            self.extract(self.sqrt_alpha_bars, timestep, clean) * clean
            + self.extract(self.sqrt_one_minus_alpha_bars, timestep, clean) * noise
        )

    def predict_x0(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the epsilon-parameterized estimate of ``x_0``."""

        if noisy.shape != predicted_noise.shape:
            raise ValueError("noisy and predicted_noise tensors must have identical shapes")
        return (
            self.extract(self.sqrt_recip_alpha_bars, timestep, noisy) * noisy
            - self.extract(self.sqrt_recipm1_alpha_bars, timestep, noisy)
            * predicted_noise
        )

    def reverse_mean(
        self,
        noisy: torch.Tensor,
        predicted_x0: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact posterior mean ``q(x_{t-1}|x_t,x_0)``."""

        mean = (
            self.extract(self.posterior_mean_coef1, timestep, noisy) * predicted_x0
            + self.extract(self.posterior_mean_coef2, timestep, noisy) * noisy
        )
        at_zero = (timestep == 0).reshape(
            (timestep.shape[0],) + (1,) * (noisy.ndim - 1)
        )
        # This branch guarantees bitwise x_0 at the final step even if a backend
        # evaluates the coefficient expression with reduced precision.
        return torch.where(at_zero, predicted_x0, mean)

    def reverse_std(
        self,
        timestep: torch.Tensor,
        target: torch.Tensor,
        variance_type: str,
    ) -> torch.Tensor:
        if variance_type == "paper_beta":
            variance = self.betas
        elif variance_type == "posterior":
            variance = self.posterior_variance
        else:
            raise ValueError("variance_type must be 'paper_beta' or 'posterior'")
        return self.extract(variance, timestep, target).clamp_min(0).sqrt()


def random_normal_like(
    reference: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """``randn_like`` equivalent with generator support on PyTorch 1.9."""

    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def ddpm_step(
    noisy: torch.Tensor,
    predicted_noise: torch.Tensor,
    timestep: torch.Tensor,
    schedule: DiffusionSchedule,
    variance_type: str = "paper_beta",
    clip_x0: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one reverse DDPM step and return ``(x_{t-1}, predicted_x0)``.

    At ``t=0`` the returned sample is exactly ``predicted_x0`` and no random
    noise contributes.  Mixed-timestep batches are handled row by row.
    """

    predicted_x0 = schedule.predict_x0(noisy, timestep, predicted_noise)
    if clip_x0 is not None:
        if float(clip_x0) <= 0:
            raise ValueError("clip_x0 must be positive or None")
        predicted_x0 = predicted_x0.clamp(-float(clip_x0), float(clip_x0))
    mean = schedule.reverse_mean(noisy, predicted_x0, timestep)
    at_zero = (timestep == 0).reshape(
        (timestep.shape[0],) + (1,) * (noisy.ndim - 1)
    )
    if bool((timestep != 0).any()):
        noise = random_normal_like(noisy, generator)
        stochastic = mean + schedule.reverse_std(
            timestep, noisy, variance_type
        ) * noise
        sample = torch.where(at_zero, predicted_x0, stochastic)
    else:
        sample = predicted_x0
    return sample, predicted_x0


# ---------------------------------------------------------------------------
# Shared denoiser blocks
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding for integer diffusion steps."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if int(dimension) < 4:
            raise ValueError("time embedding dimension must be at least four")
        self.dimension = int(dimension)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(
                half,
                device=timestep.device,
                dtype=torch.float32,
            )
            * -scale
        )
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualFiLMBlock(nn.Module):
    """Layer-normalized residual MLP modulated by time and conditions."""

    def __init__(self, hidden_dim: int, context_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim < 1 or context_dim < 1:
            raise ValueError("hidden_dim and context_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.norm = nn.LayerNorm(hidden_dim)
        self.context_to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, hidden_dim * 2),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.context_to_scale_shift(context).chunk(2, dim=-1)
        normalized = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + self.net(normalized)


def initialize_denoiser(module: nn.Module) -> None:
    """Apply the shared Xavier initialization to all linear layers."""

    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


# ---------------------------------------------------------------------------
# WSI-level joint diffusion (paper equations 8--9)
# ---------------------------------------------------------------------------


@dataclass
class WSILevelOutput:
    """One batch of generated WSI-level constraints."""

    consistency: torch.Tensor
    distribution: torch.Tensor
    distribution_latent: torch.Tensor


def project_to_probability_simplex(values: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of direct distribution features onto the simplex.

    The WSI-level tissue target is diffused directly in probability space.  A
    reverse trajectory is unconstrained until its last step, after which this
    exact simplex projection produces non-negative values that sum to one.  It
    deliberately is not a softmax/logit transform: training and sampling use
    the same direct parameterization.
    """

    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("values must have a non-empty class dimension")
    original_shape = values.shape
    flattened = values.reshape(-1, values.shape[-1])
    sorted_values, _ = torch.sort(flattened, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=-1) - 1.0
    ranks = torch.arange(
        1,
        flattened.shape[-1] + 1,
        device=flattened.device,
        dtype=flattened.dtype,
    ).unsqueeze(0)
    admissible = sorted_values - cumulative / ranks > 0
    rho = admissible.long().sum(dim=-1).clamp_min(1) - 1
    theta = cumulative.gather(1, rho.unsqueeze(1)) / (rho + 1).to(
        flattened.dtype
    ).unsqueeze(1)
    projected = (flattened - theta).clamp_min(0)
    # Normalize once to absorb the final floating-point rounding error.
    projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(projected.dtype).eps
    )
    return projected.reshape(original_shape)


class JointWSIDenoiser(nn.Module):
    """Joint epsilon predictor for WSI consistency and tissue distribution."""

    def __init__(
        self,
        wsi_dim: int,
        num_classes: int,
        hidden_dim: int,
        condition_dim: int,
        time_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = (wsi_dim, num_classes, hidden_dim, condition_dim, time_dim, num_blocks)
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("all WSI denoiser dimensions must be positive")
        self.wsi_dim = int(wsi_dim)
        self.num_classes = int(num_classes)
        input_dim = self.wsi_dim + self.num_classes
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(self.wsi_dim),
            nn.Linear(self.wsi_dim, condition_dim),
            nn.SiLU(),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualFiLMBlock(hidden_dim, condition_dim, dropout)
                for _ in range(int(num_blocks))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.consistency_head = nn.Linear(hidden_dim, self.wsi_dim)
        self.distribution_head = nn.Linear(hidden_dim, self.num_classes)
        initialize_denoiser(self)
        # Near-zero heads stabilize epsilon regression at initialization.
        nn.init.normal_(self.consistency_head.weight, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.distribution_head.weight, mean=0.0, std=1.0e-3)

    def forward(
        self,
        noisy_consistency: torch.Tensor,
        noisy_distribution: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_consistency.shape[-1] != self.wsi_dim:
            raise ValueError("noisy_consistency has the wrong feature width")
        if noisy_distribution.shape[-1] != self.num_classes:
            raise ValueError("noisy_distribution has the wrong class width")
        if condition.shape[-1] != self.wsi_dim:
            raise ValueError("condition has the wrong WSI feature width")
        batch = noisy_consistency.shape[0]
        if (
            noisy_distribution.shape[0] != batch
            or condition.shape[0] != batch
            or timestep.shape[0] != batch
        ):
            raise ValueError("all WSI denoiser inputs must share a batch size")
        hidden = self.input_projection(
            torch.cat([noisy_consistency, noisy_distribution], dim=-1)
        )
        context = self.condition_projection(condition) + self.time_embedding(timestep)
        for block in self.blocks:
            hidden = block(hidden, context)
        hidden = F.silu(self.output_norm(hidden))
        return self.consistency_head(hidden), self.distribution_head(hidden)


class WSILevelDiffusion(nn.Module):
    """Conditional joint DDPM over WSI-level ``(z, p)`` constraints.

    ``use_consistency`` and ``use_distribution`` define architectural
    ablations.  A disabled input is identically zero in both training and every
    reverse-sampling step, preventing train/sample leakage through an ablated
    branch.
    """

    def __init__(
        self,
        denoiser: JointWSIDenoiser,
        consistency_schedule: DiffusionSchedule,
        distribution_schedule: Optional[DiffusionSchedule] = None,
        consistency_weight: float = 1.0,
        distribution_weight: float = 1.0,
        variance_type: str = "paper_beta",
        clip_x0: Optional[float] = None,
        distribution_parameterization: str = "direct",
        use_consistency: bool = True,
        use_distribution: bool = True,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.consistency_schedule = consistency_schedule
        self.distribution_schedule = distribution_schedule or consistency_schedule
        if self.distribution_schedule.timesteps != self.consistency_schedule.timesteps:
            raise ValueError("both WSI schedules must have the same number of steps")
        if float(consistency_weight) < 0 or float(distribution_weight) < 0:
            raise ValueError("diffusion loss weights cannot be negative")
        if variance_type not in {"paper_beta", "posterior"}:
            raise ValueError("variance_type must be 'paper_beta' or 'posterior'")
        if distribution_parameterization != "direct":
            raise ValueError(
                "WSI-Diffusion uses direct tissue-distribution diffusion; "
                "logit-space sampling requires a different training transform"
            )
        if not use_consistency and float(consistency_weight) != 0:
            raise ValueError("a disabled consistency branch must have zero loss weight")
        if not use_distribution and float(distribution_weight) != 0:
            raise ValueError("a disabled distribution branch must have zero loss weight")
        if not use_consistency and not use_distribution:
            # A fully disabled WSI model is represented by skip_wsi_level in the
            # hierarchy, not by training an objective that is identically zero.
            if float(consistency_weight) != 0 or float(distribution_weight) != 0:
                raise ValueError("disabled WSI branches must have zero loss weights")
        self.consistency_weight = float(consistency_weight)
        self.distribution_weight = float(distribution_weight)
        self.variance_type = variance_type
        self.clip_x0 = None if clip_x0 is None else float(clip_x0)
        self.distribution_parameterization = distribution_parameterization
        self.use_consistency = bool(use_consistency)
        self.use_distribution = bool(use_distribution)

    @property
    def timesteps(self) -> int:
        return self.consistency_schedule.timesteps

    def _validate_training_batch(
        self,
        consistency: torch.Tensor,
        distribution: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        if consistency.ndim != 2 or condition.ndim != 2 or distribution.ndim != 2:
            raise ValueError("WSI diffusion inputs must be two-dimensional batches")
        batch = consistency.shape[0]
        if distribution.shape[0] != batch or condition.shape[0] != batch:
            raise ValueError("WSI diffusion inputs must share a batch size")
        if consistency.shape[1] != self.denoiser.wsi_dim:
            raise ValueError("consistency feature width does not match the denoiser")
        if condition.shape[1] != self.denoiser.wsi_dim:
            raise ValueError("condition feature width does not match the denoiser")
        if distribution.shape[1] != self.denoiser.num_classes:
            raise ValueError("distribution class width does not match the denoiser")
        if not bool(torch.isfinite(distribution).all()):
            raise ValueError("tissue distributions must be finite")
        if bool((distribution < -1.0e-6).any()):
            raise ValueError("direct tissue distributions cannot be negative")
        row_sums = distribution.sum(dim=-1)
        if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-4)):
            raise ValueError("each direct tissue distribution must sum to one")

    def training_loss(
        self,
        consistency: torch.Tensor,
        distribution: torch.Tensor,
        condition: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_training_batch(consistency, distribution, condition)
        batch = consistency.shape[0]
        timestep = torch.randint(
            self.timesteps,
            (batch,),
            device=consistency.device,
            generator=generator,
        )
        noise_z = (
            random_normal_like(consistency, generator)
            if self.use_consistency
            else torch.zeros_like(consistency)
        )
        noise_p = (
            random_normal_like(distribution, generator)
            if self.use_distribution
            else torch.zeros_like(distribution)
        )
        noisy_z = (
            self.consistency_schedule.q_sample(consistency, timestep, noise_z)
            if self.use_consistency
            else torch.zeros_like(consistency)
        )
        noisy_p = (
            self.distribution_schedule.q_sample(distribution, timestep, noise_p)
            if self.use_distribution
            else torch.zeros_like(distribution)
        )
        predicted_z, predicted_p = self.denoiser(
            noisy_z, noisy_p, condition, timestep
        )
        consistency_loss = (
            F.mse_loss(predicted_z, noise_z)
            if self.use_consistency
            else predicted_z.sum() * 0.0
        )
        distribution_loss = (
            F.mse_loss(predicted_p, noise_p)
            if self.use_distribution
            else predicted_p.sum() * 0.0
        )
        total = (
            self.consistency_weight * consistency_loss
            + self.distribution_weight * distribution_loss
        )
        return {
            "loss": total,
            "consistency_loss": consistency_loss,
            "distribution_loss": distribution_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> WSILevelOutput:
        if condition.ndim != 2 or condition.shape[1] != self.denoiser.wsi_dim:
            raise ValueError("condition must have shape [batch, wsi_dim]")
        batch = condition.shape[0]
        shape_z = (batch, self.denoiser.wsi_dim)
        shape_p = (batch, self.denoiser.num_classes)
        noisy_z = (
            torch.randn(
                shape_z,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
            if self.use_consistency
            else condition.new_zeros(shape_z)
        )
        noisy_p = (
            torch.randn(
                shape_p,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
            if self.use_distribution
            else condition.new_zeros(shape_p)
        )
        for step in reversed(range(self.timesteps)):
            timestep = torch.full(
                (batch,),
                step,
                device=condition.device,
                dtype=torch.long,
            )
            predicted_z, predicted_p = self.denoiser(
                noisy_z, noisy_p, condition, timestep
            )
            if self.use_consistency:
                noisy_z, _ = ddpm_step(
                    noisy_z,
                    predicted_z,
                    timestep,
                    self.consistency_schedule,
                    self.variance_type,
                    self.clip_x0,
                    generator,
                )
            else:
                # Keep the ablated input exactly identical to its training value.
                noisy_z.zero_()
            if self.use_distribution:
                noisy_p, _ = ddpm_step(
                    noisy_p,
                    predicted_p,
                    timestep,
                    self.distribution_schedule,
                    self.variance_type,
                    self.clip_x0,
                    generator,
                )
            else:
                noisy_p.zero_()
        distribution = (
            project_to_probability_simplex(noisy_p)
            if self.use_distribution
            else torch.full_like(noisy_p, 1.0 / noisy_p.shape[-1])
        )
        return WSILevelOutput(
            consistency=noisy_z,
            distribution=distribution,
            distribution_latent=noisy_p,
        )


# ---------------------------------------------------------------------------
# Patch-level conditional diffusion (paper equation 10)
# ---------------------------------------------------------------------------


class WSIClassConditioner(nn.Module):
    """Fuse WSI consistency and tissue class with a Hadamard product.

    The two boolean switches are applied inside this single shared function, so
    patch training and reverse sampling cannot accidentally encode different
    ablation inputs.
    """

    def __init__(
        self,
        wsi_dim: int,
        num_classes: int,
        condition_dim: int,
        use_consistency: bool = True,
        use_distribution: bool = True,
    ) -> None:
        super().__init__()
        if min(int(wsi_dim), int(num_classes), int(condition_dim)) < 1:
            raise ValueError("conditioner dimensions must be positive")
        self.wsi_dim = int(wsi_dim)
        self.num_classes = int(num_classes)
        self.condition_dim = int(condition_dim)
        self.use_consistency = bool(use_consistency)
        # This switch controls the tissue-class signal derived from p.
        self.use_distribution = bool(use_distribution)
        self.wsi_projection = nn.Sequential(
            nn.LayerNorm(self.wsi_dim),
            nn.Linear(self.wsi_dim, self.condition_dim),
            nn.SiLU(),
        )
        self.class_embedding = nn.Embedding(self.num_classes, self.condition_dim)
        self.null_condition = nn.Parameter(torch.zeros(self.condition_dim))
        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        consistency: torch.Tensor,
        tissue_class: torch.Tensor,
    ) -> torch.Tensor:
        if consistency.ndim != 2 or consistency.shape[1] != self.wsi_dim:
            raise ValueError("consistency must have shape [batch, wsi_dim]")
        tissue_class = tissue_class.reshape(-1).long().to(consistency.device)
        if tissue_class.shape[0] != consistency.shape[0]:
            raise ValueError("consistency and tissue_class batch sizes differ")
        if bool(((tissue_class < 0) | (tissue_class >= self.num_classes)).any()):
            raise ValueError("tissue_class contains an out-of-range class id")
        if not self.use_consistency and not self.use_distribution:
            return self.null_condition.to(consistency).unsqueeze(0).expand(
                consistency.shape[0], -1
            )
        consistency_condition = (
            self.wsi_projection(consistency)
            if self.use_consistency
            else consistency.new_ones(
                (consistency.shape[0], self.condition_dim)
            )
        )
        class_condition = (
            self.class_embedding(tissue_class).to(consistency_condition.dtype)
            if self.use_distribution
            else torch.ones_like(consistency_condition)
        )
        return consistency_condition * class_condition


class PatchDenoiser(nn.Module):
    """FiLM residual epsilon predictor for patch representations."""

    def __init__(
        self,
        patch_dim: int,
        condition_dim: int,
        hidden_dim: int,
        time_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = (patch_dim, condition_dim, hidden_dim, time_dim, num_blocks)
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("all patch denoiser dimensions must be positive")
        self.patch_dim = int(patch_dim)
        self.condition_dim = int(condition_dim)
        self.input_projection = nn.Linear(self.patch_dim, hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_dim),
            nn.SiLU(),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualFiLMBlock(hidden_dim, self.condition_dim, dropout)
                for _ in range(int(num_blocks))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.patch_dim),
        )
        initialize_denoiser(self)
        nn.init.normal_(self.output[-1].weight, mean=0.0, std=1.0e-3)

    def forward(
        self,
        noisy_patch: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_patch.ndim != 2 or noisy_patch.shape[1] != self.patch_dim:
            raise ValueError("noisy_patch must have shape [batch, patch_dim]")
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError("condition must have shape [batch, condition_dim]")
        if condition.shape[0] != noisy_patch.shape[0]:
            raise ValueError("patch and condition batch sizes differ")
        hidden = self.input_projection(noisy_patch)
        context = self.condition_projection(condition) + self.time_embedding(timestep)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(hidden)


class PatchLevelDiffusion(nn.Module):
    """Tissue- and WSI-conditioned patch representation DDPM."""

    def __init__(
        self,
        conditioner: WSIClassConditioner,
        denoiser: PatchDenoiser,
        schedule: DiffusionSchedule,
        variance_type: str = "paper_beta",
        clip_x0: Optional[float] = None,
    ) -> None:
        super().__init__()
        if conditioner.condition_dim != denoiser.condition_dim:
            raise ValueError("conditioner and patch denoiser widths differ")
        if variance_type not in {"paper_beta", "posterior"}:
            raise ValueError("variance_type must be 'paper_beta' or 'posterior'")
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.schedule = schedule
        self.variance_type = variance_type
        self.clip_x0 = None if clip_x0 is None else float(clip_x0)

    def training_loss(
        self,
        patch: torch.Tensor,
        consistency: torch.Tensor,
        tissue_class: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, torch.Tensor]:
        if patch.ndim != 2 or patch.shape[1] != self.denoiser.patch_dim:
            raise ValueError("patch must have shape [batch, patch_dim]")
        if patch.shape[0] != consistency.shape[0]:
            raise ValueError("patch and consistency batch sizes differ")
        batch = patch.shape[0]
        timestep = torch.randint(
            self.schedule.timesteps,
            (batch,),
            device=patch.device,
            generator=generator,
        )
        noise = random_normal_like(patch, generator)
        noisy = self.schedule.q_sample(patch, timestep, noise)
        # The same conditioner call is used verbatim by sample().
        condition = self.conditioner(consistency, tissue_class)
        predicted_noise = self.denoiser(noisy, condition, timestep)
        loss = F.mse_loss(predicted_noise, noise)
        return {"loss": loss, "patch_loss": loss}

    @torch.no_grad()
    def sample(
        self,
        consistency: torch.Tensor,
        tissue_class: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if consistency.ndim != 2 or consistency.shape[1] != self.conditioner.wsi_dim:
            raise ValueError("consistency must have shape [batch, wsi_dim]")
        noisy = torch.randn(
            (consistency.shape[0], self.denoiser.patch_dim),
            device=consistency.device,
            dtype=consistency.dtype,
            generator=generator,
        )
        condition = self.conditioner(consistency, tissue_class)
        for step in reversed(range(self.schedule.timesteps)):
            timestep = torch.full(
                (consistency.shape[0],),
                step,
                device=consistency.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(noisy, condition, timestep)
            noisy, _ = ddpm_step(
                noisy,
                predicted_noise,
                timestep,
                self.schedule,
                self.variance_type,
                self.clip_x0,
                generator,
            )
        return noisy


# ---------------------------------------------------------------------------
# Hierarchical WSI generation and augmentation strategies
# ---------------------------------------------------------------------------


@dataclass
class GeneratedSlide:
    """Generated patch bag and its WSI-level constraints."""

    consistency: torch.Tensor
    distribution: torch.Tensor
    patch_features: torch.Tensor
    tissue_labels: torch.Tensor


def number_to_generate(
    original_count: int,
    strategy: str,
    fixed_count: int,
    partial_threshold: int,
    target_total: int,
) -> int:
    """Resolve Wholly, Partially, or Unequally augmentation for one patient."""

    original_count = int(original_count)
    fixed_count = int(fixed_count)
    partial_threshold = int(partial_threshold)
    target_total = int(target_total)
    if min(original_count, fixed_count, partial_threshold, target_total) < 0:
        raise ValueError("generation counts and thresholds cannot be negative")
    if strategy == "wholly":
        return fixed_count
    if strategy == "partially":
        return fixed_count if original_count < partial_threshold else 0
    if strategy == "unequally":
        return max(0, target_total - original_count)
    raise ValueError("strategy must be wholly, partially, or unequally")


class HierarchicalWSIGenerator(nn.Module):
    """Run WSI-level diffusion and then generate a tissue-stratified patch bag."""

    def __init__(
        self,
        wsi_diffusion: WSILevelDiffusion,
        patch_diffusion: PatchLevelDiffusion,
        patches_per_slide: int,
        minimum_per_nonzero_class: int = 0,
        skip_wsi_level: bool = False,
        class_sampling: str = "generated",
        distribution_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if int(patches_per_slide) < 1:
            raise ValueError("patches_per_slide must be positive")
        if int(minimum_per_nonzero_class) < 0:
            raise ValueError("minimum_per_nonzero_class cannot be negative")
        if class_sampling not in {"generated", "uniform", "empirical_global"}:
            raise ValueError(
                "class_sampling must be generated, uniform, or empirical_global"
            )
        if float(distribution_temperature) <= 0:
            raise ValueError("distribution_temperature must be positive")
        if wsi_diffusion.denoiser.wsi_dim != patch_diffusion.conditioner.wsi_dim:
            raise ValueError("WSI and patch levels use different consistency widths")
        if wsi_diffusion.denoiser.num_classes != patch_diffusion.conditioner.num_classes:
            raise ValueError("WSI and patch levels use different tissue class counts")

        # These checks turn ablation consistency into a construction invariant.
        if skip_wsi_level and (
            patch_diffusion.conditioner.use_consistency
            or patch_diffusion.conditioner.use_distribution
        ):
            raise ValueError(
                "skip_wsi_level requires both patch conditioner inputs to be disabled "
                "during training as well as sampling"
            )
        if not skip_wsi_level:
            if (
                patch_diffusion.conditioner.use_consistency
                != wsi_diffusion.use_consistency
            ):
                raise ValueError(
                    "consistency ablation must match at the WSI and patch levels"
                )
            if (
                patch_diffusion.conditioner.use_distribution
                != wsi_diffusion.use_distribution
            ):
                raise ValueError(
                    "distribution ablation must match at the WSI and patch levels"
                )
        if class_sampling == "generated" and (
            skip_wsi_level or not wsi_diffusion.use_distribution
        ):
            raise ValueError(
                "generated class sampling requires the WSI distribution branch"
            )

        self.wsi_diffusion = wsi_diffusion
        self.patch_diffusion = patch_diffusion
        self.patches_per_slide = int(patches_per_slide)
        self.minimum_per_nonzero_class = int(minimum_per_nonzero_class)
        self.skip_wsi_level = bool(skip_wsi_level)
        self.class_sampling = class_sampling
        self.distribution_temperature = float(distribution_temperature)
        # persistent=False is supported by the paper environment (PyTorch 1.9.1)
        # and keeps a fold-specific empirical statistic out of checkpoints.
        self.register_buffer(
            "fallback_distribution",
            torch.empty(0),
            persistent=False,
        )

    def set_fallback_distribution(self, distribution: torch.Tensor) -> None:
        """Install the train-fold empirical distribution used by one ablation."""

        probabilities = distribution.detach().float().reshape(-1)
        expected_classes = self.wsi_diffusion.denoiser.num_classes
        if probabilities.numel() != expected_classes:
            raise ValueError(
                f"expected {expected_classes} tissue probabilities, "
                f"received {probabilities.numel()}"
            )
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("fallback distribution must be finite")
        probabilities = probabilities.clamp_min(0)
        if float(probabilities.sum()) <= 0:
            raise ValueError("fallback distribution must have positive mass")
        probabilities = probabilities / probabilities.sum()
        self.fallback_distribution = probabilities.to(
            device=self.fallback_distribution.device
        )

    def _allocation_distribution(
        self,
        generated_distribution: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self.class_sampling == "generated":
            probabilities = generated_distribution.reshape(-1)
        elif self.class_sampling == "uniform":
            probabilities = torch.ones_like(generated_distribution.reshape(-1))
        else:
            if self.fallback_distribution.numel() == 0:
                raise RuntimeError(
                    "empirical_global sampling requires set_fallback_distribution()"
                )
            probabilities = self.fallback_distribution.to(
                device=context.device,
                dtype=context.dtype,
            )
        probabilities = probabilities.clamp_min(0)
        if self.distribution_temperature != 1.0:
            probabilities = probabilities.clamp_min(1.0e-12).pow(
                1.0 / self.distribution_temperature
            )
        return probabilities / probabilities.sum().clamp_min(1.0e-8)

    @torch.no_grad()
    def generate_one(
        self,
        existing_wsi_features: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> GeneratedSlide:
        if (
            existing_wsi_features.ndim != 2
            or existing_wsi_features.shape[0] < 1
            or existing_wsi_features.shape[1]
            != self.wsi_diffusion.denoiser.wsi_dim
        ):
            raise ValueError("existing_wsi_features must have shape [N>=1, wsi_dim]")
        context = existing_wsi_features.mean(dim=0, keepdim=True)
        if self.skip_wsi_level:
            consistency = torch.zeros_like(context)
            generated_distribution = torch.full(
                (1, self.wsi_diffusion.denoiser.num_classes),
                1.0 / self.wsi_diffusion.denoiser.num_classes,
                device=context.device,
                dtype=context.dtype,
            )
        else:
            constraints = self.wsi_diffusion.sample(context, generator)
            consistency = constraints.consistency
            generated_distribution = constraints.distribution

        allocation_distribution = self._allocation_distribution(
            generated_distribution[0], context
        )
        counts = largest_remainder_counts(
            allocation_distribution.detach().cpu(),
            self.patches_per_slide,
            self.minimum_per_nonzero_class,
        )
        # Patch rows are conditionally independent; a deterministic quota order
        # avoids CPU/CUDA generator coupling without changing their distribution.
        labels = expand_class_counts(counts, shuffle=False).to(context.device)
        patch_consistency = consistency.expand(labels.shape[0], -1)
        patches = self.patch_diffusion.sample(
            patch_consistency,
            labels,
            generator,
        )
        return GeneratedSlide(
            consistency=consistency[0],
            distribution=allocation_distribution,
            patch_features=patches,
            tissue_labels=labels,
        )

    @torch.no_grad()
    def generate_patient(
        self,
        existing_wsi_features: torch.Tensor,
        count: int,
        generator: Optional[torch.Generator] = None,
    ) -> list[GeneratedSlide]:
        if int(count) < 0:
            raise ValueError("generated slide count cannot be negative")
        # Generated constraints do not recursively enter the patient context.
        return [
            self.generate_one(existing_wsi_features, generator)
            for _ in range(int(count))
        ]


# ---------------------------------------------------------------------------
# Hierarchical C-MIL survival model (paper equations 11--12)
# ---------------------------------------------------------------------------


class CMILPatchEncoder(nn.Module):
    """Encode a variable-size patch bag into one WSI representation.

    Kernel-size-one convolutions transform instances independently, so arbitrary
    feature-file order does not create a false spatial neighborhood.  Adaptive
    mean pooling then maps any positive patch count to the fixed output width.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if min(int(input_dim), int(hidden_dim), int(output_dim)) < 1:
            raise ValueError("C-MIL dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("C-MIL dropout must lie in [0, 1)")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, self.output_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def encode_instances(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 2:
            raise ValueError(
                f"expected a [patches, features] bag, received {tuple(patches.shape)}"
            )
        if patches.shape[0] == 0:
            raise ValueError("a WSI patch bag cannot be empty")
        if patches.shape[1] != self.input_dim:
            raise ValueError(
                f"expected patch width {self.input_dim}, received {patches.shape[1]}"
            )
        # [N,D] -> [1,D,N] -> [1,H,N] -> [N,H]
        encoded = self.network(patches.transpose(0, 1).unsqueeze(0))
        return encoded.squeeze(0).transpose(0, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_instances(patches)
        pooled = self.pool(encoded.transpose(0, 1).unsqueeze(0))
        return pooled.flatten()


class MaskedMeanPooling(nn.Module):
    """Mean-pool padded variable-size collections without padding leakage."""

    def forward(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if values.ndim < 2:
            raise ValueError("values must contain item and feature dimensions")
        if mask is None:
            return values.mean(dim=-2)
        if mask.shape != values.shape[:-1]:
            raise ValueError("mask must match every values dimension except features")
        weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=-2).clamp_min(1.0)
        return (values * weights).sum(dim=-2) / denominator


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build the patient-level risk head."""

    widths = [int(input_dim)] + [int(width) for width in hidden_dims]
    if any(width < 1 for width in widths) or int(output_dim) < 1:
        raise ValueError("MLP dimensions must be positive")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("MLP dropout must lie in [0, 1)")
    layers: list[nn.Module] = []
    for source, target in zip(widths[:-1], widths[1:]):
        layers.extend(
            [
                nn.Linear(source, target),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
    layers.append(nn.Linear(widths[-1], int(output_dim)))
    return nn.Sequential(*layers)


class HierarchicalCMILSurvival(nn.Module):
    """Patch-to-WSI-to-patient hierarchy followed by a scalar Cox risk head."""

    def __init__(
        self,
        patch_dim: int,
        cmil_hidden_dim: int = 128,
        cmil_output_dim: int = 64,
        cmil_dropout: float = 0.25,
        patient_hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.slide_encoder = CMILPatchEncoder(
            patch_dim,
            cmil_hidden_dim,
            cmil_output_dim,
            cmil_dropout,
        )
        self.risk_head = build_mlp(
            cmil_output_dim,
            patient_hidden_dims,
            1,
            dropout,
        )

    def encode_patient(
        self,
        real_slides: Sequence[torch.Tensor],
        generated_slides: Sequence[torch.Tensor] = (),
    ) -> torch.Tensor:
        all_slides = list(real_slides) + list(generated_slides)
        if not all_slides:
            raise ValueError("a patient must have at least one real or generated WSI")
        slide_features = torch.stack(
            [self.slide_encoder(slide) for slide in all_slides],
            dim=0,
        )
        return slide_features.mean(dim=0)

    def forward_one(
        self,
        patient: "PatientBags",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode_patient(
            patient.real_slides,
            patient.generated_slides,
        )
        return self.risk_head(representation).squeeze(-1), representation

    def forward(self, patients: Sequence["PatientBags"]) -> torch.Tensor:
        if not patients:
            raise ValueError("survival forward requires at least one patient")
        return torch.stack([self.forward_one(patient)[0] for patient in patients])

    def forward_with_representations(
        self,
        patients: Sequence["PatientBags"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not patients:
            raise ValueError("survival forward requires at least one patient")
        outputs = [self.forward_one(patient) for patient in patients]
        risks, representations = zip(*outputs)
        return torch.stack(list(risks)), torch.stack(list(representations))


def negative_cox_partial_log_likelihood(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    ties: str = "breslow",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the negative Cox partial log-likelihood.

    The risk-set convention is ``time >= event_time`` and larger model output
    means earlier event.  Survival times are explicitly transferred to the risk
    device as float64 and never pass through fp16.  Event indicators are also
    moved to the same device before validation, avoiding CPU/CUDA mask errors.
    Breslow and Efron tie handling are both supported.
    """

    risk = risk.reshape(-1)
    if not risk.is_floating_point():
        risk = risk.float()
    elif risk.dtype in {torch.float16, torch.bfloat16}:
        # Retains the computation graph while making log-sum-exp AMP safe.
        risk = risk.float()
    time = time.reshape(-1).to(device=risk.device, dtype=torch.float64)
    raw_event = event.reshape(-1).to(device=risk.device)
    if not (risk.numel() == time.numel() == raw_event.numel()):
        raise ValueError("risk, time, and event must have equal lengths")
    if risk.numel() == 0:
        raise ValueError("Cox loss requires at least one observation")
    if not bool(torch.isfinite(risk).all()) or not bool(torch.isfinite(time).all()):
        raise ValueError("risk and time must contain only finite values")
    if bool((time < 0).any()):
        raise ValueError("survival times cannot be negative")
    if not bool(((raw_event == 0) | (raw_event == 1)).all()):
        raise ValueError("event must contain binary 0/1 values")
    observed = raw_event.bool()
    if not bool(observed.any()):
        raise ValueError("Cox loss is undefined without an observed event")
    if ties not in {"breslow", "efron"}:
        raise ValueError("ties must be 'breslow' or 'efron'")
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    contributions: list[torch.Tensor] = []
    event_times = torch.unique(time[observed], sorted=True)
    for event_time in event_times:
        deaths = (time == event_time) & observed
        risk_set = time >= event_time
        death_risks = risk[deaths]
        number_deaths = int(deaths.long().sum().item())
        if ties == "breslow" or number_deaths == 1:
            log_denominator = torch.logsumexp(risk[risk_set], dim=0)
            contributions.append(
                -death_risks.sum() + number_deaths * log_denominator
            )
            continue

        # Efron's correction subtracts increasing fractions of the tied-death
        # exponential risk mass.  Centering by the maximum keeps it stable.
        shift = risk[risk_set].max()
        risk_mass = torch.exp(risk[risk_set] - shift).sum()
        death_mass = torch.exp(death_risks - shift).sum()
        tied_denominators: list[torch.Tensor] = []
        for rank in range(number_deaths):
            adjusted_mass = risk_mass - (float(rank) / number_deaths) * death_mass
            tied_denominators.append(
                torch.log(
                    adjusted_mass.clamp_min(torch.finfo(risk.dtype).tiny)
                )
                + shift
            )
        contributions.append(
            -death_risks.sum() + torch.stack(tied_denominators).sum()
        )

    loss = torch.stack(contributions).sum()
    if reduction == "mean":
        return loss / observed.sum().to(loss.dtype)
    return loss


class CoxPHLoss(nn.Module):
    """Module wrapper for :func:`negative_cox_partial_log_likelihood`."""

    def __init__(self, ties: str = "breslow", reduction: str = "mean") -> None:
        super().__init__()
        if ties not in {"breslow", "efron"}:
            raise ValueError("ties must be 'breslow' or 'efron'")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.ties = ties
        self.reduction = reduction

    def forward(
        self,
        risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        return negative_cox_partial_log_likelihood(
            risk,
            time,
            event,
            ties=self.ties,
            reduction=self.reduction,
        )


@dataclass
class HierarchicalDiffusionLoss:
    """Structured report of the three hierarchical epsilon objectives."""

    wsi_consistency: torch.Tensor
    wsi_distribution: torch.Tensor
    patch: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.wsi_consistency + self.wsi_distribution + self.patch

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "wsi_consistency": float(self.wsi_consistency.detach()),
            "wsi_distribution": float(self.wsi_distribution.detach()),
            "patch": float(self.patch.detach()),
        }


# ---------------------------------------------------------------------------
# Configuration-driven factories
# ---------------------------------------------------------------------------


def build_schedule(
    config: "Config",
    timesteps: int,
    schedule_override: Optional["Config"] = None,
) -> DiffusionSchedule:
    """Build a schedule from global settings or a level-specific override."""

    settings = (
        config.diffusion.schedule
        if schedule_override is None
        else schedule_override
    )
    return DiffusionSchedule(
        timesteps=int(timesteps),
        name=str(settings.name),
        beta_start=float(settings.beta_start),
        beta_end=float(settings.beta_end),
    )


def validate_ablation_wiring(config: "Config") -> None:
    """Reject configurations whose training and generation conditions differ."""

    skip_wsi = bool(config.generation.skip_wsi_level)
    wsi_consistency = bool(config.diffusion.wsi_level.use_consistency)
    wsi_distribution = bool(config.diffusion.wsi_level.use_distribution)
    patch_consistency = bool(config.diffusion.patch_level.use_consistency)
    patch_distribution = bool(config.diffusion.patch_level.use_distribution)
    errors: list[str] = []
    if not wsi_consistency and float(
        config.diffusion.wsi_level.consistency_loss_weight
    ) != 0:
        errors.append("disabled WSI consistency requires loss weight 0")
    if not wsi_distribution and float(
        config.diffusion.wsi_level.distribution_loss_weight
    ) != 0:
        errors.append("disabled WSI distribution requires loss weight 0")
    if skip_wsi:
        if patch_consistency or patch_distribution:
            errors.append(
                "skip_wsi_level requires both patch conditioning inputs disabled"
            )
    else:
        if patch_consistency != wsi_consistency:
            errors.append("WSI and patch consistency ablations must match")
        if patch_distribution != wsi_distribution:
            errors.append("WSI and patch distribution ablations must match")
    if str(config.generation.class_sampling) == "generated" and (
        skip_wsi or not wsi_distribution
    ):
        errors.append("generated class sampling requires WSI distribution diffusion")
    if errors:
        raise ValueError("inconsistent ablation wiring:\n- " + "\n- ".join(errors))


def build_wsi_diffusion(config: "Config") -> WSILevelDiffusion:
    """Construct the joint WSI-level DDPM from a resolved configuration."""

    denoiser = JointWSIDenoiser(
        wsi_dim=int(config.data.wsi_dim),
        num_classes=int(config.data.num_tissue_classes),
        hidden_dim=int(config.diffusion.hidden_dim),
        condition_dim=int(config.diffusion.condition_dim),
        time_dim=int(config.diffusion.time_embedding_dim),
        num_blocks=int(config.diffusion.num_residual_blocks),
        dropout=float(config.diffusion.dropout),
    )
    distribution_schedule = config.diffusion.wsi_level.distribution_schedule
    clip_x0 = config.diffusion.clip_x0
    return WSILevelDiffusion(
        denoiser=denoiser,
        consistency_schedule=build_schedule(
            config,
            int(config.diffusion.wsi_level.timesteps),
        ),
        distribution_schedule=(
            None
            if distribution_schedule is None
            else build_schedule(
                config,
                int(config.diffusion.wsi_level.timesteps),
                distribution_schedule,
            )
        ),
        consistency_weight=float(
            config.diffusion.wsi_level.consistency_loss_weight
        ),
        distribution_weight=float(
            config.diffusion.wsi_level.distribution_loss_weight
        ),
        variance_type=str(config.diffusion.sampling_variance),
        clip_x0=None if clip_x0 is None else float(clip_x0),
        distribution_parameterization=str(
            config.diffusion.wsi_level.distribution_parameterization
        ),
        use_consistency=bool(config.diffusion.wsi_level.use_consistency),
        use_distribution=bool(config.diffusion.wsi_level.use_distribution),
    )


def build_patch_diffusion(config: "Config") -> PatchLevelDiffusion:
    """Construct the patch-level conditional DDPM."""

    validate_ablation_wiring(config)
    conditioner = WSIClassConditioner(
        wsi_dim=int(config.data.wsi_dim),
        num_classes=int(config.data.num_tissue_classes),
        condition_dim=int(config.diffusion.condition_dim),
        use_consistency=bool(config.diffusion.patch_level.use_consistency),
        use_distribution=bool(config.diffusion.patch_level.use_distribution),
    )
    denoiser = PatchDenoiser(
        patch_dim=int(config.data.patch_dim),
        condition_dim=int(config.diffusion.condition_dim),
        hidden_dim=int(config.diffusion.hidden_dim),
        time_dim=int(config.diffusion.time_embedding_dim),
        num_blocks=int(config.diffusion.num_residual_blocks),
        dropout=float(config.diffusion.dropout),
    )
    patch_schedule = config.diffusion.patch_level.schedule
    clip_x0 = config.diffusion.clip_x0
    return PatchLevelDiffusion(
        conditioner=conditioner,
        denoiser=denoiser,
        schedule=build_schedule(
            config,
            int(config.diffusion.patch_level.timesteps),
            patch_schedule,
        ),
        variance_type=str(config.diffusion.sampling_variance),
        clip_x0=None if clip_x0 is None else float(clip_x0),
    )


def build_hierarchical_generator(config: "Config") -> HierarchicalWSIGenerator:
    """Construct the complete two-level generator."""

    validate_ablation_wiring(config)
    return HierarchicalWSIGenerator(
        wsi_diffusion=build_wsi_diffusion(config),
        patch_diffusion=build_patch_diffusion(config),
        patches_per_slide=int(config.generation.patches_per_generated_slide),
        minimum_per_nonzero_class=int(
            config.generation.min_patches_per_nonzero_class
        ),
        skip_wsi_level=bool(config.generation.skip_wsi_level),
        class_sampling=str(config.generation.class_sampling),
        distribution_temperature=float(
            config.generation.distribution_temperature
        ),
    )


def build_survival_model(config: "Config") -> HierarchicalCMILSurvival:
    """Construct the hierarchical C-MIL Cox risk model."""

    return HierarchicalCMILSurvival(
        patch_dim=int(config.data.patch_dim),
        cmil_hidden_dim=int(config.survival_model.cmil_hidden_dim),
        cmil_output_dim=int(config.survival_model.cmil_output_dim),
        cmil_dropout=float(config.survival_model.cmil_dropout),
        patient_hidden_dims=[
            int(width) for width in config.survival_model.patient_hidden_dims
        ],
        dropout=float(config.survival_model.dropout),
    )


__all__ = [
    "CMILPatchEncoder",
    "CoxPHLoss",
    "DiffusionSchedule",
    "GeneratedSlide",
    "HierarchicalCMILSurvival",
    "HierarchicalDiffusionLoss",
    "HierarchicalWSIGenerator",
    "JointWSIDenoiser",
    "MaskedMeanPooling",
    "PatchDenoiser",
    "PatchLevelDiffusion",
    "ResidualFiLMBlock",
    "SinusoidalTimeEmbedding",
    "WSIClassConditioner",
    "WSILevelDiffusion",
    "WSILevelOutput",
    "build_hierarchical_generator",
    "build_mlp",
    "build_patch_diffusion",
    "build_schedule",
    "build_survival_model",
    "build_wsi_diffusion",
    "ddpm_step",
    "initialize_denoiser",
    "make_beta_schedule",
    "negative_cox_partial_log_likelihood",
    "number_to_generate",
    "project_to_probability_simplex",
    "random_normal_like",
    "validate_ablation_wiring",
]
