"""
Runtime verification of the mathematical properties of the ILGAN dual-output
GAN system.

This module provides four verification functions that empirically validate
the core mathematical guarantees of the ILGAN architecture:

1. **Repulsion property** (``verify_repulsion_property``): verifies that
   the spatial slot attention centres of mass are separated by at least
   ``τ - ε`` (the repulsion threshold minus a small tolerance).  This is
   the primary mathematical mechanism against bounding box collapse.

2. **Gradient penalty property** (``verify_gradient_penalty_property``):
   verifies that the WGAN-GP gradient penalty constraint is satisfied:
   the mean gradient norm at interpolated points should be close to 1.0
   (the 1-Lipschitz equilibrium condition).

3. **Latent distribution property** (``verify_latent_distribution``):
   verifies that the generator's running latent statistics converge to
   the prior distribution ``𝒩(0, I)`` — mean close to 0 and variance
   close to 1.

4. **Consistency property** (``verify_consistency_property``): verifies
   that the cross-modal consistency between image and box features is
   high, as measured by cosine similarity in the shared feature space.

Each function returns a dictionary of results that can be logged, printed,
or used in assertions for automated testing.

Mathematical background
-----------------------
The ILGAN system is built on several mathematical guarantees that must hold
for the dual image-and-box generation to work correctly:

**Repulsion guarantee**: Let :math:`c_i = (cx_i, cy_i)` be the centre of
mass of slot :math:`i`'s attention distribution.  The repulsion loss
penalises pairs with :math:`\|c_i - c_j\|_2 < τ`.  At equilibrium, we
expect :math:`\|c_i - c_j\|_2 \ge τ - ε` for all :math:`i \neq j`, where
:math:`ε` is a small tolerance accounting for numerical precision and
the fact that the repulsion loss is a soft penalty, not a hard constraint.

**Gradient penalty equilibrium**: The WGAN-GP discriminator is trained to
be 1-Lipschitz by penalising deviations of the gradient norm from 1.0 at
interpolated points.  At equilibrium, :math:`\mathbb{E}[\|\nabla D(x̂)\|_2] = 1`.

**Latent prior**: The generator's latent statistics module tracks the
running mean and variance of all latent vectors seen during training.
If the latents are drawn from :math:`\mathcal{N}(0, I)`, the running
statistics should converge to mean 0 and variance 1.

**Cross-modal consistency**: The image and box feature encoders project
both modalities into a shared feature space.  The consistency loss
minimises the cosine distance between them.  A high mean cosine similarity
(> 0.5) indicates good alignment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilgan.models.attention import SpatialContentCrossAttention
from ilgan.models.generator import ILGANGenerator
from ilgan.models.discriminator import ImageDiscriminator
from ilgan.losses.consistency import ImageFeatureEncoder, BoxFeatureEncoder

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon for numerical stability."""

_DEFAULT_REPULSION_TOLERANCE: float = 0.05
"""Tolerance ``ε`` for the repulsion threshold check.  Accounts for
numerical precision and the soft nature of the repulsion penalty."""

_DEFAULT_GP_TOLERANCE: float = 0.1
"""Tolerance for the gradient penalty equilibrium check.  The mean gradient
norm should be within ``1.0 ± gp_tolerance``."""

_DEFAULT_LATENT_MEAN_TOLERANCE: float = 0.05
"""Tolerance for the latent mean check.  The running mean should be within
``0.0 ± latent_mean_tolerance``."""

_DEFAULT_LATENT_VAR_TOLERANCE: float = 0.1
"""Tolerance for the latent variance check.  The running variance should be
within ``1.0 ± latent_var_tolerance``."""

_DEFAULT_CONSISTENCY_THRESHOLD: float = 0.5
"""Threshold for the mean cosine similarity.  A value above this indicates
good cross-modal consistency."""


# ──────────────────────────────────────────────────────────────────────────────
# 1. verify_repulsion_property
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def verify_repulsion_property(
    generator: ILGANGenerator,
    num_samples: int = 1000,
    repulsion_threshold: float = 0.2,
    tolerance: float = _DEFAULT_REPULSION_TOLERANCE,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    r"""Verify the repulsion property of the spatial slot attention.

    This function generates ``num_samples`` batches of attention maps from
    the generator's ``SpatialHead`` and computes the empirical distribution
    of pairwise slot centre-of-mass distances.  It then checks whether any
    pair of slots has a centre distance smaller than ``τ - ε``, where
    ``τ = repulsion_threshold`` and ``ε = tolerance``.

    The centre of mass for slot ``n`` in batch element ``b`` is:

    .. math::

        cx_{b,n} = \sum_{h,w} A_{b,n,h,w} \cdot \frac{w}{W-1}
        \qquad
        cy_{b,n} = \sum_{h,w} A_{b,n,h,w} \cdot \frac{h}{H-1}

    where :math:`A_{b,n}` is the attention distribution over the spatial
    grid of size ``H × W``.

    **What this verifies**: The repulsion loss in the SCCA module penalises
    slots whose centres are closer than ``τ``.  At equilibrium, the
    generator should have learned to spread its slot attention centres
    across the spatial domain such that no two slots are too close.
    If the minimum observed distance is significantly below ``τ - ε``,
    the repulsion mechanism may not be working correctly.

    Parameters
    ----------
    generator : ILGANGenerator
        The ILGAN generator model.  Must be in evaluation mode
        (``generator.eval()``) for deterministic behaviour.
    num_samples : int, optional
        Total number of samples to generate for the verification.
        These are split into batches of size ``batch_size``.
        (default: ``1000``)
    repulsion_threshold : float, optional
        The repulsion threshold ``τ`` used in the SCCA module.  This should
        match the value used during training (typically ``0.2``).
        (default: ``0.2``)
    tolerance : float, optional
        The tolerance ``ε`` for the threshold check.  The minimum observed
        distance should be at least ``τ - ε``.  (default: ``0.05``)
    batch_size : int, optional
        Batch size for generating samples.  Larger batches use more GPU
        memory but are faster.  (default: ``32``)
    device : torch.device, optional
        Device to run the verification on.  If ``None``, uses the
        generator's parameter device.  (default: ``None``)

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"min_distance"``: the minimum pairwise centre distance observed
          across all samples and all slot pairs (scalar float).
        - ``"mean_distance"``: the mean pairwise centre distance across all
          samples and all slot pairs (scalar float).
        - ``"std_distance"``: the standard deviation of pairwise centre
          distances (scalar float).
        - ``"fraction_violations"``: the fraction of slot pairs (across all
          samples) whose centre distance is below ``τ - ε``.  A value of
          ``0.0`` means no violations were detected.
        - ``"num_violations"``: the absolute number of violating pairs.
        - ``"total_pairs"``: the total number of slot pairs examined.
        - ``"threshold_effective"``: the effective threshold ``τ - ε`` used
          for the violation check.
        - ``"passed"``: ``1.0`` if the fraction of violations is zero,
          ``0.0`` otherwise.

    Raises
    ------
    TypeError
        If ``generator`` is not an ``ILGANGenerator`` instance.
    ValueError
        If ``num_samples``, ``batch_size``, or ``repulsion_threshold`` are
        not positive.
    RuntimeError
        If the generator does not produce attention maps in its auxiliary
        outputs.

    Example
    -------
    >>> from ilgan.models.generator import ILGANGenerator
    >>> from ilgan.utils.config import Config
    >>> cfg = Config()
    >>> gen = ILGANGenerator(cfg).eval()
    >>> result = verify_repulsion_property(gen, num_samples=100)
    >>> print(f"Min distance: {result['min_distance']:.4f}")
    >>> print(f"Violations: {result['fraction_violations']:.4f}")
    >>> assert result['passed'] == 1.0, "Repulsion property violated!"
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not isinstance(generator, ILGANGenerator):
        raise TypeError(
            f"Expected 'generator' to be an ILGANGenerator, "
            f"got {type(generator).__name__}."
        )
    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be positive, got {num_samples}."
        )
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}."
        )
    if repulsion_threshold <= 0.0:
        raise ValueError(
            f"repulsion_threshold must be positive, got {repulsion_threshold}."
        )

    # ── Determine device ──────────────────────────────────────────────────
    if device is None:
        device = next(generator.parameters()).device

    # ── Set model to eval mode ────────────────────────────────────────────
    was_training = generator.training
    generator.eval()

    # ── Compute number of batches ──────────────────────────────────────────
    num_batches = math.ceil(num_samples / batch_size)
    effective_batch_size = min(batch_size, num_samples)

    # ── Storage for all centre distances ──────────────────────────────────
    all_min_distances: List[float] = []
    all_mean_distances: List[float] = []
    all_violation_counts: List[int] = []
    total_pairs_per_batch: int = 0

    # ── Generate samples and compute distances ────────────────────────────
    for batch_idx in range(num_batches):
        # Determine current batch size
        remaining = num_samples - batch_idx * effective_batch_size
        current_batch_size = min(effective_batch_size, remaining)

        # Sample latent vectors
        z = torch.randn(
            current_batch_size,
            generator.latent_dim,
            device=device,
        )

        # Get skip features from content decoder
        _, skip_features = generator.content_decoder(z)

        # ── Set up forward hooks to capture attention maps ──────────────
        captured_attention_maps: List[torch.Tensor] = []

        def _make_hook(container: List[torch.Tensor]) -> callable:
            def _hook(module, input, output):
                # output is (Z, A_maps, aux_losses)
                # A_maps has shape [B, N, H, W]
                container.append(output[1].detach())
            return _hook

        hooks = []
        for scca_module in generator.spatial_head.scca_modules:
            hook = scca_module.register_forward_hook(
                _make_hook(captured_attention_maps)
            )
            hooks.append(hook)

        # Run the spatial head forward to trigger the hooks
        _, _, _, _ = generator.spatial_head(skip_features)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if len(captured_attention_maps) == 0:
            raise RuntimeError(
                "No attention maps were captured.  Ensure the SCCA modules "
                "are present in the spatial head."
            )

        # Use the attention maps from the **last** SCCA module (highest
        # resolution), as this is the most refined.
        A = captured_attention_maps[-1]  # [B, N, H, W]
        B, N, H, W = A.shape

        # ── Compute centre of mass for each slot ────────────────────────
        # Build normalised coordinate grids
        x_grid = torch.linspace(0.0, 1.0, W, device=device)
        y_grid = torch.linspace(0.0, 1.0, H, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        x_coords = xx.reshape(-1)  # [HW]
        y_coords = yy.reshape(-1)  # [HW]

        # Flatten attention: [B, N, H, W] -> [B, N, HW]
        A_flat = A.view(B, N, H * W)

        # Centre of mass: [B, N]
        cx = (A_flat * x_coords[None, None, :]).sum(dim=-1)
        cy = (A_flat * y_coords[None, None, :]).sum(dim=-1)

        # ── Compute pairwise distances ──────────────────────────────────
        # dx[b, i, j] = cx[b, i] - cx[b, j]
        dx = cx[:, :, None] - cx[:, None, :]  # [B, N, N]
        dy = cy[:, :, None] - cy[:, None, :]  # [B, N, N]
        dists = torch.sqrt(dx.pow(2) + dy.pow(2) + _EPS)  # [B, N, N]

        # Upper-triangular indices (i < j)
        triu_idx = torch.triu_indices(N, N, offset=1, device=device)
        pair_dists = dists[:, triu_idx[0], triu_idx[1]]  # [B, N_pairs]

        # Store statistics
        all_min_distances.append(pair_dists.min().item())
        all_mean_distances.append(pair_dists.mean().item())

        # Count violations: pairs with distance < threshold - tolerance
        effective_threshold = repulsion_threshold - tolerance
        violations = (pair_dists < effective_threshold).sum().item()
        all_violation_counts.append(violations)

        if total_pairs_per_batch == 0:
            total_pairs_per_batch = pair_dists.shape[1]

    # ── Aggregate results ────────────────────────────────────────────────
    total_pairs_examined = total_pairs_per_batch * num_batches
    total_violations = sum(all_violation_counts)

    min_distance = min(all_min_distances) if all_min_distances else 0.0
    mean_distance = (
        sum(all_mean_distances) / len(all_mean_distances)
        if all_mean_distances
        else 0.0
    )

    # Compute std of distances across all batches
    # (We recompute from the last batch's full distance matrix for efficiency)
    if len(captured_attention_maps) > 0:
        # Use the last batch's distances for std
        std_distance = pair_dists.std().item()
    else:
        std_distance = 0.0

    fraction_violations = (
        total_violations / total_pairs_examined if total_pairs_examined > 0 else 0.0
    )

    passed = 1.0 if fraction_violations == 0.0 else 0.0

    # ── Restore original training mode ────────────────────────────────────
    if was_training:
        generator.train()

    return {
        "min_distance": min_distance,
        "mean_distance": mean_distance,
        "std_distance": std_distance,
        "fraction_violations": fraction_violations,
        "num_violations": float(total_violations),
        "total_pairs": float(total_pairs_examined),
        "threshold_effective": repulsion_threshold - tolerance,
        "passed": passed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. verify_gradient_penalty_property
# ──────────────────────────────────────────────────────────────────────────────


def verify_gradient_penalty_property(
    discriminator: ImageDiscriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    num_interpolations: int = 1000,
    tolerance: float = _DEFAULT_GP_TOLERANCE,
) -> Dict[str, float]:
    r"""Verify the WGAN-GP gradient penalty equilibrium condition.

    The WGAN-GP discriminator is trained to be 1-Lipschitz by penalising
    deviations of the gradient norm from 1.0 at interpolated points between
    real and fake images.  At equilibrium, the expected gradient norm should
    be close to 1.0:

    .. math::

        \mathbb{E}_{x̂ \sim P_{x̂}}[\|\nabla_{x̂} D(x̂)\|_2] \approx 1.0

    where :math:`x̂ = ε · x_{real} + (1-ε) · x_{fake}` with
    :math:`ε \sim \mathcal{U}(0, 1)`.

    This function samples ``num_interpolations`` interpolated points,
    computes the gradient norm at each point, and reports the mean and
    standard deviation.  It also checks whether the mean is within
    ``1.0 ± tolerance``.

    **What this verifies**: If the mean gradient norm is significantly
    different from 1.0, the discriminator is not satisfying the 1-Lipschitz
    constraint, which can lead to unstable GAN training.  A mean gradient
    norm > 1.1 indicates the discriminator is too steep (gradient explosion
    risk), while a mean < 0.9 indicates it is too flat (vanishing gradients
    for the generator).

    Parameters
    ----------
    discriminator : ImageDiscriminator
        The ILGAN image discriminator.  Must be in evaluation mode.
    real_images : torch.Tensor
        A batch of real images, shape ``[B, 3, H, W]``.  These are used as
        one endpoint of the interpolation.  The batch size ``B`` can be any
        value; the function will sample ``num_interpolations`` points by
        randomly selecting from the batch.
    fake_images : torch.Tensor
        A batch of fake (generated) images, shape ``[B, 3, H, W]``.  These
        are used as the other endpoint of the interpolation.  Must have the
        same shape as ``real_images``.
    num_interpolations : int, optional
        Number of interpolated points to sample for the gradient norm
        computation.  More points give a more accurate estimate of the
        expected gradient norm.  (default: ``1000``)
    tolerance : float, optional
        Tolerance for the equilibrium check.  The mean gradient norm should
        be within ``1.0 ± tolerance``.  (default: ``0.1``)

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"mean_grad_norm"``: the mean gradient norm across all
          interpolated points (scalar float).
        - ``"std_grad_norm"``: the standard deviation of gradient norms
          (scalar float).
        - ``"min_grad_norm"``: the minimum gradient norm observed.
        - ``"max_grad_norm"``: the maximum gradient norm observed.
        - ``"fraction_near_one"``: the fraction of points whose gradient
          norm is within ``[1 - tolerance, 1 + tolerance]``.
        - ``"mean_deviation"``: the mean absolute deviation from 1.0,
          i.e., ``mean(|grad_norm - 1.0|)``.
        - ``"passed"``: ``1.0`` if the mean gradient norm is within
          ``1.0 ± tolerance``, ``0.0`` otherwise.

    Raises
    ------
    TypeError
        If ``discriminator`` is not an ``ImageDiscriminator``.
    ValueError
        If ``real_images`` and ``fake_images`` have different shapes.
    ValueError
        If ``num_interpolations`` is not positive.

    Example
    -------
    >>> from ilgan.models.discriminator import ImageDiscriminator
    >>> disc = ImageDiscriminator(disc_base_channels=64, image_size=128).eval()
    >>> real = torch.randn(16, 3, 128, 128)
    >>> fake = torch.randn(16, 3, 128, 128)
    >>> result = verify_gradient_penalty_property(disc, real, fake)
    >>> print(f"Mean grad norm: {result['mean_grad_norm']:.4f}")
    >>> print(f"Passed: {result['passed']}")
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not isinstance(discriminator, ImageDiscriminator):
        raise TypeError(
            f"Expected 'discriminator' to be an ImageDiscriminator, "
            f"got {type(discriminator).__name__}."
        )
    if real_images.shape != fake_images.shape:
        raise ValueError(
            f"real_images shape {real_images.shape} must match "
            f"fake_images shape {fake_images.shape}."
        )
    if num_interpolations <= 0:
        raise ValueError(
            f"num_interpolations must be positive, got {num_interpolations}."
        )

    # ── Determine device and dtype ────────────────────────────────────────
    device = real_images.device
    dtype = real_images.dtype
    B, C, H, W = real_images.shape

    # ── Set model to eval mode ────────────────────────────────────────────
    was_training = discriminator.training
    discriminator.eval()

    # ── Sample interpolated points ────────────────────────────────────────
    # We sample num_interpolations points by randomly selecting pairs from
    # the batch and interpolating.
    grad_norms: List[float] = []

    # Number of iterations: we sample in mini-batches to avoid OOM
    interp_batch_size = min(num_interpolations, 128)
    num_iters = math.ceil(num_interpolations / interp_batch_size)

    for _ in range(num_iters):
        current_batch_size = min(interp_batch_size, num_interpolations - len(grad_norms))
        if current_batch_size <= 0:
            break

        # Randomly select indices from the batch
        idx_real = torch.randint(0, B, (current_batch_size,), device=device)
        idx_fake = torch.randint(0, B, (current_batch_size,), device=device)

        # Select images
        real_selected = real_images[idx_real]  # [B', 3, H, W]
        fake_selected = fake_images[idx_fake]  # [B', 3, H, W]

        # Sample epsilon ~ Uniform(0, 1)
        epsilon = torch.rand(current_batch_size, 1, 1, 1, device=device, dtype=dtype)

        # Interpolate — requires_grad for gradient computation
        x_hat = epsilon * real_selected + (1.0 - epsilon) * fake_selected
        x_hat.requires_grad_(True)

        # ── Forward pass through discriminator ──────────────────────────
        # Note: we are NOT under torch.no_grad() here because we need
        # gradients w.r.t. x_hat.  The discriminator parameters themselves
        # do not receive gradients because we don't call backward() on them.
        local_scores, global_scores = discriminator(x_hat)

        # ── Compute gradient of D(x_hat) w.r.t. x_hat ──────────────────
        # Sum local and global scores to get a single scalar output
        local_pooled = local_scores.mean(dim=(2, 3))  # [B', 1]
        d_output = local_pooled.sum() + global_scores.sum()  # scalar

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_output,
            inputs=x_hat,
            grad_outputs=torch.ones_like(d_output),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        # gradients shape: [B', 3, H, W]

        # ── Compute gradient norms ──────────────────────────────────────
        # Flatten: [B', 3*H*W]
        gradients_flat = gradients.view(current_batch_size, -1)
        norms = gradients_flat.norm(2, dim=1)  # [B']
        grad_norms.extend(norms.detach().cpu().tolist())

    # ── Aggregate results ────────────────────────────────────────────────
    if len(grad_norms) == 0:
        return {
            "mean_grad_norm": 0.0,
            "std_grad_norm": 0.0,
            "min_grad_norm": 0.0,
            "max_grad_norm": 0.0,
            "fraction_near_one": 0.0,
            "mean_deviation": 0.0,
            "passed": 0.0,
        }

    grad_norms_tensor = torch.tensor(grad_norms)
    mean_grad_norm = grad_norms_tensor.mean().item()
    std_grad_norm = grad_norms_tensor.std().item()
    min_grad_norm = grad_norms_tensor.min().item()
    max_grad_norm = grad_norms_tensor.max().item()

    # Fraction of points with grad norm in [1 - tolerance, 1 + tolerance]
    near_one = (
        (grad_norms_tensor >= 1.0 - tolerance)
        & (grad_norms_tensor <= 1.0 + tolerance)
    ).float().mean().item()

    # Mean absolute deviation from 1.0
    mean_deviation = (grad_norms_tensor - 1.0).abs().mean().item()

    # Check if mean is within tolerance
    passed = 1.0 if abs(mean_grad_norm - 1.0) <= tolerance else 0.0

    # ── Restore original training mode ────────────────────────────────────
    if was_training:
        discriminator.train()

    return {
        "mean_grad_norm": mean_grad_norm,
        "std_grad_norm": std_grad_norm,
        "min_grad_norm": min_grad_norm,
        "max_grad_norm": max_grad_norm,
        "fraction_near_one": near_one,
        "mean_deviation": mean_deviation,
        "passed": passed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. verify_latent_distribution
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def verify_latent_distribution(
    generator: ILGANGenerator,
    num_samples: int = 10000,
    mean_tolerance: float = _DEFAULT_LATENT_MEAN_TOLERANCE,
    var_tolerance: float = _DEFAULT_LATENT_VAR_TOLERANCE,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    r"""Verify that the generator's latent statistics converge to the prior
    distribution :math:`\mathcal{N}(0, I)`.

    The ILGAN generator tracks running mean and variance of all latent
    vectors passed through it during training.  If the latents are drawn
    from :math:`\mathcal{N}(0, I)`, the running statistics should converge
    to:

    - **Mean**: :math:`\mu \approx 0` (element-wise, within
      ``mean_tolerance``).
    - **Variance**: :math:`\sigma^2 \approx 1` (element-wise, within
      ``var_tolerance``).

    This function generates ``num_samples`` latent vectors from
    :math:`\mathcal{N}(0, I)`, passes them through the generator (which
    updates the running statistics), and then reads the current running
    mean and variance from the generator's ``_latent_mean`` and
    ``_latent_var`` buffers.

    **What this verifies**: If the running mean or variance deviate
    significantly from the prior, it may indicate that:

    - The generator is receiving latents from a different distribution
      (e.g., if an encoder is being used instead of direct sampling).
    - The latent statistics tracking module has a bug in its Welford
      online algorithm implementation.
    - The learnable instance noise (``noise_std`` parameter) is
      distorting the latent distribution.

    Parameters
    ----------
    generator : ILGANGenerator
        The ILGAN generator model.  The running statistics are updated
        during the forward pass, so the generator should be in evaluation
        mode (``generator.eval()``) to avoid the instance noise injection
        that could bias the statistics.
    num_samples : int, optional
        Number of latent vectors to generate and pass through the generator.
        More samples give a more accurate estimate of the running statistics.
        (default: ``10000``)
    mean_tolerance : float, optional
        Tolerance for the mean check.  The absolute value of each element
        of the running mean should be within ``mean_tolerance`` of 0.
        (default: ``0.05``)
    var_tolerance : float, optional
        Tolerance for the variance check.  Each element of the running
        variance should be within ``var_tolerance`` of 1.0.
        (default: ``0.1``)
    batch_size : int, optional
        Batch size for generating samples.  (default: ``128``)
    device : torch.device, optional
        Device to run on.  If ``None``, uses the generator's parameter
        device.  (default: ``None``)

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"mean_mean"``: the mean of the running mean vector (i.e.,
          the average over all latent dimensions of the running mean).
          Should be close to 0.
        - ``"mean_std"``: the standard deviation of the running mean
          vector across latent dimensions.
        - ``"mean_max_abs"``: the maximum absolute value in the running
          mean vector.  Should be within ``mean_tolerance``.
        - ``"var_mean"``: the mean of the running variance vector.
          Should be close to 1.0.
        - ``"var_std"``: the standard deviation of the running variance
          vector across latent dimensions.
        - ``"var_max_deviation"``: the maximum absolute deviation of the
          running variance from 1.0.  Should be within ``var_tolerance``.
        - ``"count"``: the total number of latent vectors seen by the
          generator (including any previous training).
        - ``"mean_passed"``: ``1.0`` if ``mean_max_abs <= mean_tolerance``,
          ``0.0`` otherwise.
        - ``"var_passed"``: ``1.0`` if ``var_max_deviation <= var_tolerance``,
          ``0.0`` otherwise.
        - ``"passed"``: ``1.0`` if both mean and variance checks pass,
          ``0.0`` otherwise.

    Raises
    ------
    TypeError
        If ``generator`` is not an ``ILGANGenerator`` instance.
    ValueError
        If ``num_samples`` or ``batch_size`` are not positive.

    Example
    -------
    >>> from ilgan.models.generator import ILGANGenerator
    >>> from ilgan.utils.config import Config
    >>> cfg = Config()
    >>> gen = ILGANGenerator(cfg).eval()
    >>> result = verify_latent_distribution(gen, num_samples=5000)
    >>> print(f"Mean of means: {result['mean_mean']:.4f}")
    >>> print(f"Mean of variances: {result['var_mean']:.4f}")
    >>> assert result['passed'] == 1.0, "Latent distribution property violated!"
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not isinstance(generator, ILGANGenerator):
        raise TypeError(
            f"Expected 'generator' to be an ILGANGenerator, "
            f"got {type(generator).__name__}."
        )
    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be positive, got {num_samples}."
        )
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}."
        )

    # ── Determine device ──────────────────────────────────────────────────
    if device is None:
        device = next(generator.parameters()).device

    # ── Set model to eval mode ────────────────────────────────────────────
    was_training = generator.training
    generator.eval()

    # ── Generate samples and pass through generator ───────────────────────
    num_batches = math.ceil(num_samples / batch_size)
    effective_batch_size = min(batch_size, num_samples)

    for batch_idx in range(num_batches):
        remaining = num_samples - batch_idx * effective_batch_size
        current_batch_size = min(effective_batch_size, remaining)

        # Sample from N(0, I)
        z = torch.randn(
            current_batch_size,
            generator.latent_dim,
            device=device,
        )

        # Forward pass (updates running statistics)
        _ = generator(z)

    # ── Read running statistics ──────────────────────────────────────────
    running_mean = generator._latent_mean.clone()  # [latent_dim]
    running_var = generator._latent_var.clone()  # [latent_dim]
    count = generator._latent_count.item()

    # ── Compute statistics ───────────────────────────────────────────────
    mean_mean = running_mean.mean().item()
    mean_std = running_mean.std().item()
    mean_max_abs = running_mean.abs().max().item()

    var_mean = running_var.mean().item()
    var_std = running_var.std().item()
    var_max_deviation = (running_var - 1.0).abs().max().item()

    # ── Check tolerances ─────────────────────────────────────────────────
    mean_passed = 1.0 if mean_max_abs <= mean_tolerance else 0.0
    var_passed = 1.0 if var_max_deviation <= var_tolerance else 0.0
    passed = 1.0 if (mean_passed and var_passed) else 0.0

    # ── Restore original training mode ────────────────────────────────────
    if was_training:
        generator.train()

    return {
        "mean_mean": mean_mean,
        "mean_std": mean_std,
        "mean_max_abs": mean_max_abs,
        "var_mean": var_mean,
        "var_std": var_std,
        "var_max_deviation": var_max_deviation,
        "count": float(count),
        "mean_passed": mean_passed,
        "var_passed": var_passed,
        "passed": passed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. verify_consistency_property
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def verify_consistency_property(
    generator: ILGANGenerator,
    image_encoder: ImageFeatureEncoder,
    box_encoder: BoxFeatureEncoder,
    num_samples: int = 100,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    r"""Verify the cross-modal consistency between image and box features.

    The ILGAN system uses two encoders to project generated images and
    predicted bounding boxes into a shared feature space:

    - ``ImageFeatureEncoder``: maps a generated image
      :math:`I \in \mathbb{R}^{3 \times H \times W}` to a feature vector
      :math:`f_{img} \in \mathbb{R}^{D}`.
    - ``BoxFeatureEncoder``: maps the set of predicted bounding boxes
      :math:`B \in \mathbb{R}^{N \times 4}` and confidences
      :math:`c \in \mathbb{R}^{N \times 1}` to a feature vector
      :math:`f_{box} \in \mathbb{R}^{D}`.

    The consistency loss minimises the cosine distance between these two
    feature vectors:

    .. math::

        \mathcal{L}_{cons} = 1 - \cos(f_{img}, f_{box})

    At equilibrium, the mean cosine similarity across samples should be
    high (typically > 0.5), indicating that the image and box representations
    are well-aligned in the shared feature space.

    This function generates ``num_samples`` images and bounding boxes from
    the generator, encodes both modalities, and computes the mean and
    standard deviation of the cosine similarity.

    **What this verifies**: A low mean cosine similarity (< 0.3) indicates
    that the image and box encoders are not well-aligned, which means the
    consistency loss is not effectively pulling the two modalities together.
    This could happen if:

    - The consistency weight is too low.
    - The encoders are not being trained (e.g., frozen or not receiving
      gradients).
    - The generator is producing images and boxes that are semantically
      inconsistent (e.g., boxes that don't correspond to image content).

    Parameters
    ----------
    generator : ILGANGenerator
        The ILGAN generator.  Must be in evaluation mode.
    image_encoder : ImageFeatureEncoder
        The image feature encoder.  Must be in evaluation mode.
    box_encoder : BoxFeatureEncoder
        The box feature encoder.  Must be in evaluation mode.
    num_samples : int, optional
        Number of samples to generate for the verification.
        (default: ``100``)
    batch_size : int, optional
        Batch size for generation.  (default: ``32``)
    device : torch.device, optional
        Device to run on.  If ``None``, uses the generator's parameter
        device.  (default: ``None``)

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"mean_cosine_similarity"``: the mean cosine similarity between
          image and box features across all samples (scalar float).
          Ranges in ``[-1, 1]``.  Values > 0.5 indicate good consistency.
        - ``"std_cosine_similarity"``: the standard deviation of cosine
          similarities (scalar float).  Low values indicate consistent
          alignment across samples.
        - ``"min_cosine_similarity"``: the minimum cosine similarity
          observed.  A very low value (e.g., < 0) indicates some samples
          have poor alignment.
        - ``"max_cosine_similarity"``: the maximum cosine similarity
          observed.
        - ``"fraction_above_threshold"``: the fraction of samples whose
          cosine similarity is above 0.5 (the default consistency
          threshold).  A value close to 1.0 indicates most samples are
          well-aligned.
        - ``"passed"``: ``1.0`` if the mean cosine similarity is above
          0.5, ``0.0`` otherwise.

    Raises
    ------
    TypeError
        If any of the models are not of the expected type.
    ValueError
        If ``num_samples`` or ``batch_size`` are not positive.

    Example
    -------
    >>> from ilgan.models.generator import ILGANGenerator
    >>> from ilgan.losses.consistency import ImageFeatureEncoder, BoxFeatureEncoder
    >>> from ilgan.utils.config import Config
    >>> cfg = Config()
    >>> gen = ILGANGenerator(cfg).eval()
    >>> img_enc = ImageFeatureEncoder().eval()
    >>> box_enc = BoxFeatureEncoder().eval()
    >>> result = verify_consistency_property(gen, img_enc, box_enc, num_samples=50)
    >>> print(f"Mean cosine similarity: {result['mean_cosine_similarity']:.4f}")
    >>> print(f"Passed: {result['passed']}")
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not isinstance(generator, ILGANGenerator):
        raise TypeError(
            f"Expected 'generator' to be an ILGANGenerator, "
            f"got {type(generator).__name__}."
        )
    if not isinstance(image_encoder, ImageFeatureEncoder):
        raise TypeError(
            f"Expected 'image_encoder' to be an ImageFeatureEncoder, "
            f"got {type(image_encoder).__name__}."
        )
    if not isinstance(box_encoder, BoxFeatureEncoder):
        raise TypeError(
            f"Expected 'box_encoder' to be a BoxFeatureEncoder, "
            f"got {type(box_encoder).__name__}."
        )
    if num_samples <= 0:
        raise ValueError(
            f"num_samples must be positive, got {num_samples}."
        )
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}."
        )

    # ── Determine device ──────────────────────────────────────────────────
    if device is None:
        device = next(generator.parameters()).device

    # ── Set all models to eval mode ───────────────────────────────────────
    models = [generator, image_encoder, box_encoder]
    was_training = [m.training for m in models]
    for m in models:
        m.eval()

    # ── Generate samples and compute cosine similarities ─────────────────
    num_batches = math.ceil(num_samples / batch_size)
    effective_batch_size = min(batch_size, num_samples)

    all_cosine_sims: List[float] = []

    for batch_idx in range(num_batches):
        remaining = num_samples - batch_idx * effective_batch_size
        current_batch_size = min(effective_batch_size, remaining)

        # Sample latent vectors
        z = torch.randn(
            current_batch_size,
            generator.latent_dim,
            device=device,
        )

        # Generate
        gen_outputs = generator(z)
        fake_images = gen_outputs["image"]  # [B, 3, H, W], [-1, 1]
        pred_boxes = gen_outputs["boxes"]  # [B, N, 4], (cx, cy, w, h)
        confidences = gen_outputs["confidences"]  # [B, N, 1]

        # Create valid mask: all boxes are valid for generated samples
        # (the generator always outputs max_boxes boxes)
        B = current_batch_size
        N = pred_boxes.shape[1]
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Encode images
        image_features = image_encoder(fake_images)  # [B, D]

        # Encode boxes
        box_features = box_encoder(pred_boxes, confidences, valid_mask)  # [B, D]

        # Compute cosine similarity per sample
        img_norm = F.normalize(image_features, p=2, dim=-1, eps=_EPS)
        box_norm = F.normalize(box_features, p=2, dim=-1, eps=_EPS)
        cos_sim = (img_norm * box_norm).sum(dim=-1)  # [B]

        all_cosine_sims.extend(cos_sim.detach().cpu().tolist())

    # ── Aggregate results ────────────────────────────────────────────────
    if len(all_cosine_sims) == 0:
        return {
            "mean_cosine_similarity": 0.0,
            "std_cosine_similarity": 0.0,
            "min_cosine_similarity": 0.0,
            "max_cosine_similarity": 0.0,
            "fraction_above_threshold": 0.0,
            "passed": 0.0,
        }

    cos_sim_tensor = torch.tensor(all_cosine_sims)
    mean_cos = cos_sim_tensor.mean().item()
    std_cos = cos_sim_tensor.std().item()
    min_cos = cos_sim_tensor.min().item()
    max_cos = cos_sim_tensor.max().item()

    # Fraction of samples with cosine similarity > 0.5
    fraction_above = (cos_sim_tensor > _DEFAULT_CONSISTENCY_THRESHOLD).float().mean().item()

    # Pass if mean cosine similarity > 0.5
    passed = 1.0 if mean_cos > _DEFAULT_CONSISTENCY_THRESHOLD else 0.0

    # ── Restore original training modes ─────────────────────────────────
    for m, was_train in zip(models, was_training):
        if was_train:
            m.train()

    return {
        "mean_cosine_similarity": mean_cos,
        "std_cosine_similarity": std_cos,
        "min_cosine_similarity": min_cos,
        "max_cosine_similarity": max_cos,
        "fraction_above_threshold": fraction_above,
        "passed": passed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Comprehensive verification runner
# ──────────────────────────────────────────────────────────────────────────────


def run_all_verifications(
    generator: ILGANGenerator,
    discriminator: ImageDiscriminator,
    image_encoder: ImageFeatureEncoder,
    box_encoder: BoxFeatureEncoder,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    num_samples_repulsion: int = 500,
    num_samples_latent: int = 5000,
    num_samples_consistency: int = 100,
    num_interpolations_gp: int = 500,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run all four mathematical property verifications and return the
    combined results.

    This is a convenience function that calls all four verification
    functions in sequence and returns a nested dictionary of results.
    It is useful for:

    - Automated testing: run this at the end of each training epoch to
      verify that the model is maintaining its mathematical guarantees.
    - Debugging: if any property fails, the detailed results from each
      verification function help identify the issue.
    - Logging: the results can be logged to console, file, or wandb.

    Parameters
    ----------
    generator : ILGANGenerator
        The ILGAN generator.
    discriminator : ImageDiscriminator
        The ILGAN image discriminator.
    image_encoder : ImageFeatureEncoder
        The image feature encoder.
    box_encoder : BoxFeatureEncoder
        The box feature encoder.
    real_images : torch.Tensor
        A batch of real images for the gradient penalty verification.
    fake_images : torch.Tensor
        A batch of fake (generated) images for the gradient penalty
        verification.
    num_samples_repulsion : int, optional
        Number of samples for the repulsion property verification.
        (default: ``500``)
    num_samples_latent : int, optional
        Number of samples for the latent distribution verification.
        (default: ``5000``)
    num_samples_consistency : int, optional
        Number of samples for the consistency property verification.
        (default: ``100``)
    num_interpolations_gp : int, optional
        Number of interpolated points for the gradient penalty
        verification.  (default: ``500``)
    batch_size : int, optional
        Batch size for all verifications.  (default: ``32``)
    device : torch.device, optional
        Device to run on.  If ``None``, inferred from the generator.
        (default: ``None``)
    verbose : bool, optional
        If ``True``, print a summary of each verification result to
        stdout.  (default: ``True``)

    Returns
    -------
    dict of str -> dict of str -> float
        A nested dictionary with top-level keys ``"repulsion"``,
        ``"gradient_penalty"``, ``"latent_distribution"``, and
        ``"consistency"``, each containing the results from the
        corresponding verification function.

    Example
    -------
    >>> from ilgan.models.generator import ILGANGenerator
    >>> from ilgan.models.discriminator import ImageDiscriminator
    >>> from ilgan.losses.consistency import ImageFeatureEncoder, BoxFeatureEncoder
    >>> from ilgan.utils.config import Config
    >>> cfg = Config()
    >>> gen = ILGANGenerator(cfg).eval()
    >>> disc = ImageDiscriminator(64, 128).eval()
    >>> img_enc = ImageFeatureEncoder().eval()
    >>> box_enc = BoxFeatureEncoder().eval()
    >>> real = torch.randn(16, 3, 128, 128)
    >>> fake = torch.randn(16, 3, 128, 128)
    >>> results = run_all_verifications(gen, disc, img_enc, box_enc, real, fake)
    >>> for prop, res in results.items():
    ...     print(f"{prop}: passed={res['passed']}")
    """
    if device is None:
        device = next(generator.parameters()).device

    results: Dict[str, Dict[str, float]] = {}

    # ── 1. Repulsion property ────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Verification 1/4: Repulsion Property")
        print("=" * 60)

    repulsion_result = verify_repulsion_property(
        generator=generator,
        num_samples=num_samples_repulsion,
        batch_size=batch_size,
        device=device,
    )
    results["repulsion"] = repulsion_result

    if verbose:
        print(f"  Min distance:       {repulsion_result['min_distance']:.6f}")
        print(f"  Mean distance:      {repulsion_result['mean_distance']:.6f}")
        print(f"  Fraction violations: {repulsion_result['fraction_violations']:.6f}")
        print(f"  Passed:             {repulsion_result['passed']}")
        print()

    # ── 2. Gradient penalty property ──────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Verification 2/4: Gradient Penalty Property")
        print("=" * 60)

    gp_result = verify_gradient_penalty_property(
        discriminator=discriminator,
        real_images=real_images,
        fake_images=fake_images,
        num_interpolations=num_interpolations_gp,
    )
    results["gradient_penalty"] = gp_result

    if verbose:
        print(f"  Mean grad norm:     {gp_result['mean_grad_norm']:.6f}")
        print(f"  Std grad norm:      {gp_result['std_grad_norm']:.6f}")
        print(f"  Fraction near 1:    {gp_result['fraction_near_one']:.6f}")
        print(f"  Passed:             {gp_result['passed']}")
        print()

    # ── 3. Latent distribution property ──────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Verification 3/4: Latent Distribution Property")
        print("=" * 60)

    latent_result = verify_latent_distribution(
        generator=generator,
        num_samples=num_samples_latent,
        batch_size=batch_size,
        device=device,
    )
    results["latent_distribution"] = latent_result

    if verbose:
        print(f"  Mean of means:      {latent_result['mean_mean']:.6f}")
        print(f"  Mean of variances:  {latent_result['var_mean']:.6f}")
        print(f"  Max abs mean:       {latent_result['mean_max_abs']:.6f}")
        print(f"  Max var deviation:  {latent_result['var_max_deviation']:.6f}")
        print(f"  Count:              {latent_result['count']:.0f}")
        print(f"  Mean passed:        {latent_result['mean_passed']}")
        print(f"  Var passed:         {latent_result['var_passed']}")
        print(f"  Passed:             {latent_result['passed']}")
        print()

    # ── 4. Consistency property ──────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Verification 4/4: Consistency Property")
        print("=" * 60)

    consistency_result = verify_consistency_property(
        generator=generator,
        image_encoder=image_encoder,
        box_encoder=box_encoder,
        num_samples=num_samples_consistency,
        batch_size=batch_size,
        device=device,
    )
    results["consistency"] = consistency_result

    if verbose:
        print(f"  Mean cosine sim:    {consistency_result['mean_cosine_similarity']:.6f}")
        print(f"  Std cosine sim:     {consistency_result['std_cosine_similarity']:.6f}")
        print(f"  Fraction > 0.5:     {consistency_result['fraction_above_threshold']:.6f}")
        print(f"  Passed:             {consistency_result['passed']}")
        print()

    # ── Overall summary ──────────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Overall Verification Summary")
        print("=" * 60)
        all_passed = all(
            res.get("passed", 0.0) > 0.5 for res in results.values()
        )
        for prop_name, res in results.items():
            status = "✅ PASSED" if res.get("passed", 0.0) > 0.5 else "❌ FAILED"
            print(f"  {prop_name:25s}: {status}")
        print()
        if all_passed:
            print("  ✅ All mathematical properties verified successfully.")
        else:
            print("  ⚠  Some properties failed.  See details above.")
        print("=" * 60)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "verify_repulsion_property",
    "verify_gradient_penalty_property",
    "verify_latent_distribution",
    "verify_consistency_property",
    "run_all_verifications",
]
