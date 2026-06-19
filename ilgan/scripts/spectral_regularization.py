"""
Novel spectral regularization technique for ILGAN dual-output GAN.

This module provides a mathematically grounded spectral regularizer that
penalizes the spectral norm (largest singular value) of convolutional layers
with spectral normalization, keeping the Lipschitz constant of the network
bounded and improving training stability.

Mathematical motivation
----------------------
Let :math:`f_\\theta` be a neural network (or a sub-network thereof) composed
of layers :math:`\\{f^{(1)}, f^{(2)}, \\dots, f^{(L)}\\}`.  The Lipschitz
constant of the entire network satisfies:

.. math::

    \\text{Lip}(f_\\theta) \\leq \\prod_{i=1}^{L} \\sigma_{\\max}(W^{(i)})

where :math:`\\sigma_{\\max}(W^{(i)})` is the spectral norm (largest singular
value) of the weight matrix of layer :math:`i`.  By constraining the spectral
norm of each layer, we bound the overall Lipschitz constant, which:

1. **Stabilises GAN training**: prevents discriminator/generator gradient
   explosions (Miyato et al., 2018).
2. **Prevents mode collapse**: a bounded Lipschitz constant ensures the
   discriminator cannot assign arbitrarily high scores to real samples and
   arbitrarily low scores to fake samples, which would cause vanishing
   gradients for the generator.
3. **Improves generalisation**: spectral regularization acts as a form of
   weight decay in the spectral domain, preventing the network from
   overfitting to spurious high-frequency patterns.

Spectral regularization penalty
-------------------------------
For each layer :math:`i` with spectral normalization, let
:math:`\\sigma_i = \\sigma_{\\max}(W^{(i)})` be its current spectral norm.
We define the spectral regularization penalty as:

.. math::

    \\mathcal{L}_{\\text{spec}} = \\sum_{i=1}^{L}
        \\max(0, \\sigma_i - \\sigma_{\\text{target}})^2

where :math:`\\sigma_{\\text{target}}` is the target spectral norm (default
1.0).  This is a **hinge-squared penalty**: layers whose spectral norm is
already below the target incur no penalty, while layers exceeding the target
are penalized quadratically.

The total generator loss becomes:

.. math::

    \\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{gen}}
        + \\lambda \\cdot \\mathcal{L}_{\\text{spec}}

where :math:`\\lambda` is the regularization weight (default 0.001).

Computing the spectral norm
----------------------------
For a convolutional layer with weight :math:`W \\in \\mathbb{R}^{C_{\\text{out}} \\times C_{\\text{in}} \\times k_H \\times k_W}`,
we reshape it to a 2D matrix :math:`\\tilde{W} \\in \\mathbb{R}^{C_{\\text{out}} \\times (C_{\\text{in}} \\cdot k_H \\cdot k_W)}`
and compute the largest singular value via singular value decomposition (SVD):

.. math::

    \\tilde{W} = U \\Sigma V^T, \\quad
    \\sigma_{\\max} = \\Sigma_{[0,0]}

For efficiency, we use ``torch.linalg.svd`` with ``full_matrices=False``,
which computes only the singular values without materialising the full
:math:`U` and :math:`V^T` matrices.

For layers with PyTorch's ``spectral_norm`` applied, we access the
**original (un-normalized) weight** via the ``weight_orig`` attribute and
compute its spectral norm.  The effective weight used in the forward pass
is :math:`W_{\\text{eff}} = W_{\\text{orig}} / \\sigma(W_{\\text{orig}})`,
so regularizing :math:`\\sigma(W_{\\text{orig}})` indirectly controls the
spectral norm of the effective weight.

Usage
-----
The spectral regularizer is typically used in the training loop::

    from ilgan.scripts.spectral_regularization import (
        spectral_regularizer,
        add_spectral_regularization,
    )

    # In the training loop:
    outputs = generator(z)
    gen_loss = adversarial_loss(outputs["image"], ...)

    # Add spectral regularization
    gen_loss = add_spectral_regularization(
        gen_loss,
        generator,
        weight=0.001,
        target_spectral_norm=1.0,
    )

    gen_loss.backward()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon to prevent numerical issues in SVD."""

_DEFAULT_TARGET_SPECTRAL_NORM: float = 1.0
"""Default target spectral norm :math:`\\sigma_{\\text{target}}`.  A value of
1.0 ensures the effective weight of each spectrally-normalized layer has a
spectral norm no larger than 1.0."""

_DEFAULT_REGULARIZATION_WEIGHT: float = 0.001
"""Default weight :math:`\\lambda` for the spectral regularization penalty.
This is a small value that adds a gentle constraint without overwhelming the
primary adversarial loss."""


# ──────────────────────────────────────────────────────────────────────────────
# Spectral norm computation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_spectral_norm(module: nn.Module) -> Optional[torch.Tensor]:
    """Compute the spectral norm (largest singular value) of a module's
    weight.

    This function handles two cases:

    1. **Spectral-normalized layers** (``nn.utils.spectral_norm`` applied):
       The module has a ``weight_orig`` attribute.  We compute the spectral
       norm of the original (un-normalized) weight, which is the quantity
       that spectral normalization constrains.

    2. **Regular Conv2d layers** (no spectral norm): The module has a
       ``weight`` attribute that is a plain ``nn.Parameter``.  We compute
       the spectral norm of the weight directly.

    For both cases, the weight is reshaped from
    ``[C_out, C_in, kH, kW]`` to ``[C_out, C_in * kH * kW]`` and the
    largest singular value is computed via ``torch.linalg.svd``.

    Parameters
    ----------
    module : nn.Module
        A module (typically ``nn.Conv2d`` or ``SpectralNormConv2d``) that
        has a ``weight`` or ``weight_orig`` attribute.

    Returns
    -------
    torch.Tensor or None
        A scalar tensor containing the spectral norm (largest singular
        value), or ``None`` if the module does not have a suitable weight
        attribute.

    Notes
    -----
    - ``torch.linalg.svd`` is used with ``full_matrices=False`` for
      efficiency.  For very large weight matrices (e.g., the initial
      linear projection in the generator), this can be expensive.  In
      such cases, consider using power iteration instead (see
      :func:`_estimate_spectral_norm_power_iteration`).
    - The computation is performed on the module's current device and
      does **not** modify the module's state.
    """
    # Determine the weight tensor to use
    if hasattr(module, "weight_orig") and module.weight_orig is not None:
        # Spectral normalization was applied — use the original weight
        weight = module.weight_orig
    elif hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
        # Plain Conv2d or Linear layer — use the weight directly
        weight = module.weight
    else:
        return None

    # Reshape to 2D: [C_out, C_in * kH * kW] for Conv2d, or
    # [out_features, in_features] for Linear
    if weight.dim() == 4:
        # Conv2d: [C_out, C_in, kH, kW] -> [C_out, C_in * kH * kW]
        weight_2d = weight.view(weight.size(0), -1)
    elif weight.dim() == 2:
        # Linear: [out_features, in_features] — already 2D
        weight_2d = weight
    elif weight.dim() == 1:
        # Bias or 1D parameter — spectral norm is not meaningful
        return None
    else:
        # Unsupported dimensionality
        return None

    # Compute the largest singular value via SVD
    # Use `torch.linalg.svd` with full_matrices=False for efficiency.
    # We only need the singular values (S), not U or V^H.
    try:
        # For small-to-moderate matrices, SVD is exact and fast.
        # For very large matrices, consider power iteration instead.
        _, S, _ = torch.linalg.svd(weight_2d, full_matrices=False)
        sigma = S[0]  # Largest singular value
    except RuntimeError:
        # Fallback: if SVD fails (e.g., on CUDA out of memory for very
        # large matrices), use power iteration as a fallback.
        sigma = _estimate_spectral_norm_power_iteration(weight_2d)

    return sigma


def _estimate_spectral_norm_power_iteration(
    W: torch.Tensor,
    num_iters: int = 10,
) -> torch.Tensor:
    """Estimate the spectral norm (largest singular value) of a 2D matrix
    using power iteration.

    Power iteration is a memory-efficient alternative to full SVD for very
    large weight matrices.  It approximates the largest singular value by
    iteratively applying :math:`W^T W` to a random vector.

    Algorithm
    ---------
    Given a matrix :math:`W \\in \\mathbb{R}^{m \\times n}`:

    1. Initialise :math:`u \\in \\mathbb{R}^m` with random normal values.
    2. For :math:`t = 1 \\dots T`:
       a. :math:`v = W^T u / \\|W^T u\\|_2`
       b. :math:`u = W v / \\|W v\\|_2`
    3. :math:`\\sigma \\approx u^T W v`

    The estimate converges to the true largest singular value as
    :math:`T \\to \\infty`.  With :math:`T = 10`, the estimate is typically
    within 1% of the true value for random matrices.

    Parameters
    ----------
    W : torch.Tensor
        A 2D matrix of shape ``[m, n]``.
    num_iters : int, optional
        Number of power iterations.  More iterations yield a more accurate
        estimate.  (default: 10)

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the estimated spectral norm (largest
        singular value).

    Notes
    -----
    - This function does **not** modify the input matrix.
    - The random initialisation is seeded by the current PyTorch RNG state.
    - For matrices smaller than 10,000 elements, SVD is typically faster
      and more accurate.  Power iteration is recommended for matrices
      larger than 100,000 elements.
    """
    m, n = W.shape
    device = W.device
    dtype = W.dtype

    # 1. Initialise random vector u
    u = torch.randn(m, 1, device=device, dtype=dtype)
    u = u / torch.norm(u, p=2)

    # 2. Power iteration
    for _ in range(num_iters):
        # v = W^T u / ||W^T u||
        v = W.T @ u
        v_norm = torch.norm(v, p=2)
        if v_norm > _EPS:
            v = v / v_norm

        # u = W v / ||W v||
        u = W @ v
        u_norm = torch.norm(u, p=2)
        if u_norm > _EPS:
            u = u / u_norm

    # 3. Estimate sigma = u^T W v
    sigma = (u.T @ (W @ v)).squeeze()

    return sigma


# ──────────────────────────────────────────────────────────────────────────────
# Spectral regularizer
# ──────────────────────────────────────────────────────────────────────────────


def spectral_regularizer(
    model: nn.Module,
    target_spectral_norm: float = _DEFAULT_TARGET_SPECTRAL_NORM,
    verbose: bool = False,
) -> torch.Tensor:
    """Compute the spectral regularization penalty for all convolutional
    layers with spectral normalization in a model.

    The penalty is defined as:

    .. math::

        \\mathcal{L}_{\\text{spec}} = \\sum_{i \\in \\mathcal{S}}
            \\max(0, \\sigma_i - \\sigma_{\\text{target}})^2

    where :math:`\\mathcal{S}` is the set of all sub-modules that have
    spectral normalization applied (i.e., have a ``weight_orig`` attribute),
    :math:`\\sigma_i` is the spectral norm (largest singular value) of
    layer :math:`i`, and :math:`\\sigma_{\\text{target}}` is the target
    spectral norm.

    The penalty is **hinge-squared**: layers whose spectral norm is already
    below the target incur no penalty, while layers exceeding the target are
    penalized quadratically.  This encourages the spectral norms to stay
    close to (or below) the target value.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model (typically the ILGAN generator or discriminator)
        to regularize.  The function recursively iterates over all
        sub-modules via ``model.modules()``.
    target_spectral_norm : float, optional
        The target spectral norm :math:`\\sigma_{\\text{target}}`.  Must be
        positive.  A value of 1.0 (default) ensures the effective weight of
        each spectrally-normalized layer has a spectral norm no larger than
        1.0, which is the standard choice for spectral normalization.
        (default: 1.0)
    verbose : bool, optional
        If ``True``, print diagnostic information about each layer's
        spectral norm and the penalty contribution.  (default: ``False``)

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the total spectral regularization penalty
        :math:`\\mathcal{L}_{\\text{spec}}`.  This is a zero-dimensional
        tensor on the same device as the model's parameters.

    Raises
    ------
    ValueError
        If ``target_spectral_norm`` is not positive.

    Notes
    -----
    - The function only considers modules that have a ``weight_orig``
      attribute (indicating ``nn.utils.spectral_norm`` was applied) or
      plain ``nn.Conv2d`` / ``nn.Linear`` modules with a ``weight``
      attribute.
    - The penalty is computed on the **original (un-normalized)** weight
      for spectrally-normalized layers, which is the quantity that spectral
      normalization constrains.
    - The computation uses ``torch.linalg.svd`` for exact singular value
      computation, with a fallback to power iteration for very large
      matrices.

    Example
    -------
    >>> from ilgan.scripts.spectral_regularization import spectral_regularizer
    >>>
    >>> # Compute the spectral penalty for the generator
    >>> spec_penalty = spectral_regularizer(generator, target_spectral_norm=1.0)
    >>> print(f"Spectral penalty: {spec_penalty.item():.6f}")
    """
    if target_spectral_norm <= 0.0:
        raise ValueError(
            f"target_spectral_norm must be positive, got {target_spectral_norm}."
        )

    device: Optional[torch.device] = None
    total_penalty: torch.Tensor = torch.tensor(0.0)

    # Track per-layer statistics for diagnostics
    layer_stats: List[Dict[str, float]] = []

    # ── Iterate over all sub-modules ─────────────────────────────────────
    for name, module in model.named_modules():
        # Determine if this module has a weight we can regularize
        has_spectral_norm = hasattr(module, "weight_orig") and module.weight_orig is not None
        has_plain_weight = (
            not has_spectral_norm
            and hasattr(module, "weight")
            and isinstance(module.weight, nn.Parameter)
            and module.weight.dim() in (2, 4)  # Linear or Conv2d
        )

        if not (has_spectral_norm or has_plain_weight):
            continue

        # ── Compute the spectral norm ──────────────────────────────────
        sigma = _get_spectral_norm(module)
        if sigma is None:
            continue

        # Track device from the first valid module
        if device is None:
            device = sigma.device

        # ── Compute hinge-squared penalty ──────────────────────────────
        # L_i = max(0, sigma_i - target)^2
        excess = torch.relu(sigma - target_spectral_norm)
        penalty_i = excess ** 2

        total_penalty = total_penalty + penalty_i

        # ── Diagnostics ────────────────────────────────────────────────
        if verbose:
            layer_stats.append({
                "name": name,
                "sigma": sigma.item(),
                "excess": excess.item(),
                "penalty": penalty_i.item(),
            })

    # Move total_penalty to the model's device if we found any layers
    if device is not None:
        total_penalty = total_penalty.to(device=device)

    # ── Verbose logging ──────────────────────────────────────────────────
    if verbose and layer_stats:
        print("=" * 80)
        print(f"Spectral Regularization Diagnostics (target={target_spectral_norm})")
        print("=" * 80)
        print(f"{'Layer':<40} {'σ':<12} {'Excess':<12} {'Penalty':<12}")
        print("-" * 80)
        for stat in layer_stats:
            print(
                f"{stat['name']:<40} "
                f"{stat['sigma']:<12.6f} "
                f"{stat['excess']:<12.6f} "
                f"{stat['penalty']:<12.6f}"
            )
        print("-" * 80)
        print(f"{'Total penalty':<40} {total_penalty.item():<12.6f}")
        print("=" * 80)

    return total_penalty


# ──────────────────────────────────────────────────────────────────────────────
# Add spectral regularization to generator loss
# ──────────────────────────────────────────────────────────────────────────────


def add_spectral_regularization(
    generator_loss: torch.Tensor,
    model: nn.Module,
    weight: float = _DEFAULT_REGULARIZATION_WEIGHT,
    target_spectral_norm: float = _DEFAULT_TARGET_SPECTRAL_NORM,
    verbose: bool = False,
) -> torch.Tensor:
    """Add spectral regularization to the generator loss.

    This function computes the spectral regularization penalty for the
    given model and adds it to the generator loss:

    .. math::

        \\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{gen}}
            + \\lambda \\cdot \\mathcal{L}_{\\text{spec}}

    where :math:`\\lambda` is the regularization weight and
    :math:`\\mathcal{L}_{\\text{spec}}` is the spectral regularization
    penalty from :func:`spectral_regularizer`.

    The spectral regularization term is **detached from the computation
    graph** of the generator loss, meaning it only contributes its own
    gradients and does not interfere with the gradient flow of the
    adversarial loss.  This is important because the spectral regularization
    is an auxiliary constraint, not a replacement for the adversarial loss.

    Parameters
    ----------
    generator_loss : torch.Tensor
        The primary generator loss (e.g., adversarial loss + box regression
        loss + auxiliary losses).  Must be a scalar tensor that requires
        gradient.
    model : nn.Module
        The model to regularize (typically the ILGAN generator).  The
        function recursively iterates over all sub-modules via
        ``model.modules()`` to find spectrally-normalized layers.
    weight : float, optional
        The regularization weight :math:`\\lambda`.  This controls the
        strength of the spectral penalty relative to the primary loss.
        Must be non-negative.  (default: 0.001)
    target_spectral_norm : float, optional
        The target spectral norm :math:`\\sigma_{\\text{target}}` for the
        hinge-squared penalty.  Must be positive.  (default: 1.0)
    verbose : bool, optional
        If ``True``, print diagnostic information about each layer's
        spectral norm and the penalty contribution.  (default: ``False``)

    Returns
    -------
    torch.Tensor
        The combined loss :math:`\\mathcal{L}_{\\text{total}}` as a scalar
        tensor.  This tensor has the same device and dtype as the input
        ``generator_loss``.

    Raises
    ------
    TypeError
        If ``generator_loss`` is not a ``torch.Tensor``.
    ValueError
        If ``weight`` is negative or ``target_spectral_norm`` is not
        positive.

    Example
    -------
    >>> from ilgan.scripts.spectral_regularization import (
    ...     add_spectral_regularization,
    ... )
    >>>
    >>> # In the training loop:
    >>> outputs = generator(z)
    >>> gen_loss = adversarial_loss(outputs["image"], ...)
    >>> gen_loss = add_spectral_regularization(
    ...     gen_loss,
    ...     generator,
    ...     weight=0.001,
    ...     target_spectral_norm=1.0,
    ... )
    >>> gen_loss.backward()
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if not isinstance(generator_loss, torch.Tensor):
        raise TypeError(
            f"Expected 'generator_loss' to be a torch.Tensor, "
            f"got {type(generator_loss).__name__}."
        )
    if generator_loss.dim() != 0:
        raise ValueError(
            f"Expected 'generator_loss' to be a scalar tensor (dim=0), "
            f"got tensor of shape {generator_loss.shape}."
        )
    if weight < 0.0:
        raise ValueError(
            f"Regularization weight must be non-negative, got {weight}."
        )
    if target_spectral_norm <= 0.0:
        raise ValueError(
            f"target_spectral_norm must be positive, got {target_spectral_norm}."
        )

    # ── Compute spectral regularization penalty ─────────────────────────
    spec_penalty = spectral_regularizer(
        model=model,
        target_spectral_norm=target_spectral_norm,
        verbose=verbose,
    )

    # ── Add to generator loss ───────────────────────────────────────────
    # The spectral penalty is a scalar on the model's device.  We ensure
    # it is on the same device as the generator loss.
    spec_penalty = spec_penalty.to(device=generator_loss.device, dtype=generator_loss.dtype)

    # Combined loss: L_total = L_gen + lambda * L_spec
    total_loss = generator_loss + weight * spec_penalty

    return total_loss


# ──────────────────────────────────────────────────────────────────────────────
# Spectral norm monitoring (for logging / diagnostics)
# ──────────────────────────────────────────────────────────────────────────────


def get_spectral_norms(
    model: nn.Module,
) -> Dict[str, float]:
    """Retrieve the current spectral norms of all spectrally-normalized
    layers in a model.

    This is a diagnostic utility that returns a dictionary mapping layer
    names to their current spectral norms.  It is useful for logging
    during training (e.g., to TensorBoard or WandB) to monitor whether
    the spectral norms are staying within the desired range.

    Parameters
    ----------
    model : nn.Module
        The model to inspect.

    Returns
    -------
    dict of str -> float
        A dictionary mapping layer names (as returned by
        ``model.named_modules()``) to their current spectral norm values.
        Only layers with a ``weight_orig`` attribute (spectral norm applied)
        or plain ``nn.Conv2d`` / ``nn.Linear`` layers are included.

    Example
    -------
    >>> from ilgan.scripts.spectral_regularization import get_spectral_norms
    >>>
    >>> norms = get_spectral_norms(generator)
    >>> for name, sigma in norms.items():
    ...     print(f"{name}: σ = {sigma:.4f}")
    """
    spectral_norms: Dict[str, float] = {}

    for name, module in model.named_modules():
        has_spectral_norm = hasattr(module, "weight_orig") and module.weight_orig is not None
        has_plain_weight = (
            not has_spectral_norm
            and hasattr(module, "weight")
            and isinstance(module.weight, nn.Parameter)
            and module.weight.dim() in (2, 4)
        )

        if not (has_spectral_norm or has_plain_weight):
            continue

        sigma = _get_spectral_norm(module)
        if sigma is not None:
            spectral_norms[name] = sigma.item()

    return spectral_norms


# ──────────────────────────────────────────────────────────────────────────────
# Spectral norm constraint (hard projection)
# ──────────────────────────────────────────────────────────────────────────────


def constrain_spectral_norms(
    model: nn.Module,
    max_spectral_norm: float = 1.0,
) -> None:
    """Hard-constrain the spectral norms of all spectrally-normalized
    layers by projecting the weights back onto the feasible set.

    For each layer :math:`i` with spectral norm :math:`\\sigma_i`, if
    :math:`\\sigma_i > \\sigma_{\\max}`, the weight is scaled down:

    .. math::

        W_i \\leftarrow W_i \\cdot \\frac{\\sigma_{\\max}}{\\sigma_i}

    This is a **hard projection** onto the set of matrices with spectral
    norm :math:`\\leq \\sigma_{\\max}`.  Unlike the soft penalty in
    :func:`spectral_regularizer`, this directly modifies the model's
    parameters.

    This function is useful as a post-processing step after each optimizer
    step (similar to weight clipping in WGAN-GP) to enforce a hard Lipschitz
    constraint.

    Parameters
    ----------
    model : nn.Module
        The model whose weights to constrain.
    max_spectral_norm : float, optional
        The maximum allowed spectral norm.  Must be positive.
        (default: 1.0)

    Raises
    ------
    ValueError
        If ``max_spectral_norm`` is not positive.

    Notes
    -----
    - This function modifies the model's parameters **in-place**.
    - For spectrally-normalized layers, the constraint is applied to the
      **original (un-normalized)** weight (``weight_orig``), which is the
      quantity that spectral normalization constrains.
    - This is a stronger constraint than the soft penalty and may interfere
      with training if applied too aggressively.  It is recommended to use
      the soft penalty (:func:`add_spectral_regularization`) during training
      and only use this function for evaluation or as a last resort.

    Example
    -------
    >>> from ilgan.scripts.spectral_regularization import constrain_spectral_norms
    >>>
    >>> # After each optimizer step:
    >>> constrain_spectral_norms(generator, max_spectral_norm=1.0)
    """
    if max_spectral_norm <= 0.0:
        raise ValueError(
            f"max_spectral_norm must be positive, got {max_spectral_norm}."
        )

    with torch.no_grad():
        for name, module in model.named_modules():
            # Only consider layers with spectral normalization
            if not (hasattr(module, "weight_orig") and module.weight_orig is not None):
                continue

            weight = module.weight_orig
            if weight.dim() == 4:
                weight_2d = weight.view(weight.size(0), -1)
            elif weight.dim() == 2:
                weight_2d = weight
            else:
                continue

            # Compute spectral norm
            _, S, _ = torch.linalg.svd(weight_2d, full_matrices=False)
            sigma = S[0]

            # Project if exceeding max
            if sigma > max_spectral_norm:
                scale = max_spectral_norm / sigma
                weight.mul_(scale)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "spectral_regularizer",
    "add_spectral_regularization",
    "get_spectral_norms",
    "constrain_spectral_norms",
    "_get_spectral_norm",
    "_estimate_spectral_norm_power_iteration",
]
