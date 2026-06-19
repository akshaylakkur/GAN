"""
Adversarial losses for the ILGAN dual-output GAN.

This module implements the Wasserstein GAN with Gradient Penalty (WGAN-GP)
loss functions, adapted for the ILGAN's dual-output architecture where the
discriminator produces both **local** (spatial grid) and **global** (per-sample)
realism scores.

Mathematical foundation
-----------------------
The standard WGAN value function (Arjovsky et al., 2017) is:

    min_G max_D  E_{x ~ P_r}[D(x)] - E_{z ~ P_z}[D(G(z))]

where :math:`P_r` is the real data distribution and :math:`P_z` is the latent
prior.  The discriminator (critic) :math:`D` is constrained to be 1-Lipschitz.
The gradient penalty (Gulrajani et al., 2017) enforces this constraint softly:

    L_gp = λ * E_{x̂ ~ P_{x̂}}[(‖∇_{x̂} D(x̂)‖₂ - 1)²]

where :math:`x̂ = ε x_real + (1-ε) x_fake` with :math:`ε ~ Uniform(0,1)`.

ILGAN adaptation
-----------------
The ILGAN discriminator produces two outputs:

- ``local_scores`` :math:`∈ ℝ^{B × 1 × H × W}` — a spatial grid of realism
  scores (PatchGAN-style).
- ``global_score`` :math:`∈ ℝ^{B × 1}` — a single scalar per sample.

The loss functions in this module combine both outputs into a single scalar
loss via a weighted sum with configurable weight :math:`w_{global}`:

    L_D = L_D^{local} + w_{global} · L_D^{global}
    L_G = L_G^{local} + w_{global} · L_G^{global}

where:

    L_D^{local} = E[fake_scores_local] - E[real_scores_local]
    L_D^{global} = E[fake_scores_global] - E[real_scores_global]
    L_G^{local} = -E[fake_scores_local]
    L_G^{global} = -E[fake_scores_global]

Note that the discriminator loss is the **negative** of the standard WGAN
objective because we perform gradient **descent** on the discriminator
parameters (PyTorch convention), so we minimise :math:`-L_D^{WGAN}`.

All functions are compatible with ``torch.cuda.amp.autocast()`` mixed
precision training.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# WGAN Discriminator Loss
# ──────────────────────────────────────────────────────────────────────────────


def wgan_discriminator_loss(
    real_scores_local: torch.Tensor,
    real_scores_global: torch.Tensor,
    fake_scores_local: torch.Tensor,
    fake_scores_global: torch.Tensor,
    w_global: float = 0.5,
) -> torch.Tensor:
    r"""Compute the WGAN discriminator (critic) loss.

    The discriminator aims to maximise :math:`\mathbb{E}[D(x_{real})] -
    \mathbb{E}[D(x_{fake})]`.  Since we perform gradient descent, we minimise
    the negative of this quantity:

    .. math::

        L_D^{local} &= \mathbb{E}[D_{local}(x_{fake})] -
                       \mathbb{E}[D_{local}(x_{real})] \\
        L_D^{global} &= \mathbb{E}[D_{global}(x_{fake})] -
                        \mathbb{E}[D_{global}(x_{real})] \\
        L_D &= L_D^{local} + w_{global} \cdot L_D^{global}

    The local scores are mean-pooled over spatial dimensions before computing
    expectations, so each sample contributes its average local realism score.

    Parameters
    ----------
    real_scores_local : torch.Tensor
        Local realism scores for real images, shape ``[B, 1, H, W]``.
    real_scores_global : torch.Tensor
        Global realism scores for real images, shape ``[B, 1]``.
    fake_scores_local : torch.Tensor
        Local realism scores for fake (generated) images, shape ``[B, 1, H, W]``.
    fake_scores_global : torch.Tensor
        Global realism scores for fake (generated) images, shape ``[B, 1]``.
    w_global : float, optional
        Weight for the global score component of the loss.  Must be
        non-negative.  (default: ``0.5``)

    Returns
    -------
    torch.Tensor
        Scalar discriminator loss tensor (0-dimensional).

    Raises
    ------
    ValueError
        If ``w_global`` is negative.

    Shape
    -----
    - Inputs: ``real_scores_local`` and ``fake_scores_local`` have shape
      ``[B, 1, H, W]``.
    - Inputs: ``real_scores_global`` and ``fake_scores_global`` have shape
      ``[B, 1]``.
    - Output: scalar ``[]``.

    Example
    -------
    >>> real_local = torch.randn(4, 1, 4, 4)
    >>> real_global = torch.randn(4, 1)
    >>> fake_local = torch.randn(4, 1, 4, 4)
    >>> fake_global = torch.randn(4, 1)
    >>> loss = wgan_discriminator_loss(real_local, real_global, fake_local, fake_global)
    >>> loss.shape
    torch.Size([])
    """
    if w_global < 0.0:
        raise ValueError(
            f"w_global must be non-negative, got {w_global}."
        )

    # Mean-pool local scores over spatial dimensions: [B, 1, H, W] -> [B, 1]
    real_local_pooled = real_scores_local.mean(dim=(2, 3))  # [B, 1]
    fake_local_pooled = fake_scores_local.mean(dim=(2, 3))  # [B, 1]

    # Local component: E[fake_local] - E[real_local]
    loss_local = fake_local_pooled.mean() - real_local_pooled.mean()

    # Global component: E[fake_global] - E[real_global]
    loss_global = fake_scores_global.mean() - real_scores_global.mean()

    # Combined loss
    loss = loss_local + w_global * loss_global

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# WGAN Generator Loss
# ──────────────────────────────────────────────────────────────────────────────


def wgan_generator_loss(
    fake_scores_local: torch.Tensor,
    fake_scores_global: torch.Tensor,
    w_global: float = 0.5,
) -> torch.Tensor:
    r"""Compute the WGAN generator loss.

    The generator aims to minimise :math:`-\mathbb{E}[D(G(z))]`, i.e. it
    wants the discriminator to assign high realism scores to its outputs.
    The loss is:

    .. math::

        L_G^{local} &= -\mathbb{E}[D_{local}(x_{fake})] \\
        L_G^{global} &= -\mathbb{E}[D_{global}(x_{fake})] \\
        L_G &= L_G^{local} + w_{global} \cdot L_G^{global}

    The local scores are mean-pooled over spatial dimensions before computing
    the expectation.

    Parameters
    ----------
    fake_scores_local : torch.Tensor
        Local realism scores for fake (generated) images, shape ``[B, 1, H, W]``.
    fake_scores_global : torch.Tensor
        Global realism scores for fake (generated) images, shape ``[B, 1]``.
    w_global : float, optional
        Weight for the global score component of the loss.  Must be
        non-negative.  (default: ``0.5``)

    Returns
    -------
    torch.Tensor
        Scalar generator loss tensor (0-dimensional).

    Raises
    ------
    ValueError
        If ``w_global`` is negative.

    Shape
    -----
    - Input: ``fake_scores_local`` has shape ``[B, 1, H, W]``.
    - Input: ``fake_scores_global`` has shape ``[B, 1]``.
    - Output: scalar ``[]``.

    Example
    -------
    >>> fake_local = torch.randn(4, 1, 4, 4)
    >>> fake_global = torch.randn(4, 1)
    >>> loss = wgan_generator_loss(fake_local, fake_global)
    >>> loss.shape
    torch.Size([])
    """
    if w_global < 0.0:
        raise ValueError(
            f"w_global must be non-negative, got {w_global}."
        )

    # Mean-pool local scores over spatial dimensions: [B, 1, H, W] -> [B, 1]
    fake_local_pooled = fake_scores_local.mean(dim=(2, 3))  # [B, 1]

    # Local component: -E[fake_local]
    loss_local = -fake_local_pooled.mean()

    # Global component: -E[fake_global]
    loss_global = -fake_scores_global.mean()

    # Combined loss
    loss = loss_local + w_global * loss_global

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Penalty
# ──────────────────────────────────────────────────────────────────────────────


def gradient_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    r"""Compute the WGAN-GP gradient penalty.

    The gradient penalty enforces the 1-Lipschitz constraint on the
    discriminator by penalising the gradient norm at interpolated points
    between real and fake images (Gulrajani et al., 2017).

    .. math::

        x_{hat} &= \varepsilon \cdot x_{real} + (1 - \varepsilon) \cdot x_{fake}
                  \quad \text{where} \quad \varepsilon \sim \mathcal{U}(0, 1) \\
        L_{gp} &= \lambda \cdot
                 \mathbb{E}_{x_{hat} \sim P_{x_{hat}}}
                 \left[ \left( \| \nabla_{x_{hat}} D(x_{hat}) \|_2 - 1 \right)^2 \right]

    The gradient is computed with respect to the **sum** of the local and
    global discriminator outputs:

    .. math::

        D(x_{hat}) = \text{local\_hat} + \text{global\_hat}

    where ``local_hat`` is first mean-pooled over spatial dimensions to
    produce a scalar per sample, then summed across the batch.

    This function is compatible with ``torch.cuda.amp.autocast()``.  The
    gradient computation is performed in the dtype of the input images
    (typically ``float32``) to avoid numerical issues with low-precision
    gradients.

    Parameters
    ----------
    discriminator : nn.Module
        The ILGAN ``ImageDiscriminator`` module.  Its ``forward`` must return
        a tuple ``(local_scores, global_score)`` where ``local_scores`` has
        shape ``[B, 1, H, W]`` and ``global_score`` has shape ``[B, 1]``.
    real_images : torch.Tensor
        Batch of real images, shape ``[B, 3, H, W]``, pixel values in
        ``[-1, 1]``.  Gradients are detached (no gradient flows through
        real images).
    fake_images : torch.Tensor
        Batch of fake (generated) images, shape ``[B, 3, H, W]``, pixel
        values in ``[-1, 1]``.  Gradients are detached (no gradient flows
        through fake images).
    lambda_gp : float, optional
        Coefficient scaling the gradient penalty.  The original WGAN-GP
        paper uses ``lambda_gp = 10.0``.  (default: ``10.0``)

    Returns
    -------
    torch.Tensor
        Scalar gradient penalty loss tensor (0-dimensional), already
        multiplied by ``lambda_gp``.

    Raises
    ------
    ValueError
        If ``real_images`` and ``fake_images`` have different shapes.
    RuntimeError
        If the discriminator does not return a tuple of two tensors.

    Shape
    -----
    - Inputs: ``real_images`` and ``fake_images`` have shape
      ``[B, 3, H, W]``.
    - Output: scalar ``[]``.

    Example
    -------
    >>> disc = ImageDiscriminator(disc_base_channels=64, image_size=128)
    >>> real = torch.randn(4, 3, 128, 128)
    >>> fake = torch.randn(4, 3, 128, 128)
    >>> gp = gradient_penalty(disc, real, fake, lambda_gp=10.0)
    >>> gp.shape
    torch.Size([])
    """
    if real_images.shape != fake_images.shape:
        raise ValueError(
            f"real_images and fake_images must have the same shape, "
            f"got {real_images.shape} and {fake_images.shape}."
        )

    B = real_images.shape[0]
    device = real_images.device
    dtype = real_images.dtype

    # ── 1. Sample ε ~ Uniform(0, 1) broadcastable to image shape ──────
    # Shape: [B, 1, 1, 1] — broadcasts to [B, 3, H, W] when multiplied
    epsilon = torch.rand(B, 1, 1, 1, device=device, dtype=dtype)

    # ── 2. Compute interpolated images ────────────────────────────────
    # Detach real and fake so gradients don't flow back to generator/real data
    x_hat = epsilon * real_images.detach() + (1.0 - epsilon) * fake_images.detach()
    x_hat.requires_grad_(True)

    # ── 3. Pass through discriminator ──────────────────────────────────
    # Use autocast-compatible forward: the discriminator's forward is
    # called normally; autocast context is handled by the caller.
    local_hat, global_hat = discriminator(x_hat)

    # ── 4. Compute gradients of D(x_hat) w.r.t. x_hat ──────────────────
    # Mean-pool local scores to get a scalar per sample, then sum both
    # local and global to get a single scalar for gradient computation.
    local_pooled = local_hat.mean(dim=(2, 3))  # [B, 1]
    # Sum over batch and spatial dims to get a single scalar
    d_output = local_pooled.sum() + global_hat.sum()  # scalar

    # Compute gradients: ∇_{x_hat} D(x_hat)
    gradients = torch.autograd.grad(
        outputs=d_output,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # gradients shape: [B, 3, H, W]

    # ── 5. Compute gradient norm penalty ──────────────────────────────
    # Flatten spatial and channel dims: [B, 3*H*W]
    gradients_flat = gradients.view(B, -1)

    # L2 norm per sample: ||∇D(x_hat)||_2
    grad_norm = gradients_flat.norm(2, dim=1)  # [B]

    # Penalty: (||∇D(x_hat)||_2 - 1)^2, averaged over the batch
    penalty = ((grad_norm - 1.0) ** 2).mean()

    # ── 6. Scale by lambda_gp ──────────────────────────────────────────
    gp_loss = lambda_gp * penalty

    return gp_loss


# ──────────────────────────────────────────────────────────────────────────────
# Composite Adversarial Losses
# ──────────────────────────────────────────────────────────────────────────────


def compute_adversarial_losses(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    real_scores_local: torch.Tensor,
    real_scores_global: torch.Tensor,
    fake_scores_local: torch.Tensor,
    fake_scores_global: torch.Tensor,
    lambda_gp: float = 10.0,
    adv_weight: float = 1.0,
    w_global: float = 0.5,
) -> Dict[str, torch.Tensor]:
    r"""Compute all adversarial losses in a single call.

    This function composes :func:`wgan_discriminator_loss`,
    :func:`wgan_generator_loss`, and :func:`gradient_penalty` into a single
    call, returning a dictionary with all loss components for both the
    discriminator and generator.

    The returned dictionary contains:

    - ``"d_loss"``: total discriminator loss = ``L_D + L_gp``, where
      ``L_D`` is the WGAN discriminator loss and ``L_gp`` is the gradient
      penalty.  This is the loss to minimise for the discriminator.
    - ``"g_loss"``: total generator loss = ``adv_weight * L_G``, where
      ``L_G`` is the WGAN generator loss.  This is the loss to minimise
      for the generator.
    - ``"gp_loss"``: the weighted gradient penalty ``lambda_gp * penalty``
      (same as the return value of :func:`gradient_penalty`).
    - ``"gp_value"``: the **unweighted** gradient penalty ``penalty``
      (before multiplying by ``lambda_gp``), useful for logging and
      monitoring the actual gradient norm constraint.

    Parameters
    ----------
    discriminator : nn.Module
        The ILGAN ``ImageDiscriminator`` module.
    real_images : torch.Tensor
        Batch of real images, shape ``[B, 3, H, W]``.
    fake_images : torch.Tensor
        Batch of fake (generated) images, shape ``[B, 3, H, W]``.
    real_scores_local : torch.Tensor
        Local scores for real images from the discriminator,
        shape ``[B, 1, H, W]``.
    real_scores_global : torch.Tensor
        Global scores for real images from the discriminator,
        shape ``[B, 1]``.
    fake_scores_local : torch.Tensor
        Local scores for fake images from the discriminator,
        shape ``[B, 1, H, W]``.
    fake_scores_global : torch.Tensor
        Global scores for fake images from the discriminator,
        shape ``[B, 1]``.
    lambda_gp : float, optional
        Gradient penalty coefficient.  (default: ``10.0``)
    adv_weight : float, optional
        Weight applied to the generator adversarial loss.  This can be
        used to balance the adversarial loss with other losses (e.g.
        bounding box losses) in the full generator objective.
        (default: ``1.0``)
    w_global : float, optional
        Weight for the global score component in both discriminator and
        generator losses.  (default: ``0.5``)

    Returns
    -------
    dict of str -> torch.Tensor
        A dictionary with the following keys:

        - ``"d_loss"``: total discriminator loss (scalar).
        - ``"g_loss"``: total generator adversarial loss (scalar).
        - ``"gp_loss"``: weighted gradient penalty (scalar).
        - ``"gp_value"``: unweighted gradient penalty for logging (scalar).

    Example
    -------
    >>> disc = ImageDiscriminator(disc_base_channels=64, image_size=128)
    >>> real = torch.randn(4, 3, 128, 128)
    >>> fake = torch.randn(4, 3, 128, 128)
    >>> real_local, real_global = disc(real)
    >>> fake_local, fake_global = disc(fake)
    >>> losses = compute_adversarial_losses(
    ...     disc, real, fake, real_local, real_global, fake_local, fake_global
    ... )
    >>> losses["d_loss"].shape
    torch.Size([])
    >>> losses["g_loss"].shape
    torch.Size([])
    >>> list(losses.keys())
    ['d_loss', 'g_loss', 'gp_loss', 'gp_value']
    """
    # ── 1. WGAN discriminator loss ──────────────────────────────────────
    d_loss_adv = wgan_discriminator_loss(
        real_scores_local=real_scores_local,
        real_scores_global=real_scores_global,
        fake_scores_local=fake_scores_local,
        fake_scores_global=fake_scores_global,
        w_global=w_global,
    )

    # ── 2. Gradient penalty ─────────────────────────────────────────────
    gp_loss = gradient_penalty(
        discriminator=discriminator,
        real_images=real_images,
        fake_images=fake_images,
        lambda_gp=lambda_gp,
    )

    # ── 3. Unweighted gradient penalty for logging ─────────────────────
    # Compute the unweighted penalty by calling gradient_penalty with
    # lambda_gp=1.0, but that would be wasteful.  Instead, we compute it
    # from the weighted version: gp_value = gp_loss / lambda_gp.
    # However, if lambda_gp is 0, we set gp_value to 0.
    if lambda_gp > 0.0:
        gp_value = gp_loss / lambda_gp
    else:
        gp_value = torch.tensor(0.0, device=gp_loss.device, dtype=gp_loss.dtype)

    # ── 4. WGAN generator loss ──────────────────────────────────────────
    g_loss_adv = wgan_generator_loss(
        fake_scores_local=fake_scores_local,
        fake_scores_global=fake_scores_global,
        w_global=w_global,
    )

    # ── 5. Compose total losses ─────────────────────────────────────────
    # Discriminator total loss = adversarial loss + gradient penalty
    d_loss = d_loss_adv + gp_loss

    # Generator total adversarial loss = weighted generator loss
    g_loss = adv_weight * g_loss_adv

    return {
        "d_loss": d_loss,
        "g_loss": g_loss,
        "gp_loss": gp_loss,
        "gp_value": gp_value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "wgan_discriminator_loss",
    "wgan_generator_loss",
    "gradient_penalty",
    "compute_adversarial_losses",
]
