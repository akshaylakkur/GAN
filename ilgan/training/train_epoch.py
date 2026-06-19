"""
Single training epoch for the ILGAN dual-output GAN.

This module implements the core training loop for one epoch of ILGAN
training.  It orchestrates the generator, discriminator, encoders, loss
aggregator, metrics tracker, AMP scaler, and gradient management into a
coherent training step that respects the WGAN-GP ``n_critic`` ratio,
gradient accumulation, and the novel "representation anchoring" mechanism.

Mathematical overview
---------------------
Each training step proceeds as follows:

1. **Sample latents**: :math:`z \\sim \\mathcal{N}(0, I)^{B \\times D}`

2. **Generator forward**: :math:`(I_{fake}, B_{pred}, C_{pred}, \\text{conf}) = G(z)`

3. **Discriminator forward** (real and fake):
   :math:`(s_{local}^{real}, s_{global}^{real}) = D(I_{real})`
   :math:`(s_{local}^{fake}, s_{global}^{fake}) = D(I_{fake})`

4. **Discriminator loss** (every step):
   :math:`\\mathcal{L}_D = \\mathcal{L}_{WGAN\\_D} + \\lambda_{gp} \\cdot \\mathcal{L}_{GP}`

5. **Generator loss** (every ``n_critic`` steps):
   :math:`\\mathcal{L}_G = \\mathcal{L}_{adv} + \\mathcal{L}_{box} + \\mathcal{L}_{cls} + \\mathcal{L}_{conf} + \\mathcal{L}_{collapse} + \\mathcal{L}_{consistency}`

6. **Representation anchoring** (every K steps):
   :math:`\\mathcal{L}_{anchor} = \\beta \\cdot \\left( \\|\\mu_z - \\mu_{prior}\\|_2^2 + \\|\\sigma_z - \\sigma_{prior}\\|_2^2 \\right)`

   This regularisation penalises the generator when the empirical latent
   statistics drift too far from the prior :math:`\\mathcal{N}(0, I)`,
   preventing representation collapse.

7. **Gradient penalty** (every step, attached to discriminator loss):
   :math:`\\mathcal{L}_{GP} = \\lambda \\cdot \\mathbb{E}[ (\\|\\nabla D(\\hat{x})\\|_2 - 1)^2 ]`

The ``n_critic`` ratio (default 5) ensures the discriminator is updated
multiple times per generator update, which is critical for WGAN-GP stability.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ilgan.data.structures import Batch
from ilgan.losses import LossAggregator
from ilgan.metrics.joint_metrics import MetricsTracker
from ilgan.training.gradient_utils import (
    clip_gradients,
    detect_nan_inf_gradients,
    zero_gradients,
)
from ilgan.training.mixed_precision import AMPScaler
from ilgan.utils.config import Config
from ilgan.utils.logger import Logger
from ilgan.utils.device import get_amp_device_type

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon for numerical stability."""

_DEFAULT_ANCHOR_INTERVAL: int = 100
"""Default number of steps between representation anchoring updates."""

_DEFAULT_ANCHOR_WEIGHT: float = 0.01
"""Default weight for the representation anchoring regularisation."""

_DEFAULT_ANCHOR_MOMENTUM: float = 0.99
"""Momentum for updating running latent statistics."""


# ──────────────────────────────────────────────────────────────────────────────
# Representation Anchoring
# ──────────────────────────────────────────────────────────────────────────────


class RepresentationAnchor:
    r"""Running statistics tracker for representation anchoring regularisation.

    This class maintains exponentially-weighted running estimates of the
    mean and variance of the latent vectors seen during training.  Every
    ``anchor_interval`` steps, it computes a regularisation loss that
    penalises the generator for deviating from the prior distribution
    :math:`\\mathcal{N}(0, I)`.

    The running statistics are updated with momentum :math:`\\alpha`:

    .. math::

        \\mu_{t+1} &= \\alpha \\cdot \\mu_t + (1 - \\alpha) \\cdot \\mu_{\\text{batch}} \\
        \\sigma^2_{t+1} &= \\alpha \\cdot \\sigma^2_t + (1 - \\alpha) \\cdot \\sigma^2_{\\text{batch}}

    The anchoring loss at step :math:`t` is:

    .. math::

        \\mathcal{L}_{\\text{anchor}} = \\beta \\cdot
            \\left( \\|\\mu_t - 0\\|_2^2 + \\|\\sigma_t - 1\\|_2^2 \\right)

    where :math:`\\beta` is ``anchor_weight``.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    anchor_interval : int, optional
        Number of steps between anchoring loss computations.
        (default: 100)
    anchor_weight : float, optional
        Weight :math:`\\beta` for the anchoring regularisation.
        (default: 0.01)
    momentum : float, optional
        Momentum :math:`\\alpha` for updating running statistics.
        Must be in ``(0, 1)``.  (default: 0.99)
    device : torch.device, optional
        Device for the running statistics tensors.  If ``None``, uses
        the device of the first batch of latents seen.

    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """

    def __init__(
        self,
        latent_dim: int,
        anchor_interval: int = _DEFAULT_ANCHOR_INTERVAL,
        anchor_weight: float = _DEFAULT_ANCHOR_WEIGHT,
        momentum: float = _DEFAULT_ANCHOR_MOMENTUM,
        device: Optional[torch.device] = None,
    ) -> None:
        if latent_dim < 1:
            raise ValueError(
                f"latent_dim must be positive, got {latent_dim}."
            )
        if anchor_interval < 1:
            raise ValueError(
                f"anchor_interval must be positive, got {anchor_interval}."
            )
        if anchor_weight < 0.0:
            raise ValueError(
                f"anchor_weight must be non-negative, got {anchor_weight}."
            )
        if not (0.0 < momentum < 1.0):
            raise ValueError(
                f"momentum must be in (0, 1), got {momentum}."
            )

        self._latent_dim = latent_dim
        self._anchor_interval = anchor_interval
        self._anchor_weight = anchor_weight
        self._momentum = momentum

        # Running statistics — initialised to the prior N(0, I)
        if device is not None:
            self._running_mean = torch.zeros(latent_dim, device=device)
            self._running_var = torch.ones(latent_dim, device=device)
        else:
            self._running_mean = torch.zeros(latent_dim)
            self._running_var = torch.ones(latent_dim)

        self._steps_since_anchor: int = 0
        self._initialised: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, z_batch: torch.Tensor) -> None:
        """Update running statistics with a batch of latent vectors.

        Parameters
        ----------
        z_batch : torch.Tensor
            Batch of latent vectors, shape ``[B, D]``.
        """
        if z_batch.numel() == 0:
            return

        # Move running stats to the correct device on first call
        if not self._initialised:
            self._running_mean = self._running_mean.to(z_batch.device)
            self._running_var = self._running_var.to(z_batch.device)
            self._initialised = True

        # Compute batch statistics
        batch_mean = z_batch.mean(dim=0)  # [D]
        batch_var = z_batch.var(dim=0, unbiased=False)  # [D]

        # Update running statistics with momentum
        alpha = self._momentum
        self._running_mean = alpha * self._running_mean + (1.0 - alpha) * batch_mean
        self._running_var = alpha * self._running_var + (1.0 - alpha) * batch_var

        self._steps_since_anchor += 1

    def compute_loss(self) -> Optional[torch.Tensor]:
        r"""Compute the representation anchoring loss if it is time to do so.

        The loss is computed every ``anchor_interval`` steps.  At other
        steps, this method returns ``None``.

        Returns
        -------
        torch.Tensor or None
            A scalar tensor containing the anchoring loss, or ``None`` if
            it is not time to compute the loss.

        Notes
        -----
        - The loss is detached from the running statistics (no gradients
          flow through the running mean/var).  Gradients flow through the
          latent vectors via the generator's computation graph.
        - If the running statistics have not been initialised (no batches
          seen yet), this method returns ``None``.
        """
        if not self._initialised:
            return None

        if self._steps_since_anchor < self._anchor_interval:
            return None

        # Reset the counter
        self._steps_since_anchor = 0

        # Compute the anchoring loss
        # Penalise deviation from prior N(0, I)
        mean_dev = (self._running_mean - 0.0).pow(2).sum()  # ||mu - 0||^2
        var_dev = (self._running_var - 1.0).pow(2).sum()    # ||sigma^2 - 1||^2

        anchor_loss = self._anchor_weight * (mean_dev + var_dev)

        return anchor_loss

    @property
    def running_mean(self) -> torch.Tensor:
        """Current running estimate of the latent mean."""
        return self._running_mean.clone()

    @property
    def running_var(self) -> torch.Tensor:
        """Current running estimate of the latent variance."""
        return self._running_var.clone()

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the anchor for checkpointing."""
        return {
            "running_mean": self._running_mean.cpu().clone(),
            "running_var": self._running_var.cpu().clone(),
            "steps_since_anchor": self._steps_since_anchor,
            "initialised": self._initialised,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the anchor state from a checkpoint."""
        self._running_mean = state_dict["running_mean"].to(
            self._running_mean.device
        )
        self._running_var = state_dict["running_var"].to(
            self._running_var.device
        )
        self._steps_since_anchor = state_dict["steps_since_anchor"]
        self._initialised = state_dict["initialised"]

    def __repr__(self) -> str:
        return (
            f"RepresentationAnchor(\n"
            f"  latent_dim={self._latent_dim},\n"
            f"  anchor_interval={self._anchor_interval},\n"
            f"  anchor_weight={self._anchor_weight},\n"
            f"  momentum={self._momentum},\n"
            f"  initialised={self._initialised},\n"
            f"  steps_since_anchor={self._steps_since_anchor},\n"
            f"  mean_norm={self._running_mean.norm().item():.4f},\n"
            f"  var_mean={self._running_var.mean().item():.4f},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Training Epoch
# ──────────────────────────────────────────────────────────────────────────────


def train_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    image_encoder: nn.Module,
    box_encoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    loss_aggregator: LossAggregator,
    metrics_tracker: MetricsTracker,
    epoch: int,
    global_step: int,
    config: Config,
    logger: Logger,
    amp_scaler: AMPScaler,
    grad_clip_norm: float = 1.0,
) -> Tuple[int, Dict[str, float]]:
    r"""Run one full training epoch for the ILGAN dual-output GAN.

    This function implements the core training loop with:

    - **WGAN-GP adversarial training** with the ``n_critic`` ratio.
    - **Gradient accumulation** for effective batch size scaling.
    - **Mixed precision (AMP)** support via ``amp_scaler``.
    - **Gradient clipping** to prevent exploding gradients.
    - **Representation anchoring** to prevent latent drift.
    - **NaN/Inf gradient detection** for early divergence warning.
    - **Running loss tracking** for console logging and metrics.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator (``ILGANGenerator``).  Its ``forward`` must
        accept a ``[B, latent_dim]`` tensor and return a dict with keys
        ``"image"``, ``"boxes"``, ``"class_logits"``, ``"confidences"``,
        and ``"aux"`` (containing ``"attention_maps"`` and
        ``"skip_features"``).
    discriminator : nn.Module
        The ILGAN discriminator (``ImageDiscriminator``).  Its ``forward``
        must accept a ``[B, 3, H, W]`` tensor and return a tuple
        ``(local_scores, global_score)``.
    image_encoder : nn.Module
        The ``ImageFeatureEncoder`` for cross-modal consistency.  Maps
        images to a shared feature space.
    box_encoder : nn.Module
        The ``BoxFeatureEncoder`` for cross-modal consistency.  Maps
        bounding boxes to a shared feature space.
    train_loader : torch.utils.data.DataLoader
        DataLoader yielding :class:`~ilgan.data.structures.Batch` objects.
    g_optimizer : torch.optim.Optimizer
        Optimizer for the generator (and encoder) parameters.
    d_optimizer : torch.optim.Optimizer
        Optimizer for the discriminator parameters.
    loss_aggregator : LossAggregator
        The central loss aggregator.  Provides ``discriminator_loss()``
        and ``generator_loss()`` convenience methods.
    metrics_tracker : MetricsTracker
        Stateful metrics accumulator.  Losses are pushed via
        ``update_loss_metrics()``.
    epoch : int
        Current epoch number (0-indexed).  Used for logging.
    global_step : int
        Global training step counter.  Incremented after each batch.
    config : Config
        The ILGAN configuration object.  The following keys are read:

        - ``config.training.n_critic`` (int): discriminator updates per
          generator update.
        - ``config.training.gradient_accumulation_steps`` (int): number
          of forward passes to accumulate gradients over.
        - ``config.training.clip_grad_norm`` (float): max gradient norm
          for clipping.
        - ``config.model.latent_dim`` (int): dimensionality of the latent
          space.
        - ``config.logging.log_interval`` (int): log losses every N steps.
    logger : Logger
        The ILGAN logger instance for console output.
    amp_scaler : AMPScaler
        The AMP scaler for mixed precision training.  Provides
        ``scale_loss()``, ``unscale_()``, and ``step()``.
    grad_clip_norm : float, optional
        Maximum gradient norm for clipping.  Overrides
        ``config.training.clip_grad_norm`` if provided.  (default: 1.0)

    Returns
    -------
    global_step : int
        Updated global step counter after processing all batches.
    epoch_losses : dict of str -> float
        Dictionary of epoch-averaged losses.  Contains all individual loss
        terms averaged over the epoch, plus ``"total_g_loss"`` and
        ``"total_d_loss"``.

    Raises
    ------
    RuntimeError
        If NaN/Inf gradients are detected and cannot be recovered from.
    ValueError
        If the config is missing required keys.

    Notes
    -----
    **n_critic ratio**

    The discriminator is updated every step, while the generator is updated
    only every ``n_critic`` steps.  This is the standard WGAN-GP training
    procedure (Gulrajani et al., 2017).  The ratio ensures the discriminator
    maintains a good estimate of the Wasserstein distance before the
    generator tries to minimise it.

    **Gradient accumulation**

    When ``gradient_accumulation_steps > 1``, gradients are accumulated
    over multiple forward passes before each optimizer step.  The loss is
    divided by the number of accumulation steps to keep the effective
    learning rate constant.  This allows training with larger effective
    batch sizes than GPU memory would otherwise permit.

    **Representation anchoring**

    Every ``anchor_interval`` steps (default 100), the function computes
    the running mean and variance of the latent vectors and adds a small
    regularisation to the generator loss that penalises deviation from the
    prior :math:`\\mathcal{N}(0, I)`.  This prevents the generator from
    drifting too far from the prior distribution, which is a known cause
    of mode collapse in GANs.

    **Gradient penalty**

    The gradient penalty is computed every step and attached to the
    discriminator loss.  It enforces the 1-Lipschitz constraint on the
    discriminator, which is the core of WGAN-GP training.
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Extract config values
    # ──────────────────────────────────────────────────────────────────────────
    try:
        n_critic: int = int(config.training.n_critic)
        gradient_accumulation_steps: int = int(
            config.training.gradient_accumulation_steps
        )
        latent_dim: int = int(config.model.latent_dim)
        log_interval: int = int(config.logging.log_interval)
        clip_norm: float = float(
            grad_clip_norm
            if grad_clip_norm is not None
            else config.training.clip_grad_norm
        )
        # Optional: representation anchoring config
        anchor_interval: int = int(
            getattr(config.training, "anchor_interval", _DEFAULT_ANCHOR_INTERVAL)
        )
        anchor_weight: float = float(
            getattr(config.training, "anchor_weight", _DEFAULT_ANCHOR_WEIGHT)
        )
    except (AttributeError, KeyError, TypeError) as e:
        raise ValueError(
            f"Config is missing a required key for training: {e}. "
            f"Please ensure your config has all required fields."
        ) from e

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Set all models to training mode
    # ──────────────────────────────────────────────────────────────────────────
    generator.train()
    discriminator.train()
    image_encoder.train()
    box_encoder.train()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Initialise helpers
    # ──────────────────────────────────────────────────────────────────────────
    device = next(generator.parameters()).device

    # Representation anchor
    anchor = RepresentationAnchor(
        latent_dim=latent_dim,
        anchor_interval=anchor_interval,
        anchor_weight=anchor_weight,
        device=device,
    )

    # Running loss accumulators for the epoch
    running_losses: Dict[str, float] = {}
    num_batches: int = 0

    # Gradient accumulation state
    d_accumulation_counter: int = 0
    g_accumulation_counter: int = 0
    d_optimizer_should_step: bool = False
    g_optimizer_should_step: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Iterate over batches
    # ──────────────────────────────────────────────────────────────────────────
    for batch_idx, batch in enumerate(train_loader):
        # ── 4a. Move batch to GPU ──────────────────────────────────────────────
        if isinstance(batch, Batch):
            batch = batch.to(device)
            batch_dict = {
                "images": batch.images,
                "boxes": batch.boxes,
                "labels": batch.labels,
                "valid_mask": batch.valid_mask,
            }
        elif isinstance(batch, dict):
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            # Assume (images, boxes, labels, valid_mask) tuple
            batch_dict = {
                "images": batch[0].to(device),
                "boxes": batch[1].to(device),
                "labels": batch[2].to(device),
                "valid_mask": batch[3].to(device),
            }
        else:
            raise TypeError(
                f"Unsupported batch type: {type(batch)}. "
                f"Expected Batch, dict, or tuple."
            )

        B = batch_dict["images"].shape[0]

        # ── 4b. Sample latent vectors ─────────────────────────────────────────
        z = torch.randn(B, latent_dim, device=device)

        # ── 4c. Generator forward pass ────────────────────────────────────────
        # The generator runs under autocast if AMP is enabled.
        # We use torch.amp.autocast() context if amp_scaler is enabled.
        with torch.amp.autocast(get_amp_device_type(), enabled=amp_scaler.is_enabled):
            gen_outputs = generator(z)

        # ── 4d. Discriminator forward + loss (every step) ─────────────────────
        # We need to run the discriminator on real and fake images.
        # The discriminator loss is computed via the loss_aggregator.
        with torch.amp.autocast(get_amp_device_type(), enabled=amp_scaler.is_enabled):
            d_loss = loss_aggregator.discriminator_loss(
                generator_outputs=gen_outputs,
                batch=batch_dict,
                discriminator=discriminator,
            )

        # Scale loss for gradient accumulation
        d_loss_scaled = d_loss / gradient_accumulation_steps

        # Backpropagate discriminator loss (with AMP scaling)
        amp_scaler.scale_loss(d_loss_scaled, d_optimizer).backward()

        d_accumulation_counter += 1
        d_optimizer_should_step = (
            d_accumulation_counter >= gradient_accumulation_steps
        )

        # ── 4e. Generator forward + loss (every n_critic steps) ───────────────
        # The generator loss is computed only every n_critic steps.
        # We check if this is a generator update step.
        should_update_generator = (
            global_step % n_critic == 0
        )

        # Initialise anchor_loss here so it always exists (may be None
        # on non-generator-update steps, preventing UnboundLocalError).
        anchor_loss: Optional[torch.Tensor] = None

        if should_update_generator:
            # Re-run generator forward for the generator loss computation.
            # We need to re-run because the discriminator backward may have
            # modified the computation graph (gradient penalty).
            # However, we can reuse gen_outputs if we haven't modified the graph.
            # The generator outputs are detached from the discriminator graph,
            # so we can safely reuse them.
            with torch.amp.autocast(get_amp_device_type(), enabled=amp_scaler.is_enabled):
                g_loss = loss_aggregator.generator_loss(
                    generator_outputs=gen_outputs,
                    batch=batch_dict,
                    discriminator=discriminator,
                    image_encoder=image_encoder,
                    box_encoder=box_encoder,
                    z_batch=z,
                )

                # ── Representation anchoring regularisation ───────────────────
                # Update running latent statistics
                anchor.update(z)

                # Compute anchoring loss if it is time
                anchor_loss = anchor.compute_loss()
                if anchor_loss is not None:
                    g_loss = g_loss + anchor_loss

            # Scale loss for gradient accumulation
            g_loss_scaled = g_loss / gradient_accumulation_steps

            # Backpropagate generator loss (with AMP scaling)
            amp_scaler.scale_loss(g_loss_scaled, g_optimizer).backward()

            g_accumulation_counter += 1
            g_optimizer_should_step = (
                g_accumulation_counter >= gradient_accumulation_steps
            )

        # ── 4f. Optimizer steps (with gradient accumulation) ──────────────────
        # Discriminator step
        if d_optimizer_should_step:
            # Unscale gradients before clipping
            amp_scaler.unscale_(d_optimizer)

            # Detect NaN/Inf gradients
            bad_params = detect_nan_inf_gradients(
                discriminator, logger=logger, step=global_step
            )
            if bad_params:
                logger.warning(
                    f"[Step {global_step}] NaN/Inf gradients detected in "
                    f"discriminator parameters: {bad_params[:5]}... "
                    f"Skipping discriminator step."
                )
                # Zero gradients to recover
                zero_gradients(d_optimizer)
            else:
                # Clip gradients
                d_grad_norm = clip_gradients(
                    discriminator, max_norm=clip_norm
                )

                # Step discriminator optimizer
                step_success = amp_scaler.step(d_optimizer)

                if not step_success:
                    logger.warning(
                        f"[Step {global_step}] Discriminator optimizer step "
                        f"skipped due to inf/nan gradients (AMP)."
                    )

            # Zero gradients for next accumulation cycle
            zero_gradients(d_optimizer)
            d_accumulation_counter = 0
            d_optimizer_should_step = False

        # Generator step
        if g_optimizer_should_step:
            # Unscale gradients before clipping
            amp_scaler.unscale_(g_optimizer)

            # Detect NaN/Inf gradients
            bad_params = detect_nan_inf_gradients(
                generator, logger=logger, step=global_step
            )
            if bad_params:
                logger.warning(
                    f"[Step {global_step}] NaN/Inf gradients detected in "
                    f"generator parameters: {bad_params[:5]}... "
                    f"Skipping generator step."
                )
                zero_gradients(g_optimizer)
            else:
                # Clip gradients
                g_grad_norm = clip_gradients(
                    generator, max_norm=clip_norm
                )

                # Step generator optimizer
                step_success = amp_scaler.step(g_optimizer)

                if not step_success:
                    logger.warning(
                        f"[Step {global_step}] Generator optimizer step "
                        f"skipped due to inf/nan gradients (AMP)."
                    )

            # Zero gradients for next accumulation cycle
            zero_gradients(g_optimizer)
            g_accumulation_counter = 0
            g_optimizer_should_step = False

        # ── 4g. Track running losses ─────────────────────────────────────────
        # Collect loss values from the already-computed d_loss and g_loss.
        # We avoid calling loss_aggregator.__call__() a second time because
        # it would re-run the discriminator and gradient penalty, which
        # would fail since the discriminator backward has already freed
        # the computation graph.
        step_losses: Dict[str, float] = {
            "total_d_loss": d_loss.detach().item(),
        }
        if should_update_generator:
            step_losses["total_g_loss"] = g_loss.detach().item()
        else:
            step_losses["total_g_loss"] = 0.0

        # Add the anchoring loss if it was computed
        if anchor_loss is not None:
            step_losses["anchor_loss"] = anchor_loss.detach().item()

        # Update running averages
        for key, value in step_losses.items():
            if key in running_losses:
                # Running average: update with exponential smoothing
                # Use a simple cumulative average for the epoch
                running_losses[key] = running_losses[key] + (
                    value - running_losses[key]
                ) / (num_batches + 1)
            else:
                running_losses[key] = value

        # Push losses to metrics tracker
        metrics_tracker.update_loss_metrics(step_losses)

        num_batches += 1

        # ── 4h. Logging ─────────────────────────────────────────────────────
        if global_step % log_interval == 0:
            # Build a concise log string
            log_parts: List[str] = [
                f"Epoch {epoch:>3d} | Step {global_step:>6d} | "
                f"Batch {batch_idx:>4d}/{len(train_loader):<4d}"
            ]

            # Add key losses
            if "total_g_loss" in step_losses:
                log_parts.append(
                    f"G: {step_losses['total_g_loss']:.4f}"
                )
            if "total_d_loss" in step_losses:
                log_parts.append(
                    f"D: {step_losses['total_d_loss']:.4f}"
                )
            if "box_loss" in step_losses:
                log_parts.append(
                    f"Box: {step_losses['box_loss']:.4f}"
                )
            if "consistency_loss" in step_losses:
                log_parts.append(
                    f"Cons: {step_losses['consistency_loss']:.4f}"
                )
            if "collapse_loss" in step_losses:
                log_parts.append(
                    f"Coll: {step_losses['collapse_loss']:.4f}"
                )
            if "anchor_loss" in step_losses:
                log_parts.append(
                    f"Anch: {step_losses['anchor_loss']:.6f}"
                )

            # Add gradient norms if available
            if should_update_generator and g_optimizer_should_step:
                log_parts.append(f"| G grad: {g_grad_norm:.4f}")
            if d_optimizer_should_step:
                log_parts.append(f"D grad: {d_grad_norm:.4f}")

            # Add AMP scale
            if amp_scaler.is_enabled:
                log_parts.append(f"| AMP scale: {amp_scaler.scale:.1f}")

            logger.info("  ".join(log_parts))

        # ── 4i. Increment global step ────────────────────────────────────────
        global_step += 1

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Finalise and return
    # ──────────────────────────────────────────────────────────────────────────
    # Log epoch summary
    logger.info(
        f"Epoch {epoch:>3d} complete — {num_batches} batches processed. "
        f"Global step: {global_step:>6d}."
    )

    # Log epoch-averaged losses
    if running_losses:
        loss_summary = " | ".join(
            f"{k}: {v:.6f}" for k, v in sorted(running_losses.items())
        )
        logger.info(f"Epoch {epoch:>3d} averaged losses: {loss_summary}")

    return global_step, running_losses


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "RepresentationAnchor",
    "train_epoch",
]
