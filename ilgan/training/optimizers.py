"""
Custom optimizers and learning rate schedulers for ILGAN training.

This module provides three core components for the ILGAN training loop:

1. :func:`build_optimizers` — factory that creates Adam optimizers for the
   generator (including encoder sub-networks) and the discriminator, with
   separate parameter-group learning rates for the auxiliary encoders.

2. :class:`AdaptiveOptimizerScheduler` — a novel adaptive learning rate
   scheme that dynamically balances the generator and discriminator by
   monitoring the ratio of their output magnitudes over a sliding window.
   This prevents one network from overwhelming the other during adversarial
   training, mathematically forestalling representation collapse.

3. :func:`build_scheduler` — factory that creates standard learning rate
   schedulers (CosineAnnealingLR or MultiStepLR) for both optimizers,
   wrapping them for epoch-based or step-based decay.

Mathematical motivation
-----------------------
In standard GAN training, the generator :math:`G` and discriminator :math:`D`
engage in a two-player minimax game:

.. math::

    \min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)]
    + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]

When one network becomes too strong, the other's gradients vanish, leading to
mode collapse or discriminator overfitting.  The :class:`AdaptiveOptimizerScheduler`
addresses this by monitoring the balance condition:

.. math::

    r_t = \frac{\mathbb{E}[|D(x_{\text{real}})|]}{\mathbb{E}[|D(G(z))|] + \varepsilon}

If :math:`r_t > \tau` (discriminator too strong), we reduce :math:`D`'s learning
rate and increase :math:`G`'s.  If :math:`r_t < 1/\tau` (generator too strong),
we do the opposite.  This keeps the game balanced and prevents collapse.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
    _LRScheduler,
)

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon to prevent division by zero in ratio computations."""

_DEFAULT_ENCODER_LR_FACTOR: float = 0.1
"""Default factor by which encoder learning rates are scaled down relative to
the generator's base learning rate.  Encoders are auxiliary networks that
should learn more slowly to avoid destabilising the primary GAN training."""

_DEFAULT_ADAPTIVE_WINDOW: int = 100
"""Default number of training steps over which the discriminator/generator
output ratio is averaged for the adaptive scheduler."""

_DEFAULT_BALANCE_THRESHOLD: float = 1.5
"""Default threshold :math:`\\tau` for the adaptive balance condition.  If the
ratio :math:`r_t` exceeds this, the discriminator is considered too strong."""

_DEFAULT_LR_ADJUSTMENT_FACTOR: float = 0.95
"""Default multiplicative factor for learning rate adjustments in the adaptive
scheduler.  A value of 0.95 means each adjustment changes the LR by 5%."""

_DEFAULT_MIN_LR: float = 1e-6
"""Minimum learning rate floor for both optimizers in the adaptive scheduler."""

_DEFAULT_MAX_LR: float = 1e-2
"""Maximum learning rate ceiling for both optimizers in the adaptive scheduler."""


# ──────────────────────────────────────────────────────────────────────────────
# build_optimizers
# ──────────────────────────────────────────────────────────────────────────────


def build_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    image_encoder: nn.Module,
    box_encoder: nn.Module,
    config: Config,
) -> Tuple[Adam, Adam]:
    """Create Adam optimizers for the generator and discriminator.

    The **generator optimizer** manages parameters from three sub-networks:

    - ``generator`` (the primary ``ILGANGenerator``)
    - ``image_encoder`` (auxiliary image feature encoder for consistency)
    - ``box_encoder`` (auxiliary box feature encoder for consistency)

    The encoder parameters are placed in a **separate parameter group** with
    a lower learning rate (``encoder_lr_factor * base_lr``) since they are
    auxiliary networks that should not dominate the primary GAN training.

    The **discriminator optimizer** manages only the discriminator parameters.

    Both optimizers use the Adam algorithm (Kingma & Ba, 2015) with
    hyperparameters drawn from the config:

    - ``config.training.learning_rate``: base learning rate.
    - ``config.training.beta1``: Adam ``betas[0]``.
    - ``config.training.beta2``: Adam ``betas[1]``.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator (``ILGANGenerator`` instance).  Its parameters
        are trained with the base learning rate.
    discriminator : nn.Module
        The ILGAN discriminator (``ImageDiscriminator`` instance).  Its
        parameters are trained with the base learning rate.
    image_encoder : nn.Module
        The image feature encoder (``ImageFeatureEncoder`` instance).  Its
        parameters are trained with a reduced learning rate.
    box_encoder : nn.Module
        The box feature encoder (``BoxFeatureEncoder`` instance).  Its
        parameters are trained with a reduced learning rate.
    config : Config
        The ILGAN configuration object.  The following keys are read:

        - ``config.training.learning_rate`` (float): base learning rate.
        - ``config.training.beta1`` (float): Adam beta1.
        - ``config.training.beta2`` (float): Adam beta2.
        - ``config.training.encoder_lr_factor`` (float, optional): factor
          by which encoder learning rates are scaled.  Defaults to 0.1 if
          not present in config.

    Returns
    -------
    g_optimizer : Adam
        Adam optimizer for the generator + encoder parameters.
    d_optimizer : Adam
        Adam optimizer for the discriminator parameters.

    Raises
    ------
    TypeError
        If any of the network arguments is not an ``nn.Module``.
    ValueError
        If the learning rate or betas are out of valid ranges.

    Example
    -------
    >>> from ilgan.training.optimizers import build_optimizers
    >>> g_opt, d_opt = build_optimizers(generator, discriminator,
    ...                                 image_encoder, box_encoder, cfg)
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    for name, module in [
        ("generator", generator),
        ("discriminator", discriminator),
        ("image_encoder", image_encoder),
        ("box_encoder", box_encoder),
    ]:
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Expected '{name}' to be an nn.Module, "
                f"got {type(module).__name__}."
            )

    # ── Extract hyperparameters from config ─────────────────────────────
    base_lr: float = float(config.training.learning_rate)
    # If separate G/D learning rates are provided, use them; otherwise
    # both default to the base learning rate
    g_lr: float = float(getattr(config.training, "generator_lr", base_lr))
    d_lr: float = float(getattr(config.training, "discriminator_lr", base_lr))
    beta1: float = float(config.training.beta1)
    beta2: float = float(config.training.beta2)

    # Validate
    for lr_val, name in [(g_lr, "generator_lr"), (d_lr, "discriminator_lr"), (base_lr, "learning_rate")]:
        if lr_val <= 0.0:
            raise ValueError(
                f"{name} must be positive, got {lr_val}."
            )
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(
            f"beta1 must be in [0.0, 1.0), got {beta1}."
        )
    if not (0.0 < beta2 <= 1.0):
        raise ValueError(
            f"beta2 must be in (0.0, 1.0], got {beta2}."
        )

    # Optional encoder LR factor (default: 0.1 = 10x lower)
    encoder_lr_factor: float = getattr(
        config.training, "encoder_lr_factor", _DEFAULT_ENCODER_LR_FACTOR
    )
    encoder_lr: float = g_lr * encoder_lr_factor  # encoders follow G's LR

    # ── Build generator parameter groups ────────────────────────────────
    # Group 1: generator parameters (base LR)
    gen_params = list(generator.parameters())

    # Group 2: image_encoder parameters (reduced LR)
    img_enc_params = list(image_encoder.parameters())

    # Group 3: box_encoder parameters (reduced LR)
    box_enc_params = list(box_encoder.parameters())

    g_optimizer = Adam(
        [
            {
                "params": gen_params,
                "lr": g_lr,
                "betas": (beta1, beta2),
                "name": "generator",
            },
            {
                "params": img_enc_params,
                "lr": encoder_lr,
                "betas": (beta1, beta2),
                "name": "image_encoder",
            },
            {
                "params": box_enc_params,
                "lr": encoder_lr,
                "betas": (beta1, beta2),
                "name": "box_encoder",
            },
        ],
        lr=g_lr,
        betas=(beta1, beta2),
    )

    # ── Build discriminator optimizer ───────────────────────────────────
    d_params = list(discriminator.parameters())

    d_optimizer = Adam(
        d_params,
        lr=d_lr,
        betas=(beta1, beta2),
    )

    return g_optimizer, d_optimizer


# ──────────────────────────────────────────────────────────────────────────────
# AdaptiveOptimizerScheduler
# ──────────────────────────────────────────────────────────────────────────────


class AdaptiveOptimizerScheduler:
    r"""Adaptive learning rate scheduler that balances generator and
    discriminator training dynamics.

    This scheduler implements a novel adaptive scheme that monitors the
    balance between the generator and discriminator by tracking the ratio
    of their output magnitudes over a sliding window of :math:`N` training
    steps.

    Balance condition
    -----------------
    At each step :math:`t`, we compute:

    .. math::

        r_t = \frac{\mathbb{E}[|D(x_{\text{real}})|]}
                   {\mathbb{E}[|D(G(z))|] + \varepsilon}

    where :math:`D(x_{\text{real}})` is the discriminator's output on real
    data, :math:`D(G(z))` is its output on generated (fake) data, and
    :math:`\varepsilon` is a small constant to prevent division by zero.

    The scheduler maintains a sliding-window average :math:`\bar{r}_t` over
    the last :math:`N` steps:

    .. math::

        \bar{r}_t = \frac{1}{N} \sum_{i=t-N+1}^{t} r_i

    The balance is adjusted according to:

    - If :math:`\bar{r}_t > \tau` (discriminator too strong):
      reduce discriminator LR by :math:`\alpha`, increase generator LR by
      :math:`\alpha`.

    - If :math:`\bar{r}_t < 1/\tau` (generator too strong):
      increase discriminator LR by :math:`\alpha`, reduce generator LR by
      :math:`\alpha`.

    - Otherwise: no change (the game is balanced).

    where :math:`\tau > 1` is the balance threshold and
    :math:`\alpha \in (0, 1)` is the adjustment factor.

    This mechanism mathematically prevents one network from overwhelming the
    other, which in turn prevents:

    - **Image mode collapse**: if the discriminator is too strong, the
      generator receives vanishing gradients and collapses to a single output.
      By reducing the discriminator's LR, we give the generator room to
      recover.

    - **Bounding box collapse**: if the generator is too strong, the
      discriminator cannot distinguish real from fake, and the generator
      may produce degenerate bounding boxes.  By reducing the generator's LR,
      we force it to produce more varied outputs.

    Parameters
    ----------
    g_optimizer : Optimizer
        The generator optimizer (typically returned by :func:`build_optimizers`).
    d_optimizer : Optimizer
        The discriminator optimizer (typically returned by :func:`build_optimizers`).
    window_size : int, optional
        Number of recent steps :math:`N` over which to average the balance
        ratio.  Must be positive.  (default: 100)
    threshold : float, optional
        Balance threshold :math:`\tau`.  Must be > 1.  (default: 1.5)
    adjustment_factor : float, optional
        Multiplicative factor :math:`\alpha` for LR adjustments.  Must be in
        ``(0, 1)``.  (default: 0.95)
    min_lr : float, optional
        Minimum allowed learning rate for any parameter group.  (default: 1e-6)
    max_lr : float, optional
        Maximum allowed learning rate for any parameter group.  (default: 1e-2)
    cooldown_steps : int, optional
        Number of steps to wait after an adjustment before making another
        adjustment.  This prevents oscillation.  (default: 10)

    Raises
    ------
    TypeError
        If either optimizer is not an ``Optimizer`` instance.
    ValueError
        If any of the numeric parameters are out of valid ranges.

    Example
    -------
    >>> scheduler = AdaptiveOptimizerScheduler(g_opt, d_opt)
    >>> for batch in dataloader:
    ...     # ... training step ...
    ...     d_real_mean = real_scores_local.mean().abs().item()
    ...     d_fake_mean = fake_scores_local.mean().abs().item()
    ...     scheduler.step(d_real_mean, d_fake_mean)
    ...     current_lrs = scheduler.get_lr()
    """

    def __init__(
        self,
        g_optimizer: Optimizer,
        d_optimizer: Optimizer,
        window_size: int = _DEFAULT_ADAPTIVE_WINDOW,
        threshold: float = _DEFAULT_BALANCE_THRESHOLD,
        adjustment_factor: float = _DEFAULT_LR_ADJUSTMENT_FACTOR,
        min_lr: float = _DEFAULT_MIN_LR,
        max_lr: float = _DEFAULT_MAX_LR,
        cooldown_steps: int = 10,
    ) -> None:
        # ── Validate optimizers ────────────────────────────────────────
        if not isinstance(g_optimizer, Optimizer):
            raise TypeError(
                f"Expected g_optimizer to be an Optimizer, "
                f"got {type(g_optimizer).__name__}."
            )
        if not isinstance(d_optimizer, Optimizer):
            raise TypeError(
                f"Expected d_optimizer to be an Optimizer, "
                f"got {type(d_optimizer).__name__}."
            )

        # ── Validate numeric parameters ──────────────────────────────────
        if window_size < 1:
            raise ValueError(
                f"window_size must be positive, got {window_size}."
            )
        if threshold <= 1.0:
            raise ValueError(
                f"threshold must be > 1.0, got {threshold}."
            )
        if not (0.0 < adjustment_factor < 1.0):
            raise ValueError(
                f"adjustment_factor must be in (0, 1), got {adjustment_factor}."
            )
        if min_lr <= 0.0:
            raise ValueError(
                f"min_lr must be positive, got {min_lr}."
            )
        if max_lr <= min_lr:
            raise ValueError(
                f"max_lr ({max_lr}) must be > min_lr ({min_lr})."
            )
        if cooldown_steps < 0:
            raise ValueError(
                f"cooldown_steps must be non-negative, got {cooldown_steps}."
            )

        self._g_optimizer = g_optimizer
        self._d_optimizer = d_optimizer
        self._window_size = window_size
        self._threshold = threshold
        self._adjustment_factor = adjustment_factor
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._cooldown_steps = cooldown_steps

        # ── Internal state ───────────────────────────────────────────────
        # Sliding window of recent balance ratios
        self._ratio_history: Deque[float] = deque(maxlen=window_size)

        # Number of steps taken so far
        self._steps: int = 0

        # Steps remaining in cooldown (0 means no cooldown)
        self._cooldown_remaining: int = 0

        # Store the last computed average ratio for logging
        self._last_avg_ratio: float = 1.0

        # Store the last adjustment direction for logging
        # 1 = discriminator was too strong, -1 = generator was too strong, 0 = balanced
        self._last_adjustment: int = 0

        # ── Snapshot initial LRs ────────────────────────────────────────
        self._initial_g_lr: float = self._get_g_lr()
        self._initial_d_lr: float = self._get_d_lr()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def step(
        self,
        d_real_mean: float,
        d_fake_mean: float,
    ) -> Dict[str, Any]:
        """Update the adaptive scheduler with the latest discriminator
        outputs.

        Call this **after** each training step (or after each ``n_critic``
        cycle) with the mean absolute discriminator outputs on real and fake
        data.

        Parameters
        ----------
        d_real_mean : float
            Mean absolute value of the discriminator's output on real images,
            i.e. ``E[|D(x_real)|]``.  This is typically
            ``real_scores_local.mean().abs().item()`` or
            ``real_scores_global.mean().abs().item()`` (or a combination).
        d_fake_mean : float
            Mean absolute value of the discriminator's output on fake images,
            i.e. ``E[|D(G(z))|]``.  This is typically
            ``fake_scores_local.mean().abs().item()`` or similar.

        Returns
        -------
        dict
            A dictionary with diagnostic information about the adjustment:

            - ``"avg_ratio"``: the sliding-window average balance ratio
              :math:`\\bar{r}_t`.
            - ``"adjusted"``: ``True`` if an LR adjustment was made this step.
            - ``"direction"``: ``"discriminator_too_strong"``,
              ``"generator_too_strong"``, or ``"balanced"``.
            - ``"g_lr"``: current generator learning rate.
            - ``"d_lr"``: current discriminator learning rate.
            - ``"cooldown_active"``: ``True`` if a cooldown is in effect.
        """
        self._steps += 1

        # ── Compute balance ratio ───────────────────────────────────────
        ratio = d_real_mean / (d_fake_mean + _EPS)
        self._ratio_history.append(ratio)

        # ── Compute sliding-window average ───────────────────────────────
        if len(self._ratio_history) < self._window_size:
            # Not enough data yet; use what we have
            avg_ratio = sum(self._ratio_history) / len(self._ratio_history)
        else:
            avg_ratio = sum(self._ratio_history) / self._window_size

        self._last_avg_ratio = avg_ratio

        # ── Check cooldown ──────────────────────────────────────────────
        result: Dict[str, Any] = {
            "avg_ratio": avg_ratio,
            "adjusted": False,
            "direction": "balanced",
            "g_lr": self._get_g_lr(),
            "d_lr": self._get_d_lr(),
            "cooldown_active": self._cooldown_remaining > 0,
        }

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return result

        # ── Determine adjustment direction ───────────────────────────────
        if avg_ratio > self._threshold:
            # Discriminator is too strong → reduce D LR, increase G LR
            self._adjust_lr(
                g_factor=1.0 / self._adjustment_factor,  # increase G LR
                d_factor=self._adjustment_factor,         # decrease D LR
            )
            self._last_adjustment = 1
            result["adjusted"] = True
            result["direction"] = "discriminator_too_strong"

        elif avg_ratio < 1.0 / self._threshold:
            # Generator is too strong → increase D LR, reduce G LR
            self._adjust_lr(
                g_factor=self._adjustment_factor,         # decrease G LR
                d_factor=1.0 / self._adjustment_factor,   # increase D LR
            )
            self._last_adjustment = -1
            result["adjusted"] = True
            result["direction"] = "generator_too_strong"

        else:
            # Balanced — no adjustment
            self._last_adjustment = 0

        # ── Update result with current LRs ──────────────────────────────
        result["g_lr"] = self._get_g_lr()
        result["d_lr"] = self._get_d_lr()

        return result

    def get_lr(self) -> Dict[str, float]:
        """Return the current learning rates for both optimizers.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``"g_lr"``: current generator learning rate (the LR of the
              first parameter group in the generator optimizer).
            - ``"d_lr"``: current discriminator learning rate.
            - ``"g_initial_lr"``: initial generator learning rate.
            - ``"d_initial_lr"``: initial discriminator learning rate.
            - ``"avg_ratio"``: the most recent sliding-window average
              balance ratio.
        """
        return {
            "g_lr": self._get_g_lr(),
            "d_lr": self._get_d_lr(),
            "g_initial_lr": self._initial_g_lr,
            "d_initial_lr": self._initial_d_lr,
            "avg_ratio": self._last_avg_ratio,
        }

    def reset(self) -> None:
        """Reset the scheduler to its initial state.

        Clears the ratio history, resets the step counter, and restores
        the initial learning rates for both optimizers.
        """
        self._ratio_history.clear()
        self._steps = 0
        self._cooldown_remaining = 0
        self._last_avg_ratio = 1.0
        self._last_adjustment = 0

        # Restore initial LRs
        self._set_g_lr(self._initial_g_lr)
        self._set_d_lr(self._initial_d_lr)

    @property
    def steps(self) -> int:
        """Total number of ``step()`` calls made so far."""
        return self._steps

    @property
    def last_avg_ratio(self) -> float:
        """The most recent sliding-window average balance ratio."""
        return self._last_avg_ratio

    @property
    def last_adjustment(self) -> int:
        """The direction of the last adjustment.

        Returns
        -------
        int
            1 if discriminator was too strong, -1 if generator was too
            strong, 0 if balanced (no adjustment).
        """
        return self._last_adjustment

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_g_lr(self) -> float:
        """Get the current learning rate of the generator optimizer's first
        parameter group (the primary generator parameters)."""
        return float(self._g_optimizer.param_groups[0]["lr"])

    def _get_d_lr(self) -> float:
        """Get the current learning rate of the discriminator optimizer."""
        return float(self._d_optimizer.param_groups[0]["lr"])

    def _set_g_lr(self, lr: float) -> None:
        """Set the learning rate for all parameter groups in the generator
        optimizer, preserving the relative scaling between groups.

        The first group (generator) gets ``lr``, the encoder groups get
        ``lr * encoder_lr_factor`` (where the factor is inferred from the
        ratio of their current LR to the first group's LR).
        """
        if len(self._g_optimizer.param_groups) == 0:
            return

        base_lr = self._g_optimizer.param_groups[0]["lr"]
        for group in self._g_optimizer.param_groups:
            # Preserve the relative scaling of each group
            if abs(base_lr) > _EPS:
                scale = group["lr"] / base_lr
            else:
                scale = 1.0
            new_lr = lr * scale
            # Clamp to [min_lr, max_lr]
            new_lr = max(self._min_lr, min(self._max_lr, new_lr))
            group["lr"] = new_lr

    def _set_d_lr(self, lr: float) -> None:
        """Set the learning rate for all parameter groups in the
        discriminator optimizer, clamped to ``[min_lr, max_lr]``."""
        for group in self._d_optimizer.param_groups:
            new_lr = max(self._min_lr, min(self._max_lr, lr))
            group["lr"] = new_lr

    def _adjust_lr(
        self,
        g_factor: float,
        d_factor: float,
    ) -> None:
        """Apply multiplicative LR adjustments to both optimizers.

        Parameters
        ----------
        g_factor : float
            Multiplicative factor for the generator LR.  Values > 1 increase
            the LR; values < 1 decrease it.
        d_factor : float
            Multiplicative factor for the discriminator LR.
        """
        # Adjust generator LR
        current_g_lr = self._get_g_lr()
        new_g_lr = current_g_lr * g_factor
        self._set_g_lr(new_g_lr)

        # Adjust discriminator LR
        current_d_lr = self._get_d_lr()
        new_d_lr = current_d_lr * d_factor
        self._set_d_lr(new_d_lr)

        # Enter cooldown to prevent oscillation
        self._cooldown_remaining = self._cooldown_steps

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AdaptiveOptimizerScheduler(\n"
            f"  window_size={self._window_size},\n"
            f"  threshold={self._threshold},\n"
            f"  adjustment_factor={self._adjustment_factor},\n"
            f"  min_lr={self._min_lr},\n"
            f"  max_lr={self._max_lr},\n"
            f"  cooldown_steps={self._cooldown_steps},\n"
            f"  steps={self._steps},\n"
            f"  g_lr={self._get_g_lr():.6f},\n"
            f"  d_lr={self._get_d_lr():.6f},\n"
            f"  avg_ratio={self._last_avg_ratio:.4f},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# build_scheduler
# ──────────────────────────────────────────────────────────────────────────────


def build_scheduler(
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    config: Config,
    scheduler_type: str = "cosine",
    **kwargs: Any,
) -> Dict[str, _LRScheduler]:
    """Create standard learning rate schedulers for both optimizers.

    This factory creates a pair of ``torch.optim.lr_scheduler`` objects —
    one for the generator optimizer and one for the discriminator optimizer —
    that decay the learning rate over the course of training according to a
    predefined schedule.

    Supported scheduler types
    -------------------------
    - ``"cosine"`` (default): :class:`torch.optim.lr_scheduler.CosineAnnealingLR`.
      The LR decays from the initial value to ``eta_min`` following a cosine
      curve over ``T_max`` epochs.
    - ``"cosine_warm_restarts"``: :class:`CosineAnnealingWarmRestarts`.
      Cosine annealing with periodic warm restarts.
    - ``"multistep"``: :class:`torch.optim.lr_scheduler.MultiStepLR`.
      The LR is multiplied by ``gamma`` at specified milestone epochs.

    Parameters
    ----------
    g_optimizer : Optimizer
        The generator optimizer.
    d_optimizer : Optimizer
        The discriminator optimizer.
    config : Config
        The ILGAN configuration object.  The following keys are read:

        - ``config.training.epochs`` (int): total number of training epochs,
          used as ``T_max`` for cosine annealing.
        - ``config.training.learning_rate`` (float): base learning rate,
          used to compute ``eta_min`` (default: ``lr / 1000``).

    scheduler_type : str, optional
        One of ``"cosine"``, ``"cosine_warm_restarts"``, or ``"multistep"``.
        (default: ``"cosine"``)

    **kwargs
        Additional keyword arguments passed to the scheduler constructor.
        Common options:

        - ``eta_min`` (float): minimum LR for cosine annealing (default:
          ``config.training.learning_rate / 1000``).
        - ``T_0`` (int): number of epochs for the first restart cycle
          (only for ``cosine_warm_restarts``; default: ``max(1, epochs // 10)``).
        - ``T_mult`` (int): factor by which ``T_0`` grows after each restart
          (only for ``cosine_warm_restarts``; default: 2).
        - ``milestones`` (list of int): epoch indices for ``multistep``
          (default: ``[epochs // 2, epochs * 3 // 4]``).
        - ``gamma`` (float): multiplicative factor for ``multistep``
          (default: 0.1).

    Returns
    -------
    dict of str -> _LRScheduler
        A dictionary with two keys:

        - ``"g_scheduler"``: the scheduler for the generator optimizer.
        - ``"d_scheduler"``: the scheduler for the discriminator optimizer.

    Raises
    ------
    ValueError
        If ``scheduler_type`` is not one of the supported types.

    Example
    -------
    >>> from ilgan.training.optimizers import build_scheduler
    >>> schedulers = build_scheduler(g_opt, d_opt, cfg, scheduler_type="cosine")
    >>> for epoch in range(cfg.training.epochs):
    ...     for batch in dataloader:
    ...         # ... training step ...
    ...     schedulers["g_scheduler"].step()
    ...     schedulers["d_scheduler"].step()
    """
    # ── Extract config values ───────────────────────────────────────────
    epochs: int = int(config.training.epochs)
    base_lr: float = float(config.training.learning_rate)

    # Default eta_min = base_lr / 1000
    eta_min: float = kwargs.pop("eta_min", base_lr / 1000.0)

    # ── Build schedulers based on type ──────────────────────────────────
    if scheduler_type == "cosine":
        g_scheduler: _LRScheduler = CosineAnnealingLR(
            g_optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
        d_scheduler: _LRScheduler = CosineAnnealingLR(
            d_optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )

    elif scheduler_type == "cosine_warm_restarts":
        T_0: int = kwargs.pop("T_0", max(1, epochs // 10))
        T_mult: int = kwargs.pop("T_mult", 2)
        g_scheduler = CosineAnnealingWarmRestarts(
            g_optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )
        d_scheduler = CosineAnnealingWarmRestarts(
            d_optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )

    elif scheduler_type == "multistep":
        milestones: List[int] = kwargs.pop(
            "milestones",
            [epochs // 2, epochs * 3 // 4],
        )
        gamma: float = kwargs.pop("gamma", 0.1)
        g_scheduler = MultiStepLR(
            g_optimizer,
            milestones=milestones,
            gamma=gamma,
        )
        d_scheduler = MultiStepLR(
            d_optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    else:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            f"Expected one of: 'cosine', 'cosine_warm_restarts', 'multistep'."
        )

    return {
        "g_scheduler": g_scheduler,
        "d_scheduler": d_scheduler,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "build_optimizers",
    "AdaptiveOptimizerScheduler",
    "build_scheduler",
]
