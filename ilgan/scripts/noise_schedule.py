"""
Learnable noise injection schedule for the ILGAN generator.

This module provides a :class:`NoiseScheduler` that wraps the generator's
learnable ``noise_std`` parameter and implements a deterministic annealing
schedule.  The schedule ensures high noise early in training (encouraging
exploration and preventing mode collapse) and low noise later (allowing
fine-tuning of details).

Mathematical formulation
------------------------
Let :math:`t` be the current training step (0-indexed), :math:`T` the total
number of training steps, :math:`\\sigma_0` the initial noise standard
deviation, and :math:`\\sigma_{\\min}` the minimum noise standard deviation.
The noise standard deviation at step :math:`t` is:

.. math::

    \\sigma(t) = (\\sigma_0 - \\sigma_{\\min}) \\cdot \\left(1 - \\frac{t}{T}\\right) + \\sigma_{\\min}

This is a **linear annealing schedule** that decays from :math:`\\sigma_0`
at :math:`t=0` to :math:`\\sigma_{\\min}` at :math:`t=T`.

The generator's ``noise_std`` parameter is updated in-place at each call
to :meth:`NoiseScheduler.step`, so the generator's forward pass
automatically uses the scheduled noise level without any code changes.

Why a noise schedule?
---------------------
Instance noise (also called "input noise" or "latent noise") is a
regularisation technique where a small amount of Gaussian noise is added
to the latent vector before it is passed through the generator.  This
has several benefits:

1. **Exploration (early training)**: High noise levels encourage the
   generator to explore a wider region of the latent space, preventing
   it from collapsing to a small set of modes.

2. **Fine-tuning (late training)**: Low noise levels allow the generator
   to fine-tune details without being perturbed by noise, leading to
   sharper images and more precise bounding boxes.

3. **Collapse prevention**: By maintaining a non-zero noise level
   throughout training, the generator is prevented from overfitting to
   a fixed set of latent vectors, which is a known cause of mode collapse
   in GANs.

4. **Bounding box diversity**: The noise schedule is particularly
   important for the ILGAN's dual output.  High noise early encourages
   the spatial head to explore different bounding box configurations,
   preventing all boxes from collapsing to the same location.

Usage
-----
The noise scheduler is typically created at the start of training and
called at each training step::

    from ilgan.scripts.noise_schedule import get_noise_scheduler

    # Create the scheduler
    noise_scheduler = get_noise_scheduler(generator, config)

    # In the training loop, at each step:
    noise_scheduler.step(global_step)

    # The generator's noise_std is now updated automatically.
    # The generator forward pass will use the new noise level.

The scheduler can also be used with a warmup phase where the noise is
held constant at :math:`\\sigma_0` for the first ``warmup_steps`` steps,
then annealed linearly to :math:`\\sigma_{\\min}` over the remaining steps.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_INITIAL_NOISE_STD: float = 0.1
"""Default initial noise standard deviation :math:`\\sigma_0`.  This is
set to 0.1, which is 10% of the standard normal latent distribution
:math:`\\mathcal{N}(0, I)`.  At this level, the noise significantly
perturbs the latent vector, encouraging exploration."""

_DEFAULT_MIN_NOISE_STD: float = 0.001
"""Default minimum noise standard deviation :math:`\\sigma_{\\min}`.  This
is set to 0.001, which is a very small perturbation that allows fine-tuning
without destabilising the generator."""

_DEFAULT_WARMUP_STEPS: int = 1000
"""Default number of warmup steps where the noise is held constant at
:math:`\\sigma_0`.  This prevents the noise from decaying too quickly
at the very start of training, giving the generator time to establish
a basic representation before fine-tuning begins."""

_EPS: float = 1e-8
"""Small epsilon to prevent division by zero in schedule computation."""


# ──────────────────────────────────────────────────────────────────────────────
# NoiseScheduler
# ──────────────────────────────────────────────────────────────────────────────


class NoiseScheduler:
    """Learnable noise injection schedule for the ILGAN generator.

    This class wraps the generator's ``noise_std`` parameter and implements
    a deterministic annealing schedule that decays the noise standard
    deviation from an initial value :math:`\\sigma_0` to a minimum value
    :math:`\\sigma_{\\min}` over the course of training.

    The schedule is:

    .. math::

        \\sigma(t) =
        \\begin{cases}
        \\sigma_0, & t < t_{\\text{warmup}} \\\\
        (\\sigma_0 - \\sigma_{\\min}) \\cdot \\left(1 - \\frac{t - t_{\\text{warmup}}}{T - t_{\\text{warmup}}}\\right) + \\sigma_{\\min}, & t \\geq t_{\\text{warmup}}
        \\end{cases}

    where :math:`t` is the current step, :math:`T` is the total number of
    steps, :math:`t_{\\text{warmup}}` is the number of warmup steps,
    :math:`\\sigma_0` is the initial noise std, and :math:`\\sigma_{\\min}`
    is the minimum noise std.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator.  Must have a ``noise_std`` attribute that is
        an ``nn.Parameter`` (scalar tensor).  This parameter is updated
        in-place by :meth:`step`.
    initial_noise_std : float, optional
        The initial noise standard deviation :math:`\\sigma_0`.  Must be
        positive.  (default: 0.1)
    min_noise_std : float, optional
        The minimum noise standard deviation :math:`\\sigma_{\\min}`.  Must
        be non-negative and less than ``initial_noise_std``.
        (default: 0.001)
    total_steps : int
        The total number of training steps :math:`T`.  Must be positive.
    warmup_steps : int, optional
        The number of warmup steps :math:`t_{\\text{warmup}}` during which
        the noise is held constant at :math:`\\sigma_0`.  Must be
        non-negative.  (default: 1000)

    Raises
    ------
    TypeError
        If *generator* does not have a ``noise_std`` attribute or it is
        not an ``nn.Parameter``.
    ValueError
        If any parameter is out of valid range.

    Notes
    -----
    - The generator's ``noise_std`` parameter is updated **in-place** at
      each call to :meth:`step`.  This means the generator's forward pass
      automatically uses the scheduled noise level.
    - The scheduler does **not** modify the computation graph.  The
      ``noise_std`` parameter remains a learnable parameter, but its value
      is overwritten by the schedule.  This allows the gradient to flow
      through the noise injection during backpropagation.
    - If ``total_steps <= warmup_steps``, the noise is held constant at
      :math:`\\sigma_0` for the entire training (no annealing).

    Examples
    --------
    >>> from ilgan.scripts.noise_schedule import NoiseScheduler
    >>> from ilgan.models import ILGANGenerator
    >>> from ilgan.utils.config import Config
    >>>
    >>> config = Config()
    >>> generator = ILGANGenerator(config)
    >>>
    >>> # Create scheduler for 100,000 total steps
    >>> scheduler = NoiseScheduler(
    ...     generator=generator,
    ...     initial_noise_std=0.1,
    ...     min_noise_std=0.001,
    ...     total_steps=100000,
    ...     warmup_steps=1000,
    ... )
    >>>
    >>> # At each training step:
    >>> scheduler.step(global_step)
    >>> # generator.noise_std is now updated
    >>> print(f"Current noise std: {generator.noise_std.item():.6f}")
    """

    def __init__(
        self,
        generator: nn.Module,
        initial_noise_std: float = _DEFAULT_INITIAL_NOISE_STD,
        min_noise_std: float = _DEFAULT_MIN_NOISE_STD,
        total_steps: int = 100_000,
        warmup_steps: int = _DEFAULT_WARMUP_STEPS,
    ) -> None:
        # ── Validate generator ──────────────────────────────────────────
        if not hasattr(generator, "noise_std"):
            raise TypeError(
                f"The generator must have a 'noise_std' attribute. "
                f"Got generator of type {type(generator).__name__} "
                f"which lacks this attribute."
            )
        if not isinstance(generator.noise_std, nn.Parameter):
            raise TypeError(
                f"The generator's 'noise_std' must be an nn.Parameter. "
                f"Got {type(generator.noise_std).__name__}."
            )
        if generator.noise_std.numel() != 1:
            raise ValueError(
                f"The generator's 'noise_std' must be a scalar parameter "
                f"(numel=1), got numel={generator.noise_std.numel()}."
            )

        self._generator = generator

        # ── Validate parameters ─────────────────────────────────────────
        if initial_noise_std <= 0.0:
            raise ValueError(
                f"initial_noise_std must be positive, got {initial_noise_std}."
            )
        if min_noise_std < 0.0:
            raise ValueError(
                f"min_noise_std must be non-negative, got {min_noise_std}."
            )
        if min_noise_std >= initial_noise_std:
            raise ValueError(
                f"min_noise_std ({min_noise_std}) must be less than "
                f"initial_noise_std ({initial_noise_std})."
            )
        if total_steps <= 0:
            raise ValueError(
                f"total_steps must be positive, got {total_steps}."
            )
        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {warmup_steps}."
            )

        self._initial_noise_std = initial_noise_std
        self._min_noise_std = min_noise_std
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps

        # ── Initialise the generator's noise_std ───────────────────────
        with torch.no_grad():
            self._generator.noise_std.fill_(initial_noise_std)

        # ── State tracking ─────────────────────────────────────────────
        self._current_step: int = 0
        self._current_noise_std: float = initial_noise_std

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, t: int) -> float:
        """Update the generator's ``noise_std`` parameter according to the
        annealing schedule at step ``t``.

        The schedule is:

        .. math::

            \\sigma(t) =
            \\begin{cases}
            \\sigma_0, & t < t_{\\text{warmup}} \\\\
            (\\sigma_0 - \\sigma_{\\min}) \\cdot \\left(1 - \\frac{t - t_{\\text{warmup}}}{T - t_{\\text{warmup}}}\\right) + \\sigma_{\\min}, & t \\geq t_{\\text{warmup}}
            \\end{cases}

        where :math:`t` is the current step, :math:`T` is the total number
        of steps, :math:`t_{\\text{warmup}}` is the number of warmup steps,
        :math:`\\sigma_0` is the initial noise std, and :math:`\\sigma_{\\min}`
        is the minimum noise std.

        Parameters
        ----------
        t : int
            The current training step (0-indexed).  Must be non-negative.

        Returns
        -------
        float
            The new noise standard deviation value that was set on the
            generator's ``noise_std`` parameter.

        Raises
        ------
        ValueError
            If ``t`` is negative.

        Notes
        -----
        - If ``t`` exceeds ``total_steps - 1``, the noise is clamped to
          :math:`\\sigma_{\\min}` (the minimum value).  This prevents the
          noise from going below the minimum if training continues beyond
          the scheduled number of steps.
        - The generator's ``noise_std`` parameter is updated **in-place**
          using ``data.fill_()``, which does not break the computation
          graph.  The parameter remains a learnable ``nn.Parameter``.
        """
        if t < 0:
            raise ValueError(
                f"Step t must be non-negative, got {t}."
            )

        self._current_step = t

        # ── Compute the scheduled noise std ─────────────────────────────
        if t < self._warmup_steps:
            # Warmup phase: hold at initial value
            sigma = self._initial_noise_std
        elif t >= self._total_steps:
            # Beyond total steps: clamp to minimum
            sigma = self._min_noise_std
        else:
            # Annealing phase: linear decay from initial to minimum
            # Effective progress through the annealing phase
            progress = (t - self._warmup_steps) / max(
                self._total_steps - self._warmup_steps, _EPS
            )
            # Clamp progress to [0, 1] to avoid numerical issues
            progress = max(0.0, min(1.0, progress))

            # Linear interpolation:
            #   sigma = (sigma_0 - sigma_min) * (1 - progress) + sigma_min
            # At progress=0: sigma = sigma_0
            # At progress=1: sigma = sigma_min
            sigma_range = self._initial_noise_std - self._min_noise_std
            sigma = sigma_range * (1.0 - progress) + self._min_noise_std

        # ── Update the generator's noise_std parameter in-place ─────────
        with torch.no_grad():
            self._generator.noise_std.fill_(sigma)

        self._current_noise_std = sigma

        return sigma

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def current_noise_std(self) -> float:
        """The current noise standard deviation value.

        This is the value that was set at the most recent call to
        :meth:`step`.  If :meth:`step` has not been called yet, this
        returns the initial value :math:`\\sigma_0`.
        """
        return self._current_noise_std

    @property
    def current_step(self) -> int:
        """The most recent step number passed to :meth:`step`."""
        return self._current_step

    @property
    def initial_noise_std(self) -> float:
        """The initial noise standard deviation :math:`\\sigma_0`."""
        return self._initial_noise_std

    @property
    def min_noise_std(self) -> float:
        """The minimum noise standard deviation :math:`\\sigma_{\\min}`."""
        return self._min_noise_std

    @property
    def total_steps(self) -> int:
        """The total number of training steps :math:`T`."""
        return self._total_steps

    @property
    def warmup_steps(self) -> int:
        """The number of warmup steps :math:`t_{\\text{warmup}}`."""
        return self._warmup_steps

    @property
    def progress(self) -> float:
        """The current progress through the annealing phase as a fraction
        in ``[0, 1]``.

        Returns 0.0 during the warmup phase, 1.0 after the total steps
        have been completed, and a value in ``(0, 1)`` during the
        annealing phase.
        """
        if self._current_step < self._warmup_steps:
            return 0.0
        if self._current_step >= self._total_steps:
            return 1.0
        return (self._current_step - self._warmup_steps) / max(
            self._total_steps - self._warmup_steps, _EPS
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Any]:
        """Return the scheduler's state for checkpointing.

        Returns
        -------
        dict
            A dictionary containing the scheduler's configuration and
            current state.  Can be passed to :meth:`load_state_dict` to
            restore the scheduler.

        Notes
        -----
        - The generator's ``noise_std`` parameter is **not** included in
          the state dict because it is part of the generator's own state
          dict and is saved/loaded separately.
        - The state dict includes the configuration parameters
          (``initial_noise_std``, ``min_noise_std``, ``total_steps``,
          ``warmup_steps``) and the current step, so the scheduler can be
          resumed from the exact same point.
        """
        return {
            "initial_noise_std": self._initial_noise_std,
            "min_noise_std": self._min_noise_std,
            "total_steps": self._total_steps,
            "warmup_steps": self._warmup_steps,
            "current_step": self._current_step,
            "current_noise_std": self._current_noise_std,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore the scheduler's state from a checkpoint.

        Parameters
        ----------
        state_dict : dict
            A dictionary produced by :meth:`state_dict`.  Must contain
            all keys returned by :meth:`state_dict`.

        Raises
        ------
        KeyError
            If *state_dict* is missing required keys.
        ValueError
            If the configuration in *state_dict* does not match the
            scheduler's current configuration.

        Notes
        -----
        - This method updates the generator's ``noise_std`` parameter to
          the value stored in ``current_noise_std``.
        - The configuration parameters (``initial_noise_std``,
          ``min_noise_std``, ``total_steps``, ``warmup_steps``) must match
          the scheduler's current configuration.  If they do not match,
          a ``ValueError`` is raised.
        """
        # Validate configuration matches
        for key in ["initial_noise_std", "min_noise_std", "total_steps", "warmup_steps"]:
            if key not in state_dict:
                raise KeyError(
                    f"State dict is missing required key '{key}'."
                )
            stored_value = state_dict[key]
            current_value = getattr(self, f"_{key}")
            if abs(stored_value - current_value) > _EPS:
                raise ValueError(
                    f"Scheduler configuration mismatch for '{key}': "
                    f"stored={stored_value}, current={current_value}. "
                    f"Cannot load state dict from a different configuration."
                )

        # Restore state
        self._current_step = state_dict["current_step"]
        self._current_noise_std = state_dict["current_noise_std"]

        # Update the generator's noise_std parameter
        with torch.no_grad():
            self._generator.noise_std.fill_(self._current_noise_std)

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"NoiseScheduler(\n"
            f"  initial_noise_std={self._initial_noise_std},\n"
            f"  min_noise_std={self._min_noise_std},\n"
            f"  total_steps={self._total_steps},\n"
            f"  warmup_steps={self._warmup_steps},\n"
            f"  current_step={self._current_step},\n"
            f"  current_noise_std={self._current_noise_std:.6f},\n"
            f"  progress={self.progress:.4f},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────────────────────


def get_noise_scheduler(
    generator: nn.Module,
    config: Config,
) -> NoiseScheduler:
    """Create a :class:`NoiseScheduler` with parameters extracted from the
    ILGAN configuration.

    This factory function reads the following keys from the config:

    - ``training.noise_schedule.initial_noise_std`` (float, optional):
      Initial noise standard deviation :math:`\\sigma_0`.  Default: 0.1.
    - ``training.noise_schedule.min_noise_std`` (float, optional):
      Minimum noise standard deviation :math:`\\sigma_{\\min}`.
      Default: 0.001.
    - ``training.noise_schedule.warmup_steps`` (int, optional):
      Number of warmup steps.  Default: 1000.
    - ``training.epochs`` (int): Total number of training epochs.
    - ``data.batch_size`` (int): Batch size.
    - ``training.n_critic`` (int, optional): Number of discriminator
      updates per generator update.  Default: 5.

    The total number of steps :math:`T` is computed as:

    .. math::

        T = \\text{epochs} \\times \\left\\lceil \\frac{N_{\\text{train}}}{\\text{batch\\_size}} \\right\\rceil

    where :math:`N_{\\text{train}}` is the approximate number of training
    samples.  If the exact number of training samples is not known, a
    default of 100,000 is used.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator.  Must have a ``noise_std`` attribute that is
        an ``nn.Parameter``.
    config : Config
        The ILGAN configuration object.

    Returns
    -------
    NoiseScheduler
        A configured :class:`NoiseScheduler` instance.

    Raises
    ------
    TypeError
        If *generator* does not have a ``noise_std`` attribute or
        *config* is not a :class:`Config` instance.
    ValueError
        If the config is missing required keys or contains invalid values.

    Examples
    --------
    >>> from ilgan.scripts.noise_schedule import get_noise_scheduler
    >>> from ilgan.models import ILGANGenerator
    >>> from ilgan.utils.config import Config
    >>>
    >>> config = Config()
    >>> generator = ILGANGenerator(config)
    >>> scheduler = get_noise_scheduler(generator, config)
    >>> print(scheduler)
    """
    # ── Validate inputs ─────────────────────────────────────────────────
    if not isinstance(config, Config):
        raise TypeError(
            f"Expected 'config' to be a Config instance, "
            f"got {type(config).__name__}."
        )
    if not hasattr(generator, "noise_std"):
        raise TypeError(
            f"The generator must have a 'noise_std' attribute. "
            f"Got generator of type {type(generator).__name__} "
            f"which lacks this attribute."
        )

    # ── Extract noise schedule parameters from config ───────────────────
    # Try to read from training.noise_schedule sub-section first
    try:
        noise_schedule_cfg = config.training.noise_schedule
        initial_noise_std = float(
            getattr(noise_schedule_cfg, "initial_noise_std", _DEFAULT_INITIAL_NOISE_STD)
        )
        min_noise_std = float(
            getattr(noise_schedule_cfg, "min_noise_std", _DEFAULT_MIN_NOISE_STD)
        )
        warmup_steps = int(
            getattr(noise_schedule_cfg, "warmup_steps", _DEFAULT_WARMUP_STEPS)
        )
    except AttributeError:
        # Fallback: use defaults or read from flat config
        initial_noise_std = float(
            getattr(config, "noise_initial_std", _DEFAULT_INITIAL_NOISE_STD)
        )
        min_noise_std = float(
            getattr(config, "noise_min_std", _DEFAULT_MIN_NOISE_STD)
        )
        warmup_steps = int(
            getattr(config, "noise_warmup_steps", _DEFAULT_WARMUP_STEPS)
        )

    # ── Compute total steps ─────────────────────────────────────────────
    # Estimate total steps from epochs, batch size, and n_critic
    try:
        epochs: int = int(config.training.epochs)
        batch_size: int = int(config.data.batch_size)
        n_critic: int = int(
            getattr(config.training, "n_critic", 5)
        )
    except (AttributeError, KeyError, TypeError) as e:
        raise ValueError(
            f"Config is missing required keys for noise scheduler: {e}. "
            f"Please ensure your config has 'training.epochs', "
            f"'data.batch_size', and optionally 'training.n_critic'."
        ) from e

    # Estimate the number of training batches per epoch.
    # We use a heuristic: assume ~1000 batches per epoch if we don't know
    # the exact dataset size.  This is a reasonable default for most
    # datasets (e.g., COCO has ~118k training images → ~7,375 batches at
    # batch_size=16).
    #
    # The total number of generator steps is:
    #   total_steps = epochs * num_batches_per_epoch / n_critic
    #
    # Because the generator is updated only every n_critic steps, the
    # noise schedule should be aligned with generator updates, not
    # discriminator updates.
    #
    # We use a conservative estimate of 1000 batches per epoch, which
    # corresponds to a dataset of ~16,000 images at batch_size=16.
    # This can be overridden by setting training.noise_schedule.total_steps
    # in the config.
    try:
        total_steps = int(config.training.noise_schedule.total_steps)
    except (AttributeError, KeyError, TypeError):
        # Estimate: assume ~1000 batches per epoch
        estimated_batches_per_epoch: int = 1000
        total_steps = epochs * estimated_batches_per_epoch

    # ── Create and return the scheduler ─────────────────────────────────
    return NoiseScheduler(
        generator=generator,
        initial_noise_std=initial_noise_std,
        min_noise_std=min_noise_std,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: create and attach to training loop
# ──────────────────────────────────────────────────────────────────────────────


def create_noise_scheduler_from_config(
    generator: nn.Module,
    config: Config,
    num_training_samples: Optional[int] = None,
) -> NoiseScheduler:
    """Create a :class:`NoiseScheduler` with an accurate total step count
    based on the actual number of training samples.

    This is an alternative to :func:`get_noise_scheduler` that allows
    specifying the exact number of training samples for a more accurate
    total step calculation.

    The total number of steps :math:`T` is computed as:

    .. math::

        T = \\text{epochs} \\times \\left\\lceil \\frac{N_{\\text{train}}}{\\text{batch\\_size}} \\right\\rceil

    where :math:`N_{\\text{train}}` is the number of training samples.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator.
    config : Config
        The ILGAN configuration object.
    num_training_samples : int, optional
        The exact number of training samples.  If ``None``, uses the
        default estimate of 1000 batches per epoch.

    Returns
    -------
    NoiseScheduler
        A configured :class:`NoiseScheduler` instance.

    Examples
    --------
    >>> from ilgan.scripts.noise_schedule import (
    ...     create_noise_scheduler_from_config,
    ... )
    >>>
    >>> # After creating the dataloader:
    >>> num_train = len(train_loader.dataset)
    >>> scheduler = create_noise_scheduler_from_config(
    ...     generator, config, num_training_samples=num_train,
    ... )
    """
    if num_training_samples is not None and num_training_samples > 0:
        try:
            epochs: int = int(config.training.epochs)
            batch_size: int = int(config.data.batch_size)
            batches_per_epoch: int = math.ceil(num_training_samples / batch_size)
            total_steps: int = epochs * batches_per_epoch
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Config is missing required keys: {e}."
            ) from e
    else:
        total_steps = None

    # Extract noise schedule parameters
    try:
        noise_schedule_cfg = config.training.noise_schedule
        initial_noise_std = float(
            getattr(noise_schedule_cfg, "initial_noise_std", _DEFAULT_INITIAL_NOISE_STD)
        )
        min_noise_std = float(
            getattr(noise_schedule_cfg, "min_noise_std", _DEFAULT_MIN_NOISE_STD)
        )
        warmup_steps = int(
            getattr(noise_schedule_cfg, "warmup_steps", _DEFAULT_WARMUP_STEPS)
        )
    except AttributeError:
        initial_noise_std = _DEFAULT_INITIAL_NOISE_STD
        min_noise_std = _DEFAULT_MIN_NOISE_STD
        warmup_steps = _DEFAULT_WARMUP_STEPS

    if total_steps is None:
        total_steps = epochs * 1000  # fallback estimate

    return NoiseScheduler(
        generator=generator,
        initial_noise_std=initial_noise_std,
        min_noise_std=min_noise_std,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "NoiseScheduler",
    "get_noise_scheduler",
    "create_noise_scheduler_from_config",
]
