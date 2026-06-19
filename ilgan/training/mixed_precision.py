"""
Mixed precision training support for ILGAN.

This module provides three core components for Automatic Mixed Precision (AMP)
training using PyTorch's native ``torch.amp`` package:

1. :class:`AMPScaler` — a wrapper around ``torch.amp.GradScaler`` that
   provides a clean API for scaling losses, unscaling gradients, and stepping
   optimizers, with graceful fallback to no-ops when CUDA is unavailable.

2. :func:`should_use_amp` — a decision function that checks both CUDA
   availability and the user's configuration setting.

3. :func:`autocast_context` — a context manager that yields the appropriate
   autocast context (``torch.amp.autocast("cuda")`` if AMP is enabled, otherwise
   a no-op context manager).

Mathematical motivation
-----------------------
Mixed precision training (Micikevicius et al., 2018) accelerates training and
reduces VRAM usage by performing most operations in ``float16`` while keeping
critical operations (e.g., loss scaling, gradient updates) in ``float32``.
This is especially important for ILGAN because:

- **VRAM efficiency**: ILGAN's dual-output generator and multi-head attention
  blocks are memory-intensive.  AMP reduces activation memory by ~40%,
  allowing larger batch sizes or higher-resolution images.

- **Gradient scaling**: Small gradients from the adversarial loss can underflow
  in ``float16``.  The :class:`AMPScaler` applies a dynamic loss scale factor
  :math:`S` that grows/shrinks to keep gradients in the representable range
  of ``float16``:

  .. math::

      \mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}

      \nabla_{\theta} = \frac{1}{S} \cdot \nabla_{\theta} \mathcal{L}_{\text{scaled}}

  where :math:`S` is adjusted dynamically: if inf/nan gradients are detected,
  :math:`S` is halved; otherwise, :math:`S` is increased by a small factor
  every ``growth_interval`` steps.

- **Autocast**: Operations within the ``autocast`` context are automatically
  cast to ``float16`` or ``float32`` based on their operator-specific
  precision rules, ensuring numerical stability for sensitive operations
  (e.g., softmax, layer norm) while accelerating compute-bound operations
  (e.g., convolutions, matmuls).
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Any, Generator, Optional, Union

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler, OptState

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CUDA_AVAILABLE: bool = torch.cuda.is_available()
"""Cached flag indicating whether CUDA is available on this system."""

_DEFAULT_GROWTH_FACTOR: float = 2.0
"""Default multiplicative growth factor for the GradScaler's scale."""

_DEFAULT_BACKOFF_FACTOR: float = 0.5
"""Default multiplicative backoff factor when inf/nan gradients are detected."""

_DEFAULT_GROWTH_INTERVAL: int = 2000
"""Default number of steps without inf/nan before increasing the scale."""

_DEFAULT_INIT_SCALE: float = 2.0 ** 16
"""Default initial loss scale (65536.0)."""

_DEFAULT_ENABLED: bool = True
"""Default enabled state for the GradScaler when CUDA is available."""


# ──────────────────────────────────────────────────────────────────────────────
# AMPScaler
# ──────────────────────────────────────────────────────────────────────────────


class AMPScaler:
    """Wrapper around ``torch.amp.GradScaler`` with graceful CPU fallback.

    The :class:`AMPScaler` provides a unified interface for mixed precision
    training that works seamlessly whether or not CUDA is available.  When
    CUDA is available, it delegates to ``torch.amp.GradScaler`` for
    dynamic loss scaling.  When CUDA is not available, all methods become
    no-ops, allowing the same training code to run on CPU without
    modification.

    The scaler implements dynamic loss scaling to prevent gradient underflow
    in ``float16``.  The loss scale factor :math:`S` is adjusted according
    to the following algorithm:

    .. math::

        S_{t+1} =
        \\begin{cases}
        S_t \\cdot \\text{growth\\_factor}
            & \\text{if no inf/nan for growth\\_interval steps} \\\\
        S_t \\cdot \\text{backoff\\_factor}
            & \\text{if inf/nan detected} \\\\
        S_t & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    init_scale : float, optional
        Initial loss scale factor :math:`S_0`.  (default: 65536.0)
    growth_factor : float, optional
        Multiplicative factor by which the scale grows after a period of
        no inf/nan gradients.  (default: 2.0)
    backoff_factor : float, optional
        Multiplicative factor by which the scale shrinks when inf/nan
        gradients are detected.  (default: 0.5)
    growth_interval : int, optional
        Number of consecutive steps without inf/nan before the scale is
        increased.  (default: 2000)
    enabled : bool, optional
        Whether the scaler is enabled.  If ``False``, all methods are no-ops.
        (default: ``True`` when CUDA is available, ``False`` otherwise)

    Raises
    ------
    ValueError
        If any of the numeric parameters are out of valid ranges.

    Example
    -------
    >>> scaler = AMPScaler()
    >>> for batch in dataloader:
    ...     with autocast_context(scaler.is_enabled):
    ...         loss = criterion(model(batch))
    ...     scaler.scale_loss(loss, optimizer).backward()
    ...     scaler.unscale_(optimizer)
    ...     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    ...     scaler.step(optimizer)
    """

    def __init__(
        self,
        init_scale: float = _DEFAULT_INIT_SCALE,
        growth_factor: float = _DEFAULT_GROWTH_FACTOR,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
        growth_interval: int = _DEFAULT_GROWTH_INTERVAL,
        enabled: Optional[bool] = None,
    ) -> None:
        # ── Determine enabled state ─────────────────────────────────────
        if enabled is None:
            self._enabled: bool = _CUDA_AVAILABLE and _DEFAULT_ENABLED
        else:
            self._enabled = enabled

        # ── Validate parameters ──────────────────────────────────────────
        if init_scale <= 0.0:
            raise ValueError(
                f"init_scale must be positive, got {init_scale}."
            )
        if growth_factor <= 1.0:
            raise ValueError(
                f"growth_factor must be > 1.0, got {growth_factor}."
            )
        if not (0.0 < backoff_factor < 1.0):
            raise ValueError(
                f"backoff_factor must be in (0, 1), got {backoff_factor}."
            )
        if growth_interval < 1:
            raise ValueError(
                f"growth_interval must be positive, got {growth_interval}."
            )

        # ── Internal state ───────────────────────────────────────────────
        self._scaler: Optional[GradScaler] = None
        self._init_scale: float = init_scale
        self._growth_factor: float = growth_factor
        self._backoff_factor: float = backoff_factor
        self._growth_interval: int = growth_interval

        # Create the underlying GradScaler if CUDA is available and enabled
        if self._enabled and _CUDA_AVAILABLE:
            self._scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=True,
            )
        elif self._enabled and not _CUDA_AVAILABLE:
            warnings.warn(
                "AMPScaler is enabled but CUDA is not available. "
                "Falling back to no-op mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._enabled = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def scale_loss(
        self,
        loss: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """Scale the loss tensor for mixed precision training.

        When AMP is enabled, this scales the loss by the current scale factor
        :math:`S` so that gradients computed in ``float16`` remain in the
        representable range.  The *optimizer* argument is accepted for API
        consistency but is not used by the underlying ``GradScaler.scale()``
        method (the scaler applies the same scale to all losses regardless
        of optimizer).

        When AMP is disabled, this is a no-op that returns the loss unchanged.

        Parameters
        ----------
        loss : torch.Tensor
            The unscaled loss tensor (a scalar tensor).
        optimizer : torch.optim.Optimizer, optional
            Ignored; accepted for API consistency with ``GradScaler.scale()``.

        Returns
        -------
        torch.Tensor
            The scaled loss tensor (or the original loss if AMP is disabled).

        Raises
        ------
        TypeError
            If *loss* is not a ``torch.Tensor``.
        """
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"Expected loss to be a torch.Tensor, "
                f"got {type(loss).__name__}."
            )

        if self._enabled and self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale the gradients for the given optimizer.

        When AMP is enabled, this divides the gradients of all parameters
        managed by *optimizer* by the current scale factor :math:`S`,
        restoring them to their true magnitudes before gradient clipping or
        the optimizer step.

        When AMP is disabled, this is a no-op.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose parameter gradients should be unscaled.

        Raises
        ------
        TypeError
            If *optimizer* is not a ``torch.optim.Optimizer``.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Expected optimizer to be a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}."
            )

        if self._enabled and self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step the optimizer and update the loss scale.

        When AMP is enabled, this:

        1. Calls ``scaler.step(optimizer)``, which checks for inf/nan
           gradients.  If any are found, the optimizer step is skipped
           and the scale is reduced by ``backoff_factor``.
        2. Calls ``scaler.update()``, which adjusts the scale factor
           based on whether inf/nan were detected.

        When AMP is disabled, this simply calls ``optimizer.step()``.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to step.

        Returns
        -------
        bool
            ``True`` if the optimizer step was successfully applied,
            ``False`` if it was skipped due to inf/nan gradients (only
            possible when AMP is enabled).

        Raises
        ------
        TypeError
            If *optimizer* is not a ``torch.optim.Optimizer``.
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Expected optimizer to be a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}."
            )

        if self._enabled and self._scaler is not None:
            # step() returns a boolean indicating whether the step was skipped
            self._scaler.step(optimizer)
            self._scaler.update()
            # Check if the step was skipped (inf/nan detected)
            # We can infer this from the scaler's internal state
            return self._scaler._found_inf_per_device is None or not any(
                v.item() > 0 for v in self._scaler._found_inf_per_device.values()
            )
        else:
            optimizer.step()
            return True

    def state_dict(self) -> dict:
        """Return the scaler's state dictionary for checkpointing.

        When AMP is enabled, this returns the underlying ``GradScaler``'s
        state dict, which includes the current scale factor, the number of
        steps since the last growth, and other internal state.

        When AMP is disabled, this returns an empty dict.

        Returns
        -------
        dict
            The scaler state dictionary.
        """
        if self._enabled and self._scaler is not None:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the scaler's state from a checkpoint.

        When AMP is enabled, this restores the underlying ``GradScaler``'s
        state from *state_dict*.

        When AMP is disabled, this is a no-op.

        Parameters
        ----------
        state_dict : dict
            The state dictionary to load (as returned by :meth:`state_dict`).
        """
        if self._enabled and self._scaler is not None:
            self._scaler.load_state_dict(state_dict)

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        """Whether AMP is active.

        Returns ``True`` if the scaler was constructed with ``enabled=True``
        **and** CUDA is available.  Returns ``False`` otherwise.

        This property is read-only.  To change the enabled state, construct
        a new :class:`AMPScaler` with the desired ``enabled`` parameter.
        """
        return self._enabled

    @property
    def scale(self) -> float:
        """The current loss scale factor :math:`S`.

        When AMP is enabled, this returns the current scale from the
        underlying ``GradScaler``.  When AMP is disabled, this returns 1.0.

        Returns
        -------
        float
            The current loss scale factor.
        """
        if self._enabled and self._scaler is not None:
            return self._scaler.get_scale()
        return 1.0

    @property
    def growth_factor(self) -> float:
        """The multiplicative growth factor for the loss scale."""
        return self._growth_factor

    @property
    def backoff_factor(self) -> float:
        """The multiplicative backoff factor for the loss scale."""
        return self._backoff_factor

    @property
    def growth_interval(self) -> int:
        """The number of steps between scale growth events."""
        return self._growth_interval

    @property
    def init_scale(self) -> float:
        """The initial loss scale factor."""
        return self._init_scale

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AMPScaler(\n"
            f"  enabled={self._enabled},\n"
            f"  scale={self.scale:.1f},\n"
            f"  init_scale={self._init_scale:.1f},\n"
            f"  growth_factor={self._growth_factor},\n"
            f"  backoff_factor={self._backoff_factor},\n"
            f"  growth_interval={self._growth_interval},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# should_use_amp
# ──────────────────────────────────────────────────────────────────────────────


def should_use_amp(config: Config) -> bool:
    """Determine whether mixed precision training should be used.

    Returns ``True`` if **both** of the following conditions are met:

    1. CUDA is available (``torch.cuda.is_available()`` returns ``True``).
    2. The configuration has ``config.training.use_mixed_precision`` set to
       ``True``.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.  The following key is read:

        - ``config.training.use_mixed_precision`` (bool): whether the user
          has requested mixed precision training.

    Returns
    -------
    bool
        ``True`` if AMP should be used, ``False`` otherwise.

    Example
    -------
    >>> from ilgan.training.mixed_precision import should_use_amp
    >>> use_amp = should_use_amp(config)
    >>> scaler = AMPScaler(enabled=use_amp)
    """
    if not _CUDA_AVAILABLE:
        return False

    try:
        return bool(config.training.use_mixed_precision)
    except (AttributeError, KeyError, TypeError):
        warnings.warn(
            "Config key 'training.use_mixed_precision' not found. "
            "Defaulting to AMP disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


# ──────────────────────────────────────────────────────────────────────────────
# autocast_context
# ──────────────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def autocast_context(use_amp: bool) -> Generator[Any, None, None]:
    """Context manager that yields the appropriate autocast context.

    When ``use_amp`` is ``True`` **and** CUDA is available, this yields
    ``torch.amp.autocast("cuda")``, which automatically casts operations to
    ``float16`` or ``float32`` as appropriate.

    When ``use_amp`` is ``False`` or CUDA is not available, this yields a
    no-op context manager (``contextlib.nullcontext``), so the same code
    can be used without modification on CPU-only systems.

    Parameters
    ----------
    use_amp : bool
        Whether to enable autocast.  Typically the return value of
        :func:`should_use_amp`.

    Yields
    ------
    context manager
        Either ``torch.amp.autocast(device_type)`` or a no-op context manager.

    Example
    -------
    >>> from ilgan.training.mixed_precision import (
    ...     should_use_amp, autocast_context, AMPScaler
    ... )
    >>> use_amp = should_use_amp(config)
    >>> scaler = AMPScaler(enabled=use_amp)
    >>> for batch in dataloader:
    ...     with autocast_context(use_amp):
    ...         output = model(batch)
    ...         loss = loss_fn(output, target)
    ...     scaler.scale_loss(loss, optimizer).backward()
    ...     scaler.step(optimizer)
    """
    if use_amp:
        from ilgan.utils.device import get_amp_device_type
        device_type = get_amp_device_type()
        if device_type in ("cuda", "mps"):
            with autocast(device_type):
                yield
        else:
            yield
    else:
        yield


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: create_amp_scaler
# ──────────────────────────────────────────────────────────────────────────────


def create_amp_scaler(
    config: Config,
    **kwargs: Any,
) -> AMPScaler:
    """Create an :class:`AMPScaler` from a configuration object.

    This is a convenience factory that calls :func:`should_use_amp` to
    determine whether AMP should be enabled, then constructs an
    :class:`AMPScaler` with the appropriate settings.

    Additional keyword arguments are passed directly to the
    :class:`AMPScaler` constructor, allowing overrides of the default
    growth factor, backoff factor, etc.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.
    **kwargs
        Additional keyword arguments passed to :class:`AMPScaler`.

    Returns
    -------
    AMPScaler
        An :class:`AMPScaler` instance configured according to the config.

    Example
    -------
    >>> from ilgan.training.mixed_precision import create_amp_scaler
    >>> scaler = create_amp_scaler(config)
    >>> print(f"AMP enabled: {scaler.is_enabled}")
    """
    use_amp = should_use_amp(config)
    return AMPScaler(enabled=use_amp, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "AMPScaler",
    "should_use_amp",
    "autocast_context",
    "create_amp_scaler",
]
