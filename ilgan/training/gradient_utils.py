"""
Gradient management utilities for ILGAN training.

This module provides four core utilities for managing gradients during
ILGAN training:

1. :func:`clip_gradients` — clips gradients to a specified max norm and
   returns the total gradient norm **before** clipping for logging.

2. :func:`log_gradient_statistics` — logs the mean and max gradient norm
   for each parameter group (useful for debugging training stability and
   detecting vanishing/exploding gradients early).

3. :func:`detect_nan_inf_gradients` — checks for NaN or Inf gradients
   across all parameters and returns a list of parameter names with
   problematic gradients.  Critical for early detection of training
   divergence.

4. :func:`zero_gradients` — zeros gradients for all provided optimizers
   in a single call (convenience for the training loop).

Mathematical motivation
-----------------------
Gradient management is critical for stable GAN training.  The ILGAN
dual-output architecture is particularly sensitive to gradient issues
because:

- **Vanishing gradients**: if the discriminator becomes too strong, the
  generator receives near-zero gradients, causing both image and bounding
  box predictions to collapse.  Monitoring gradient norms per parameter
  group allows early detection of this condition.

- **Exploding gradients**: spectral normalisation mitigates this, but
  the cross-attention modules in the spatial head can still produce
  large gradients if attention weights become concentrated.  Clipping
  prevents these from destabilising training.

- **NaN/Inf propagation**: a single NaN gradient in any parameter will
  corrupt the entire optimizer state.  Early detection via
  :func:`detect_nan_inf_gradients` allows the training loop to skip
  the step or reset the problematic parameters before divergence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ilgan.utils.logger import Logger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon to prevent division by zero in gradient norm computations."""

_DEFAULT_MAX_NORM: float = 1.0
"""Default maximum gradient norm for clipping."""


# ──────────────────────────────────────────────────────────────────────────────
# clip_gradients
# ──────────────────────────────────────────────────────────────────────────────


def clip_gradients(
    model: nn.Module,
    max_norm: float = _DEFAULT_MAX_NORM,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> float:
    """Clip gradients of all parameters in *model* to a specified max norm.

    This function wraps ``torch.nn.utils.clip_grad_norm_`` but **returns the
    total gradient norm before clipping**, which is essential for logging
    and monitoring training dynamics.  The clipping is applied in-place to
    the model's parameter gradients.

    The total gradient norm :math:`N` is computed as:

    .. math::

        N = \\left( \\sum_{p \\in \\text{model.parameters()}}
            \\|\\nabla_p \\mathcal{L}\\|_{\\text{norm\\_type}}
            ^{\\text{norm\\_type}} \\right)^{1 / \\text{norm\\_type}}

    where :math:`\\nabla_p \\mathcal{L}` is the gradient of the loss with
    respect to parameter :math:`p`.  If :math:`N > \\text{max\\_norm}`, all
    gradients are scaled by :math:`\\text{max\\_norm} / N`.

    Parameters
    ----------
    model : nn.Module
        The model whose gradients should be clipped.  All parameters with
        ``requires_grad=True`` and non-None gradients are included.
    max_norm : float, optional
        Maximum allowed gradient norm.  Gradients are scaled down if the
        total norm exceeds this value.  Must be positive.  (default: 1.0)
    norm_type : float, optional
        Type of norm to use.  ``2.0`` for L2 norm (Euclidean), ``1.0`` for
        L1 norm, ``float('inf')`` for max norm.  (default: 2.0)
    error_if_nonfinite : bool, optional
        If ``True``, raise an error if any gradient is NaN or Inf (in
        addition to clipping).  This is useful for debugging but may
        interrupt training.  (default: ``False``)

    Returns
    -------
    float
        The total gradient norm **before** clipping.  This value can be
        logged to monitor training stability:

        - **Too small** (e.g., < 1e-6): gradients are vanishing; the
          discriminator may be too strong, or the generator has collapsed.
        - **Too large** (e.g., > 100): gradients are exploding; consider
          increasing spectral normalisation or reducing the learning rate.
        - **Stable range**: typically 0.1–10.0 for well-tuned GANs.

    Raises
    ------
    TypeError
        If *model* is not an ``nn.Module``.
    ValueError
        If *max_norm* is not positive.
    RuntimeError
        If *error_if_nonfinite* is ``True`` and any gradient is NaN or Inf.

    Example
    -------
    >>> from ilgan.training.gradient_utils import clip_gradients
    >>> total_norm = clip_gradients(generator, max_norm=1.0)
    >>> logger.info(f"Generator gradient norm before clipping: {total_norm:.4f}")
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected 'model' to be an nn.Module, "
            f"got {type(model).__name__}."
        )
    if max_norm <= 0.0:
        raise ValueError(
            f"max_norm must be positive, got {max_norm}."
        )
    if norm_type <= 0.0 and norm_type != float("inf"):
        raise ValueError(
            f"norm_type must be positive or float('inf'), got {norm_type}."
        )

    # ── Collect all parameters with gradients ─────────────────────────────
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    # ── Compute total norm before clipping ──────────────────────────────
    # We use torch.nn.utils.clip_grad_norm_ which returns the total norm
    # before clipping.  This is the most numerically stable approach since
    # it handles the norm computation internally.
    total_norm: float = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )

    return total_norm


# ──────────────────────────────────────────────────────────────────────────────
# log_gradient_statistics
# ──────────────────────────────────────────────────────────────────────────────


def log_gradient_statistics(
    model: nn.Module,
    logger: Logger,
    step: int,
    param_groups: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Log the mean and max gradient norm for each parameter group.

    This function computes per-parameter-group gradient statistics and logs
    them via the provided *logger*.  It also returns the statistics as a
    dictionary for further analysis or metric tracking.

    A *parameter group* is a named subset of the model's parameters (e.g.,
    ``"generator"``, ``"image_encoder"``, ``"box_encoder"``,
    ``"spatial_head.box_head"``).  If *param_groups* is not provided, the
    function groups parameters by their top-level module name (the first
    component of their fully-qualified name).

    For each group, the following statistics are computed:

    - **mean_norm**: the mean L2 norm of gradients across all parameters
      in the group.  A very small value (< 1e-6) indicates vanishing
      gradients.
    - **max_norm**: the maximum L2 norm among all parameters in the group.
      A very large value (> 100) indicates exploding gradients.
    - **fraction_zero**: the fraction of parameters in the group that have
      zero gradients (useful for detecting dead neurons).

    Parameters
    ----------
    model : nn.Module
        The model whose gradient statistics to compute.
    logger : Logger
        The ILGAN logger instance (``ilgan.utils.logger.Logger``) for
        logging the statistics at INFO level.
    step : int
        The current training step (used in the log message for context).
    param_groups : dict of str -> list of str, optional
        A dictionary mapping group names to lists of parameter name
        prefixes.  For example::

            {
                "generator": ["content_decoder", "spatial_head.box_head"],
                "image_encoder": ["image_encoder"],
                "box_encoder": ["box_encoder"],
            }

        If ``None``, parameters are grouped by their top-level module name
        (the first component of the parameter name before the first dot).

    Returns
    -------
    dict of str -> dict of str -> float
        A nested dictionary mapping group names to their statistics::

            {
                "generator": {
                    "mean_norm": 0.042,
                    "max_norm": 0.87,
                    "fraction_zero": 0.0,
                },
                "image_encoder": {
                    "mean_norm": 0.015,
                    "max_norm": 0.32,
                    "fraction_zero": 0.05,
                },
                ...
            }

    Raises
    ------
    TypeError
        If *model* is not an ``nn.Module`` or *logger* is not a ``Logger``.

    Example
    -------
    >>> from ilgan.training.gradient_utils import log_gradient_statistics
    >>> stats = log_gradient_statistics(generator, logger, step=100)
    >>> print(stats["generator"]["mean_norm"])
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected 'model' to be an nn.Module, "
            f"got {type(model).__name__}."
        )
    if not isinstance(logger, Logger):
        raise TypeError(
            f"Expected 'logger' to be a Logger, "
            f"got {type(logger).__name__}."
        )

    # ── Build parameter name → gradient mapping ──────────────────────────
    # We iterate over all named parameters and collect those with gradients.
    param_grads: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_grads[name] = param.grad.detach()

    if len(param_grads) == 0:
        logger.warning(
            f"[Step {step}] No gradients found in model '{model.__class__.__name__}'."
        )
        return {}

    # ── Determine parameter groups ───────────────────────────────────────
    if param_groups is None:
        # Group by top-level module name (first component before '.')
        groups: Dict[str, List[str]] = {}
        for name in param_grads:
            top_level = name.split(".")[0]
            if top_level not in groups:
                groups[top_level] = []
            groups[top_level].append(name)
    else:
        # Use the provided groups: for each group, collect all parameter
        # names that start with any of the prefixes.
        groups = {}
        for group_name, prefixes in param_groups.items():
            matched = []
            for name in param_grads:
                if any(name.startswith(prefix) for prefix in prefixes):
                    matched.append(name)
            if matched:
                groups[group_name] = matched

    # ── Compute statistics per group ─────────────────────────────────────
    stats: Dict[str, Dict[str, float]] = {}

    for group_name, param_names in groups.items():
        if len(param_names) == 0:
            continue

        # Collect gradient norms for all parameters in this group
        grad_norms: List[float] = []
        zero_count: int = 0

        for pname in param_names:
            grad = param_grads[pname]
            # Compute L2 norm of this parameter's gradient
            p_norm = grad.norm(2).item()
            grad_norms.append(p_norm)
            if p_norm < _EPS:
                zero_count += 1

        if len(grad_norms) == 0:
            continue

        # Compute statistics
        mean_norm = float(torch.tensor(grad_norms).mean().item())
        max_norm = float(max(grad_norms))
        fraction_zero = zero_count / len(grad_norms)

        stats[group_name] = {
            "mean_norm": mean_norm,
            "max_norm": max_norm,
            "fraction_zero": fraction_zero,
        }

        # Log at INFO level
        logger.info(
            f"[Step {step}] Gradient stats — {group_name}: "
            f"mean_norm={mean_norm:.6f}, "
            f"max_norm={max_norm:.6f}, "
            f"zero_frac={fraction_zero:.4f}"
        )

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# detect_nan_inf_gradients
# ──────────────────────────────────────────────────────────────────────────────


def detect_nan_inf_gradients(
    model: nn.Module,
    logger: Optional[Logger] = None,
    step: Optional[int] = None,
) -> List[str]:
    """Check for NaN or Inf gradients across all model parameters.

    This function iterates over all parameters with non-None gradients and
    checks whether any gradient value is NaN (not a number) or Inf
    (infinity).  Such gradients indicate numerical instability and, if
    left unchecked, will corrupt the optimizer state and cause training
    to diverge.

    The function returns a list of parameter names (fully-qualified) that
    have problematic gradients.  If *logger* is provided, it logs a warning
    for each problematic parameter.

    Common causes of NaN/Inf gradients in ILGAN
    --------------------------------------------
    - **Attention softmax overflow**: the ``SpatialContentCrossAttention``
      module computes softmax over large logits, which can overflow in
      ``float16``.  This is mitigated by AMP but can still occur.
    - **Loss scale underflow/overflow**: when AMP is enabled, the loss
      scale factor may become too large or too small, causing gradients
      to underflow to zero or overflow to Inf.
    - **Division by zero**: any ``/ (x + eps)`` operation where ``x`` is
      very large negative can produce Inf.
    - **Sigmoid/Tanh saturation**: extreme input values to sigmoid or
      tanh can produce NaN due to floating-point precision limits.

    Parameters
    ----------
    model : nn.Module
        The model to check for problematic gradients.
    logger : Logger, optional
        If provided, warnings are logged for each problematic parameter.
    step : int, optional
        The current training step (used in log messages for context).
        Only used if *logger* is also provided.

    Returns
    -------
    list of str
        A list of fully-qualified parameter names (as returned by
        ``model.named_parameters()``) whose gradients contain NaN or Inf
        values.  An empty list indicates no problematic gradients were
        found.

    Raises
    ------
    TypeError
        If *model* is not an ``nn.Module``.

    Example
    -------
    >>> from ilgan.training.gradient_utils import detect_nan_inf_gradients
    >>> bad_params = detect_nan_inf_gradients(generator, logger, step=42)
    >>> if bad_params:
    ...     logger.error(f"NaN/Inf gradients detected in: {bad_params}")
    ...     # Optionally: skip this step, reset parameters, or reduce LR
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected 'model' to be an nn.Module, "
            f"got {type(model).__name__}."
        )

    # ── Check each parameter's gradient ──────────────────────────────────
    bad_params: List[str] = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()

        # Check for NaN or Inf
        has_nan = torch.isnan(grad).any().item()
        has_inf = torch.isinf(grad).any().item()

        if has_nan or has_inf:
            bad_params.append(name)

            if logger is not None:
                issue_type = "NaN" if has_nan else "Inf"
                step_str = f"[Step {step}] " if step is not None else ""
                logger.warning(
                    f"{step_str}Gradient issue detected — "
                    f"parameter '{name}' contains {issue_type} values. "
                    f"Shape: {list(grad.shape)}, "
                    f"dtype: {grad.dtype}, "
                    f"device: {grad.device}"
                )

    return bad_params


# ──────────────────────────────────────────────────────────────────────────────
# zero_gradients
# ──────────────────────────────────────────────────────────────────────────────


def zero_gradients(
    optimizers_list: Union[Optimizer, Sequence[Optimizer]],
    set_to_none: bool = True,
) -> None:
    """Zero gradients for all provided optimizers in a single call.

    This is a convenience function that calls ``optimizer.zero_grad()`` on
    each optimizer in the list.  By default, it uses ``set_to_none=True``,
    which sets gradients to ``None`` instead of zero tensors.  This is
    more memory-efficient (PyTorch will free the gradient memory) and
    slightly faster, but it means that any code that checks
    ``param.grad is not None`` will see ``None`` instead of a zero tensor.

    Parameters
    ----------
    optimizers_list : Optimizer or list of Optimizer
        One or more PyTorch optimizers whose gradients should be zeroed.
        A single optimizer (not wrapped in a list) is also accepted.
    set_to_none : bool, optional
        If ``True``, set gradients to ``None`` (more memory-efficient).
        If ``False``, set gradients to zero tensors (preserves the
        ``param.grad`` object).  (default: ``True``)

    Raises
    ------
    TypeError
        If any element in *optimizers_list* is not a ``torch.optim.Optimizer``.

    Example
    -------
    >>> from ilgan.training.gradient_utils import zero_gradients
    >>> # Single optimizer
    >>> zero_gradients(g_optimizer)
    >>> # Multiple optimizers
    >>> zero_gradients([g_optimizer, d_optimizer])
    """
    # ── Normalise to a list ──────────────────────────────────────────────
    if isinstance(optimizers_list, Optimizer):
        optimizers: List[Optimizer] = [optimizers_list]
    elif isinstance(optimizers_list, (list, tuple)):
        optimizers = list(optimizers_list)
    else:
        raise TypeError(
            f"Expected an Optimizer or a sequence of Optimizers, "
            f"got {type(optimizers_list).__name__}."
        )

    # ── Validate and zero each optimizer ─────────────────────────────────
    for opt in optimizers:
        if not isinstance(opt, Optimizer):
            raise TypeError(
                f"Expected all elements to be torch.optim.Optimizer instances, "
                f"got {type(opt).__name__}."
            )
        opt.zero_grad(set_to_none=set_to_none)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "clip_gradients",
    "log_gradient_statistics",
    "detect_nan_inf_gradients",
    "zero_gradients",
]
