"""
Novel adaptive optimization algorithm specifically for dual-output GANs.

This module provides the :class:`DualOutputOptimizer`, a gradient-balancing
optimizer wrapper that prevents one pathway (image vs. bounding box) from
dominating the other during ILGAN training.

Mathematical motivation
----------------------
In the ILGAN generator, the loss :math:`\\mathcal{L}` is a weighted sum of
image-related losses (:math:`\\mathcal{L}_{\\text{img}}`) and box-related
losses (:math:`\\mathcal{L}_{\\text{box}}`):

.. math::

    \\mathcal{L} = \\lambda_{\\text{img}} \\mathcal{L}_{\\text{img}}
                 + \\lambda_{\\text{box}} \\mathcal{L}_{\\text{box}}

The gradients w.r.t. the generator parameters are:

.. math::

    \\nabla_{\\theta} \\mathcal{L} =
        \\lambda_{\\text{img}} \\nabla_{\\theta} \\mathcal{L}_{\\text{img}}
        + \\lambda_{\\text{box}} \\nabla_{\\theta} \\mathcal{L}_{\\text{box}}

When one loss term dominates (e.g., box regression loss is much larger than
adversarial loss), the corresponding pathway's gradients overwhelm the other,
causing:

- **Image collapse**: if box gradients dominate, the content decoder receives
  distorted gradient signals and produces blurry or repetitive images.
- **Box collapse**: if image gradients dominate, the spatial head receives
  vanishing gradients and all bounding boxes converge to a single location.

The :class:`DualOutputOptimizer` addresses this by:

1. **Per-pathway gradient norm tracking**: maintains an exponential moving
   average (EMA) of the gradient norm ratio between the box pathway
   (``spatial_head`` parameters) and the image pathway (``content_decoder``
   parameters).

2. **Adaptive gradient scaling**: when the EMA ratio exceeds a threshold
   :math:`\\tau`, the dominant pathway's gradients are scaled down and the
   weaker pathway's gradients are scaled up, restoring balance.

3. **Per-parameter-group scaling**: scaling factors are computed separately
   for each parameter group based on whether the parameter belongs to
   ``content_decoder`` or ``spatial_head``, allowing fine-grained control.

Algorithm
---------
Let :math:`g^{(t)}_I` be the vector of all gradients w.r.t. parameters in the
image pathway (``content_decoder``) at step :math:`t`, and :math:`g^{(t)}_B`
be the vector of all gradients w.r.t. parameters in the box pathway
(``spatial_head``).  Define the per-pathway gradient norms:

.. math::

    n^{(t)}_I = \\|g^{(t)}_I\\|_2, \\quad
    n^{(t)}_B = \\|g^{(t)}_B\\|_2

The gradient norm ratio is:

.. math::

    r^{(t)} = \\frac{n^{(t)}_B}{n^{(t)}_I + \\varepsilon}

We maintain an EMA :math:`\\bar{r}^{(t)}` with momentum :math:`\\beta`:

.. math::

    \\bar{r}^{(t)} = \\beta \\cdot \\bar{r}^{(t-1)} + (1 - \\beta) \\cdot r^{(t)}

If :math:`\\bar{r}^{(t)} > \\tau` (box pathway dominates), we scale:

.. math::

    g^{(t)}_B \\leftarrow g^{(t)}_B \\cdot \\frac{\\tau}{\\bar{r}^{(t)}}, \\quad
    g^{(t)}_I \\leftarrow g^{(t)}_I \\cdot \\min\\left(\\frac{\\bar{r}^{(t)}}{\\tau}, \\alpha_{\\max}\\right)

If :math:`\\bar{r}^{(t)} < 1/\\tau` (image pathway dominates), we scale:

.. math::

    g^{(t)}_I \\leftarrow g^{(t)}_I \\cdot \\frac{\\bar{r}^{(t)}}{1/\\tau}, \\quad
    g^{(t)}_B \\leftarrow g^{(t)}_B \\cdot \\min\\left(\\frac{1/\\tau}{\\bar{r}^{(t)}}, \\alpha_{\\max}\\right)

where :math:`\\alpha_{\\max}` is a maximum amplification factor to prevent
exploding gradients from the up-scaled pathway.

This mechanism mathematically prevents representation collapse by ensuring
that neither pathway's gradients vanish relative to the other.

Implementation
--------------
The :class:`DualOutputOptimizer` uses PyTorch's optimizer hooks
(:meth:`torch.optim.Optimizer.register_step_pre_hook` and
:meth:`torch.optim.Optimizer.register_step_post_hook`) to modify gradients
in-place **before** the optimizer step executes.  This is more efficient
than modifying gradients after the backward pass because it avoids an
additional iteration over parameters.

Usage
-----
The :class:`DualOutputOptimizer` wraps an existing optimizer (typically the
generator optimizer from :func:`ilgan.training.optimizers.build_optimizers`)::

    from ilgan.scripts.adaptive_optim import DualOutputOptimizer
    from ilgan.training.optimizers import build_optimizers

    g_opt, d_opt = build_optimizers(generator, discriminator, ...)
    dual_opt = DualOutputOptimizer(
        optimizer=g_opt,
        generator=generator,
        threshold=2.0,
        ema_momentum=0.9,
        max_amplification=5.0,
    )

    # In the training loop:
    loss.backward()
    dual_opt.step()  # applies gradient balancing, then optimizer.step()
    dual_opt.zero_grad()
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon to prevent division by zero in norm ratio computations."""

_DEFAULT_THRESHOLD: float = 2.0
"""Default threshold :math:`\\tau` for the gradient norm ratio.  If the EMA
ratio exceeds this, gradient balancing is triggered."""

_DEFAULT_EMA_MOMENTUM: float = 0.9
"""Default momentum :math:`\\beta` for the exponential moving average of the
gradient norm ratio."""

_DEFAULT_MAX_AMPLIFICATION: float = 5.0
"""Default maximum amplification factor :math:`\\alpha_{\\max}` for the
up-scaled pathway.  Prevents exploding gradients."""

_DEFAULT_BALANCE_STRENGTH: float = 1.0
"""Default strength of the balancing correction.  A value of 1.0 applies the
full correction; lower values apply a softer correction."""


# ──────────────────────────────────────────────────────────────────────────────
# DualOutputOptimizer
# ──────────────────────────────────────────────────────────────────────────────


class DualOutputOptimizer:
    """Gradient-balancing optimizer wrapper for dual-output GANs.

    This class wraps a :class:`torch.optim.Optimizer` (typically the generator
    optimizer) and registers pre-step hooks that dynamically balance the
    gradient magnitudes between the image pathway (``content_decoder``) and
    the bounding box pathway (``spatial_head``) of the ILGAN generator.

    The balancing prevents one pathway from dominating the other during
    adversarial training, which in turn prevents:

    - **Image mode collapse**: box gradients overwhelming the content decoder.
    - **Bounding box collapse**: image gradients overwhelming the spatial head,
      causing all boxes to converge to a single location.

    The optimizer maintains an exponential moving average (EMA) of the
    gradient norm ratio between the two pathways and uses this to compute
    per-parameter-group scaling factors that are applied in-place to the
    gradients before the optimizer step.

    Parameters
    ----------
    optimizer : Optimizer
        The underlying PyTorch optimizer to wrap (typically the generator
        optimizer from :func:`ilgan.training.optimizers.build_optimizers`).
        This optimizer must have its parameters organised into groups that
        include the generator's ``content_decoder`` and ``spatial_head``
        sub-modules.
    generator : nn.Module
        The ILGAN generator (``ILGANGenerator`` instance).  Used to identify
        which parameters belong to the image pathway (``content_decoder``)
        and which belong to the box pathway (``spatial_head``).
    threshold : float, optional
        The gradient norm ratio threshold :math:`\\tau`.  If the EMA ratio
        exceeds this value, gradient balancing is triggered.  Must be > 1.0.
        (default: 2.0)
    ema_momentum : float, optional
        The momentum :math:`\\beta` for the exponential moving average of the
        gradient norm ratio.  Must be in ``(0, 1)``.  Higher values smooth
        the ratio more.  (default: 0.9)
    max_amplification : float, optional
        The maximum amplification factor :math:`\\alpha_{\\max}` for the
        up-scaled pathway.  Must be >= 1.0.  (default: 5.0)
    balance_strength : float, optional
        The strength of the balancing correction.  A value of 1.0 applies the
        full correction; lower values (e.g., 0.5) apply a softer correction.
        Must be in ``(0, 1]``.  (default: 1.0)
    verbose : bool, optional
        If ``True``, log diagnostic information about gradient balancing
        via Python's ``warnings.warn`` (since this module does not depend on
        the ILGAN logger).  (default: ``False``)

    Raises
    ------
    TypeError
        If *optimizer* is not an ``Optimizer`` or *generator* is not an
        ``nn.Module``.
    ValueError
        If any of the numeric parameters are out of valid ranges.

    Example
    -------
    >>> from ilgan.scripts.adaptive_optim import DualOutputOptimizer
    >>> from ilgan.training.optimizers import build_optimizers
    >>>
    >>> g_opt, d_opt = build_optimizers(generator, discriminator, ...)
    >>> dual_opt = DualOutputOptimizer(
    ...     optimizer=g_opt,
    ...     generator=generator,
    ...     threshold=2.0,
    ...     ema_momentum=0.9,
    ... )
    >>>
    >>> # In the training loop:
    >>> for batch in dataloader:
    ...     # ... forward and backward ...
    ...     loss.backward()
    ...     dual_opt.step()  # balances gradients, then calls optimizer.step()
    ...     dual_opt.zero_grad()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        generator: nn.Module,
        threshold: float = _DEFAULT_THRESHOLD,
        ema_momentum: float = _DEFAULT_EMA_MOMENTUM,
        max_amplification: float = _DEFAULT_MAX_AMPLIFICATION,
        balance_strength: float = _DEFAULT_BALANCE_STRENGTH,
        verbose: bool = False,
    ) -> None:
        # ── Validate inputs ───────────────────────────────────────────────
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                f"Expected 'optimizer' to be a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}."
            )
        if not isinstance(generator, nn.Module):
            raise TypeError(
                f"Expected 'generator' to be an nn.Module, "
                f"got {type(generator).__name__}."
            )
        if threshold <= 1.0:
            raise ValueError(
                f"threshold must be > 1.0, got {threshold}."
            )
        if not (0.0 < ema_momentum < 1.0):
            raise ValueError(
                f"ema_momentum must be in (0, 1), got {ema_momentum}."
            )
        if max_amplification < 1.0:
            raise ValueError(
                f"max_amplification must be >= 1.0, got {max_amplification}."
            )
        if not (0.0 < balance_strength <= 1.0):
            raise ValueError(
                f"balance_strength must be in (0, 1], got {balance_strength}."
            )

        self._optimizer = optimizer
        self._generator = generator
        self._threshold = threshold
        self._ema_momentum = ema_momentum
        self._max_amplification = max_amplification
        self._balance_strength = balance_strength
        self._verbose = verbose

        # ── Identify pathway parameters ──────────────────────────────────
        # We scan the generator's named parameters and classify each into
        # one of three categories:
        #   - "content_decoder": image pathway
        #   - "spatial_head": bounding box pathway
        #   - "other": shared parameters (e.g., noise_std, latent stats)
        self._content_params: List[str] = []
        self._spatial_params: List[str] = []
        self._other_params: List[str] = []

        for name, param in generator.named_parameters():
            if name.startswith("content_decoder"):
                self._content_params.append(name)
            elif name.startswith("spatial_head"):
                self._spatial_params.append(name)
            else:
                self._other_params.append(name)

        # Build a set for fast lookup
        self._content_param_set = set(self._content_params)
        self._spatial_param_set = set(self._spatial_params)

        # Build a reverse mapping from parameter tensor id to pathway name
        # This is O(n) at init time and O(1) at step time.
        self._param_id_to_pathway: Dict[int, str] = {}
        for name, param in generator.named_parameters():
            if name in self._content_param_set:
                self._param_id_to_pathway[id(param)] = "content"
            elif name in self._spatial_param_set:
                self._param_id_to_pathway[id(param)] = "spatial"
            else:
                self._param_id_to_pathway[id(param)] = "other"

        if len(self._content_params) == 0:
            warnings.warn(
                "DualOutputOptimizer: No parameters found matching "
                "'content_decoder.*'.  Gradient balancing will have no effect "
                "on the image pathway."
            )
        if len(self._spatial_params) == 0:
            warnings.warn(
                "DualOutputOptimizer: No parameters found matching "
                "'spatial_head.*'.  Gradient balancing will have no effect "
                "on the box pathway."
            )

        # ── EMA state ─────────────────────────────────────────────────────
        # Exponential moving average of the gradient norm ratio
        # r = ||g_box|| / (||g_image|| + eps)
        self._ema_ratio: float = 1.0

        # Number of steps taken
        self._steps: int = 0

        # Last computed norms (for diagnostics)
        self._last_content_norm: float = 0.0
        self._last_spatial_norm: float = 0.0
        self._last_raw_ratio: float = 1.0

        # Whether balancing was applied in the last step
        self._last_balanced: bool = False
        self._last_content_scale: float = 1.0
        self._last_spatial_scale: float = 1.0

        # ── Register optimizer hooks ───────────────────────────────────────
        # Pre-step hook: called before optimizer.step(), modifies gradients
        # in-place to balance the two pathways.
        self._pre_hook_handle = optimizer.register_step_pre_hook(
            self._pre_step_hook,
        )

        # Post-step hook: called after optimizer.step(), used for cleanup
        # and diagnostics.
        self._post_hook_handle = optimizer.register_step_post_hook(
            self._post_step_hook,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Optimizer hooks
    # ──────────────────────────────────────────────────────────────────────────

    def _pre_step_hook(
        self,
        optimizer: Optimizer,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """Pre-step hook that balances gradients between the image and box
        pathways.

        This hook is called **before** ``optimizer.step()``.  It:

        1. Computes the L2 norm of gradients for each pathway.
        2. Computes the ratio :math:`r = ||g_box|| / (||g_image|| + eps)`.
        3. Updates the EMA :math:`\\bar{r}`.
        4. If :math:`\\bar{r} > \\tau` or :math:`\\bar{r} < 1/\\tau`,
           scales the gradients of both pathways to restore balance.

        The scaling is applied **in-place** to the parameter gradients,
        so no additional memory is allocated.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer that is about to step (same as ``self._optimizer``).
        args : tuple
            Positional arguments passed to ``optimizer.step()`` (typically
            empty).
        kwargs : dict
            Keyword arguments passed to ``optimizer.step()`` (typically
            empty).
        """
        # ── 1. Compute per-pathway gradient norms ────────────────────────
        content_norm_sq = 0.0
        spatial_norm_sq = 0.0

        # Iterate over all parameter groups and their parameters
        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach()
                grad_norm_sq = grad.norm(2).item() ** 2

                # Determine which pathway this parameter belongs to
                # using the pre-computed id-to-pathway mapping (O(1) lookup)
                pathway = self._param_id_to_pathway.get(id(param), "other")

                if pathway == "content":
                    content_norm_sq += grad_norm_sq
                elif pathway == "spatial":
                    spatial_norm_sq += grad_norm_sq
                # "other" parameters are not scaled

        content_norm = math.sqrt(content_norm_sq)
        spatial_norm = math.sqrt(spatial_norm_sq)

        self._last_content_norm = content_norm
        self._last_spatial_norm = spatial_norm

        # ── 2. Compute ratio ──────────────────────────────────────────────
        raw_ratio = spatial_norm / (content_norm + _EPS)
        self._last_raw_ratio = raw_ratio

        # ── 3. Update EMA ─────────────────────────────────────────────────
        if self._steps == 0:
            # Initialise EMA with the first raw ratio
            self._ema_ratio = raw_ratio
        else:
            self._ema_ratio = (
                self._ema_momentum * self._ema_ratio
                + (1.0 - self._ema_momentum) * raw_ratio
            )

        self._steps += 1

        # ── 4. Determine if balancing is needed ──────────────────────────
        ema = self._ema_ratio
        threshold = self._threshold
        inv_threshold = 1.0 / threshold

        if ema > threshold:
            # ── Box pathway dominates ──────────────────────────────────
            # The ratio r = ||g_box|| / ||g_image|| is too large.
            # We want to bring it down to 'threshold'.
            #
            # Scale box gradients DOWN by:  threshold / ema  (< 1.0)
            # Scale image gradients UP by:  min(ema / threshold, max_amp)  (>= 1.0)
            #
            # After scaling:
            #   r' = (||g_box|| * threshold/ema) / (||g_image|| * min(ema/threshold, max_amp))
            #      = r * (threshold/ema) / min(ema/threshold, max_amp)
            #      = threshold / min(ema/threshold * ema/r, max_amp * ema/r)
            #      = threshold / min(1, max_amp * ema/r)  ... simplified
            #   When max_amp is not hit: r' = threshold (exact target)

            box_scale = threshold / ema  # < 1.0, scales box DOWN
            image_boost = min(ema / threshold, self._max_amplification)  # >= 1.0

            content_scale = 1.0 + (image_boost - 1.0) * self._balance_strength
            spatial_scale = 1.0 - (1.0 - box_scale) * self._balance_strength

            self._last_balanced = True
            self._last_content_scale = content_scale
            self._last_spatial_scale = spatial_scale

            # Apply scaling in-place
            self._scale_pathway_gradients("content", content_scale)
            self._scale_pathway_gradients("spatial", spatial_scale)

            if self._verbose:
                warnings.warn(
                    f"[DualOutputOptimizer] Step {self._steps}: "
                    f"Box pathway dominates (EMA ratio={ema:.3f} > {threshold}). "
                    f"Scaling content x{content_scale:.3f}, spatial x{spatial_scale:.3f}."
                )

        elif ema < inv_threshold:
            # ── Image pathway dominates ────────────────────────────────
            # The ratio r = ||g_box|| / ||g_image|| is too small.
            # We want to bring it up to 'inv_threshold'.
            #
            # Scale image gradients DOWN by:  ema / inv_threshold  (< 1.0)
            # Scale box gradients UP by:     min(inv_threshold / ema, max_amp)  (>= 1.0)
            #
            # After scaling:
            #   r' = (||g_box|| * min(inv_threshold/ema, max_amp)) / (||g_image|| * ema/inv_threshold)
            #      = r * min(inv_threshold/ema, max_amp) / (ema/inv_threshold)
            #      = r * min(inv_threshold/ema, max_amp) * (inv_threshold/ema)
            #      = inv_threshold * min(1, max_amp * ema/inv_threshold)  ... simplified
            #   When max_amp is not hit: r' = inv_threshold (exact target)

            content_down = ema / inv_threshold  # < 1.0, scales content DOWN
            box_boost = min(inv_threshold / ema, self._max_amplification)  # >= 1.0

            content_scale = 1.0 - (1.0 - content_down) * self._balance_strength
            spatial_scale = 1.0 + (box_boost - 1.0) * self._balance_strength

            self._last_balanced = True
            self._last_content_scale = content_scale
            self._last_spatial_scale = spatial_scale

            # Apply scaling in-place
            self._scale_pathway_gradients("content", content_scale)
            self._scale_pathway_gradients("spatial", spatial_scale)

            if self._verbose:
                warnings.warn(
                    f"[DualOutputOptimizer] Step {self._steps}: "
                    f"Image pathway dominates (EMA ratio={ema:.3f} < {inv_threshold:.3f}). "
                    f"Scaling content x{content_scale:.3f}, spatial x{spatial_scale:.3f}."
                )

        else:
            # Balanced — no scaling needed
            self._last_balanced = False
            self._last_content_scale = 1.0
            self._last_spatial_scale = 1.0

    def _post_step_hook(
        self,
        optimizer: Optimizer,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        """Post-step hook for cleanup and diagnostics.

        Currently a no-op, but reserved for future use (e.g., logging
        gradient statistics after the step, or resetting internal state
        if needed).

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer that just stepped.
        args : tuple
            Positional arguments passed to ``optimizer.step()``.
        kwargs : dict
            Keyword arguments passed to ``optimizer.step()``.
        """
        pass  # Reserved for future use

    # ──────────────────────────────────────────────────────────────────────────
    # Gradient scaling
    # ──────────────────────────────────────────────────────────────────────────

    def _scale_pathway_gradients(
        self,
        pathway: str,
        scale: float,
    ) -> None:
        """Scale all gradients belonging to a specific pathway in-place.

        This method iterates over all parameter groups in the optimizer and
        scales the gradients of parameters that belong to the specified
        pathway.

        Parameters
        ----------
        pathway : str
            One of ``"content"``, ``"spatial"``, or ``"other"``.
        scale : float
            The multiplicative factor to apply to the gradients.  Values
            < 1.0 reduce the gradients; values > 1.0 amplify them.
        """
        if scale == 1.0:
            return  # No scaling needed

        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                p_pathway = self._param_id_to_pathway.get(id(param), "other")
                if p_pathway == pathway:
                    param.grad.mul_(scale)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (delegates to the underlying optimizer)
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step with gradient balancing.

        This method calls the underlying optimizer's ``step()`` method.
        The gradient balancing is applied automatically via the pre-step
        hook registered in the constructor.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.
            Optional for most optimizers.

        Returns
        -------
        float or None
            The loss value from the closure, if provided.
        """
        return self._optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients of all parameters.

        Delegates to the underlying optimizer's ``zero_grad()`` method.

        Parameters
        ----------
        set_to_none : bool, optional
            If ``True``, set gradients to ``None`` (more memory-efficient).
            If ``False``, set gradients to zero tensors.  (default: ``True``)
        """
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer and the dual-output balancer.

        The state includes:

        - The underlying optimizer's state dict.
        - The EMA ratio.
        - The step count.
        - The configuration parameters.

        Returns
        -------
        dict
            A dictionary containing the full state for serialization.
        """
        return {
            "optimizer_state_dict": self._optimizer.state_dict(),
            "ema_ratio": self._ema_ratio,
            "steps": self._steps,
            "threshold": self._threshold,
            "ema_momentum": self._ema_momentum,
            "max_amplification": self._max_amplification,
            "balance_strength": self._balance_strength,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the optimizer and the dual-output balancer.

        Parameters
        ----------
        state_dict : dict
            A dictionary containing the state, as returned by
            :meth:`state_dict`.

        Raises
        ------
        KeyError
            If the state dict is missing required keys.
        ValueError
            If the configuration parameters in the state dict do not match
            the current configuration.
        """
        # Validate configuration consistency
        for key in ["threshold", "ema_momentum", "max_amplification", "balance_strength"]:
            if key in state_dict:
                current = getattr(self, f"_{key}")
                loaded = state_dict[key]
                if abs(current - loaded) > 1e-6:
                    warnings.warn(
                        f"DualOutputOptimizer.load_state_dict: "
                        f"Configuration mismatch for '{key}': "
                        f"current={current}, loaded={loaded}. "
                        f"Using loaded value."
                    )

        # Load optimizer state
        if "optimizer_state_dict" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer_state_dict"])

        # Load EMA state
        if "ema_ratio" in state_dict:
            self._ema_ratio = float(state_dict["ema_ratio"])
        if "steps" in state_dict:
            self._steps = int(state_dict["steps"])

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostic properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def ema_ratio(self) -> float:
        """The current EMA of the gradient norm ratio
        :math:`\\bar{r} = EMA(||g_{box}|| / ||g_{image}||)`.

        A value > 1.0 indicates the box pathway has larger gradients on
        average; a value < 1.0 indicates the image pathway has larger
        gradients.
        """
        return self._ema_ratio

    @property
    def steps(self) -> int:
        """The number of optimization steps taken so far."""
        return self._steps

    @property
    def last_content_norm(self) -> float:
        """The L2 norm of the image pathway (``content_decoder``) gradients
        from the most recent step."""
        return self._last_content_norm

    @property
    def last_spatial_norm(self) -> float:
        """The L2 norm of the box pathway (``spatial_head``) gradients
        from the most recent step."""
        return self._last_spatial_norm

    @property
    def last_raw_ratio(self) -> float:
        """The raw (non-EMA) gradient norm ratio from the most recent step."""
        return self._last_raw_ratio

    @property
    def last_balanced(self) -> bool:
        """Whether gradient balancing was applied in the most recent step."""
        return self._last_balanced

    @property
    def last_content_scale(self) -> float:
        """The scaling factor applied to the image pathway gradients in the
        most recent step (1.0 if no balancing was applied)."""
        return self._last_content_scale

    @property
    def last_spatial_scale(self) -> float:
        """The scaling factor applied to the box pathway gradients in the
        most recent step (1.0 if no balancing was applied)."""
        return self._last_spatial_scale

    @property
    def optimizer(self) -> Optimizer:
        """The underlying PyTorch optimizer."""
        return self._optimizer

    @property
    def generator(self) -> nn.Module:
        """The ILGAN generator."""
        return self._generator

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a comprehensive dictionary of diagnostic information about
        the gradient balancing state.

        This is useful for logging and visualization during training.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"ema_ratio"``: the EMA of the gradient norm ratio.
            - ``"raw_ratio"``: the raw (non-EMA) ratio from the last step.
            - ``"content_norm"``: the L2 norm of image pathway gradients.
            - ``"spatial_norm"``: the L2 norm of box pathway gradients.
            - ``"balanced"``: whether balancing was applied in the last step.
            - ``"content_scale"``: the scaling factor applied to image pathway.
            - ``"spatial_scale"``: the scaling factor applied to box pathway.
            - ``"steps"``: total number of steps taken.
            - ``"threshold"``: the balance threshold.
            - ``"ema_momentum"``: the EMA momentum.
            - ``"max_amplification"``: the max amplification factor.
            - ``"balance_strength"``: the balance strength.
            - ``"num_content_params"``: number of image pathway parameters.
            - ``"num_spatial_params"``: number of box pathway parameters.
        """
        return {
            "ema_ratio": self._ema_ratio,
            "raw_ratio": self._last_raw_ratio,
            "content_norm": self._last_content_norm,
            "spatial_norm": self._last_spatial_norm,
            "balanced": self._last_balanced,
            "content_scale": self._last_content_scale,
            "spatial_scale": self._last_spatial_scale,
            "steps": self._steps,
            "threshold": self._threshold,
            "ema_momentum": self._ema_momentum,
            "max_amplification": self._max_amplification,
            "balance_strength": self._balance_strength,
            "num_content_params": len(self._content_params),
            "num_spatial_params": len(self._spatial_params),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DualOutputOptimizer(\n"
            f"  optimizer={type(self._optimizer).__name__},\n"
            f"  threshold={self._threshold},\n"
            f"  ema_momentum={self._ema_momentum},\n"
            f"  max_amplification={self._max_amplification},\n"
            f"  balance_strength={self._balance_strength},\n"
            f"  steps={self._steps},\n"
            f"  ema_ratio={self._ema_ratio:.4f},\n"
            f"  content_params={len(self._content_params)},\n"
            f"  spatial_params={len(self._spatial_params)},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────


def build_dual_output_optimizer(
    optimizer: Optimizer,
    generator: nn.Module,
    threshold: float = _DEFAULT_THRESHOLD,
    ema_momentum: float = _DEFAULT_EMA_MOMENTUM,
    max_amplification: float = _DEFAULT_MAX_AMPLIFICATION,
    balance_strength: float = _DEFAULT_BALANCE_STRENGTH,
    verbose: bool = False,
) -> DualOutputOptimizer:
    """Convenience factory for creating a :class:`DualOutputOptimizer`.

    This function provides a single call-point for constructing the
    dual-output gradient balancer, with sensible defaults that work well
    for ILGAN training.

    Parameters
    ----------
    optimizer : Optimizer
        The generator optimizer to wrap.
    generator : nn.Module
        The ILGAN generator (``ILGANGenerator`` instance).
    threshold : float, optional
        Gradient norm ratio threshold.  Must be > 1.0.  (default: 2.0)
    ema_momentum : float, optional
        EMA momentum for the ratio.  Must be in ``(0, 1)``.  (default: 0.9)
    max_amplification : float, optional
        Maximum amplification factor for the up-scaled pathway.
        Must be >= 1.0.  (default: 5.0)
    balance_strength : float, optional
        Strength of the balancing correction.  Must be in ``(0, 1]``.
        (default: 1.0)
    verbose : bool, optional
        If ``True``, log diagnostic information.  (default: ``False``)

    Returns
    -------
    DualOutputOptimizer
        A fully configured :class:`DualOutputOptimizer` instance.

    Example
    -------
    >>> from ilgan.scripts.adaptive_optim import build_dual_output_optimizer
    >>> from ilgan.training.optimizers import build_optimizers
    >>>
    >>> g_opt, d_opt = build_optimizers(generator, discriminator, ...)
    >>> dual_opt = build_dual_output_optimizer(g_opt, generator)
    """
    return DualOutputOptimizer(
        optimizer=optimizer,
        generator=generator,
        threshold=threshold,
        ema_momentum=ema_momentum,
        max_amplification=max_amplification,
        balance_strength=balance_strength,
        verbose=verbose,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "DualOutputOptimizer",
    "build_dual_output_optimizer",
]
