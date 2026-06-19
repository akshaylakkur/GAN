"""
Memory profiling utilities for ILGAN.

This module provides three core functions for understanding and optimising
GPU memory usage during ILGAN training:

1. :func:`profile_model_memory` — runs a forward and backward pass through
   a model and measures per-component CUDA memory allocation, peak memory,
   and reserved memory.  Returns a structured dictionary of results.

2. :func:`optimize_memory_usage` — analyzes a configuration and model sizes
   to recommend optimal settings for batch size, gradient accumulation steps,
   gradient checkpointing, and mixed precision.  Prints a human-readable
   report with estimated memory usage.

3. :func:`estimate_vram_requirements` — estimates the VRAM required for a
   given configuration, helping users choose appropriate settings before
   launching a training run.

Mathematical motivation
-----------------------
The total VRAM usage during ILGAN training can be decomposed into four
components:

.. math::

    M_{\\text{total}} = M_{\\text{params}} + M_{\\text{grads}} + M_{\\text{opt}}
                       + M_{\\text{activations}}

where:

- :math:`M_{\\text{params}} = \\sum_i P_i \\cdot s_i` is the memory for
  model parameters, with :math:`P_i` the number of parameters in module
  :math:`i` and :math:`s_i` the byte size per parameter (4 for float32,
  2 for float16).
- :math:`M_{\\text{grads}} = M_{\\text{params}}` (gradients have the same
  shape as parameters).
- :math:`M_{\\text{opt}} = 2 \\cdot M_{\\text{params}}` for Adam (stores
  ``exp_avg`` and ``exp_avg_sq``, each the same size as parameters).
- :math:`M_{\\text{activations}}` depends on the batch size, image size,
  model depth, and whether gradient checkpointing is enabled.

With gradient checkpointing, activation memory is reduced by approximately
:math:`\\sqrt{L}` where :math:`L` is the number of layers, because only a
subset of activations are stored and the rest are recomputed during the
backward pass.

With mixed precision (AMP), parameters and activations are stored in
float16, reducing :math:`M_{\\text{params}}` and
:math:`M_{\\text{activations}}` by approximately 2×.

Usage
-----
::

    from ilgan.utils.config import Config
    from ilgan.scripts.profile_memory import (
        profile_model_memory,
        optimize_memory_usage,
        estimate_vram_requirements,
    )

    cfg = Config()

    # Estimate VRAM for a given config
    estimate = estimate_vram_requirements(cfg)
    print(f"Estimated VRAM: {estimate['total_mb']:.0f} MB")

    # Get optimisation recommendations
    optimize_memory_usage(cfg)

    # Profile a specific model
    from ilgan.models import ILGANGenerator
    model = ILGANGenerator(cfg).cuda()
    input_tensor = torch.randn(4, cfg.model.latent_dim).cuda()
    profile = profile_model_memory(model, input_tensor)
    print(f"Peak memory: {profile['peak_mb']:.2f} MB")
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BYTES_PER_MB: float = 1024.0 * 1024.0
"""Conversion factor from bytes to megabytes."""

_BYTES_PER_GB: float = 1024.0 * 1024.0 * 1024.0
"""Conversion factor from bytes to gigabytes."""

_FP32_BYTES: int = 4
"""Number of bytes per float32 parameter."""

_FP16_BYTES: int = 2
"""Number of bytes per float16 parameter."""

_ADAM_OPT_STATES: int = 2
"""Number of optimizer states per parameter for Adam (exp_avg, exp_avg_sq)."""

_ACTIVATION_FACTOR: float = 3.0
"""Empirical factor relating parameter memory to activation memory for
convolutional GAN architectures.  This is a rough heuristic based on
typical ILGAN configurations."""

_SAFETY_MARGIN: float = 1.2
"""Safety margin factor applied to all VRAM estimates to account for
PyTorch's CUDA caching allocator overhead and temporary tensors."""

_DEFAULT_IMAGE_SIZE: int = 128
"""Default image size used when not specified in config."""

_DEFAULT_BATCH_SIZE: int = 16
"""Default batch size used when not specified in config."""

_DEFAULT_LATENT_DIM: int = 256
"""Default latent dimension used when not specified in config."""

_DEFAULT_GEN_BASE_CHANNELS: int = 64
"""Default generator base channels used when not specified in config."""

_DEFAULT_DISC_BASE_CHANNELS: int = 64
"""Default discriminator base channels used when not specified in config."""

_DEFAULT_MAX_BOXES: int = 20
"""Default maximum boxes used when not specified in config."""

_DEFAULT_NUM_CLASSES: int = 80
"""Default number of classes used when not specified in config."""

_DEFAULT_NUM_ATTENTION_HEADS: int = 8
"""Default number of attention heads used when not specified in config."""


# ──────────────────────────────────────────────────────────────────────────────
# Helper: format bytes
# ──────────────────────────────────────────────────────────────────────────────


def _format_bytes(num_bytes: float) -> str:
    """Format a byte count into a human-readable string.

    Parameters
    ----------
    num_bytes : float
        Number of bytes.

    Returns
    -------
    str
        Human-readable string (e.g., ``"8.00 GiB"``, ``"512.00 MiB"``).
    """
    if num_bytes >= 1024.0 ** 3:
        return f"{num_bytes / (1024.0 ** 3):.2f} GiB"
    elif num_bytes >= 1024.0 ** 2:
        return f"{num_bytes / (1024.0 ** 2):.2f} MiB"
    elif num_bytes >= 1024.0:
        return f"{num_bytes / 1024.0:.2f} KiB"
    else:
        return f"{num_bytes:.0f} B"


def _format_mb(mb: float) -> str:
    """Format a megabyte value into a human-readable string.

    Parameters
    ----------
    mb : float
        Value in megabytes.

    Returns
    -------
    str
        Human-readable string (e.g., ``"2.50 GiB"``, ``"512.00 MiB"``).
    """
    return _format_bytes(mb * _BYTES_PER_MB)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: count parameters
# ──────────────────────────────────────────────────────────────────────────────


def _count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to analyze.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"total"``: total number of parameters.
        - ``"trainable"``: number of trainable (requires_grad=True) parameters.
        - ``"frozen"``: number of frozen (requires_grad=False) parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def _count_parameters_by_module(
    model: nn.Module,
    prefix: str = "",
) -> Dict[str, int]:
    """Count parameters recursively by module name.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to analyze.
    prefix : str, optional
        Prefix for module names (used internally for recursion).

    Returns
    -------
    dict of str -> int
        A dictionary mapping module names to their parameter counts.
    """
    result: Dict[str, int] = {}
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        child_params = sum(p.numel() for p in child.parameters())
        result[full_name] = child_params
        # Recurse into sub-modules
        sub_counts = _count_parameters_by_module(child, prefix=full_name)
        for sub_name, sub_count in sub_counts.items():
            # Only add if it's a leaf module (has no further children with params)
            if sub_count > 0 and sub_name not in result:
                result[sub_name] = sub_count
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Helper: estimate parameter memory
# ──────────────────────────────────────────────────────────────────────────────


def _estimate_parameter_memory(
    num_params: int,
    use_mixed_precision: bool = False,
) -> float:
    """Estimate the memory required to store model parameters.

    Parameters
    ----------
    num_params : int
        Number of parameters.
    use_mixed_precision : bool, optional
        If True, parameters are stored in float16 (2 bytes each).
        If False, parameters are stored in float32 (4 bytes each).
        (default: False)

    Returns
    -------
    float
        Estimated memory in megabytes.
    """
    bytes_per_param = _FP16_BYTES if use_mixed_precision else _FP32_BYTES
    return (num_params * bytes_per_param) / _BYTES_PER_MB


def _estimate_gradient_memory(
    num_params: int,
    use_mixed_precision: bool = False,
) -> float:
    """Estimate the memory required to store gradients.

    Gradients are always stored in float32 (the parameter's original
    precision) regardless of whether mixed precision is used, because
    the ``GradScaler`` unscales gradients to float32 before the optimizer
    step.

    Parameters
    ----------
    num_params : int
        Number of parameters.
    use_mixed_precision : bool, optional
        Ignored — gradients are always float32.  (default: False)

    Returns
    -------
    float
        Estimated memory in megabytes.
    """
    # Gradients are always stored in float32
    return (num_params * _FP32_BYTES) / _BYTES_PER_MB


def _estimate_optimizer_memory(
    num_params: int,
    optimizer_type: str = "adam",
    use_mixed_precision: bool = False,
) -> float:
    """Estimate the memory required for optimizer states.

    For Adam, each parameter has two optimizer states (``exp_avg`` and
    ``exp_avg_sq``), each stored in float32 regardless of mixed precision.

    For SGD with momentum, each parameter has one state (momentum buffer),
    stored in float32.

    Parameters
    ----------
    num_params : int
        Number of parameters.
    optimizer_type : str, optional
        Type of optimizer.  One of ``"adam"``, ``"sgd"``, ``"adamw"``.
        (default: ``"adam"``)
    use_mixed_precision : bool, optional
        Ignored — optimizer states are always float32.  (default: False)

    Returns
    -------
    float
        Estimated memory in megabytes.

    Raises
    ------
    ValueError
        If *optimizer_type* is unknown.
    """
    if optimizer_type.lower() in ("adam", "adamw"):
        num_states = _ADAM_OPT_STATES  # exp_avg + exp_avg_sq
    elif optimizer_type.lower() == "sgd":
        num_states = 1  # momentum buffer
    else:
        raise ValueError(
            f"Unknown optimizer type '{optimizer_type}'. "
            f"Expected one of: adam, adamw, sgd."
        )

    # Optimizer states are always float32
    return (num_params * num_states * _FP32_BYTES) / _BYTES_PER_MB


def _estimate_activation_memory(
    batch_size: int,
    image_size: int,
    gen_base_channels: int,
    disc_base_channels: int,
    num_blocks_gen: int,
    num_blocks_disc: int,
    max_boxes: int,
    num_classes: int,
    num_attention_heads: int,
    use_mixed_precision: bool = False,
    use_checkpointing: bool = False,
) -> float:
    """Estimate the memory required for intermediate activations during
    a forward+backward pass.

    This is a heuristic estimate based on the ILGAN architecture.  The
    activation memory is dominated by:

    1. **Generator activations**: feature maps at each resolution level
       in the ``ContentDecoder`` and attention maps in the ``SpatialHead``.
    2. **Discriminator activations**: feature maps at each resolution
       level in the ``ImageDiscriminator``.
    3. **Loss computation activations**: intermediate tensors for GIoU,
       gradient penalty, consistency, etc.

    The estimate uses the following model:

    .. math::

        M_{\\text{act}} = M_{\\text{gen}} + M_{\\text{disc}} + M_{\\text{loss}}

    where each component is estimated by summing the feature map sizes
    at each resolution level, multiplied by the batch size and bytes per
    element.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.
    image_size : int
        Spatial size of images (square).
    gen_base_channels : int
        Base channel count for the generator.
    disc_base_channels : int
        Base channel count for the discriminator.
    num_blocks_gen : int
        Number of up-blocks in the generator (``log2(image_size) - 2``).
    num_blocks_disc : int
        Number of down-blocks in the discriminator (``log2(image_size) - 2``).
    max_boxes : int
        Maximum number of bounding boxes per image.
    num_classes : int
        Number of object classes.
    num_attention_heads : int
        Number of attention heads in SCCA modules.
    use_mixed_precision : bool, optional
        If True, activations are stored in float16.  (default: False)
    use_checkpointing : bool, optional
        If True, gradient checkpointing reduces activation memory by
        approximately ``sqrt(L)`` where ``L`` is the number of layers.
        (default: False)

    Returns
    -------
    float
        Estimated activation memory in megabytes.
    """
    bytes_per_elem = _FP16_BYTES if use_mixed_precision else _FP32_BYTES

    # ── 1. Generator activation memory ──────────────────────────────────
    # The ContentDecoder has num_blocks_gen up-blocks, each producing a
    # feature map at 2× the previous resolution.  The channel count starts
    # at gen_base_channels * 16 and halves each block until gen_base_channels.
    gen_activation_mb = 0.0
    channels = gen_base_channels * 16
    spatial = 4  # Starting at 4×4

    for block_idx in range(num_blocks_gen):
        # After upsampling, spatial size doubles
        spatial *= 2
        # Channel count halves (but not below gen_base_channels)
        channels = max(channels // 2, gen_base_channels)

        # Feature map size: [B, C, H, W]
        feat_map_bytes = (
            batch_size * channels * spatial * spatial * bytes_per_elem
        )
        gen_activation_mb += feat_map_bytes / _BYTES_PER_MB

        # Each UpBlock also produces a skip feature of the same size
        gen_activation_mb += feat_map_bytes / _BYTES_PER_MB

    # The SpatialHead processes each resolution level with cross-attention.
    # The attention maps are [B, max_boxes, H, W] at each level.
    spatial = 4
    for block_idx in range(num_blocks_gen):
        spatial *= 2
        attn_map_bytes = (
            batch_size * max_boxes * spatial * spatial * bytes_per_elem
        )
        gen_activation_mb += attn_map_bytes / _BYTES_PER_MB

    # ── 2. Discriminator activation memory ──────────────────────────────
    disc_activation_mb = 0.0
    channels = disc_base_channels
    spatial = image_size

    for block_idx in range(num_blocks_disc):
        # Feature map before down-sampling: [B, C, H, W]
        feat_map_bytes = (
            batch_size * channels * spatial * spatial * bytes_per_elem
        )
        disc_activation_mb += feat_map_bytes / _BYTES_PER_MB

        # After down-sampling, spatial halves and channels double
        spatial //= 2
        channels = min(channels * 2, disc_base_channels * 16)

    # Final feature map at 4×4
    feat_map_bytes = (
        batch_size * channels * 4 * 4 * bytes_per_elem
    )
    disc_activation_mb += feat_map_bytes / _BYTES_PER_MB

    # ── 3. Loss computation memory ──────────────────────────────────────
    # GIoU, gradient penalty, consistency, etc. create intermediate tensors.
    # Estimate as a fraction of the total activation memory.
    loss_memory_mb = 0.1 * (gen_activation_mb + disc_activation_mb)

    # ── 4. Total activation memory ──────────────────────────────────────
    total_activation_mb = gen_activation_mb + disc_activation_mb + loss_memory_mb

    # ── 5. Apply gradient checkpointing reduction ─────────────────────────
    if use_checkpointing:
        # Gradient checkpointing reduces activation memory by approximately
        # sqrt(L) where L is the number of layers.  For ILGAN, L ≈
        # num_blocks_gen + num_blocks_disc + attention layers.
        num_layers = num_blocks_gen + num_blocks_disc + num_blocks_gen  # + attn
        reduction_factor = math.sqrt(max(num_layers, 1))
        total_activation_mb /= reduction_factor

    return total_activation_mb


# ──────────────────────────────────────────────────────────────────────────────
# profile_model_memory
# ──────────────────────────────────────────────────────────────────────────────


def profile_model_memory(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    loss_fn: Optional[callable] = None,
    use_mixed_precision: bool = False,
    num_repetitions: int = 3,
) -> Dict[str, Any]:
    """Run a forward and backward pass through a model and measure per-
    component CUDA memory usage.

    This function:

    1. Moves the model and input to CUDA (if not already).
    2. Runs a forward pass, measuring memory before and after.
    3. Computes a loss (either from *loss_fn* or by summing the output).
    4. Runs a backward pass, measuring peak memory.
    5. Returns a dictionary with per-component memory usage.

    The function runs *num_repetitions* passes and reports the median
    values to reduce noise from CUDA caching allocator behavior.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to profile.  Must be on CUDA or will be moved.
    input_tensor : torch.Tensor
        Input tensor for the model.  Must be on CUDA or will be moved.
    target : torch.Tensor, optional
        Optional target tensor for loss computation.  If not provided,
        the loss is computed as the sum of the model output.
    loss_fn : callable, optional
        Loss function ``loss_fn(output, target) -> Tensor``.  If not
        provided, ``torch.nn.MSELoss()`` is used when *target* is given,
        otherwise the output is summed.
    use_mixed_precision : bool, optional
        If True, run the forward pass under ``torch.cuda.amp.autocast``.
        (default: False)
    num_repetitions : int, optional
        Number of forward+backward passes to run.  The median of the
        measurements is reported.  (default: 3)

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"model_name"``: ``str`` — the model's class name.
        - ``"num_params"``: ``int`` — total number of parameters.
        - ``"num_trainable"``: ``int`` — number of trainable parameters.
        - ``"param_memory_mb"``: ``float`` — memory for parameters (MB).
        - ``"grad_memory_mb"``: ``float`` — memory for gradients (MB).
        - ``"opt_memory_mb"``: ``float`` — estimated optimizer memory (MB).
        - ``"activation_memory_mb"``: ``float`` — estimated activation
          memory (MB), computed as ``peak_mb - param_mb - grad_mb``.
        - ``"forward_memory_mb"``: ``float`` — memory allocated after
          forward pass (MB).
        - ``"peak_mb"``: ``float`` — peak memory allocated during the
          backward pass (MB).
        - ``"reserved_mb"``: ``float`` — total CUDA memory reserved (MB).
        - ``"input_shape"``: ``tuple`` — shape of the input tensor.
        - ``"batch_size"``: ``int`` — batch size from the input tensor.
        - ``"use_mixed_precision"``: ``bool`` — whether AMP was used.
        - ``"device"``: ``str`` — the CUDA device used.
        - ``"per_module"``: ``dict`` — per-sub-module parameter counts
          and estimated memory.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    ValueError
        If *num_repetitions* is less than 1.

    Notes
    -----
    - The function calls ``torch.cuda.reset_peak_memory_stats()`` and
      ``torch.cuda.empty_cache()`` before each repetition to get clean
      measurements.
    - The model is set to training mode during profiling.
    - The reported ``activation_memory_mb`` is an estimate computed as
      ``peak_mb - param_memory_mb - grad_memory_mb``.  The actual
      activation memory may differ due to CUDA caching allocator overhead.
    - For multi-GPU systems, the current device (``torch.cuda.current_device()``)
      is used.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for memory profiling. "
            "No CUDA-capable device detected."
        )

    if num_repetitions < 1:
        raise ValueError(
            f"num_repetitions must be >= 1, got {num_repetitions}."
        )

    # ── Move model and input to CUDA ────────────────────────────────────
    device = torch.device("cuda")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if target is not None:
        target = target.to(device)

    # ── Set up loss function ─────────────────────────────────────────────
    if loss_fn is None:
        if target is not None:
            loss_fn = torch.nn.MSELoss()
        else:
            loss_fn = lambda x: x.sum()  # noqa: E731

    # ── Count parameters ─────────────────────────────────────────────────
    param_counts = _count_parameters(model)
    num_params = param_counts["total"]
    num_trainable = param_counts["trainable"]

    # Parameter memory (float32)
    param_memory_mb = _estimate_parameter_memory(num_params, use_mixed_precision=False)
    grad_memory_mb = _estimate_gradient_memory(num_params, use_mixed_precision=False)
    opt_memory_mb = _estimate_optimizer_memory(num_params, optimizer_type="adam")

    # ── Per-module breakdown ────────────────────────────────────────────
    per_module = _count_parameters_by_module(model)
    per_module_memory: Dict[str, Dict[str, Any]] = {}
    for mod_name, mod_params in per_module.items():
        per_module_memory[mod_name] = {
            "num_params": mod_params,
            "param_memory_mb": _estimate_parameter_memory(mod_params, use_mixed_precision=False),
            "grad_memory_mb": _estimate_gradient_memory(mod_params, use_mixed_precision=False),
        }

    # ── Run profiling repetitions ──────────────────────────────────────
    all_forward_mb: List[float] = []
    all_peak_mb: List[float] = []
    all_reserved_mb: List[float] = []

    model.train()

    for _ in range(num_repetitions):
        # Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Record memory before
        mem_before = torch.cuda.memory_allocated(device)

        # ── Forward pass ────────────────────────────────────────────────
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(input_tensor)
                if isinstance(output, dict):
                    # For ILGANGenerator, use the image output for loss
                    loss_input = output.get("image", next(iter(output.values())))
                else:
                    loss_input = output
                loss = loss_fn(loss_input, target) if target is not None else loss_fn(loss_input)
        else:
            output = model(input_tensor)
            if isinstance(output, dict):
                loss_input = output.get("image", next(iter(output.values())))
            else:
                loss_input = output
            loss = loss_fn(loss_input, target) if target is not None else loss_fn(loss_input)

        # Record memory after forward
        torch.cuda.synchronize(device)
        mem_after_forward = torch.cuda.memory_allocated(device)
        forward_mb = (mem_after_forward - mem_before) / _BYTES_PER_MB
        all_forward_mb.append(forward_mb)

        # ── Backward pass ──────────────────────────────────────────────
        loss.backward()

        # Record peak memory
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mb = (peak_mem - mem_before) / _BYTES_PER_MB
        all_peak_mb.append(peak_mb)

        reserved_mem = torch.cuda.memory_reserved(device)
        reserved_mb = (reserved_mem - mem_before) / _BYTES_PER_MB
        all_reserved_mb.append(reserved_mb)

        # Zero gradients for next repetition
        model.zero_grad(set_to_none=True)

    # ── Compute median values ────────────────────────────────────────────
    sorted_forward = sorted(all_forward_mb)
    sorted_peak = sorted(all_peak_mb)
    sorted_reserved = sorted(all_reserved_mb)

    median_idx = num_repetitions // 2
    median_forward_mb = sorted_forward[median_idx]
    median_peak_mb = sorted_peak[median_idx]
    median_reserved_mb = sorted_reserved[median_idx]

    # ── Estimate activation memory ──────────────────────────────────────
    # Activation memory ≈ peak - param_memory - grad_memory
    # (optimizer memory is not allocated during forward/backward)
    activation_memory_mb = max(
        0.0, median_peak_mb - param_memory_mb - grad_memory_mb
    )

    # ── Build result dictionary ─────────────────────────────────────────
    result: Dict[str, Any] = {
        "model_name": model.__class__.__name__,
        "num_params": num_params,
        "num_trainable": num_trainable,
        "param_memory_mb": round(param_memory_mb, 2),
        "grad_memory_mb": round(grad_memory_mb, 2),
        "opt_memory_mb": round(opt_memory_mb, 2),
        "activation_memory_mb": round(activation_memory_mb, 2),
        "forward_memory_mb": round(median_forward_mb, 2),
        "peak_mb": round(median_peak_mb, 2),
        "reserved_mb": round(median_reserved_mb, 2),
        "input_shape": tuple(input_tensor.shape),
        "batch_size": input_tensor.shape[0],
        "use_mixed_precision": use_mixed_precision,
        "device": str(device),
        "per_module": per_module_memory,
        # Raw measurements for advanced analysis
        "raw_forward_mb": [round(v, 2) for v in all_forward_mb],
        "raw_peak_mb": [round(v, 2) for v in all_peak_mb],
        "raw_reserved_mb": [round(v, 2) for v in all_reserved_mb],
    }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# estimate_vram_requirements
# ──────────────────────────────────────────────────────────────────────────────


def estimate_vram_requirements(
    config: Config,
    optimizer_type: str = "adam",
) -> Dict[str, Any]:
    """Estimate the VRAM required for a given ILGAN configuration.

    This function computes a detailed estimate of GPU memory usage for
    training with the given configuration, without actually running any
    model forward/backward passes.  The estimate is based on parameter
    counts and heuristic activation memory models.

    The estimate includes:

    - **Parameter memory**: memory for all model parameters (generator,
      discriminator, image encoder, box encoder).
    - **Gradient memory**: memory for gradients (same size as parameters).
    - **Optimizer memory**: memory for Adam optimizer states (2× parameter
      memory for Adam).
    - **Activation memory**: memory for intermediate activations during
      forward/backward pass.
    - **Total estimated VRAM**: sum of all components with a safety margin.
    - **Recommended settings**: suggested batch size, gradient accumulation,
      checkpointing, and mixed precision based on available VRAM.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.  The following keys are used:

        - ``data.image_size`` (int): image spatial size.
        - ``data.batch_size`` (int): batch size.
        - ``model.latent_dim`` (int): latent vector dimension.
        - ``model.gen_base_channels`` (int): generator base channels.
        - ``model.disc_base_channels`` (int): discriminator base channels.
        - ``model.max_boxes`` (int): maximum boxes per image.
        - ``model.num_classes`` (int): number of object classes.
        - ``model.num_attention_heads`` (int): attention heads.
        - ``training.use_mixed_precision`` (bool): whether AMP is enabled.
        - ``training.grad_checkpoint`` (bool): whether checkpointing is enabled.
        - ``training.gradient_accumulation_steps`` (int): gradient accumulation.

    optimizer_type : str, optional
        Type of optimizer.  One of ``"adam"``, ``"adamw"``, ``"sgd"``.
        (default: ``"adam"``)

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"image_size"``: ``int`` — image spatial size.
        - ``"batch_size"``: ``int`` — batch size.
        - ``"use_mixed_precision"``: ``bool`` — whether AMP is enabled.
        - ``"use_checkpointing"``: ``bool`` — whether checkpointing is enabled.
        - ``"gradient_accumulation_steps"``: ``int`` — gradient accumulation.
        - ``"num_params_generator"``: ``int`` — estimated generator params.
        - ``"num_params_discriminator"``: ``int`` — estimated discriminator params.
        - ``"num_params_encoders"``: ``int`` — estimated encoder params.
        - ``"num_params_total"``: ``int`` — total parameters.
        - ``"param_memory_mb"``: ``float`` — parameter memory (MB).
        - ``"grad_memory_mb"``: ``float`` — gradient memory (MB).
        - ``"opt_memory_mb"``: ``float`` — optimizer memory (MB).
        - ``"activation_memory_mb"``: ``float`` — activation memory (MB).
        - ``"total_mb"``: ``float`` — total estimated VRAM (MB) with safety margin.
        - ``"total_gb"``: ``float`` — total estimated VRAM (GB).
        - ``"breakdown"``: ``dict`` — per-component breakdown with labels.
        - ``"recommendations"``: ``dict`` — recommended settings.

    Raises
    ------
    TypeError
        If *config* is not a :class:`Config` instance.

    Notes
    -----
    - The estimate includes a 20% safety margin (``_SAFETY_MARGIN = 1.2``)
      to account for PyTorch's CUDA caching allocator overhead.
    - The parameter counts for the generator and discriminator are estimated
      using analytical formulas based on the architecture.  These may differ
      slightly from the actual counts due to implementation details.
    - The activation memory estimate is a heuristic and may be off by
      ±30% depending on the specific input sizes and model configuration.
    """
    if not isinstance(config, Config):
        raise TypeError(
            f"Expected 'config' to be a Config instance, "
            f"got {type(config).__name__}."
        )

    # ── Extract config values with defaults ──────────────────────────────
    try:
        image_size = int(getattr(config.data, "image_size", _DEFAULT_IMAGE_SIZE))
        batch_size = int(getattr(config.data, "batch_size", _DEFAULT_BATCH_SIZE))
        latent_dim = int(getattr(config.model, "latent_dim", _DEFAULT_LATENT_DIM))
        gen_base_channels = int(
            getattr(config.model, "gen_base_channels", _DEFAULT_GEN_BASE_CHANNELS)
        )
        disc_base_channels = int(
            getattr(config.model, "disc_base_channels", _DEFAULT_DISC_BASE_CHANNELS)
        )
        max_boxes = int(getattr(config.model, "max_boxes", _DEFAULT_MAX_BOXES))
        num_classes = int(getattr(config.model, "num_classes", _DEFAULT_NUM_CLASSES))
        num_attention_heads = int(
            getattr(config.model, "num_attention_heads", _DEFAULT_NUM_ATTENTION_HEADS)
        )
        use_mixed_precision = bool(
            getattr(config.training, "use_mixed_precision", False)
        )
        use_checkpointing = bool(
            getattr(config.training, "grad_checkpoint", False)
        )
        gradient_accumulation_steps = int(
            getattr(config.training, "gradient_accumulation_steps", 1)
        )
    except (AttributeError, KeyError, TypeError) as e:
        raise ValueError(
            f"Config is missing a required key for VRAM estimation: {e}. "
            f"Please ensure your config has all required fields."
        ) from e

    # ── Compute number of blocks ─────────────────────────────────────────
    num_blocks_gen = int(math.log2(image_size)) - 2  # log2(image_size) - 2
    num_blocks_disc = int(math.log2(image_size)) - 2

    # ── Estimate parameter counts ────────────────────────────────────────
    # Generator parameter estimate:
    #   - Initial linear: latent_dim * (gen_base_channels * 16 * 4 * 4)
    #   - UpBlocks: sum over blocks of (in_C * 3*3 * out_C + out_C * 3*3 * out_C) * 2 (norm)
    #   - Final conv: in_C * 3*3 * 3
    #   - SpatialHead: queries (max_boxes * slot_dim) + projectors + SCCA + MLP + heads
    #   - slot_dim = gen_base_channels * 2
    slot_dim = gen_base_channels * 2

    # Initial projection
    init_proj_params = latent_dim * (gen_base_channels * 16 * 4 * 4)

    # UpBlocks
    upblock_params = 0
    in_ch = gen_base_channels * 16
    for _ in range(num_blocks_gen):
        out_ch = max(in_ch // 2, gen_base_channels)
        # Conv1: in_ch * 9 * out_ch (3x3 conv)
        # Conv2: out_ch * 9 * out_ch
        # Norm1: out_ch * 2 (weight + bias)
        # Norm2: out_ch * 2
        upblock_params += in_ch * 9 * out_ch + out_ch * 9 * out_ch + out_ch * 4
        in_ch = out_ch

    # Final conv: in_ch * 9 * 3
    final_conv_params = in_ch * 9 * 3

    # SpatialHead
    # Queries: max_boxes * slot_dim
    queries_params = max_boxes * slot_dim
    # Feature projectors: sum over levels of (C_i * 1*1 * slot_dim + slot_dim * 2)
    projector_params = 0
    in_ch = gen_base_channels * 16
    for _ in range(num_blocks_gen):
        out_ch = max(in_ch // 2, gen_base_channels)
        projector_params += in_ch * slot_dim + slot_dim * 2  # conv + norm
        in_ch = out_ch
    # SCCA modules: QKV projections + output projection
    # Each SCCA: 3 * slot_dim * proj_channels + proj_channels * slot_dim
    scca_params = num_blocks_gen * (4 * slot_dim * slot_dim)
    # SlotMLP: slot_dim * 2*slot_dim + 2*slot_dim * slot_dim
    mlp_params = slot_dim * 2 * slot_dim + 2 * slot_dim * slot_dim
    # Output heads: box (slot_dim * 4), class (slot_dim * num_classes), confidence (slot_dim * 1)
    head_params = slot_dim * 4 + slot_dim * num_classes + slot_dim * 1

    gen_params_est = (
        init_proj_params
        + upblock_params
        + final_conv_params
        + queries_params
        + projector_params
        + scca_params
        + mlp_params
        + head_params
    )

    # Discriminator parameter estimate:
    #   - DownBlocks: sum over blocks of (in_C * 16 * out_C) for 4x4 conv
    #   - Score convs: 2 * (in_C * 9 * in_C) + in_C * 9 * 1
    #   - Global head: in_C * 1
    disc_params_est = 0
    in_ch = 3  # RGB input
    for block_idx in range(num_blocks_disc):
        out_ch = min(disc_base_channels * (2 ** block_idx), disc_base_channels * 16)
        # 4x4 conv: in_ch * 16 * out_ch
        disc_params_est += in_ch * 16 * out_ch
        # Norm: out_ch * 2
        disc_params_est += out_ch * 2
        in_ch = out_ch

    # Score convs (after minibatch stddev, so +1 channel)
    score_input_ch = in_ch + 1
    disc_params_est += score_input_ch * 9 * in_ch  # score_conv1
    disc_params_est += in_ch * 9 * 1  # score_conv2
    # Global head
    disc_params_est += in_ch * 1  # linear

    # Encoder parameter estimates (small MLPs)
    # ImageFeatureEncoder: ~128 * 128 * 3 layers ≈ 50K params
    # BoxFeatureEncoder: ~128 * 128 * 3 layers ≈ 50K params
    encoder_params_est = 100_000  # Rough estimate

    total_params_est = gen_params_est + disc_params_est + encoder_params_est

    # ── Compute memory components ────────────────────────────────────────
    param_memory_mb = _estimate_parameter_memory(
        total_params_est, use_mixed_precision
    )
    grad_memory_mb = _estimate_gradient_memory(total_params_est)
    opt_memory_mb = _estimate_optimizer_memory(
        total_params_est, optimizer_type=optimizer_type
    )
    activation_memory_mb = _estimate_activation_memory(
        batch_size=batch_size,
        image_size=image_size,
        gen_base_channels=gen_base_channels,
        disc_base_channels=disc_base_channels,
        num_blocks_gen=num_blocks_gen,
        num_blocks_disc=num_blocks_disc,
        max_boxes=max_boxes,
        num_classes=num_classes,
        num_attention_heads=num_attention_heads,
        use_mixed_precision=use_mixed_precision,
        use_checkpointing=use_checkpointing,
    )

    # ── Total with safety margin ─────────────────────────────────────────
    total_mb = (
        param_memory_mb
        + grad_memory_mb
        + opt_memory_mb
        + activation_memory_mb
    ) * _SAFETY_MARGIN

    total_gb = total_mb / 1024.0

    # ── Build breakdown ─────────────────────────────────────────────────
    breakdown = OrderedDict([
        ("Parameters", {
            "mb": round(param_memory_mb, 2),
            "pct": round(param_memory_mb / total_mb * 100, 1) if total_mb > 0 else 0.0,
            "description": f"Model weights ({'float16' if use_mixed_precision else 'float32'})",
        }),
        ("Gradients", {
            "mb": round(grad_memory_mb, 2),
            "pct": round(grad_memory_mb / total_mb * 100, 1) if total_mb > 0 else 0.0,
            "description": "Gradient tensors (always float32)",
        }),
        ("Optimizer States", {
            "mb": round(opt_memory_mb, 2),
            "pct": round(opt_memory_mb / total_mb * 100, 1) if total_mb > 0 else 0.0,
            "description": f"Adam states ({_ADAM_OPT_STATES}× float32 per param)",
        }),
        ("Activations", {
            "mb": round(activation_memory_mb, 2),
            "pct": round(activation_memory_mb / total_mb * 100, 1) if total_mb > 0 else 0.0,
            "description": f"Intermediate tensors ({'checkpointing ON' if use_checkpointing else 'checkpointing OFF'})",
        }),
        ("Safety Margin", {
            "mb": round(total_mb - total_mb / _SAFETY_MARGIN, 2),
            "pct": round((1 - 1.0 / _SAFETY_MARGIN) * 100, 1),
            "description": f"{(_SAFETY_MARGIN - 1.0) * 100:.0f}% overhead for CUDA allocator",
        }),
    ])

    # ── Build result ─────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "image_size": image_size,
        "batch_size": batch_size,
        "use_mixed_precision": use_mixed_precision,
        "use_checkpointing": use_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_params_generator": gen_params_est,
        "num_params_discriminator": disc_params_est,
        "num_params_encoders": encoder_params_est,
        "num_params_total": total_params_est,
        "param_memory_mb": round(param_memory_mb, 2),
        "grad_memory_mb": round(grad_memory_mb, 2),
        "opt_memory_mb": round(opt_memory_mb, 2),
        "activation_memory_mb": round(activation_memory_mb, 2),
        "total_mb": round(total_mb, 2),
        "total_gb": round(total_gb, 2),
        "breakdown": breakdown,
        "recommendations": _generate_vram_recommendations(
            total_mb=total_mb,
            batch_size=batch_size,
            image_size=image_size,
            gen_base_channels=gen_base_channels,
            disc_base_channels=disc_base_channels,
            use_mixed_precision=use_mixed_precision,
            use_checkpointing=use_checkpointing,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
    }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Internal: generate VRAM recommendations
# ──────────────────────────────────────────────────────────────────────────────


def _generate_vram_recommendations(
    total_mb: float,
    batch_size: int,
    image_size: int,
    gen_base_channels: int,
    disc_base_channels: int,
    use_mixed_precision: bool,
    use_checkpointing: bool,
    gradient_accumulation_steps: int,
) -> Dict[str, Any]:
    """Generate recommended settings based on estimated VRAM usage.

    This function compares the estimated VRAM against common GPU memory
    sizes and suggests optimal settings.

    Parameters
    ----------
    total_mb : float
        Estimated total VRAM usage in MB.
    batch_size : int
        Current batch size.
    image_size : int
        Image spatial size.
    gen_base_channels : int
        Generator base channels.
    disc_base_channels : int
        Discriminator base channels.
    use_mixed_precision : bool
        Whether mixed precision is enabled.
    use_checkpointing : bool
        Whether gradient checkpointing is enabled.
    gradient_accumulation_steps : int
        Current gradient accumulation steps.

    Returns
    -------
    dict
        A dictionary with recommended settings.
    """
    # Common GPU memory sizes in MB
    gpu_memory_options = [
        ("4 GB", 4096),
        ("6 GB", 6144),
        ("8 GB", 8192),
        ("10 GB", 10240),
        ("12 GB", 12288),
        ("16 GB", 16384),
        ("20 GB", 20480),
        ("24 GB", 24576),
        ("32 GB", 32768),
        ("40 GB", 40960),
        ("48 GB", 49152),
        ("80 GB", 81920),
    ]

    # Find the smallest GPU that can fit this config
    recommended_gpu = None
    for label, size_mb in gpu_memory_options:
        if size_mb >= total_mb:
            recommended_gpu = label
            break

    if recommended_gpu is None:
        recommended_gpu = f">{gpu_memory_options[-1][0]}"

    # Determine if we can increase batch size
    if recommended_gpu is not None:
        # Find the max batch size for the recommended GPU
        max_batch_factor = 1.0
        for label, size_mb in gpu_memory_options:
            if label == recommended_gpu:
                max_batch_factor = size_mb / total_mb
                break
        max_batch_size = max(1, int(batch_size * max_batch_factor * 0.8))
    else:
        max_batch_size = batch_size

    # Recommendations
    recommendations: Dict[str, Any] = {
        "recommended_gpu": recommended_gpu,
        "current_fits_on_gpu": recommended_gpu is not None,
        "max_batch_size": max_batch_size,
        "suggested_batch_size": min(max_batch_size, batch_size * 2),
        "suggested_gradient_accumulation": max(
            1, gradient_accumulation_steps
        ),
        "suggest_mixed_precision": not use_mixed_precision and total_mb > 4096,
        "suggest_checkpointing": not use_checkpointing and total_mb > 6144,
        "can_double_batch_size": max_batch_size >= batch_size * 2,
        "can_halve_batch_size": batch_size > 1 and total_mb > 8192,
        "estimated_memory_utilization": min(
            100.0, (total_mb / 8192) * 100  # Assume 8 GB baseline
        ),
    }

    return recommendations


# ──────────────────────────────────────────────────────────────────────────────
# optimize_memory_usage
# ──────────────────────────────────────────────────────────────────────────────


def optimize_memory_usage(
    config: Config,
    available_vram_mb: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Analyze the configuration and model sizes, then recommend optimal
    settings for memory-efficient training.

    This function:

    1. Estimates VRAM requirements for the current configuration.
    2. If *available_vram_mb* is provided, checks whether the config fits.
    3. Recommends optimal settings for:
       - Batch size
       - Gradient accumulation steps
       - Gradient checkpointing (on/off)
       - Mixed precision (on/off)
       - Model size (base channels)
    4. Prints a human-readable report with estimated memory usage.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.
    available_vram_mb : float, optional
        Available VRAM in megabytes.  If provided, the function checks
        whether the current configuration fits and suggests adjustments.
        If not provided, the function uses common GPU memory sizes to
        make recommendations.
    verbose : bool, optional
        If True, print a detailed report to stdout.  (default: True)

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"current_estimate"``: ``dict`` — the VRAM estimate for the
          current configuration (see :func:`estimate_vram_requirements`).
        - ``"fits_in_vram"``: ``bool`` or ``None`` — whether the current
          config fits in the available VRAM (``None`` if not provided).
        - ``"recommendations"``: ``dict`` — recommended settings.
        - ``"config_changes"``: ``dict`` — suggested changes to the config
          to reduce memory usage.
        - ``"report"``: ``str`` — a human-readable report string.

    Raises
    ------
    TypeError
        If *config* is not a :class:`Config` instance.

    Notes
    -----
    - The function does **not** modify the config object.  It returns
      suggested changes that the user can apply manually.
    - The recommendations are based on heuristics and may not be optimal
      for all use cases.  Users should experiment with different settings
      to find the best configuration for their specific hardware.
    """
    if not isinstance(config, Config):
        raise TypeError(
            f"Expected 'config' to be a Config instance, "
            f"got {type(config).__name__}."
        )

    # ── 1. Estimate current VRAM requirements ───────────────────────────
    estimate = estimate_vram_requirements(config)
    total_mb = estimate["total_mb"]

    # ── 2. Check if config fits in available VRAM ────────────────────────
    fits_in_vram: Optional[bool] = None
    if available_vram_mb is not None:
        fits_in_vram = total_mb <= available_vram_mb

    # ── 3. Generate recommendations ─────────────────────────────────────
    recommendations = estimate["recommendations"]

    # ── 4. Generate config changes ───────────────────────────────────────
    config_changes: Dict[str, Any] = {}

    # Batch size
    if recommendations["suggested_batch_size"] != estimate["batch_size"]:
        config_changes["batch_size"] = {
            "current": estimate["batch_size"],
            "suggested": recommendations["suggested_batch_size"],
            "reason": (
                f"Adjust batch size to fit within available VRAM. "
                f"Current: {estimate['batch_size']}, "
                f"Suggested: {recommendations['suggested_batch_size']}."
            ),
        }

    # Mixed precision
    if recommendations["suggest_mixed_precision"]:
        config_changes["use_mixed_precision"] = {
            "current": False,
            "suggested": True,
            "reason": (
                "Mixed precision (AMP) reduces activation memory by ~40% "
                "and parameter memory by 2× with minimal quality loss. "
                "Strongly recommended for VRAM > 4 GB."
            ),
        }

    # Gradient checkpointing
    if recommendations["suggest_checkpointing"]:
        config_changes["grad_checkpoint"] = {
            "current": False,
            "suggested": True,
            "reason": (
                "Gradient checkpointing reduces activation memory by "
                "recomputing activations during backward pass. "
                "Recommended when VRAM is limited."
            ),
        }

    # Gradient accumulation
    if estimate["gradient_accumulation_steps"] < 2 and total_mb > 8192:
        config_changes["gradient_accumulation_steps"] = {
            "current": estimate["gradient_accumulation_steps"],
            "suggested": max(2, estimate["gradient_accumulation_steps"]),
            "reason": (
                "Increasing gradient accumulation allows using a smaller "
                "batch size while maintaining effective batch size. "
                "This reduces VRAM usage."
            ),
        }

    # Model size (base channels)
    if estimate["total_mb"] > 16384:  # > 16 GB
        config_changes["gen_base_channels"] = {
            "current": estimate.get("gen_base_channels", 64),
            "suggested": max(32, estimate.get("gen_base_channels", 64) // 2),
            "reason": (
                "Reducing generator base channels halves the model size "
                "and activation memory.  Consider reducing if VRAM is "
                "limited."
            ),
        }
        config_changes["disc_base_channels"] = {
            "current": estimate.get("disc_base_channels", 64),
            "suggested": max(32, estimate.get("disc_base_channels", 64) // 2),
            "reason": (
                "Reducing discriminator base channels halves the model size "
                "and activation memory."
            ),
        }

    # ── 5. Build report string ──────────────────────────────────────────
    report_lines: List[str] = []
    report_lines.append("=" * 72)
    report_lines.append("  ILGAN — Memory Optimisation Report")
    report_lines.append("=" * 72)
    report_lines.append("")

    # Configuration summary
    report_lines.append("  ┌─ Current Configuration")
    report_lines.append(f"  │  Image size:              {estimate['image_size']}")
    report_lines.append(f"  │  Batch size:              {estimate['batch_size']}")
    report_lines.append(f"  │  Mixed precision:          {'ON' if estimate['use_mixed_precision'] else 'OFF'}")
    report_lines.append(f"  │  Gradient checkpointing:  {'ON' if estimate['use_checkpointing'] else 'OFF'}")
    report_lines.append(f"  │  Gradient accumulation:   {estimate['gradient_accumulation_steps']}")
    report_lines.append(f"  │  Generator params:         {estimate['num_params_generator']:,}")
    report_lines.append(f"  │  Discriminator params:     {estimate['num_params_discriminator']:,}")
    report_lines.append(f"  │  Total params:             {estimate['num_params_total']:,}")
    report_lines.append("  └" + "─" * 50)
    report_lines.append("")

    # Memory breakdown
    report_lines.append("  ┌─ Estimated VRAM Breakdown")
    report_lines.append(f"  │  {'Component':<25s} {'Memory':>10s} {'%':>6s}")
    report_lines.append("  │  " + "-" * 43)
    for component, data in estimate["breakdown"].items():
        report_lines.append(
            f"  │  {component:<25s} {_format_mb(data['mb']):>10s} {data['pct']:>5.1f}%"
        )
    report_lines.append("  │  " + "-" * 43)
    report_lines.append(
        f"  │  {'TOTAL (with margin)':<25s} {_format_mb(estimate['total_mb']):>10s} 100.0%"
    )
    report_lines.append("  └" + "─" * 50)
    report_lines.append("")

    # Fit check
    if available_vram_mb is not None:
        if fits_in_vram:
            report_lines.append(
                f"  ✅  Configuration fits in {_format_mb(available_vram_mb)} VRAM "
                f"(uses {estimate['total_mb'] / available_vram_mb * 100:.1f}%)."
            )
        else:
            report_lines.append(
                f"  ❌  Configuration does NOT fit in {_format_mb(available_vram_mb)} VRAM "
                f"(needs {estimate['total_mb']:.0f} MB, "
                f"has {available_vram_mb:.0f} MB)."
            )
        report_lines.append("")

    # Recommended GPU
    report_lines.append(
        f"  Recommended GPU: {recommendations['recommended_gpu']}"
    )
    report_lines.append("")

    # Suggested changes
    if config_changes:
        report_lines.append("  ┌─ Suggested Configuration Changes")
        for key, change in config_changes.items():
            report_lines.append(f"  │  • {key}:")
            report_lines.append(
                f"  │    Current:  {change['current']}"
            )
            report_lines.append(
                f"  │    Suggested: {change['suggested']}"
            )
            report_lines.append(f"  │    Reason: {change['reason']}")
            report_lines.append("  │")
        report_lines.append("  └" + "─" * 50)
        report_lines.append("")
    else:
        report_lines.append("  ✅  No configuration changes suggested.")
        report_lines.append("")

    # Tips
    report_lines.append("  ┌─ Memory Optimisation Tips")
    report_lines.append("  │  • Use --mixed-precision to enable AMP (reduces memory by ~40%)")
    report_lines.append("  │  • Use --grad-checkpoint to enable checkpointing (reduces activations)")
    report_lines.append("  │  • Reduce --batch-size to lower activation memory")
    report_lines.append("  │  • Increase --gradient-accumulation to compensate for smaller batch")
    report_lines.append("  │  • Reduce --gen-base-channels and --disc-base-channels for smaller models")
    report_lines.append("  │  • Reduce --image-size for significantly lower memory usage")
    report_lines.append("  └" + "─" * 50)
    report_lines.append("")
    report_lines.append("=" * 72)

    report = "\n".join(report_lines)

    # ── Print report if verbose ──────────────────────────────────────────
    if verbose:
        print(report)

    # ── Build result ────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "current_estimate": estimate,
        "fits_in_vram": fits_in_vram,
        "recommendations": recommendations,
        "config_changes": config_changes,
        "report": report,
    }

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: print_profile_summary
# ──────────────────────────────────────────────────────────────────────────────


def print_profile_summary(profile: Dict[str, Any]) -> None:
    """Print a human-readable summary of a memory profile result.

    Parameters
    ----------
    profile : dict
        A dictionary returned by :func:`profile_model_memory`.

    Raises
    ------
    TypeError
        If *profile* is not a dictionary.
    """
    if not isinstance(profile, dict):
        raise TypeError(
            f"Expected 'profile' to be a dictionary, "
            f"got {type(profile).__name__}."
        )

    lines: List[str] = []
    lines.append("─" * 60)
    lines.append(f"  Memory Profile: {profile.get('model_name', 'Unknown')}")
    lines.append("─" * 60)
    lines.append(f"  Device:              {profile.get('device', 'N/A')}")
    lines.append(f"  Input shape:         {profile.get('input_shape', 'N/A')}")
    lines.append(f"  Batch size:          {profile.get('batch_size', 'N/A')}")
    lines.append(f"  Mixed precision:     {'ON' if profile.get('use_mixed_precision') else 'OFF'}")
    lines.append("")
    lines.append(f"  Parameters:          {profile.get('num_params', 0):,}")
    lines.append(f"  Trainable:           {profile.get('num_trainable', 0):,}")
    lines.append("")
    lines.append(f"  Parameter memory:    {_format_mb(profile.get('param_memory_mb', 0))}")
    lines.append(f"  Gradient memory:     {_format_mb(profile.get('grad_memory_mb', 0))}")
    lines.append(f"  Optimizer memory:    {_format_mb(profile.get('opt_memory_mb', 0))}")
    lines.append(f"  Activation memory:   {_format_mb(profile.get('activation_memory_mb', 0))}")
    lines.append("")
    lines.append(f"  Forward memory:      {_format_mb(profile.get('forward_memory_mb', 0))}")
    lines.append(f"  Peak memory:         {_format_mb(profile.get('peak_mb', 0))}")
    lines.append(f"  Reserved memory:     {_format_mb(profile.get('reserved_mb', 0))}")
    lines.append("")
    lines.append("  ┌─ Per-Module Breakdown")
    lines.append(f"  │  {'Module':<30s} {'Params':>10s} {'Param Mem':>10s} {'Grad Mem':>10s}")
    lines.append("  │  " + "-" * 62)
    per_module = profile.get("per_module", {})
    for mod_name, mod_data in per_module.items():
        lines.append(
            f"  │  {mod_name:<30s} "
            f"{mod_data['num_params']:>10,} "
            f"{_format_mb(mod_data['param_memory_mb']):>10s} "
            f"{_format_mb(mod_data['grad_memory_mb']):>10s}"
        )
    lines.append("  └" + "─" * 62)
    lines.append("─" * 60)

    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "profile_model_memory",
    "estimate_vram_requirements",
    "optimize_memory_usage",
    "print_profile_summary",
]
