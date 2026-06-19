"""
Intelligent gradient checkpointing strategy for ILGAN.

This module provides two core functions for selectively applying gradient
checkpointing to the ILGAN generator's ``ContentDecoder`` up-blocks, rather
than the default all-or-nothing approach:

1. :func:`analyze_checkpoint_tradeoff` — profiles each ``UpBlock`` in the
   generator's ``ContentDecoder``, measuring forward-pass time and CUDA
   memory allocation with and without gradient checkpointing.  Returns a
   ranked list of block indices where checkpointing is most beneficial
   (high memory savings with low compute overhead).

2. :func:`apply_selective_checkpointing` — uses the analysis results to
   enable gradient checkpointing **only** on the blocks that benefit most,
   leaving the remaining blocks without checkpointing overhead.

Mathematical motivation
-----------------------
Gradient checkpointing (Chen et al., 2016) trades compute for memory.
During the forward pass, intermediate activations are discarded; during
the backward pass, they are recomputed from the nearest saved checkpoint.
The memory savings :math:`\\Delta M` and compute overhead :math:`\\Delta T`
vary per block depending on:

- **Feature map size**: larger spatial resolutions (closer to the output)
  consume more activation memory, so checkpointing them saves more memory.
- **Channel count**: blocks with more channels have larger activations.
- **Compute intensity**: convolution-heavy blocks have a higher
  compute-to-memory ratio, making checkpointing more expensive relative
  to the memory saved.

The trade-off score for block :math:`i` is defined as:

.. math::

    S_i = \\frac{\\Delta M_i}{\\Delta T_i + \\varepsilon}

where :math:`\\Delta M_i = M_i^{\\text{(no ckpt)}} - M_i^{\\text{(ckpt)}}`
is the memory saved, and :math:`\\Delta T_i = T_i^{\\text{(ckpt)}} - T_i^{\\text{(no ckpt)}}`
is the additional compute time.  Blocks with :math:`S_i > \\tau` (a
configurable threshold) are selected for checkpointing.

Usage
-----
::

    from ilgan.utils.config import Config
    from ilgan.models.generator import ILGANGenerator
    from ilgan.scripts.checkpoint_strategy import (
        analyze_checkpoint_tradeoff,
        apply_selective_checkpointing,
    )

    cfg = Config()
    generator = ILGANGenerator(cfg).cuda()
    z = torch.randn(4, cfg.model.latent_dim).cuda()

    # Analyze which blocks benefit from checkpointing
    beneficial_indices = analyze_checkpoint_tradeoff(generator, z)
    print(f"Checkpointing beneficial on blocks: {beneficial_indices}")

    # Apply selective checkpointing
    apply_selective_checkpointing(generator, cfg)
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from ilgan.models.generator import ContentDecoder, ILGANGenerator
from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon to prevent division by zero in trade-off score computation."""

_DEFAULT_MEMORY_WEIGHT: float = 1.0
"""Weight of the memory-savings term in the trade-off score."""

_DEFAULT_TIME_WEIGHT: float = 1.0
"""Weight of the time-overhead term in the trade-off score."""

_DEFAULT_THRESHOLD: float = 0.5
"""Default trade-off score threshold for selecting a block."""

_DEFAULT_NUM_WARMUP: int = 3
"""Number of warm-up iterations before profiling."""

_DEFAULT_NUM_ITERATIONS: int = 5
"""Number of profiling iterations per block (median is reported)."""

_BYTES_PER_MB: float = 1024.0 * 1024.0
"""Conversion factor from bytes to megabytes."""


# ──────────────────────────────────────────────────────────────────────────────
# BlockProfileResult
# ──────────────────────────────────────────────────────────────────────────────


class BlockProfileResult:
    """Stores the profiling result for a single ``UpBlock``.

    Attributes
    ----------
    block_index : int
        Index of the block in ``ContentDecoder.up_blocks``.
    block_name : str
        Human-readable name (e.g. ``"up_blocks.3"``).
    in_channels : int
        Number of input channels to the block.
    out_channels : int
        Number of output channels from the block.
    spatial_size : int
        Spatial resolution (height/width) **after** the block's up-sampling.
    time_no_ckpt_ms : float
        Median forward-pass time **without** checkpointing (milliseconds).
    time_with_ckpt_ms : float
        Median forward-pass time **with** checkpointing (milliseconds).
    mem_no_ckpt_mb : float
        Median CUDA memory allocated **without** checkpointing (MB).
    mem_with_ckpt_mb : float
        Median CUDA memory allocated **with** checkpointing (MB).
    memory_savings_mb : float
        ``mem_no_ckpt_mb - mem_with_ckpt_mb`` (MB).  Positive means
        checkpointing saves memory.
    time_overhead_ms : float
        ``time_with_ckpt_ms - time_no_ckpt_ms`` (ms).  Positive means
        checkpointing adds compute overhead.
    tradeoff_score : float
        ``memory_savings_mb / (time_overhead_ms + EPS)``.  Higher values
        indicate blocks where checkpointing is more beneficial.
    is_beneficial : bool
        Whether ``tradeoff_score >= threshold``.
    """

    def __init__(
        self,
        block_index: int,
        block_name: str,
        in_channels: int,
        out_channels: int,
        spatial_size: int,
        time_no_ckpt_ms: float,
        time_with_ckpt_ms: float,
        mem_no_ckpt_mb: float,
        mem_with_ckpt_mb: float,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.block_index = block_index
        self.block_name = block_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        self.time_no_ckpt_ms = time_no_ckpt_ms
        self.time_with_ckpt_ms = time_with_ckpt_ms
        self.mem_no_ckpt_mb = mem_no_ckpt_mb
        self.mem_with_ckpt_mb = mem_with_ckpt_mb

        # Derived quantities
        self.memory_savings_mb = mem_no_ckpt_mb - mem_with_ckpt_mb
        self.time_overhead_ms = time_with_ckpt_ms - time_no_ckpt_ms

        # Trade-off score: memory saved per unit time overhead
        self.tradeoff_score = self.memory_savings_mb / (
            self.time_overhead_ms + _EPS
        )

        self.is_beneficial = self.tradeoff_score >= threshold

    def to_dict(self) -> Dict[str, float]:
        """Return all numeric fields as a flat dictionary."""
        return {
            "block_index": float(self.block_index),
            "in_channels": float(self.in_channels),
            "out_channels": float(self.out_channels),
            "spatial_size": float(self.spatial_size),
            "time_no_ckpt_ms": self.time_no_ckpt_ms,
            "time_with_ckpt_ms": self.time_with_ckpt_ms,
            "mem_no_ckpt_mb": self.mem_no_ckpt_mb,
            "mem_with_ckpt_mb": self.mem_with_ckpt_mb,
            "memory_savings_mb": self.memory_savings_mb,
            "time_overhead_ms": self.time_overhead_ms,
            "tradeoff_score": self.tradeoff_score,
            "is_beneficial": float(self.is_beneficial),
        }

    def __repr__(self) -> str:
        return (
            f"BlockProfileResult(block={self.block_index}, "
            f"name={self.block_name!r}, "
            f"mem_save={self.memory_savings_mb:.2f} MB, "
            f"time_overhead={self.time_overhead_ms:.2f} ms, "
            f"score={self.tradeoff_score:.3f}, "
            f"beneficial={self.is_beneficial})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _has_cuda() -> bool:
    """Return ``True`` if CUDA is available and usable."""
    return torch.cuda.is_available()


def _get_device(model: nn.Module) -> torch.device:
    """Return the device of the first parameter in *model*.

    If the model has no parameters, returns the current CUDA device (if
    available) or CPU.
    """
    for p in model.parameters():
        return p.device
    if _has_cuda():
        return torch.device("cuda")
    return torch.device("cpu")


def _measure_memory() -> float:
    """Return the current CUDA memory allocation in megabytes.

    Returns 0.0 if CUDA is not available.
    """
    if _has_cuda():
        return torch.cuda.memory_allocated() / _BYTES_PER_MB
    return 0.0


def _sync_device(device: torch.device) -> None:
    """Synchronise the device to ensure accurate timing.

    For CUDA devices, calls ``torch.cuda.synchronize()``.  For CPU, this
    is a no-op.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median(values: Sequence[float]) -> float:
    """Return the median of a sequence of floats."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Per-block profiling
# ──────────────────────────────────────────────────────────────────────────────


def _profile_single_block(
    content_decoder: ContentDecoder,
    block_idx: int,
    input_tensor: torch.Tensor,
    num_warmup: int = _DEFAULT_NUM_WARMUP,
    num_iterations: int = _DEFAULT_NUM_ITERATIONS,
    threshold: float = _DEFAULT_THRESHOLD,
) -> BlockProfileResult:
    """Profile a single ``UpBlock`` with and without gradient checkpointing.

    This function:

    1. Temporarily patches the ``ContentDecoder`` so that only the block at
       *block_idx* is executed (the rest are skipped via a monkey-patch).
    2. Runs *num_warmup* forward passes to warm up the GPU.
    3. Runs *num_iterations* forward passes **without** checkpointing,
       measuring time and memory.
    4. Runs *num_iterations* forward passes **with** checkpointing,
       measuring time and memory.
    5. Restores the original forward method.
    6. Returns a :class:`BlockProfileResult` with median measurements.

    Parameters
    ----------
    content_decoder : ContentDecoder
        The content decoder whose blocks to profile.
    block_idx : int
        Index of the block to profile (0-indexed).
    input_tensor : torch.Tensor
        Input tensor to the content decoder (latent vector ``z``).
        Shape ``[B, latent_dim]``.
    num_warmup : int, optional
        Number of warm-up iterations.  (default: 3)
    num_iterations : int, optional
        Number of profiling iterations per mode.  (default: 5)
    threshold : float, optional
        Trade-off score threshold for marking a block as beneficial.
        (default: 0.5)

    Returns
    -------
    BlockProfileResult
        Profiling result for the block.

    Raises
    ------
    IndexError
        If *block_idx* is out of range.
    RuntimeError
        If the content decoder has no up-blocks.
    """
    num_blocks = len(content_decoder.up_blocks)
    if num_blocks == 0:
        raise RuntimeError("ContentDecoder has no up-blocks to profile.")
    if block_idx < 0 or block_idx >= num_blocks:
        raise IndexError(
            f"block_idx {block_idx} out of range for {num_blocks} blocks."
        )

    device = _get_device(content_decoder)
    block = content_decoder.up_blocks[block_idx]

    # Get block metadata
    in_ch = block.in_channels
    out_ch = block.out_channels
    # Spatial size after this block: 4 * 2^{block_idx + 1}
    spatial_size = 4 * (2 ** (block_idx + 1))

    # ── Build a wrapper that runs only this block ───────────────────────
    # We need to feed the correct input to this block.  The easiest
    # approach: run the full decoder up to (but not including) this block,
    # then run only this block with/without checkpointing.

    def _run_prefix(h: torch.Tensor) -> torch.Tensor:
        """Run blocks 0..block_idx-1 to produce the input for block_idx."""
        for i in range(block_idx):
            h, _ = content_decoder.up_blocks[i](h)
        return h

    # ── Warm-up ─────────────────────────────────────────────────────────
    with torch.no_grad():
        # Run the full decoder once to ensure all CUDA kernels are loaded
        _ = content_decoder(input_tensor)
        _sync_device(device)

        # Warm up the prefix + block
        h_prefix = _run_prefix(content_decoder.init_linear(input_tensor).view(
            input_tensor.shape[0],
            content_decoder._init_channels,
            content_decoder._init_spatial,
            content_decoder._init_spatial,
        ))
        for _ in range(num_warmup):
            _ = block(h_prefix)
            _sync_device(device)

    # ── Profile WITHOUT checkpointing ──────────────────────────────────
    times_no_ckpt: List[float] = []
    mems_no_ckpt: List[float] = []

    for _ in range(num_iterations):
        # Recompute prefix each iteration to get clean memory measurements
        with torch.no_grad():
            h_prefix = _run_prefix(content_decoder.init_linear(input_tensor).view(
                input_tensor.shape[0],
                content_decoder._init_channels,
                content_decoder._init_spatial,
                content_decoder._init_spatial,
            ))

        _sync_device(device)
        mem_before = _measure_memory()
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = block(h_prefix)

        _sync_device(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mem_after = _measure_memory()

        times_no_ckpt.append(elapsed_ms)
        mems_no_ckpt.append(mem_after - mem_before)

    # ── Profile WITH checkpointing ─────────────────────────────────────
    times_with_ckpt: List[float] = []
    mems_with_ckpt: List[float] = []

    for _ in range(num_iterations):
        with torch.no_grad():
            h_prefix = _run_prefix(content_decoder.init_linear(input_tensor).view(
                input_tensor.shape[0],
                content_decoder._init_channels,
                content_decoder._init_spatial,
                content_decoder._init_spatial,
            ))

        _sync_device(device)
        mem_before = _measure_memory()
        start_time = time.perf_counter()

        with torch.no_grad():
            _ = checkpoint.checkpoint(block, h_prefix, use_reentrant=False)

        _sync_device(device)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mem_after = _measure_memory()

        times_with_ckpt.append(elapsed_ms)
        mems_with_ckpt.append(mem_after - mem_before)

    # ── Compute medians ─────────────────────────────────────────────────
    median_time_no_ckpt = _median(times_no_ckpt)
    median_time_with_ckpt = _median(times_with_ckpt)
    median_mem_no_ckpt = _median(mems_no_ckpt)
    median_mem_with_ckpt = _median(mems_with_ckpt)

    return BlockProfileResult(
        block_index=block_idx,
        block_name=f"up_blocks.{block_idx}",
        in_channels=in_ch,
        out_channels=out_ch,
        spatial_size=spatial_size,
        time_no_ckpt_ms=median_time_no_ckpt,
        time_with_ckpt_ms=median_time_with_ckpt,
        mem_no_ckpt_mb=median_mem_no_ckpt,
        mem_with_ckpt_mb=median_mem_with_ckpt,
        threshold=threshold,
    )


# ──────────────────────────────────────────────────────────────────────────────
# analyze_checkpoint_tradeoff
# ──────────────────────────────────────────────────────────────────────────────


def analyze_checkpoint_tradeoff(
    model: ILGANGenerator,
    input_tensor: torch.Tensor,
    threshold: float = _DEFAULT_THRESHOLD,
    num_warmup: int = _DEFAULT_NUM_WARMUP,
    num_iterations: int = _DEFAULT_NUM_ITERATIONS,
    verbose: bool = True,
) -> List[int]:
    """Analyze the memory-time trade-off of gradient checkpointing for each
    ``UpBlock`` in the generator's ``ContentDecoder``.

    This function profiles every up-block in the content decoder, measuring
    the forward-pass time and CUDA memory allocation with and without
    gradient checkpointing.  It then computes a **trade-off score** for each
    block:

    .. math::

        S_i = \\frac{\\text{memory\\_savings}_i}{\\text{time\\_overhead}_i + \\varepsilon}

    Blocks with :math:`S_i \\geq \\text{threshold}` are considered
    "beneficial" for checkpointing.

    Parameters
    ----------
    model : ILGANGenerator
        The ILGAN generator instance.  Must contain a ``ContentDecoder``
        with at least one ``UpBlock``.
    input_tensor : torch.Tensor
        A batch of latent vectors of shape ``[B, latent_dim]``.  The batch
        size should match the intended training batch size for realistic
        memory measurements.
    threshold : float, optional
        Minimum trade-off score for a block to be considered beneficial.
        Higher values select fewer blocks (more conservative).  Must be
        non-negative.  (default: 0.5)
    num_warmup : int, optional
        Number of warm-up iterations before profiling.  (default: 3)
    num_iterations : int, optional
        Number of profiling iterations per block per mode (with/without
        checkpointing).  The median is reported.  (default: 5)
    verbose : bool, optional
        If ``True``, print a summary table of the profiling results to
        stdout.  (default: ``True``)

    Returns
    -------
    list of int
        Sorted list of block indices (0-indexed) where checkpointing is
        beneficial (``tradeoff_score >= threshold``).  Empty if no blocks
        benefit.

    Raises
    ------
    TypeError
        If *model* is not an ``ILGANGenerator``.
    ValueError
        If *threshold* is negative, or if the content decoder has no blocks.
    RuntimeError
        If CUDA is not available (profiling requires CUDA for accurate
        memory measurements).

    Example
    -------
    >>> from ilgan.utils.config import Config
    >>> from ilgan.models.generator import ILGANGenerator
    >>> from ilgan.scripts.checkpoint_strategy import analyze_checkpoint_tradeoff
    >>> cfg = Config()
    >>> gen = ILGANGenerator(cfg).cuda()
    >>> z = torch.randn(4, cfg.model.latent_dim).cuda()
    >>> beneficial = analyze_checkpoint_tradeoff(gen, z, threshold=0.5)
    >>> print(f"Checkpoint these blocks: {beneficial}")
    """
    # ── Validate inputs ─────────────────────────────────────────────────
    if not isinstance(model, ILGANGenerator):
        raise TypeError(
            f"Expected 'model' to be an ILGANGenerator, "
            f"got {type(model).__name__}."
        )

    if threshold < 0.0:
        raise ValueError(
            f"threshold must be non-negative, got {threshold}."
        )

    if not _has_cuda():
        raise RuntimeError(
            "CUDA is required for checkpoint trade-off analysis. "
            "Memory measurements are not meaningful on CPU."
        )

    content_decoder = model.content_decoder
    num_blocks = len(content_decoder.up_blocks)

    if num_blocks == 0:
        raise ValueError(
            "ContentDecoder has no up-blocks to analyze."
        )

    # Ensure the model and input are on CUDA
    device = _get_device(model)
    if device.type != "cuda":
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        device = torch.device("cuda")

    # ── Profile each block ──────────────────────────────────────────────
    results: List[BlockProfileResult] = []

    for block_idx in range(num_blocks):
        if verbose:
            print(
                f"  Profiling block {block_idx}/{num_blocks - 1} "
                f"({content_decoder.up_blocks[block_idx].__class__.__name__}) ...",
                end=" ",
                flush=True,
            )

        result = _profile_single_block(
            content_decoder=content_decoder,
            block_idx=block_idx,
            input_tensor=input_tensor,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            threshold=threshold,
        )
        results.append(result)

        if verbose:
            print(
                f"mem_save={result.memory_savings_mb:.2f} MB, "
                f"time_overhead={result.time_overhead_ms:.2f} ms, "
                f"score={result.tradeoff_score:.3f}, "
                f"{'✅' if result.is_beneficial else '❌'}"
            )

    # ── Collect beneficial block indices ─────────────────────────────────
    beneficial_indices = [
        r.block_index for r in results if r.is_beneficial
    ]

    # ── Print summary ────────────────────────────────────────────────────
    if verbose:
        _print_analysis_summary(results, beneficial_indices, threshold)

    return beneficial_indices


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────


def _print_analysis_summary(
    results: List[BlockProfileResult],
    beneficial_indices: List[int],
    threshold: float,
) -> None:
    """Print a formatted summary table of the profiling results."""
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  Gradient Checkpointing Trade-off Analysis")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"  Threshold: {threshold:.2f}  |  "
        f"Beneficial blocks: {len(beneficial_indices)}/{len(results)}"
    )
    lines.append("")

    # Table header
    header = (
        f"  {'Block':<8s} {'Name':<18s} {'Size':<8s} "
        f"{'Mem No Ckpt':>12s} {'Mem Ckpt':>10s} {'Save':>8s} "
        f"{'Time No Ckpt':>12s} {'Time Ckpt':>10s} {'Overhead':>10s} "
        f"{'Score':>8s} {'Sel':>5s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in results:
        size_str = f"{r.in_channels}→{r.out_channels} @ {r.spatial_size}²"
        sel_marker = "✅" if r.is_beneficial else " "
        lines.append(
            f"  {r.block_index:<8d} "
            f"{r.block_name:<18s} "
            f"{size_str:<8s} "
            f"{r.mem_no_ckpt_mb:>8.2f} MB "
            f"{r.mem_with_ckpt_mb:>8.2f} MB "
            f"{r.memory_savings_mb:>+7.2f} MB "
            f"{r.time_no_ckpt_ms:>8.2f} ms "
            f"{r.time_with_ckpt_ms:>8.2f} ms "
            f"{r.time_overhead_ms:>+8.2f} ms "
            f"{r.tradeoff_score:>8.3f} "
            f"{sel_marker:>5s}"
        )

    lines.append("  " + "-" * (len(header) - 2))
    lines.append("")

    if beneficial_indices:
        lines.append(
            f"  ✅ Selected blocks for checkpointing: {beneficial_indices}"
        )
    else:
        lines.append(
            "  ℹ️  No blocks meet the threshold. "
            "Consider lowering the threshold or disabling checkpointing."
        )

    # Recommendation
    lines.append("")
    lines.append("  ┌─ Recommendation")
    if len(beneficial_indices) == len(results):
        lines.append("  │  All blocks benefit from checkpointing — use all-or-nothing mode.")
    elif len(beneficial_indices) == 0:
        lines.append("  │  No blocks benefit — consider disabling checkpointing entirely.")
    else:
        lines.append(
            f"  │  Enable checkpointing on blocks {beneficial_indices} "
            f"for optimal memory-compute trade-off."
        )
    lines.append("  └" + "─" * 50)
    lines.append("")

    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# apply_selective_checkpointing
# ──────────────────────────────────────────────────────────────────────────────


def apply_selective_checkpointing(
    generator: ILGANGenerator,
    config: Config,
    threshold: float = _DEFAULT_THRESHOLD,
    num_warmup: int = _DEFAULT_NUM_WARMUP,
    num_iterations: int = _DEFAULT_NUM_ITERATIONS,
    verbose: bool = True,
) -> List[int]:
    """Analyze the generator and apply gradient checkpointing **only** on
    the blocks that benefit from it.

    This function:

    1. Calls :func:`analyze_checkpoint_tradeoff` to identify which
       ``UpBlock`` indices benefit from checkpointing.
    2. Modifies the ``ContentDecoder`` to enable checkpointing only on
       those blocks (via the ``checkpoint_block_indices`` set).
    3. Returns the list of selected block indices.

    The modification is **in-place** on the generator's
    ``content_decoder``.  After calling this function, the generator will
    use checkpointing only on the selected blocks during training.

    Parameters
    ----------
    generator : ILGANGenerator
        The ILGAN generator to modify.  Must be on CUDA for profiling.
    config : Config
        The ILGAN configuration object.  Used to create a dummy input
        tensor for profiling (reads ``model.latent_dim`` and
        ``data.batch_size``).
    threshold : float, optional
        Minimum trade-off score for a block to be selected.  Passed
        directly to :func:`analyze_checkpoint_tradeoff`.  (default: 0.5)
    num_warmup : int, optional
        Number of warm-up iterations for profiling.  (default: 3)
    num_iterations : int, optional
        Number of profiling iterations per block.  (default: 5)
    verbose : bool, optional
        If ``True``, print profiling output.  (default: ``True``)

    Returns
    -------
    list of int
        Sorted list of block indices where checkpointing was enabled.

    Raises
    ------
    TypeError
        If *generator* is not an ``ILGANGenerator`` or *config* is not a
        :class:`Config`.
    RuntimeError
        If CUDA is not available.

    Example
    -------
    >>> from ilgan.utils.config import Config
    >>> from ilgan.models.generator import ILGANGenerator
    >>> from ilgan.scripts.checkpoint_strategy import apply_selective_checkpointing
    >>> cfg = Config()
    >>> gen = ILGANGenerator(cfg).cuda()
    >>> selected = apply_selective_checkpointing(gen, cfg)
    >>> print(f"Checkpointing enabled on blocks: {selected}")
    """
    # ── Validate inputs ─────────────────────────────────────────────────
    if not isinstance(generator, ILGANGenerator):
        raise TypeError(
            f"Expected 'generator' to be an ILGANGenerator, "
            f"got {type(generator).__name__}."
        )

    if not isinstance(config, Config):
        raise TypeError(
            f"Expected 'config' to be a Config instance, "
            f"got {type(config).__name__}."
        )

    if not _has_cuda():
        raise RuntimeError(
            "CUDA is required for selective checkpointing analysis. "
            "Cannot profile memory on CPU."
        )

    # ── Create a dummy input tensor for profiling ───────────────────────
    latent_dim = int(config.model.latent_dim)
    batch_size = int(config.data.batch_size)
    device = _get_device(generator)

    input_tensor = torch.randn(batch_size, latent_dim, device=device)

    # ── Run the trade-off analysis ──────────────────────────────────────
    if verbose:
        print("Running checkpoint trade-off analysis ...")
        print(f"  Model: {generator.__class__.__name__}")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Number of up-blocks: {len(generator.content_decoder.up_blocks)}")
        print()

    beneficial_indices = analyze_checkpoint_tradeoff(
        model=generator,
        input_tensor=input_tensor,
        threshold=threshold,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
        verbose=verbose,
    )

    # ── Apply selective checkpointing ───────────────────────────────────
    content_decoder = generator.content_decoder

    if len(beneficial_indices) == len(content_decoder.up_blocks):
        # All blocks benefit — use the simple all-or-nothing flag
        content_decoder.use_checkpointing = True
        content_decoder.checkpoint_block_indices = set(
            range(len(content_decoder.up_blocks))
        )
        if verbose:
            print(
                "  ✅ All blocks benefit — enabled global checkpointing."
            )
    elif len(beneficial_indices) == 0:
        # No blocks benefit — disable checkpointing entirely
        content_decoder.use_checkpointing = False
        content_decoder.checkpoint_block_indices = set()
        if verbose:
            print(
                "  ℹ️  No blocks benefit — checkpointing disabled."
            )
    else:
        # Selective: enable checkpointing only on beneficial blocks
        content_decoder.use_checkpointing = True
        content_decoder.checkpoint_block_indices = set(beneficial_indices)
        if verbose:
            print(
                f"  ✅ Selective checkpointing enabled on blocks: "
                f"{beneficial_indices}"
            )

    return beneficial_indices


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: get_checkpoint_status
# ──────────────────────────────────────────────────────────────────────────────


def get_checkpoint_status(
    generator: ILGANGenerator,
) -> Dict[str, object]:
    """Return a dictionary describing the current checkpointing state of
    the generator's ``ContentDecoder``.

    Parameters
    ----------
    generator : ILGANGenerator
        The generator to inspect.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"use_checkpointing"`` (bool): whether the global flag is set.
        - ``"checkpoint_block_indices"`` (set of int): indices of blocks
          with checkpointing enabled.
        - ``"num_blocks"`` (int): total number of up-blocks.
        - ``"num_checkpointed"`` (int): number of blocks with checkpointing.
        - ``"mode"`` (str): one of ``"all"``, ``"none"``, ``"selective"``.
        - ``"blocks"`` (list of dict): per-block status.

    Raises
    ------
    TypeError
        If *generator* is not an ``ILGANGenerator``.
    """
    if not isinstance(generator, ILGANGenerator):
        raise TypeError(
            f"Expected 'generator' to be an ILGANGenerator, "
            f"got {type(generator).__name__}."
        )

    decoder = generator.content_decoder
    num_blocks = len(decoder.up_blocks)

    # Determine the set of checkpointed block indices
    if hasattr(decoder, "checkpoint_block_indices") and decoder.checkpoint_block_indices:
        ckpt_indices = set(decoder.checkpoint_block_indices)
    elif decoder.use_checkpointing:
        ckpt_indices = set(range(num_blocks))
    else:
        ckpt_indices = set()

    # Determine mode
    if len(ckpt_indices) == num_blocks:
        mode = "all"
    elif len(ckpt_indices) == 0:
        mode = "none"
    else:
        mode = "selective"

    # Per-block status
    blocks = []
    for i, block in enumerate(decoder.up_blocks):
        blocks.append({
            "index": i,
            "name": f"up_blocks.{i}",
            "type": block.__class__.__name__,
            "in_channels": block.in_channels,
            "out_channels": block.out_channels,
            "checkpointed": i in ckpt_indices,
        })

    return {
        "use_checkpointing": decoder.use_checkpointing,
        "checkpoint_block_indices": ckpt_indices,
        "num_blocks": num_blocks,
        "num_checkpointed": len(ckpt_indices),
        "mode": mode,
        "blocks": blocks,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "BlockProfileResult",
    "analyze_checkpoint_tradeoff",
    "apply_selective_checkpointing",
    "get_checkpoint_status",
]
