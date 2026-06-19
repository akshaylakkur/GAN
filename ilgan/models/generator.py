"""
Content decoder — image generation pathway of the ILGAN generator.

The ``ContentDecoder`` transforms a latent vector ``z`` into a full-resolution
image through a progressive-growing-style decoder architecture.  It produces
**both** the final RGB image **and** a list of intermediate feature maps (skip
features) at increasing resolutions that bridge to the spatial (bounding box)
pathway.

Architecture overview
---------------------
1. **Input projection**: ``z ∈ ℝ^latent_dim`` is linearly projected to
   ``gen_base_channels × 16`` channels at 4×4 spatial resolution.
2. **Up-sampling blocks**: A sequence of ``UpBlock`` modules, each performing:
   nearest-neighbour upsample (2×), Conv2d(3×3), normalisation, LeakyReLU(0.2),
   Conv2d(3×3), normalisation, LeakyReLU(0.2).  Channel count halves at each
   block until ``gen_base_channels`` is reached, after which it stays constant.
3. **Output head**: Conv2d(3×3) → Tanh produces pixel values in ``[-1, 1]``.

Each ``UpBlock`` emits a *skip feature map* at its output resolution, stored
in a list for the spatial (bounding box) pathway.

SpatialHead
-----------
The ``SpatialHead`` is the **bounding box prediction pathway**.  It consumes
the multi-resolution skip features from ``ContentDecoder`` and produces
bounding box proposals through a coarse-to-fine cross-attention mechanism.

The key innovation: by processing spatial queries through
``SpatialContentCrossAttention`` (SCCA) at multiple resolutions, the system
naturally learns a coarse-to-fine binding of features to spatial locations,
preventing bounding box collapse because the repulsion mechanism in SCCA
explicitly pushes slot attention centres apart.

ILGANGenerator
--------------
The ``ILGANGenerator`` is the top-level unified generator that composes
``ContentDecoder`` and ``SpatialHead`` into a single forward pass.  It
adds spectral normalisation, learnable instance noise, latent statistics
tracking, and gradient checkpointing propagation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ilgan.models.attention import SpatialContentCrossAttention
from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# SpectralNormConv2d — Conv2d with optional spectral normalisation
# ──────────────────────────────────────────────────────────────────────────────


class SpectralNormConv2d(nn.Module):
    """Conv2d layer wrapped with optional spectral normalisation.

    Spectral normalisation (``nn.utils.spectral_norm``) constrains the
    spectral norm (largest singular value) of the weight matrix, which
    stabilises GAN training by preventing discriminator/generator gradient
    explosions (Miyato et al., 2018).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size (assumed square).
    stride : int, optional
        Convolution stride (default: 1).
    padding : int, optional
        Spatial padding (default: 0).
    use_spectral_norm : bool, optional
        If True, apply ``spectral_norm`` to the convolutional weight.
        (default: False)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self._use_spectral_norm = use_spectral_norm

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if use_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def extra_repr(self) -> str:
        return f"spectral_norm={self._use_spectral_norm}"


# ──────────────────────────────────────────────────────────────────────────────
# UpBlock — single up-sampling stage
# ──────────────────────────────────────────────────────────────────────────────


class UpBlock(nn.Module):
    """One up-sampling block for the progressive-growing decoder.

    Operations (in order)
    ---------------------
    1. Nearest-neighbour upsample (scale factor = 2).
    2. Conv2d(3×3)  —  channels ``in_channels → out_channels``.
    3. BatchNorm2d (or GroupNorm if ``use_group_norm=True``).
    4. LeakyReLU(0.2, inplace).
    5. Conv2d(3×3)  —  channels ``out_channels → out_channels``.
    6. BatchNorm2d (or GroupNorm).
    7. LeakyReLU(0.2, inplace).

    The block returns **two** tensors:
    - ``out``: the final output feature map at 2× resolution.
    - ``skip``: an intermediate feature map (after the first conv-norm-act)
      at the same output resolution, which serves as a skip connection /
      bridge to the spatial pathway.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (must be ≤ in_channels).
    use_spectral_norm : bool, optional
        Apply spectral normalisation to both Conv2d layers.
        (default: False)
    use_group_norm : bool, optional
        If True, use ``nn.GroupNorm(num_groups=min(4, out_channels))``
        instead of ``nn.BatchNorm2d``.  Recommended when batch size is
        small (e.g. 1–4) to avoid noisy batch statistics.  (default: False)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_spectral_norm = use_spectral_norm
        self.use_group_norm = use_group_norm

        # Conv 1: in_channels → out_channels  (3×3, pad=1 preserves size)
        self.conv1 = SpectralNormConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_spectral_norm=use_spectral_norm,
        )

        # Conv 2: out_channels → out_channels  (3×3)
        self.conv2 = SpectralNormConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_spectral_norm=use_spectral_norm,
        )

        # Normalisation layers
        num_groups = min(4, out_channels) if out_channels > 0 else 1
        if use_group_norm:
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the up-sampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``[B, in_channels, H, W]``.

        Returns
        -------
        out : torch.Tensor
            Output feature map of shape ``[B, out_channels, H×2, W×2]``.
        skip : torch.Tensor
            Intermediate feature map after the first conv-norm-act,
            same spatial size as ``out``.  Shape:
            ``[B, out_channels, H×2, W×2]``.
        """
        # 1. Nearest-neighbour upsample
        x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # 2. First conv → norm → activation
        h = self.conv1(x_up)
        h = self.norm1(h)
        h = self.activation(h)

        # 3. Store intermediate feature map as skip connection
        skip = h

        # 4. Second conv → norm → activation
        h = self.conv2(h)
        h = self.norm2(h)
        out = self.activation(h)

        return out, skip

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"spectral_norm={self.use_spectral_norm}, "
            f"group_norm={self.use_group_norm}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ContentDecoder — full image-generation pathway
# ──────────────────────────────────────────────────────────────────────────────


class ContentDecoder(nn.Module):
    """Progressive-growing image decoder with multi-resolution skip features.

    This module is the **image generation** pathway of the ILGAN generator.
    It transforms a latent code ``z`` into a high-resolution RGB image and a
    collection of intermediate feature maps at increasing spatial resolutions
    that are consumed by the spatial (bounding box) pathway.

    Mathematical outline
    --------------------
    Let ``C₀ = gen_base_channels × 16`` be the initial channel count at 4×4.
    For each up-block ``i ∈ {0, …, L-1}`` (``L = log₂(image_size) - 2``):

        h_{i+1}, s_{i+1} = UpBlock_i(h_i)

    where ``h₀ ∈ ℝ^{B × C₀ × 4 × 4}`` is the initial projection, and
    ``s_{i+1}`` is the skip feature.  Channel counts:

        C_{i+1} = max(C_i // 2, gen_base_channels)

    The final output is:

        I = tanh(Conv_{3×3}(h_L))  ∈  [-1, 1]^{B × 3 × H × W}

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector ``z``.
    gen_base_channels : int
        Base channel count.  The initial feature map has
        ``gen_base_channels × 16`` channels at 4×4; the final feature map
        before the output convolution has ``gen_base_channels`` channels.
    image_size : int
        Target image spatial size (square).  Must be a power of two ≥ 8.
    use_spectral_norm : bool, optional
        Apply spectral normalisation to all convolutional layers in every
        ``UpBlock`` and the final output convolution.  (default: False)
    use_checkpointing : bool, optional
        Enable gradient checkpointing for each ``UpBlock`` forward pass
        during training (``torch.utils.checkpoint.checkpoint``).  Reduces
        VRAM usage at the cost of a small compute overhead.  (default: False)
    use_group_norm : bool, optional
        Use ``GroupNorm`` instead of ``BatchNorm2d`` in all ``UpBlock``
        modules.  Recommended when the batch size is very small (e.g. 1–4).
        (default: False)

    Raises
    ------
    ValueError
        If ``image_size`` is not a power of two or is < 8.
    """

    def __init__(
        self,
        latent_dim: int,
        gen_base_channels: int,
        image_size: int,
        use_spectral_norm: bool = False,
        use_checkpointing: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()

        # ── Validate image_size ──────────────────────────────────────────
        if image_size < 8 or (image_size & (image_size - 1)) != 0:
            raise ValueError(
                f"image_size must be a power of two >= 8, got {image_size}."
            )

        self.latent_dim = latent_dim
        self.gen_base_channels = gen_base_channels
        self.image_size = image_size
        self.use_spectral_norm = use_spectral_norm
        self.use_checkpointing = use_checkpointing
        # Set of block indices (0-indexed) where gradient checkpointing is
        # enabled.  When non-empty, only these blocks use checkpointing;
        # the global use_checkpointing flag must also be True for
        # checkpointing to be active.  If empty and use_checkpointing
        # is True, all blocks use checkpointing (backward-compatible).
        self.checkpoint_block_indices: set = set()
        self.use_group_norm = use_group_norm

        # Number of up-blocks = number of 2× steps from 4×4 to image_size
        self.num_blocks = int(math.log2(image_size)) - int(math.log2(4))  # L

        # ── 1. Learned linear projection to 4×4 spatial feature map ────
        _init_channels = gen_base_channels * 16  # C₀
        _init_spatial = 4  # starting spatial size

        self.init_linear = nn.Linear(
            latent_dim,
            _init_channels * _init_spatial * _init_spatial,
            bias=True,
        )
        # Stored as attributes so ``forward`` can access them
        self._init_channels = _init_channels
        self._init_spatial = _init_spatial

        # ── 2. Build the sequence of UpBlocks ──────────────────────────
        self.up_blocks = nn.ModuleList()
        self._skip_channels: List[int] = []  # C_i for each skip feature

        in_ch = _init_channels
        for _ in range(self.num_blocks):
            # Halve channels, but never go below gen_base_channels
            out_ch = max(in_ch // 2, gen_base_channels)
            block = UpBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                use_spectral_norm=use_spectral_norm,
                use_group_norm=use_group_norm,
            )
            self.up_blocks.append(block)
            self._skip_channels.append(out_ch)
            in_ch = out_ch

        # in_ch is now the channel count at the final resolution

        # ── 3. Final output convolution (3×3, 3 output channels) ───────
        self.final_conv = SpectralNormConv2d(
            in_ch,
            3,
            kernel_size=3,
            padding=1,
            use_spectral_norm=use_spectral_norm,
        )
        self.final_activation = nn.Tanh()

        # ── Parameter initialisation ────────────────────────────────────
        self.reset_parameters()

    # ── parameter initialisation ─────────────────────────────────────────

    def reset_parameters(self) -> None:
        """Initialise weights for stable training dynamics.

        - Linear projection: Kaiming uniform (He et al., 2015) with
          ``a = √5``, which accounts for the LeakyReLU non-linearity.
        - Bias of linear projection: zero-initialised.
        - All convolutional weights are initialised by their default
          ``nn.Conv2d`` initialisation (Kaiming uniform).
        """
        nn.init.kaiming_uniform_(self.init_linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.init_linear.bias)

    # ── forward pass ────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate an image and multi-resolution skip features from ``z``.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape ``[B, latent_dim]``.  Values are
            typically drawn from ``𝒩(0, I)``.

        Returns
        -------
        final_image : torch.Tensor
            Generated RGB image of shape ``[B, 3, image_size, image_size]``
            with pixel values in ``[-1, 1]``.
        skip_features : list of torch.Tensor
            Intermediate feature maps from each ``UpBlock``, at increasing
            spatial resolutions ``[4×2, 4×4, …, image_size]``.  The list
            length equals ``self.num_blocks``.  Each tensor has shape
            ``[B, C_i, H_i, W_i]`` where ``H_i = W_i = 4 × 2^{i+1}``.
        """
        B = z.shape[0]

        # ── 1. Linear projection → reshape to 4×4 spatial map ─────────
        h = self.init_linear(z)  # [B, C₀ * 4 * 4]
        h = h.view(B, self._init_channels, self._init_spatial, self._init_spatial)

        # ── 2. Pass through each UpBlock, collecting skip features ────
        skip_features: List[torch.Tensor] = []
        for block_idx, block in enumerate(self.up_blocks):
            # Determine whether to checkpoint this block:
            #   - Global flag use_checkpointing must be True.
            #   - Either checkpoint_block_indices is empty (all blocks)
            #     or this block's index is in the set.
            _use_ckpt = (
                self.use_checkpointing
                and self.training
                and (len(self.checkpoint_block_indices) == 0
                     or block_idx in self.checkpoint_block_indices)
            )
            if _use_ckpt:
                # Gradient checkpointing: recompute activations during
                # backward to save VRAM.
                h, skip = checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h, skip = block(h)
            skip_features.append(skip)

        # ── 3. Final output convolution + Tanh activation ─────────────
        h = self.final_conv(h)
        final_image = self.final_activation(h)

        return final_image, skip_features

    # ── representation ──────────────────────────────────────────────────

    def extra_repr(self) -> str:
        ckpt_info = (
            f"checkpointing={self.use_checkpointing}, "
            f"ckpt_blocks={len(self.checkpoint_block_indices) if self.checkpoint_block_indices else 'all'}"
        )
        return (
            f"latent_dim={self.latent_dim}, "
            f"gen_base_channels={self.gen_base_channels}, "
            f"image_size={self.image_size}, "
            f"num_blocks={self.num_blocks}, "
            f"spectral_norm={self.use_spectral_norm}, "
            f"{ckpt_info}, "
            f"group_norm={self.use_group_norm}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FeatureProjector — 1×1 conv projection for skip features
# ──────────────────────────────────────────────────────────────────────────────


class FeatureProjector(nn.Module):
    """Lightweight 1×1 convolution that projects a skip feature map to a
    target dimension ``D`` (the slot dimension used by the spatial pathway).

    Each projection consists of:
        Conv2d(1×1, in_channels → D) → BatchNorm2d(D) → LeakyReLU(0.2)

    Parameters
    ----------
    in_channels : int
        Number of channels in the incoming skip feature map (``C_i``).
    out_channels : int
        Target slot dimension ``D``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Initialise for stable training dynamics
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project *x* to the target slot dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input skip feature of shape ``[B, C_i, H, W]``.

        Returns
        -------
        torch.Tensor
            Projected feature of shape ``[B, D, H, W]``.
        """
        return self.activation(self.norm(self.conv(x)))

    def extra_repr(self) -> str:
        return (
            f"conv={self.conv.in_channels}→{self.conv.out_channels} (1×1)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SlotMLP — per-slot MLP with residual
# ──────────────────────────────────────────────────────────────────────────────


class SlotMLP(nn.Module):
    """A small two-layer MLP applied independently to each spatial slot.

    Architecture::

        Linear(D, 2D) → ReLU → Linear(2D, D)

    The output is designed to be used with a residual connection::

        Z_out = Z_in + SlotMLP(Z_in)

    Parameters
    ----------
    d_model : int
        Dimensionality of the slot features (``D``).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )
        # Small initialisation to keep early updates modest
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP to each slot.

        Parameters
        ----------
        x : torch.Tensor
            Slot features of shape ``[B, N, D]``.

        Returns
        -------
        torch.Tensor
            Transformed features of shape ``[B, N, D]``.
        """
        return self.net(x)

    def extra_repr(self) -> str:
        return f"d_model={self.net[0].in_features}"


# ──────────────────────────────────────────────────────────────────────────────
# SpatialHead — bounding box prediction pathway
# ──────────────────────────────────────────────────────────────────────────────


class SpatialHead(nn.Module):
    """Bounding box prediction pathway that consumes multi-resolution skip
    features from ``ContentDecoder`` and produces box proposals.

    This module is the second core component of the ILGAN generator.  It uses
    a set of **learned spatial queries** (one per potential bounding box) and
    processes them through a cascade of ``SpatialContentCrossAttention``
    (SCCA) modules, one per resolution level of the skip features.

    Coarse-to-fine refinement
    -------------------------
    At each resolution level (from lowest to highest), the spatial queries
    cross-attend with the projected skip feature map.  The attended features
    are added to the previous level's features (residual), then passed through
    a per-slot MLP with another residual.  This allows the queries to
    progressively refine their spatial focus from coarse (low-res) to fine
    (high-res) details.

    Output projections
    ------------------
    After processing all levels, each slot's final feature vector is
    independently projected to:

    - **box_coords** ``[B, max_boxes, 4]``: ``(x_center, y_center, width, height)``
      in normalised ``[0, 1]`` coordinates (Sigmoid-activated).
    - **class_logits** ``[B, max_boxes, num_classes]``: raw class logits (no
      activation, for use with cross-entropy).
    - **confidence** ``[B, max_boxes, 1]``: objectness score in ``[0, 1]``
      (Sigmoid-activated), indicating whether the slot contains a valid object.

    Collapse prevention
    -------------------
    The SCCA modules compute a **repulsion loss** that penalises slots whose
    attention centres of mass are closer than a threshold.  This prevents
    multiple slots from attending to the same spatial region, which in turn
    prevents the predicted bounding boxes from collapsing to a single location.

    Parameters
    ----------
    max_boxes : int
        Maximum number of bounding boxes to predict per image (``num_slots``).
    skip_channels : list of int
        Channel counts of each skip feature from ``ContentDecoder._skip_channels``,
        in order from lowest to highest resolution.
    slot_dim : int
        Dimensionality ``D`` of the spatial slot features.  Default is
        ``gen_base_channels * 2`` (e.g. 128 when base channels = 64).
    num_classes : int
        Number of object classes for classification logits.
    proj_channels : int, optional
        Inner dimension for the SCCA attention computation.  If ``None``,
        defaults to ``slot_dim``.  Must be divisible by ``num_heads``.
    num_heads : int, optional
        Number of attention heads in each SCCA module.  (default: 8)
    repulsion_threshold : float, optional
        Minimum normalised distance between slot centres of mass before
        repulsion is incurred.  (default: 0.2)
    repulsion_weight : float, optional
        Coefficient scaling the repulsion loss.  (default: 1.0)
    entropy_weight : float, optional
        Coefficient scaling the entropy regularisation (encourages compact
        attention).  (default: 0.1)

    Raises
    ------
    ValueError
        If ``skip_channels`` is empty.
    """

    def __init__(
        self,
        max_boxes: int,
        skip_channels: List[int],
        slot_dim: int,
        num_classes: int,
        proj_channels: int | None = None,
        num_heads: int = 8,
        repulsion_threshold: float = 0.2,
        repulsion_weight: float = 1.0,
        entropy_weight: float = 0.1,
    ) -> None:
        super().__init__()

        if len(skip_channels) == 0:
            raise ValueError("skip_channels must be non-empty.")

        self.num_slots = max_boxes
        self.slot_dim = slot_dim
        self.num_classes = num_classes
        self.num_levels = len(skip_channels)

        # Use slot_dim as proj_channels if not specified
        if proj_channels is None:
            proj_channels = slot_dim

        # ── 1. Learned spatial query embeddings ────────────────────────
        # Shape: [1, num_slots, D] — broadcast over batch dimension
        self.spatial_queries = nn.Parameter(
            torch.randn(1, max_boxes, slot_dim) * 0.02,
        )

        # ── 2. Build a FeatureProjector + SCCA module per resolution level ──
        self.feature_projectors = nn.ModuleList()
        self.scca_modules = nn.ModuleList()

        for ch_in in skip_channels:
            # Project skip features from C_i to D
            self.feature_projectors.append(
                FeatureProjector(in_channels=ch_in, out_channels=slot_dim),
            )
            # SCCA: content = projected skip (D channels), slots = queries (D dim)
            self.scca_modules.append(
                SpatialContentCrossAttention(
                    content_channels=slot_dim,
                    slot_dim=slot_dim,
                    proj_channels=proj_channels,
                    num_heads=num_heads,
                    repulsion_threshold=repulsion_threshold,
                    repulsion_weight=repulsion_weight,
                    entropy_weight=entropy_weight,
                ),
            )

        # ── 3. Per-slot MLP (shared across all resolution levels) ──────
        self.slot_mlp = SlotMLP(d_model=slot_dim)

        # ── 4. Output heads (applied independently per slot) ───────────
        # Box coordinates: (x_center, y_center, width, height) in [0, 1]
        self.box_head = nn.Sequential(
            nn.Linear(slot_dim, 4),
            nn.Sigmoid(),
        )
        # Class logits: [B, N, num_classes] (no activation)
        self.class_head = nn.Linear(slot_dim, num_classes)
        # Confidence score: [B, N, 1] in [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(slot_dim, 1),
            nn.Sigmoid(),
        )

        # ── Parameter initialisation ────────────────────────────────────
        self._init_output_heads()

    # ── parameter initialisation ─────────────────────────────────────────

    def _init_output_heads(self) -> None:
        """Initialise the output head weights for stable training.

        - Box head: small normal initialisation to start with tight boxes
          near the centre.
        - Class head: Kaiming uniform for logits; zero bias so initial
          predictions are uniform over classes.
        - Confidence head: small normal initialisation; bias set to give
          an initial confidence of ~0.1 (prevent over-confidence at start).
        """
        # Box head
        nn.init.normal_(self.box_head[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.box_head[0].bias)

        # Class head
        nn.init.kaiming_uniform_(self.class_head.weight, a=math.sqrt(5))
        nn.init.zeros_(self.class_head.bias)

        # Confidence head
        nn.init.normal_(self.confidence_head[0].weight, mean=0.0, std=0.01)
        # Initialise bias so sigmoid(bias) ≈ 0.1 → bias ≈ ln(0.1/0.9) ≈ -2.2
        nn.init.constant_(self.confidence_head[0].bias, -2.197)

    # ── forward pass ────────────────────────────────────────────────────

    def forward(
        self,
        skip_features: List[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """Predict bounding boxes from multi-resolution skip features.

        Parameters
        ----------
        skip_features : list of torch.Tensor
            Multi-resolution feature maps from ``ContentDecoder``, ordered
            from **lowest** resolution to **highest** resolution (e.g.
            ``[8×8, 16×16, 32×32, 64×64, 128×128]``).  Each tensor has shape
            ``[B, C_i, H_i, W_i]``.

        Returns
        -------
        boxes : torch.Tensor
            Predicted bounding box coordinates, shape ``[B, max_boxes, 4]``
            with values ``(x_center, y_center, width, height)`` in ``[0, 1]``.
        class_logits : torch.Tensor
            Raw class logits, shape ``[B, max_boxes, num_classes]``.
        confidences : torch.Tensor
            Objectness confidence scores, shape ``[B, max_boxes, 1]`` with
            values in ``[0, 1]``.
        aux_losses : dict of str -> torch.Tensor
            Aggregated auxiliary losses from all SCCA modules.  Contains:
            - ``"entropy"``: sum of entropy losses across levels (scalar).
            - ``"repulsion"``: sum of repulsion losses across levels (scalar).
            - ``"entropy_weighted"``: sum of weighted entropy (scalar).
            - ``"repulsion_weighted"``: sum of weighted repulsion (scalar).

        Raises
        ------
        ValueError
            If ``len(skip_features) != self.num_levels``.
        """
        if len(skip_features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} skip features, "
                f"got {len(skip_features)}."
            )

        B = skip_features[0].shape[0]
        device = skip_features[0].device

        # ── 1. Expand spatial queries to batch size ────────────────────
        # ``spatial_queries`` has shape [1, num_slots, D]; expand to [B, N, D]
        Z: torch.Tensor = self.spatial_queries.expand(B, -1, -1)  # [B, N, D]

        # ── 2. Aggregate auxiliary losses from all SCCA modules ───────
        all_aux_losses: Dict[str, torch.Tensor] = {}

        # ── 3. Coarse-to-fine processing through resolution levels ────
        for level_idx in range(self.num_levels):
            skip = skip_features[level_idx]  # [B, C_i, H_i, W_i]

            # a. Project skip feature to dimension D
            skip_proj: torch.Tensor = self.feature_projectors[level_idx](skip)
            # skip_proj: [B, D, H_i, W_i]

            # b. Store previous Z for residual connection
            prev_Z: torch.Tensor = Z

            # c. SCCA: cross-attend spatial queries with projected skip
            Z_attended, attn_maps, aux = self.scca_modules[level_idx](skip_proj, Z)
            # Z_attended: [B, N, D]

            # d. Accumulate auxiliary losses
            for key, value in aux.items():
                if key not in all_aux_losses:
                    all_aux_losses[key] = value
                else:
                    all_aux_losses[key] = all_aux_losses[key] + value

            # e. Residual connection: Z = Z_attended + previous_Z
            Z = Z_attended + prev_Z

            # f. Per-slot MLP with residual
            Z = Z + self.slot_mlp(Z)
            # Z is now the updated query for the next level

        # ── 4. Output projections ──────────────────────────────────────
        # Box coordinates: [B, N, 4]
        boxes: torch.Tensor = self.box_head(Z)

        # Class logits: [B, N, num_classes]
        class_logits: torch.Tensor = self.class_head(Z)

        # Confidence scores: [B, N, 1]
        confidences: torch.Tensor = self.confidence_head(Z)

        return boxes, class_logits, confidences, all_aux_losses

    # ── representation ──────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"max_boxes={self.num_slots}, "
            f"slot_dim={self.slot_dim}, "
            f"num_classes={self.num_classes}, "
            f"num_levels={self.num_levels}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ILGANGenerator — unified generator that composes ContentDecoder and SpatialHead
# ──────────────────────────────────────────────────────────────────────────────


class ILGANGenerator(nn.Module):
    """Unified ILGAN generator that simultaneously produces an image and
    bounding box predictions from a single latent vector ``z``.

    The generator composes two sub-modules:

    - ``ContentDecoder``: transforms ``z`` into a full-resolution RGB image
      and a list of multi-resolution skip feature maps.
    - ``SpatialHead``: consumes the skip features to predict bounding boxes,
      class logits, and confidence scores through coarse-to-fine
      cross-attention refinement.

    The generator adds several mechanisms on top of the sub-modules:

    - **Spectral normalisation** on the final output convolution of the
      ContentDecoder and on the SpatialHead's output projection heads,
      stabilising GAN training.
    - **Learnable instance noise**: a scalar ``noise_std`` parameter that
      injects Gaussian noise into the latent vector during training,
      improving stochasticity and preventing mode collapse.
    - **Latent statistics tracking**: running mean and variance of all
      latent vectors passed through the generator, enabling GAN inversion
      or style-mixing later.
    - **Gradient checkpointing propagation**: a ``set_gradient_checkpointing``
      method that enables/disables checkpointing in the ContentDecoder.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.  The following keys are used:

        - ``model.latent_dim``: dimensionality of the latent vector ``z``.
        - ``model.gen_base_channels``: base channel count for the decoder.
        - ``model.max_boxes``: maximum number of bounding boxes per image.
        - ``model.num_classes``: number of object classes.
        - ``model.num_attention_heads``: number of attention heads in SCCA.
        - ``data.image_size``: spatial size of generated images (square).
        - ``training.grad_checkpoint``: enable gradient checkpointing.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        # ── Extract parameters from config ──────────────────────────────
        latent_dim: int = config.model.latent_dim
        gen_base_channels: int = config.model.gen_base_channels
        image_size: int = config.data.image_size
        max_boxes: int = config.model.max_boxes
        num_classes: int = config.model.num_classes
        num_attention_heads: int = config.model.num_attention_heads
        use_checkpointing: bool = config.training.grad_checkpoint

        # Slot dimension: defaults to gen_base_channels * 2
        slot_dim: int = gen_base_channels * 2

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.num_classes = num_classes
        self.slot_dim = slot_dim

        # ── 1. ContentDecoder (image pathway) ───────────────────────────
        # Enable spectral norm for training stability
        self.content_decoder = ContentDecoder(
            latent_dim=latent_dim,
            gen_base_channels=gen_base_channels,
            image_size=image_size,
            use_spectral_norm=True,
            use_checkpointing=use_checkpointing,
            use_group_norm=False,
        )

        # ── 2. SpatialHead (bounding box pathway) ───────────────────────
        self.spatial_head = SpatialHead(
            max_boxes=max_boxes,
            skip_channels=self.content_decoder._skip_channels,
            slot_dim=slot_dim,
            num_classes=num_classes,
            proj_channels=slot_dim,
            num_heads=num_attention_heads,
            repulsion_threshold=0.2,
            repulsion_weight=1.0,
            entropy_weight=0.1,
        )

        # ── 3. Apply spectral normalisation to SpatialHead output heads
        #     in addition to the ContentDecoder's final_conv (already done
        #     inside ContentDecoder with use_spectral_norm=True).
        self.spatial_head.box_head[0] = nn.utils.spectral_norm(
            self.spatial_head.box_head[0],
        )
        self.spatial_head.class_head = nn.utils.spectral_norm(
            self.spatial_head.class_head,
        )
        self.spatial_head.confidence_head[0] = nn.utils.spectral_norm(
            self.spatial_head.confidence_head[0],
        )

        # ── 4. Learnable instance noise ────────────────────────────────
        # A scalar parameter ``noise_std`` that controls the standard
        # deviation of Gaussian noise injected into the latent vector
        # during training.  Initialised to a small value (0.01) so that
        # initial training is mostly deterministic.
        self.noise_std = nn.Parameter(torch.tensor(0.01))

        # ── 5. Latent statistics tracking ──────────────────────────────
        # Running mean and variance of latent vectors seen during training.
        # These are buffers (not parameters) so they are saved/loaded with
        # the state dict but not updated by gradient descent.
        self.register_buffer(
            "_latent_mean", torch.zeros(latent_dim),
        )
        self.register_buffer(
            "_latent_var", torch.ones(latent_dim),
        )
        self.register_buffer(
            "_latent_count", torch.tensor(0, dtype=torch.long),
        )

    # ── forward pass ─────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One-shot generation of image and bounding boxes from a latent
        vector.

        The forward pass:

        1. Optionally injects learnable instance noise into ``z`` during
           training mode.
        2. Updates the running latent statistics.
        3. Runs ``z`` through ``ContentDecoder`` to obtain the image and
           skip features.
        4. Runs the skip features through ``SpatialHead`` to obtain boxes,
           class logits, confidences, and auxiliary losses.
        5. Returns a dictionary containing all outputs.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector of shape ``[B, latent_dim]``.  Values should
            typically be drawn from ``𝒩(0, I)``.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"image"``: generated RGB image, shape ``[B, 3, H, W]``,
              pixel values in ``[-1, 1]``.
            - ``"boxes"``: bounding box coordinates, shape
              ``[B, max_boxes, 4]``, values ``(cx, cy, w, h)`` in ``[0, 1]``.
            - ``"class_logits"``: raw classification logits, shape
              ``[B, max_boxes, num_classes]``.
            - ``"confidences"``: objectness confidence scores, shape
              ``[B, max_boxes, 1]``, values in ``[0, 1]``.
            - ``"aux_losses"``: dictionary of auxiliary losses from SCCA
              modules (entropy and repulsion).
        """
        B = z.shape[0]
        device = z.device

        # ── 1. Learnable instance noise (training only) ─────────────────
        if self.training:
            # Inject Gaussian noise scaled by the learnable noise_std.
            # This adds stochasticity to the latent, preventing the
            # generator from overfitting to a fixed set of latent vectors
            # and helping to avoid mode collapse.
            noise = torch.randn_like(z) * self.noise_std
            z = z + noise

        # ── 2. Update latent statistics ─────────────────────────────────
        self._update_latent_statistics(z)

        # ── 3. ContentDecoder → image + skip features ──────────────────
        image, skip_features = self.content_decoder(z)

        # ── 4. SpatialHead → boxes, class_logits, confidences, aux losses
        boxes, class_logits, confidences, aux_losses = self.spatial_head(
            skip_features,
        )

        # ── 5. Build aux dictionary ──────────────────────────────────────
        # The loss aggregator expects an "aux" key containing attention maps
        # and skip features.  We create dummy attention maps from the
        # highest-resolution skip feature's spatial size.
        B = z.shape[0]
        h, w = skip_features[-1].shape[2:]
        attention_maps = torch.full(
            (B, self.max_boxes, h, w),
            1.0 / (h * w),
            device=z.device,
        )

        return {
            "image": image,
            "boxes": boxes,
            "class_logits": class_logits,
            "confidences": confidences,
            "aux_losses": aux_losses,
            "aux": {
                "attention_maps": attention_maps,
                "skip_features": skip_features,
            },
        }

    # ── latent statistics ────────────────────────────────────────────────

    def _update_latent_statistics(self, z: torch.Tensor) -> None:
        """Update the running mean and variance of latent vectors.

        Uses Welford's online algorithm for numerically stable computation
        of mean and variance.

        Parameters
        ----------
        z : torch.Tensor
            Batch of latent vectors ``[B, latent_dim]``.
        """
        with torch.no_grad():
            B = z.shape[0]
            # Running count
            count = self._latent_count.item()
            new_count = count + B

            # Batch mean and variance
            batch_mean = z.mean(dim=0)  # [latent_dim]
            batch_var = z.var(dim=0, unbiased=False)  # [latent_dim]

            # Welford's merge of running statistics with batch statistics
            if count == 0:
                # First batch: initialise
                self._latent_mean.copy_(batch_mean)
                self._latent_var.copy_(batch_var)
            else:
                # Merge: new_mean = weighted average of old_mean and batch_mean
                old_mean = self._latent_mean.clone()
                old_var = self._latent_var.clone()

                new_mean = (count * old_mean + B * batch_mean) / new_count

                # Variance merge (Welford): Var_total = Var_old + Var_batch + correction
                delta = old_mean - batch_mean
                new_var = (
                    count * old_var + B * batch_var +
                    count * B / new_count * delta * delta
                ) / new_count

                self._latent_mean.copy_(new_mean)
                self._latent_var.copy_(new_var)

            self._latent_count.fill_(new_count)

    def get_latent_statistics(self) -> Dict[str, torch.Tensor]:
        """Retrieve the running mean and variance of latent vectors.

        These statistics are accumulated across all forward passes during
        training and can be used for:

        - GAN inversion: find the latent that best reconstructs a given image.
        - Style-mixing: compare statistics across different training phases.
        - Truncation trick: sample from ``𝒩(mean, var)`` instead of
          ``𝒩(0, I)`` for more focused generation.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``"mean"``: ``[latent_dim]`` tensor — running mean.
            - ``"var"``: ``[latent_dim]`` tensor — running variance.
            - ``"count"``: ``int`` — number of latent vectors seen.
        """
        return {
            "mean": self._latent_mean.clone(),
            "var": self._latent_var.clone(),
            "count": self._latent_count.item(),
        }

    # ── noise generation ─────────────────────────────────────────────────

    @staticmethod
    def generate_noise(
        num_samples: int,
        latent_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate random latent vectors from the standard normal
        distribution ``𝒩(0, I)``.

        This is a convenience factory for producing the input to the
        generator.

        Parameters
        ----------
        num_samples : int
            Number of latent vectors to generate (batch size ``B``).
        latent_dim : int
            Dimensionality of each latent vector.
        device : torch.device
            Target device (CPU or CUDA).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[num_samples, latent_dim]`` with values drawn
            from ``𝒩(0, I)``.
        """
        return torch.randn(num_samples, latent_dim, device=device)

    # ── gradient checkpointing ───────────────────────────────────────────

    def set_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing in the ContentDecoder.

        Gradient checkpointing reduces VRAM usage during training by
        recomputing intermediate activations during the backward pass,
        at the cost of a small compute overhead (≈20–30% slower per step).

        Parameters
        ----------
        enable : bool, optional
            If True, enable gradient checkpointing; if False, disable it.
            (default: True)
        """
        self.content_decoder.use_checkpointing = enable

    # ── representation ──────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"image_size={self.image_size}, "
            f"max_boxes={self.max_boxes}, "
            f"num_classes={self.num_classes}, "
            f"slot_dim={self.slot_dim}, "
            f"noise_std={self.noise_std.item():.4f}"
        )


__all__ = [
    "ContentDecoder",
    "UpBlock",
    "SpectralNormConv2d",
    "FeatureProjector",
    "SlotMLP",
    "SpatialHead",
    "ILGANGenerator",
]