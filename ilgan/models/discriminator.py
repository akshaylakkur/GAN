"""
Image discriminator for the ILGAN dual-output GAN.

The ``ImageDiscriminator`` is a PatchGAN-style discriminator that evaluates
the realism of generated or real images.  Rather than producing a single
scalar per sample, it outputs a **spatial grid of realism scores** (local
scores) that provide fine-grained feedback to the generator, together with
a **single global realism score** per sample.

Architecture overview
---------------------
1. A sequence of ``DownBlock`` modules progressively downsample the input
   image while doubling the channel count.  Each block::

        Conv2d(4×4, stride=2, padding=1)
        → FrozenBatchNorm2d  (or InstanceNorm / LayerNorm)
        → LeakyReLU(0.2)

2. After the down-sampling stack, two Conv2d(3×3) layers with spectral
   normalisation produce a spatial score map of shape
   ``[B, 1, grid_h, grid_w]`` where ``grid_h = grid_w = image_size / 2^L``
   and ``L`` is the number of down blocks.  Each spatial position scores the
   realism of a local image patch.

3. A **global realism head** applies global average pooling over the final
   feature map followed by a linear layer, producing a single scalar per
   sample.

Collapse prevention — minibatch discrimination
----------------------------------------------
An optional minibatch standard deviation feature is concatenated to the final
feature map before the score heads.  This gives the discriminator access to
batch-level statistics, making it harder for the generator to collapse all
samples to the same output (mode collapse).  When enabled, a single-channel
feature map containing the average per-position standard deviation across the
batch is concatenated to the feature map.

FrozenBatchNorm2d
-----------------
Standard ``nn.BatchNorm2d`` accumulates running statistics during training.
In a GAN setting, real and fake batches have different distributions, so
leaking statistics between them is undesirable.  ``FrozenBatchNorm2d`` never
updates its running statistics — they remain at their initial (or calibrated)
values for both training and evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilgan.models.generator import SpectralNormConv2d

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# FrozenBatchNorm2d — batch normalisation with frozen statistics
# ──────────────────────────────────────────────────────────────────────────────


class FrozenBatchNorm2d(nn.Module):
    """Batch normalisation with frozen (non-trainable) running statistics.

    Unlike ``nn.BatchNorm2d``, this module **never** updates ``running_mean``
    or ``running_var`` during training.  Both training and evaluation use the
    same frozen statistics.  This prevents statistics leakage between real
    and fake batches in GAN training, where the two distributions differ.

    The affine parameters ``weight`` (gamma) and ``bias`` (beta) are trainable
    as usual.  The running statistics are initialised to ``(0, 1)`` by default
    and can be calibrated to the data distribution via the ``calibrate()``
    method at any point.

    Parameters
    ----------
    num_features : int
        Number of input channels ``C``.
    eps : float, optional
        Small constant added to the variance for numerical stability.
        (default: ``1e-5``)
    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Trainable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Frozen running statistics (buffers, not parameters)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frozen batch normalisation.

        Always uses ``training=False``, so the running statistics are never
        updated regardless of the module's training mode.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[B, C, H, W]``.

        Returns
        -------
        torch.Tensor
            Normalised output of the same shape.
        """
        return F.batch_norm(
            x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,  # Never update running stats
            eps=self.eps,
        )

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        """Calibrate the running statistics from a batch of data.

        This sets ``running_mean`` and ``running_var`` to the mean and
        variance of the provided batch.  Call this once with a representative
        batch (e.g., the first training batch) to initialise sensible
        statistics.

        Parameters
        ----------
        x : torch.Tensor
            Batch of feature maps of shape ``[B, C, H, W]`` from which to
            compute statistics.
        """
        batch_mean = x.mean(dim=(0, 2, 3))
        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
        self.running_mean.copy_(batch_mean)
        self.running_var.copy_(batch_var)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation factory
# ──────────────────────────────────────────────────────────────────────────────


def _build_norm(norm_type: str, num_features: int, num_groups: int = 4) -> nn.Module:
    """Construct a normalisation layer by type name.

    Parameters
    ----------
    norm_type : str
        One of ``"frozen_bn"``, ``"instance"``, ``"layer"``, or
        ``"group"``.
    num_features : int
        Number of channels (used by frozen_bn, instance norms).
    num_groups : int, optional
        Number of groups for group normalisation (default: 4).

    Returns
    -------
    nn.Module
        The normalisation layer.

    Raises
    ------
    ValueError
        If ``norm_type`` is unknown.
    """
    if norm_type == "frozen_bn":
        return FrozenBatchNorm2d(num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    elif norm_type == "layer":
        # GroupNorm with 1 group is equivalent to LayerNorm applied over
        # (C, H, W) dimensions for each sample independently.
        return nn.GroupNorm(num_groups=1, num_channels=num_features)
    elif norm_type == "group":
        num_groups = min(num_groups, num_features)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. "
            f"Expected one of: frozen_bn, instance, layer, group."
        )


# ──────────────────────────────────────────────────────────────────────────────
# DownBlock — single down-sampling stage for the discriminator
# ──────────────────────────────────────────────────────────────────────────────


class DownBlock(nn.Module):
    """One down-sampling block for the discriminator.

    Operations (in order)
    ---------------------
    1. Conv2d(4×4, stride=2, padding=1) — halves spatial resolution.
    2. Normalisation (FrozenBatchNorm2d, InstanceNorm, or LayerNorm).
    3. LeakyReLU(0.2, inplace).

    Spectral normalisation is applied to the convolution weight to stabilise
    GAN training (Miyato et al., 2018).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels (always >= in_channels in the discriminator).
    use_spectral_norm : bool, optional
        If True, apply ``nn.utils.spectral_norm`` to the convolutional
        weight.  (default: True)
    norm_type : str, optional
        Type of normalisation to use.  One of ``"frozen_bn"``, ``"instance"``,
        ``"layer"``, ``"group"``.  (default: ``"frozen_bn"``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = True,
        norm_type: str = "frozen_bn",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_spectral_norm = use_spectral_norm
        self.norm_type = norm_type

        # 4x4 convolution with stride 2 and padding 1 halves spatial size
        self.conv = SpectralNormConv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            use_spectral_norm=use_spectral_norm,
        )

        # Normalisation layer
        self.norm = _build_norm(norm_type, out_channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the down-sampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``[B, in_channels, H, W]``.

        Returns
        -------
        torch.Tensor
            Output feature map of shape ``[B, out_channels, H/2, W/2]``.
        """
        h = self.conv(x)
        h = self.norm(h)
        h = self.activation(h)
        return h

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"spectral_norm={self.use_spectral_norm}, "
            f"norm={self.norm_type}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Minibatch standard deviation helper
# ──────────────────────────────────────────────────────────────────────────────


def minibatch_stddev(x: torch.Tensor) -> torch.Tensor:
    """Compute minibatch standard deviation and concatenate as an extra
    feature channel.

    For each spatial location and channel, the standard deviation across
    the batch is computed.  These per-position-per-channel standard
    deviations are then averaged to a single scalar, which is expanded
    to a ``[B, 1, H, W]`` feature map and concatenated to ``x`` along the
    channel dimension.

    This technique, introduced by Karras et al. (StyleGAN, 2019), gives the
    discriminator access to batch-level statistics, making it harder for the
    generator to collapse all samples to the same output (mode collapse).

    The function **always** returns ``[B, C + 1, H, W]`` for consistency,
    even when ``B < 2`` (in which case the standard deviation is zero).

    Parameters
    ----------
    x : torch.Tensor
        Feature map of shape ``[B, C, H, W]``.

    Returns
    -------
    torch.Tensor
        Augmented feature map of shape ``[B, C + 1, H, W]``.
    """
    B, C, H, W = x.shape

    # Compute std over the batch dimension at each position and channel
    if B < 2:
        # With B < 2, std over the batch is undefined.  Return a zero
        # channel to maintain consistent output shape [B, C+1, H, W].
        std_feat_map = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        return torch.cat([x, std_feat_map], dim=1)

    x_mean = x.mean(dim=0, keepdim=True)  # [1, C, H, W]
    x_var = ((x - x_mean) ** 2).mean(dim=0, keepdim=False)  # [C, H, W]
    x_std = torch.sqrt(x_var + _EPS)  # [C, H, W]

    # Average over all remaining dimensions (C, H, W) -> scalar
    std_feat = x_std.mean()  # scalar

    # Expand and concatenate
    std_feat_map = std_feat.expand(B, 1, H, W)  # [B, 1, H, W]
    return torch.cat([x, std_feat_map], dim=1)  # [B, C + 1, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# ImageDiscriminator -- PatchGAN-style image discriminator
# ──────────────────────────────────────────────────────────────────────────────


class ImageDiscriminator(nn.Module):
    """PatchGAN-style discriminator that evaluates image realism.

    Given an input image of shape ``[B, 3, H, W]``, the discriminator
    produces two outputs:

    - ``local_scores`` (``[B, 1, grid_h, grid_w]``): a spatial grid of
      realism scores.  Each position scores the realism of a local image
      patch.  The grid resolution is ``grid_h = grid_w = image_size / 2^L``
      where ``L`` is the number of down blocks.  For the default
      configuration (L chosen such that the final spatial size is 4),
      this gives a 4x4 grid.

    - ``global_score`` (``[B, 1]``): a single scalar per sample representing
      the overall realism of the entire image.  Obtained by global average
      pooling the final feature map followed by a linear projection.

    Architecture
    ------------
    1. **Down blocks**: a sequence of ``DownBlock`` modules.  The number of
       blocks is ``L = log2(image_size) - 2``, so the final spatial size is
       4x4.

       Channel counts double each block:
       ``disc_base_channels -> 2*d -> 4*d -> 8*d -> ...``
       capped at a maximum of ``disc_base_channels * 16``.

    2. **Minibatch standard deviation** (optional): a single-channel std-dev
       feature is concatenated to the final feature map before the score
       heads.

    3. **Local score head**: two Conv2d(3x3) layers with spectral
       normalisation.  The first keeps the channel count the same, the
       second projects to 1 channel.

    4. **Global score head**: global average pooling -> ``nn.Linear`` -> 1.
       Uses the feature map **before** minibatch stddev is applied, so
       the linear layer dimension always matches ``_final_channels``.

    Collapse prevention mechanisms
    ------------------------------
    - **Spectral normalisation** on all convolutional layers constrains the
      Lipschitz constant of the discriminator, preventing gradient
      explosions.
    - **Minibatch discrimination** gives the discriminator access to
      batch-level statistics, making mode collapse immediately costly.
    - **FrozenBatchNorm2d** prevents statistics leakage between real and
      fake batches, stabilising the adversarial game.

    Parameters
    ----------
    disc_base_channels : int
        Base channel count for the first down block.  Channels double each
        subsequent block.
    image_size : int
        Input image spatial size (square).  Must be a power of two >= 8.
    use_spectral_norm : bool, optional
        Apply spectral normalisation to all convolutional layers (in down
        blocks and score heads).  (default: True)
    norm_type : str, optional
        Type of normalisation for down blocks.  One of ``"frozen_bn"``,
        ``"instance"``, ``"layer"``, ``"group"``.  (default: ``"frozen_bn"``)
    use_minibatch_stddev : bool, optional
        Enable minibatch standard deviation feature concatenation before the
        score heads.  (default: True)
    max_channels : int, optional
        Maximum number of channels in any down block.  If ``None``, defaults
        to ``disc_base_channels * 16``.  (default: None)

    Raises
    ------
    ValueError
        If ``image_size`` is not a power of two or is < 8.
    """

    def __init__(
        self,
        disc_base_channels: int,
        image_size: int,
        use_spectral_norm: bool = True,
        norm_type: str = "frozen_bn",
        use_minibatch_stddev: bool = True,
        max_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # -- Validate image_size ---------------------------------------------
        if image_size < 8 or (image_size & (image_size - 1)) != 0:
            raise ValueError(
                f"image_size must be a power of two >= 8, got {image_size}."
            )

        self.disc_base_channels = disc_base_channels
        self.image_size = image_size
        self.use_spectral_norm = use_spectral_norm
        self.norm_type = norm_type
        self.use_minibatch_stddev = use_minibatch_stddev

        # Compute the number of down blocks to reach a 4x4 spatial grid
        # (stride 2 each block -> 2^L = image_size / 4 -> L = log2(image_size) - 2)
        self.num_blocks = int(math.log2(image_size)) - 2  # L
        if max_channels is None:
            max_channels = disc_base_channels * 16
        self._max_channels = max_channels

        # -- 1. Build the down-sampling blocks --------------------------------
        self.down_blocks = nn.ModuleList()

        in_ch = 3  # Input: RGB image
        for block_idx in range(self.num_blocks):
            out_ch = min(
                disc_base_channels * (2 ** block_idx),
                max_channels,
            )
            block = DownBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                use_spectral_norm=use_spectral_norm,
                norm_type=norm_type,
            )
            self.down_blocks.append(block)
            in_ch = out_ch

        # Store the final feature channel count after down blocks
        self._final_channels = in_ch

        # -- 2. Local score head -- two Conv2d(3x3) with spectral norm --------
        # The local score head operates AFTER minibatch stddev, so it
        # potentially has an extra channel.
        local_input_channels = in_ch + (1 if use_minibatch_stddev else 0)
        self.score_conv1 = SpectralNormConv2d(
            local_input_channels,
            in_ch,
            kernel_size=3,
            padding=1,
            use_spectral_norm=use_spectral_norm,
        )
        self.score_conv2 = SpectralNormConv2d(
            in_ch,
            1,
            kernel_size=3,
            padding=1,
            use_spectral_norm=use_spectral_norm,
        )
        self.score_activation = nn.LeakyReLU(0.2, inplace=True)

        # -- 3. Global realism head -------------------------------------------
        # The global head operates on the feature map BEFORE minibatch
        # stddev, so its linear layer dimension is always _final_channels.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_head = nn.Linear(in_ch, 1)

        # Also apply spectral norm to the global head for consistency
        if use_spectral_norm:
            self.global_head = nn.utils.spectral_norm(self.global_head)

        # -- Parameter initialisation -----------------------------------------
        self.reset_parameters()

    # -- parameter initialisation ---------------------------------------------

    def reset_parameters(self) -> None:
        """Initialise the global head weights with Kaiming uniform.

        The convolutional weights are initialised by their default PyTorch
        initialisation (Kaiming uniform for Conv2d).  The linear head is
        initialised with a small normal to keep initial global scores near
        zero.
        """
        if hasattr(self.global_head, "weight_orig"):
            nn.init.normal_(self.global_head.weight_orig, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.global_head.weight, mean=0.0, std=0.02)
        if hasattr(self.global_head, "bias") and self.global_head.bias is not None:
            nn.init.zeros_(self.global_head.bias)

    # -- forward pass ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the realism of input images.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape ``[B, 3, H, W]`` with pixel values in
            any range (typically ``[-1, 1]`` for GAN-generated or real
            images).

        Returns
        -------
        local_scores : torch.Tensor
            Spatial grid of realism scores, shape ``[B, 1, grid_h, grid_w]``.
            ``grid_h = grid_w = image_size / 2^{num_blocks}`` (typically 4).
            Each spatial position scores the realism of the corresponding
            local image patch.
        global_score : torch.Tensor
            Single scalar realism score per sample, shape ``[B, 1]``.
            Represents the overall realism of the entire image.
        """
        B = x.shape[0]

        # -- 1. Down-sampling blocks -----------------------------------------
        h = x
        for block in self.down_blocks:
            h = block(h)
        # h shape: [B, C_final, grid_h, grid_w] where grid_h = grid_w = 4

        # -- 2. Global realism head (before minibatch stddev) -----------------
        # Pool the feature map BEFORE minibatch stddev is applied so the
        # linear layer dimension is always _final_channels.
        h_global = self.global_pool(h)  # [B, C_final, 1, 1]
        h_global = h_global.view(B, -1)  # [B, C_final]
        global_score = self.global_head(h_global)  # [B, 1]

        # -- 3. (Optional) Minibatch standard deviation ----------------------
        if self.use_minibatch_stddev:
            h = minibatch_stddev(h)
        # h shape: [B, C_final + (1 if minibatch_stddev else 0), 4, 4]

        # -- 4. Local score head ---------------------------------------------
        h_local = self.score_conv1(h)
        h_local = self.score_activation(h_local)
        local_scores = self.score_conv2(h_local)  # [B, 1, 4, 4]

        return local_scores, global_score

    # -- utilities ------------------------------------------------------------

    @property
    def grid_size(self) -> int:
        """Return the spatial size of the local score grid.

        This is ``image_size / 2^{num_blocks}``, typically 4.
        """
        return self.image_size // (2 ** self.num_blocks)

    # -- representation -------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"disc_base_channels={self.disc_base_channels}, "
            f"image_size={self.image_size}, "
            f"num_blocks={self.num_blocks}, "
            f"final_channels={self._final_channels}, "
            f"grid_size={self.grid_size}, "
            f"spectral_norm={self.use_spectral_norm}, "
            f"norm={self.norm_type}, "
            f"minibatch_stddev={self.use_minibatch_stddev}"
        )


__all__ = [
    "FrozenBatchNorm2d",
    "DownBlock",
    "minibatch_stddev",
    "ImageDiscriminator",
]