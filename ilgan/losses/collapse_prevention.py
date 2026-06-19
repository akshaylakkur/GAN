"""
Collapse prevention losses for the ILGAN dual-output GAN.

This module implements the mathematical mechanisms that prevent both **image
mode collapse** (the generator producing the same image for all latents) and
**bounding box collapse** (all predicted boxes converging to the same spatial
location).  These are novel loss functions designed specifically for the
ILGAN's dual image-and-box generation paradigm.

Mathematical foundation
----------------------
Mode collapse in GANs occurs when the generator maps different latent codes
to the same output.  In the ILGAN setting, collapse can manifest in two ways:

1. **Image collapse**: all generated images look the same regardless of the
   latent vector.  This is the classic GAN mode collapse problem.
2. **Bounding box collapse**: all predicted bounding boxes converge to the
   same spatial location (e.g., the centre of the image), regardless of the
   latent vector or the image content.

The losses in this module address both forms of collapse through four
complementary mechanisms:

- **Attention entropy minimisation** (``attention_entropy_loss``): encourages
  each spatial slot's attention distribution to be concentrated on a compact
  region, which is desirable for localisation.  Low entropy = focused
  attention = precise bounding box.

- **Slot repulsion** (``repulsion_loss``): penalises pairs of slots whose
  attention centres of mass are too close together.  This is the primary
  mechanism against bounding box collapse: it forces slots to spread out
  across the spatial domain.

- **Feature diversity** (``feature_diversity_loss``): penalises intermediate
  feature maps that have high pairwise cosine similarity between spatial
  positions.  This encourages the generator to produce diverse features
  across the spatial grid, preventing the image from collapsing to a uniform
  appearance.

- **Latent diversity** (``latent_diversity_loss``): encourages the latent
  vectors in a batch to be well-separated.  If all latents are close together,
  the generator outputs will be similar.  This loss pushes them apart.

All functions are numerically stable (using ``_EPS = 1e-8`` for logarithms
and divisions), fully differentiable, and compatible with
``torch.cuda.amp.autocast()`` mixed precision training.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon for numerical stability in logarithms and divisions."""


# ──────────────────────────────────────────────────────────────────────────────
# Attention Entropy Loss
# ──────────────────────────────────────────────────────────────────────────────


def attention_entropy_loss(attention_maps: torch.Tensor) -> torch.Tensor:
    r"""Compute the mean entropy of slot attention distributions.

    For each slot ``n`` in each batch element ``b``, the attention map
    :math:`A_{b,n} \in \mathbb{R}^{H \times W}` is a probability distribution
    over spatial positions (it sums to 1).  The entropy of this distribution
    is:

    .. math::

        H_{b,n} = -\sum_{h=1}^{H} \sum_{w=1}^{W}
                   A_{b,n,h,w} \cdot \log(A_{b,n,h,w} + \varepsilon)

    where :math:`\varepsilon = 10^{-8}` prevents :math:`\log(0)`.

    The loss is the mean entropy over all slots and batch elements:

    .. math::

        \mathcal{L}_{\text{entropy}} = \frac{1}{B \cdot N} \sum_{b,n} H_{b,n}

    **Why this prevents collapse**: low entropy means the slot is focused on
    a compact spatial region, which is desirable for bounding box localisation.
    By minimising entropy, we encourage each slot to "zoom in" on a specific
    object or part, rather than spreading its attention diffusely.  This is a
    regulariser that complements the repulsion loss: entropy makes each slot
    precise, repulsion makes them distinct.

    Parameters
    ----------
    attention_maps : torch.Tensor
        Attention weight maps from SCCA modules, shape ``[B, N, H, W]``.
        Each ``[b, n]`` slice must be a valid probability distribution
        (non-negative and summing to 1 over ``H × W``).

    Returns
    -------
    torch.Tensor
        Scalar entropy loss (0-dimensional).  Returns 0.0 if the tensor
        has zero elements.

    Raises
    ------
    ValueError
        If ``attention_maps`` contains NaN or Inf values.

    Shape
    -----
    - Input: ``[B, N, H, W]``
    - Output: scalar ``[]``

    Gradient flow
    -------------
    Gradients flow through the attention maps to the parameters that produce
    them (the SCCA query and key projections).  The loss is differentiable
    everywhere except where ``A = 0`` (where the gradient of ``x log x`` is
    defined as 0 by convention in PyTorch).

    Example
    -------
    >>> # Uniform attention -> high entropy
    >>> B, N, H, W = 2, 4, 8, 8
    >>> uniform_attn = torch.ones(B, N, H, W) / (H * W)
    >>> loss = attention_entropy_loss(uniform_attn)
    >>> loss.item()  # doctest: +ELLIPSIS
    4.1588...

    >>> # Concentrated attention -> low entropy
    >>> concentrated_attn = torch.zeros(B, N, H, W)
    >>> concentrated_attn[:, :, 4, 4] = 1.0
    >>> loss = attention_entropy_loss(concentrated_attn)
    >>> abs(loss.item()) < 1e-4
    True

    >>> # Empty tensor
    >>> empty = torch.empty(0, 0, 0, 0)
    >>> loss = attention_entropy_loss(empty)
    >>> loss.item()
    0.0
    """
    if attention_maps.numel() == 0:
        return torch.tensor(0.0, device=attention_maps.device,
                            dtype=attention_maps.dtype)

    if torch.isnan(attention_maps).any():
        raise ValueError("attention_maps contains NaN values.")
    if torch.isinf(attention_maps).any():
        raise ValueError("attention_maps contains Inf values.")

    # Flatten spatial dimensions: [B, N, H, W] -> [B, N, H*W]
    A_flat = attention_maps.view(*attention_maps.shape[:2], -1)

    # Compute entropy per slot: H = -sum(p * log(p + eps))
    # Shape: [B, N]
    entropy = -(A_flat * torch.log(A_flat + _EPS)).sum(dim=-1)

    # Mean over all slots and batch elements
    loss = entropy.mean()

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Repulsion Loss
# ──────────────────────────────────────────────────────────────────────────────


def repulsion_loss(
    attention_maps: torch.Tensor,
    repulsion_threshold: float = 0.2,
) -> torch.Tensor:
    r"""Compute the pairwise repulsion loss between slot attention centres
    of mass.

    For each slot ``n`` in each batch element ``b``, compute the spatial
    centre of mass :math:`(cx_{b,n}, cy_{b,n})` in normalised ``[0, 1]``
    coordinates using the attention distribution as weights:

    .. math::

        cx_{b,n} = \sum_{h,w} A_{b,n,h,w} \cdot \frac{w}{W-1}
        \qquad
        cy_{b,n} = \sum_{h,w} A_{b,n,h,w} \cdot \frac{h}{H-1}

    Then compute a pairwise repulsion penalty:

    .. math::

        \mathcal{L}_{\text{rep}} = \frac{1}{B \cdot N_{\text{pairs}}}
            \sum_{b} \sum_{i < j}
            \max(0, \tau - d_{b,i,j})^2

    where :math:`d_{b,i,j} = \|(cx_{b,i}, cy_{b,i}) - (cx_{b,j}, cy_{b,j})\|_2`
    is the Euclidean distance between the centres of slots ``i`` and ``j``,
    and :math:`\tau` is ``repulsion_threshold``.

    **Why this prevents bounding box collapse**: if two slots attend to the
    same spatial region, their centres of mass will be close, incurring a
    repulsion penalty.  This forces slots to spread out across the spatial
    domain, ensuring that the predicted bounding boxes cover different
    objects or parts rather than collapsing to a single location.

    Parameters
    ----------
    attention_maps : torch.Tensor
        Attention weight maps, shape ``[B, N, H, W]``.  Each ``[b, n]``
        slice must be a valid probability distribution over ``H × W``.
    repulsion_threshold : float, optional
        Minimum allowed normalised distance between slot centres of mass.
        Slots whose centres are closer than this threshold incur a quadratic
        penalty.  Must be in ``(0, 1]``.  (default: ``0.2``)

    Returns
    -------
    torch.Tensor
        Scalar repulsion loss (0-dimensional).  Returns 0.0 if ``N <= 1``
        (no pairs to compute) or if the tensor has zero elements.

    Raises
    ------
    ValueError
        If ``repulsion_threshold`` is not in ``(0, 1]``.
    ValueError
        If ``attention_maps`` contains NaN or Inf values.

    Shape
    -----
    - Input: ``[B, N, H, W]``
    - Output: scalar ``[]``

    Gradient flow
    -------------
    Gradients flow through the attention maps to the parameters that produce
    them.  The loss is differentiable everywhere; the ``max(0, ...)``
    operation has a subgradient of 1 where the argument is positive and 0
    where it is negative.

    Example
    -------
    >>> # Two slots at opposite corners -> no repulsion
    >>> B, N, H, W = 1, 2, 8, 8
    >>> A = torch.zeros(B, N, H, W)
    >>> A[0, 0, 0, 0] = 1.0       # top-left
    >>> A[0, 1, 7, 7] = 1.0       # bottom-right
    >>> loss = repulsion_loss(A, repulsion_threshold=0.2)
    >>> abs(loss.item()) < 1e-6
    True

    >>> # Two slots at the same position -> high repulsion
    >>> A2 = torch.zeros(B, N, H, W)
    >>> A2[0, 0, 4, 4] = 1.0
    >>> A2[0, 1, 4, 4] = 1.0
    >>> loss2 = repulsion_loss(A2, repulsion_threshold=0.2)
    >>> loss2.item() > 0.0
    True

    >>> # Single slot -> no repulsion
    >>> A3 = torch.randn(2, 1, 8, 8).softmax(dim=-1).view(2, 1, 8, 8)
    >>> loss3 = repulsion_loss(A3)
    >>> abs(loss3.item()) < 1e-6
    True
    """
    if not (0.0 < repulsion_threshold <= 1.0):
        raise ValueError(
            f"repulsion_threshold must be in (0, 1], got {repulsion_threshold}."
        )

    if attention_maps.numel() == 0:
        return torch.tensor(0.0, device=attention_maps.device,
                            dtype=attention_maps.dtype)

    if torch.isnan(attention_maps).any():
        raise ValueError("attention_maps contains NaN values.")
    if torch.isinf(attention_maps).any():
        raise ValueError("attention_maps contains Inf values.")

    B, N, H, W = attention_maps.shape

    # No pairs to compute if N <= 1
    if N <= 1:
        return torch.tensor(0.0, device=attention_maps.device,
                            dtype=attention_maps.dtype)

    device = attention_maps.device
    dtype = attention_maps.dtype

    # ── 1. Build normalised coordinate grids ───────────────────────────
    # x in [0, 1], y in [0, 1] with (W) and (H) points respectively.
    x_grid = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
    y_grid = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)

    # Create a [HW] flat grid with x and y coordinates
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    x_coords = xx.reshape(-1)  # [HW]
    y_coords = yy.reshape(-1)  # [HW]

    # ── 2. Flatten attention maps ──────────────────────────────────────
    A_flat = attention_maps.view(B, N, H * W)  # [B, N, HW]

    # ── 3. Compute centre of mass for each slot ────────────────────────
    # cx[b, n] = sum_i A[b, n, i] * x_coords[i]
    cx = (A_flat * x_coords[None, None, :]).sum(dim=-1)  # [B, N]
    cy = (A_flat * y_coords[None, None, :]).sum(dim=-1)  # [B, N]

    # ── 4. Compute pairwise distances ──────────────────────────────────
    # dx[b, i, j] = cx[b, i] - cx[b, j], shape [B, N, N]
    dx = cx[:, :, None] - cx[:, None, :]   # [B, N, N]
    dy = cy[:, :, None] - cy[:, None, :]   # [B, N, N]
    dists = torch.sqrt(dx.pow(2) + dy.pow(2) + _EPS)  # [B, N, N]

    # ── 5. Compute repulsion penalty ───────────────────────────────────
    tau = repulsion_threshold
    # Upper-triangular indices (i < j)
    triu_idx = torch.triu_indices(N, N, offset=1, device=device)

    # Extract pairwise distances for i < j: [B, N_pairs]
    pair_dists = dists[:, triu_idx[0], triu_idx[1]]
    penalties = torch.clamp(tau - pair_dists, min=0.0).pow(2)

    # Mean over all pairs and batch
    loss = penalties.mean()

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Feature Diversity Loss
# ──────────────────────────────────────────────────────────────────────────────


def feature_diversity_loss(feature_maps: List[torch.Tensor]) -> torch.Tensor:
    r"""Compute the feature diversity loss to prevent image mode collapse.

    This is a **novel mechanism** designed specifically for ILGAN.  The idea
    is to encourage diversity in the intermediate feature representations of
    the generator.  If the generator is collapsing (producing the same image
    for all latents), the intermediate feature maps will also be similar
    across spatial positions.  By penalising pairwise similarity between
    feature vectors at different spatial positions, we force the generator
    to produce spatially diverse features, which in turn prevents image
    collapse.

    For each feature map :math:`F \in \mathbb{R}^{B \times C \times H \times W}`
    in the list, we:

    1. Normalise each spatial feature vector to unit norm:
       :math:`\hat{F}_{b,c,h,w} = F_{b,c,h,w} / \|F_{b,:,h,w}\|_2`
    2. Compute the pairwise cosine similarity matrix between all spatial
       positions:
       :math:`S_{b,(h,w),(h',w')} = \hat{F}_{b,:,h,w} \cdot \hat{F}_{b,:,h',w'}`
    3. Take the absolute value of all off-diagonal entries and compute the
       mean:
       :math:`\mathcal{L}_{\text{div}}^{(m)} = \frac{1}{B \cdot (HW)^2 - HW}
           \sum_{b} \sum_{i \neq j} |S_{b,i,j}|`

    The final loss is the mean over all feature maps in the list:

    .. math::

        \mathcal{L}_{\text{div}} = \frac{1}{M} \sum_{m=1}^{M}
            \mathcal{L}_{\text{div}}^{(m)}

    where :math:`M` is the number of feature maps (resolution levels).

    **Why this prevents image collapse**: if the generator is collapsing, the
    feature maps will be spatially uniform (all positions have similar
    features), leading to high pairwise cosine similarity.  By penalising
    this, we force the generator to produce spatially varied features, which
    is a necessary condition for generating diverse images.

    Parameters
    ----------
    feature_maps : list of torch.Tensor
        A list of feature maps from the ``ContentDecoder``'s skip features
        (or any intermediate layer).  Each tensor has shape
        ``[B, C_i, H_i, W_i]``.  The list can contain feature maps at
        different resolutions.

    Returns
    -------
    torch.Tensor
        Scalar feature diversity loss (0-dimensional).  Returns 0.0 if the
        list is empty or if any feature map has zero spatial positions.

    Raises
    ------
    ValueError
        If any feature map contains NaN or Inf values.

    Shape
    -----
    - Input: list of tensors, each ``[B, C_i, H_i, W_i]``
    - Output: scalar ``[]``

    Gradient flow
    -------------
    Gradients flow through all feature maps to the generator parameters that
    produce them.  The loss is fully differentiable.

    Example
    -------
    >>> # Two feature maps with diverse features -> low loss
    >>> B, C, H, W = 2, 16, 4, 4
    >>> f1 = torch.randn(B, C, H, W)  # random features are diverse
    >>> loss = feature_diversity_loss([f1])
    >>> loss.item()  # typically around 0.2-0.3 for random features
    ...               # doctest: +ELLIPSIS
    0.2...

    >>> # Uniform feature map -> high loss (all positions similar)
    >>> f_uniform = torch.ones(B, C, H, W) * 0.5
    >>> loss_uniform = feature_diversity_loss([f_uniform])
    >>> loss_uniform.item() > 0.5  # high similarity
    True

    >>> # Empty list
    >>> loss = feature_diversity_loss([])
    >>> loss.item()
    0.0
    """
    if len(feature_maps) == 0:
        return torch.tensor(0.0)

    total_loss = torch.tensor(0.0, device=feature_maps[0].device,
                               dtype=feature_maps[0].dtype)
    num_maps = 0

    for fmap in feature_maps:
        if fmap.numel() == 0:
            continue

        if torch.isnan(fmap).any():
            raise ValueError("feature_maps contains NaN values.")
        if torch.isinf(fmap).any():
            raise ValueError("feature_maps contains Inf values.")

        B, C, H, W = fmap.shape
        num_spatial = H * W

        # Skip if only one spatial position (no pairs to compare)
        if num_spatial <= 1:
            continue

        # ── 0. Subsample spatial positions to cap memory ───────────────
        # The pairwise similarity matrix is [B, num_spatial, num_spatial].
        # For large spatial sizes (e.g. 64x64 = 4096), this is ~128 MB
        # per batch element.  We subsample to at most MAX_SPATIAL_SAMPLES
        # to avoid OOM on memory-constrained devices (MPS / small GPUs).
        MAX_SPATIAL_SAMPLES = 256
        if num_spatial > MAX_SPATIAL_SAMPLES:
            with torch.no_grad():
                # Randomly sample spatial indices (same across batch)
                idx = torch.randperm(num_spatial, device=fmap.device)[:MAX_SPATIAL_SAMPLES]
            # Subsample: [B, C, H, W] -> flatten spatial dim -> [B, C, HW] -> index -> [B, C, K]
            f_flat = fmap.view(B, C, num_spatial)        # [B, C, HW]
            f_flat = f_flat[:, :, idx]                    # [B, C, K]
            K = MAX_SPATIAL_SAMPLES
        else:
            f_flat = fmap.view(B, C, num_spatial)        # [B, C, HW]
            K = num_spatial

        f_flat = f_flat.transpose(1, 2)                  # [B, K, C]

        # ── 2. Normalise each spatial feature vector to unit norm ──────
        f_norm = F.normalize(f_flat, p=2, dim=-1)        # [B, K, C]

        # ── 3. Compute pairwise cosine similarity matrix ───────────────
        # S[b, i, j] = f_norm[b, i] · f_norm[b, j]
        # Shape: [B, K, K]
        sim_matrix = torch.bmm(f_norm, f_norm.transpose(1, 2))

        # ── 4. Take absolute value of off-diagonal entries ─────────────
        abs_sim = sim_matrix.abs()                       # [B, K, K]
        total_sum = abs_sim.sum(dim=(1, 2))              # [B]
        diag_sum = float(K)
        off_diag_sum = total_sum - diag_sum              # [B]
        num_off_diag = K * (K - 1)
        mean_abs_sim = off_diag_sum / float(num_off_diag)  # [B]
        map_loss = mean_abs_sim.mean()

        total_loss = total_loss + map_loss
        num_maps += 1

    # Mean over all feature maps
    if num_maps > 0:
        total_loss = total_loss / float(num_maps)

    return total_loss


# ──────────────────────────────────────────────────────────────────────────────
# Latent Diversity Loss
# ──────────────────────────────────────────────────────────────────────────────


def latent_diversity_loss(z_batch: torch.Tensor) -> torch.Tensor:
    r"""Compute the latent diversity loss to encourage separation in the
    latent space.

    If all latent vectors in a batch are close together, the generator will
    produce similar outputs, leading to mode collapse.  This loss encourages
    the latents to be well-separated by maximising the pairwise distance
    between them.

    The loss is:

    .. math::

        \mathcal{L}_{\text{latent}} = -\frac{1}{B \cdot (B-1)}
            \sum_{i \neq j} \|z_i - z_j\|_2

    i.e., the **negative** mean pairwise Euclidean distance.  By minimising
    this loss (making it more negative), we maximise the pairwise distances,
    pushing latents apart.

    **Why this prevents image collapse**: if the generator is trained with
    latents that are forced to be diverse, the generator must learn to map
    different regions of the latent space to different outputs.  This directly
    counteracts mode collapse, where the generator would otherwise learn to
    map all latents to the same output.

    **Important usage note**: This loss should only be applied to the
    generator's latent sampling, not to an encoder (if one exists).  It is
    intended for use as a regulariser on the input latents during generator
    training.  Apply it with a small weight (e.g., ``0.01``) to avoid
    distorting the latent distribution too much.

    Parameters
    ----------
    z_batch : torch.Tensor
        Batch of latent vectors, shape ``[B, D]`` where ``B`` is the batch
        size and ``D`` is the latent dimensionality.

    Returns
    -------
    torch.Tensor
        Scalar latent diversity loss (0-dimensional).  Returns 0.0 if
        ``B <= 1`` (no pairs to compute).

    Raises
    ------
    ValueError
        If ``z_batch`` contains NaN or Inf values.

    Shape
    -----
    - Input: ``[B, D]``
    - Output: scalar ``[]``

    Gradient flow
    -------------
    Gradients flow through the latent vectors.  If the latents are
    ``torch.randn`` samples (not parameters), this loss will not produce
    gradients for the generator unless the latents are connected to the
    computation graph (e.g., through a reparameterisation trick or if
    they are learnable embeddings).

    Example
    -------
    >>> # Two very close latents -> high loss (more negative)
    >>> z_close = torch.tensor([[0.0, 0.0], [0.01, 0.01]])
    >>> loss_close = latent_diversity_loss(z_close)
    >>> loss_close.item()  # doctest: +ELLIPSIS
    -0.0141...

    >>> # Two far apart latents -> low loss (less negative)
    >>> z_far = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    >>> loss_far = latent_diversity_loss(z_far)
    >>> loss_far.item()  # doctest: +ELLIPSIS
    -14.1421...

    >>> # Single latent -> no pairs -> loss = 0
    >>> z_single = torch.randn(1, 64)
    >>> loss = latent_diversity_loss(z_single)
    >>> abs(loss.item()) < 1e-6
    True

    >>> # Empty tensor
    >>> z_empty = torch.empty(0, 64)
    >>> loss = latent_diversity_loss(z_empty)
    >>> abs(loss.item()) < 1e-6
    True
    """
    if z_batch.numel() == 0:
        return torch.tensor(0.0, device=z_batch.device, dtype=z_batch.dtype)

    if torch.isnan(z_batch).any():
        raise ValueError("z_batch contains NaN values.")
    if torch.isinf(z_batch).any():
        raise ValueError("z_batch contains Inf values.")

    B, D = z_batch.shape

    # No pairs to compute if B <= 1
    if B <= 1:
        return torch.tensor(0.0, device=z_batch.device, dtype=z_batch.dtype)

    # ── 1. Compute pairwise Euclidean distance matrix ──────────────────
    # z_i - z_j for all i, j: [B, 1, D] - [1, B, D] -> [B, B, D]
    diff = z_batch[:, None, :] - z_batch[None, :, :]  # [B, B, D]

    # Euclidean distance: ||z_i - z_j||_2
    # Shape: [B, B]
    dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + _EPS)

    # ── 2. Extract upper-triangular entries (i < j) ────────────────────
    triu_idx = torch.triu_indices(B, B, offset=1, device=z_batch.device)
    pair_dists = dist_matrix[triu_idx[0], triu_idx[1]]  # [B*(B-1)/2]

    # ── 3. Mean pairwise distance ──────────────────────────────────────
    mean_pair_dist = pair_dists.mean()

    # ── 4. Negative: we want to maximise distance, so minimise -distance
    loss = -mean_pair_dist

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Composite Collapse Losses
# ──────────────────────────────────────────────────────────────────────────────


def compute_collapse_losses(
    attention_maps: torch.Tensor,
    skip_features: List[torch.Tensor],
    z_batch: torch.Tensor,
    diversity_weight: float = 0.1,
    entropy_weight: float = 0.1,
    repulsion_weight: float = 1.0,
    latent_diversity_weight: float = 0.01,
    repulsion_threshold: float = 0.2,
) -> Dict[str, torch.Tensor]:
    r"""Compute all collapse-prevention losses in a single call.

    This function composes :func:`attention_entropy_loss`,
    :func:`repulsion_loss`, :func:`feature_diversity_loss`, and
    :func:`latent_diversity_loss` into a single call, returning a dictionary
    with all loss components and their weighted sum.

    The total collapse loss is:

    .. math::

        \mathcal{L}_{\text{collapse}} =
            w_{\text{entropy}} \cdot \mathcal{L}_{\text{entropy}}
            + w_{\text{repulsion}} \cdot \mathcal{L}_{\text{repulsion}}
            + w_{\text{diversity}} \cdot \mathcal{L}_{\text{diversity}}
            + w_{\text{latent}} \cdot \mathcal{L}_{\text{latent}}

    where the weights control the contribution of each mechanism.

    Parameters
    ----------
    attention_maps : torch.Tensor
        Attention weight maps from SCCA modules, shape ``[B, N, H, W]``.
        Each ``[b, n]`` slice must be a valid probability distribution over
        ``H × W``.
    skip_features : list of torch.Tensor
        Multi-resolution skip feature maps from ``ContentDecoder``.  Each
        tensor has shape ``[B, C_i, H_i, W_i]``.
    z_batch : torch.Tensor
        Batch of latent vectors, shape ``[B, D]``.
    diversity_weight : float, optional
        Weight for the feature diversity loss.  Must be non-negative.
        (default: ``0.1``)
    entropy_weight : float, optional
        Weight for the attention entropy loss.  Must be non-negative.
        (default: ``0.1``)
    repulsion_weight : float, optional
        Weight for the slot repulsion loss.  Must be non-negative.
        (default: ``1.0``)
    latent_diversity_weight : float, optional
        Weight for the latent diversity loss.  Must be non-negative.
        Should typically be small (e.g., ``0.01``) to avoid distorting the
        latent distribution.  (default: ``0.01``)
    repulsion_threshold : float, optional
        Minimum normalised distance between slot centres of mass before
        repulsion is incurred.  Passed to :func:`repulsion_loss`.
        (default: ``0.2``)

    Returns
    -------
    dict of str -> torch.Tensor
        A dictionary with the following keys:

        - ``"entropy"``: the attention entropy loss (scalar).
        - ``"repulsion"``: the slot repulsion loss (scalar).
        - ``"feature_diversity"``: the feature diversity loss (scalar).
        - ``"latent_diversity"``: the latent diversity loss (scalar).
        - ``"collapse_loss"``: the weighted sum of all four losses (scalar).

    Raises
    ------
    ValueError
        If any weight is negative.

    Example
    -------
    >>> B, N, H, W = 2, 4, 8, 8
    >>> D = 128
    >>> attn = torch.rand(B, N, H, W)
    >>> attn = attn / attn.view(B, N, -1).sum(dim=-1, keepdim=True).view(B, N, 1, 1)
    >>> skip = [torch.randn(B, 16, 8, 8), torch.randn(B, 8, 16, 16)]
    >>> z = torch.randn(B, D)
    >>> losses = compute_collapse_losses(attn, skip, z)
    >>> list(losses.keys())
    ['entropy', 'repulsion', 'feature_diversity', 'latent_diversity', 'collapse_loss']
    >>> all(loss.ndim == 0 for loss in losses.values())
    True
    """
    # ── Validate weights ──────────────────────────────────────────────────
    if diversity_weight < 0.0:
        raise ValueError(
            f"diversity_weight must be non-negative, got {diversity_weight}."
        )
    if entropy_weight < 0.0:
        raise ValueError(
            f"entropy_weight must be non-negative, got {entropy_weight}."
        )
    if repulsion_weight < 0.0:
        raise ValueError(
            f"repulsion_weight must be non-negative, got {repulsion_weight}."
        )
    if latent_diversity_weight < 0.0:
        raise ValueError(
            f"latent_diversity_weight must be non-negative, "
            f"got {latent_diversity_weight}."
        )

    # ── 1. Attention entropy loss ─────────────────────────────────────────
    entropy = attention_entropy_loss(attention_maps)

    # ── 2. Slot repulsion loss ────────────────────────────────────────────
    repulsion = repulsion_loss(attention_maps, repulsion_threshold)

    # ── 3. Feature diversity loss ────────────────────────────────────────
    feature_div = feature_diversity_loss(skip_features)

    # ── 4. Latent diversity loss ─────────────────────────────────────────
    latent_div = latent_diversity_loss(z_batch)

    # ── 5. Weighted sum ─────────────────────────────────────────────────
    collapse_loss = (
        entropy_weight * entropy
        + repulsion_weight * repulsion
        + diversity_weight * feature_div
        + latent_diversity_weight * latent_div
    )

    return {
        "entropy": entropy,
        "repulsion": repulsion,
        "feature_diversity": feature_div,
        "latent_diversity": latent_div,
        "collapse_loss": collapse_loss,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "attention_entropy_loss",
    "repulsion_loss",
    "feature_diversity_loss",
    "latent_diversity_loss",
    "compute_collapse_losses",
]
