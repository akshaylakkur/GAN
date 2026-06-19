"""
Spatial-Content Cross-Attention (SCCA) module for ILGAN.

The SCCA module is the architectural centerpiece that binds image generation
to bounding box prediction. It enables the generator to jointly represent
visual features and their spatial significance through a cross-attention
mechanism between learned spatial slots and content feature maps.

The module produces:
- ``Z``: Content-aware slot features for bounding box prediction.
- ``A``: Attention weight maps for interpretability.
- Auxiliary losses: entropy regularisation (encourages spatial concentration)
  and repulsion loss (the first mathematical mechanism against bounding box
  collapse by preventing multiple slots from attending to the same region).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS = 1e-8


# ──────────────────────────────────────────────────────────────────────────────
# SCCA Module
# ──────────────────────────────────────────────────────────────────────────────


class SpatialContentCrossAttention(nn.Module):
    """Spatial-Content Cross-Attention (SCCA) module.

    This module takes two inputs:

    1. A **content** feature map ``X`` of shape ``[B, C, H, W]`` representing
       visual features (e.g., from a generator's intermediate layer).
    2. A **spatial query** tensor ``Q`` of shape ``[B, N, D]`` representing
       *N* learnable spatial slots that will bind to bounding box locations.

    For each spatial slot, the module computes a soft attention map over all
    spatial positions in the content feature map, then aggregates visual
    features from the attended region. The result is a set of *N* visual
    feature summaries that encode *what* is at *where* each slot is looking.

    The module uses a **query projection** (``slot_dim -> proj_channels``) so
    that the spatial slot dimension can differ from the projected key/value
    channel dimension, following the standard cross-attention formulation.

    Multi-head attention splits the projected channels across heads: each of
    ``num_heads`` heads attends to ``head_dim = proj_channels // num_heads``
    channels independently, then the outputs are concatenated.

    The module incorporates two built-in regularisation mechanisms:

    - **Entropy minimisation** on each slot's attention distribution,
      encouraging each slot to focus on a compact spatial region.
    - **Repulsion** between the spatial centres of mass of different slots,
      preventing multiple slots from attending to the same region (the first
      mathematical mechanism against bounding box collapse).

    Parameters
    ----------
    content_channels : int
        Number of input channels ``C`` of the content feature map.
    slot_dim : int
        Dimensionality ``D`` of each spatial slot (last dimension of ``Q``).
    proj_channels : int
        Number of channels ``C'`` to project both the content feature map
        **and** the spatial queries to.  This is the inner dimension of the
        attention computation.  Must be divisible by ``num_heads``.
    num_heads : int, optional
        Number of attention heads.  The projected channels are split across
        heads for parallel attention computation.  (default: ``1``)
    repulsion_threshold : float, optional
        Minimum allowed normalised distance between slot centres of mass.
        Slots whose centres are closer than this threshold incur a repulsion
        penalty.  (default: ``0.2``)
    repulsion_weight : float, optional
        Coefficient scaling the repulsion loss term.  (default: ``1.0``)
    entropy_weight : float, optional
        Coefficient scaling the entropy regularisation term.
        (default: ``0.1``)
    """

    def __init__(
        self,
        content_channels: int,
        slot_dim: int,
        proj_channels: int,
        num_heads: int = 1,
        repulsion_threshold: float = 0.2,
        repulsion_weight: float = 1.0,
        entropy_weight: float = 0.1,
    ) -> None:
        super().__init__()

        if proj_channels % num_heads != 0:
            raise ValueError(
                f"proj_channels ({proj_channels}) must be divisible by "
                f"num_heads ({num_heads}) for multi-head attention."
            )

        self.content_channels = content_channels
        self.slot_dim = slot_dim
        self.proj_channels = proj_channels
        self.num_heads = num_heads
        self.head_dim = proj_channels // num_heads
        self.repulsion_threshold = repulsion_threshold
        self.repulsion_weight = repulsion_weight
        self.entropy_weight = entropy_weight

        # ── feature projections (content -> key/value) ───────────────────
        # Both are 1x1 convolutions: C -> C'
        self.key_proj = nn.Conv2d(
            content_channels, proj_channels, kernel_size=1, bias=False,
        )
        self.value_proj = nn.Conv2d(
            content_channels, proj_channels, kernel_size=1, bias=False,
        )

        # ── query projection (slot -> projected space) ───────────────────
        # Maps each spatial slot from its intrinsic D-dim space to C',
        # allowing the slot dimension to differ from the key/value dimension.
        self.query_proj = nn.Linear(slot_dim, proj_channels, bias=False)

        # ── output projection (back to slot dimension) ───────────────────
        # After attending, project C'-dim features back to slot dimension D.
        self.output_proj = nn.Linear(proj_channels, slot_dim, bias=False)

        # ── parameter initialisation ─────────────────────────────────────
        self.reset_parameters()

    # ── parameter initialisation ─────────────────────────────────────────

    def reset_parameters(self) -> None:
        """Initialise weights for stable training.

        Uses Kaiming uniform initialisation for the convolutional projections
        and the query projection.  The output projection uses a small normal
        initialisation to start training with small feature perturbations.
        """
        nn.init.kaiming_uniform_(self.key_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.query_proj.weight, a=math.sqrt(5))
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)

    # ── forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,
        Q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the SCCA module.

        Parameters
        ----------
        X : torch.Tensor
            Content feature map of shape ``[B, C, H, W]``.
        Q : torch.Tensor
            Spatial query tensor of shape ``[B, N, D]``, where ``N`` is the
            number of spatial slots and ``D`` is the slot dimension.

        Returns
        -------
        Z : torch.Tensor
            Attended feature summaries per slot, shape ``[B, N, D]`` (same
            dimension as the input spatial queries).
        A : torch.Tensor
            Attention weight maps, shape ``[B, N, H, W]``.  Each ``[b, n]``
            slice is a soft spatial distribution over the ``H x W`` grid.
        aux_losses : dict of str -> torch.Tensor
            Dictionary of auxiliary loss terms:
            - ``"entropy"``: mean entropy of attention distributions (scalar).
            - ``"repulsion"``: repulsion loss between slot centres (scalar).
            - ``"entropy_weighted"``: ``entropy_weight * entropy``.
            - ``"repulsion_weighted"``: ``repulsion_weight * repulsion``.
        """
        B, C, H, W = X.shape
        B_q, N, D = Q.shape

        if B != B_q:
            raise ValueError(
                f"Batch sizes of X ({B}) and Q ({B_q}) must match."
            )

        _validate_tensor(X, "X")
        _validate_tensor(Q, "Q")

        device = X.device
        C_proj = self.proj_channels

        # ── 1. Project content feature map ───────────────────────────────
        V: torch.Tensor = self.value_proj(X)  # [B, C', H, W]
        K: torch.Tensor = self.key_proj(X)    # [B, C', H, W]

        # ── 2. Project spatial queries ───────────────────────────────────
        Q_proj: torch.Tensor = self.query_proj(Q)  # [B, N, C']

        # ── 3. Flatten spatial dimensions ────────────────────────────────
        V_flat = V.view(B, C_proj, H * W)  # [B, C', HW]
        K_flat = K.view(B, C_proj, H * W)  # [B, C', HW]

        # ── 4. Compute attention scores ──────────────────────────────────
        if self.num_heads > 1:
            A = self._multi_head_attention(Q_proj, K_flat, N, H, W, device)
        else:
            # Single head: standard dot-product attention
            # A = softmax(Q_proj @ K^T / sqrt(C'))
            scale = 1.0 / math.sqrt(C_proj)
            attn_logits = torch.bmm(Q_proj, K_flat) * scale  # [B, N, HW]
            A = F.softmax(attn_logits, dim=-1)               # [B, N, HW]

        # ── 5. Compute attended features ─────────────────────────────────
        # Z[b, n, :] = sum_i A[b, n, i] * V[b, :, i]
        Z_flat = torch.bmm(A, V_flat.transpose(1, 2))  # [B, N, C']

        # ── 6. Output projection back to slot dimension ──────────────────
        Z = self.output_proj(Z_flat)  # [B, N, D]

        # ── 7. Reshape attention maps for interpretability ────────────────
        A_maps = A.view(B, N, H, W)

        # ── 8. Compute auxiliary losses ──────────────────────────────────
        aux_losses = self._compute_auxiliary_losses(A, B, N, H, W, device)

        return Z, A_maps, aux_losses

    # ── multi-head attention ─────────────────────────────────────────────

    def _multi_head_attention(
        self,
        Q_proj: torch.Tensor,
        K_flat: torch.Tensor,
        N: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute multi-head dot-product attention.

        Splits the projected channels ``C'`` across ``num_heads`` heads.
        Each head processes ``head_dim`` channels independently.

        Parameters
        ----------
        Q_proj : torch.Tensor
            Projected queries ``[B, N, C']``.
        K_flat : torch.Tensor
            Flattened keys ``[B, C', HW]``.
        N, H, W : int
            Number of slots, grid height, grid width.
        device : torch.device
            Current device.

        Returns
        -------
        A : torch.Tensor
            Attention weights ``[B, N, HW]``.
        """
        num_heads = self.num_heads
        head_dim = self.head_dim
        C_proj = self.proj_channels

        # Reshape Q: [B, N, C'] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        Q_mh = Q_proj.view(-1, N, num_heads, head_dim).transpose(1, 2)

        # Reshape K: [B, C', HW] -> [B, num_heads, head_dim, HW]
        K_mh = K_flat.view(-1, num_heads, head_dim, H * W)

        # Scaled dot-product attention per head
        scale = 1.0 / math.sqrt(head_dim)
        attn_logits = torch.matmul(Q_mh, K_mh) * scale  # [B, num_heads, N, HW]
        A_head = F.softmax(attn_logits, dim=-1)         # [B, num_heads, N, HW]

        # Merge heads: [B, num_heads, N, HW] -> [B, N, HW]
        A = A_head.mean(dim=1)  # [B, num_heads, N, HW] -> [B, N, HW]

        return A

    # ── auxiliary losses ─────────────────────────────────────────────────

    def _compute_auxiliary_losses(
        self,
        A: torch.Tensor,
        B: int,
        N: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Compute entropy regularisation and repulsion losses.

        Parameters
        ----------
        A : torch.Tensor
            Attention weights ``[B, N, HW]`` (softmax-normalised over HW).
        B, N, H, W : int
            Batch size, number of slots, grid height, grid width.
        device : torch.device
            Current device.

        Returns
        -------
        dict
            Dictionary with keys ``"entropy"``, ``"repulsion"``,
            ``"entropy_weighted"``, ``"repulsion_weighted"``.
        """
        losses: Dict[str, torch.Tensor] = {}

        # ── Entropy regularisation ──────────────────────────────────────
        # H(p) = -sum_i p_i * log(p_i + eps)
        # Minimising entropy encourages each slot to focus on a compact region.
        entropy = -(A * torch.log(A + _EPS)).sum(dim=-1)  # [B, N]
        mean_entropy = entropy.mean()

        losses["entropy"] = mean_entropy
        losses["entropy_weighted"] = self.entropy_weight * mean_entropy

        # ── Repulsion loss ──────────────────────────────────────────────
        # Compute the spatial centre of mass for each slot's attention
        # distribution, then penalise pairs that are too close.
        repulsion = self._compute_repulsion_loss(A, B, N, H, W, device)
        losses["repulsion"] = repulsion
        losses["repulsion_weighted"] = self.repulsion_weight * repulsion

        return losses

    def compute_repulsion_loss(
        self,
        A: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Public helper to compute repulsion loss from raw attention weights.

        Useful for testing and for external use when attention weights have
        been computed separately.

        Parameters
        ----------
        A : torch.Tensor
            Attention weights ``[B, N, HW]`` (softmax-normalised over HW).
        H : int
            Height of the spatial grid.
        W : int
            Width of the spatial grid.

        Returns
        -------
        torch.Tensor
            Scalar repulsion loss.
        """
        B, N, _ = A.shape
        device = A.device
        return self._compute_repulsion_loss(A, B, N, H, W, device)

    def _compute_repulsion_loss(
        self,
        A: torch.Tensor,
        B: int,
        N: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the repulsion loss between slot centres of mass.

        For each slot ``n`` in each batch element ``b``, compute the spatial
        centre of mass ``(cx_n, cy_n)`` in normalised ``[0, 1]`` coordinates
        using the attention distribution ``A[b, n, :]`` as weights.

        Then compute a pairwise repulsion penalty:

            L_rep = (1 / (B * N_pairs)) * sum_b sum_{i < j} max(0, tau - d_ij)^2

        where ``d_{ij}`` is the Euclidean distance between the centres of
        slots ``i`` and ``j``, and ``tau`` is ``repulsion_threshold``.

        Parameters
        ----------
        A : torch.Tensor
            Attention weights ``[B, N, HW]``.
        B, N, H, W : int
            Dimensions.
        device : torch.device
            Current device.

        Returns
        -------
        torch.Tensor
            Scalar repulsion loss (``0`` if no pairs are too close, or if
            ``N <= 1``).
        """
        if N <= 1:
            return torch.tensor(0.0, device=device, dtype=A.dtype)

        # ── Build normalised coordinate grids ───────────────────────────
        # x in [0, 1], y in [0, 1] with (W) and (H) points respectively.
        x_grid = torch.linspace(0.0, 1.0, W, device=device)
        y_grid = torch.linspace(0.0, 1.0, H, device=device)

        # Create a [HW] flat grid with x and y coordinates
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        x_coords = xx.reshape(-1)  # [HW]
        y_coords = yy.reshape(-1)  # [HW]

        # ── Compute centre of mass for each slot ────────────────────────
        # cx[b, n] = sum_i A[b, n, i] * x_coords[i]
        cx = (A * x_coords[None, None, :]).sum(dim=-1)  # [B, N]
        cy = (A * y_coords[None, None, :]).sum(dim=-1)  # [B, N]

        # ── Compute pairwise distances ──────────────────────────────────
        # dx[b, i, j] = cx[b, i] - cx[b, j], shape [B, N, N]
        dx = cx[:, :, None] - cx[:, None, :]   # [B, N, N]
        dy = cy[:, :, None] - cy[:, None, :]   # [B, N, N]
        dists = torch.sqrt(dx.pow(2) + dy.pow(2) + _EPS)  # [B, N, N]

        # ── Compute repulsion penalty ───────────────────────────────────
        tau = self.repulsion_threshold
        # Upper-triangular indices (i < j)
        triu_idx = torch.triu_indices(N, N, offset=1, device=device)

        # Extract pairwise distances for i < j: [B, N_pairs]
        pair_dists = dists[:, triu_idx[0], triu_idx[1]]
        penalties = torch.clamp(tau - pair_dists, min=0.0).pow(2)

        # Mean over all pairs and batch
        repulsion = penalties.mean()

        return repulsion

    # ── representation ──────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"content_channels={self.content_channels}, "
            f"slot_dim={self.slot_dim}, "
            f"proj_channels={self.proj_channels}, "
            f"num_heads={self.num_heads}, "
            f"repulsion_threshold={self.repulsion_threshold}, "
            f"repulsion_weight={self.repulsion_weight}, "
            f"entropy_weight={self.entropy_weight}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _validate_tensor(t: torch.Tensor, name: str) -> None:
    """Raise ``ValueError`` if *t* contains any NaN or Inf values."""
    if torch.isnan(t).any():
        raise ValueError(f"{name} contains NaN values.")
    if torch.isinf(t).any():
        raise ValueError(f"{name} contains Inf values.")


__all__ = [
    "SpatialContentCrossAttention",
]