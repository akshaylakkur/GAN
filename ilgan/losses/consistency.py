"""
Cross-modal consistency loss for the ILGAN dual-output GAN.

This module implements a **novel cross-modal consistency mechanism** that
aligns the image and bounding box representations in a shared feature space.
The core idea is that a generated image and its corresponding bounding boxes
should encode the same semantic content — they are two views of the same
underlying scene.  By enforcing consistency between their feature embeddings,
we encourage the generator to produce images and boxes that are semantically
coherent with each other.

Mathematical foundation
-----------------------
Let :math:`I = G_{img}(z)` be the generated image and
:math:`B = G_{box}(z)` be the predicted bounding boxes from the same latent
:math:`z`.  We define two encoders:

- :math:`E_{img}`: ``ImageFeatureEncoder`` — a small CNN that maps
  :math:`I \in \mathbb{R}^{3 \times H \times W}` to a feature vector
  :math:`f_{img} \in \mathbb{R}^{D}` (where :math:`D = 128`).
- :math:`E_{box}`: ``BoxFeatureEncoder`` — a per-box MLP with
  confidence-weighted attention pooling that maps the set of boxes
  :math:`B \in \mathbb{R}^{N \times 4}` and confidences
  :math:`c \in \mathbb{R}^{N \times 1}` to a feature vector
  :math:`f_{box} \in \mathbb{R}^{D}`.

The consistency loss is the **cosine embedding loss**:

.. math::

    \mathcal{L}_{cons} = 1 - \cos(f_{img}, f_{box}) = 1 -
    \frac{f_{img} \cdot f_{box}}{\|f_{img}\|_2 \|f_{box}\|_2}

This pulls the image and box representations together in the shared feature
space.  By minimising this loss, the generator learns to produce images and
bounding boxes that are semantically aligned — the boxes highlight the
regions of interest that correspond to the content of the generated image.

Why this prevents representation collapse
------------------------------------------
If the generator starts to collapse (producing the same image or the same
boxes for all latents), the consistency loss will be low trivially (both
representations collapse to the same point).  However, the **adversarial
loss** and **collapse prevention losses** (from ``collapse_prevention.py``)
prevent this.  The consistency loss acts as a **regulariser** that ensures
the two output modalities stay aligned, rather than diverging into
incoherent representations.

The encoders are intentionally small (under 1M parameters total) to avoid
excessive memory usage.  They are trained jointly with the generator.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon for numerical stability in normalisation."""

_PROJ_DIM: int = 128
"""Dimensionality of the shared feature space for cross-modal consistency."""


# ──────────────────────────────────────────────────────────────────────────────
# ImageFeatureEncoder — maps images to the shared feature space
# ──────────────────────────────────────────────────────────────────────────────


class ImageFeatureEncoder(nn.Module):
    r"""Small CNN that maps a generated image to a fixed-size feature vector.

    This encoder projects an RGB image :math:`I \in \mathbb{R}^{3 \times H \times W}`
    into a :math:`D`-dimensional feature vector :math:`f_{img} \in \mathbb{R}^{D}`
    (where :math:`D = 128`).  The architecture is a strided convolutional
    network with spectral normalisation on all layers, ending with adaptive
    average pooling and a linear projection.

    Architecture
    ------------
    ::

        Input: [B, 3, H, W]
        ┌──────────────────────────────────────────────┐
        │ Conv2d(3 → 64, 4×4, stride=2, padding=1)     │  ← spectral norm
        │ LeakyReLU(0.2)                                │
        ├──────────────────────────────────────────────┤
        │ Conv2d(64 → 128, 4×4, stride=2, padding=1)   │  ← spectral norm
        │ LeakyReLU(0.2)                                │
        ├──────────────────────────────────────────────┤
        │ Conv2d(128 → 256, 4×4, stride=2, padding=1)  │  ← spectral norm
        │ LeakyReLU(0.2)                                │
        ├──────────────────────────────────────────────┤
        │ AdaptiveAvgPool2d(1)                          │
        │ Flatten                                       │
        │ Linear(256 → proj_dim)                        │
        └──────────────────────────────────────────────┘
        Output: [B, proj_dim]

    The three stride-2 convolutions reduce the spatial resolution by a factor
    of 8 overall (2 × 2 × 2).  For a 256×256 input image, the spatial size
    after the three conv layers is 32×32, which is then pooled to 1×1.

    Spectral normalisation (``nn.utils.spectral_norm``) is applied to all
    convolutional layers to stabilise training and constrain the Lipschitz
    constant of the encoder, which is important when training jointly with
    the GAN generator.

    Parameters
    ----------
    proj_dim : int, optional
        Dimensionality of the output feature vector.  (default: ``128``)

    Shape
    -----
    - Input: ``[B, 3, H, W]`` — batch of RGB images, any spatial size
      (at least 8×8 due to three stride-2 convolutions).
    - Output: ``[B, proj_dim]`` — feature vectors in the shared space.

    Example
    -------
    >>> encoder = ImageFeatureEncoder(proj_dim=128)
    >>> images = torch.randn(4, 3, 256, 256)
    >>> features = encoder(images)
    >>> features.shape
    torch.Size([4, 128])
    """

    def __init__(self, proj_dim: int = _PROJ_DIM) -> None:
        super().__init__()

        self.proj_dim = proj_dim

        # ── 1. Strided convolutional layers with spectral norm ──────────
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True)
        )
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)
        )
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)
        )

        # ── 2. Activation ──────────────────────────────────────────────
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # ── 3. Pooling + projection head ────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, proj_dim, bias=True)

        # ── 4. Parameter initialisation ────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights for stable training dynamics.

        - Convolutional weights: Kaiming uniform (He et al., 2015) with
          ``a = √5``, which accounts for the LeakyReLU non-linearity.
        - Convolutional biases: zero-initialised.
        - Linear projection: small normal initialisation to keep initial
          feature vectors near zero.
        - Linear bias: zero-initialised.
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            # Access the underlying conv weight (spectral_norm wraps it)
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into the shared feature space.

        Parameters
        ----------
        images : torch.Tensor
            Batch of RGB images, shape ``[B, 3, H, W]``.  Pixel values
            should be in ``[-1, 1]`` (as produced by the generator's Tanh
            output).  The spatial size ``H, W`` must be at least 8 due to
            the three stride-2 convolutions.

        Returns
        -------
        torch.Tensor
            Feature vectors of shape ``[B, proj_dim]``.
        """
        # ── 1. Strided convolutions ────────────────────────────────────
        h = self.activation(self.conv1(images))   # [B, 64, H/2, W/2]
        h = self.activation(self.conv2(h))         # [B, 128, H/4, W/4]
        h = self.activation(self.conv3(h))         # [B, 256, H/8, W/8]

        # ── 2. Global pooling + flatten ────────────────────────────────
        h = self.pool(h)                           # [B, 256, 1, 1]
        h = h.view(h.shape[0], -1)                 # [B, 256]

        # ── 3. Linear projection ─────────────────────────────────────
        features = self.fc(h)                      # [B, proj_dim]

        return features

    def extra_repr(self) -> str:
        return f"proj_dim={self.proj_dim}"


# ──────────────────────────────────────────────────────────────────────────────
# BoxFeatureEncoder — maps bounding boxes to the shared feature space
# ──────────────────────────────────────────────────────────────────────────────


class BoxFeatureEncoder(nn.Module):
    r"""Encodes a set of bounding boxes into the shared feature space using
    a per-box MLP with confidence-weighted attention pooling.

    This encoder takes a set of :math:`N` bounding boxes (with associated
    confidence scores) and produces a single :math:`D`-dimensional feature
    vector that represents the "box content" of the image.  The architecture
    is:

    1. **Per-box MLP**: each box :math:`(cx, cy, w, h) \in \mathbb{R}^4` is
       independently processed through a 3-layer MLP:

       .. math::

           h_1 &= \text{ReLU}(W_1 \cdot (cx, cy, w, h) + b_1) \quad (4 \to 64) \\
           h_2 &= \text{ReLU}(W_2 \cdot h_1 + b_2) \quad (64 \to 128) \\
           f_i &= W_3 \cdot h_2 + b_3 \quad (128 \to D)

       where :math:`f_i \in \mathbb{R}^D` is the feature vector for box ``i``.

    2. **Confidence-weighted attention pooling**: the box features are
       aggregated into a single vector using attention, where the attention
       weights are computed from the box features themselves and modulated
       by the confidence scores.  Invalid boxes (where ``valid_mask`` is
       ``False``) are masked out.

       Specifically, for each box ``i``:

       .. math::

           a_i &= \frac{\exp(\text{score}(f_i) \cdot c_i)}{\sum_{j \in valid}
                  \exp(\text{score}(f_j) \cdot c_j)} \\
           f_{box} &= \sum_{i \in valid} a_i \cdot f_i

       where :math:`c_i` is the confidence score for box ``i`` and
       :math:`\text{score}(\cdot)` is a learned linear projection from
       :math:`\mathbb{R}^D \to \mathbb{R}^1`.

    The output is a single feature vector :math:`f_{box} \in \mathbb{R}^D`
    that represents the entire set of bounding boxes.

    Parameters
    ----------
    proj_dim : int, optional
        Dimensionality of the output feature vector.  (default: ``128``)

    Shape
    -----
    - Input boxes: ``[B, N, 4]`` — bounding box coordinates in
      ``(cx, cy, w, h)`` format, normalised to ``[0, 1]``.
    - Input confidences: ``[B, N, 1]`` — objectness scores in ``[0, 1]``.
    - Input valid_mask: ``[B, N]`` — boolean mask, ``True`` for valid boxes.
    - Output: ``[B, proj_dim]`` — feature vectors in the shared space.

    Example
    -------
    >>> encoder = BoxFeatureEncoder(proj_dim=128)
    >>> boxes = torch.rand(4, 10, 4)  # [B, N, 4]
    >>> confidences = torch.rand(4, 10, 1)  # [B, N, 1]
    >>> valid_mask = torch.rand(4, 10) > 0.5  # [B, N]
    >>> features = encoder(boxes, confidences, valid_mask)
    >>> features.shape
    torch.Size([4, 128])
    """

    def __init__(self, proj_dim: int = _PROJ_DIM) -> None:
        super().__init__()

        self.proj_dim = proj_dim

        # ── 1. Per-box MLP ──────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(4, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim, bias=True),
        )

        # ── 2. Attention scoring function ──────────────────────────────
        # Projects each box feature to a scalar attention logit.
        self.attention_score = nn.Linear(proj_dim, 1, bias=True)

        # ── 3. Parameter initialisation ────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights for stable training.

        - MLP layers: Kaiming uniform for linear layers with ReLU.
        - Attention scoring: small normal to keep initial attention
          weights near uniform.
        - All biases: zero-initialised.
        """
        for i in range(0, 5, 2):  # layers 0, 2, 4 of the MLP
            nn.init.kaiming_uniform_(self.mlp[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.mlp[i].bias)

        nn.init.normal_(self.attention_score.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.attention_score.bias)

    def forward(
        self,
        boxes: torch.Tensor,
        confidences: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a set of bounding boxes into the shared feature space.

        Parameters
        ----------
        boxes : torch.Tensor
            Bounding box coordinates, shape ``[B, N, 4]`` in
            ``(cx, cy, w, h)`` format, normalised to ``[0, 1]``.
        confidences : torch.Tensor
            Objectness confidence scores, shape ``[B, N, 1]`` with values
            in ``[0, 1]``.
        valid_mask : torch.Tensor
            Boolean mask of shape ``[B, N]``.  ``True`` entries indicate
            valid (non-padded) boxes that contribute to the pooled feature.
            Invalid boxes are masked out during attention pooling.

        Returns
        -------
        torch.Tensor
            Feature vectors of shape ``[B, proj_dim]``.

        Raises
        ------
        ValueError
            If ``boxes``, ``confidences``, or ``valid_mask`` have
            incompatible shapes.
        """
        # ── Input validation ──────────────────────────────────────────────
        if boxes.shape[:-1] != confidences.shape[:-1]:
            raise ValueError(
                f"boxes shape {boxes.shape} and confidences shape "
                f"{confidences.shape} are incompatible (expected same "
                f"batch and box dimensions)."
            )
        if boxes.shape[:-1] != valid_mask.shape:
            raise ValueError(
                f"boxes shape {boxes.shape} and valid_mask shape "
                f"{valid_mask.shape} are incompatible."
            )
        if valid_mask.dtype != torch.bool:
            raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")

        B, N, _ = boxes.shape

        # ── 1. Per-box MLP ──────────────────────────────────────────────
        # Flatten boxes to [B*N, 4] for the MLP, then reshape back.
        boxes_flat = boxes.view(B * N, 4)          # [B*N, 4]
        box_features = self.mlp(boxes_flat)        # [B*N, proj_dim]
        box_features = box_features.view(B, N, self.proj_dim)  # [B, N, proj_dim]

        # ── 2. Compute attention logits ─────────────────────────────────
        # Score each box feature: [B, N, 1]
        attn_logits = self.attention_score(box_features)  # [B, N, 1]

        # Modulate by confidence scores: multiply logits by confidence
        # so that high-confidence boxes get higher attention weight.
        # confidences: [B, N, 1]
        attn_logits = attn_logits * confidences  # [B, N, 1]

        # ── 3. Mask out invalid boxes ───────────────────────────────────
        # Set logits of invalid boxes to a very negative number so they
        # get zero attention weight after softmax.
        # valid_mask: [B, N] -> [B, N, 1]
        valid_mask_expanded = valid_mask.unsqueeze(-1).to(attn_logits.dtype)  # [B, N, 1]
        attn_logits = attn_logits * valid_mask_expanded + (
            -1e9 * (1.0 - valid_mask_expanded)
        )

        # ── 4. Softmax over boxes (N dimension) ────────────────────────
        # Shape: [B, N, 1]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, N, 1]

        # ── 5. Weighted sum of box features ────────────────────────────
        # box_features: [B, N, proj_dim]
        # attn_weights: [B, N, 1]
        # pooled: [B, proj_dim]
        pooled_features = (box_features * attn_weights).sum(dim=1)  # [B, proj_dim]

        return pooled_features

    def extra_repr(self) -> str:
        return f"proj_dim={self.proj_dim}"


# ──────────────────────────────────────────────────────────────────────────────
# Consistency Loss
# ──────────────────────────────────────────────────────────────────────────────


def consistency_loss(
    image_features: torch.Tensor,
    box_features: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the cross-modal consistency loss as a cosine embedding loss.

    The loss pulls the image and box feature representations together in the
    shared feature space by minimising their cosine distance:

    .. math::

        \mathcal{L}_{cons} = 1 - \cos(f_{img}, f_{box}) = 1 -
        \frac{f_{img} \cdot f_{box}}{\|f_{img}\|_2 \|f_{box}\|_2}

    The loss is bounded in :math:`[0, 2]`:

    - :math:`\mathcal{L}_{cons} = 0` when the two vectors are perfectly
      aligned (cosine similarity = 1).
    - :math:`\mathcal{L}_{cons} = 1` when they are orthogonal (cosine
      similarity = 0).
    - :math:`\mathcal{L}_{cons} = 2` when they are opposite (cosine
      similarity = -1).

    Parameters
    ----------
    image_features : torch.Tensor
        Image feature vectors from ``ImageFeatureEncoder``, shape
        ``[B, proj_dim]``.
    box_features : torch.Tensor
        Box feature vectors from ``BoxFeatureEncoder``, shape
        ``[B, proj_dim]``.

    Returns
    -------
    torch.Tensor
        Scalar consistency loss (0-dimensional), averaged over the batch.

    Raises
    ------
    ValueError
        If ``image_features`` and ``box_features`` have different shapes.
    ValueError
        If either tensor contains NaN or Inf values.

    Shape
    -----
    - Inputs: ``[B, proj_dim]`` for both tensors.
    - Output: scalar ``[]``.

    Example
    -------
    >>> img_feat = torch.randn(4, 128)
    >>> box_feat = torch.randn(4, 128)
    >>> loss = consistency_loss(img_feat, box_feat)
    >>> loss.shape
    torch.Size([])
    >>> 0.0 <= loss.item() <= 2.0
    True

    >>> # Perfectly aligned features -> loss = 0
    >>> aligned = torch.randn(4, 128)
    >>> loss_aligned = consistency_loss(aligned, aligned)
    >>> abs(loss_aligned.item()) < 1e-6
    True
    """
    # ── Input validation ──────────────────────────────────────────────────
    if image_features.shape != box_features.shape:
        raise ValueError(
            f"image_features shape {image_features.shape} must match "
            f"box_features shape {box_features.shape}."
        )

    if torch.isnan(image_features).any():
        raise ValueError("image_features contains NaN values.")
    if torch.isinf(image_features).any():
        raise ValueError("image_features contains Inf values.")
    if torch.isnan(box_features).any():
        raise ValueError("box_features contains NaN values.")
    if torch.isinf(box_features).any():
        raise ValueError("box_features contains Inf values.")

    # ── 1. Normalise both feature vectors to unit norm ──────────────────
    # Use F.normalize with eps for numerical stability.
    img_norm = F.normalize(image_features, p=2, dim=-1, eps=_EPS)  # [B, D]
    box_norm = F.normalize(box_features, p=2, dim=-1, eps=_EPS)    # [B, D]

    # ── 2. Compute cosine similarity ────────────────────────────────────
    # cos_sim = sum(img_norm * box_norm, dim=-1), shape [B]
    cos_sim = (img_norm * box_norm).sum(dim=-1)  # [B]

    # ── 3. Compute loss: L = 1 - cos_sim, averaged over batch ──────────
    loss_per_sample = 1.0 - cos_sim  # [B]
    loss = loss_per_sample.mean()     # scalar

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Cosine Similarity (for logging)
# ──────────────────────────────────────────────────────────────────────────────


def cosine_similarity(
    image_features: torch.Tensor,
    box_features: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the mean cosine similarity between image and box features.

    This is a utility function for logging and monitoring.  It computes:

    .. math::

        \text{cos\_sim} = \frac{1}{B} \sum_{b=1}^{B}
        \frac{f_{img}^{(b)} \cdot f_{box}^{(b)}}
        {\|f_{img}^{(b)}\|_2 \|f_{box}^{(b)}\|_2}

    Parameters
    ----------
    image_features : torch.Tensor
        Image feature vectors, shape ``[B, proj_dim]``.
    box_features : torch.Tensor
        Box feature vectors, shape ``[B, proj_dim]``.

    Returns
    -------
    torch.Tensor
        Scalar mean cosine similarity (0-dimensional).  Ranges in ``[-1, 1]``.

    Example
    -------
    >>> img_feat = torch.randn(4, 128)
    >>> box_feat = torch.randn(4, 128)
    >>> sim = cosine_similarity(img_feat, box_feat)
    >>> sim.shape
    torch.Size([])
    >>> -1.0 <= sim.item() <= 1.0
    True
    """
    img_norm = F.normalize(image_features, p=2, dim=-1, eps=_EPS)
    box_norm = F.normalize(box_features, p=2, dim=-1, eps=_EPS)
    cos_sim = (img_norm * box_norm).sum(dim=-1)  # [B]
    return cos_sim.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Composite Consistency Loss
# ──────────────────────────────────────────────────────────────────────────────


def compute_consistency_loss(
    generated_images: torch.Tensor,
    predicted_boxes: torch.Tensor,
    confidences: torch.Tensor,
    valid_mask: torch.Tensor,
    image_encoder: ImageFeatureEncoder,
    box_encoder: BoxFeatureEncoder,
    consistency_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    r"""Compute the full cross-modal consistency loss in a single call.

    This function orchestrates the entire consistency computation:

    1. Encodes the generated images through ``ImageFeatureEncoder`` to
       obtain :math:`f_{img}`.
    2. Encodes the predicted boxes, confidences, and valid mask through
       ``BoxFeatureEncoder`` to obtain :math:`f_{box}`.
    3. Computes the consistency loss :math:`\mathcal{L}_{cons}`.
    4. Returns a dictionary with the loss and the mean cosine similarity
       for logging.

    The total consistency loss is:

    .. math::

        \mathcal{L}_{total} = w_{cons} \cdot \mathcal{L}_{cons}

    where :math:`w_{cons}` is ``consistency_weight``.

    Parameters
    ----------
    generated_images : torch.Tensor
        Batch of generated images, shape ``[B, 3, H, W]``, pixel values in
        ``[-1, 1]`` (as produced by the generator's Tanh output).
    predicted_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format, normalised to ``[0, 1]``.
    confidences : torch.Tensor
        Objectness confidence scores, shape ``[B, N, 1]`` with values in
        ``[0, 1]``.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` entries indicate valid
        (non-padded) boxes.
    image_encoder : ImageFeatureEncoder
        The image feature encoder module (``ImageFeatureEncoder`` instance).
    box_encoder : BoxFeatureEncoder
        The box feature encoder module (``BoxFeatureEncoder`` instance).
    consistency_weight : float, optional
        Weight scaling factor for the consistency loss.  Must be
        non-negative.  (default: ``0.5``)

    Returns
    -------
    dict of str -> torch.Tensor
        A dictionary with the following keys:

        - ``"consistency_loss"``: the weighted consistency loss
          ``consistency_weight * L_cons`` (scalar).  This is the loss to
          add to the generator's total objective.
        - ``"cosine_similarity"``: the mean cosine similarity between image
          and box features (scalar), useful for logging and monitoring how
          well the two modalities are aligned.

    Raises
    ------
    ValueError
        If ``consistency_weight`` is negative.

    Example
    -------
    >>> img_enc = ImageFeatureEncoder(proj_dim=128)
    >>> box_enc = BoxFeatureEncoder(proj_dim=128)
    >>> images = torch.randn(4, 3, 256, 256)
    >>> boxes = torch.rand(4, 10, 4)
    >>> confs = torch.rand(4, 10, 1)
    >>> mask = torch.rand(4, 10) > 0.5
    >>> result = compute_consistency_loss(
    ...     images, boxes, confs, mask, img_enc, box_enc,
    ...     consistency_weight=0.5,
    ... )
    >>> list(result.keys())
    ['consistency_loss', 'cosine_similarity']
    >>> result['consistency_loss'].shape
    torch.Size([])
    >>> result['cosine_similarity'].shape
    torch.Size([])
    """
    if consistency_weight < 0.0:
        raise ValueError(
            f"consistency_weight must be non-negative, got {consistency_weight}."
        )

    # ── 1. Encode images ─────────────────────────────────────────────────
    # Pass through ImageFeatureEncoder: [B, 3, H, W] -> [B, proj_dim]
    image_features = image_encoder(generated_images)

    # ── 2. Encode boxes ──────────────────────────────────────────────────
    # Pass through BoxFeatureEncoder: [B, N, 4] + [B, N, 1] + [B, N] -> [B, proj_dim]
    box_features = box_encoder(predicted_boxes, confidences, valid_mask)

    # ── 3. Compute consistency loss ──────────────────────────────────────
    cons_loss = consistency_loss(image_features, box_features)

    # ── 4. Compute cosine similarity for logging ─────────────────────────
    cos_sim = cosine_similarity(image_features, box_features)

    # ── 5. Weight the loss ───────────────────────────────────────────────
    weighted_loss = consistency_weight * cons_loss

    return {
        "consistency_loss": weighted_loss,
        "cosine_similarity": cos_sim,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ImageFeatureEncoder",
    "BoxFeatureEncoder",
    "consistency_loss",
    "cosine_similarity",
    "compute_consistency_loss",
]
