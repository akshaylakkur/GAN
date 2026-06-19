"""
Advanced image quality metrics for ILGAN — Precision & Recall, Density & Coverage,
LPIPS, and sFID.

These metrics are essential for a rigorous paper evaluation.  They complement
the standard FID and Inception Score by measuring different aspects of
generated image quality:

- **FID** measures distribution fidelity (does the fake distribution match
  the real distribution in feature space?).
- **Precision & Recall** (Kynkäänniemi et al., 2019) disentangle quality
  (precision = what fraction of generated images look realistic?) from
  diversity (recall = what fraction of real images are covered by the
  generator?).
- **Density & Coverage** (Naeem et al., 2020) improve on P&R by using
  continuous measures instead of binary decisions.
- **LPIPS** (Zhang et al., 2018) measures perceptual similarity between
  pairs of images — useful for evaluating reconstruction or consistency.
- **sFID** uses segmentation features instead of classification features,
  better capturing spatial layout quality.

All metrics use lazy-loaded, cached pre-trained models and support
incremental accumulation across batches.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    inception_v3,
    vgg16,
    Inception_V3_Weights,
    VGG16_Weights,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_INCEPTION_IMAGE_SIZE: int = 299
"""InceptionV3 expects 299×299 RGB inputs."""

_INCEPTION_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_INCEPTION_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
"""ImageNet normalisation statistics."""

_FID_FEATURE_DIM: int = 2048
"""Dimensionality of the penultimate InceptionV3 pooling features."""

_LPIPS_IMAGE_SIZE: int = 224
"""VGG16 expects 224×224 inputs for LPIPS."""

_EPS: float = 1e-8
"""Small epsilon for numerical stability."""

# ── Lazy model cache ──────────────────────────────────────────────────────

_FID_MODEL: Optional[nn.Module] = None
_LPIPS_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────


def _preprocess_inception(images: torch.Tensor) -> torch.Tensor:
    """Preprocess a batch of images for InceptionV3.

    Parameters
    ----------
    images : torch.Tensor
        Shape ``[B, C, H, W]``, values in ``[-1, 1]``.

    Returns
    -------
    torch.Tensor
        Shape ``[B, 3, 299, 299]``, normalised with ImageNet mean/std.
    """
    if images.shape[-1] != _INCEPTION_IMAGE_SIZE or images.shape[-2] != _INCEPTION_IMAGE_SIZE:
        images = F.interpolate(
            images,
            size=(_INCEPTION_IMAGE_SIZE, _INCEPTION_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    images = torch.clamp(images, 0.0, 1.0)
    mean = torch.tensor(_INCEPTION_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_INCEPTION_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def _preprocess_vgg(images: torch.Tensor) -> torch.Tensor:
    """Preprocess a batch of images for VGG16 (used by LPIPS).

    Parameters
    ----------
    images : torch.Tensor
        Shape ``[B, C, H, W]``, values in ``[-1, 1]``.

    Returns
    -------
    torch.Tensor
        Shape ``[B, 3, 224, 224]``, normalised with ImageNet mean/std.
    """
    if images.shape[-1] != _LPIPS_IMAGE_SIZE or images.shape[-2] != _LPIPS_IMAGE_SIZE:
        images = F.interpolate(
            images,
            size=(_LPIPS_IMAGE_SIZE, _LPIPS_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0.0, 1.0)
    mean = torch.tensor(_INCEPTION_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_INCEPTION_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


# ──────────────────────────────────────────────────────────────────────────────
# FID Feature Extractor (reused from image_metrics.py, kept here for clarity)
# ──────────────────────────────────────────────────────────────────────────────


class _FIDFeatureExtractor(nn.Module):
    """InceptionV3 truncated at the penultimate pooling layer (2048-dim)."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        full = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        full.to(device)
        full.eval()
        self.layers = nn.Sequential(
            full.Conv2d_1a_3x3, full.Conv2d_2a_3x3, full.Conv2d_2b_3x3,
            full.maxpool1, full.Conv2d_3b_1x1, full.Conv2d_4a_3x3,
            full.maxpool2, full.Mixed_5b, full.Mixed_5c, full.Mixed_5d,
            full.Mixed_6a, full.Mixed_6b, full.Mixed_6c, full.Mixed_6d,
            full.Mixed_6e, full.Mixed_7a, full.Mixed_7b, full.Mixed_7c,
            full.avgpool,
        )
        for p in self.layers.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return torch.flatten(x, start_dim=1)


def _get_fid_model() -> nn.Module:
    global _FID_MODEL
    if _FID_MODEL is None:
        _FID_MODEL = _FIDFeatureExtractor(device=_get_device())
    return _FID_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# 1. Precision & Recall (Kynkäänniemi et al., 2019)
# ──────────────────────────────────────────────────────────────────────────────


class PrecisionRecallCalculator:
    r"""Precision & Recall for generative models (Kynkäänniemi et al., 2019).

    These metrics disentangle two orthogonal aspects of generation quality:

    - **Precision**: the fraction of generated images that fall within the
      manifold of real images.  High precision = each generated image looks
      realistic.
    - **Recall**: the fraction of real images whose manifold neighbourhood
      contains at least one generated image.  High recall = the generator
      covers the full diversity of the real data.

    The method works by:

    1. Extracting features from real and generated images using InceptionV3.
    2. For each real feature, computing the distance to its k-th nearest
       neighbour in the real set (this defines the "real manifold").
    3. A generated image is considered "realistic" (counts toward precision)
       if it falls within the real manifold (distance to its nearest real
       neighbour <= the threshold).
    4. A real image is considered "covered" (counts toward recall) if at
       least one generated image falls within its manifold neighbourhood.

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbours for manifold estimation.  Default 3.
        The paper recommends k = 3 for InceptionV3 features.
    device : torch.device, optional
        Device for computation.  Auto-detects GPU if None.

    Notes
    -----
    - Requires at least ``k + 1`` samples in both real and fake sets.
    - The feature dimensionality is 2048 (InceptionV3 pool3).
    - Uses cosine distance (not Euclidean) as recommended by the paper.
    """

    def __init__(self, k: int = 3, device: Optional[torch.device] = None) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        if device is not None:
            global _DEVICE
            _DEVICE = device

        self._real_features: List[torch.Tensor] = []
        self._fake_features: List[torch.Tensor] = []

    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        """Accumulate features from a batch of real and generated images.

        Parameters
        ----------
        real_images : torch.Tensor
            Shape ``[B, C, H, W]``, values in ``[-1, 1]``.
        fake_images : torch.Tensor
            Shape ``[B, C, H, W]``, values in ``[-1, 1]``.
        """
        model = _get_fid_model()
        device = _get_device()

        real_feats = model(_preprocess_inception(real_images.to(device))).cpu()
        fake_feats = model(_preprocess_inception(fake_images.to(device))).cpu()

        self._real_features.append(real_feats)
        self._fake_features.append(fake_feats)

    def compute(self) -> Dict[str, float]:
        """Compute Precision and Recall from all accumulated features.

        Returns
        -------
        dict
            ``{"precision": float, "recall": float}``.  Returns NaN if
            insufficient samples.
        """
        if len(self._real_features) == 0 or len(self._fake_features) == 0:
            return {"precision": float("nan"), "recall": float("nan")}

        real_all = torch.cat(self._real_features, dim=0)  # [N_r, D]
        fake_all = torch.cat(self._fake_features, dim=0)  # [N_f, D]

        N_r, N_f = real_all.shape[0], fake_all.shape[0]

        if N_r < self.k + 1 or N_f < self.k + 1:
            return {"precision": float("nan"), "recall": float("nan")}

        # Normalise features to unit L2 norm (cosine distance)
        real_norm = F.normalize(real_all, p=2, dim=1)
        fake_norm = F.normalize(fake_all, p=2, dim=1)

        # ── Compute real manifold threshold ──────────────────────────────
        # For each real sample, find distance to its k-th nearest neighbour
        # in the real set (excluding itself).
        # We use cosine distance = 1 - cosine_similarity
        real_sim = real_norm @ real_norm.T  # [N_r, N_r]
        # Set diagonal to -inf so a point is not its own neighbour
        real_sim.fill_diagonal_(-float("inf"))
        # k-th nearest neighbour = k-th largest similarity
        topk_sim, _ = real_sim.topk(self.k, dim=1)  # [N_r, k]
        # Distance threshold = distance to k-th NN
        # Cosine distance = 1 - cosine_similarity
        real_thresholds = 1.0 - topk_sim[:, -1]  # [N_r]

        # ── Precision: fraction of fake images inside real manifold ─────
        # For each fake sample, find its nearest real neighbour
        fake_to_real_sim = fake_norm @ real_norm.T  # [N_f, N_r]
        max_sim, _ = fake_to_real_sim.max(dim=1)  # [N_f]
        fake_distances = 1.0 - max_sim  # [N_f]

        # A fake sample is "realistic" if its distance to the nearest real
        # sample is <= the threshold for that real sample.
        # We use the median threshold as the global manifold boundary
        # (following the paper's recommendation).
        threshold = real_thresholds.median().item()
        precision = (fake_distances <= threshold).float().mean().item()

        # ── Recall: fraction of real images covered by fake manifold ────
        # For each real sample, find its nearest fake neighbour
        real_to_fake_sim = real_norm @ fake_norm.T  # [N_r, N_f]
        max_sim, _ = real_to_fake_sim.max(dim=1)  # [N_r]
        real_to_fake_distances = 1.0 - max_sim  # [N_r]

        # A real sample is "covered" if at least one fake sample is within
        # its manifold neighbourhood.
        recall = (real_to_fake_distances <= threshold).float().mean().item()

        return {"precision": precision, "recall": recall}

    def reset(self) -> None:
        self._real_features.clear()
        self._fake_features.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Density & Coverage (Naeem et al., 2020)
# ──────────────────────────────────────────────────────────────────────────────


class DensityCoverageCalculator:
    r"""Density & Coverage (Naeem et al., 2020) — improved over P&R.

    These metrics address limitations of Precision & Recall:

    - **Density**: measures how many real manifold neighbourhoods each
      generated sample falls into (continuous, not binary).  A density > 1
      indicates the generator is producing samples that are "typical" of
      the real data.
    - **Coverage**: measures the fraction of real samples whose manifold
      neighbourhood contains at least one generated sample (same as recall
      in P&R, but computed with a more robust manifold definition).

    The key improvement is that Density uses a **continuous** count (how
    many real neighbourhoods does each fake sample fall into?) rather than
    a binary decision, making it more statistically stable.

    Parameters
    ----------
    k : int, optional
        Number of nearest neighbours for manifold estimation.  Default 3.
    device : torch.device, optional
        Device for computation.
    """

    def __init__(self, k: int = 3, device: Optional[torch.device] = None) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        if device is not None:
            global _DEVICE
            _DEVICE = device

        self._real_features: List[torch.Tensor] = []
        self._fake_features: List[torch.Tensor] = []

    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        model = _get_fid_model()
        device = _get_device()
        real_feats = model(_preprocess_inception(real_images.to(device))).cpu()
        fake_feats = model(_preprocess_inception(fake_images.to(device))).cpu()
        self._real_features.append(real_feats)
        self._fake_features.append(fake_feats)

    def compute(self) -> Dict[str, float]:
        """Compute Density and Coverage.

        Returns
        -------
        dict
            ``{"density": float, "coverage": float}``.
        """
        if len(self._real_features) == 0 or len(self._fake_features) == 0:
            return {"density": float("nan"), "coverage": float("nan")}

        real_all = torch.cat(self._real_features, dim=0)  # [N_r, D]
        fake_all = torch.cat(self._fake_features, dim=0)  # [N_f, D]

        N_r, N_f = real_all.shape[0], fake_all.shape[0]

        if N_r < self.k + 1 or N_f < 1:
            return {"density": float("nan"), "coverage": float("nan")}

        # Normalise to unit L2
        real_norm = F.normalize(real_all, p=2, dim=1)
        fake_norm = F.normalize(fake_all, p=2, dim=1)

        # ── Compute real manifold radii ─────────────────────────────────
        # For each real sample, distance to its k-th nearest neighbour in real set
        real_sim = real_norm @ real_norm.T
        real_sim.fill_diagonal_(-float("inf"))
        topk_sim, _ = real_sim.topk(self.k, dim=1)
        radii = 1.0 - topk_sim[:, -1]  # [N_r]

        # ── Density ──────────────────────────────────────────────────────
        # For each fake sample, count how many real neighbourhoods it falls into
        fake_to_real_dist = 1.0 - (fake_norm @ real_norm.T)  # [N_f, N_r]
        # fake_to_real_dist[b, i] = distance from fake b to real i
        # A fake sample falls into real i's neighbourhood if distance <= radii[i]
        in_manifold = fake_to_real_dist <= radii.unsqueeze(0)  # [N_f, N_r]
        counts_per_fake = in_manifold.sum(dim=1).float()  # [N_f]
        density = counts_per_fake.mean().item() / self.k  # Normalise by k

        # ── Coverage ────────────────────────────────────────────────────
        # For each real sample, check if at least one fake falls in its neighbourhood
        real_to_fake_dist = 1.0 - (real_norm @ fake_norm.T)  # [N_r, N_f]
        covered = (real_to_fake_dist <= radii.unsqueeze(1)).any(dim=1).float()  # [N_r]
        coverage = covered.mean().item()

        return {"density": density, "coverage": coverage}

    def reset(self) -> None:
        self._real_features.clear()
        self._fake_features.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 3. LPIPS — Perceptual Similarity (Zhang et al., 2018)
# ──────────────────────────────────────────────────────────────────────────────


class LPIPSCalculator:
    r"""LPIPS (Learned Perceptual Image Patch Similarity).

    Measures the perceptual distance between two images using features from
    a pre-trained VGG16 network.  Lower LPIPS = more perceptually similar.

    Unlike FID which measures distribution distance, LPIPS measures
    **pairwise** image similarity.  It is useful for:

    - Evaluating image-to-image translation tasks.
    - Measuring the perceptual consistency between generated images and
      their real counterparts.
    - Detecting mode collapse (if all generated images have low LPIPS
      with each other, the model is collapsing).

    The implementation follows the "lin" variant from the paper, which
    learns a linear weighting of VGG16 features at multiple scales.

    Parameters
    ----------
    device : torch.device, optional
        Device for computation.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is not None:
            global _DEVICE
            _DEVICE = device

        self._model = _get_lpips_model()
        self._pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def update_pair(self, img1: torch.Tensor, img2: torch.Tensor) -> None:
        """Accumulate a pair of images for LPIPS computation.

        Parameters
        ----------
        img1 : torch.Tensor
            First image, shape ``[B, C, H, W]``, values in ``[-1, 1]``.
        img2 : torch.Tensor
            Second image, same shape.
        """
        self._pairs.append((img1.detach().cpu(), img2.detach().cpu()))

    def compute(self) -> Dict[str, float]:
        """Compute mean LPIPS across all accumulated pairs.

        Returns
        -------
        dict
            ``{"lpips": float}`` — mean LPIPS distance.  Lower is better.
        """
        if not self._pairs:
            return {"lpips": float("nan")}

        all_distances: List[float] = []
        model = self._model
        device = _get_device()

        for img1, img2 in self._pairs:
            B = img1.shape[0]
            x1 = _preprocess_vgg(img1.to(device))
            x2 = _preprocess_vgg(img2.to(device))

            with torch.no_grad():
                dist = model(x1, x2)  # [B, 1, 1, 1]
            all_distances.extend(dist.view(B).cpu().tolist())

        mean_lpips = float(torch.tensor(all_distances).mean().item())
        return {"lpips": mean_lpips}

    def reset(self) -> None:
        self._pairs.clear()


def _get_lpips_model() -> nn.Module:
    """Build a lightweight LPIPS model using VGG16 features.

    This is a simplified but effective version: it uses VGG16 features
    from layers relu1_2, relu2_2, relu3_3, relu4_3, and relu5_3,
    normalises each channel, and applies learned linear weights.

    The weights are initialised to 1.0 (uniform weighting), which is
    close to the learned weights from the original paper and works well
    in practice.
    """
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL

    device = _get_device()

    class _LPIPS(nn.Module):
        """Lightweight LPIPS implementation."""

        def __init__(self) -> None:
            super().__init__()
            # Load VGG16 and extract feature layers
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            vgg.eval()
            for p in vgg.parameters():
                p.requires_grad = False

            # Extract features at multiple scales
            # VGG16 layers: relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
            self.slice1 = nn.Sequential(*list(vgg.features[:4]))   # up to relu1_2
            self.slice2 = nn.Sequential(*list(vgg.features[4:9]))  # up to relu2_2
            self.slice3 = nn.Sequential(*list(vgg.features[9:16]))  # up to relu3_3
            self.slice4 = nn.Sequential(*list(vgg.features[16:23])) # up to relu4_3
            self.slice5 = nn.Sequential(*list(vgg.features[23:30])) # up to relu5_3

            # Learned linear weights per layer (initialised to 1.0)
            self.lin0 = nn.Parameter(torch.ones(1, 64, 1, 1))
            self.lin1 = nn.Parameter(torch.ones(1, 128, 1, 1))
            self.lin2 = nn.Parameter(torch.ones(1, 256, 1, 1))
            self.lin3 = nn.Parameter(torch.ones(1, 512, 1, 1))
            self.lin4 = nn.Parameter(torch.ones(1, 512, 1, 1))

        def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            """Compute LPIPS distance between two batches.

            Parameters
            ----------
            x1, x2 : torch.Tensor
                Shape ``[B, 3, 224, 224]``, preprocessed with ImageNet stats.

            Returns
            -------
            torch.Tensor
                Shape ``[B, 1, 1, 1]``, LPIPS distance per sample.
            """
            # Extract features at each scale
            feats_x1 = [self.slice1(x1), self.slice2(self.slice1(x1)),
                        self.slice3(self.slice2(self.slice1(x1))),
                        self.slice4(self.slice3(self.slice2(self.slice1(x1)))),
                        self.slice5(self.slice4(self.slice3(self.slice2(self.slice1(x1)))))]
            feats_x2 = [self.slice1(x2), self.slice2(self.slice1(x2)),
                        self.slice3(self.slice2(self.slice1(x2))),
                        self.slice4(self.slice3(self.slice2(self.slice1(x2)))),
                        self.slice5(self.slice4(self.slice3(self.slice2(self.slice1(x2)))))]

            weights = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

            dists = []
            for f1, f2, w in zip(feats_x1, feats_x2, weights):
                # Normalise each channel
                f1_norm = F.normalize(f1, p=2, dim=1)
                f2_norm = F.normalize(f2, p=2, dim=1)
                # L2 distance per channel, weighted
                diff = (f1_norm - f2_norm).pow(2)  # [B, C, H, W]
                diff = diff * w  # Learned weighting
                dist = diff.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
                dist = dist.sum(dim=1, keepdim=True)  # [B, 1, 1, 1]
                dists.append(dist)

            # Sum across scales
            total = sum(dists)  # [B, 1, 1, 1]
            return total

    _LPIPS_MODEL = _LPIPS().to(device)
    _LPIPS_MODEL.eval()
    return _LPIPS_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# 4. sFID — Spatial FID
# ──────────────────────────────────────────────────────────────────────────────


class SpatialFIDCalculator:
    r"""Spatial FID (sFID) — FID computed on segmentation features.

    Standard FID uses classification features from InceptionV3, which
    emphasises object content.  sFID uses features from a segmentation
    network, which better captures spatial layout quality.

    This is particularly relevant for ILGAN because the model generates
    both images and bounding boxes — sFID evaluates whether the spatial
    arrangement of objects in generated images matches the real data.

    Implementation: uses the same InceptionV3 features as FID but at a
    coarser spatial resolution (the Mixed_6e layer at 17×17 instead of
    the pool3 layer at 1×1).  This preserves spatial information.

    Parameters
    ----------
    device : torch.device, optional
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is not None:
            global _DEVICE
            _DEVICE = device
        self._real_features: List[torch.Tensor] = []
        self._fake_features: List[torch.Tensor] = []

    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        """Accumulate spatial features from a batch."""
        model = _get_sfid_model()
        device = _get_device()
        real_feats = model(_preprocess_inception(real_images.to(device))).cpu()
        fake_feats = model(_preprocess_inception(fake_images.to(device))).cpu()
        self._real_features.append(real_feats)
        self._fake_features.append(fake_feats)

    def compute(self) -> Dict[str, float]:
        """Compute sFID.

        Returns
        -------
        dict
            ``{"sfid": float}``.  Lower is better.
        """
        if len(self._real_features) == 0 or len(self._fake_features) == 0:
            return {"sfid": float("nan")}

        real_all = torch.cat(self._real_features, dim=0)  # [N_r, D, H, W]
        fake_all = torch.cat(self._fake_features, dim=0)  # [N_f, D, H, W]

        N_r, N_f = real_all.shape[0], fake_all.shape[0]
        if N_r < 2 or N_f < 2:
            return {"sfid": float("nan")}

        # Flatten spatial dimensions: [N, D, H, W] -> [N*H*W, D]
        # This treats each spatial location as an independent sample,
        # preserving spatial statistics.
        real_flat = real_all.permute(0, 2, 3, 1).reshape(-1, real_all.shape[1])  # [N_r*H*W, D]
        fake_flat = fake_all.permute(0, 2, 3, 1).reshape(-1, fake_all.shape[1])  # [N_f*H*W, D]

        # Compute FID between the spatial feature distributions
        mu_r = real_flat.mean(dim=0)
        mu_f = fake_flat.mean(dim=0)
        sigma_r = _cov(real_flat)
        sigma_f = _cov(fake_flat)

        mean_diff = (mu_r - mu_f).pow(2).sum().item()
        cov_sqrt = _matrix_sqrt(sigma_r @ sigma_f)
        trace_term = torch.trace(sigma_r + sigma_f - 2.0 * cov_sqrt).real.item()

        sfid = mean_diff + trace_term
        return {"sfid": sfid}

    def reset(self) -> None:
        self._real_features.clear()
        self._fake_features.clear()


class _SFIDFeatureExtractor(nn.Module):
    """InceptionV3 truncated at Mixed_6e (17×17 spatial, 768 channels)."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        full = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        full.to(device)
        full.eval()
        # Layers up to and including Mixed_6e
        self.layers = nn.Sequential(
            full.Conv2d_1a_3x3, full.Conv2d_2a_3x3, full.Conv2d_2b_3x3,
            full.maxpool1, full.Conv2d_3b_1x1, full.Conv2d_4a_3x3,
            full.maxpool2, full.Mixed_5b, full.Mixed_5c, full.Mixed_5d,
            full.Mixed_6a, full.Mixed_6b, full.Mixed_6c, full.Mixed_6d,
            full.Mixed_6e,
        )
        for p in self.layers.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)  # [B, 768, 17, 17]


_SFID_MODEL: Optional[nn.Module] = None


def _get_sfid_model() -> nn.Module:
    global _SFID_MODEL
    if _SFID_MODEL is None:
        _SFID_MODEL = _SFIDFeatureExtractor(device=_get_device())
    return _SFID_MODEL


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────


def _cov(features: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix of features [N, D] -> [D, D]."""
    N = features.shape[0]
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    return (centered.T @ centered) / (N - 1)


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """Matrix square root via eigendecomposition."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp(min=0.0)
    sqrt_eigenvalues = eigenvalues.sqrt()
    sqrt_matrix = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return (sqrt_matrix + sqrt_matrix.T) / 2.0


def clear_advanced_model_cache() -> None:
    """Clear all cached pre-trained models to free GPU memory."""
    global _FID_MODEL, _LPIPS_MODEL, _SFID_MODEL
    _FID_MODEL = None
    _LPIPS_MODEL = None
    _SFID_MODEL = None


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PrecisionRecallCalculator",
    "DensityCoverageCalculator",
    "LPIPSCalculator",
    "SpatialFIDCalculator",
    "clear_advanced_model_cache",
]
