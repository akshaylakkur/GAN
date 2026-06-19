"""
Image quality assessment metrics for ILGAN.

Provides three core metric classes/functions for evaluating the quality of
generated images:

1. :class:`FIDCalculator` — Fréchet Inception Distance (FID) between real
   and generated image distributions.  Uses a pre-trained InceptionV3 model
   (penultimate 2048-dim pooling features) and computes the Wasserstein-2
   distance between multivariate Gaussians fitted to the feature
   distributions.

2. :class:`InceptionScoreCalculator` — Inception Score (IS), which measures
   both the diversity and the class-conditional fidelity of generated images.
   Uses a pre-trained InceptionV3 with the full classification head.

3. :func:`compute_image_statistics` — Basic image-level statistics: mean
   pixel value, standard deviation, mean gradient magnitude (sharpness),
   and colour histogram entropy.

All pre-trained model loading is **lazy** (deferred until first use) and
**cached** (a singleton per process).  All feature extraction runs under
``torch.no_grad()`` to minimise memory usage.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_INCEPTION_IMAGE_SIZE: int = 299
"""InceptionV3 expects 299×299 RGB inputs."""

_INCEPTION_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_INCEPTION_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
"""ImageNet normalisation statistics used by InceptionV3."""

_FID_FEATURE_DIM: int = 2048
"""Dimensionality of the penultimate InceptionV3 pooling features."""

_IS_FEATURE_DIM: int = 1000
"""Dimensionality of the InceptionV3 classification logits (ImageNet)."""

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model cache (module-level singletons)
# ──────────────────────────────────────────────────────────────────────────────

_FID_MODEL: Optional[nn.Module] = None
"""Cached InceptionV3 model for FID feature extraction (fc layer removed)."""

_IS_MODEL: Optional[nn.Module] = None
"""Cached InceptionV3 model for Inception Score (full classification head)."""

_DEVICE: Optional[torch.device] = None
"""Device on which the models are placed."""


def _get_device() -> torch.device:
    """Return the current device (GPU if available, else CPU)."""
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
        Shape ``[B, C, H, W]``.  Images are expected in ``[-1, 1]`` range
        (as produced by the ILGAN generator and data pipeline).

    Returns
    -------
    torch.Tensor
        Shape ``[B, 3, 299, 299]``, normalised with ImageNet mean/std,
        on the same device as the input.
    """
    # Resize to 299×299 using bilinear interpolation
    if images.shape[-1] != _INCEPTION_IMAGE_SIZE or images.shape[-2] != _INCEPTION_IMAGE_SIZE:
        images = F.interpolate(
            images,
            size=(_INCEPTION_IMAGE_SIZE, _INCEPTION_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    # Normalise from [-1, 1] to [0, 1] then apply ImageNet stats
    images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    images = torch.clamp(images, 0.0, 1.0)

    mean = torch.tensor(_INCEPTION_MEAN, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(_INCEPTION_STD, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    images = (images - mean) / std

    return images


# ──────────────────────────────────────────────────────────────────────────────
# FID Calculator
# ──────────────────────────────────────────────────────────────────────────────


class _FIDInceptionV3(nn.Module):
    """InceptionV3 truncated at the penultimate pooling layer.

    Returns the 2048-dimensional feature vector after adaptive average
    pooling and flattening, **before** the final dropout and fc layer.
    This is the standard feature space used for FID computation.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        # Load the full InceptionV3 with ImageNet weights
        full_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        )
        full_model.to(device)
        full_model.eval()

        # Extract all layers up to and including Mixed_7c, avgpool, and flatten
        # We need: Conv2d_1a_3x3 through Mixed_7c, then avgpool, then flatten
        self.layers = nn.Sequential(
            full_model.Conv2d_1a_3x3,
            full_model.Conv2d_2a_3x3,
            full_model.Conv2d_2b_3x3,
            full_model.maxpool1,
            full_model.Conv2d_3b_1x1,
            full_model.Conv2d_4a_3x3,
            full_model.maxpool2,
            full_model.Mixed_5b,
            full_model.Mixed_5c,
            full_model.Mixed_5d,
            full_model.Mixed_6a,
            full_model.Mixed_6b,
            full_model.Mixed_6c,
            full_model.Mixed_6d,
            full_model.Mixed_6e,
            full_model.Mixed_7a,
            full_model.Mixed_7b,
            full_model.Mixed_7c,
            full_model.avgpool,
        )
        # Freeze all parameters
        for p in self.layers.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim features.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input, shape ``[B, 3, 299, 299]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 2048]``.
        """
        x = self.layers(x)  # [B, 2048, 1, 1]
        x = torch.flatten(x, start_dim=1)  # [B, 2048]
        return x


def _get_fid_model() -> nn.Module:
    """Lazy-load and cache the FID InceptionV3 feature extractor."""
    global _FID_MODEL
    if _FID_MODEL is None:
        device = _get_device()
        _FID_MODEL = _FIDInceptionV3(device=device)
    return _FID_MODEL


class FIDCalculator:
    r"""Fréchet Inception Distance (FID) calculator.

    The FID measures the distance between the real and generated image
    distributions in the feature space of a pre-trained InceptionV3 network.
    It is defined as:

    .. math::

        \text{FID} = \|\mu_r - \mu_f\|^2_2
            + \operatorname{Tr}\left(
                \Sigma_r + \Sigma_f - 2 (\Sigma_r \Sigma_f)^{1/2}
              \right)

    where :math:`(\mu_r, \Sigma_r)` and :math:`(\mu_f, \Sigma_f)` are the
    mean and covariance of the real and generated feature distributions,
    respectively.

    The calculator supports **incremental accumulation** via
    :meth:`update`, allowing FID to be computed over the entire dataset
    without loading all images into memory at once.

    Parameters
    ----------
    device : torch.device, optional
        Device for feature extraction.  If ``None``, auto-detects GPU.

    Notes
    -----
    - If fewer than 2 samples are provided for either distribution, the
      FID is undefined and :meth:`compute` returns ``NaN``.
    - The pre-trained InceptionV3 model is loaded lazily on the first call
      to :meth:`update` or :meth:`compute_features`.
    - All operations run under ``torch.no_grad()``.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is not None:
            global _DEVICE
            _DEVICE = device

        self._real_features: List[torch.Tensor] = []
        self._fake_features: List[torch.Tensor] = []

    # ── Public API ───────────────────────────────────────────────────────

    def compute_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim InceptionV3 features from a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``[B, C, H, W]``, values in ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 2048]``, extracted features.
        """
        model = _get_fid_model()
        device = _get_device()

        images = images.to(device)
        preprocessed = _preprocess_inception(images)

        with torch.no_grad():
            features = model(preprocessed)

        return features.cpu()

    def update(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> None:
        """Accumulate features from a batch of real and generated images.

        Parameters
        ----------
        real_images : torch.Tensor
            Shape ``[B, C, H, W]``, real images in ``[-1, 1]``.
        fake_images : torch.Tensor
            Shape ``[B, C, H, W]``, generated images in ``[-1, 1]``.
        """
        real_feats = self.compute_features(real_images)
        fake_feats = self.compute_features(fake_images)

        self._real_features.append(real_feats)
        self._fake_features.append(fake_feats)

    def compute(self) -> float:
        """Compute the FID from all accumulated features.

        Returns
        -------
        float
            The FID score.  Returns ``float('nan')`` if either distribution
            has fewer than 2 samples.
        """
        if len(self._real_features) == 0 or len(self._fake_features) == 0:
            return float("nan")

        real_all = torch.cat(self._real_features, dim=0)  # [N_r, 2048]
        fake_all = torch.cat(self._fake_features, dim=0)  # [N_f, 2048]

        n_real = real_all.shape[0]
        n_fake = fake_all.shape[0]

        if n_real < 2 or n_fake < 2:
            return float("nan")

        return self._fid_from_features(real_all, fake_all)

    def reset(self) -> None:
        """Clear all accumulated features."""
        self._real_features.clear()
        self._fake_features.clear()

    @property
    def num_real_samples(self) -> int:
        """Number of real samples accumulated."""
        if not self._real_features:
            return 0
        return sum(f.shape[0] for f in self._real_features)

    @property
    def num_fake_samples(self) -> int:
        """Number of fake (generated) samples accumulated."""
        if not self._fake_features:
            return 0
        return sum(f.shape[0] for f in self._fake_features)

    # ── Static / private helpers ──────────────────────────────────────────

    @staticmethod
    def _fid_from_features(
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> float:
        """Compute FID between two sets of features.

        Parameters
        ----------
        real_features : torch.Tensor
            Shape ``[N_r, D]``.
        fake_features : torch.Tensor
            Shape ``[N_f, D]``.

        Returns
        -------
        float
            The FID score.
        """
        # Compute means
        mu_r = real_features.mean(dim=0)  # [D]
        mu_f = fake_features.mean(dim=0)  # [D]

        # Compute covariances
        sigma_r = FIDCalculator._cov(real_features)  # [D, D]
        sigma_f = FIDCalculator._cov(fake_features)  # [D, D]

        # Mean squared difference
        mean_diff = (mu_r - mu_f).pow(2).sum().item()

        # Trace term: Tr(sigma_r + sigma_f - 2 * (sigma_r * sigma_f)^(1/2))
        # Compute the matrix square root of (sigma_r @ sigma_f)
        cov_sqrt = FIDCalculator._matrix_sqrt(sigma_r @ sigma_f)
        trace_term = (
            torch.trace(sigma_r + sigma_f - 2.0 * cov_sqrt).real.item()
        )

        return mean_diff + trace_term

    @staticmethod
    def _cov(features: torch.Tensor) -> torch.Tensor:
        """Compute the covariance matrix of features.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``[N, D]``.

        Returns
        -------
        torch.Tensor
            Shape ``[D, D]``, the covariance matrix.
        """
        N = features.shape[0]
        mean = features.mean(dim=0, keepdim=True)  # [1, D]
        centered = features - mean  # [N, D]
        cov = (centered.T @ centered) / (N - 1)  # [D, D]
        return cov

    @staticmethod
    def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        """Compute the matrix square root using eigendecomposition.

        For a symmetric positive semi-definite matrix :math:`A`, the
        principal square root :math:`A^{1/2}` satisfies
        :math:`A^{1/2} A^{1/2} = A`.

        Parameters
        ----------
        matrix : torch.Tensor
            Shape ``[D, D]``, assumed symmetric PSD.

        Returns
        -------
        torch.Tensor
            Shape ``[D, D]``, the matrix square root.
        """
        # Symmetrize to avoid numerical asymmetry
        matrix = (matrix + matrix.T) / 2.0

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

        # Clamp eigenvalues to avoid negative values from numerical issues
        eigenvalues = eigenvalues.clamp(min=0.0)

        # sqrt of eigenvalues
        sqrt_eigenvalues = eigenvalues.sqrt()

        # Reconstruct: V * diag(sqrt(lambda)) * V^T
        sqrt_matrix = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T

        # Symmetrize the result
        sqrt_matrix = (sqrt_matrix + sqrt_matrix.T) / 2.0

        return sqrt_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Inception Score Calculator
# ──────────────────────────────────────────────────────────────────────────────


class _ISInceptionV3(nn.Module):
    """InceptionV3 with the full classification head for Inception Score.

    Returns the 1000-dimensional class logits (pre-softmax) for each image.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        full_model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        )
        full_model.to(device)
        full_model.eval()

        # We need the full model including fc layer.
        # In eval mode, InceptionV3.forward returns a plain Tensor [B, 1000].
        self.model = full_model

        # Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits.

        Parameters
        ----------
        x : torch.Tensor
            Preprocessed input, shape ``[B, 3, 299, 299]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, 1000]``, class logits.
        """
        # InceptionV3.forward returns:
        #   - In eval mode: a plain Tensor of shape [B, 1000]
        #   - In train mode with aux_logits: InceptionOutputs namedtuple
        # Since we always run in eval mode, we get a plain tensor.
        output = self.model(x)
        if isinstance(output, torch.Tensor):
            return output
        # Fallback for namedtuple (shouldn't happen in eval mode)
        return output[0]


def _get_is_model() -> nn.Module:
    """Lazy-load and cache the InceptionV3 model for Inception Score."""
    global _IS_MODEL
    if _IS_MODEL is None:
        device = _get_device()
        _IS_MODEL = _ISInceptionV3(device=device)
    return _IS_MODEL


class InceptionScoreCalculator:
    r"""Inception Score (IS) calculator.

    The Inception Score measures the quality and diversity of generated
    images by evaluating how well a pre-trained InceptionV3 classifier
    can distinguish classes in the generated images.  It is defined as:

    .. math::

        \text{IS} = \exp\left(
            \mathbb{E}_{x \sim p_g}\left[
                \text{KL}\big(p(y \mid x) \,\|\, p(y)\big)
            \right]
        \right)

    where :math:`p(y \mid x)` is the conditional label distribution predicted
    by InceptionV3, and :math:`p(y) = \mathbb{E}_{x \sim p_g}[p(y \mid x)]`
    is the marginal label distribution over all generated images.

    A higher IS indicates that the generated images are both **sharp**
    (high confidence in a single class) and **diverse** (the marginal
    distribution is spread across many classes).

    Parameters
    ----------
    device : torch.device, optional
        Device for feature extraction.  If ``None``, auto-detects GPU.
    splits : int, optional
        Number of splits for computing the mean and standard deviation of
        the IS (default: 10).  The IS is computed independently on each
        split and the results are averaged.

    Notes
    -----
    - The pre-trained InceptionV3 model is loaded lazily on the first call
      to :meth:`compute` or :meth:`update`.
    - All operations run under ``torch.no_grad()``.
    - The IS is only meaningful when computed over a sufficiently large
      number of samples (typically > 5000).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        splits: int = 10,
    ) -> None:
        if device is not None:
            global _DEVICE
            _DEVICE = device

        if splits < 1:
            raise ValueError(f"splits must be >= 1, got {splits}")
        self._splits = splits
        self._all_logits: List[torch.Tensor] = []

    # ── Public API ───────────────────────────────────────────────────────

    def compute(self, images: torch.Tensor) -> Tuple[float, float]:
        """Compute the Inception Score for a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``[B, C, H, W]``, values in ``[-1, 1]``.

        Returns
        -------
        mean : float
            Mean Inception Score across splits.
        std : float
            Standard deviation of the Inception Score across splits.
        """
        # Reset and update
        self.reset()
        self.update(images)
        return self.compute_accumulated()

    def update(self, images: torch.Tensor) -> None:
        """Accumulate logits from a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Shape ``[B, C, H, W]``, values in ``[-1, 1]``.
        """
        model = _get_is_model()
        device = _get_device()

        images = images.to(device)
        preprocessed = _preprocess_inception(images)

        with torch.no_grad():
            logits = model(preprocessed)  # [B, 1000]

        self._all_logits.append(logits.cpu())

    def compute_accumulated(self) -> Tuple[float, float]:
        """Compute the Inception Score from all accumulated logits.

        Returns
        -------
        mean : float
            Mean Inception Score across splits.
        std : float
            Standard deviation of the Inception Score across splits.
        """
        if not self._all_logits:
            return float("nan"), float("nan")

        logits = torch.cat(self._all_logits, dim=0)  # [N, 1000]
        N = logits.shape[0]

        if N < 2:
            return float("nan"), float("nan")

        # Convert logits to probabilities via softmax
        probs = F.softmax(logits, dim=1)  # [N, 1000], p(y|x)

        # Split into chunks
        split_size = max(1, N // self._splits)
        splits = torch.split(probs, split_size, dim=0)

        scores = []
        for split in splits:
            # p(y) = mean over samples of p(y|x)
            p_y = split.mean(dim=0, keepdim=True)  # [1, 1000]

            # KL divergence for each sample: sum_y p(y|x) * log(p(y|x) / p(y))
            kl_per_sample = (split * (split.log() - p_y.log())).sum(dim=1)  # [split_size]

            # Mean KL over the split
            mean_kl = kl_per_sample.mean().item()

            # IS = exp(mean KL)
            scores.append(math.exp(mean_kl))

        if not scores:
            return float("nan"), float("nan")

        mean_score = float(torch.tensor(scores).mean())
        std_score = float(torch.tensor(scores).std()) if len(scores) > 1 else 0.0

        return mean_score, std_score

    def reset(self) -> None:
        """Clear all accumulated logits."""
        self._all_logits.clear()

    @property
    def num_samples(self) -> int:
        """Number of samples accumulated."""
        if not self._all_logits:
            return 0
        return sum(l.shape[0] for l in self._all_logits)


# ──────────────────────────────────────────────────────────────────────────────
# Image Statistics
# ──────────────────────────────────────────────────────────────────────────────


def compute_image_statistics(images: torch.Tensor) -> Dict[str, float]:
    """Compute basic image statistics for a batch of images.

    Parameters
    ----------
    images : torch.Tensor
        Shape ``[B, C, H, W]``, values in ``[-1, 1]``.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``"mean_pixel"``: Mean pixel value across all channels and pixels
          (scaled to ``[0, 1]`` range).
        - ``"std_pixel"``: Standard deviation of pixel values.
        - ``"mean_gradient_magnitude"``: Mean gradient magnitude (a measure
          of image sharpness).  Computed via Sobel-like finite differences
          on the luminance channel.
        - ``"color_histogram_entropy"``: Mean entropy of the per-channel
          colour histograms (a measure of colour diversity).
        - ``"mean_channel_mean"``: Mean value per channel (R, G, B).
        - ``"mean_channel_std"``: Standard deviation per channel (R, G, B).

    Notes
    -----
    - All statistics are computed on the **un-normalised** images in
      ``[0, 1]`` range (converted internally from ``[-1, 1]``).
    - Gradient magnitude is computed on the luminance (grayscale) channel
      using central finite differences.
    - Colour histogram entropy uses 64 bins per channel.
    """
    if images.numel() == 0:
        return {
            "mean_pixel": float("nan"),
            "std_pixel": float("nan"),
            "mean_gradient_magnitude": float("nan"),
            "color_histogram_entropy": float("nan"),
            "mean_channel_mean": float("nan"),
            "mean_channel_std": float("nan"),
        }

    # Convert from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0.0, 1.0)

    B, C, H, W = images.shape

    # ── Mean and std pixel values ────────────────────────────────────────
    mean_pixel = images.mean().item()
    std_pixel = images.std().item()

    # ── Per-channel statistics ────────────────────────────────────────────
    channel_means = images.mean(dim=(2, 3))  # [B, C]
    channel_stds = images.std(dim=(2, 3))    # [B, C]

    mean_channel_mean = channel_means.mean(dim=0).tolist()  # list of 3 floats
    mean_channel_std = channel_stds.mean(dim=0).tolist()     # list of 3 floats

    # ── Gradient magnitude (sharpness) on luminance ──────────────────────
    # Convert to grayscale using luminance weights
    if C >= 3:
        luminance = (
            0.2989 * images[:, 0, :, :]
            + 0.5870 * images[:, 1, :, :]
            + 0.1140 * images[:, 2, :, :]
        )  # [B, H, W]
    else:
        luminance = images[:, 0, :, :]  # [B, H, W]

    # Central finite differences for gradient
    # dI/dx: forward difference in x-direction
    grad_x = luminance[:, :, 1:] - luminance[:, :, :-1]  # [B, H, W-1]
    # dI/dy: forward difference in y-direction
    grad_y = luminance[:, 1:, :] - luminance[:, :-1, :]  # [B, H-1, W]

    # Pad to same size for consistent shape
    grad_x = F.pad(grad_x, (0, 1), mode="replicate")  # [B, H, W]
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode="replicate")  # [B, H, W]

    # Gradient magnitude
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)  # [B, H, W]
    mean_gradient_magnitude = grad_mag.mean().item()

    # ── Colour histogram entropy ────────────────────────────────────────
    num_bins = 64
    entropy_per_image: List[float] = []

    for i in range(B):
        img = images[i]  # [C, H, W]
        per_channel_entropy: List[float] = []

        for c in range(C):
            channel = img[c]  # [H, W]
            # Compute histogram with 64 bins over [0, 1]
            hist = torch.histc(channel, bins=num_bins, min=0.0, max=1.0)  # [num_bins]
            # Normalise to probability distribution
            hist = hist / (hist.sum() + 1e-10)
            # Entropy: -sum(p * log(p))
            entropy = -(hist * (hist + 1e-10).log()).sum().item()
            per_channel_entropy.append(entropy)

        # Average entropy across channels for this image
        entropy_per_image.append(float(torch.tensor(per_channel_entropy).mean()))

    color_histogram_entropy = float(torch.tensor(entropy_per_image).mean())

    return {
        "mean_pixel": mean_pixel,
        "std_pixel": std_pixel,
        "mean_gradient_magnitude": mean_gradient_magnitude,
        "color_histogram_entropy": color_histogram_entropy,
        "mean_channel_mean": mean_channel_mean,
        "mean_channel_std": mean_channel_std,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Utility: clear model cache
# ──────────────────────────────────────────────────────────────────────────────


def clear_model_cache() -> None:
    """Clear the cached InceptionV3 models to free GPU memory.

    Call this after you are done computing metrics to release the GPU
    memory held by the pre-trained models.
    """
    global _FID_MODEL, _IS_MODEL
    _FID_MODEL = None
    _IS_MODEL = None
