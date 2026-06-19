"""
ILGAN metrics — image quality assessment, bounding box evaluation, and
joint quality tracking for the one-shot image + bounding box generation
pipeline.

The ``ilgan.metrics`` package provides four categories of evaluation tools:

1. **Image metrics** (:mod:`ilgan.metrics.image_metrics`) — FID, Inception
   Score, and basic image statistics (mean pixel, gradient magnitude,
   colour histogram entropy).

2. **Advanced image metrics** (:mod:`ilgan.metrics.advanced_image_metrics`) —
   Precision & Recall (Kynkäänniemi et al., 2019), Density & Coverage
   (Naeem et al., 2020), LPIPS (Zhang et al., 2018), and sFID (spatial
   FID using segmentation features).

3. **Box metrics** (:mod:`ilgan.metrics.box_metrics`) — mAP, GIoU, box
   statistics (confidence, size, spatial diversity), and detection accuracy.

4. **Joint metrics** (:mod:`ilgan.metrics.joint_metrics`) — A heuristic
   joint quality score combining FID, mAP, and Inception Score, a stateful
   :class:`MetricsTracker` for accumulating metrics across batches, and a
   :func:`format_metrics` utility for console logging.

The :func:`build_metrics_tracker` factory function provides a convenient
way to construct a fully configured :class:`MetricsTracker` from an ILGAN
:class:`~ilgan.utils.config.Config` object.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Re-export all public symbols from submodules
# ──────────────────────────────────────────────────────────────────────────────

from ilgan.metrics.image_metrics import (
    FIDCalculator,
    InceptionScoreCalculator,
    clear_model_cache,
    compute_image_statistics,
)
from ilgan.metrics.advanced_image_metrics import (
    PrecisionRecallCalculator,
    DensityCoverageCalculator,
    LPIPSCalculator,
    SpatialFIDCalculator,
    clear_advanced_model_cache,
)
from ilgan.metrics.box_metrics import (
    compute_map,
    compute_giou,
    compute_box_statistics,
    compute_detection_accuracy,
)
from ilgan.metrics.joint_metrics import (
    compute_joint_score,
    MetricsTracker,
    format_metrics,
)

# ──────────────────────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────────────────────


def build_metrics_tracker(config: Any) -> MetricsTracker:
    """Build a fully configured :class:`MetricsTracker` from an ILGAN
    :class:`~ilgan.utils.config.Config` object.

    This factory reads the following fields from *config*:

    - ``model.num_classes`` → number of object classes for mAP computation.
    - ``model.image_size`` → image spatial size (used for device selection
      heuristics; the actual size is handled by the data pipeline).
    - ``loss.adv_weight``, ``loss.box_weight``, ``loss.diversity_weight``,
      ``loss.consistency_weight`` → used to derive the joint score weights
      (see notes below).

    Parameters
    ----------
    config : Config
        An ILGAN configuration object (instance of
        :class:`ilgan.utils.config.Config`).  Must contain at minimum
        ``model.num_classes``.

    Returns
    -------
    MetricsTracker
        A :class:`MetricsTracker` instance configured with the correct
        number of classes, device, and joint score weights derived from
        the loss configuration.

    Raises
    ------
    KeyError
        If ``model.num_classes`` is missing from the config.

    Notes
    -----
    **Joint score weight derivation**

    The joint score weights are derived from the loss weights in the config
    to reflect the relative importance of each component during training:

    .. math::

        w_{\text{FID}} &= \\frac{w_{\\text{adv}}}{w_{\\text{adv}} + w_{\\text{box}} + w_{\\text{consistency}}} \\\\
        w_{\text{mAP}} &= \\frac{w_{\\text{box}}}{w_{\\text{adv}} + w_{\\text{box}} + w_{\\text{consistency}}} \\\\
        w_{\text{IS}}   &= \\frac{w_{\\text{consistency}}}{w_{\\text{adv}} + w_{\\text{box}} + w_{\\text{consistency}}}

    This ensures that when the box loss weight is high (e.g., 5.0), the mAP
    component contributes more to the joint score, and when the adversarial
    loss weight is high, the FID component dominates.  If all loss weights
    are zero, the default weights ``{fid: 0.4, map: 0.4, is: 0.2}`` are
    used.

    **Device selection**

    The device is automatically set to the first available CUDA GPU, or
    falls back to CPU if no GPU is available.

    **Image size**

    The ``model.image_size`` field is read from the config but is **not**
    passed to the :class:`MetricsTracker` (which operates on arbitrary-size
    images).  It is included in the factory for forward compatibility and
    for potential future use in normalisation heuristics.

    Examples
    --------
    >>> from ilgan.utils.config import Config
    >>> cfg = Config()
    >>> tracker = build_metrics_tracker(cfg)
    >>> tracker.num_classes
    80
    """
    # ── Extract required fields ──────────────────────────────────────────
    try:
        num_classes = int(config["model.num_classes"])
    except (KeyError, TypeError) as e:
        raise KeyError(
            "build_metrics_tracker requires 'model.num_classes' in config. "
            f"Got error: {e}"
        ) from e

    # ── Device selection ─────────────────────────────────────────────────
    from ilgan.utils.device import get_device
    device = get_device()

    # ── Derive joint score weights from loss configuration ──────────────
    # Read loss weights with safe defaults
    adv_weight = _safe_get_float(config, "loss.adv_weight", 1.0)
    box_weight = _safe_get_float(config, "loss.box_weight", 5.0)
    consistency_weight = _safe_get_float(config, "loss.consistency_weight", 0.5)

    total = adv_weight + box_weight + consistency_weight

    if total > 0.0:
        joint_weights: Dict[str, float] = {
            "fid": adv_weight / total,
            "map": box_weight / total,
            "is": consistency_weight / total,
        }
    else:
        # Fallback to default weights if all loss weights are zero
        joint_weights = {
            "fid": 0.4,
            "map": 0.4,
            "is": 0.2,
        }

    # ── Construct and return the tracker ─────────────────────────────────
    return MetricsTracker(
        num_classes=num_classes,
        device=device,
        joint_weights=joint_weights,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_get_float(config: Any, key: str, default: float) -> float:
    """Safely extract a float value from a config object.

    Parameters
    ----------
    config : Config
        The configuration object.
    key : str
        Dotted key path (e.g., ``"loss.adv_weight"``).
    default : float
        Default value if the key is missing or the value is ``None``.

    Returns
    -------
    float
        The extracted value, or *default* if the key is not present.
    """
    try:
        val = config[key]
        if val is None:
            return default
        return float(val)
    except (KeyError, TypeError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Image metrics
    "FIDCalculator",
    "InceptionScoreCalculator",
    "compute_image_statistics",
    "clear_model_cache",
    # Advanced image metrics
    "PrecisionRecallCalculator",
    "DensityCoverageCalculator",
    "LPIPSCalculator",
    "SpatialFIDCalculator",
    "clear_advanced_model_cache",
    # Box metrics
    "compute_map",
    "compute_giou",
    "compute_box_statistics",
    "compute_detection_accuracy",
    # Joint metrics
    "compute_joint_score",
    "MetricsTracker",
    "format_metrics",
    # Factory
    "build_metrics_tracker",
]
