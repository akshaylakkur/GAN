"""
Joint quality metrics for ILGAN — combining image quality and bounding box
accuracy into a single evaluation framework.

This module provides three core components:

1. :func:`compute_joint_score` — A heuristic weighted combination of FID,
   mAP, and Inception Score that produces a single scalar for tracking
   overall model progress during training.

2. :class:`MetricsTracker` — A stateful accumulator that maintains running
   statistics for image metrics (FID, IS, image statistics), box metrics
   (mAP, GIoU, detection accuracy, box statistics), and loss metrics
   (generator loss, discriminator loss, box loss, etc.).  It provides
   methods to update, compute, reset, and log all metrics.

3. :func:`format_metrics` — A utility that formats a metrics dictionary
   for console logging with appropriate per-metric precision.

Mathematical grounding
----------------------
The joint score is defined as:

.. math::

    J = w_{\text{mAP}} \cdot \text{mAP}
        + w_{\text{IS}} \cdot \frac{\text{IS}}{10}
        - w_{\text{FID}} \cdot \frac{\text{FID}}{100}

where :math:`w_{\text{mAP}} + w_{\text{IS}} + w_{\text{FID}} = 1`.

This formulation ensures that:

- **Higher mAP** (better detection) increases the joint score.
- **Higher IS** (better image diversity/sharpness) increases the joint score,
  normalised by 10 since IS typically ranges in :math:`[1, 10]`.
- **Lower FID** (closer to real distribution) increases the joint score
  (subtraction of the normalised FID).

The joint score is a **heuristic** — it is not a mathematically rigorous
metric but provides a convenient single number for tracking overall
progress during training.  For rigorous evaluation, inspect the individual
metrics separately.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ilgan.metrics.image_metrics import (
    FIDCalculator,
    InceptionScoreCalculator,
    compute_image_statistics,
    clear_model_cache,
)
from ilgan.metrics.box_metrics import (
    compute_map,
    compute_giou,
    compute_box_statistics,
    compute_detection_accuracy,
)
from ilgan.utils.logger import Logger


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_JOINT_WEIGHTS: Dict[str, float] = {
    "fid": 0.4,
    "map": 0.4,
    "is": 0.2,
}
"""Default weights for the joint score computation.

- ``fid``: weight for the (negative) normalised FID contribution.
- ``map``: weight for the mAP contribution.
- ``is``: weight for the normalised Inception Score contribution.

These sum to 1.0.
"""

_FID_NORMALISER: float = 100.0
"""Denominator for normalising FID values.  Typical FID ranges from 0
(perfect) to several hundred (poor).  Dividing by 100 brings it into a
:math:`[0, \sim 3]` range."""

_IS_NORMALISER: float = 10.0
"""Denominator for normalising Inception Score values.  Typical IS ranges
from 1 (poor) to :math:`\sim 10` (excellent).  Dividing by 10 brings it
into a :math:`[0.1, 1.0]` range."""


# ──────────────────────────────────────────────────────────────────────────────
# Joint Score
# ──────────────────────────────────────────────────────────────────────────────


def compute_joint_score(
    fid_score: float,
    mAP_score: float,
    inception_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    r"""Compute a heuristic joint score combining image quality and detection
    accuracy.

    The joint score is a weighted combination that normalises each
    component to a comparable scale:

    .. math::

        J = w_{\text{mAP}} \cdot \text{mAP}
            + w_{\text{IS}} \cdot \frac{\text{IS}}{10}
            - w_{\text{FID}} \cdot \frac{\text{FID}}{100}

    Parameters
    ----------
    fid_score : float
        Fréchet Inception Distance (lower is better).  If ``NaN`` or
        ``inf``, the FID contribution is set to 0.
    mAP_score : float
        Mean Average Precision (higher is better, range :math:`[0, 1]`).
        If ``NaN`` or ``inf``, the mAP contribution is set to 0.
    inception_score : float
        Inception Score (higher is better, typical range :math:`[1, 10]`).
        If ``NaN`` or ``inf``, the IS contribution is set to 0.
    weights : dict of str -> float, optional
        Weights for each component.  Must contain keys ``"fid"``,
        ``"map"``, and ``"is"``.  If ``None``, defaults to
        ``{"fid": 0.4, "map": 0.4, "is": 0.2}``.

    Returns
    -------
    float
        The joint score.  Higher values indicate better overall performance.

    Notes
    -----
    - The joint score is **not bounded** but typically falls in
      :math:`[-1, 1]` for reasonable models.
    - A score of 0 indicates a baseline model (random or untrained).
    - Positive scores indicate better-than-baseline performance.
    - The weights should sum to 1.0 for interpretability, but this is
      not enforced.
    """
    if weights is None:
        weights = _DEFAULT_JOINT_WEIGHTS

    w_fid = weights.get("fid", 0.4)
    w_map = weights.get("map", 0.4)
    w_is = weights.get("is", 0.2)

    # Sanitise inputs: replace NaN/inf with 0 so the joint score remains
    # computable even when some metrics are unavailable.
    if math.isnan(fid_score) or math.isinf(fid_score):
        fid_score = 0.0
    if math.isnan(mAP_score) or math.isinf(mAP_score):
        mAP_score = 0.0
    if math.isnan(inception_score) or math.isinf(inception_score):
        inception_score = 0.0

    # Clamp to reasonable ranges to avoid extreme values
    fid_score = max(0.0, fid_score)
    mAP_score = max(0.0, min(1.0, mAP_score))
    inception_score = max(1.0, inception_score)

    # Normalised components
    fid_term = fid_score / _FID_NORMALISER
    is_term = inception_score / _IS_NORMALISER

    # Joint score
    joint = w_map * mAP_score + w_is * is_term - w_fid * fid_term

    return float(joint)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ──────────────────────────────────────────────────────────────────────────────


class MetricsTracker:
    """Stateful accumulator for all ILGAN evaluation metrics.

    The :class:`MetricsTracker` maintains running statistics across
    multiple batches and provides a single point of access for computing,
    logging, and resetting all metrics.  It is designed to be used in the
    evaluation loop of the ILGAN training pipeline.

    The tracker internally maintains three categories of metrics:

    - **Image metrics**: FID, Inception Score, image statistics (mean
      pixel, gradient magnitude, colour entropy, etc.).
    - **Box metrics**: mAP, GIoU, detection accuracy, box statistics
      (confidence, size, spatial diversity).
    - **Loss metrics**: generator loss, discriminator loss, box regression
      loss, consistency loss, collapse prevention loss, and any other
      scalar losses.

    All metrics are accumulated as running averages (mean across batches)
    unless otherwise noted.

    Parameters
    ----------
    num_classes : int, optional
        Number of object classes for mAP computation.  Required if box
        metrics will be computed.  Default is ``None`` (box metrics
        disabled).
    device : torch.device, optional
        Device for metric computation.  If ``None``, auto-detects GPU.
    joint_weights : dict of str -> float, optional
        Weights for the joint score.  Passed to
        :func:`compute_joint_score`.  If ``None``, uses defaults.

    Examples
    --------
    >>> tracker = MetricsTracker(num_classes=10)
    >>> tracker.update_image_metrics(real_imgs, fake_imgs)
    >>> tracker.update_box_metrics(pred_boxes, pred_scores, pred_labels,
    ...                            target_boxes, target_labels, valid_mask)
    >>> tracker.update_loss_metrics({"g_loss": 1.2, "d_loss": 0.8})
    >>> all_metrics = tracker.compute_all()
    >>> tracker.log_summary(epoch=5, logger=my_logger)
    >>> tracker.reset()
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        device: Optional[torch.device] = None,
        joint_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._num_classes = num_classes
        self._device = device
        self._joint_weights = joint_weights or _DEFAULT_JOINT_WEIGHTS.copy()

        # ── Image metric accumulators ─────────────────────────────────────
        self._fid_calculator = FIDCalculator(device=device)
        self._is_calculator = InceptionScoreCalculator(device=device)
        self._image_stats_batches: List[Dict[str, Any]] = []

        # ── Box metric accumulators ──────────────────────────────────────
        self._map_results: List[Dict[str, Any]] = []
        self._giou_results: List[Dict[str, float]] = []
        self._box_stat_results: List[Dict[str, float]] = []
        self._detection_acc_results: List[Dict[str, float]] = []

        # ── Loss metric accumulators ──────────────────────────────────────
        self._loss_batches: List[Dict[str, float]] = []

        # ── Timing ───────────────────────────────────────────────────────
        self._start_time: Optional[float] = None
        self._batch_count: int = 0

    # ── Public update methods ───────────────────────────────────────────

    def update_image_metrics(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> None:
        """Update image quality metrics with a batch of real and generated
        images.

        This method:

        1. Accumulates features for FID computation.
        2. Accumulates logits for Inception Score computation.
        3. Computes and stores image statistics (mean pixel, gradient
           magnitude, colour entropy, etc.).

        Parameters
        ----------
        real_images : torch.Tensor
            Real images, shape ``[B, C, H, W]``, values in ``[-1, 1]``.
        fake_images : torch.Tensor
            Generated (fake) images, same shape and range.

        Notes
        -----
        - All operations run under ``torch.no_grad()`` internally.
        - The Inception Score is accumulated across all calls to
          :meth:`update_image_metrics` and computed once in
          :meth:`compute_all`.
        - Image statistics are stored as a running list and averaged
          in :meth:`compute_all`.
        """
        if real_images.numel() == 0 or fake_images.numel() == 0:
            return

        # Accumulate FID features
        self._fid_calculator.update(real_images, fake_images)

        # Accumulate Inception Score logits (use fake images)
        self._is_calculator.update(fake_images)

        # Compute and store image statistics
        stats = compute_image_statistics(fake_images)
        # Filter to only scalar values (exclude list-valued entries like
        # mean_channel_mean and mean_channel_std)
        scalar_stats: Dict[str, float] = {
            k: v for k, v in stats.items()
            if isinstance(v, (int, float))
        }
        self._image_stats_batches.append(scalar_stats)

        self._batch_count += 1

    def update_box_metrics(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Update bounding box metrics with a batch of predictions and
        targets.

        This method computes and stores:

        1. mAP (mean Average Precision) — requires ``num_classes`` to be
           set at initialisation.
        2. GIoU (Generalised IoU) — element-wise between matched boxes.
        3. Box statistics — confidence, size, spatial diversity.
        4. Detection accuracy — fraction of high-confidence predictions
           that are correct.

        Parameters
        ----------
        pred_boxes : torch.Tensor
            Predicted bounding boxes, shape ``[B, N, 4]`` in
            ``(cx, cy, w, h)`` format, normalised to ``[0, 1]``.
        pred_scores : torch.Tensor
            Predicted confidence scores, shape ``[B, N]``.
        pred_labels : torch.Tensor
            Predicted class labels, shape ``[B, N]``.
        target_boxes : torch.Tensor
            Ground-truth bounding boxes, shape ``[B, N, 4]``.
        target_labels : torch.Tensor
            Ground-truth class labels, shape ``[B, N]``.
        valid_mask : torch.Tensor
            Boolean mask of shape ``[B, N]``.  ``True`` for valid entries.

        Notes
        -----
        - If ``num_classes`` was not set at initialisation, mAP is
          skipped (set to ``NaN`` in the results).
        - All computations are performed on CPU tensors to minimise GPU
          memory usage during evaluation.
        """
        if pred_boxes.numel() == 0 or not valid_mask.any():
            return

        # Move to CPU for metric computation to save GPU memory
        pred_boxes_cpu = pred_boxes.detach().cpu()
        pred_scores_cpu = pred_scores.detach().cpu()
        pred_labels_cpu = pred_labels.detach().cpu()
        target_boxes_cpu = target_boxes.detach().cpu()
        target_labels_cpu = target_labels.detach().cpu()
        valid_mask_cpu = valid_mask.detach().cpu()

        # ── mAP ──────────────────────────────────────────────────────────
        if self._num_classes is not None and self._num_classes > 0:
            map_result = compute_map(
                pred_boxes=pred_boxes_cpu,
                pred_scores=pred_scores_cpu,
                pred_labels=pred_labels_cpu,
                target_boxes=target_boxes_cpu,
                target_labels=target_labels_cpu,
                valid_mask=valid_mask_cpu,
                num_classes=self._num_classes,
            )
            self._map_results.append(map_result)

        # ── GIoU ─────────────────────────────────────────────────────────
        giou_result = compute_giou(
            pred_boxes=pred_boxes_cpu,
            target_boxes=target_boxes_cpu,
            valid_mask=valid_mask_cpu,
        )
        self._giou_results.append(giou_result)

        # ── Box statistics ───────────────────────────────────────────────
        box_stat_result = compute_box_statistics(
            pred_boxes=pred_boxes_cpu,
            confidences=pred_scores_cpu,
            valid_mask=valid_mask_cpu,
        )
        self._box_stat_results.append(box_stat_result)

        # ── Detection accuracy ──────────────────────────────────────────
        det_acc_result = compute_detection_accuracy(
            pred_boxes=pred_boxes_cpu,
            pred_labels=pred_labels_cpu,
            pred_confidences=pred_scores_cpu,
            target_boxes=target_boxes_cpu,
            target_labels=target_labels_cpu,
            valid_mask=valid_mask_cpu,
        )
        self._detection_acc_results.append(det_acc_result)

        self._batch_count += 1

    def update_loss_metrics(
        self,
        loss_dict: Dict[str, float],
    ) -> None:
        """Update loss metrics with a dictionary of scalar losses from a
        training step.

        Parameters
        ----------
        loss_dict : dict of str -> float
            Dictionary mapping loss names to scalar values.  Typical keys
            include:

            - ``"g_loss"``: generator adversarial loss.
            - ``"d_loss"``: discriminator adversarial loss.
            - ``"box_loss"``: bounding box regression loss.
            - ``"consistency_loss"``: image-box consistency loss.
            - ``"collapse_penalty"``: collapse prevention regularisation.
            - ``"total_loss"``: combined generator loss.

        Notes
        -----
        - Loss values are accumulated as a running mean across all calls
          to :meth:`update_loss_metrics`.
        - The tracker does **not** reset loss accumulators when
          :meth:`reset` is called — call :meth:`reset` explicitly at the
          start of each evaluation epoch.
        """
        self._loss_batches.append(loss_dict)
        self._batch_count += 1

    # ── Compute all metrics ──────────────────────────────────────────────

    def compute_all(self) -> Dict[str, Any]:
        """Compute all accumulated metrics and return a single dictionary.

        This method:

        1. Computes FID from accumulated real/fake features.
        2. Computes Inception Score from accumulated fake logits.
        3. Averages image statistics across batches.
        4. Averages mAP, GIoU, box statistics, and detection accuracy
           across batches.
        5. Averages loss metrics across batches.
        6. Computes the joint score from FID, mAP, and IS.

        Returns
        -------
        dict of str -> Any
            A flat dictionary with the following keys (grouped by prefix):

            **Image metrics** (prefix ``"image/"``):

            - ``"image/fid"``: FID score (float).  ``NaN`` if insufficient
              samples.
            - ``"image/inception_score"``: Inception Score mean (float).
            - ``"image/inception_score_std"``: Inception Score std (float).
            - ``"image/mean_pixel"``: Mean pixel value (float).
            - ``"image/std_pixel"``: Pixel std (float).
            - ``"image/mean_gradient_magnitude"``: Sharpness (float).
            - ``"image/color_histogram_entropy"``: Colour diversity (float).

            **Box metrics** (prefix ``"box/"``):

            - ``"box/mAP"``: mean Average Precision (float).
            - ``"box/mean_giou"``: mean GIoU (float).
            - ``"box/detection_accuracy"``: detection accuracy (float).
            - ``"box/recall"``: recall (float).
            - ``"box/mean_confidence"``: mean confidence (float).
            - ``"box/mean_box_size"``: mean box area (float).
            - ``"box/std_cx"``: spatial diversity in x (float).
            - ``"box/std_cy"``: spatial diversity in y (float).
            - ``"box/num_predictions"``: total predictions (int).
            - ``"box/num_targets"``: total targets (int).

            **Loss metrics** (prefix ``"loss/"``):

            - Each key from the loss dict, prefixed with ``"loss/"``.

            **Joint metric**:

            - ``"joint_score"``: the heuristic joint score (float).

            **Meta**:

            - ``"num_batches"``: number of batches accumulated (int).
        """
        results: Dict[str, Any] = {}
        results["num_batches"] = self._batch_count

        # ── Image metrics ─────────────────────────────────────────────────
        fid = self._fid_calculator.compute()
        is_mean, is_std = self._is_calculator.compute_accumulated()

        results["image/fid"] = fid
        results["image/inception_score"] = is_mean
        results["image/inception_score_std"] = is_std

        # Average image statistics
        if self._image_stats_batches:
            avg_stats = self._average_dicts(self._image_stats_batches)
            for key, value in avg_stats.items():
                results[f"image/{key}"] = value
        else:
            results["image/mean_pixel"] = float("nan")
            results["image/std_pixel"] = float("nan")
            results["image/mean_gradient_magnitude"] = float("nan")
            results["image/color_histogram_entropy"] = float("nan")

        # ── Box metrics ──────────────────────────────────────────────────
        # mAP
        if self._map_results:
            map_values = [
                r["mAP"] for r in self._map_results
                if not math.isnan(r["mAP"])
            ]
            if map_values:
                results["box/mAP"] = float(torch.tensor(map_values).mean().item())
            else:
                results["box/mAP"] = float("nan")

            total_preds = sum(r.get("num_predictions", 0) for r in self._map_results)
            total_targets = sum(r.get("num_targets", 0) for r in self._map_results)
            results["box/num_predictions"] = total_preds
            results["box/num_targets"] = total_targets
        else:
            results["box/mAP"] = float("nan")
            results["box/num_predictions"] = 0
            results["box/num_targets"] = 0

        # GIoU
        if self._giou_results:
            giou_values = [
                r["mean_giou"] for r in self._giou_results
                if not math.isnan(r["mean_giou"])
            ]
            if giou_values:
                results["box/mean_giou"] = float(torch.tensor(giou_values).mean().item())
            else:
                results["box/mean_giou"] = float("nan")
        else:
            results["box/mean_giou"] = float("nan")

        # Box statistics
        if self._box_stat_results:
            conf_values = [
                r["mean_confidence"] for r in self._box_stat_results
                if not math.isnan(r["mean_confidence"])
            ]
            size_values = [
                r["mean_box_size"] for r in self._box_stat_results
                if not math.isnan(r["mean_box_size"])
            ]
            std_cx_values = [
                r["std_cx"] for r in self._box_stat_results
                if not math.isnan(r["std_cx"])
            ]
            std_cy_values = [
                r["std_cy"] for r in self._box_stat_results
                if not math.isnan(r["std_cy"])
            ]

            results["box/mean_confidence"] = (
                float(torch.tensor(conf_values).mean().item()) if conf_values else float("nan")
            )
            results["box/mean_box_size"] = (
                float(torch.tensor(size_values).mean().item()) if size_values else float("nan")
            )
            results["box/std_cx"] = (
                float(torch.tensor(std_cx_values).mean().item()) if std_cx_values else float("nan")
            )
            results["box/std_cy"] = (
                float(torch.tensor(std_cy_values).mean().item()) if std_cy_values else float("nan")
            )
        else:
            results["box/mean_confidence"] = float("nan")
            results["box/mean_box_size"] = float("nan")
            results["box/std_cx"] = float("nan")
            results["box/std_cy"] = float("nan")

        # Detection accuracy
        if self._detection_acc_results:
            det_acc_values = [
                r["detection_accuracy"] for r in self._detection_acc_results
                if not math.isnan(r["detection_accuracy"])
            ]
            recall_values = [
                r["recall"] for r in self._detection_acc_results
                if not math.isnan(r["recall"])
            ]

            results["box/detection_accuracy"] = (
                float(torch.tensor(det_acc_values).mean().item()) if det_acc_values else float("nan")
            )
            results["box/recall"] = (
                float(torch.tensor(recall_values).mean().item()) if recall_values else float("nan")
            )
        else:
            results["box/detection_accuracy"] = float("nan")
            results["box/recall"] = float("nan")

        # ── Loss metrics ─────────────────────────────────────────────────
        if self._loss_batches:
            # Collect all unique keys across batches
            all_loss_keys: set = set()
            for batch in self._loss_batches:
                all_loss_keys.update(batch.keys())

            for key in sorted(all_loss_keys):
                values = [
                    batch[key] for batch in self._loss_batches
                    if key in batch and not (math.isnan(batch[key]) or math.isinf(batch[key]))
                ]
                if values:
                    results[f"loss/{key}"] = float(torch.tensor(values).mean().item())
                else:
                    results[f"loss/{key}"] = float("nan")
        else:
            # No loss metrics accumulated — this is fine for pure eval
            pass

        # ── Joint score ───────────────────────────────────────────────────
        map_for_joint = results.get("box/mAP", float("nan"))
        if math.isnan(map_for_joint):
            map_for_joint = 0.0

        is_for_joint = results.get("image/inception_score", float("nan"))
        if math.isnan(is_for_joint):
            is_for_joint = 1.0

        fid_for_joint = results.get("image/fid", float("nan"))
        if math.isnan(fid_for_joint):
            fid_for_joint = 100.0  # Penalise unknown FID

        results["joint_score"] = compute_joint_score(
            fid_score=fid_for_joint,
            mAP_score=map_for_joint,
            inception_score=is_for_joint,
            weights=self._joint_weights,
        )

        return results

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all accumulators for a new evaluation epoch.

        This clears:

        - FID and Inception Score accumulators.
        - All image statistics, box metrics, and loss metric buffers.
        - The batch counter.

        Call this at the start of each evaluation epoch to ensure clean
        accumulation.
        """
        self._fid_calculator.reset()
        self._is_calculator.reset()
        self._image_stats_batches.clear()
        self._map_results.clear()
        self._giou_results.clear()
        self._box_stat_results.clear()
        self._detection_acc_results.clear()
        self._loss_batches.clear()
        self._batch_count = 0
        self._start_time = None

    # ── Logging ─────────────────────────────────────────────────────────

    def log_summary(
        self,
        epoch: int,
        logger: Logger,
        phase: str = "Eval",
    ) -> None:
        """Format and log a summary of all current metrics.

        This method computes all metrics via :meth:`compute_all` and logs
        a formatted summary string using the provided logger.

        Parameters
        ----------
        epoch : int
            Current epoch number (for display).
        logger : Logger
            An instance of :class:`ilgan.utils.logger.Logger` for
            output.
        phase : str, optional
            Phase description (e.g., ``"Eval"``, ``"Train"``, ``"Val"``).
            Default is ``"Eval"``.

        Notes
        -----
        - The summary is logged at INFO level.
        - Metrics are formatted using :func:`format_metrics`.
        - The joint score is highlighted separately.
        """
        metrics = self.compute_all()
        formatted = format_metrics(metrics)

        # Build summary string
        lines: List[str] = []
        lines.append(f"{'=' * 72}")
        lines.append(f"  {phase} — Epoch {epoch:>4d}  |  "
                      f"Batches: {metrics.get('num_batches', 0):>5d}")
        lines.append(f"{'=' * 72}")

        # Image metrics section
        lines.append("  ┌─ Image Metrics")
        lines.append(f"  │  FID:              {formatted.get('image/fid', 'N/A'):>12s}")
        lines.append(f"  │  Inception Score:  {formatted.get('image/inception_score', 'N/A'):>12s}  "
                      f"(std: {formatted.get('image/inception_score_std', 'N/A'):>12s})")
        lines.append(f"  │  Mean Pixel:       {formatted.get('image/mean_pixel', 'N/A'):>12s}")
        lines.append(f"  │  Gradient Mag:     {formatted.get('image/mean_gradient_magnitude', 'N/A'):>12s}")
        lines.append(f"  │  Colour Entropy:   {formatted.get('image/color_histogram_entropy', 'N/A'):>12s}")

        # Box metrics section
        lines.append("  ├─ Box Metrics")
        lines.append(f"  │  mAP:              {formatted.get('box/mAP', 'N/A'):>12s}")
        lines.append(f"  │  Mean GIoU:        {formatted.get('box/mean_giou', 'N/A'):>12s}")
        lines.append(f"  │  Detection Acc:    {formatted.get('box/detection_accuracy', 'N/A'):>12s}")
        lines.append(f"  │  Recall:           {formatted.get('box/recall', 'N/A'):>12s}")
        lines.append(f"  │  Mean Confidence:  {formatted.get('box/mean_confidence', 'N/A'):>12s}")
        lines.append(f"  │  Box Size:         {formatted.get('box/mean_box_size', 'N/A'):>12s}")
        lines.append(f"  │  Spatial Std (cx): {formatted.get('box/std_cx', 'N/A'):>12s}")
        lines.append(f"  │  Spatial Std (cy): {formatted.get('box/std_cy', 'N/A'):>12s}")

        # Loss metrics section
        loss_keys = [k for k in sorted(metrics.keys()) if k.startswith("loss/")]
        if loss_keys:
            lines.append("  ├─ Loss Metrics")
            for key in loss_keys:
                short_key = key.replace("loss/", "")
                lines.append(f"  │  {short_key:20s}: {formatted.get(key, 'N/A'):>12s}")

        # Joint score
        lines.append("  └─ Joint Score")
        joint_score = metrics.get("joint_score", float("nan"))
        if not math.isnan(joint_score):
            lines.append(f"     Joint Score:     {joint_score:>+12.6f}")
        else:
            lines.append(f"     Joint Score:     {'N/A':>12s}")

        lines.append(f"{'=' * 72}")

        logger.info("\n".join(lines))

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def num_batches(self) -> int:
        """Number of batches accumulated since last reset."""
        return self._batch_count

    @property
    def num_classes(self) -> Optional[int]:
        """Number of classes for mAP computation."""
        return self._num_classes

    @num_classes.setter
    def num_classes(self, n: int) -> None:
        """Set the number of classes for mAP computation."""
        if n < 1:
            raise ValueError(f"num_classes must be >= 1, got {n}.")
        self._num_classes = n

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _average_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average a list of dictionaries with the same keys.

        Only scalar (int/float) values are averaged; non-scalar values
        (e.g., lists) are silently skipped.

        Parameters
        ----------
        dicts : list of dict
            List of dictionaries with numeric values.

        Returns
        -------
        dict of str -> float
            Dictionary with the mean of each key across all input dicts.
            Returns ``float('nan')`` for keys that have no valid values.
        """
        if not dicts:
            return {}

        # Collect all keys
        all_keys: set = set()
        for d in dicts:
            all_keys.update(d.keys())

        result: Dict[str, float] = {}
        for key in all_keys:
            # Collect only scalar numeric values (skip lists, tensors, etc.)
            scalar_values = []
            for d in dicts:
                if key not in d:
                    continue
                val = d[key]
                if isinstance(val, (int, float)) and not (math.isnan(val) or math.isinf(val)):
                    scalar_values.append(val)
            if scalar_values:
                result[key] = float(torch.tensor(scalar_values).mean().item())
            else:
                result[key] = float("nan")

        return result


# ──────────────────────────────────────────────────────────────────────────────
# Formatting utility
# ──────────────────────────────────────────────────────────────────────────────


def format_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, str]:
    """Format a metrics dictionary for console logging with appropriate
    per-metric precision.

    The formatting rules are:

    - **FID** (``image/fid``): 2 decimal places.
    - **mAP** (``box/mAP``): 4 decimal places.
    - **GIoU** (``box/mean_giou``): 4 decimal places.
    - **Detection accuracy / Recall** (``box/detection_accuracy``,
      ``box/recall``): 4 decimal places.
    - **Confidence / Box size / Spatial std** (``box/mean_confidence``,
      ``box/mean_box_size``, ``box/std_cx``, ``box/std_cy``): 4 decimal
      places.
    - **Inception Score** (``image/inception_score``,
      ``image/inception_score_std``): 2 decimal places.
    - **Image statistics** (``image/mean_pixel``, ``image/std_pixel``,
      ``image/mean_gradient_magnitude``, ``image/color_histogram_entropy``):
      4 decimal places.
    - **Losses** (``loss/*``): 6 decimal places.
    - **Joint score** (``joint_score``): 6 decimal places.
    - **Integers** (``box/num_predictions``, ``box/num_targets``,
      ``num_batches``): formatted as integers.
    - **NaN** values are formatted as ``"N/A"``.

    Parameters
    ----------
    metrics_dict : dict of str -> Any
        A flat dictionary of metric names to values (as produced by
        :meth:`MetricsTracker.compute_all`).

    Returns
    -------
    dict of str -> str
        A dictionary with the same keys, where each value is a formatted
        string with the appropriate precision.

    Examples
    --------
    >>> metrics = {"image/fid": 12.3456, "box/mAP": 0.8765, "loss/g_loss": 1.234567}
    >>> formatted = format_metrics(metrics)
    >>> formatted["image/fid"]
    '12.35'
    >>> formatted["box/mAP"]
    '0.8765'
    >>> formatted["loss/g_loss"]
    '1.234567'
    """
    formatted: Dict[str, str] = {}

    for key, value in metrics_dict.items():
        # Handle NaN / None
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            formatted[key] = "N/A"
            continue

        # Integer types
        if isinstance(value, (int,)) or key in (
            "box/num_predictions",
            "box/num_targets",
            "num_batches",
        ):
            formatted[key] = f"{int(value):d}"
            continue

        # Float formatting by key pattern
        if not isinstance(value, float):
            # Unknown type — convert to string
            formatted[key] = str(value)
            continue

        if key == "image/fid":
            formatted[key] = f"{value:.2f}"
        elif key in ("image/inception_score", "image/inception_score_std"):
            formatted[key] = f"{value:.2f}"
        elif key.startswith("box/"):
            if key in ("box/num_predictions", "box/num_targets"):
                formatted[key] = f"{int(value):d}"
            else:
                formatted[key] = f"{value:.4f}"
        elif key.startswith("loss/"):
            formatted[key] = f"{value:.6f}"
        elif key == "joint_score":
            formatted[key] = f"{value:.6f}"
        elif key.startswith("image/"):
            formatted[key] = f"{value:.4f}"
        else:
            # Fallback: 4 decimal places
            formatted[key] = f"{value:.4f}"

    return formatted


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "compute_joint_score",
    "MetricsTracker",
    "format_metrics",
]
