"""
Bounding box quality assessment metrics for ILGAN.

Provides four core functions for evaluating the quality of predicted bounding
boxes from the ILGAN dual-output generator:

1. :func:`compute_map` — mean Average Precision (mAP) for object detection,
   computed per-class and averaged across all classes.

2. :func:`compute_giou` — mean Generalized IoU (GIoU) between predicted and
   target boxes, averaged over valid (non-padded) entries.

3. :func:`compute_box_statistics` — Statistics about the predicted boxes:
   mean confidence, mean box size, spatial diversity (std of box centres),
   and the number of high-confidence boxes.

4. :func:`compute_detection_accuracy` — Percentage of predicted boxes that
   correctly localise and classify a ground-truth object.

All functions operate on padded batch tensors (shape ``[B, N, ...]``) and
accept a ``valid_mask`` (``[B, N]`` bool) to exclude padding entries.  They
are numerically stable and handle edge cases such as empty predictions,
missing targets, or single-class datasets.

Coordinate conventions
----------------------
All box coordinates are in the normalised ``(cx, cy, w, h)`` format used
throughout ILGAN, where each value lies in ``[0, 1]`` relative to the image
dimensions.  Internally, functions convert to ``(x1, y1, x2, y2)`` format
for IoU/GIoU computation.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversion
# ──────────────────────────────────────────────────────────────────────────────


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    r"""Convert boxes from ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)`` format.

    Parameters
    ----------
    boxes : torch.Tensor
        Tensor of shape ``[..., 4]`` with the last dimension being
        ``(cx, cy, w, h)``.  All values are assumed to be in ``[0, 1]``
        (normalised relative to image dimensions).

    Returns
    -------
    torch.Tensor
        Tensor of the same shape with the last dimension being
        ``(x1, y1, x2, y2)``, where ``x1 = cx - w/2``, etc.
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# IoU computation (core building block)
# ──────────────────────────────────────────────────────────────────────────────


def _compute_iou(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
) -> torch.Tensor:
    r"""Compute IoU between two sets of boxes in ``(x1, y1, x2, y2)`` format.

    Parameters
    ----------
    pred_xyxy : torch.Tensor
        Predicted boxes, shape ``[N_pred, 4]``.
    target_xyxy : torch.Tensor
        Target boxes, shape ``[N_target, 4]``.

    Returns
    -------
    torch.Tensor
        IoU matrix of shape ``[N_pred, N_target]``, where element ``[i, j]``
        is the IoU between ``pred_xyxy[i]`` and ``target_xyxy[j]``.
    """
    # Expand dimensions for broadcasting: [N_pred, 1, 4] vs [1, N_target, 4]
    pred = pred_xyxy.unsqueeze(1)   # [N_pred, 1, 4]
    target = target_xyxy.unsqueeze(0)  # [1, N_target, 4]

    # Intersection coordinates
    inter_x1 = torch.max(pred[..., 0], target[..., 0])  # [N_pred, N_target]
    inter_y1 = torch.max(pred[..., 1], target[..., 1])  # [N_pred, N_target]
    inter_x2 = torch.min(pred[..., 2], target[..., 2])  # [N_pred, N_target]
    inter_y2 = torch.min(pred[..., 3], target[..., 3])  # [N_pred, N_target]

    # Intersection area (clamp to 0 for non-overlapping)
    inter_h = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_w = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_h * inter_w  # [N_pred, N_target]

    # Areas of each box
    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])  # [N_pred, 1]
    target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])  # [1, N_target]

    # Union area
    union_area = pred_area + target_area - inter_area  # [N_pred, N_target]

    # IoU with epsilon for numerical stability
    eps = 1e-7
    iou = inter_area / (union_area + eps)

    return iou


def _compute_giou_pair(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
) -> torch.Tensor:
    r"""Compute GIoU between two sets of boxes in ``(x1, y1, x2, y2)`` format.

    Parameters
    ----------
    pred_xyxy : torch.Tensor
        Predicted boxes, shape ``[N_pred, 4]``.
    target_xyxy : torch.Tensor
        Target boxes, shape ``[N_target, 4]``.

    Returns
    -------
    torch.Tensor
        GIoU matrix of shape ``[N_pred, N_target]``.
    """
    pred = pred_xyxy.unsqueeze(1)   # [N_pred, 1, 4]
    target = target_xyxy.unsqueeze(0)  # [1, N_target, 4]

    # Intersection
    inter_x1 = torch.max(pred[..., 0], target[..., 0])
    inter_y1 = torch.max(pred[..., 1], target[..., 1])
    inter_x2 = torch.min(pred[..., 2], target[..., 2])
    inter_y2 = torch.min(pred[..., 3], target[..., 3])

    inter_h = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_w = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_h * inter_w

    pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])

    union_area = pred_area + target_area - inter_area
    eps = 1e-7
    iou = inter_area / (union_area + eps)

    # Smallest enclosing box
    enclose_x1 = torch.min(pred[..., 0], target[..., 0])
    enclose_y1 = torch.min(pred[..., 1], target[..., 1])
    enclose_x2 = torch.max(pred[..., 2], target[..., 2])
    enclose_y2 = torch.max(pred[..., 3], target[..., 3])

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    enclose_area = enclose_area.clamp(min=eps)

    # GIoU = IoU - (enclosing_area - union_area) / enclosing_area
    giou = iou - (enclose_area - union_area) / enclose_area

    return giou


# ──────────────────────────────────────────────────────────────────────────────
# mAP Computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_map(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    r"""Compute mean Average Precision (mAP) for object detection.

    The mAP is computed by:

    1. For each class, collecting all predicted boxes (with their confidence
       scores) and all ground-truth boxes.
    2. Sorting predictions by descending confidence.
    3. Computing the precision-recall curve and the Average Precision (AP)
       using the 101-point interpolation method (as in the Pascal VOC
       challenge).
    4. Averaging AP across all classes.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format, normalised to ``[0, 1]``.
    pred_scores : torch.Tensor
        Predicted confidence (objectness) scores, shape ``[B, N]``.  Values
        in ``(0, 1)``.
    pred_labels : torch.Tensor
        Predicted class labels, shape ``[B, N]``.  Integer class IDs in
        ``[0, num_classes)``.
    target_boxes : torch.Tensor
        Ground-truth bounding boxes, shape ``[B, N, 4]`` in
        ``(cx, cy, w, h)`` format.
    target_labels : torch.Tensor
        Ground-truth class labels, shape ``[B, N]``.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` for valid (non-padded)
        entries in both predictions and targets.
    num_classes : int
        Total number of classes.  Must be >= 1.
    iou_threshold : float, optional
        IoU threshold for considering a prediction as a true positive.
        (default: ``0.5``)

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"mAP"``: mean Average Precision across all classes (float).
        - ``"AP_per_class"``: list of AP values per class (length
          ``num_classes``).  Classes with no ground-truth boxes have
          ``AP = float('nan')``.
        - ``"num_predictions"``: total number of valid predictions.
        - ``"num_targets"``: total number of valid ground-truth boxes.

    Notes
    -----
    - If there are no valid predictions, mAP is 0.0 (no true positives).
    - If a class has no ground-truth boxes, its AP is ``float('nan')`` and
      it is excluded from the mAP average.
    - The 101-point interpolation evaluates precision at 101 equally-spaced
      recall levels: ``[0.00, 0.01, ..., 1.00]``.

    Raises
    ------
    ValueError
        If ``num_classes < 1`` or ``iou_threshold`` is not in ``(0, 1]``.

    Shape
    -----
    - ``pred_boxes``: ``[B, N, 4]``
    - ``pred_scores``: ``[B, N]``
    - ``pred_labels``: ``[B, N]``
    - ``target_boxes``: ``[B, N, 4]``
    - ``target_labels``: ``[B, N]``
    - ``valid_mask``: ``[B, N]``
    """
    # ── Input validation ──────────────────────────────────────────────────
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(
            f"iou_threshold must be in (0, 1], got {iou_threshold}."
        )

    # ── Flatten batch dimension ────────────────────────────────────────────
    # Convert all boxes to (x1, y1, x2, y2) format
    pred_xyxy = _cxcywh_to_xyxy(pred_boxes)        # [B, N, 4]
    target_xyxy = _cxcywh_to_xyxy(target_boxes)    # [B, N, 4]

    # Clamp to [0, 1] for numerical safety
    pred_xyxy = pred_xyxy.clamp(min=0.0, max=1.0)
    target_xyxy = target_xyxy.clamp(min=0.0, max=1.0)

    # Gather all valid predictions and targets across the batch
    valid_pred_mask = valid_mask & (pred_scores > 0.0)
    valid_target_mask = valid_mask

    # ── Collect predictions ────────────────────────────────────────────────
    pred_list: List[Tuple[float, int, torch.Tensor]] = []  # (score, label, box_xyxy)
    for b in range(pred_boxes.shape[0]):
        for n in range(pred_boxes.shape[1]):
            if valid_pred_mask[b, n]:
                pred_list.append((
                    pred_scores[b, n].item(),
                    int(pred_labels[b, n].item()),
                    pred_xyxy[b, n].clone(),
                ))

    # ── Collect targets ────────────────────────────────────────────────────
    target_list: List[Tuple[int, torch.Tensor]] = []  # (label, box_xyxy)
    for b in range(target_boxes.shape[0]):
        for n in range(target_boxes.shape[1]):
            if valid_target_mask[b, n]:
                target_list.append((
                    int(target_labels[b, n].item()),
                    target_xyxy[b, n].clone(),
                ))

    num_predictions = len(pred_list)
    num_targets = len(target_list)

    # ── Edge case: no predictions ──────────────────────────────────────────
    if num_predictions == 0:
        ap_per_class: List[float] = []
        for c in range(num_classes):
            num_targets_c = sum(1 for t_label, _ in target_list if t_label == c)
            ap_per_class.append(float("nan") if num_targets_c == 0 else 0.0)

        # mAP excludes NaN classes
        valid_ap = [v for v in ap_per_class if not math.isnan(v)]
        map_val = float(torch.tensor(valid_ap).mean().item()) if valid_ap else 0.0

        return {
            "mAP": map_val,
            "AP_per_class": ap_per_class,
            "num_predictions": num_predictions,
            "num_targets": num_targets,
        }

    # ── Sort predictions by descending confidence ───────────────────────────
    pred_list.sort(key=lambda x: x[0], reverse=True)

    # ── Group targets by class ─────────────────────────────────────────────
    targets_by_class: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
    for t_label, t_box in target_list:
        targets_by_class[t_label].append(t_box)

    # ── Compute AP per class ───────────────────────────────────────────────
    ap_per_class = []
    for c in range(num_classes):
        # Get predictions for this class
        class_preds = [(score, box) for score, label, box in pred_list if label == c]
        class_targets = targets_by_class[c]

        num_class_targets = len(class_targets)

        # If no ground-truth boxes for this class, AP is undefined
        if num_class_targets == 0:
            ap_per_class.append(float("nan"))
            continue

        # If no predictions for this class, AP is 0.0
        if len(class_preds) == 0:
            ap_per_class.append(0.0)
            continue

        # ── Match predictions to targets ──────────────────────────────────
        # Track which targets have been matched (one prediction per target)
        target_matched = [False] * num_class_targets

        # Stack target boxes for IoU computation
        target_boxes_c = torch.stack(class_targets, dim=0)  # [N_t, 4]

        tp = []  # 1 for true positive, 0 for false positive
        fp = []

        for score, pred_box in class_preds:
            # Compute IoU between this prediction and all targets of this class
            pred_box_expanded = pred_box.unsqueeze(0)  # [1, 4]
            ious = _compute_iou(pred_box_expanded, target_boxes_c)  # [1, N_t]
            ious = ious.squeeze(0)  # [N_t]

            # Find the best matching target
            best_iou, best_idx = ious.max(dim=0)

            if best_iou.item() >= iou_threshold and not target_matched[best_idx.item()]:
                # True positive: matches a target and that target hasn't been matched yet
                tp.append(1.0)
                fp.append(0.0)
                target_matched[best_idx.item()] = True
            else:
                # False positive: either low IoU or target already matched
                tp.append(0.0)
                fp.append(1.0)

        # ── Compute precision-recall curve ────────────────────────────────
        tp_tensor = torch.tensor(tp, dtype=torch.float64)
        fp_tensor = torch.tensor(fp, dtype=torch.float64)

        # Cumulative sums
        tp_cum = torch.cumsum(tp_tensor, dim=0)
        fp_cum = torch.cumsum(fp_tensor, dim=0)

        # Precision and recall at each rank
        recall = tp_cum / max(num_class_targets, 1)
        precision = tp_cum / (tp_cum + fp_cum + 1e-10)

        # ── 101-point interpolation ──────────────────────────────────────
        # Append sentinel values for interpolation
        recall = torch.cat([torch.tensor([0.0], dtype=torch.float64), recall])
        precision = torch.cat([torch.tensor([1.0], dtype=torch.float64), precision])

        # For each recall level r in [0.00, 0.01, ..., 1.00], take the
        # maximum precision for any recall >= r (this is the "all-points"
        # interpolation used in Pascal VOC 2010+).
        ap = 0.0
        for r in torch.linspace(0.0, 1.0, 101, dtype=torch.float64):
            # Find indices where recall >= r
            mask = recall >= r
            if mask.any():
                p = precision[mask].max().item()
            else:
                p = 0.0
            ap += p

        ap /= 101.0
        ap_per_class.append(ap)

    # ── Compute mAP (average over classes with valid AP) ──────────────────
    valid_ap = [v for v in ap_per_class if not math.isnan(v)]
    if valid_ap:
        map_val = float(torch.tensor(valid_ap).mean().item())
    else:
        map_val = 0.0

    return {
        "mAP": map_val,
        "AP_per_class": ap_per_class,
        "num_predictions": num_predictions,
        "num_targets": num_targets,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mean GIoU
# ──────────────────────────────────────────────────────────────────────────────


def compute_giou(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    r"""Compute the mean GIoU between predicted and target boxes.

    For each valid (non-padded) box pair, the GIoU is computed.  The mean
    GIoU across all valid pairs is returned, along with the minimum and
    maximum GIoU for diagnostic purposes.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format.
    target_boxes : torch.Tensor
        Ground-truth bounding boxes, same shape and format.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` for valid entries.

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"mean_giou"``: mean GIoU across all valid box pairs (float).
          Returns ``float('nan')`` if no valid boxes exist.
        - ``"min_giou"``: minimum GIoU (float).
        - ``"max_giou"``: maximum GIoU (float).
        - ``"num_valid"``: number of valid box pairs (int).

    Notes
    -----
    - GIoU is bounded in ``[-1, 1]``.  A GIoU of 1 indicates perfect
      overlap, while -1 indicates complete non-overlap.
    - This function is designed for **logging during training**, where
      predicted boxes are matched to target boxes by index (i.e., the
      ``i``-th predicted box is compared to the ``i``-th target box).
      For evaluation metrics like mAP, use :func:`compute_map` instead.

    Shape
    -----
    - ``pred_boxes``: ``[B, N, 4]``
    - ``target_boxes``: ``[B, N, 4]``
    - ``valid_mask``: ``[B, N]``
    """
    # ── Input validation ──────────────────────────────────────────────────
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} must match "
            f"target_boxes shape {target_boxes.shape}."
        )
    if pred_boxes.shape[:-1] != valid_mask.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and valid_mask shape "
            f"{valid_mask.shape} are incompatible."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")

    # ── Early exit if no valid boxes ─────────────────────────────────────
    if not valid_mask.any():
        return {
            "mean_giou": float("nan"),
            "min_giou": float("nan"),
            "max_giou": float("nan"),
            "num_valid": 0,
        }

    # ── Convert to (x1, y1, x2, y2) ──────────────────────────────────────
    pred_xyxy = _cxcywh_to_xyxy(pred_boxes).clamp(min=0.0, max=1.0)
    target_xyxy = _cxcywh_to_xyxy(target_boxes).clamp(min=0.0, max=1.0)

    # ── Compute GIoU element-wise ─────────────────────────────────────────
    # Intersection
    inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])

    inter_h = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_w = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_h * inter_w

    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * \
                (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]) * \
                  (target_xyxy[..., 3] - target_xyxy[..., 1])

    union_area = pred_area + target_area - inter_area
    eps = 1e-7
    iou = inter_area / (union_area + eps)

    # Smallest enclosing box
    enclose_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
    enclose_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
    enclose_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
    enclose_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    enclose_area = enclose_area.clamp(min=eps)

    giou = iou - (enclose_area - union_area) / enclose_area  # [B, N]

    # ── Mask and compute statistics ───────────────────────────────────────
    giou_valid = giou[valid_mask]  # [num_valid]

    if giou_valid.numel() == 0:
        return {
            "mean_giou": float("nan"),
            "min_giou": float("nan"),
            "max_giou": float("nan"),
            "num_valid": 0,
        }

    return {
        "mean_giou": float(giou_valid.mean().item()),
        "min_giou": float(giou_valid.min().item()),
        "max_giou": float(giou_valid.max().item()),
        "num_valid": int(giou_valid.numel()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Box Statistics
# ──────────────────────────────────────────────────────────────────────────────


def compute_box_statistics(
    pred_boxes: torch.Tensor,
    confidences: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    r"""Compute statistics about the predicted bounding boxes.

    This function is designed for **monitoring training dynamics**,
    particularly to detect bounding box collapse (where all predicted boxes
    converge to the same location).  It computes:

    - **Mean confidence**: average objectness score across valid predictions.
    - **Mean box size**: average area (width × height) of valid boxes.
    - **Spatial diversity**: standard deviation of box centre coordinates
      ``(cx, cy)`` across valid predictions.  A low standard deviation
      (e.g., < 0.01) indicates that boxes are collapsing to a single
      location.
    - **High-confidence count**: number of valid boxes with confidence > 0.5.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format.
    confidences : torch.Tensor
        Predicted confidence (objectness) scores, shape ``[B, N]``.  Values
        in ``(0, 1)``.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` for valid entries.

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"mean_confidence"``: mean confidence score (float).  Returns
          ``float('nan')`` if no valid boxes.
        - ``"mean_box_size"``: mean box area (width × height) (float).
          Returns ``float('nan')`` if no valid boxes.
        - ``"std_cx"``: standard deviation of box centre x-coordinates
          (float).  Returns ``float('nan')`` if fewer than 2 valid boxes.
        - ``"std_cy"``: standard deviation of box centre y-coordinates
          (float).  Returns ``float('nan')`` if fewer than 2 valid boxes.
        - ``"mean_cx"``: mean of box centre x-coordinates (float).
        - ``"mean_cy"``: mean of box centre y-coordinates (float).
        - ``"mean_box_width"``: mean box width (float).
        - ``"mean_box_height"``: mean box height (float).
        - ``"num_high_confidence"``: number of boxes with confidence > 0.5
          (int).
        - ``"num_valid"``: number of valid boxes (int).

    Shape
    -----
    - ``pred_boxes``: ``[B, N, 4]``
    - ``confidences``: ``[B, N]``
    - ``valid_mask``: ``[B, N]``
    """
    # ── Input validation ──────────────────────────────────────────────────
    if pred_boxes.shape[:-1] != confidences.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and confidences shape "
            f"{confidences.shape} are incompatible."
        )
    if pred_boxes.shape[:-1] != valid_mask.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and valid_mask shape "
            f"{valid_mask.shape} are incompatible."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")

    # ── Early exit if no valid boxes ─────────────────────────────────────
    if not valid_mask.any():
        return {
            "mean_confidence": float("nan"),
            "mean_box_size": float("nan"),
            "std_cx": float("nan"),
            "std_cy": float("nan"),
            "mean_cx": float("nan"),
            "mean_cy": float("nan"),
            "mean_box_width": float("nan"),
            "mean_box_height": float("nan"),
            "num_high_confidence": 0,
            "num_valid": 0,
        }

    # ── Extract valid entries ────────────────────────────────────────────
    valid_boxes = pred_boxes[valid_mask]       # [V, 4]
    valid_conf = confidences[valid_mask]        # [V]

    # ── Centre coordinates ────────────────────────────────────────────────
    cx = valid_boxes[:, 0]  # [V]
    cy = valid_boxes[:, 1]  # [V]
    w = valid_boxes[:, 2]   # [V]
    h = valid_boxes[:, 3]   # [V]

    # ── Box size (area) ──────────────────────────────────────────────────
    box_area = w * h  # [V]

    # ── Statistics ────────────────────────────────────────────────────────
    mean_confidence = float(valid_conf.mean().item())
    mean_box_size = float(box_area.mean().item())
    mean_cx = float(cx.mean().item())
    mean_cy = float(cy.mean().item())
    mean_box_width = float(w.mean().item())
    mean_box_height = float(h.mean().item())

    # Standard deviation of centres (spatial diversity)
    if valid_boxes.shape[0] >= 2:
        std_cx = float(cx.std().item())
        std_cy = float(cy.std().item())
    else:
        std_cx = float("nan")
        std_cy = float("nan")

    # High-confidence count
    num_high_confidence = int((valid_conf > 0.5).sum().item())
    num_valid = int(valid_boxes.shape[0])

    return {
        "mean_confidence": mean_confidence,
        "mean_box_size": mean_box_size,
        "std_cx": std_cx,
        "std_cy": std_cy,
        "mean_cx": mean_cx,
        "mean_cy": mean_cy,
        "mean_box_width": mean_box_width,
        "mean_box_height": mean_box_height,
        "num_high_confidence": num_high_confidence,
        "num_valid": num_valid,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Detection Accuracy
# ──────────────────────────────────────────────────────────────────────────────


def compute_detection_accuracy(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_confidences: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    valid_mask: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    r"""Compute the percentage of predicted boxes that correctly detect an
    object.

    A prediction is considered a **correct detection** if:

    1. Its confidence score is > 0.5 (the standard detection threshold).
    2. It has an IoU >= ``iou_threshold`` with some ground-truth box.
    3. The predicted class label matches the ground-truth class label.
    4. That ground-truth box has not already been matched to another
       prediction (one-to-one matching).

    The detection accuracy is defined as:

    .. math::

        \text{DetAcc} = \frac{N_{correct}}{N_{pred}}

    where :math:`N_{correct}` is the number of predictions satisfying all
    four conditions, and :math:`N_{pred}` is the total number of valid
    predictions with confidence > 0.5.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format.
    pred_labels : torch.Tensor
        Predicted class labels, shape ``[B, N]``.
    pred_confidences : torch.Tensor
        Predicted confidence scores, shape ``[B, N]``.
    target_boxes : torch.Tensor
        Ground-truth bounding boxes, shape ``[B, N, 4]``.
    target_labels : torch.Tensor
        Ground-truth class labels, shape ``[B, N]``.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` for valid entries.
    iou_threshold : float, optional
        IoU threshold for considering a prediction as localised correctly.
        (default: ``0.5``)

    Returns
    -------
    dict of str -> float
        A dictionary with the following keys:

        - ``"detection_accuracy"``: the fraction of high-confidence
          predictions that are correct (float).  Returns ``float('nan')``
          if no predictions have confidence > 0.5.
        - ``"num_correct"``: number of correct detections (int).
        - ``"num_predictions"``: number of predictions with confidence > 0.5
          (int).
        - ``"num_targets"``: total number of valid ground-truth boxes (int).
        - ``"recall"``: fraction of ground-truth boxes that were matched
          by at least one prediction (float).  Returns ``float('nan')`` if
          no targets exist.

    Notes
    -----
    - This function uses **greedy one-to-one matching**: each ground-truth
      box can be matched to at most one prediction.  This prevents multiple
      predictions from claiming the same target.
    - Predictions are processed in order of descending confidence (greedy
      assignment).

    Shape
    -----
    - ``pred_boxes``: ``[B, N, 4]``
    - ``pred_labels``: ``[B, N]``
    - ``pred_confidences``: ``[B, N]``
    - ``target_boxes``: ``[B, N, 4]``
    - ``target_labels``: ``[B, N]``
    - ``valid_mask``: ``[B, N]``
    """
    # ── Input validation ──────────────────────────────────────────────────
    if pred_boxes.shape != target_boxes.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} must match "
            f"target_boxes shape {target_boxes.shape}."
        )
    if pred_boxes.shape[:-1] != pred_labels.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and pred_labels shape "
            f"{pred_labels.shape} are incompatible."
        )
    if pred_boxes.shape[:-1] != pred_confidences.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and pred_confidences shape "
            f"{pred_confidences.shape} are incompatible."
        )
    if pred_boxes.shape[:-1] != target_labels.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and target_labels shape "
            f"{target_labels.shape} are incompatible."
        )
    if pred_boxes.shape[:-1] != valid_mask.shape:
        raise ValueError(
            f"pred_boxes shape {pred_boxes.shape} and valid_mask shape "
            f"{valid_mask.shape} are incompatible."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(
            f"iou_threshold must be in (0, 1], got {iou_threshold}."
        )

    # ── Convert to (x1, y1, x2, y2) ──────────────────────────────────────
    pred_xyxy = _cxcywh_to_xyxy(pred_boxes).clamp(min=0.0, max=1.0)
    target_xyxy = _cxcywh_to_xyxy(target_boxes).clamp(min=0.0, max=1.0)

    # ── Flatten batch dimension ────────────────────────────────────────────
    B, N = valid_mask.shape

    # Collect all predictions with confidence > 0.5
    pred_entries: List[Tuple[float, int, torch.Tensor, int, int]] = []
    # (confidence, label, box_xyxy, batch_idx, box_idx)
    for b in range(B):
        for n in range(N):
            if valid_mask[b, n] and pred_confidences[b, n] > 0.5:
                pred_entries.append((
                    pred_confidences[b, n].item(),
                    int(pred_labels[b, n].item()),
                    pred_xyxy[b, n].clone(),
                    b,
                    n,
                ))

    # Collect all targets
    target_entries: List[Tuple[int, torch.Tensor, int, int]] = []
    # (label, box_xyxy, batch_idx, box_idx)
    for b in range(B):
        for n in range(N):
            if valid_mask[b, n]:
                target_entries.append((
                    int(target_labels[b, n].item()),
                    target_xyxy[b, n].clone(),
                    b,
                    n,
                ))

    num_predictions = len(pred_entries)
    num_targets = len(target_entries)

    # ── Edge case: no predictions with confidence > 0.5 ───────────────────
    if num_predictions == 0:
        recall = float("nan") if num_targets == 0 else 0.0
        return {
            "detection_accuracy": float("nan"),
            "num_correct": 0,
            "num_predictions": 0,
            "num_targets": num_targets,
            "recall": recall,
        }

    # ── Edge case: no targets ──────────────────────────────────────────────
    if num_targets == 0:
        return {
            "detection_accuracy": 0.0,
            "num_correct": 0,
            "num_predictions": num_predictions,
            "num_targets": 0,
            "recall": float("nan"),
        }

    # ── Sort predictions by descending confidence ───────────────────────────
    pred_entries.sort(key=lambda x: x[0], reverse=True)

    # ── Greedy matching ────────────────────────────────────────────────────
    # Track which targets have been matched
    target_matched = [False] * num_targets
    num_correct = 0

    for conf, pred_label, pred_box, pb, pn in pred_entries:
        best_iou = -1.0
        best_target_idx = -1

        for t_idx, (t_label, t_box, tb, tn) in enumerate(target_entries):
            if target_matched[t_idx]:
                continue

            # Compute IoU
            pred_expanded = pred_box.unsqueeze(0)  # [1, 4]
            target_expanded = t_box.unsqueeze(0)   # [1, 4]
            iou_matrix = _compute_iou(pred_expanded, target_expanded)  # [1, 1]
            iou = iou_matrix[0, 0].item()

            if iou > best_iou:
                best_iou = iou
                best_target_idx = t_idx

        # Check if this prediction is a correct detection
        if best_target_idx >= 0 and best_iou >= iou_threshold:
            # Check class label match
            t_label = target_entries[best_target_idx][0]
            if pred_label == t_label:
                num_correct += 1
                target_matched[best_target_idx] = True

    # ── Compute metrics ───────────────────────────────────────────────────
    detection_accuracy = num_correct / max(num_predictions, 1)
    num_matched_targets = sum(target_matched)
    recall = num_matched_targets / max(num_targets, 1)

    return {
        "detection_accuracy": detection_accuracy,
        "num_correct": num_correct,
        "num_predictions": num_predictions,
        "num_targets": num_targets,
        "recall": recall,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "compute_map",
    "compute_giou",
    "compute_box_statistics",
    "compute_detection_accuracy",
]
