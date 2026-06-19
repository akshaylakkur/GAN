"""
Bounding box regression losses for the ILGAN dual-output GAN.

This module implements loss functions for the bounding box prediction head
of the ILGAN generator.  The generator produces, for each of ``N`` predicted
boxes per sample, a tuple of ``(cx, cy, w, h)`` coordinates, a class
logit vector, and a confidence (objectness) score.

The losses defined here are:

* :func:`giou_loss` — Generalized IoU (GIoU) loss, which directly optimises
  the spatial overlap between predicted and ground-truth boxes.
* :func:`l1_box_loss` — Smooth L1 loss on the normalised box coordinates,
  providing a stable regression signal.
* :func:`compute_box_losses` — Convenience function that combines GIoU and
  L1 losses into a single weighted sum.
* :func:`class_loss` — Cross-entropy loss for class label prediction.
* :func:`confidence_loss` — Binary cross-entropy loss for objectness scores.

All functions operate on padded batches (shape ``[B, N, ...]``) and accept a
``valid_mask`` (``[B, N]`` bool) to exclude padding entries from the loss.
When no valid boxes exist in a batch, the loss returns 0.0.

Mathematical grounding
----------------------
The GIoU loss (Rezatofighi et al., 2019) addresses a key limitation of
standard IoU: when two boxes do not overlap, IoU is zero and provides no
gradient.  GIoU extends IoU by also considering the area of the smallest
enclosing box:

    IoU = |A ∩ B| / |A ∪ B|

    GIoU = IoU - (|C| - |A ∪ B|) / |C|

where ``C`` is the smallest convex box that encloses both ``A`` and ``B``.
The GIoU loss is ``L_GIoU = 1 - GIoU``, which is bounded in ``[0, 2]``.

The smooth L1 loss (also known as Huber loss) provides a piecewise
quadratic-linear loss that is less sensitive to outliers than L2 loss:

    smooth_L1(x) = { 0.5 * x^2,          if |x| < 1
                   { |x| - 0.5,          otherwise

We apply it to the difference between predicted and target box coordinates
in the normalised ``(cx, cy, w, h)`` space.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversion helpers
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

    Example
    -------
    >>> boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    >>> _cxcywh_to_xyxy(boxes)
    tensor([[0.4000, 0.3000, 0.6000, 0.7000]])
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# GIoU Loss
# ──────────────────────────────────────────────────────────────────────────────


def giou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the Generalized IoU (GIoU) loss.

    The GIoU loss is defined as:

    .. math::

        L_{GIoU} = \frac{1}{N_{valid}} \sum_{i \in valid} (1 - GIoU_i)

    where

    .. math::

        GIoU_i = IoU_i - \frac{|C_i| - |A_i \cup B_i|}{|C_i|}

    with ``A`` = predicted box, ``B`` = target box, and ``C`` = smallest
    enclosing box.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format, normalised to ``[0, 1]``.
    target_boxes : torch.Tensor
        Ground-truth bounding boxes, same shape and format as
        ``pred_boxes``.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` entries indicate valid
        (non-padded) boxes that contribute to the loss.

    Returns
    -------
    torch.Tensor
        Scalar GIoU loss (0-dimensional).  Returns 0.0 if no valid boxes
        exist.

    Raises
    ------
    ValueError
        If ``pred_boxes``, ``target_boxes``, or ``valid_mask`` have
        incompatible shapes.

    Shape
    -----
    - ``pred_boxes``: ``[B, N, 4]``
    - ``target_boxes``: ``[B, N, 4]``
    - ``valid_mask``: ``[B, N]``
    - Output: scalar ``[]``

    Example
    -------
    >>> pred = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    >>> target = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    >>> mask = torch.tensor([[True]])
    >>> loss = giou_loss(pred, target, mask)
    >>> abs(loss.item()) < 1e-4
    True

    >>> # Non-overlapping boxes -> GIoU < 0 -> loss > 1
    >>> pred = torch.tensor([[[0.1, 0.1, 0.1, 0.1]]])
    >>> target = torch.tensor([[[0.9, 0.9, 0.1, 0.1]]])
    >>> mask = torch.tensor([[True]])
    >>> loss = giou_loss(pred, target, mask)
    >>> loss.item() > 1.0
    True

    >>> # Empty valid mask -> loss = 0
    >>> pred = torch.randn(2, 3, 4)
    >>> target = torch.randn(2, 3, 4)
    >>> mask = torch.zeros(2, 3, dtype=torch.bool)
    >>> loss = giou_loss(pred, target, mask)
    >>> loss.item()
    0.0
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
            f"{valid_mask.shape} are incompatible (expected "
            f"{pred_boxes.shape[:-1]} for valid_mask)."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")

    # ── Early exit if no valid boxes ─────────────────────────────────────
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

    # ── Convert from (cx, cy, w, h) to (x1, y1, x2, y2) ──────────────────
    pred_xyxy = _cxcywh_to_xyxy(pred_boxes)      # [B, N, 4]
    target_xyxy = _cxcywh_to_xyxy(target_boxes)   # [B, N, 4]

    # Clamp coordinates to [0, 1] to avoid numerical issues from
    # predictions that slightly exceed the normalised range.
    pred_xyxy = pred_xyxy.clamp(min=0.0, max=1.0)
    target_xyxy = target_xyxy.clamp(min=0.0, max=1.0)

    # ── Compute intersection ──────────────────────────────────────────────
    # Intersection top-left: max of x1, y1
    inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])  # [B, N]
    inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])  # [B, N]
    # Intersection bottom-right: min of x2, y2
    inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])  # [B, N]
    inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])  # [B, N]

    # Intersection area (clamp to 0 to handle non-overlapping boxes)
    inter_hw = (inter_x2 - inter_x1).clamp(min=0.0) * (inter_y2 - inter_y1).clamp(min=0.0)
    # inter_hw shape: [B, N]

    # ── Compute areas of predicted and target boxes ────────────────────────
    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * \
                (pred_xyxy[..., 3] - pred_xyxy[..., 1])  # [B, N]
    target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]) * \
                  (target_xyxy[..., 3] - target_xyxy[..., 1])  # [B, N]

    # ── Compute union area ────────────────────────────────────────────────
    union_area = pred_area + target_area - inter_hw  # [B, N]

    # ── Compute IoU ──────────────────────────────────────────────────────
    # Add small epsilon to avoid division by zero
    eps = 1e-7
    iou = inter_hw / (union_area + eps)  # [B, N]

    # ── Compute smallest enclosing box ───────────────────────────────────
    enclose_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])  # [B, N]
    enclose_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])  # [B, N]
    enclose_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])  # [B, N]
    enclose_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])  # [B, N]

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)  # [B, N]
    enclose_area = enclose_area.clamp(min=eps)  # avoid division by zero

    # ── Compute GIoU ─────────────────────────────────────────────────────
    # GIoU = IoU - (enclosing_area - union_area) / enclosing_area
    giou = iou - (enclose_area - union_area) / enclose_area  # [B, N]

    # ── Compute loss: L = 1 - GIoU, averaged over valid entries ──────────
    loss_per_box = 1.0 - giou  # [B, N]

    # Mask out invalid entries
    loss_per_box = loss_per_box * valid_mask.to(loss_per_box.dtype)

    # Average over valid entries
    num_valid = valid_mask.sum().clamp(min=1).to(loss_per_box.dtype)
    loss = loss_per_box.sum() / num_valid

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Smooth L1 Box Loss
# ──────────────────────────────────────────────────────────────────────────────


def l1_box_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the smooth L1 loss between predicted and target box coordinates.

    The smooth L1 (Huber) loss is applied element-wise to the 4 box
    coordinates ``(cx, cy, w, h)``:

    .. math::

        smooth_{L1}(x) = \begin{cases}
            0.5 \cdot x^2, & \text{if } |x| < 1 \\
            |x| - 0.5,     & \text{otherwise}
        \end{cases}

    The loss is averaged over the 4 coordinates and then over all valid
    (non-padded) boxes in the batch.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format.
    target_boxes : torch.Tensor
        Ground-truth bounding boxes, same shape and format.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.

    Returns
    -------
    torch.Tensor
        Scalar smooth L1 loss (0-dimensional).  Returns 0.0 if no valid
        boxes exist.

    Raises
    ------
    ValueError
        If shapes are incompatible.

    Shape
    -----
    - ``pred_boxes``: ``[B, N, 4]``
    - ``target_boxes``: ``[B, N, 4]``
    - ``valid_mask``: ``[B, N]``
    - Output: scalar ``[]``

    Example
    -------
    >>> pred = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    >>> target = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    >>> mask = torch.tensor([[True]])
    >>> loss = l1_box_loss(pred, target, mask)
    >>> loss.item()
    0.0

    >>> # Empty mask
    >>> pred = torch.randn(2, 3, 4)
    >>> target = torch.randn(2, 3, 4)
    >>> mask = torch.zeros(2, 3, dtype=torch.bool)
    >>> loss = l1_box_loss(pred, target, mask)
    >>> loss.item()
    0.0
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
        return torch.tensor(0.0, device=pred_boxes.device, dtype=pred_boxes.dtype)

    # ── Compute smooth L1 loss ──────────────────────────────────────────
    # F.smooth_l1_loss computes element-wise smooth L1 and returns a tensor
    # of the same shape as the inputs.  We use reduction='none' to manually
    # mask and average.
    element_loss = F.smooth_l1_loss(
        pred_boxes, target_boxes, reduction="none"
    )  # [B, N, 4]

    # Average over the 4 coordinates -> per-box loss
    per_box_loss = element_loss.mean(dim=-1)  # [B, N]

    # Mask out invalid entries
    per_box_loss = per_box_loss * valid_mask.to(per_box_loss.dtype)

    # Average over valid entries
    num_valid = valid_mask.sum().clamp(min=1).to(per_box_loss.dtype)
    loss = per_box_loss.sum() / num_valid

    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Combined Box Losses
# ──────────────────────────────────────────────────────────────────────────────


def compute_box_losses(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    valid_mask: torch.Tensor,
    box_weight: float = 5.0,
) -> Dict[str, torch.Tensor]:
    r"""Compute all bounding box regression losses.

    This function computes the GIoU loss and the smooth L1 loss, then
    combines them into a weighted sum:

    .. math::

        L_{box} = w_{box} \cdot (L_{GIoU} + L_{smooth L1})

    The weight :math:`w_{box}` controls the contribution of the box
    regression losses relative to other losses (e.g. adversarial, class,
    confidence) in the full generator objective.

    Parameters
    ----------
    pred_boxes : torch.Tensor
        Predicted bounding boxes, shape ``[B, N, 4]`` in ``(cx, cy, w, h)``
        format.
    target_boxes : torch.Tensor
        Ground-truth bounding boxes, same shape and format.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.
    box_weight : float, optional
        Weight scaling factor for the combined box loss.  Must be
        non-negative.  (default: ``5.0``)

    Returns
    -------
    dict of str -> torch.Tensor
        A dictionary with the following keys:

        - ``"giou_loss"``: the GIoU loss (scalar).
        - ``"l1_loss"``: the smooth L1 loss (scalar).
        - ``"box_loss"``: the weighted sum ``box_weight * (giou_loss + l1_loss)``
          (scalar).

    Raises
    ------
    ValueError
        If ``box_weight`` is negative.

    Example
    -------
    >>> pred = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]])
    >>> target = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]])
    >>> mask = torch.tensor([[True, True]])
    >>> losses = compute_box_losses(pred, target, mask, box_weight=5.0)
    >>> abs(losses["giou_loss"].item()) < 1e-4
    True
    >>> abs(losses["l1_loss"].item()) < 1e-7
    True
    >>> abs(losses["box_loss"].item()) < 1e-3
    True
    >>> list(losses.keys())
    ['giou_loss', 'l1_loss', 'box_loss']
    """
    if box_weight < 0.0:
        raise ValueError(
            f"box_weight must be non-negative, got {box_weight}."
        )

    giou = giou_loss(pred_boxes, target_boxes, valid_mask)
    l1 = l1_box_loss(pred_boxes, target_boxes, valid_mask)
    box = box_weight * (giou + l1)

    return {
        "giou_loss": giou,
        "l1_loss": l1,
        "box_loss": box,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Class Loss
# ──────────────────────────────────────────────────────────────────────────────


def class_loss(
    class_logits: torch.Tensor,
    target_labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Compute the cross-entropy loss for class label prediction.

    The loss is computed as:

    .. math::

        L_{cls} = \frac{1}{N_{valid}} \sum_{i \in valid}
                  \text{CrossEntropy}(\text{logits}_i, \text{label}_i)

    where ``CrossEntropy`` is the standard softmax cross-entropy.

    Parameters
    ----------
    class_logits : torch.Tensor
        Predicted class logits, shape ``[B, N, num_classes]``.
    target_labels : torch.Tensor
        Ground-truth class labels, shape ``[B, N]``.  Values are integer
        class IDs in ``[0, num_classes)``.  Invalid (padding) entries may
        contain arbitrary values (they are masked out).
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.

    Returns
    -------
    dict of str -> torch.Tensor
        A dictionary with key ``"class_loss"`` mapping to the scalar
        cross-entropy loss (0-dimensional).  Returns 0.0 if no valid
        boxes exist.

    Raises
    ------
    ValueError
        If shapes are incompatible.

    Shape
    -----
    - ``class_logits``: ``[B, N, C]`` where ``C`` is the number of classes.
    - ``target_labels``: ``[B, N]``.
    - ``valid_mask``: ``[B, N]``.
    - Output: dict with scalar ``"class_loss"``.

    Example
    -------
    >>> logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    >>> labels = torch.tensor([[0, 1]])
    >>> mask = torch.tensor([[True, True]])
    >>> result = class_loss(logits, labels, mask)
    >>> result["class_loss"].item()  # doctest: +ELLIPSIS
    0.1269...

    >>> # Empty mask
    >>> logits = torch.randn(2, 3, 5)
    >>> labels = torch.zeros(2, 3, dtype=torch.long)
    >>> mask = torch.zeros(2, 3, dtype=torch.bool)
    >>> result = class_loss(logits, labels, mask)
    >>> result["class_loss"].item()
    0.0
    """
    # ── Input validation ──────────────────────────────────────────────────
    if class_logits.shape[:-1] != target_labels.shape:
        raise ValueError(
            f"class_logits shape {class_logits.shape} and target_labels "
            f"shape {target_labels.shape} are incompatible (expected "
            f"{class_logits.shape[:-1]} for target_labels)."
        )
    if class_logits.shape[:-1] != valid_mask.shape:
        raise ValueError(
            f"class_logits shape {class_logits.shape} and valid_mask shape "
            f"{valid_mask.shape} are incompatible."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")

    # ── Early exit if no valid boxes ─────────────────────────────────────
    if not valid_mask.any():
        return {"class_loss": torch.tensor(0.0, device=class_logits.device,
                                            dtype=class_logits.dtype)}

    # ── Compute cross-entropy with reduction='none' ──────────────────────
    # F.cross_entropy expects [N, C] and [N] inputs.  We flatten the
    # batch and box dimensions, then filter to only valid entries.
    B, N, C = class_logits.shape
    logits_flat = class_logits.view(B * N, C)        # [B*N, C]
    labels_flat = target_labels.view(B * N)           # [B*N]
    mask_flat = valid_mask.view(B * N)                 # [B*N]

    # Only compute loss on valid (non-padded) entries to avoid
    # IndexError from -1 labels.
    if mask_flat.any():
        valid_logits = logits_flat[mask_flat]          # [V, C]
        valid_labels = labels_flat[mask_flat]          # [V]
        per_element_loss = F.cross_entropy(
            valid_logits, valid_labels, reduction="none"
        )  # [V]
        loss = per_element_loss.mean()
    else:
        loss = torch.tensor(0.0, device=class_logits.device,
                            dtype=class_logits.dtype)

    return {"class_loss": loss}


# ──────────────────────────────────────────────────────────────────────────────
# Confidence (Objectness) Loss
# ──────────────────────────────────────────────────────────────────────────────


def confidence_loss(
    confidences: torch.Tensor,
    target_confidence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Compute the binary cross-entropy loss for objectness scores.

    The confidence (objectness) score indicates whether a predicted box
    corresponds to a real object.  For valid (non-padded) boxes, the target
    confidence is 1.0; for invalid (padded) boxes, the target is 0.0.

    The loss is:

    .. math::

        L_{conf} = \frac{1}{N_{total}} \sum_{i}
                   \text{BCE}(\text{conf}_i, \text{target}_i)

    where ``BCE`` is the binary cross-entropy loss.  Note that the loss is
    averaged over **all** entries (both valid and invalid), because the
    model must learn to predict 0.0 confidence for padded positions.

    Parameters
    ----------
    confidences : torch.Tensor
        Predicted confidence (objectness) scores, shape ``[B, N]``.  Values
        should be in ``(0, 1)`` after sigmoid activation.
    target_confidence : torch.Tensor
        Target confidence scores, shape ``[B, N]``.  Should be 1.0 for
        valid boxes and 0.0 for invalid (padded) boxes.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, N]``.  ``True`` for valid boxes.

    Returns
    -------
    dict of str -> torch.Tensor
        A dictionary with key ``"confidence_loss"`` mapping to the scalar
        binary cross-entropy loss (0-dimensional).  Returns 0.0 if the
        batch is empty.

    Raises
    ------
    ValueError
        If shapes are incompatible.

    Shape
    -----
    - ``confidences``: ``[B, N]``.
    - ``target_confidence``: ``[B, N]``.
    - ``valid_mask``: ``[B, N]``.
    - Output: dict with scalar ``"confidence_loss"``.

    Example
    -------
    >>> conf = torch.tensor([[0.9, 0.1]])
    >>> target = torch.tensor([[1.0, 0.0]])
    >>> mask = torch.tensor([[True, False]])
    >>> result = confidence_loss(conf, target, mask)
    >>> result["confidence_loss"].item()  # doctest: +ELLIPSIS
    0.1053...

    >>> # All boxes invalid -> loss computed over all entries
    >>> conf = torch.tensor([[0.1, 0.2]])
    >>> target = torch.tensor([[0.0, 0.0]])
    >>> mask = torch.tensor([[False, False]])
    >>> result = confidence_loss(conf, target, mask)
    >>> result["confidence_loss"].item() > 0.0
    True
    """
    # ── Input validation ──────────────────────────────────────────────────
    if confidences.shape != target_confidence.shape:
        raise ValueError(
            f"confidences shape {confidences.shape} must match "
            f"target_confidence shape {target_confidence.shape}."
        )
    if confidences.shape != valid_mask.shape:
        raise ValueError(
            f"confidences shape {confidences.shape} must match "
            f"valid_mask shape {valid_mask.shape}."
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool, got {valid_mask.dtype}.")

    # ── Early exit if batch is empty ─────────────────────────────────────
    if confidences.numel() == 0:
        return {"confidence_loss": torch.tensor(0.0, device=confidences.device,
                                                  dtype=confidences.dtype)}

    # ── Compute binary cross-entropy ─────────────────────────────────────
    # F.binary_cross_entropy expects values in [0, 1].  We clamp to avoid
    # log(0) numerical issues.
    confidences_clamped = confidences.clamp(min=1e-7, max=1.0 - 1e-7)

    # Use reduction='none' so we can manually handle masking
    per_element_loss = F.binary_cross_entropy(
        confidences_clamped, target_confidence, reduction="none"
    )  # [B, N]

    # Average over ALL entries (both valid and invalid), because the model
    # must learn to assign low confidence to padded positions.
    loss = per_element_loss.mean()

    return {"confidence_loss": loss}


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "giou_loss",
    "l1_box_loss",
    "compute_box_losses",
    "class_loss",
    "confidence_loss",
]
