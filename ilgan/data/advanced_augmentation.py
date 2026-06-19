"""
Advanced augmentation techniques for ILGAN.

Provides multi-sample augmentations (Mosaic, MixUp) and single-sample
augmentations (RandomErasing) that go beyond the basic transforms in
``ilgan.data.augmentation``.  These techniques are inspired by YOLOv4
and modern object-detection training pipelines.

All transforms operate on the same ``(image, boxes, labels, valid_mask)``
contract as the basic augmentations, with the addition of multi-sample
transforms that accept lists of samples and return a single fused sample.

Available transforms
--------------------
- MosaicAugmentation   — combine 4 images into a 2×2 grid (YOLOv4-style)
- MixUpAugmentation    — blend 2 images and their labels proportionally
- RandomErasing        — erase rectangular regions with random aspect ratios
- BatchAugmentationPipeline — composes single-sample and multi-sample augs
- build_advanced_augmentation_pipeline — factory for the full pipeline
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np

from ilgan.data.structures import Sample
from ilgan.data.augmentation import (
    Augmentation,
    Compose,
    build_default_augmentation_pipeline,
    AugReturn,
)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-sample augmentation base class
# ──────────────────────────────────────────────────────────────────────────────


class MultiSampleAugmentation(ABC):
    """Abstract base for augmentations that operate on multiple input samples
    and produce a single fused output sample.

    Subclasses must implement ``forward(samples)`` which accepts a list of
    ``Sample`` objects and returns a single ``Sample``.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Probability p must be in [0, 1], got {p}.")
        self.p = p

    @abstractmethod
    def forward(self, samples: List[Sample]) -> Sample:
        """Apply the multi-sample augmentation with *guaranteed* application.

        Parameters
        ----------
        samples : list of Sample
            Input samples to fuse (e.g. 4 for Mosaic, 2 for MixUp).

        Returns
        -------
        Sample
            A single fused sample.
        """
        ...

    def __call__(
        self,
        samples: List[Sample],
        rng_seed: Optional[int] = None,
    ) -> Sample:
        """Apply the augmentation (probabilistically if ``p < 1``).

        Parameters
        ----------
        samples : list of Sample
            Input samples.
        rng_seed : int, optional
            Deterministic seed for reproducibility.

        Returns
        -------
        Sample
            Fused sample (or the first sample if the augmentation is
            skipped due to probability).
        """
        if rng_seed is not None:
            state = torch.random.get_rng_state()
            torch.manual_seed(rng_seed)
            result = self._apply_probabilistic(samples)
            torch.random.set_rng_state(state)
            return result

        return self._apply_probabilistic(samples)

    def _apply_probabilistic(self, samples: List[Sample]) -> Sample:
        """Apply with probability ``self.p``, otherwise return first sample."""
        if self.p >= 1.0:
            return self.forward(samples)
        if self.p <= 0.0:
            return samples[0]
        if torch.rand(1).item() < self.p:
            return self.forward(samples)
        return samples[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# ──────────────────────────────────────────────────────────────────────────────
# MosaicAugmentation
# ──────────────────────────────────────────────────────────────────────────────


class MosaicAugmentation(MultiSampleAugmentation):
    """YOLOv4-style Mosaic augmentation.

    Combines 4 images into a single 2×2 grid mosaic.  A random centre point
    determines how much of each image is visible, increasing the diversity
    of object scales and contexts.  Box coordinates are transformed to the
    mosaic coordinate space, and boxes that fall entirely outside the visible
    area are discarded.

    The output image has the same spatial dimensions as the input images
    (all inputs must share the same spatial size).

    Parameters
    ----------
    p : float
        Probability of applying the mosaic transform.
    center_ratio_range : tuple of (float, float)
        Range ``(min, max)`` for the random centre point offset, expressed
        as a fraction of the image size.  Default ``(0.3, 0.7)``.
    pad_fill_value : float
        Value used to fill empty regions of the mosaic canvas
        (default 0.0, which is the mean of the ``[-1, 1]`` range).
    """

    def __init__(
        self,
        p: float = 1.0,
        center_ratio_range: Tuple[float, float] = (0.3, 0.7),
        pad_fill_value: float = 0.0,
    ) -> None:
        super().__init__(p=p)
        self.center_ratio_range = center_ratio_range
        self.pad_fill_value = pad_fill_value

    def forward(self, samples: List[Sample]) -> Sample:
        """Apply mosaic fusion to 4 input samples.

        Parameters
        ----------
        samples : list of Sample
            Exactly 4 samples to combine.  All must share the same spatial
            dimensions ``(H, W)``.

        Returns
        -------
        Sample
            A single mosaic sample with all valid boxes from the 4 inputs
            transformed to the mosaic coordinate space.
        """
        if len(samples) != 4:
            raise ValueError(
                f"MosaicAugmentation requires exactly 4 samples, got {len(samples)}."
            )

        # ── validate consistent spatial size ─────────────────────────────
        ref_h, ref_w = samples[0].image.shape[1:]
        for i, s in enumerate(samples[1:], start=1):
            if s.image.shape[1:] != (ref_h, ref_w):
                raise ValueError(
                    f"All mosaic samples must share the same spatial size. "
                    f"Sample 0: ({ref_h}, {ref_w}), sample {i}: "
                    f"({s.image.shape[1]}, {s.image.shape[2]})."
                )

        H, W = ref_h, ref_w
        device = samples[0].image.device

        # ── random centre point (in pixels) ──────────────────────────────
        cx_min = int(W * self.center_ratio_range[0])
        cx_max = int(W * self.center_ratio_range[1])
        cy_min = int(H * self.center_ratio_range[0])
        cy_max = int(H * self.center_ratio_range[1])
        cx = int(torch.randint(cx_min, max(cx_min + 1, cx_max + 1), (1,)).item())
        cy = int(torch.randint(cy_min, max(cy_min + 1, cy_max + 1), (1,)).item())
        cx = max(1, min(cx, W - 1))
        cy = max(1, min(cy, H - 1))

        # ── placement regions for each of the 4 images ────────────────────
        # Each image is resized to fill its quadrant and placed so that
        # the centre point (cx, cy) is the meeting point of all 4 quadrants.
        placements = [
            (0, 0, cx, cy),                    # top-left     (index 0)
            (cx, 0, W - cx, cy),               # top-right    (index 1)
            (0, cy, cx, H - cy),               # bottom-left  (index 2)
            (cx, cy, W - cx, H - cy),          # bottom-right (index 3)
        ]

        # ── create mosaic canvas ─────────────────────────────────────────
        mosaic = torch.full(
            (3, H, W), fill_value=self.pad_fill_value, dtype=torch.float32, device=device
        )

        all_boxes: List[List[float]] = []
        all_labels: List[int] = []

        for i, sample in enumerate(samples):
            x, y, w, h = placements[i]
            if w <= 0 or h <= 0:
                continue  # degenerate quadrant (very rare with reasonable ranges)

            # Resize image to (h, w) — note F.interpolate uses (H, W) order
            img = sample.image.unsqueeze(0)  # [1, 3, H_i, W_i]
            img_resized = F.interpolate(
                img, size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)  # [3, h, w]

            # Place in mosaic
            mosaic[:, y : y + h, x : x + w] = img_resized

            # ── transform boxes ──────────────────────────────────────────
            for j in range(sample.boxes.size(0)):
                if not sample.valid_mask[j]:
                    continue

                xc_norm, yc_norm, bw_norm, bh_norm = sample.boxes[j].tolist()

                # Convert from normalised [0,1] → absolute pixels in the
                # resized (quadrant) image
                xc_abs = xc_norm * w
                yc_abs = yc_norm * h
                bw_abs = bw_norm * w
                bh_abs = bh_norm * h

                # Shift to mosaic absolute coordinates
                xc_mosaic = xc_abs + x
                yc_mosaic = yc_abs + y
                bw_mosaic = bw_abs
                bh_mosaic = bh_abs

                # Convert back to normalised [0, 1] relative to mosaic size
                xc_out = xc_mosaic / W
                yc_out = yc_mosaic / H
                bw_out = bw_mosaic / W
                bh_out = bh_mosaic / H

                # Compute corners to check visibility
                x1 = xc_out - bw_out / 2.0
                y1 = yc_out - bh_out / 2.0
                x2 = xc_out + bw_out / 2.0
                y2 = yc_out + bh_out / 2.0

                # Keep box if at least some portion is visible
                if x2 > 0.0 and y2 > 0.0 and x1 < 1.0 and y1 < 1.0:
                    # Clamp to [0, 1]
                    x1_c = max(0.0, x1)
                    y1_c = max(0.0, y1)
                    x2_c = min(1.0, x2)
                    y2_c = min(1.0, y2)
                    new_w = x2_c - x1_c
                    new_h = y2_c - y1_c
                    if new_w > 0.0 and new_h > 0.0:
                        new_xc = (x1_c + x2_c) / 2.0
                        new_yc = (y1_c + y2_c) / 2.0
                        all_boxes.append([new_xc, new_yc, new_w, new_h])
                        all_labels.append(int(sample.labels[j].item()))

        # ── build output tensors ─────────────────────────────────────────
        n = len(all_boxes)
        if n > 0:
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=device)
            valid_mask = torch.ones(n, dtype=torch.bool, device=device)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels_tensor = torch.zeros(0, dtype=torch.long, device=device)
            valid_mask = torch.zeros(0, dtype=torch.bool, device=device)

        # ── build output sample ───────────────────────────────────────────
        # Use the first sample's path as a representative path
        return Sample(
            image=mosaic,
            boxes=boxes_tensor,
            labels=labels_tensor,
            valid_mask=valid_mask,
            image_path=samples[0].image_path,
            metadata={
                "augmented": True,
                "mosaic": True,
                "mosaic_center": (cx, cy),
                "num_source_images": 4,
            },
        )

    def __repr__(self) -> str:
        return (
            f"MosaicAugmentation("
            f"p={self.p}, "
            f"center_range={self.center_ratio_range})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# MixUpAugmentation
# ──────────────────────────────────────────────────────────────────────────────


class MixUpAugmentation(MultiSampleAugmentation):
    """MixUp augmentation for images with bounding boxes.

    Blends two images and their annotations using a mixing coefficient
    ``λ`` sampled from a Beta distribution.  The resulting image is a
    convex combination of the two inputs, and the box sets are concatenated
    (both sets of labels are preserved).  This acts as a strong regulariser
    that improves generalisation and reduces overfitting.

    Both input images must share the same spatial dimensions.

    Parameters
    ----------
    p : float
        Probability of applying the mixup.
    alpha : float
        Shape parameter for the Beta distribution from which ``λ`` is
        sampled.  Default 0.5 (symmetric).  Lower values produce more
        extreme mixes; higher values produce mixes closer to 0.5.
    """

    def __init__(self, p: float = 1.0, alpha: float = 0.5) -> None:
        super().__init__(p=p)
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        self.alpha = alpha

    def forward(self, samples: List[Sample]) -> Sample:
        """Apply MixUp to 2 input samples.

        Parameters
        ----------
        samples : list of Sample
            Exactly 2 samples to blend.  Both must share the same spatial
            dimensions ``(H, W)``.

        Returns
        -------
        Sample
            A single blended sample with concatenated boxes and labels.
        """
        if len(samples) != 2:
            raise ValueError(
                f"MixUpAugmentation requires exactly 2 samples, got {len(samples)}."
            )

        img1, boxes1, labels1, mask1 = (
            samples[0].image, samples[0].boxes, samples[0].labels, samples[0].valid_mask
        )
        img2, boxes2, labels2, mask2 = (
            samples[1].image, samples[1].boxes, samples[1].labels, samples[1].valid_mask
        )

        # Validate spatial consistency
        if img1.shape[1:] != img2.shape[1:]:
            raise ValueError(
                f"MixUp images must share spatial size. "
                f"Got {img1.shape[1:]} and {img2.shape[1:]}."
            )

        device = img1.device

        # ── sample mixing coefficient λ from Beta(alpha, alpha) ──────────
        # Use PyTorch's Gamma sampler for Beta: if x ~ Gamma(a,1), y ~ Gamma(b,1)
        # then x/(x+y) ~ Beta(a,b).  We use a = b = alpha for symmetric mixing.
        if self.alpha > 0.0:
            gamma_a = torch.distributions.Gamma(
                torch.tensor(self.alpha, device=device),
                torch.tensor(1.0, device=device),
            )
            gamma_b = torch.distributions.Gamma(
                torch.tensor(self.alpha, device=device),
                torch.tensor(1.0, device=device),
            )
            x = gamma_a.sample()
            y = gamma_b.sample()
            lam = x / (x + y)
            lam = lam.clamp(0.0, 1.0)
        else:
            lam = torch.tensor(0.5, device=device)

        # ── blend images ──────────────────────────────────────────────────
        blended_img = lam * img1 + (1.0 - lam) * img2

        # ── concatenate boxes and labels ─────────────────────────────────
        # Both sets of labels are preserved in the blended sample.
        n1 = boxes1.size(0)
        n2 = boxes2.size(0)

        if n1 > 0 and n2 > 0:
            boxes_cat = torch.cat([boxes1, boxes2], dim=0)
            labels_cat = torch.cat([labels1, labels2], dim=0)
            valid_cat = torch.cat([mask1, mask2], dim=0)
        elif n1 > 0:
            boxes_cat = boxes1.clone()
            labels_cat = labels1.clone()
            valid_cat = mask1.clone()
        elif n2 > 0:
            boxes_cat = boxes2.clone()
            labels_cat = labels2.clone()
            valid_cat = mask2.clone()
        else:
            boxes_cat = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels_cat = torch.zeros(0, dtype=torch.long, device=device)
            valid_cat = torch.zeros(0, dtype=torch.bool, device=device)

        # ── build output sample ───────────────────────────────────────────
        return Sample(
            image=blended_img,
            boxes=boxes_cat,
            labels=labels_cat,
            valid_mask=valid_cat,
            image_path=samples[0].image_path,
            metadata={
                "augmented": True,
                "mixup": True,
                "mixup_lambda": lam.item(),
                "num_source_images": 2,
            },
        )

    def __repr__(self) -> str:
        return f"MixUpAugmentation(p={self.p}, alpha={self.alpha})"


# ──────────────────────────────────────────────────────────────────────────────
# RandomErasing
# ──────────────────────────────────────────────────────────────────────────────


class RandomErasing(Augmentation):
    """Randomly erase rectangular regions of the image.

    Similar to Cutout but with random aspect ratios and positions for each
    erased rectangle.  The erased region can be filled with a constant
    value, random noise, or the original image mean.  Boxes and labels are
    passed through unchanged.

    This is a single-sample augmentation (extends ``Augmentation``).

    Parameters
    ----------
    p : float
        Probability of applying the erasing.
    scale_range : tuple of (float, float)
        Range for the erased area as a fraction of the total image area.
        Default ``(0.02, 0.2)``.
    aspect_ratio_range : tuple of (float, float)
        Range for the aspect ratio (width/height) of the erased rectangle.
        Default ``(0.3, 3.0)``.
    fill_mode : str
        One of ``"constant"``, ``"random"``, or ``"mean"``.
        - ``"constant"``: fill with ``fill_value``.
        - ``"random"``: fill with random uniform noise in ``[-1, 1]``.
        - ``"mean"``: fill with the per-channel mean of the image.
    fill_value : float
        Value used when ``fill_mode="constant"`` (default 0.0, the mean
        of the ``[-1, 1]`` range).
    max_erasures : int
        Maximum number of erased rectangles per image (default 1).
    """

    def __init__(
        self,
        p: float = 0.5,
        scale_range: Tuple[float, float] = (0.02, 0.2),
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.0),
        fill_mode: str = "constant",
        fill_value: float = 0.0,
        max_erasures: int = 1,
    ) -> None:
        super().__init__(p=p)
        if not (0.0 < scale_range[0] <= scale_range[1] < 1.0):
            raise ValueError(
                f"scale_range must be (min, max) with 0 < min <= max < 1, "
                f"got {scale_range}."
            )
        if aspect_ratio_range[0] <= 0.0 or aspect_ratio_range[1] < aspect_ratio_range[0]:
            raise ValueError(
                f"aspect_ratio_range must be (min, max) with min > 0, "
                f"got {aspect_ratio_range}."
            )
        valid_modes = {"constant", "random", "mean"}
        if fill_mode not in valid_modes:
            raise ValueError(
                f"fill_mode must be one of {valid_modes}, got {fill_mode!r}."
            )

        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.max_erasures = max_erasures

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> AugReturn:
        """Apply random erasing to the image.

        Parameters
        ----------
        image : torch.Tensor
            ``[3, H, W]`` in ``[-1, 1]``.
        boxes : torch.Tensor
            ``[N, 4]`` YOLO format.
        labels : torch.Tensor
            ``[N]`` class IDs.
        valid_mask : torch.Tensor
            ``[N]`` bool mask.

        Returns
        -------
        AugReturn
            (erased_image, boxes, labels, valid_mask) — boxes unchanged.
        """
        *_, H, W = image.shape
        area = H * W
        device = image.device
        output = image.clone()

        # Pre-compute per-channel mean if needed
        channel_mean: Optional[torch.Tensor] = None
        if self.fill_mode == "mean":
            channel_mean = image.mean(dim=[1, 2])  # [3]

        num_erasures = int(
            torch.randint(1, self.max_erasures + 1, (1,)).item()
        )

        for _ in range(num_erasures):
            # Sample erased area
            target_area = area * torch.empty(1).uniform_(
                self.scale_range[0], self.scale_range[1]
            ).item()

            # Sample aspect ratio
            aspect_ratio = torch.empty(1).uniform_(
                self.aspect_ratio_range[0], self.aspect_ratio_range[1]
            ).item()

            # Compute height and width
            h_erase = int(round(math.sqrt(target_area / aspect_ratio)))
            w_erase = int(round(math.sqrt(target_area * aspect_ratio)))

            # Clamp to image dimensions
            h_erase = min(h_erase, H - 1)
            w_erase = min(w_erase, W - 1)
            h_erase = max(h_erase, 1)
            w_erase = max(w_erase, 1)

            # Sample top-left corner
            x_start = int(torch.randint(0, W - w_erase + 1, (1,)).item())
            y_start = int(torch.randint(0, H - h_erase + 1, (1,)).item())

            # Determine fill values
            if self.fill_mode == "constant":
                fill_region = torch.full(
                    (3, h_erase, w_erase),
                    fill_value=self.fill_value,
                    dtype=torch.float32,
                    device=device,
                )
            elif self.fill_mode == "random":
                fill_region = torch.empty(
                    3, h_erase, w_erase, dtype=torch.float32, device=device
                ).uniform_(-1.0, 1.0)
            elif self.fill_mode == "mean":
                # channel_mean is [3]; reshape to [3, 1, 1] for broadcasting
                fill_region = channel_mean.view(3, 1, 1).expand(
                    -1, h_erase, w_erase
                )
            else:
                # Should never reach here due to validation in __init__
                fill_region = torch.zeros(
                    3, h_erase, w_erase, dtype=torch.float32, device=device
                )

            # Apply erasure
            output[:, y_start : y_start + h_erase, x_start : x_start + w_erase] = (
                fill_region
            )

        return output, boxes, labels, valid_mask

    def __repr__(self) -> str:
        return (
            f"RandomErasing("
            f"p={self.p}, "
            f"scale={self.scale_range}, "
            f"aspect={self.aspect_ratio_range}, "
            f"fill={self.fill_mode}, "
            f"max_erasures={self.max_erasures})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# BatchAugmentationPipeline
# ──────────────────────────────────────────────────────────────────────────────


class BatchAugmentationPipeline:
    """Composes single-sample and multi-sample augmentations into a single
    pipeline that operates on a list of samples (a batch).

    The pipeline applies multi-sample transforms (Mosaic, MixUp) first,
    then applies single-sample transforms (flip, jitter, affine, erasing,
    cutout, etc.) to each resulting sample independently.

    This allows seamless integration of both augmentation types in the
    training loop.

    Parameters
    ----------
    multi_sample_transforms : list of MultiSampleAugmentation, optional
        Multi-sample transforms applied to groups of samples (e.g. Mosaic
        on groups of 4, MixUp on pairs).  Applied in order.
    single_sample_transforms : Compose or list of Augmentation, optional
        Single-sample transforms applied to each sample after multi-sample
        fusion.  If a list is given, it is wrapped in a ``Compose``.
    mosaic_group_size : int
        Number of samples per mosaic group (default 4).  Samples are
        partitioned into groups of this size before applying mosaic.
    mixup_group_size : int
        Number of samples per mixup group (default 2).  Samples are
        partitioned into pairs before applying mixup.
    """

    def __init__(
        self,
        multi_sample_transforms: Optional[List[MultiSampleAugmentation]] = None,
        single_sample_transforms: Optional[Union[Compose, List[Augmentation]]] = None,
        mosaic_group_size: int = 4,
        mixup_group_size: int = 2,
    ) -> None:
        self.multi_sample_transforms = multi_sample_transforms or []
        self.mixup_group_size = mixup_group_size
        self.mosaic_group_size = mosaic_group_size

        if single_sample_transforms is None:
            self.single_sample_compose: Optional[Compose] = None
        elif isinstance(single_sample_transforms, Compose):
            self.single_sample_compose = single_sample_transforms
        else:
            self.single_sample_compose = Compose(single_sample_transforms)

    def __call__(
        self,
        samples: List[Sample],
        rng_seed: Optional[int] = None,
    ) -> List[Sample]:
        """Apply the full augmentation pipeline to a list of samples.

        Parameters
        ----------
        samples : list of Sample
            Input samples (typically a batch).
        rng_seed : int, optional
            Base seed for deterministic augmentation.  Sub-seeds are
            derived for each transform and each sample.

        Returns
        -------
        list of Sample
            Augmented samples.  The length may differ from the input
            length due to multi-sample fusion (e.g. 4 → 1 for mosaic).
        """
        current = samples

        # ── 1. Multi-sample transforms ──────────────────────────────────
        for transform in self.multi_sample_transforms:
            if isinstance(transform, MosaicAugmentation):
                current = self._apply_mosaic_to_batch(
                    current, transform, rng_seed
                )
            elif isinstance(transform, MixUpAugmentation):
                current = self._apply_mixup_to_batch(
                    current, transform, rng_seed
                )
            else:
                # Generic multi-sample: apply to groups
                current = self._apply_generic_multi_sample(
                    current, transform, rng_seed
                )

        # ── 2. Single-sample transforms ──────────────────────────────────
        if self.single_sample_compose is not None:
            augmented: List[Sample] = []
            for i, sample in enumerate(current):
                sub_seed = _mix_seed(rng_seed, i + 1000) if rng_seed is not None else None
                aug_sample = self.single_sample_compose.apply_to_sample(
                    sample, rng_seed=sub_seed
                )
                augmented.append(aug_sample)
            current = augmented

        return current

    # ── internal helpers ────────────────────────────────────────────────

    def _apply_mosaic_to_batch(
        self,
        samples: List[Sample],
        mosaic: MosaicAugmentation,
        rng_seed: Optional[int],
    ) -> List[Sample]:
        """Partition samples into groups of 4 and apply mosaic to each."""
        result: List[Sample] = []
        gs = self.mosaic_group_size
        for i in range(0, len(samples), gs):
            group = samples[i : i + gs]
            if len(group) < gs:
                # Pad with the last sample if we don't have enough
                while len(group) < gs:
                    group.append(samples[-1])
            sub_seed = _mix_seed(rng_seed, i + 2000) if rng_seed is not None else None
            result.append(mosaic(group, rng_seed=sub_seed))
        return result

    def _apply_mixup_to_batch(
        self,
        samples: List[Sample],
        mixup: MixUpAugmentation,
        rng_seed: Optional[int],
    ) -> List[Sample]:
        """Partition samples into pairs and apply mixup to each."""
        result: List[Sample] = []
        gs = self.mixup_group_size
        for i in range(0, len(samples), gs):
            group = samples[i : i + gs]
            if len(group) < gs:
                # Pad with the last sample
                while len(group) < gs:
                    group.append(samples[-1])
            sub_seed = _mix_seed(rng_seed, i + 3000) if rng_seed is not None else None
            result.append(mixup(group, rng_seed=sub_seed))
        return result

    @staticmethod
    def _apply_generic_multi_sample(
        samples: List[Sample],
        transform: MultiSampleAugmentation,
        rng_seed: Optional[int],
    ) -> List[Sample]:
        """Apply a generic multi-sample transform to pairs (default)."""
        result: List[Sample] = []
        for i in range(0, len(samples), 2):
            group = samples[i : i + 2]
            if len(group) < 2:
                group.append(samples[-1])
            sub_seed = _mix_seed(rng_seed, i + 4000) if rng_seed is not None else None
            result.append(transform(group, rng_seed=sub_seed))
        return result

    def __repr__(self) -> str:
        ms = [str(t) for t in self.multi_sample_transforms]
        ss = str(self.single_sample_compose) if self.single_sample_compose else "None"
        return (
            f"BatchAugmentationPipeline(\n"
            f"  multi_sample={ms},\n"
            f"  single_sample={ss}\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build_advanced_augmentation_pipeline
# ──────────────────────────────────────────────────────────────────────────────


def build_advanced_augmentation_pipeline(
    config: Optional[Dict] = None,
    image_size: int = 128,
    use_mosaic: bool = True,
    use_mixup: bool = True,
    use_random_erasing: bool = True,
    mosaic_prob: float = 0.5,
    mixup_prob: float = 0.5,
    mixup_alpha: float = 0.5,
    erasing_prob: float = 0.5,
    erasing_scale_range: Tuple[float, float] = (0.02, 0.2),
    erasing_aspect_range: Tuple[float, float] = (0.3, 3.0),
    erasing_fill_mode: str = "random",
    erasing_max_erasures: int = 1,
    hflip_prob: float = 0.5,
    color_jitter_prob: float = 0.8,
    affine_prob: float = 0.5,
    cutout_prob: float = 0.5,
    degrees: float = 5.0,
    translate: float = 0.05,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.05,
    max_holes: int = 3,
    max_cutout_size: float = 0.1,
    shuffle_order: bool = True,
) -> BatchAugmentationPipeline:
    """Build the full advanced augmentation pipeline for ILGAN training.

    Composes multi-sample augmentations (Mosaic, MixUp) with single-sample
    augmentations (flip, colour jitter, affine, cutout, random erasing) into
    a single ``BatchAugmentationPipeline``.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary (e.g. from ``Config.to_dict()``).  If
        provided, individual parameter values are overridden by the
        corresponding config keys under ``data.augmentation``.
    image_size : int
        Target image size (used for mosaic placement calculations).
    use_mosaic : bool
        Whether to include Mosaic augmentation.
    use_mixup : bool
        Whether to include MixUp augmentation.
    use_random_erasing : bool
        Whether to include RandomErasing.
    mosaic_prob : float
        Probability of applying Mosaic to each group of 4 samples.
    mixup_prob : float
        Probability of applying MixUp to each pair of samples.
    mixup_alpha : float
        Beta distribution alpha for MixUp.
    erasing_prob : float
        Probability of applying RandomErasing to each sample.
    erasing_scale_range : tuple
        Scale range for erased regions.
    erasing_aspect_range : tuple
        Aspect ratio range for erased regions.
    erasing_fill_mode : str
        Fill mode for erased regions.
    erasing_max_erasures : int
        Maximum number of erased rectangles per image.
    hflip_prob : float
        Horizontal flip probability.
    color_jitter_prob : float
        Colour jitter probability.
    affine_prob : float
        Affine transform probability.
    cutout_prob : float
        Cutout probability.
    degrees : float
        Max rotation degrees for affine.
    translate : float
        Max translation fraction for affine.
    scale_range : tuple
        Scale range for affine.
    brightness : float
        Brightness jitter strength.
    contrast : float
        Contrast jitter strength.
    saturation : float
        Saturation jitter strength.
    hue : float
        Hue jitter strength.
    max_holes : int
        Max cutout holes.
    max_cutout_size : float
        Max cutout hole size as fraction.
    shuffle_order : bool
        Whether to shuffle single-sample transform order.

    Returns
    -------
    BatchAugmentationPipeline
        Composed pipeline ready to be called on batches of samples.
    """
    # ── extract overrides from config dict ──────────────────────────────
    if config is not None:
        aug_cfg = config.get("data", {}).get("augmentation", {})
        if aug_cfg:
            use_mosaic = aug_cfg.get("use_mosaic", use_mosaic)
            use_mixup = aug_cfg.get("use_mixup", use_mixup)
            use_random_erasing = aug_cfg.get("use_random_erasing", use_random_erasing)
            mosaic_prob = aug_cfg.get("mosaic_prob", mosaic_prob)
            mixup_prob = aug_cfg.get("mixup_prob", mixup_prob)
            mixup_alpha = aug_cfg.get("mixup_alpha", mixup_alpha)
            erasing_prob = aug_cfg.get("erasing_prob", erasing_prob)
            erasing_scale_range = aug_cfg.get(
                "erasing_scale_range", erasing_scale_range
            )
            erasing_aspect_range = aug_cfg.get(
                "erasing_aspect_range", erasing_aspect_range
            )
            erasing_fill_mode = aug_cfg.get("erasing_fill_mode", erasing_fill_mode)
            erasing_max_erasures = aug_cfg.get(
                "erasing_max_erasures", erasing_max_erasures
            )
            hflip_prob = aug_cfg.get("hflip_prob", hflip_prob)
            color_jitter_prob = aug_cfg.get("color_jitter_prob", color_jitter_prob)
            affine_prob = aug_cfg.get("affine_prob", affine_prob)
            cutout_prob = aug_cfg.get("cutout_prob", cutout_prob)
            degrees = aug_cfg.get("degrees", degrees)
            translate = aug_cfg.get("translate", translate)
            scale_range = aug_cfg.get("scale_range", scale_range)
            brightness = aug_cfg.get("brightness", brightness)
            contrast = aug_cfg.get("contrast", contrast)
            saturation = aug_cfg.get("saturation", saturation)
            hue = aug_cfg.get("hue", hue)
            max_holes = aug_cfg.get("max_holes", max_holes)
            max_cutout_size = aug_cfg.get("max_cutout_size", max_cutout_size)
            shuffle_order = aug_cfg.get("shuffle_order", shuffle_order)

    # ── build multi-sample transforms ──────────────────────────────────
    multi_sample: List[MultiSampleAugmentation] = []
    if use_mosaic:
        multi_sample.append(MosaicAugmentation(p=mosaic_prob))
    if use_mixup:
        multi_sample.append(MixUpAugmentation(p=mixup_prob, alpha=mixup_alpha))

    # ── build single-sample transforms ──────────────────────────────────
    single_transforms: List[Augmentation] = []

    # Import the basic augmentations
    from ilgan.data.augmentation import (
        RandomHorizontalFlip,
        RandomColorJitter,
        RandomAffine,
        Cutout,
    )

    single_transforms.append(RandomHorizontalFlip(p=hflip_prob))
    single_transforms.append(
        RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=color_jitter_prob,
        )
    )
    single_transforms.append(
        RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale_range,
            p=affine_prob,
        )
    )
    single_transforms.append(
        Cutout(p=cutout_prob, max_holes=max_holes, max_size=max_cutout_size)
    )

    if use_random_erasing:
        single_transforms.append(
            RandomErasing(
                p=erasing_prob,
                scale_range=erasing_scale_range,
                aspect_ratio_range=erasing_aspect_range,
                fill_mode=erasing_fill_mode,
                max_erasures=erasing_max_erasures,
            )
        )

    single_compose = Compose(single_transforms, shuffle_order=shuffle_order)

    # ── build and return the full pipeline ──────────────────────────────
    return BatchAugmentationPipeline(
        multi_sample_transforms=multi_sample,
        single_sample_transforms=single_compose,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _mix_seed(base_seed: Optional[int], index: int) -> int:
    """Derive a deterministic sub-seed from a base seed and an index.

    Uses a simple multiplicative hash to spread seeds across the 32-bit
    integer space, avoiding correlation between adjacent indices.
    """
    if base_seed is None:
        return index
    return (base_seed ^ (index * 0x9E3779B9)) & 0xFFFFFFFF


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "MultiSampleAugmentation",
    "MosaicAugmentation",
    "MixUpAugmentation",
    "RandomErasing",
    "BatchAugmentationPipeline",
    "build_advanced_augmentation_pipeline",
]
