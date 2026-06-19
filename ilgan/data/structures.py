"""
Core data structures for ILGAN.

Defines ``Sample``, ``Batch``, ``DatasetMetadata``, and the YOLO label parsing
function ``parse_yolo_label``. These types are the fundamental currency of the
data pipeline: every dataset, loader, and collation step produces or consumes
these structures.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Sample
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    """A single training sample consisting of an image and its bounding box
    annotations.

    Fields
    ------
    image : torch.Tensor
        Image tensor of shape ``[3, H, W]`` with values normalized to ``[-1, 1]``.
    boxes : torch.Tensor
        Bounding box tensor of shape ``[N, 4]`` where each row is
        ``[x_center, y_center, width, height]``, all values in ``[0, 1]``
        (normalised relative to image dimensions).
    labels : torch.Tensor
        Integer class IDs of shape ``[N]``.
    valid_mask : torch.Tensor
        Boolean tensor of shape ``[N]``; ``True`` for real (non-padded) boxes.
        For a freshly loaded sample every entry is ``True``.
    image_path : str
        Filesystem path to the source image file.
    metadata : dict
        Free-form dictionary for extra information (e.g. original filename,
        dataset split, augmentation history).
    """

    image: torch.Tensor
    boxes: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor
    image_path: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Lightweight validation of tensor shapes and value ranges.

        Raises
        ------
        ValueError
            If any tensor has an incorrect shape or contains out-of-range
            values.
        """
        # --- image ---
        if self.image.dim() != 3 or self.image.size(0) != 3:
            raise ValueError(
                f"image must be [3, H, W], got {tuple(self.image.shape)}"
            )
        if self.image.min() < -1.001 or self.image.max() > 1.001:
            raise ValueError(
                f"image values must be in [-1, 1], "
                f"got range [{self.image.min():.4f}, {self.image.max():.4f}]"
            )

        # --- boxes ---
        if self.boxes.dim() != 2 or self.boxes.size(1) != 4:
            raise ValueError(
                f"boxes must be [N, 4], got {tuple(self.boxes.shape)}"
            )
        n = self.boxes.size(0)

        # --- labels (shape only; dtype checked implicitly later) ---
        if self.labels.dim() != 1 or self.labels.size(0) != n:
            raise ValueError(
                f"labels must be [N] with N={n}, "
                f"got {tuple(self.labels.shape)}"
            )

        # --- valid_mask (check shape + dtype BEFORE any indexing) ---
        if self.valid_mask.dim() != 1 or self.valid_mask.size(0) != n:
            raise ValueError(
                f"valid_mask must be [N] with N={n}, "
                f"got {tuple(self.valid_mask.shape)}"
            )
        if self.valid_mask.dtype != torch.bool:
            raise ValueError(
                f"valid_mask must be bool, got {self.valid_mask.dtype}"
            )

        # Now it's safe to use valid_mask for indexing
        _assert_finite(self.boxes, "boxes")
        if n > 0 and self.valid_mask.any():
            valid_boxes = self.boxes[self.valid_mask]
            if valid_boxes.min() < -0.001 or valid_boxes.max() > 1.001:
                raise ValueError(
                    f"box coordinates must be in [0, 1], "
                    f"got range [{valid_boxes.min():.4f}, {valid_boxes.max():.4f}]"
                )

    # ── convenience helpers ─────────────────────────────────────────────

    @property
    def num_boxes(self) -> int:
        """Number of valid (non-padded) boxes in this sample."""
        return int(self.valid_mask.sum().item())

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.image.size(1)

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.image.size(2)

    def to(self, device: torch.device) -> Sample:
        """Move all tensors to *device* and return a new Sample."""
        return Sample(
            image=self.image.to(device),
            boxes=self.boxes.to(device),
            labels=self.labels.to(device),
            valid_mask=self.valid_mask.to(device),
            image_path=self.image_path,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Sample(image={tuple(self.image.shape)}, "
            f"boxes={tuple(self.boxes.shape)}, "
            f"labels={tuple(self.labels.shape)}, "
            f"valid={self.num_boxes}/{self.boxes.size(0)}, "
            f"path={self.image_path!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Batch:
    """A collated batch of ``Sample`` objects with padded tensor fields.

    Fields
    ------
    images : torch.Tensor
        Stacked images of shape ``[B, 3, H, W]``.
    boxes : torch.Tensor
        Padded bounding boxes of shape ``[B, max_boxes, 4]``.  Invalid
        (padding) entries are filled with ``-1.0``.
    labels : torch.Tensor
        Padded class IDs of shape ``[B, max_boxes]``.  Invalid entries are
        filled with ``-1``.
    valid_mask : torch.Tensor
        Boolean mask of shape ``[B, max_boxes]``; ``True`` where a box is
        real.
    image_paths : list of str
        Source image paths for each sample in the batch.
    metadata : list of dict
        Per-sample metadata dictionaries.
    batch_size : int
        Actual batch size (may be smaller than the configured value for the
        final partial batch).
    max_boxes_in_batch : int
        The dynamic per-batch maximum box count (never exceeds the
        model-configured global maximum).
    """

    images: torch.Tensor
    boxes: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor
    image_paths: List[str]
    metadata: List[Dict]
    batch_size: int
    max_boxes_in_batch: int

    # ── collation ───────────────────────────────────────────────────────

    @staticmethod
    def collate(
        samples: List[Sample],
        global_max_boxes: Optional[int] = None,
    ) -> Batch:
        """Collate a list of ``Sample`` objects into a padded ``Batch``.

        Parameters
        ----------
        samples : list of Sample
            The samples to collate.  All images must share the same spatial
            dimensions ``(H, W)``.
        global_max_boxes : int or None
            Hard upper bound on the number of boxes per sample
            (e.g. ``model.max_boxes`` from the config).  If None, no
            cap is applied (the dynamic max for the batch is used directly).

        Returns
        -------
        Batch
            A single batch ready for model consumption.

        Raises
        ------
        ValueError
            If the list is empty or images have inconsistent spatial sizes.
        """
        if not samples:
            raise ValueError("Cannot collate an empty list of samples.")

        # Validate consistent image size
        ref_h, ref_w = samples[0].image.shape[1:]
        for i, s in enumerate(samples):
            if s.image.shape[1:] != (ref_h, ref_w):
                raise ValueError(
                    f"All images must share the same spatial dimensions. "
                    f"Sample 0: ({ref_h}, {ref_w}), sample {i}: "
                    f"({s.image.shape[1]}, {s.image.shape[2]})."
                )

        # --- stack images ---
        images = torch.stack([s.image for s in samples], dim=0)  # [B, 3, H, W]

        # --- determine max boxes for this batch ---
        max_boxes = max((s.boxes.size(0) for s in samples), default=0)
        if global_max_boxes is not None:
            max_boxes = min(max_boxes, global_max_boxes)

        batch_size = len(samples)
        device = samples[0].image.device

        # --- allocate padded tensors ---
        boxes = torch.full((batch_size, max_boxes, 4), fill_value=-1.0, dtype=torch.float32, device=device)
        labels = torch.full((batch_size, max_boxes), fill_value=-1, dtype=torch.long, device=device)
        valid_mask = torch.zeros(batch_size, max_boxes, dtype=torch.bool, device=device)

        image_paths: List[str] = []
        metadata_list: List[Dict] = []

        for i, s in enumerate(samples):
            n = s.boxes.size(0)
            n_effective = min(n, max_boxes)

            if n_effective > 0:
                boxes[i, :n_effective] = s.boxes[:n_effective]
                labels[i, :n_effective] = s.labels[:n_effective]
                valid_mask[i, :n_effective] = s.valid_mask[:n_effective]

            # Warn if a sample has more boxes than the allowed max
            if n > max_boxes and global_max_boxes is not None:
                warnings.warn(
                    f"Sample has {n} boxes but global_max_boxes={global_max_boxes}. "
                    f"Truncating to {max_boxes} boxes.",
                    stacklevel=2,
                )

            image_paths.append(s.image_path)
            metadata_list.append(s.metadata)

        return Batch(
            images=images,
            boxes=boxes,
            labels=labels,
            valid_mask=valid_mask,
            image_paths=image_paths,
            metadata=metadata_list,
            batch_size=batch_size,
            max_boxes_in_batch=max_boxes,
        )

    # ── device movement ─────────────────────────────────────────────────

    def to(self, device: torch.device) -> Batch:
        """Move all tensors to *device* and return a new Batch."""
        return Batch(
            images=self.images.to(device),
            boxes=self.boxes.to(device),
            labels=self.labels.to(device),
            valid_mask=self.valid_mask.to(device),
            image_paths=list(self.image_paths),
            metadata=list(self.metadata),
            batch_size=self.batch_size,
            max_boxes_in_batch=self.max_boxes_in_batch,
        )

    # ── slicing ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        return (
            f"Batch(images={tuple(self.images.shape)}, "
            f"boxes={tuple(self.boxes.shape)}, "
            f"labels={tuple(self.labels.shape)}, "
            f"valid_mask={tuple(self.valid_mask.shape)}, "
            f"B={self.batch_size}, max_boxes={self.max_boxes_in_batch})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# YOLO label parser
# ──────────────────────────────────────────────────────────────────────────────


def parse_yolo_label(txt_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read a YOLO-format label file and return its contents as tensors.

    Each line in the file must follow the format::

        class_id x_center y_center width height

    where all coordinate values are in ``[0, 1]`` (normalised relative to
    image dimensions), ``width`` and ``height`` are positive, and
    ``class_id`` is a non-negative integer.

    Parameters
    ----------
    txt_path : str
        Path to the ``.txt`` label file.

    Returns
    -------
    boxes : torch.Tensor
        Shape ``[N, 4]``, dtype ``float32``.
    labels : torch.Tensor
        Shape ``[N]``, dtype ``int64``.
    valid_mask : torch.Tensor
        Shape ``[N]``, dtype ``bool``.  All entries are ``True`` for real
        (non-padded) boxes.  If the file does not exist or is empty, all
        tensors have ``N = 0``.

    Raises
    ------
    ValueError
        If any line cannot be parsed or contains out-of-range values.
    """
    if not os.path.isfile(txt_path):
        # Return empty tensors for missing files
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
        )

    class_ids: List[int] = []
    coords_list: List[List[float]] = []

    with open(txt_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue  # skip blank lines

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 values per line, got {len(parts)} "
                    f"at {txt_path}:{line_no} ('{line}')."
                )

            try:
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse numeric values at {txt_path}:{line_no} "
                    f"('{line}'): {exc}"
                ) from exc

            # Validate ranges
            if cls_id < 0:
                raise ValueError(
                    f"class_id must be non-negative, got {cls_id} "
                    f"at {txt_path}:{line_no}."
                )
            if not (0.0 <= x_c <= 1.0):
                raise ValueError(
                    f"x_center must be in [0, 1], got {x_c} "
                    f"at {txt_path}:{line_no}."
                )
            if not (0.0 <= y_c <= 1.0):
                raise ValueError(
                    f"y_center must be in [0, 1], got {y_c} "
                    f"at {txt_path}:{line_no}."
                )
            if w <= 0.0 or w > 1.0:
                raise ValueError(
                    f"width must be in (0, 1], got {w} "
                    f"at {txt_path}:{line_no}."
                )
            if h <= 0.0 or h > 1.0:
                raise ValueError(
                    f"height must be in (0, 1], got {h} "
                    f"at {txt_path}:{line_no}."
                )

            class_ids.append(cls_id)
            coords_list.append([x_c, y_c, w, h])

    N = len(class_ids)
    if N == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
        )

    boxes = torch.tensor(coords_list, dtype=torch.float32)   # [N, 4]
    labels = torch.tensor(class_ids, dtype=torch.long)        # [N]
    valid_mask = torch.ones(N, dtype=torch.bool)              # [N]

    return boxes, labels, valid_mask


# ──────────────────────────────────────────────────────────────────────────────
# DatasetMetadata
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetMetadata:
    """Immutable dataset-level metadata.

    Attributes
    ----------
    class_names : list of str
        Human-readable class names, indexed by class ID.
    class_to_idx : dict
        Mapping from class name (str) to integer index.
    num_classes : int
        Total number of classes (``len(class_names)``).
    train_size : int
        Number of training samples.
    val_size : int
        Number of validation / test samples.
    image_size : tuple of (int, int)
        Target spatial dimensions ``(height, width)`` in pixels.
    """

    class_names: List[str]
    class_to_idx: Dict[str, int]
    num_classes: int
    train_size: int
    val_size: int
    image_size: Tuple[int, int]

    def __post_init__(self) -> None:
        """Validate consistency between fields."""
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) must match "
                f"num_classes ({self.num_classes})."
            )
        if len(self.class_to_idx) != self.num_classes:
            raise ValueError(
                f"class_to_idx size ({len(self.class_to_idx)}) must match "
                f"num_classes ({self.num_classes})."
            )
        if self.train_size < 0:
            raise ValueError(f"train_size must be non-negative, got {self.train_size}.")
        if self.val_size < 0:
            raise ValueError(f"val_size must be non-negative, got {self.val_size}.")
        if not (isinstance(self.image_size, tuple) and len(self.image_size) == 2):
            raise ValueError(
                f"image_size must be a tuple of (height, width), "
                f"got {self.image_size!r}."
            )

    @property
    def total_size(self) -> int:
        """Total number of samples across all splits."""
        return self.train_size + self.val_size

    def idx_to_name(self, idx: int) -> str:
        """Return the class name for a given integer index."""
        return self.class_names[idx]

    def name_to_idx(self, name: str) -> int:
        """Return the integer index for a given class name."""
        return self.class_to_idx[name]

    def __repr__(self) -> str:
        return (
            f"DatasetMetadata(classes={self.num_classes}, "
            f"train={self.train_size}, val={self.val_size}, "
            f"image_size={self.image_size})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _assert_finite(t: torch.Tensor, name: str) -> None:
    """Raise ``ValueError`` if *t* contains any NaN or Inf values."""
    if torch.isnan(t).any():
        raise ValueError(f"{name} contains NaN values.")
    if torch.isinf(t).any():
        raise ValueError(f"{name} contains Inf values.")


__all__ = [
    "Sample",
    "Batch",
    "DatasetMetadata",
    "parse_yolo_label",
]