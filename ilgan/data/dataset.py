"""
YOLO-format dataset for ILGAN.

Provides ``YOLODataset`` — a ``torch.utils.data.Dataset`` that reads images
and corresponding YOLO-format label files, resizes them with aspect-ratio-
preserving padding, and returns ``Sample`` objects ready for model
consumption.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

from ilgan.data.structures import Sample, parse_yolo_label

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

TransformCallable = Optional[Callable[[torch.Tensor], torch.Tensor]]


# ──────────────────────────────────────────────────────────────────────────────
# Utility: resize with padding
# ──────────────────────────────────────────────────────────────────────────────


def resize_with_pad(
    image: Image.Image,
    target_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, Tuple[int, int]]:
    """Resize the longer side of *image* to *target_size*, pad the shorter side
    with zeros (value -1 before normalization), and return the normalised tensor
    together with metadata.

    The padding colour is black (0,0,0) which becomes -1 after the ``[-1,1]``
    mapping — this ensures padded regions do not leak false signal into the
    network.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in RGB mode.
    target_size : int
        Desired size of the longer side in pixels.

    Returns
    -------
    resized_tensor : torch.Tensor
        Shape ``[3, target_size, target_size]``, values in ``[-1, 1]``.
    padding_mask : torch.Tensor
        Shape ``[1, target_size, target_size]``, dtype ``bool``.
        ``True`` where the image is present (non-padded).
    scale_factor : float
        Ratio by which the original image was scaled
        (``target_size / original_longer_side``).
    pad_amounts : tuple of (int, int)
        ``(pad_left, pad_top)`` applied to the shorter sides.
    """
    # ── ensure RGB ──────────────────────────────────────────────────────
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    longer_side = max(orig_w, orig_h)
    scale_factor = target_size / longer_side

    # ── resize longer side to target_size ───────────────────────────────
    new_w = int(round(orig_w * scale_factor))
    new_h = int(round(orig_h * scale_factor))
    # Clamp to at least 1 pixel
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Use high-quality bilinear resampling
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # ── create square canvas and paste ──────────────────────────────────
    square = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    square.paste(resized, (pad_left, pad_top))

    # ── convert to tensor in [-1, 1] ────────────────────────────────────
    arr = np.asarray(square, dtype=np.float32)          # [H, W, 3]
    arr = arr.transpose((2, 0, 1))                      # [3, H, W]
    tensor = torch.from_numpy(arr) / 127.5 - 1.0        # [0,255] → [-1, 1]
    tensor = tensor.clamp(-1.0, 1.0)

    # ── padding mask (1 = image, 0 = padded border) ─────────────────────
    mask = torch.zeros(1, target_size, target_size, dtype=torch.bool)
    mask[:, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = True

    return tensor, mask, scale_factor, (pad_left, pad_top)


# ──────────────────────────────────────────────────────────────────────────────
# YOLODataset
# ──────────────────────────────────────────────────────────────────────────────


class YOLODataset(torch.utils.data.Dataset):
    """Dataset for images with YOLO-format bounding box annotations.

    The dataset expects the following directory structure under *root_dir*::

        root_dir/
        ├── images/       # *.jpg (or *.png) image files
        ├── labels/       # *.txt YOLO label files (same stem as images)
        ├── train.txt     # (optional) list of image stems for training split
        └── val.txt       # (optional) list of image stems for validation split

    If the split files do not exist, the dataset performs a deterministic
    80/20 random split (seeded for reproducibility).

    Parameters
    ----------
    root_dir : str
        Path to the dataset root directory.
    image_size : int
        Target size for the longer side of the image (pixels). The final
        tensor is a square of shape ``[3, image_size, image_size]``.
    split : str
        One of ``"train"`` or ``"val"`` (or ``"test"``, treated as ``"val"``).
    max_boxes : int
        Maximum number of bounding boxes per sample. If a sample has fewer,
        it is padded with sentinel values (``-1``). If it has more, the
        excess boxes are discarded.
    transform : callable or None
        Optional per-sample transform applied to the image tensor after
        resizing/padding.  Signature: ``(Tensor) -> Tensor``.
    seed : int
        Random seed used when creating the split file list (default 42).
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int,
        split: str = "train",
        max_boxes: int = 50,
        transform: TransformCallable = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._root_dir = Path(root_dir)
        if not self._root_dir.is_dir():
            raise NotADirectoryError(f"root_dir does not exist: {root_dir}")

        self._image_size = image_size
        self._split = split
        self._max_boxes = max_boxes
        self._transform = transform

        # Normalise split name
        split_lower = split.lower()
        if split_lower in ("val", "test", "valid"):
            self._split_name = "val"
        elif split_lower == "train":
            self._split_name = "train"
        else:
            raise ValueError(f"Unknown split '{split}'; expected 'train' or 'val'.")

        # ── resolve image / label directories ────────────────────────────
        self._img_dir = self._root_dir / "images"
        self._label_dir = self._root_dir / "labels"

        if not self._img_dir.is_dir():
            raise NotADirectoryError(
                f"Images directory not found: {self._img_dir}"
            )

        # ── build file list ──────────────────────────────────────────────
        split_file = self._root_dir / f"{self._split_name}.txt"
        if split_file.is_file():
            # Load stems from the split file
            stems = self._load_stems_from_file(split_file)
        else:
            # Fall back to all available images, then split randomly
            all_stems = self._discover_stems()
            if not all_stems:
                raise FileNotFoundError(
                    f"No image files found in {self._img_dir}"
                )
            stems = self._create_split(all_stems, self._split_name, seed)

        # Filter stems that have a corresponding label file (or allow
        # samples with no labels — label file is optional).
        self._samples: List[Dict[str, str]] = []
        for stem in stems:
            img_path = self._find_image(stem)
            if img_path is None:
                continue
            label_path = self._label_dir / f"{stem}.txt"
            self._samples.append({
                "image": str(img_path),
                "label": str(label_path) if label_path.is_file() else None,
            })

        if not self._samples:
            raise FileNotFoundError(
                f"No valid samples found for split '{self._split_name}' "
                f"in {root_dir}"
            )

    # ── public helpers ─────────────────────────────────────────────────

    @property
    def image_size(self) -> int:
        """Target image size (longer side)."""
        return self._image_size

    @property
    def split(self) -> str:
        """Dataset split name."""
        return self._split_name

    @property
    def root_dir(self) -> str:
        """Root directory of the dataset."""
        return str(self._root_dir)

    @property
    def max_boxes(self) -> int:
        """Maximum number of boxes per sample."""
        return self._max_boxes

    # ── required Dataset overrides ─────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        """Load and return a single sample.

        Parameters
        ----------
        idx : int
            Index into the dataset.

        Returns
        -------
        Sample
            A fully-populated ``Sample`` with the image resized, padded,
            normalised to ``[-1, 1]``, boxes rescaled to the new image
            dimensions, and padded to ``max_boxes``.
        """
        entry = self._samples[idx]
        img_path = entry["image"]

        # ── 1. Read image ───────────────────────────────────────────────
        pil_image = Image.open(img_path)
        orig_w, orig_h = pil_image.size

        # ── 2. Resize with padding ──────────────────────────────────────
        image_tensor, padding_mask, scale_factor, (pad_left, pad_top) = \
            resize_with_pad(pil_image, self._image_size)

        # ── 3. Optional transform ───────────────────────────────────────
        if self._transform is not None:
            image_tensor = self._transform(image_tensor)

        # ── 4. Parse labels / boxes ─────────────────────────────────────
        label_path = entry["label"]
        if label_path is not None and os.path.isfile(label_path):
            boxes, labels, valid_mask = parse_yolo_label(label_path)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
            valid_mask = torch.zeros(0, dtype=torch.bool)

        # ── 5. Rescale boxes to padded image coordinates ────────────────
        if boxes.size(0) > 0:
            boxes = self._rescale_boxes(
                boxes, orig_w, orig_h,
                scale_factor, pad_left, pad_top,
                self._image_size,
            )

        # ── 6. Truncate / pad to max_boxes ──────────────────────────────
        n = boxes.size(0)
        if n > self._max_boxes:
            # Truncate — keep the first max_boxes
            boxes = boxes[:self._max_boxes]
            labels = labels[:self._max_boxes]
            valid_mask = valid_mask[:self._max_boxes]
            n = self._max_boxes

        # Pad with sentinel values
        if n < self._max_boxes:
            pad_n = self._max_boxes - n
            boxes = torch.cat([
                boxes,
                torch.full((pad_n, 4), fill_value=-1.0, dtype=torch.float32),
            ], dim=0)
            labels = torch.cat([
                labels,
                torch.full((pad_n,), fill_value=-1, dtype=torch.long),
            ], dim=0)
            valid_mask = torch.cat([
                valid_mask,
                torch.zeros(pad_n, dtype=torch.bool),
            ], dim=0)

        # ── 7. Build Sample ─────────────────────────────────────────────
        sample = Sample(
            image=image_tensor,
            boxes=boxes,
            labels=labels,
            valid_mask=valid_mask,
            image_path=img_path,
            metadata={
                "split": self._split_name,
                "padding_mask": padding_mask,
                "orig_size": (orig_w, orig_h),
                "scale_factor": scale_factor,
                "pad_amounts": (pad_left, pad_top),
            },
        )

        return sample

    # ── internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _load_stems_from_file(split_file: Path) -> List[str]:
        """Read one stem per line from *split_file*.

        Lines starting with ``#`` are treated as comments and skipped.
        """
        stems: List[str] = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Remove extension if present (e.g. "image001.jpg" → "image001")
                stem = Path(line).stem
                stems.append(stem)
        return stems

    def _discover_stems(self) -> List[str]:
        """Collect all image file stems from the images directory.

        Supported extensions: ``.jpg``, ``.jpeg``, ``.png``, ``.bmp``,
        ``.webp``.
        """
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        stems: List[str] = []
        for fname in os.listdir(self._img_dir):
            p = Path(fname)
            if p.suffix.lower() in extensions:
                stems.append(p.stem)
        stems.sort()
        return stems

    @staticmethod
    def _create_split(
        all_stems: List[str],
        split_name: str,
        seed: int,
        val_ratio: float = 0.2,
    ) -> List[str]:
        """Deterministically split *all_stems* into train/val.

        The split is reproducible given the same *seed* and
        *all_stems* order.
        """
        rng = random.Random(seed)
        indices = list(range(len(all_stems)))
        rng.shuffle(indices)

        n_val = max(1, int(round(len(all_stems) * val_ratio)))
        if split_name == "val":
            sel = indices[:n_val]
        else:
            sel = indices[n_val:]

        return [all_stems[i] for i in sel]

    def _find_image(self, stem: str) -> Optional[Path]:
        """Return the full path for *stem* with any supported extension, or
        ``None`` if no matching image file exists.
        """
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        for ext in extensions:
            candidate = self._img_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
            candidate = self._img_dir / f"{stem}{ext.upper()}"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _rescale_boxes(
        boxes: torch.Tensor,
        orig_w: int,
        orig_h: int,
        scale_factor: float,
        pad_left: int,
        pad_top: int,
        target_size: int,
    ) -> torch.Tensor:
        """Rescale YOLO-normalised boxes from original image coordinates to
        the padded square image coordinates.

        YOLO format stores ``[x_center, y_center, width, height]`` where all
        values are normalised relative to the original image dimensions.

        After rescaling, the coordinates remain in ``[0, 1]`` relative to the
        *target_size* square.

        Parameters
        ----------
        boxes : torch.Tensor
            Shape ``[N, 4]`` in YOLO format (``[xc, yc, w, h]``).
        orig_w : int
            Original image width.
        orig_h : int
            Original image height.
        scale_factor : float
            Resize scale factor.
        pad_left : int
            Number of pixels padded on the left.
        pad_top : int
            Number of pixels padded on the top.
        target_size : int
            Target square size.

        Returns
        -------
        torch.Tensor
            Rescaled boxes, same shape ``[N, 4]``.
        """
        if boxes.size(0) == 0:
            return boxes

        # Convert from normalised → absolute pixel coordinates (original size)
        xc = boxes[:, 0] * orig_w
        yc = boxes[:, 1] * orig_h
        bw = boxes[:, 2] * orig_w
        bh = boxes[:, 3] * orig_h

        # Apply resize scaling
        xc = xc * scale_factor
        yc = yc * scale_factor
        bw = bw * scale_factor
        bh = bh * scale_factor

        # Apply padding offset
        xc = xc + pad_left
        yc = yc + pad_top

        # Convert back to normalised coordinates relative to target_size
        xc = xc / target_size
        yc = yc / target_size
        bw = bw / target_size
        bh = bh / target_size

        # Clamp to valid range [0, 1] (accounts for partial truncation at edges)
        xc = xc.clamp(0.0, 1.0)
        yc = yc.clamp(0.0, 1.0)
        bw = bw.clamp(0.0, 1.0)
        bh = bh.clamp(0.0, 1.0)

        rescaled = torch.stack([xc, yc, bw, bh], dim=1)
        return rescaled

    # ── representation ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root={self._root_dir!s}, "
            f"split={self._split_name!r}, "
            f"size={self._image_size}, "
            f"samples={len(self._samples)}, "
            f"max_boxes={self._max_boxes})"
        )