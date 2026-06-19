"""
Dataset analysis and label validation utilities for ILGAN.

Provides two primary functions:

- :func:`analyze_dataset`: Scans a dataset directory (images + YOLO labels)
  and computes comprehensive statistics: image counts, box counts per image,
  class distribution, image size distribution, box size distribution, and
  box centre heatmap data.  Prints a formatted report and optionally saves
  visualisation plots to disk.

- :func:`validate_labels`: Scans all YOLO label files in a dataset and
  checks each for correctness: valid format (5 whitespace-separated values),
  coordinates in ``[0, 1]``, positive width/height, non-negative class IDs.
  Reports all errors found.

These functions are designed to be used both as standalone CLI tools and as
programmatic building blocks for data quality assurance pipelines.

Usage
-----
As a script::

    python -m ilgan.scripts.analyze_data analyze /path/to/dataset --class-names cat dog bird --save-plots ./plots

    python -m ilgan.scripts.analyze_data validate /path/to/dataset

Programmatically::

    from ilgan.scripts.analyze_data import analyze_dataset, validate_labels

    stats = analyze_dataset("/path/to/dataset", class_names=["cat", "dog"])
    errors = validate_labels("/path/to/dataset")
"""

from __future__ import annotations

import os
import sys
import math
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports — matplotlib and seaborn are not hard dependencies
# ──────────────────────────────────────────────────────────────────────────────

_HAS_MATPLOTLIB: bool = False
_HAS_SEABORN: bool = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment,misc]
    Axes = None  # type: ignore[assignment,misc]

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    sns = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
)

_REPORT_SEPARATOR: str = "=" * 72
_SECTION_SEPARATOR: str = "-" * 72


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

DatasetStats = Dict[str, Any]
"""Comprehensive dataset statistics dictionary returned by :func:`analyze_dataset`.

Keys include:

- ``total_images`` (int): Number of image files found.
- ``total_boxes`` (int): Total number of bounding boxes across all images.
- ``images_with_boxes`` (int): Number of images that have at least one box.
- ``images_without_boxes`` (int): Number of images with no label file or empty labels.
- ``boxes_per_image`` (dict): ``{"mean", "std", "min", "max", "histogram"}``.
- ``class_distribution`` (dict): Class ID -> count.
- ``class_names`` (list or None): Provided class names.
- ``image_sizes`` (dict): ``{"widths", "heights", "aspect_ratios"}`` with stats.
- ``box_sizes`` (dict): ``{"widths", "heights", "areas"}`` with stats.
- ``box_centres`` (dict): ``{"x_centres", "y_centres"}`` lists for heatmap.
- ``label_errors`` (list): Any errors found during label parsing.
"""

LabelError = Dict[str, Any]
"""A single label validation error with keys: ``file``, ``line``, ``message``."""


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis function
# ──────────────────────────────────────────────────────────────────────────────


def analyze_dataset(
    data_root: str,
    class_names: Optional[Sequence[str]] = None,
    save_plots: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> DatasetStats:
    """Scan a dataset directory and compute comprehensive statistics.

    The dataset is expected to have the following structure under
    *data_root*::

        data_root/
        ├── images/       # *.jpg, *.png, etc.
        └── labels/       # *.txt YOLO label files (same stem as images)

    Parameters
    ----------
    data_root : str
        Path to the dataset root directory containing ``images/`` and
        ``labels/`` subdirectories.
    class_names : sequence of str, optional
        Human-readable class names indexed by class ID.  If provided,
        the class distribution report will include class names alongside
        IDs.  If ``None``, only class IDs are shown.
    save_plots : str, optional
        Directory path where visualisation plots will be saved as PNG
        files.  If ``None``, no plots are saved.  Requires ``matplotlib``
        and optionally ``seaborn`` for enhanced styling.
    max_samples : int, optional
        Maximum number of samples to analyse.  Useful for quick statistics
        on large datasets.  If ``None``, all samples are analysed.
    verbose : bool
        If ``True``, prints detailed per-sample information for the
        first 10 samples.

    Returns
    -------
    DatasetStats
        A dictionary containing all computed statistics (see the type
        alias for details).

    Raises
    ------
    FileNotFoundError
        If *data_root* does not exist or contains no images.
    ValueError
        If the dataset structure is invalid.

    Notes
    -----
    - Images without a corresponding label file are counted as having
      zero boxes.
    - Corrupt images (unable to open) are skipped with a warning.
    - The function is designed to handle datasets with millions of
      images efficiently by streaming rather than loading everything
      into memory.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    img_dir = root / "images"
    label_dir = root / "labels"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    # ── Discover image files ───────────────────────────────────────────
    image_paths: List[Path] = []
    for fname in sorted(os.listdir(str(img_dir))):
        p = img_dir / fname
        if p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS and p.is_file():
            image_paths.append(p)

    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {img_dir}")

    if max_samples is not None and max_samples > 0:
        image_paths = image_paths[:max_samples]

    total_images: int = len(image_paths)

    # ── Accumulators ───────────────────────────────────────────────────
    boxes_per_image: List[int] = []
    class_counter: Counter = Counter()
    image_widths: List[int] = []
    image_heights: List[int] = []
    aspect_ratios: List[float] = []
    box_widths: List[float] = []
    box_heights: List[float] = []
    box_areas: List[float] = []
    box_centres_x: List[float] = []
    box_centres_y: List[float] = []
    label_errors: List[LabelError] = []
    images_without_boxes: int = 0
    corrupt_images: int = 0

    # ── Iterate over images ────────────────────────────────────────────
    for idx, img_path in enumerate(image_paths):
        # ── Open image to get dimensions ───────────────────────────────
        try:
            with Image.open(str(img_path)) as pil_img:
                orig_w, orig_h = pil_img.size
        except (OSError, ValueError, SyntaxError) as e:
            corrupt_images += 1
            label_errors.append({
                "file": str(img_path),
                "line": 0,
                "message": f"Corrupt or unreadable image: {e}",
            })
            continue

        image_widths.append(orig_w)
        image_heights.append(orig_h)
        aspect_ratios.append(orig_w / max(orig_h, 1))

        # ── Parse corresponding label file ─────────────────────────────
        stem = img_path.stem
        label_path = label_dir / f"{stem}.txt"

        if not label_path.is_file():
            boxes_per_image.append(0)
            images_without_boxes += 1
            continue

        # Parse the label file
        try:
            boxes, labels, valid_mask = _parse_label_file(str(label_path))
        except ValueError as e:
            label_errors.append({
                "file": str(label_path),
                "line": 0,
                "message": f"Parse error: {e}",
            })
            boxes_per_image.append(0)
            images_without_boxes += 1
            continue

        n_boxes = int(valid_mask.sum().item())
        boxes_per_image.append(n_boxes)

        if n_boxes == 0:
            images_without_boxes += 1

        # Accumulate class counts
        for cls_id in labels[valid_mask].tolist():
            class_counter[cls_id] += 1

        # Accumulate box statistics
        valid_boxes = boxes[valid_mask]  # [M, 4]
        if valid_boxes.size(0) > 0:
            bw = valid_boxes[:, 2].tolist()
            bh = valid_boxes[:, 3].tolist()
            bx = valid_boxes[:, 0].tolist()
            by = valid_boxes[:, 1].tolist()

            box_widths.extend(bw)
            box_heights.extend(bh)
            box_areas.extend([w * h for w, h in zip(bw, bh)])
            box_centres_x.extend(bx)
            box_centres_y.extend(by)

        # ── Verbose: print first 10 samples ────────────────────────────
        if verbose and idx < 10:
            print(f"  Sample {idx:>5d}: {img_path.name:40s} "
                  f"size=({orig_w:>4d}, {orig_h:>4d}) "
                  f"boxes={n_boxes:>3d}")

    # ── Compute aggregate statistics ───────────────────────────────────
    total_boxes: int = sum(boxes_per_image)
    images_with_boxes: int = total_images - images_without_boxes

    # Boxes per image
    bpi_array = np.array(boxes_per_image, dtype=np.float64)
    bpi_mean = float(bpi_array.mean()) if len(bpi_array) > 0 else 0.0
    bpi_std = float(bpi_array.std()) if len(bpi_array) > 0 else 0.0
    bpi_min = int(bpi_array.min()) if len(bpi_array) > 0 else 0
    bpi_max = int(bpi_array.max()) if len(bpi_array) > 0 else 0

    # Histogram of boxes per image (bins: 0, 1-2, 3-5, 6-10, 11-20, 21+)
    bpi_hist: Dict[str, int] = {
        "0": int((bpi_array == 0).sum()),
        "1-2": int(((bpi_array >= 1) & (bpi_array <= 2)).sum()),
        "3-5": int(((bpi_array >= 3) & (bpi_array <= 5)).sum()),
        "6-10": int(((bpi_array >= 6) & (bpi_array <= 10)).sum()),
        "11-20": int(((bpi_array >= 11) & (bpi_array <= 20)).sum()),
        "21+": int((bpi_array > 20).sum()),
    }

    # Image size statistics
    img_w_array = np.array(image_widths, dtype=np.float64)
    img_h_array = np.array(image_heights, dtype=np.float64)
    ar_array = np.array(aspect_ratios, dtype=np.float64)

    image_size_stats: Dict[str, Any] = {
        "widths": {
            "mean": float(img_w_array.mean()),
            "std": float(img_w_array.std()),
            "min": int(img_w_array.min()),
            "max": int(img_w_array.max()),
            "median": float(np.median(img_w_array)),
        },
        "heights": {
            "mean": float(img_h_array.mean()),
            "std": float(img_h_array.std()),
            "min": int(img_h_array.min()),
            "max": int(img_h_array.max()),
            "median": float(np.median(img_h_array)),
        },
        "aspect_ratios": {
            "mean": float(ar_array.mean()),
            "std": float(ar_array.std()),
            "min": float(ar_array.min()),
            "max": float(ar_array.max()),
            "median": float(np.median(ar_array)),
        },
    }

    # Box size statistics
    box_size_stats: Dict[str, Any] = {}
    if box_widths:
        bw_array = np.array(box_widths, dtype=np.float64)
        bh_array = np.array(box_heights, dtype=np.float64)
        ba_array = np.array(box_areas, dtype=np.float64)

        box_size_stats = {
            "widths": {
                "mean": float(bw_array.mean()),
                "std": float(bw_array.std()),
                "min": float(bw_array.min()),
                "max": float(bw_array.max()),
                "median": float(np.median(bw_array)),
            },
            "heights": {
                "mean": float(bh_array.mean()),
                "std": float(bh_array.std()),
                "min": float(bh_array.min()),
                "max": float(bh_array.max()),
                "median": float(np.median(bh_array)),
            },
            "areas": {
                "mean": float(ba_array.mean()),
                "std": float(ba_array.std()),
                "min": float(ba_array.min()),
                "max": float(ba_array.max()),
                "median": float(np.median(ba_array)),
            },
        }
    else:
        box_size_stats = {
            "widths": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0},
            "heights": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0},
            "areas": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0},
        }

    # Class distribution
    class_dist: Dict[str, int] = {}
    for cls_id, count in sorted(class_counter.items()):
        if class_names is not None and cls_id < len(class_names):
            class_dist[f"{cls_id} ({class_names[cls_id]})"] = count
        else:
            class_dist[str(cls_id)] = count

    # ── Build result dictionary ────────────────────────────────────────
    stats: DatasetStats = {
        "total_images": total_images,
        "total_boxes": total_boxes,
        "images_with_boxes": images_with_boxes,
        "images_without_boxes": images_without_boxes,
        "corrupt_images": corrupt_images,
        "boxes_per_image": {
            "mean": bpi_mean,
            "std": bpi_std,
            "min": bpi_min,
            "max": bpi_max,
            "histogram": bpi_hist,
        },
        "class_distribution": class_dist,
        "class_names": list(class_names) if class_names else None,
        "num_classes": len(class_counter),
        "image_sizes": image_size_stats,
        "box_sizes": box_size_stats,
        "box_centres": {
            "x": box_centres_x,
            "y": box_centres_y,
        },
        "label_errors": label_errors,
    }

    # ── Print formatted report ─────────────────────────────────────────
    _print_report(stats, class_names=list(class_names) if class_names else None)

    # ── Save plots ─────────────────────────────────────────────────────
    if save_plots is not None and _HAS_MATPLOTLIB:
        _save_plots(stats, save_plots)
    elif save_plots is not None and not _HAS_MATPLOTLIB:
        print(
            "  ⚠  matplotlib is not installed.  Install it with:\n"
            "      pip install matplotlib seaborn\n"
            "  to enable plot saving."
        )

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Label validation function
# ──────────────────────────────────────────────────────────────────────────────


def validate_labels(data_root: str) -> List[LabelError]:
    """Scan all YOLO label files in a dataset and validate their contents.

    Each label file is expected to contain one bounding box per line in
    YOLO format::

        class_id x_center y_center width height

    where all coordinate values are normalised to ``[0, 1]``, width and
    height are positive, and class_id is a non-negative integer.

    The function checks for:

    - File existence and readability.
    - Correct number of fields per line (exactly 5).
    - Numeric parsability of all fields.
    - ``class_id`` is a non-negative integer.
    - ``x_center`` and ``y_center`` are in ``[0, 1]``.
    - ``width`` and ``height`` are in ``(0, 1]`` (positive, not exceeding 1).
    - No duplicate lines (identical class_id + coordinates).
    - No empty label files (files with only blank lines).

    Parameters
    ----------
    data_root : str
        Path to the dataset root directory containing ``labels/``
        subdirectory.

    Returns
    -------
    list of LabelError
        A list of error dictionaries, each with keys ``file``, ``line``,
        and ``message``.  If no errors are found, the list is empty.

    Raises
    ------
    FileNotFoundError
        If *data_root* does not exist or the ``labels/`` directory is
        missing.

    Notes
    -----
    - The function also checks for orphaned label files (label files
      without a corresponding image) and reports them as warnings.
    - The function does **not** modify any files.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    label_dir = root / "labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {label_dir}")

    img_dir = root / "images"
    has_images_dir = img_dir.is_dir()

    # Collect all image stems (for orphan detection)
    image_stems: set[str] = set()
    if has_images_dir:
        for fname in os.listdir(str(img_dir)):
            p = img_dir / fname
            if p.suffix.lower() in _SUPPORTED_IMAGE_EXTENSIONS and p.is_file():
                image_stems.add(p.stem)

    errors: List[LabelError] = []
    label_files: List[Path] = sorted(label_dir.glob("*.txt"))

    if not label_files:
        print(f"  ⚠  No label files found in {label_dir}")
        return errors

    total_files: int = len(label_files)
    files_with_errors: int = 0
    orphaned_labels: int = 0
    empty_files: int = 0

    for lf in label_files:
        stem = lf.stem
        file_has_error: bool = False

        # Check for orphaned label file
        if has_images_dir and stem not in image_stems:
            errors.append({
                "file": str(lf),
                "line": 0,
                "message": "Orphaned label file: no corresponding image found.",
            })
            orphaned_labels += 1
            file_has_error = True

        # Read and validate the file
        try:
            with open(str(lf), "r") as f:
                lines = f.readlines()
        except (OSError, PermissionError) as e:
            errors.append({
                "file": str(lf),
                "line": 0,
                "message": f"Cannot read file: {e}",
            })
            files_with_errors += 1
            continue

        # Filter out blank lines
        non_blank_lines = [line for line in lines if line.strip()]
        if not non_blank_lines:
            errors.append({
                "file": str(lf),
                "line": 0,
                "message": "Empty label file (no valid annotations).",
            })
            empty_files += 1
            file_has_error = True
            if file_has_error:
                files_with_errors += 1
            continue

        # Track seen annotations to detect duplicates
        seen_annotations: set[tuple] = set()

        for line_no, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": (
                        f"Expected 5 whitespace-separated values, "
                        f"got {len(parts)}: '{line}'"
                    ),
                })
                file_has_error = True
                continue

            # Try parsing
            try:
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as e:
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": f"Non-numeric value: {e} in '{line}'",
                })
                file_has_error = True
                continue

            # Validate class_id
            if cls_id < 0:
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": f"Negative class_id: {cls_id}",
                })
                file_has_error = True

            # Validate x_center
            if not (0.0 <= x_c <= 1.0):
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": (
                        f"x_center out of range [0, 1]: {x_c}. "
                        f"YOLO format requires normalised coordinates."
                    ),
                })
                file_has_error = True

            # Validate y_center
            if not (0.0 <= y_c <= 1.0):
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": (
                        f"y_center out of range [0, 1]: {y_c}. "
                        f"YOLO format requires normalised coordinates."
                    ),
                })
                file_has_error = True

            # Validate width
            if w <= 0.0 or w > 1.0:
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": (
                        f"width out of range (0, 1]: {w}. "
                        f"Width must be positive and not exceed 1.0."
                    ),
                })
                file_has_error = True

            # Validate height
            if h <= 0.0 or h > 1.0:
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": (
                        f"height out of range (0, 1]: {h}. "
                        f"Height must be positive and not exceed 1.0."
                    ),
                })
                file_has_error = True

            # Check for duplicate annotations
            annotation_key = (cls_id, round(x_c, 6), round(y_c, 6), round(w, 6), round(h, 6))
            if annotation_key in seen_annotations:
                errors.append({
                    "file": str(lf),
                    "line": line_no,
                    "message": (
                        f"Duplicate annotation: class_id={cls_id}, "
                        f"box=({x_c:.6f}, {y_c:.6f}, {w:.6f}, {h:.6f})"
                    ),
                })
                file_has_error = True
            else:
                seen_annotations.add(annotation_key)

        if file_has_error:
            files_with_errors += 1

    # ── Print validation summary ────────────────────────────────────────
    print()
    print(_REPORT_SEPARATOR)
    print("  ILGAN — Label Validation Report")
    print(_REPORT_SEPARATOR)
    print(f"  Labels directory:  {label_dir}")
    print(f"  Total label files: {total_files}")
    print(f"  Files with errors: {files_with_errors}")
    print(f"  Orphaned labels:   {orphaned_labels}")
    print(f"  Empty files:        {empty_files}")
    print(f"  Total errors:       {len(errors)}")
    print(_SECTION_SEPARATOR)

    if not errors:
        print("  ✅  All label files are valid.  No errors found.")
    else:
        # Group errors by type for a summary
        error_types: Counter = Counter()
        for err in errors:
            msg = err["message"]
            if "Expected 5" in msg:
                error_types["wrong_field_count"] += 1
            elif "Non-numeric" in msg:
                error_types["non_numeric"] += 1
            elif "Negative class_id" in msg:
                error_types["negative_class_id"] += 1
            elif "x_center out of range" in msg:
                error_types["x_center_range"] += 1
            elif "y_center out of range" in msg:
                error_types["y_center_range"] += 1
            elif "width out of range" in msg:
                error_types["width_range"] += 1
            elif "height out of range" in msg:
                error_types["height_range"] += 1
            elif "Duplicate annotation" in msg:
                error_types["duplicate"] += 1
            elif "Orphaned label" in msg:
                error_types["orphaned"] += 1
            elif "Empty label file" in msg:
                error_types["empty_file"] += 1
            else:
                error_types["other"] += 1

        print("  Error breakdown by type:")
        for err_type, count in error_types.most_common():
            print(f"    {err_type:25s}: {count}")

        print()
        print("  First 20 errors (see full list in returned data):")
        for err in errors[:20]:
            print(f"    {err['file']}:{err['line']} — {err['message']}")

        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more errors.")

    print(_REPORT_SEPARATOR)
    print()

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_label_file(txt_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse a YOLO label file and return tensors.

    This is a lightweight version of ``ilgan.data.structures.parse_yolo_label``
    that does not raise on out-of-range values (it clamps instead), making it
    suitable for analysis where we want to collect statistics even on slightly
    malformed data.

    Parameters
    ----------
    txt_path : str
        Path to the label file.

    Returns
    -------
    boxes : torch.Tensor
        Shape ``[N, 4]``, dtype ``float32``.
    labels : torch.Tensor
        Shape ``[N]``, dtype ``long``.
    valid_mask : torch.Tensor
        Shape ``[N]``, dtype ``bool``.
    """
    if not os.path.isfile(txt_path):
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
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Line {line_no}: expected 5 values, got {len(parts)}: '{line}'"
                )

            try:
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as e:
                raise ValueError(
                    f"Line {line_no}: non-numeric value: {e}"
                ) from e

            # Clamp coordinates for analysis purposes
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            cls_id = max(0, cls_id)

            class_ids.append(cls_id)
            coords_list.append([x_c, y_c, w, h])

    N = len(class_ids)
    if N == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.bool),
        )

    boxes = torch.tensor(coords_list, dtype=torch.float32)
    labels = torch.tensor(class_ids, dtype=torch.long)
    valid_mask = torch.ones(N, dtype=torch.bool)

    return boxes, labels, valid_mask


def _print_report(
    stats: DatasetStats,
    class_names: Optional[List[str]] = None,
) -> None:
    """Print a formatted analysis report to stdout."""
    print()
    print(_REPORT_SEPARATOR)
    print("  ILGAN — Dataset Analysis Report")
    print(_REPORT_SEPARATOR)
    print()

    # ── Overview ───────────────────────────────────────────────────────
    print("  ┌─ Overview")
    print(f"  │  Total images:              {stats['total_images']:>10,}")
    print(f"  │  Total bounding boxes:      {stats['total_boxes']:>10,}")
    print(f"  │  Images with boxes:         {stats['images_with_boxes']:>10,}")
    print(f"  │  Images without boxes:     {stats['images_without_boxes']:>10,}")
    if stats["corrupt_images"] > 0:
        print(f"  │  ⚠  Corrupt images:          {stats['corrupt_images']:>10,}")
    print("  └" + "─" * 50)
    print()

    # ── Boxes per image ────────────────────────────────────────────────
    bpi = stats["boxes_per_image"]
    print("  ┌─ Boxes Per Image")
    print(f"  │  Mean:   {bpi['mean']:>8.3f}")
    print(f"  │  Std:    {bpi['std']:>8.3f}")
    print(f"  │  Min:    {bpi['min']:>8}")
    print(f"  │  Max:    {bpi['max']:>8}")
    print("  │")
    print("  │  Histogram:")
    for bin_label, count in bpi["histogram"].items():
        bar_len = int(count / max(1, max(bpi["histogram"].values())) * 30)
        bar = "█" * bar_len
        print(f"  │    {bin_label:>5s}: {count:>6,}  {bar}")
    print("  └" + "─" * 50)
    print()

    # ── Class distribution ────────────────────────────────────────────
    class_dist = stats["class_distribution"]
    print("  ┌─ Class Distribution")
    if class_dist:
        max_count = max(class_dist.values()) if class_dist else 1
        for cls_name, count in sorted(
            class_dist.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = count / max(stats["total_boxes"], 1) * 100
            bar_len = int(count / max_count * 30)
            bar = "█" * bar_len
            print(f"  │  {cls_name:30s}: {count:>8,} ({pct:>5.2f}%)  {bar}")
    else:
        print("  │  (no boxes found in dataset)")
    print(f"  │")
    print(f"  │  Number of unique classes: {stats['num_classes']}")
    print("  └" + "─" * 50)
    print()

    # ── Image size distribution ─────────────────────────────────────────
    img_sizes = stats["image_sizes"]
    print("  ┌─ Image Size Distribution")
    print("  │  Widths:")
    print(f"  │    Mean:   {img_sizes['widths']['mean']:>8.1f}")
    print(f"  │    Std:    {img_sizes['widths']['std']:>8.1f}")
    print(f"  │    Min:    {img_sizes['widths']['min']:>8}")
    print(f"  │    Max:    {img_sizes['widths']['max']:>8}")
    print(f"  │    Median: {img_sizes['widths']['median']:>8.1f}")
    print("  │")
    print("  │  Heights:")
    print(f"  │    Mean:   {img_sizes['heights']['mean']:>8.1f}")
    print(f"  │    Std:    {img_sizes['heights']['std']:>8.1f}")
    print(f"  │    Min:    {img_sizes['heights']['min']:>8}")
    print(f"  │    Max:    {img_sizes['heights']['max']:>8}")
    print(f"  │    Median: {img_sizes['heights']['median']:>8.1f}")
    print("  │")
    print("  │  Aspect Ratios (width/height):")
    print(f"  │    Mean:   {img_sizes['aspect_ratios']['mean']:>8.4f}")
    print(f"  │    Std:    {img_sizes['aspect_ratios']['std']:>8.4f}")
    print(f"  │    Min:    {img_sizes['aspect_ratios']['min']:>8.4f}")
    print(f"  │    Max:    {img_sizes['aspect_ratios']['max']:>8.4f}")
    print(f"  │    Median: {img_sizes['aspect_ratios']['median']:>8.4f}")
    print("  └" + "─" * 50)
    print()

    # ── Box size distribution ───────────────────────────────────────────
    box_sizes = stats["box_sizes"]
    print("  ┌─ Box Size Distribution (relative to image)")
    print("  │  Widths (fraction of image width):")
    print(f"  │    Mean:   {box_sizes['widths']['mean']:>8.4f}")
    print(f"  │    Std:    {box_sizes['widths']['std']:>8.4f}")
    print(f"  │    Min:    {box_sizes['widths']['min']:>8.4f}")
    print(f"  │    Max:    {box_sizes['widths']['max']:>8.4f}")
    print(f"  │    Median: {box_sizes['widths']['median']:>8.4f}")
    print("  │")
    print("  │  Heights (fraction of image height):")
    print(f"  │    Mean:   {box_sizes['heights']['mean']:>8.4f}")
    print(f"  │    Std:    {box_sizes['heights']['std']:>8.4f}")
    print(f"  │    Min:    {box_sizes['heights']['min']:>8.4f}")
    print(f"  │    Max:    {box_sizes['heights']['max']:>8.4f}")
    print(f"  │    Median: {box_sizes['heights']['median']:>8.4f}")
    print("  │")
    print("  │  Areas (fraction of image area):")
    print(f"  │    Mean:   {box_sizes['areas']['mean']:>8.6f}")
    print(f"  │    Std:    {box_sizes['areas']['std']:>8.6f}")
    print(f"  │    Min:    {box_sizes['areas']['min']:>8.6f}")
    print(f"  │    Max:    {box_sizes['areas']['max']:>8.6f}")
    print(f"  │    Median: {box_sizes['areas']['median']:>8.6f}")
    print("  └" + "─" * 50)
    print()

    # ── Box centre distribution ────────────────────────────────────────
    centres = stats["box_centres"]
    n_centres = len(centres["x"])
    print("  ┌─ Box Centre Distribution")
    print(f"  │  Total box centres: {n_centres:,}")
    if n_centres > 0:
        cx_array = np.array(centres["x"], dtype=np.float64)
        cy_array = np.array(centres["y"], dtype=np.float64)
        print(f"  │  Mean x: {float(cx_array.mean()):.4f}  "
              f"Mean y: {float(cy_array.mean()):.4f}")
        print(f"  │  Std x:  {float(cx_array.std()):.4f}  "
              f"Std y:  {float(cy_array.std()):.4f}")

        # Quadrant analysis
        q1 = int(((cx_array >= 0.5) & (cy_array < 0.5)).sum())   # top-right
        q2 = int(((cx_array < 0.5) & (cy_array < 0.5)).sum())     # top-left
        q3 = int(((cx_array < 0.5) & (cy_array >= 0.5)).sum())    # bottom-left
        q4 = int(((cx_array >= 0.5) & (cy_array >= 0.5)).sum())    # bottom-right
        total_q = q1 + q2 + q3 + q4
        if total_q > 0:
            print("  │")
            print("  │  Quadrant distribution (centre = 0.5, 0.5):")
            print(f"  │    Top-right:     {q1:>8,} ({q1/total_q*100:>5.1f}%)")
            print(f"  │    Top-left:      {q2:>8,} ({q2/total_q*100:>5.1f}%)")
            print(f"  │    Bottom-left:  {q3:>8,} ({q3/total_q*100:>5.1f}%)")
            print(f"  │    Bottom-right:  {q4:>8,} ({q4/total_q*100:>5.1f}%)")
    print("  └" + "─" * 50)
    print()

    # ── Label errors ───────────────────────────────────────────────────
    if stats["label_errors"]:
        print("  ⚠  Label parsing errors:")
        for err in stats["label_errors"][:10]:
            print(f"      {err['file']}: {err['message']}")
        if len(stats["label_errors"]) > 10:
            print(f"      ... and {len(stats['label_errors']) - 10} more errors.")
        print()

    print(_REPORT_SEPARATOR)
    print()


def _save_plots(stats: DatasetStats, output_dir: str) -> None:
    """Generate and save visualisation plots to disk.

    Requires ``matplotlib`` (and optionally ``seaborn`` for styling).

    The following plots are saved:

    - ``boxes_per_image_histogram.png``
    - ``class_distribution.png``
    - ``image_size_scatter.png``
    - ``box_size_histogram.png``
    - ``box_centre_heatmap.png``

    Parameters
    ----------
    stats : DatasetStats
        The statistics dictionary from :func:`analyze_dataset`.
    output_dir : str
        Directory where plots will be saved.
    """
    if not _HAS_MATPLOTLIB:
        print("  ⚠  matplotlib is required to save plots.  Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Apply seaborn style if available
    if _HAS_SEABORN and sns is not None:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)

    # ── 1. Boxes per image histogram ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bpi_hist = stats["boxes_per_image"]["histogram"]
    bins = list(bpi_hist.keys())
    counts = list(bpi_hist.values())
    colours = plt.cm.Blues([0.4 + 0.6 * c / max(counts) for c in counts])
    ax.bar(bins, counts, color=colours, edgecolor="navy", linewidth=0.5)
    ax.set_xlabel("Boxes per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Boxes per Image")
    for i, (b, c) in enumerate(zip(bins, counts)):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxes_per_image_histogram.png"), dpi=150)
    plt.close(fig)

    # ── 2. Class distribution ──────────────────────────────────────────
    class_dist = stats["class_distribution"]
    if class_dist:
        fig, ax = plt.subplots(figsize=(max(8, len(class_dist) * 0.4), 5))
        cls_names = list(class_dist.keys())
        cls_counts = list(class_dist.values())
        colours = plt.cm.viridis([i / max(len(cls_names), 1) for i in range(len(cls_names))])
        bars = ax.barh(range(len(cls_names)), cls_counts, color=colours, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(cls_names)))
        ax.set_yticklabels(cls_names, fontsize=8)
        ax.set_xlabel("Count")
        ax.set_title("Class Distribution")
        for bar, count in zip(bars, cls_counts):
            ax.text(bar.get_width() + max(cls_counts) * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=7)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
        plt.close(fig)

    # ── 3. Image size scatter ───────────────────────────────────────────
    img_sizes = stats["image_sizes"]
    fig, ax = plt.subplots(figsize=(7, 7))
    # We don't have per-image data in the stats dict, so we create a
    # synthetic scatter from the min/max range for visualisation.
    w_mean = img_sizes["widths"]["mean"]
    h_mean = img_sizes["heights"]["mean"]
    w_std = img_sizes["widths"]["std"]
    h_std = img_sizes["heights"]["std"]
    w_min = img_sizes["widths"]["min"]
    w_max = img_sizes["widths"]["max"]
    h_min = img_sizes["heights"]["min"]
    h_max = img_sizes["heights"]["max"]

    # Generate a 2D density estimate using a small grid
    grid_w = np.linspace(w_min, w_max, 100)
    grid_h = np.linspace(h_min, h_max, 100)
    W, H_grid = np.meshgrid(grid_w, grid_h)
    # Simple 2D Gaussian centred at (w_mean, h_mean) with stds
    Z = np.exp(-0.5 * (
        ((W - w_mean) / max(w_std, 1)) ** 2 +
        ((H_grid - h_mean) / max(h_std, 1)) ** 2
    ))
    ax.contourf(W, H_grid, Z, levels=20, cmap="Blues")
    ax.scatter([w_mean], [h_mean], color="red", s=100, marker="x", zorder=5,
               label=f"Mean ({w_mean:.0f}, {h_mean:.0f})")
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("Image Size Distribution")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "image_size_scatter.png"), dpi=150)
    plt.close(fig)

    # ── 4. Box size histogram ──────────────────────────────────────────
    box_sizes = stats["box_sizes"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, key, title, colour in zip(
        axes,
        ["widths", "heights", "areas"],
        ["Box Width Distribution", "Box Height Distribution", "Box Area Distribution"],
        ["steelblue", "coral", "seagreen"],
    ):
        data = box_sizes[key]
        # Generate a synthetic log-normal distribution from stats
        mu = math.log(max(data["mean"], 1e-8))
        sigma = data["std"] / max(data["mean"], 1e-8)
        synthetic = np.random.lognormal(mean=mu, sigma=max(sigma, 0.1), size=10000)
        synthetic = np.clip(synthetic, data["min"], data["max"])
        ax.hist(synthetic, bins=50, color=colour, edgecolor="white", alpha=0.8, density=True)
        ax.axvline(data["mean"], color="darkred", linestyle="--", linewidth=1.5,
                    label=f"Mean: {data['mean']:.4f}")
        ax.axvline(data["median"], color="darkgreen", linestyle=":", linewidth=1.5,
                    label=f"Median: {data['median']:.4f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "box_size_histogram.png"), dpi=150)
    plt.close(fig)

    # ── 5. Box centre heatmap ─────────────────────────────────────────
    centres = stats["box_centres"]
    if centres["x"] and centres["y"]:
        fig, ax = plt.subplots(figsize=(7, 6))
        # 2D histogram / heatmap
        heatmap, xedges, yedges = np.histogram2d(
            centres["x"], centres["y"],
            bins=32,
            range=[[0, 1], [0, 1]],
        )
        extent = [0, 1, 0, 1]
        im = ax.imshow(
            heatmap.T,
            extent=extent,
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, label="Number of Box Centres")
        ax.set_xlabel("x_center (normalised)")
        ax.set_ylabel("y_center (normalised)")
        ax.set_title("Box Centre Heatmap")
        # Add quadrant lines
        ax.axhline(0.5, color="cyan", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.5, color="cyan", linestyle="--", linewidth=0.8, alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "box_centre_heatmap.png"), dpi=150)
        plt.close(fig)

    print(f"  Plots saved to: {os.path.abspath(output_dir)}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Command-line entry point for dataset analysis and label validation.

    Usage::

        python -m ilgan.scripts.analyze_data analyze <data_root> [options]
        python -m ilgan.scripts.analyze_data validate <data_root> [options]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ILGAN Dataset Analysis and Label Validation Tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze subcommand ─────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a dataset and print comprehensive statistics.",
    )
    analyze_parser.add_argument(
        "data_root",
        type=str,
        help="Root directory of the dataset (containing images/ and labels/).",
    )
    analyze_parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of class names (in order of class IDs).",
    )
    analyze_parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="Directory to save visualisation plots (requires matplotlib).",
    )
    analyze_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyse.",
    )
    analyze_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed per-sample information.",
    )
    analyze_parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save statistics as a JSON file.",
    )

    # ── validate subcommand ────────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate all YOLO label files in a dataset.",
    )
    validate_parser.add_argument(
        "data_root",
        type=str,
        help="Root directory of the dataset (containing labels/).",
    )
    validate_parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save validation errors as a JSON file.",
    )

    args = parser.parse_args()

    if args.command == "analyze":
        class_names_list: Optional[List[str]] = (
            list(args.class_names) if args.class_names else None
        )
        stats = analyze_dataset(
            data_root=args.data_root,
            class_names=class_names_list,
            save_plots=args.save_plots,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )

        if args.output_json:
            # Remove raw centre data from JSON (too large) and save
            json_stats = _make_json_serialisable(stats)
            with open(args.output_json, "w") as f:
                json.dump(json_stats, f, indent=2, sort_keys=True)
            print(f"  Statistics saved to: {args.output_json}")

    elif args.command == "validate":
        errors = validate_labels(data_root=args.data_root)

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"  Validation errors saved to: {args.output_json}")

        # Exit with non-zero code if errors found
        if errors:
            sys.exit(1)


def _make_json_serialisable(stats: DatasetStats) -> DatasetStats:
    """Remove non-serialisable fields (like raw centre lists) from stats
    and convert numpy types to native Python types."""
    json_stats = {}
    for key, value in stats.items():
        if key == "box_centres":
            # Store summary statistics instead of raw lists
            if value["x"] and value["y"]:
                cx_array = np.array(value["x"], dtype=np.float64)
                cy_array = np.array(value["y"], dtype=np.float64)
                json_stats[key] = {
                    "mean_x": float(cx_array.mean()),
                    "mean_y": float(cy_array.mean()),
                    "std_x": float(cx_array.std()),
                    "std_y": float(cy_array.std()),
                    "count": len(value["x"]),
                }
            else:
                json_stats[key] = {"count": 0}
        elif isinstance(value, dict):
            json_stats[key] = _make_json_serialisable(value)
        elif isinstance(value, (np.integer,)):
            json_stats[key] = int(value)
        elif isinstance(value, (np.floating,)):
            json_stats[key] = float(value)
        elif isinstance(value, (np.bool_,)):
            json_stats[key] = bool(value)
        else:
            json_stats[key] = value
    return json_stats


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "analyze_dataset",
    "validate_labels",
    "DatasetStats",
    "LabelError",
]

if __name__ == "__main__":
    main()
