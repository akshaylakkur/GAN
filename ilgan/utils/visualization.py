"""
Visualization utilities for ILGAN — drawing bounding boxes, creating image
grids, saving sample outputs, and plotting training curves.

This module provides five core functions:

1. :func:`draw_boxes_on_image` — Draw bounding boxes with labels and
   confidence scores on a PIL image.
2. :func:`make_grid` — Create a grid of images from a batch (wraps
   ``torchvision.utils.make_grid``).
3. :func:`save_image_grid` — Save a grid of images to disk.
4. :func:`save_sample_outputs` — Save generated/real images with boxes
   drawn, plus YOLO-format label files for generated boxes.
5. :func:`plot_loss_curves` — Plot training loss curves using matplotlib.

All functions are designed to be memory-efficient (operate on CPU tensors
or PIL images) and safe to call during training without blocking the GPU
pipeline.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless training
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from torchvision.utils import make_grid as tv_make_grid
from torchvision.utils import save_image as tv_save_image

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_COLORMAP: List[Tuple[int, int, int]] = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky Blue
    (128, 255, 0),    # Lime
    (255, 0, 128),    # Pink
    (0, 255, 128),    # Spring Green
    (128, 128, 255),  # Light Purple
    (255, 128, 128),  # Light Red
    (128, 255, 128),  # Light Green
    (128, 128, 128),  # Gray
]
"""A 16-colour palette for bounding box visualisation.  Class IDs beyond 15
wrap around modulo 16."""

_DEFAULT_FONT_SIZE: int = 12
"""Default font size for label text on bounding boxes."""

_BOX_OUTLINE_WIDTH: int = 2
"""Width of the bounding box rectangle outline in pixels."""

_LABEL_BG_PADDING: int = 2
"""Padding (in pixels) around the label text background."""


# ──────────────────────────────────────────────────────────────────────────────
# 1. draw_boxes_on_image
# ──────────────────────────────────────────────────────────────────────────────


def draw_boxes_on_image(
    image_tensor: torch.Tensor,
    boxes: torch.Tensor,
    confidences: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.5,
) -> Image.Image:
    """Draw bounding boxes on a PIL image from a normalised image tensor.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Image tensor of shape ``[3, H, W]`` with pixel values in ``[-1, 1]``.
    boxes : torch.Tensor
        Bounding box tensor of shape ``[N, 4]`` in YOLO format
        ``(cx, cy, w, h)``, with all values normalised to ``[0, 1]``
        relative to the image dimensions.
    confidences : torch.Tensor, optional
        Confidence scores of shape ``[N]`` or ``[N, 1]``.  Values in
        ``(0, 1)``.  If ``None``, no confidence text is drawn.
    labels : torch.Tensor, optional
        Integer class IDs of shape ``[N]``.  If ``None``, no class label
        text is drawn.
    class_names : list of str, optional
        Human-readable class names indexed by class ID.  If ``None``,
        class IDs are displayed as integers.
    confidence_threshold : float, optional
        Minimum confidence score for a box to be drawn.  Boxes with
        confidence below this threshold are skipped.  Default ``0.5``.

    Returns
    -------
    PIL.Image.Image
        A PIL image (RGB) with bounding boxes, labels, and confidence
        scores drawn on it.

    Notes
    -----
    - The input tensor is converted from ``[-1, 1]`` to ``[0, 255]``
      uint8 for PIL rendering.
    - Boxes with all coordinates equal to ``-1.0`` (padding sentinels)
      are automatically skipped.
    - Each class gets a distinct colour from the internal 16-colour
      palette (wraps around for >16 classes).
    - If a font cannot be loaded, a default PIL font is used (may be
      smaller on some systems).
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if image_tensor.dim() != 3 or image_tensor.size(0) != 3:
        raise ValueError(
            f"image_tensor must be [3, H, W], got {tuple(image_tensor.shape)}"
        )
    if boxes.dim() != 2 or boxes.size(1) != 4:
        raise ValueError(
            f"boxes must be [N, 4], got {tuple(boxes.shape)}"
        )

    N = boxes.size(0)

    if confidences is not None:
        if confidences.dim() == 2 and confidences.size(1) == 1:
            confidences = confidences.squeeze(1)  # [N, 1] -> [N]
        if confidences.dim() != 1 or confidences.size(0) != N:
            raise ValueError(
                f"confidences must be [N] or [N, 1] with N={N}, "
                f"got {tuple(confidences.shape)}"
            )

    if labels is not None:
        if labels.dim() != 1 or labels.size(0) != N:
            raise ValueError(
                f"labels must be [N] with N={N}, got {tuple(labels.shape)}"
            )

    # ── Convert tensor to PIL image ───────────────────────────────────────
    # Move to CPU, convert from [-1, 1] to [0, 255] uint8
    img_cpu = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    img_cpu = (img_cpu + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    img_cpu = (img_cpu * 255.0).byte()  # [0, 1] -> [0, 255]
    img_np = img_cpu.permute(1, 2, 0).numpy()  # [3, H, W] -> [H, W, 3]
    pil_image = Image.fromarray(img_np, mode="RGB")

    H, W = pil_image.size[1], pil_image.size[0]

    # ── Prepare drawing context ───────────────────────────────────────────
    draw = ImageDraw.Draw(pil_image)

    # Try to load a TrueType font; fall back to default if unavailable
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.isfile(font_path):
            try:
                font = ImageFont.truetype(font_path, size=_DEFAULT_FONT_SIZE)
                break
            except (IOError, OSError):
                continue
    if font is None:
        font = ImageFont.load_default()

    # ── Draw each box ─────────────────────────────────────────────────────
    for i in range(N):
        # Skip padding sentinel boxes (all -1.0)
        if torch.all(boxes[i] < -0.5):
            continue

        # Skip low-confidence boxes
        if confidences is not None and confidences[i].item() < confidence_threshold:
            continue

        # Convert from (cx, cy, w, h) normalised to absolute pixel coords
        cx, cy, bw, bh = boxes[i].tolist()
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        # Clamp to image boundaries
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))

        # Skip degenerate boxes (zero area after clamping)
        if x2 <= x1 or y2 <= y1:
            continue

        # Determine colour
        if labels is not None:
            cls_id = int(labels[i].item())
            colour = _COLORMAP[cls_id % len(_COLORMAP)]
        else:
            colour = _COLORMAP[0]  # Default red

        # Draw rectangle outline
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=colour,
            width=_BOX_OUTLINE_WIDTH,
        )

        # Build label text
        label_parts: List[str] = []
        if labels is not None:
            cls_id = int(labels[i].item())
            if class_names is not None and cls_id < len(class_names):
                label_parts.append(class_names[cls_id])
            else:
                label_parts.append(str(cls_id))
        if confidences is not None:
            label_parts.append(f"{confidences[i].item():.2f}")

        if label_parts:
            label_text = ": ".join(label_parts)

            # Measure text bounding box
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Draw label background
            label_bg_x1 = x1
            label_bg_y1 = y1 - text_h - 2 * _LABEL_BG_PADDING
            label_bg_x2 = x1 + text_w + 2 * _LABEL_BG_PADDING
            label_bg_y2 = y1

            # If label would go above the image, place it below the box
            if label_bg_y1 < 0:
                label_bg_y1 = y2
                label_bg_y2 = y2 + text_h + 2 * _LABEL_BG_PADDING

            draw.rectangle(
                [label_bg_x1, label_bg_y1, label_bg_x2, label_bg_y2],
                fill=colour,
            )

            # Draw label text in white
            text_x = label_bg_x1 + _LABEL_BG_PADDING
            text_y = label_bg_y1 + _LABEL_BG_PADDING
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)

    return pil_image


# ──────────────────────────────────────────────────────────────────────────────
# 2. make_grid
# ──────────────────────────────────────────────────────────────────────────────


def make_grid(
    samples: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Create a grid of images from a batch tensor.

    This is a thin wrapper around ``torchvision.utils.make_grid`` that
    provides sensible defaults for ILGAN (images in ``[-1, 1]`` range).

    Parameters
    ----------
    samples : torch.Tensor
        Batch of images of shape ``[B, C, H, W]``.  Values are expected
        in ``[-1, 1]`` (ILGAN default) unless ``value_range`` is
        specified.
    nrow : int, optional
        Number of images per row in the grid.  Default ``8``.
    padding : int, optional
        Padding (in pixels) between images in the grid.  Default ``2``.
    normalize : bool, optional
        If ``True`` (default), scale pixel values to ``[0, 1]`` for
        proper display.  Set to ``False`` if the images are already in
        ``[0, 1]``.
    value_range : tuple of (float, float), optional
        The min/max range of the input pixel values.  If ``None``,
        inferred from the data (assumes ``[-1, 1]`` if any value is
        negative, otherwise ``[0, 1]``).

    Returns
    -------
    torch.Tensor
        A grid image of shape ``[3, grid_H, grid_W]`` with values in
        ``[0, 1]`` (if ``normalize=True``) or the original range.

    Notes
    -----
    - The grid is returned as a CPU tensor.
    - If the batch size is smaller than ``nrow``, the grid will have a
      single row.
    """
    # Determine value range if not provided
    if value_range is None and normalize:
        samples_cpu = samples.detach().cpu()
        if samples_cpu.min() < 0:
            value_range = (-1.0, 1.0)
        else:
            value_range = (0.0, 1.0)

    grid = tv_make_grid(
        samples.detach().cpu(),
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )

    return grid


# ──────────────────────────────────────────────────────────────────────────────
# 3. save_image_grid
# ──────────────────────────────────────────────────────────────────────────────


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> str:
    """Save a grid of images to disk.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images of shape ``[B, C, H, W]``.  Values in ``[-1, 1]``
        (ILGAN default) unless ``value_range`` is specified.
    path : str
        Filesystem path where the grid image will be saved.  Parent
        directories are created if they do not exist.
    nrow : int, optional
        Number of images per row in the grid.  Default ``8``.
    padding : int, optional
        Padding (in pixels) between images.  Default ``2``.
    normalize : bool, optional
        If ``True`` (default), scale pixel values to ``[0, 1]``.
    value_range : tuple of (float, float), optional
        The min/max range of the input pixel values.  If ``None``,
        inferred from the data.

    Returns
    -------
    str
        The absolute path to the saved grid image.

    Notes
    -----
    - The image is saved as a PNG file.
    - If the file already exists, it is overwritten.
    """
    # Create parent directory if needed
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Determine value range if not provided
    if value_range is None and normalize:
        images_cpu = images.detach().cpu()
        if images_cpu.min() < 0:
            value_range = (-1.0, 1.0)
        else:
            value_range = (0.0, 1.0)

    # Create the grid
    grid = tv_make_grid(
        images.detach().cpu(),
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )

    # Save to disk
    tv_save_image(grid, path)

    return os.path.abspath(path)


# ──────────────────────────────────────────────────────────────────────────────
# 4. save_sample_outputs
# ──────────────────────────────────────────────────────────────────────────────


def save_sample_outputs(
    generator_outputs: Dict[str, torch.Tensor],
    batch: Any,
    output_dir: str,
    step: int,
    class_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.5,
    max_samples: int = 16,
) -> Dict[str, str]:
    """Save generated and real sample outputs for visual inspection.

    This function saves:

    1. A grid of generated images (``gen_grid_<step>.png``).
    2. A grid of real images for comparison (``real_grid_<step>.png``).
    3. Generated images with predicted boxes drawn on them
       (``gen_boxes_<step>.png``).
    4. Real images with ground-truth boxes drawn on them
       (``real_boxes_<step>.png``).
    5. YOLO-format label files for the generated boxes
       (``gen_labels_<step>_<batch_idx>.txt``).

    Parameters
    ----------
    generator_outputs : dict of str -> torch.Tensor
        Output dictionary from the ILGAN generator.  Must contain keys:

        - ``"image"``: generated images, shape ``[B, 3, H, W]``, values
          in ``[-1, 1]``.
        - ``"boxes"``: predicted boxes, shape ``[B, N, 4]`` in
          ``(cx, cy, w, h)`` format, values in ``[0, 1]``.
        - ``"class_logits"``: class logits, shape ``[B, N, num_classes]``.
        - ``"confidences"``: confidence scores, shape ``[B, N, 1]``,
          values in ``[0, 1]``.

    batch : Batch or dict
        A batch of real data.  If a :class:`ilgan.data.structures.Batch`,
        it must have attributes ``images``, ``boxes``, ``labels``,
        ``valid_mask``.  If a dict, it must have keys ``"images"``,
        ``"boxes"``, ``"labels"``, ``"valid_mask"``.
    output_dir : str
        Directory where output files will be saved.  Created if it does
        not exist.
    step : int
        Global training step (used in filenames for disambiguation).
    class_names : list of str, optional
        Human-readable class names indexed by class ID.  If ``None``,
        class IDs are displayed as integers.
    confidence_threshold : float, optional
        Minimum confidence for drawing a predicted box.  Default ``0.5``.
    max_samples : int, optional
        Maximum number of samples to visualise (to avoid excessive I/O).
        Default ``16``.

    Returns
    -------
    dict of str -> str
        A dictionary mapping logical names to absolute file paths:

        - ``"gen_grid"``: path to the generated image grid.
        - ``"real_grid"``: path to the real image grid.
        - ``"gen_boxes"``: path to the generated image with boxes.
        - ``"real_boxes"``: path to the real image with boxes.
        - ``"gen_labels_dir"``: path to the directory containing YOLO
          label files.

    Notes
    -----
    - All images are saved as PNG files.
    - YOLO label files are saved in a subdirectory ``gen_labels`` inside
      ``output_dir``.
    - The function is safe to call during training (operates on CPU).
    """
    # ── Create output directory ───────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "gen_labels")
    os.makedirs(labels_dir, exist_ok=True)

    # ── Extract tensors ───────────────────────────────────────────────────
    gen_images = generator_outputs["image"].detach().cpu()  # [B, 3, H, W]
    gen_boxes = generator_outputs["boxes"].detach().cpu()    # [B, N, 4]
    gen_logits = generator_outputs["class_logits"].detach().cpu()  # [B, N, C]
    gen_confidences = generator_outputs["confidences"].detach().cpu()  # [B, N, 1]

    # Predicted class labels (argmax over logits)
    gen_labels = gen_logits.argmax(dim=-1)  # [B, N]

    # Real data
    if hasattr(batch, "images"):
        real_images = batch.images.detach().cpu()
        real_boxes = batch.boxes.detach().cpu()
        real_labels = batch.labels.detach().cpu()
        real_valid = batch.valid_mask.detach().cpu()
    elif isinstance(batch, dict):
        real_images = batch["images"].detach().cpu()
        real_boxes = batch["boxes"].detach().cpu()
        real_labels = batch["labels"].detach().cpu()
        real_valid = batch["valid_mask"].detach().cpu()
    else:
        raise TypeError(
            f"batch must be a Batch object or a dict, got {type(batch)}"
        )

    B = min(gen_images.size(0), max_samples)

    # ── 1. Save generated image grid ─────────────────────────────────────
    gen_grid_path = os.path.join(output_dir, f"gen_grid_{step:08d}.png")
    save_image_grid(gen_images[:B], gen_grid_path, nrow=int(math.sqrt(B)))

    # ── 2. Save real image grid ───────────────────────────────────────────
    real_grid_path = os.path.join(output_dir, f"real_grid_{step:08d}.png")
    save_image_grid(real_images[:B], real_grid_path, nrow=int(math.sqrt(B)))

    # ── 3. Save generated images with boxes drawn ─────────────────────────
    gen_boxes_path = os.path.join(output_dir, f"gen_boxes_{step:08d}.png")
    _save_annotated_grid(
        images=gen_images[:B],
        boxes=gen_boxes[:B],
        confidences=gen_confidences[:B],
        labels=gen_labels[:B],
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        save_path=gen_boxes_path,
    )

    # ── 4. Save real images with ground-truth boxes drawn ────────────────
    real_boxes_path = os.path.join(output_dir, f"real_boxes_{step:08d}.png")
    _save_annotated_grid(
        images=real_images[:B],
        boxes=real_boxes[:B],
        confidences=None,
        labels=real_labels[:B],
        class_names=class_names,
        confidence_threshold=0.0,  # Show all ground-truth boxes
        valid_mask=real_valid[:B] if real_valid is not None else None,
        save_path=real_boxes_path,
    )

    # ── 5. Save YOLO-format label files for generated boxes ──────────────
    for b in range(B):
        label_path = os.path.join(
            labels_dir, f"gen_labels_{step:08d}_{b:04d}.txt"
        )
        _save_yolo_labels(
            boxes=gen_boxes[b],
            class_ids=gen_labels[b],
            confidences=gen_confidences[b],
            confidence_threshold=confidence_threshold,
            save_path=label_path,
        )

    return {
        "gen_grid": os.path.abspath(gen_grid_path),
        "real_grid": os.path.abspath(real_grid_path),
        "gen_boxes": os.path.abspath(gen_boxes_path),
        "real_boxes": os.path.abspath(real_boxes_path),
        "gen_labels_dir": os.path.abspath(labels_dir),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. plot_loss_curves
# ──────────────────────────────────────────────────────────────────────────────


def plot_loss_curves(
    loss_history: Dict[str, List[float]],
    save_path: str,
    title: str = "ILGAN Training Loss Curves",
    dpi: int = 150,
    figsize: Tuple[int, int] = (14, 10),
) -> str:
    """Plot training loss curves from a loss history dictionary.

    This function creates a multi-panel figure where each subplot shows
    one or more related loss curves over training steps.  The layout is
    organised as follows:

    - **Top-left**: Generator losses (all keys containing ``"g_"`` or
      ``"gen_"`` or ``"generator"``).
    - **Top-right**: Discriminator losses (all keys containing ``"d_"``
      or ``"disc_"`` or ``"discriminator"``).
    - **Middle-left**: Box-related losses (keys containing ``"box"``,
      ``"bbox"``, ``"regression"``).
    - **Middle-right**: Collapse prevention and consistency losses
      (keys containing ``"collapse"``, ``"consistency"``, ``"diversity"``).
    - **Bottom**: All losses overlaid (for comparing scales).

    Parameters
    ----------
    loss_history : dict of str -> list of float
        A dictionary mapping loss names to lists of scalar values recorded
        at each training step.  All lists must have the same length.
    save_path : str
        Filesystem path where the plot will be saved (e.g.
        ``"loss_curves.png"``).  Parent directories are created if they
        do not exist.
    title : str, optional
        Overall figure title.  Default ``"ILGAN Training Loss Curves"``.
    dpi : int, optional
        Figure resolution in dots per inch.  Default ``150``.
    figsize : tuple of (int, int), optional
        Figure dimensions in inches ``(width, height)``.  Default
        ``(14, 10)``.

    Returns
    -------
    str
        The absolute path to the saved plot image.

    Notes
    -----
    - The plot is saved as a PNG file.
    - If ``loss_history`` is empty, a warning message is written to the
      plot instead of raising an error.
    - All curves are smoothed with a simple moving average (window size
      = max(1, len // 50)) for readability.
    - The legend is placed outside the plot area to avoid occlusion.
    """
    # Create parent directory if needed
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # ── Handle empty history ──────────────────────────────────────────────
    if not loss_history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "No loss history available.",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title(title)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(save_path)

    # ── Validate that all lists have the same length ──────────────────────
    lengths = [len(v) for v in loss_history.values()]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All loss history lists must have the same length. "
            f"Got lengths: {dict(zip(loss_history.keys(), lengths))}"
        )

    num_steps = lengths[0] if lengths else 0
    steps = list(range(1, num_steps + 1))

    # ── Categorise loss keys ──────────────────────────────────────────────
    def _categorise(key: str) -> str:
        """Categorise a loss key into a group for subplot placement."""
        key_lower = key.lower()
        if any(x in key_lower for x in ("g_", "gen_", "generator", "g_loss")):
            return "generator"
        if any(x in key_lower for x in ("d_", "disc_", "discriminator", "d_loss")):
            return "discriminator"
        if any(x in key_lower for x in ("box", "bbox", "regression", "giou", "iou")):
            return "box"
        if any(x in key_lower for x in ("collapse", "consistency", "diversity", "repulsion")):
            return "collapse"
        if any(x in key_lower for x in ("adv", "adversarial", "wgan", "gp", "gradient_penalty")):
            return "adversarial"
        if any(x in key_lower for x in ("cls", "class", "classification")):
            return "classification"
        if any(x in key_lower for x in ("conf", "objectness")):
            return "confidence"
        return "other"

    categories: Dict[str, Dict[str, List[float]]] = {
        "generator": {},
        "discriminator": {},
        "box": {},
        "collapse": {},
        "adversarial": {},
        "classification": {},
        "confidence": {},
        "other": {},
    }

    for key, values in loss_history.items():
        cat = _categorise(key)
        categories[cat][key] = values

    # ── Determine subplot layout ──────────────────────────────────────────
    # We create subplots for non-empty categories, up to 6 panels.
    non_empty_cats = {k: v for k, v in categories.items() if v}
    num_panels = len(non_empty_cats)

    if num_panels == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "No categorised loss data to plot.",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title(title)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return os.path.abspath(save_path)

    # Layout: 2 columns, enough rows
    ncols = 2
    nrows = (num_panels + 1) // 2  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # ── Smoothing helper ──────────────────────────────────────────────────
    def _smooth(values: List[float], window: int) -> np.ndarray:
        """Apply a simple moving average to *values*."""
        if window <= 1 or len(values) <= window:
            return np.array(values)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    window = max(1, num_steps // 50)

    # ── Plot each category ────────────────────────────────────────────────
    panel_idx = 0
    cat_order = [
        "generator", "discriminator", "adversarial", "box",
        "collapse", "classification", "confidence", "other",
    ]

    for cat_name in cat_order:
        cat_dict = non_empty_cats.get(cat_name)
        if cat_dict is None or not cat_dict:
            continue

        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row, col]

        for key, values in cat_dict.items():
            smoothed = _smooth(values, window)
            # Adjust x-axis to account for convolution offset
            x = np.arange(1, len(smoothed) + 1)
            if len(smoothed) < len(values):
                # Offset to align end points
                offset = len(values) - len(smoothed)
                x = np.arange(1 + offset, len(values) + 1)
            ax.plot(x, smoothed, label=key, linewidth=1.5)

        ax.set_title(cat_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            framealpha=0.8,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        panel_idx += 1

    # ── Hide unused subplots ──────────────────────────────────────────────
    for idx in range(panel_idx, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    # ── Adjust layout and save ────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _save_annotated_grid(
    images: torch.Tensor,
    boxes: torch.Tensor,
    confidences: Optional[torch.Tensor],
    labels: torch.Tensor,
    class_names: Optional[List[str]],
    confidence_threshold: float,
    save_path: str,
    valid_mask: Optional[torch.Tensor] = None,
) -> str:
    """Draw boxes on each image in a batch, tile them into a grid, and save.

    Parameters
    ----------
    images : torch.Tensor
        Shape ``[B, 3, H, W]``, values in ``[-1, 1]``.
    boxes : torch.Tensor
        Shape ``[B, N, 4]`` in ``(cx, cy, w, h)`` format.
    confidences : torch.Tensor, optional
        Shape ``[B, N]`` or ``[B, N, 1]``.
    labels : torch.Tensor
        Shape ``[B, N]``, integer class IDs.
    class_names : list of str, optional
    confidence_threshold : float
    save_path : str
    valid_mask : torch.Tensor, optional
        Shape ``[B, N]``, bool.  If provided, only boxes with ``True``
        in the mask are drawn.

    Returns
    -------
    str
        Absolute path to the saved grid image.
    """
    B = images.size(0)
    annotated_pils: List[Image.Image] = []

    for b in range(B):
        # Determine which boxes to draw
        if valid_mask is not None:
            # Mask out invalid boxes by setting them to sentinel values
            box_b = boxes[b].clone()
            box_b[~valid_mask[b]] = -1.0
            label_b = labels[b].clone()
            label_b[~valid_mask[b]] = -1
        else:
            box_b = boxes[b]
            label_b = labels[b]

        conf_b = confidences[b] if confidences is not None else None

        pil_img = draw_boxes_on_image(
            image_tensor=images[b],
            boxes=box_b,
            confidences=conf_b,
            labels=label_b,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
        )
        annotated_pils.append(pil_img)

    # Convert PIL images to tensors and create a grid
    annotated_tensors: List[torch.Tensor] = []
    for pil_img in annotated_pils:
        # Convert PIL to numpy [H, W, C] then to tensor [C, H, W]
        np_img = np.array(pil_img, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1)  # [3, H, W]
        annotated_tensors.append(tensor_img)

    # Stack into a batch
    batch_tensor = torch.stack(annotated_tensors, dim=0)  # [B, 3, H, W]

    # Determine nrow for the grid
    nrow = int(math.ceil(math.sqrt(B)))

    # Save the grid
    return save_image_grid(
        batch_tensor,
        path=save_path,
        nrow=nrow,
        padding=2,
        normalize=False,  # Already in [0, 1]
    )


def _save_yolo_labels(
    boxes: torch.Tensor,
    class_ids: torch.Tensor,
    confidences: torch.Tensor,
    confidence_threshold: float,
    save_path: str,
) -> str:
    """Save predicted boxes in YOLO-format label file.

    Each line in the output file follows the format::

        class_id x_center y_center width_height confidence

    where all coordinate values are in ``[0, 1]`` (normalised relative to
    image dimensions).

    Parameters
    ----------
    boxes : torch.Tensor
        Shape ``[N, 4]`` in ``(cx, cy, w, h)`` format.
    class_ids : torch.Tensor
        Shape ``[N]``, integer class IDs.
    confidences : torch.Tensor
        Shape ``[N]`` or ``[N, 1]``.
    confidence_threshold : float
        Minimum confidence for a box to be included.
    save_path : str
        Path to the output ``.txt`` file.

    Returns
    -------
    str
        Absolute path to the saved label file.
    """
    # Squeeze confidences if needed
    if confidences.dim() == 2 and confidences.size(1) == 1:
        confidences = confidences.squeeze(1)

    N = boxes.size(0)
    lines: List[str] = []

    for i in range(N):
        # Skip padding sentinel boxes
        if torch.all(boxes[i] < -0.5):
            continue

        # Skip low-confidence boxes
        if confidences[i].item() < confidence_threshold:
            continue

        cx, cy, w, h = boxes[i].tolist()
        cls_id = int(class_ids[i].item())
        conf = confidences[i].item()

        # Clamp coordinates to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        # Skip degenerate boxes
        if w <= 0.0 or h <= 0.0:
            continue

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")

    # Write to file
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")  # Trailing newline

    return os.path.abspath(save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "draw_boxes_on_image",
    "make_grid",
    "save_image_grid",
    "save_sample_outputs",
    "plot_loss_curves",
]
