"""
Validation epoch for the ILGAN dual-output GAN.

This module implements the validation loop that evaluates the generator and
discriminator on a held-out validation set.  It computes all losses (for
monitoring, not backprop), updates image and box metrics, computes the
"realism gap" :math:`\\mathbb{E}[D(\\text{real})] - \\mathbb{E}[D(\\text{fake})]`,
and generates a fixed grid of sample images for visual inspection.

Mathematical overview
---------------------
The validation loop proceeds as follows for each batch:

1. **Sample latents**: :math:`z \\sim \\mathcal{N}(0, I)^{B \\times D}`

2. **Generator forward**: :math:`(I_{fake}, B_{pred}, C_{pred}, \\text{conf}) = G(z)`

3. **Discriminator forward** (real and fake):
   :math:`(s_{local}^{real}, s_{global}^{real}) = D(I_{real})`
   :math:`(s_{local}^{fake}, s_{global}^{fake}) = D(I_{fake})`

4. **Loss computation** (all losses, no backprop):
   :math:`\\mathcal{L}_{total} = \\mathcal{L}_{adv} + \\mathcal{L}_{box} + \\mathcal{L}_{cls} + \\mathcal{L}_{conf} + \\mathcal{L}_{collapse} + \\mathcal{L}_{consistency}`

5. **Realism gap**: :math:`\\Delta = \\mathbb{E}[D_{global}(I_{real})] - \\mathbb{E}[D_{global}(I_{fake})]`

   A positive realism gap indicates the discriminator can distinguish real
   from fake on average.  As the generator improves, this gap should shrink
   toward zero.

6. **Metrics update**: Image metrics (FID, IS) and box metrics (mAP, GIoU)
   are accumulated across the validation set.

7. **Sample grid**: A fixed set of latent vectors is used to generate a
   consistent grid of sample images for visual inspection across epochs.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

from ilgan.data.structures import Batch
from ilgan.losses import LossAggregator
from ilgan.metrics.joint_metrics import MetricsTracker
from ilgan.utils.config import Config
from ilgan.utils.logger import Logger
from ilgan.utils.visualization import draw_boxes_on_image  # noqa: F811
from ilgan.data.streaming_voc import VOC_CLASSES

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_EPS: float = 1e-8
"""Small epsilon for numerical stability."""

_DEFAULT_NUM_SAMPLE_GRID: int = 16
"""Default number of samples in the fixed sample grid (must be a perfect
square for ``make_grid``)."""

_DEFAULT_SAMPLE_GRID_SIZE: int = 8
"""Default spatial size of each sample in the grid (rows = cols = sqrt)."""


# ──────────────────────────────────────────────────────────────────────────────
# Validation Epoch
# ──────────────────────────────────────────────────────────────────────────────


def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    image_encoder: nn.Module,
    box_encoder: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_aggregator: LossAggregator,
    metrics_tracker: MetricsTracker,
    epoch: int,
    config: Config,
    logger: Logger,
) -> Dict[str, Any]:
    r"""Run one full validation epoch for the ILGAN dual-output GAN.

    This function evaluates the current state of the generator and
    discriminator on the validation set **without** updating any parameters.
    It computes all losses for monitoring, updates image and box metrics,
    computes the realism gap, and generates a fixed grid of sample images.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator (``ILGANGenerator``).  Its ``forward`` must
        accept a ``[B, latent_dim]`` tensor and return a dict with keys
        ``"image"``, ``"boxes"``, ``"class_logits"``, ``"confidences"``,
        and ``"aux"`` (containing ``"attention_maps"`` and
        ``"skip_features"``).
    discriminator : nn.Module
        The ILGAN discriminator (``ImageDiscriminator``).  Its ``forward``
        must accept a ``[B, 3, H, W]`` tensor and return a tuple
        ``(local_scores, global_score)``.
    image_encoder : nn.Module
        The ``ImageFeatureEncoder`` for cross-modal consistency.  Maps
        images to a shared feature space.
    box_encoder : nn.Module
        The ``BoxFeatureEncoder`` for cross-modal consistency.  Maps
        bounding boxes to a shared feature space.
    val_loader : torch.utils.data.DataLoader
        DataLoader yielding :class:`~ilgan.data.structures.Batch` objects
        for the validation set.
    loss_aggregator : LossAggregator
        The central loss aggregator.  Provides ``__call__`` for computing
        all losses.
    metrics_tracker : MetricsTracker
        Stateful metrics accumulator.  Will be **reset** at the start of
        this function.  After validation, the caller can use
        ``metrics_tracker.compute_all()`` or
        ``metrics_tracker.log_summary()`` to retrieve results.
    epoch : int
        Current epoch number (0-indexed).  Used for logging and sample
        grid filenames.
    config : Config
        The ILGAN configuration object.  The following keys are read:

        - ``config.model.latent_dim`` (int): dimensionality of the latent
          space.
        - ``config.model.num_classes`` (int): number of object classes.
        - ``config.model.max_boxes`` (int): maximum number of predicted
          boxes per image.
        - ``config.data.image_size`` (int): spatial size of images.
        - ``config.paths.log_dir`` (str): directory for saving sample grids.
        - ``config.logging.log_interval`` (int): log progress every N
          batches.
    logger : Logger
        The ILGAN logger instance for console output.

    Returns
    -------
    dict of str -> Any
        A dictionary of validation metrics with the following keys:

        - **Loss metrics** (all individual loss terms averaged over the
          validation set, prefixed with ``"val/"``):

          - ``"val/total_g_loss"``: total generator loss.
          - ``"val/total_d_loss"``: total discriminator loss.
          - ``"val/g_loss_adv"``: generator adversarial loss.
          - ``"val/d_loss_adv"``: discriminator WGAN loss.
          - ``"val/gp_loss"``: gradient penalty (weighted).
          - ``"val/box_loss"``: box regression loss.
          - ``"val/class_loss"``: class cross-entropy loss.
          - ``"val/confidence_loss"``: confidence BCE loss.
          - ``"val/collapse_loss"``: collapse prevention loss.
          - ``"val/consistency_loss"``: cross-modal consistency loss.

        - **Realism gap**:

          - ``"val/realism_gap"``: :math:`\\mathbb{E}[D(\\text{real})] -
            \\mathbb{E}[D(\\text{fake})]` (float).  Positive means the
            discriminator can distinguish real from fake.

        - **Image metrics** (from ``metrics_tracker``, prefixed with
          ``"val/"``):

          - ``"val/image/fid"``: FID score.
          - ``"val/image/inception_score"``: Inception Score mean.
          - ``"val/image/inception_score_std"``: Inception Score std.

        - **Box metrics** (from ``metrics_tracker``, prefixed with
          ``"val/"``):

          - ``"val/box/mAP"``: mean Average Precision.
          - ``"val/box/mean_giou"``: mean GIoU.
          - ``"val/box/detection_accuracy"``: detection accuracy.
          - ``"val/box/recall"``: recall.

        - **Joint score**:

          - ``"val/joint_score"``: heuristic joint quality score.

        - **Sample grid path**:

          - ``"val/sample_grid_path"``: filesystem path to the saved
            sample grid image.

    Raises
    ------
    ValueError
        If the config is missing required keys.
    RuntimeError
        If the discriminator does not return a tuple of two tensors.

    Notes
    -----
    **No gradient computation**

    This function is decorated with ``@torch.no_grad()``, so no gradients
    are computed for any operation.  This minimises GPU memory usage during
    validation and ensures that validation does not interfere with training
    state.

    **Fixed sample grid**

    A fixed set of latent vectors (``num_grid_samples``, default 16) is
    generated once and reused across all validation epochs.  This provides
    a consistent visual comparison of generator progress over time.  The
    grid is saved as a PNG image in the log directory.

    **Realism gap**

    The realism gap is computed as the difference between the mean global
    discriminator score on real images and the mean global discriminator
    score on fake images:

    .. math::

        \\Delta = \\frac{1}{B} \\sum_{i=1}^{B} D_{global}(I_{real}^{(i)})
                - \\frac{1}{B} \\sum_{i=1}^{B} D_{global}(I_{fake}^{(i)})

    A positive gap indicates the discriminator can distinguish real from
    fake.  As the generator improves, this gap should shrink toward zero.
    A negative gap suggests the generator is fooling the discriminator
    (which is the goal, but may indicate an overconfident generator or
    a weak discriminator).

    **Metrics tracker**

    The ``metrics_tracker`` is **reset** at the start of this function.
    After validation, the caller can use ``metrics_tracker.compute_all()``
    or ``metrics_tracker.log_summary()`` to retrieve and log the results.
    The returned dictionary also contains the key metrics for convenience.
    """
    # ──────────────────────────────────────────────────────────────────────────
    # 1. Extract config values
    # ──────────────────────────────────────────────────────────────────────────
    try:
        latent_dim: int = int(config.model.latent_dim)
        num_classes: int = int(config.model.num_classes)
        max_boxes: int = int(config.model.max_boxes)
        image_size: int = int(config.data.image_size)
        log_dir: str = str(config.paths.log_dir)
        log_interval: int = int(config.logging.log_interval)
    except (AttributeError, KeyError, TypeError) as e:
        raise ValueError(
            f"Config is missing a required key for validation: {e}. "
            f"Please ensure your config has all required fields."
        ) from e

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Set all models to evaluation mode
    # ──────────────────────────────────────────────────────────────────────────
    generator.eval()
    discriminator.eval()
    image_encoder.eval()
    box_encoder.eval()

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Initialise helpers
    # ──────────────────────────────────────────────────────────────────────────
    device = next(generator.parameters()).device

    # Reset the metrics tracker for a clean validation run
    metrics_tracker.reset()

    # Running loss accumulators for the validation set
    running_losses: Dict[str, float] = {}
    num_batches: int = 0

    # Realism gap accumulators
    running_real_scores: List[float] = []
    running_fake_scores: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Fixed sample grid latents (generated once per validation call)
    # ──────────────────────────────────────────────────────────────────────────
    num_grid_samples: int = _DEFAULT_NUM_SAMPLE_GRID
    grid_rows: int = int(math.isqrt(num_grid_samples))
    grid_cols: int = grid_rows
    # Ensure we have a perfect square
    if grid_rows * grid_cols < num_grid_samples:
        grid_cols += 1
    actual_grid_size: int = grid_rows * grid_cols

    fixed_z: torch.Tensor = torch.randn(actual_grid_size, latent_dim, device=device)
    """Fixed latent vectors for consistent sample grid generation across epochs."""

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Iterate over validation batches
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(
        f"Starting validation for epoch {epoch} — "
        f"{len(val_loader)} batches in validation set."
    )

    for batch_idx, batch in enumerate(val_loader):
        # ── 5a. Move batch to device ──────────────────────────────────────────
        if isinstance(batch, Batch):
            batch = batch.to(device)
            batch_dict: Dict[str, torch.Tensor] = {
                "images": batch.images,
                "boxes": batch.boxes,
                "labels": batch.labels,
                "valid_mask": batch.valid_mask,
            }
        elif isinstance(batch, dict):
            batch_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            # Assume (images, boxes, labels, valid_mask) tuple
            batch_dict = {
                "images": batch[0].to(device),
                "boxes": batch[1].to(device),
                "labels": batch[2].to(device),
                "valid_mask": batch[3].to(device),
            }
        else:
            raise TypeError(
                f"Unsupported batch type: {type(batch)}. "
                f"Expected Batch, dict, or tuple."
            )

        B: int = batch_dict["images"].shape[0]

        # ── 5b. Sample latent vectors ─────────────────────────────────────────
        z: torch.Tensor = torch.randn(B, latent_dim, device=device)

        # ── 5c. Generator forward ────────────────────────────────────────────
        # No autocast needed since we are in no_grad mode
        gen_outputs: Dict[str, Any] = generator(z)

        # ── 5d. Discriminator forward on real and fake images ────────────────
        real_images: torch.Tensor = batch_dict["images"]
        fake_images: torch.Tensor = gen_outputs["image"]

        real_scores_local, real_scores_global = discriminator(real_images)
        fake_scores_local, fake_scores_global = discriminator(fake_images)

        # ── 5e. Compute all losses (for monitoring, not backprop) ────────────
        full_losses: Dict[str, torch.Tensor] = loss_aggregator(
            generator_outputs=gen_outputs,
            batch=batch_dict,
            discriminator=discriminator,
            image_encoder=image_encoder,
            box_encoder=box_encoder,
            z_batch=z,
        )

        # ── 5f. Accumulate realism gap ───────────────────────────────────────
        # Realism gap = E[D(real)] - E[D(fake)]
        # Use the global score (scalar per sample) for the gap.
        # real_scores_global: [B], fake_scores_global: [B]
        running_real_scores.extend(real_scores_global.detach().cpu().tolist())
        running_fake_scores.extend(fake_scores_global.detach().cpu().tolist())

        # ── 5g. Accumulate losses ────────────────────────────────────────────
        step_losses: Dict[str, float] = {}
        for key, value in full_losses.items():
            if isinstance(value, torch.Tensor):
                step_losses[key] = value.detach().item()
            elif isinstance(value, (int, float)):
                step_losses[key] = float(value)

        # Update running averages (cumulative mean)
        for key, value in step_losses.items():
            if key in running_losses:
                running_losses[key] = running_losses[key] + (
                    value - running_losses[key]
                ) / (num_batches + 1)
            else:
                running_losses[key] = value

        # ── 5h. Update metrics tracker ───────────────────────────────────────
        # Image metrics: FID and IS are accumulated across the validation set
        metrics_tracker.update_image_metrics(
            real_images=real_images,
            fake_images=fake_images,
        )

        # Box metrics: extract predictions from generator outputs
        pred_boxes: torch.Tensor = gen_outputs["boxes"]          # [B, N, 4]
        pred_class_logits: torch.Tensor = gen_outputs["class_logits"]  # [B, N, C]
        pred_confidences: torch.Tensor = gen_outputs["confidences"]    # [B, N, 1]

        # Convert class logits to hard labels
        pred_labels: torch.Tensor = pred_class_logits.argmax(dim=-1)  # [B, N]

        # Squeeze confidence from [B, N, 1] to [B, N]
        pred_scores: torch.Tensor = pred_confidences.squeeze(-1)  # [B, N]

        # Ground truth
        target_boxes: torch.Tensor = batch_dict["boxes"]
        target_labels: torch.Tensor = batch_dict["labels"]
        valid_mask: torch.Tensor = batch_dict["valid_mask"]

        metrics_tracker.update_box_metrics(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            target_boxes=target_boxes,
            target_labels=target_labels,
            valid_mask=valid_mask,
        )

        # Push loss metrics
        metrics_tracker.update_loss_metrics(step_losses)

        num_batches += 1

        # ── 5i. Logging ──────────────────────────────────────────────────────
        if batch_idx % log_interval == 0 or batch_idx == len(val_loader) - 1:
            log_parts: List[str] = [
                f"Val Epoch {epoch:>3d} | Batch {batch_idx:>4d}/{len(val_loader):<4d}"
            ]

            # Add key losses
            if "total_g_loss" in step_losses:
                log_parts.append(f"G: {step_losses['total_g_loss']:.4f}")
            if "total_d_loss" in step_losses:
                log_parts.append(f"D: {step_losses['total_d_loss']:.4f}")
            if "box_loss" in step_losses:
                log_parts.append(f"Box: {step_losses['box_loss']:.4f}")
            if "consistency_loss" in step_losses:
                log_parts.append(f"Cons: {step_losses['consistency_loss']:.4f}")
            if "collapse_loss" in step_losses:
                log_parts.append(f"Coll: {step_losses['collapse_loss']:.4f}")

            # Add realism gap estimate for this batch
            batch_real_mean: float = float(real_scores_global.mean().item())
            batch_fake_mean: float = float(fake_scores_global.mean().item())
            batch_gap: float = batch_real_mean - batch_fake_mean
            log_parts.append(f"| Realism gap: {batch_gap:.4f}")

            logger.info("  ".join(log_parts))

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Compute final metrics
    # ──────────────────────────────────────────────────────────────────────────

    # ── 6a. Realism gap ────────────────────────────────────────────────────────
    if running_real_scores and running_fake_scores:
        mean_real_score: float = float(torch.tensor(running_real_scores).mean().item())
        mean_fake_score: float = float(torch.tensor(running_fake_scores).mean().item())
        realism_gap: float = mean_real_score - mean_fake_score
    else:
        mean_real_score = float("nan")
        mean_fake_score = float("nan")
        realism_gap = float("nan")

    logger.info(
        f"Val Epoch {epoch:>3d} — Realism gap: {realism_gap:.6f} "
        f"(E[D(real)]={mean_real_score:.6f}, "
        f"E[D(fake)]={mean_fake_score:.6f})"
    )

    # ── 6b. Compute all metrics from the tracker ──────────────────────────────
    all_metrics: Dict[str, Any] = metrics_tracker.compute_all()

    # ── 6c. Log summary ──────────────────────────────────────────────────────
    metrics_tracker.log_summary(epoch=epoch, logger=logger, phase="Val")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Generate fixed sample grid for visual inspection
    # ──────────────────────────────────────────────────────────────────────────
    sample_grid_path: str = _generate_sample_grid(
        generator=generator,
        fixed_z=fixed_z,
        epoch=epoch,
        log_dir=log_dir,
        image_size=image_size,
        nrow=grid_cols,
    )
    logger.info(f"Sample grid saved to: {sample_grid_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Assemble and return validation metrics dictionary
    # ──────────────────────────────────────────────────────────────────────────
    val_metrics: Dict[str, Any] = {}

    # Add loss metrics with "val/" prefix
    for key, value in running_losses.items():
        val_metrics[f"val/{key}"] = value

    # Add realism gap
    val_metrics["val/realism_gap"] = realism_gap
    val_metrics["val/mean_real_score"] = mean_real_score
    val_metrics["val/mean_fake_score"] = mean_fake_score

    # Add image metrics from tracker
    val_metrics["val/image/fid"] = all_metrics.get("image/fid", float("nan"))
    val_metrics["val/image/inception_score"] = all_metrics.get(
        "image/inception_score", float("nan")
    )
    val_metrics["val/image/inception_score_std"] = all_metrics.get(
        "image/inception_score_std", float("nan")
    )

    # Add box metrics from tracker
    val_metrics["val/box/mAP"] = all_metrics.get("box/mAP", float("nan"))
    val_metrics["val/box/mean_giou"] = all_metrics.get("box/mean_giou", float("nan"))
    val_metrics["val/box/detection_accuracy"] = all_metrics.get(
        "box/detection_accuracy", float("nan")
    )
    val_metrics["val/box/recall"] = all_metrics.get("box/recall", float("nan"))
    val_metrics["val/box/mean_confidence"] = all_metrics.get(
        "box/mean_confidence", float("nan")
    )
    val_metrics["val/box/mean_box_size"] = all_metrics.get(
        "box/mean_box_size", float("nan")
    )
    val_metrics["val/box/std_cx"] = all_metrics.get("box/std_cx", float("nan"))
    val_metrics["val/box/std_cy"] = all_metrics.get("box/std_cy", float("nan"))

    # Add joint score
    val_metrics["val/joint_score"] = all_metrics.get("joint_score", float("nan"))

    # Add sample grid path
    val_metrics["val/sample_grid_path"] = sample_grid_path

    # Add num batches processed
    val_metrics["val/num_batches"] = num_batches

    # Log final validation summary
    logger.info(
        f"Validation complete for epoch {epoch:>3d} — "
        f"{num_batches} batches processed. "
        f"Realism gap: {realism_gap:.6f}, "
        f"Joint score: {val_metrics['val/joint_score']:.6f}"
    )

    return val_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Sample Grid Generation
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _generate_sample_grid(
    generator: nn.Module,
    fixed_z: torch.Tensor,
    epoch: int,
    log_dir: str,
    image_size: int,
    nrow: int = 4,
) -> str:
    """Generate a grid of sample images from fixed latent vectors and save
    as a PNG file.

    This function:

    1. Runs the generator on the fixed latent vectors.
    2. Normalises the generated images from ``[-1, 1]`` to ``[0, 1]`` for
       ``save_image``.
    3. Creates a grid using ``torchvision.utils.make_grid``.
    4. Saves the grid as ``sample_grid_epoch_{epoch:04d}.png`` in the log
       directory.
    5. Also saves a grid with predicted bounding boxes drawn on each image
       as ``sample_grid_boxes_epoch_{epoch:04d}.png``.

    Parameters
    ----------
    generator : nn.Module
        The ILGAN generator in evaluation mode.
    fixed_z : torch.Tensor
        Fixed latent vectors of shape ``[N, latent_dim]``.
    epoch : int
        Current epoch number (used in the filename).
    log_dir : str
        Directory where the sample grid image will be saved.
    image_size : int
        Spatial size of generated images (used for logging only).
    nrow : int, optional
        Number of images per row in the grid.  (default: 4)

    Returns
    -------
    str
        Absolute filesystem path to the saved sample grid image.

    Notes
    -----
    - The grid is saved as a PNG with pixel values in ``[0, 1]``.
    - The filename format is ``sample_grid_epoch_{epoch:04d}.png``.
    - If the log directory does not exist, it is created.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate images from fixed latents
    gen_outputs: Dict[str, Any] = generator(fixed_z)
    fake_images: torch.Tensor = gen_outputs["image"]  # [N, 3, H, W], values in [-1, 1]
    pred_boxes: torch.Tensor = gen_outputs["boxes"]  # [N, max_boxes, 4]
    class_logits: torch.Tensor = gen_outputs["class_logits"]  # [N, max_boxes, C]
    confidences: torch.Tensor = gen_outputs["confidences"]  # [N, max_boxes, 1]
    pred_labels: torch.Tensor = class_logits.argmax(dim=-1)  # [N, max_boxes]

    # ── Save plain image grid (no boxes) ────────────────────────────────
    grid_images: torch.Tensor = (fake_images + 1.0) / 2.0
    grid_images = torch.clamp(grid_images, 0.0, 1.0)
    grid: torch.Tensor = make_grid(grid_images, nrow=nrow, padding=2, normalize=False)
    filename: str = f"sample_grid_epoch_{epoch:04d}.png"
    filepath: str = os.path.join(log_dir, filename)
    save_image(grid, filepath)

    # ── Save box-overlaid grid ──────────────────────────────────────────
    # Draw bounding boxes on each generated image using the visualisation
    # utility, then tile them into a grid and save.
    N = fake_images.size(0)
    annotated_pils: List[Image.Image] = []

    for b in range(N):
        pil_img = draw_boxes_on_image(
            image_tensor=fake_images[b],
            boxes=pred_boxes[b],
            confidences=confidences[b],
            labels=pred_labels[b],
            class_names=VOC_CLASSES,
            confidence_threshold=0.3,  # Show all reasonably confident boxes
        )
        annotated_pils.append(pil_img)

    # Convert PIL images to tensors for grid creation
    annotated_tensors: List[torch.Tensor] = []
    for pil_img in annotated_pils:
        np_img = np.array(pil_img, dtype=np.float32) / 255.0  # [H, W, 3], [0, 1]
        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1)  # [3, H, W]
        annotated_tensors.append(tensor_img)

    batch_tensor = torch.stack(annotated_tensors, dim=0)  # [N, 3, H, W]
    boxes_grid: torch.Tensor = make_grid(batch_tensor, nrow=nrow, padding=2, normalize=False)
    boxes_filename: str = f"sample_grid_boxes_epoch_{epoch:04d}.png"
    boxes_filepath: str = os.path.join(log_dir, boxes_filename)
    save_image(boxes_grid, boxes_filepath)

    return os.path.abspath(filepath)


# ──────────────────────────────────────────────────────────────────────────────
# Module exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "validate",
    "_generate_sample_grid",
]
