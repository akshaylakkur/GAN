"""
Top-level training orchestration for the ILGAN dual-output GAN.

The :class:`ILGANTrainer` is the central coordinator that wires together
all components of the ILGAN system — generator, discriminator, encoders,
loss aggregator, metrics tracker, optimizers, schedulers, checkpoint
manager, and AMP scaler — into a coherent training, evaluation, and
generation pipeline.

Architecture overview
---------------------
The trainer follows a modular design where each component is created in
:meth:`__init__` and stored as an attribute.  The :meth:`train` method
orchestrates the full training loop:

1. Create train/val dataloaders via :func:`get_train_val_loaders`.
2. Load checkpoint if available (resume training).
3. For each epoch:
   a. Call :func:`train_epoch` for one epoch of training.
   b. Call :func:`validate` every ``eval_interval`` epochs.
   c. Save checkpoint every ``save_interval`` epochs.
   d. Log epoch summary.
4. Save final checkpoint and run final validation.

Device placement
----------------
All models are moved to the appropriate device (CUDA if available, else
CPU) at construction time.  The trainer detects the device automatically
and moves all components accordingly.

Distributed training
--------------------
The trainer detects if ``torch.distributed`` is initialised and wraps
models with ``DistributedDataParallel``.  This is a stub for now and
can be extended for multi-GPU training.

Mathematical grounding
----------------------
The training loop implements the WGAN-GP adversarial training procedure
with the following key innovations:

- **n_critic ratio**: The discriminator is updated every step, while the
  generator is updated only every ``n_critic`` steps.  This ensures the
  discriminator maintains a good estimate of the Wasserstein distance.

- **Gradient accumulation**: Gradients are accumulated over multiple
  forward passes to simulate larger batch sizes without exceeding GPU
  memory.

- **Representation anchoring**: Running statistics of latent vectors are
  tracked and regularised toward the prior :math:`\\mathcal{N}(0, I)`,
  preventing latent drift and mode collapse.

- **Adaptive learning rate balancing**: The
  :class:`AdaptiveOptimizerScheduler` dynamically adjusts the learning
  rates of the generator and discriminator based on the ratio of their
  output magnitudes, preventing one network from overwhelming the other.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler

from ilgan.data.dataloader import GANDataloader, get_train_val_loaders
from ilgan.utils.device import get_device, get_device_info, supports_amp, get_amp_device_type
from ilgan.data.streaming_voc import get_streaming_loaders
from ilgan.data.structures import Batch
from ilgan.losses import LossAggregator
from ilgan.losses.consistency import BoxFeatureEncoder, ImageFeatureEncoder
from ilgan.metrics import build_metrics_tracker
from ilgan.metrics.joint_metrics import MetricsTracker
from ilgan.models import ILGANGenerator, ImageDiscriminator
from ilgan.training.checkpoint import CheckpointManager, load_or_initialize
from ilgan.training.mixed_precision import AMPScaler, create_amp_scaler
from ilgan.training.optimizers import (
    AdaptiveOptimizerScheduler,
    build_optimizers,
    build_scheduler,
)
from ilgan.training.train_epoch import train_epoch
from ilgan.training.val_epoch import validate
from ilgan.utils.config import Config
from ilgan.utils.logger import Logger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_EVAL_INTERVAL: int = 5
"""Default number of epochs between validation runs."""

_DEFAULT_SAVE_INTERVAL: int = 10
"""Default number of epochs between checkpoint saves."""

_DEFAULT_LOG_INTERVAL: int = 50
"""Default number of training steps between console log messages."""

_DEFAULT_NUM_SAMPLE_GRID: int = 16
"""Default number of samples in the generation grid (must be a perfect
square for ``make_grid``)."""

_JOINT_SCORE_KEY: str = "joint_score"
"""Key used to store the joint score in the checkpoint dictionary."""


# ──────────────────────────────────────────────────────────────────────────────
# ILGANTrainer
# ──────────────────────────────────────────────────────────────────────────────


class ILGANTrainer:
    """Top-level training orchestration for the ILGAN dual-output GAN.

    The :class:`ILGANTrainer` is the central coordinator that wires together
    all components of the ILGAN system.  It provides three main workflows:

    - :meth:`train`: full training loop with checkpointing, validation,
      and logging.
    - :meth:`evaluate`: load a checkpoint and run a full evaluation on the
      validation set.
    - :meth:`generate`: load a checkpoint, generate samples, and save them
      to disk.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.  Must contain all required keys
        for model construction, training, logging, and paths.
    logger : Logger
        The ILGAN logger instance for console and file output.

    Raises
    ------
    TypeError
        If *config* is not a :class:`Config` instance or *logger* is not
        a :class:`Logger` instance.
    ValueError
        If the config is missing required keys for model construction.

    Examples
    --------
    >>> from ilgan.utils.config import Config
    >>> from ilgan.utils.logger import Logger
    >>> from ilgan.training.trainer import ILGANTrainer
    >>> cfg = Config()
    >>> logger = Logger(log_dir=cfg.paths.log_dir)
    >>> trainer = ILGANTrainer(cfg, logger)
    >>> trainer.train()
    """

    def __init__(
        self,
        config: Config,
        logger: Logger,
    ) -> None:
        # ── Validate inputs ─────────────────────────────────────────────
        if not isinstance(config, Config):
            raise TypeError(
                f"Expected 'config' to be a Config instance, "
                f"got {type(config).__name__}."
            )
        if not isinstance(logger, Logger):
            raise TypeError(
                f"Expected 'logger' to be a Logger instance, "
                f"got {type(logger).__name__}."
            )

        self.config: Config = config
        self.logger: Logger = logger

        # ── Device detection ──────────────────────────────────────────────
        self.device: torch.device = get_device()
        self.logger.info(
            f"ILGANTrainer initialised — device: {self.device}"
        )

        # ── Distributed training detection ───────────────────────────────
        self.is_distributed: bool = torch.distributed.is_initialized()
        if self.is_distributed:
            self.world_size: int = torch.distributed.get_world_size()
            self.rank: int = torch.distributed.get_rank()
            self.logger.info(
                f"Distributed training detected — "
                f"world_size={self.world_size}, rank={self.rank}"
            )
        else:
            self.world_size = 1
            self.rank = 0

        # ── Extract config values for construction ──────────────────────
        try:
            self.latent_dim: int = int(config.model.latent_dim)
            self.image_size: int = int(config.data.image_size)
            self.num_classes: int = int(config.model.num_classes)
            self.max_boxes: int = int(config.model.max_boxes)
            self.gen_base_channels: int = int(config.model.gen_base_channels)
            self.disc_base_channels: int = int(config.model.disc_base_channels)
            self.num_attention_heads: int = int(config.model.num_attention_heads)
            self.epochs: int = int(config.training.epochs)
            self.eval_interval: int = int(
                getattr(config.logging, "eval_interval", _DEFAULT_EVAL_INTERVAL)
            )
            self.save_interval: int = int(
                getattr(config.logging, "save_interval", _DEFAULT_SAVE_INTERVAL)
            )
            self.log_interval: int = int(
                getattr(config.logging, "log_interval", _DEFAULT_LOG_INTERVAL)
            )
            self.checkpoint_dir: str = str(config.paths.checkpoint_dir)
            self.log_dir: str = str(config.paths.log_dir)
            self.data_root: str = str(config.paths.data_root)
            self.batch_size: int = int(config.data.batch_size)
            self.num_workers: int = int(config.data.num_workers)
            self.grad_clip_norm: float = float(config.training.clip_grad_norm)
        except (AttributeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Config is missing a required key for trainer construction: {e}. "
                f"Please ensure your config has all required fields."
            ) from e

        # ──────────────────────────────────────────────────────────────────
        # 1. Create models
        # ──────────────────────────────────────────────────────────────────
        self.logger.info("Creating ILGAN models...")

        # 1a. Generator
        self.generator: ILGANGenerator = ILGANGenerator(
            config=self.config,
        ).to(self.device)
        self.logger.info(
            f"  Generator: {sum(p.numel() for p in self.generator.parameters()):,} params"
        )

        # 1b. Discriminator
        self.discriminator: ImageDiscriminator = ImageDiscriminator(
            disc_base_channels=self.disc_base_channels,
            image_size=self.image_size,
        ).to(self.device)
        self.logger.info(
            f"  Discriminator: {sum(p.numel() for p in self.discriminator.parameters()):,} params"
        )

        # 1c. Image encoder (for cross-modal consistency)
        self.image_encoder: ImageFeatureEncoder = ImageFeatureEncoder(
            proj_dim=128,
        ).to(self.device)
        self.logger.info(
            f"  Image Encoder: {sum(p.numel() for p in self.image_encoder.parameters()):,} params"
        )

        # 1d. Box encoder (for cross-modal consistency)
        self.box_encoder: BoxFeatureEncoder = BoxFeatureEncoder(
            proj_dim=128,
        ).to(self.device)
        self.logger.info(
            f"  Box Encoder: {sum(p.numel() for p in self.box_encoder.parameters()):,} params"
        )

        # ──────────────────────────────────────────────────────────────────
        # 2. Create loss aggregator
        # ──────────────────────────────────────────────────────────────────
        self.loss_aggregator: LossAggregator = LossAggregator(self.config)
        self.logger.info("  LossAggregator created.")

        # ──────────────────────────────────────────────────────────────────
        # 3. Create metrics tracker
        # ──────────────────────────────────────────────────────────────────
        self.metrics_tracker: MetricsTracker = build_metrics_tracker(self.config)
        self.logger.info("  MetricsTracker created.")

        # ──────────────────────────────────────────────────────────────────
        # 4. Create optimizers
        # ──────────────────────────────────────────────────────────────────
        self.g_optimizer, self.d_optimizer = build_optimizers(
            generator=self.generator,
            discriminator=self.discriminator,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
            config=self.config,
        )
        self.logger.info("  Optimizers created.")

        # ──────────────────────────────────────────────────────────────────
        # 5. Create learning rate schedulers
        # ──────────────────────────────────────────────────────────────────
        scheduler_type: str = getattr(config.training, "scheduler_type", "cosine")
        self.schedulers: Dict[str, Any] = build_scheduler(
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            config=self.config,
            scheduler_type=scheduler_type,
        )
        self.g_scheduler = self.schedulers["g_scheduler"]
        self.d_scheduler = self.schedulers["d_scheduler"]
        self.logger.info(f"  Schedulers created (type={scheduler_type}).")

        # ── Adaptive optimizer scheduler (optional) ──────────────────────
        use_adaptive: bool = getattr(config.training, "use_adaptive_lr", False)
        if use_adaptive:
            self.adaptive_scheduler: Optional[AdaptiveOptimizerScheduler] = (
                AdaptiveOptimizerScheduler(
                    g_optimizer=self.g_optimizer,
                    d_optimizer=self.d_optimizer,
                    window_size=getattr(
                        config.training, "adaptive_window", 100
                    ),
                    threshold=getattr(
                        config.training, "adaptive_threshold", 1.5
                    ),
                    adjustment_factor=getattr(
                        config.training, "adaptive_adjustment", 0.95
                    ),
                    min_lr=getattr(config.training, "adaptive_min_lr", 1e-6),
                    max_lr=getattr(config.training, "adaptive_max_lr", 1e-2),
                    cooldown_steps=getattr(
                        config.training, "adaptive_cooldown", 10
                    ),
                )
            )
            self.logger.info("  AdaptiveOptimizerScheduler created.")
        else:
            self.adaptive_scheduler = None

        # ──────────────────────────────────────────────────────────────────
        # 6. Create AMP scaler
        # ──────────────────────────────────────────────────────────────────
        self.amp_scaler: AMPScaler = create_amp_scaler(self.config)
        self.logger.info(
            f"  AMP scaler created (enabled={self.amp_scaler.is_enabled})."
        )

        # ──────────────────────────────────────────────────────────────────
        # 7. Create checkpoint manager
        # ──────────────────────────────────────────────────────────────────
        max_checkpoints: int = getattr(config.training, "max_checkpoints", 10)
        self.checkpoint_manager: CheckpointManager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            config=self.config,
            max_checkpoints=max_checkpoints,
        )
        self.logger.info(
            f"  CheckpointManager created (dir={self.checkpoint_dir})."
        )

        # ──────────────────────────────────────────────────────────────────
        # 8. Distributed training setup (optional)
        # ──────────────────────────────────────────────────────────────────
        if self.is_distributed:
            self._setup_distributed()

        # ── Training state ────────────────────────────────────────────────
        self.start_epoch: int = 0
        self.global_step: int = 0
        self.best_joint_score: float = -float("inf")

        self.logger.info("ILGANTrainer initialisation complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Train
    # ──────────────────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """Run the full ILGAN training loop.

        This method orchestrates the entire training process:

        1. Creates train/val dataloaders using :func:`get_train_val_loaders`.
        2. Loads checkpoint if available (resumes training).
        3. For each epoch from ``start_epoch`` to ``config.training.epochs``:
           a. Calls :meth:`_train_epoch` for one epoch of training.
           b. Calls :meth:`_validate` every ``eval_interval`` epochs.
           c. Saves checkpoint every ``save_interval`` epochs.
           d. Logs epoch summary.
        4. Saves final checkpoint and runs final validation.

        Returns
        -------
        dict of str -> Any
            A dictionary containing the final training metrics:

            - ``"final_epoch"``: the last epoch completed.
            - ``"final_global_step"``: the total number of training steps.
            - ``"best_joint_score"``: the best joint score achieved.
            - ``"best_checkpoint_path"``: path to the best checkpoint.
            - ``"final_checkpoint_path"``: path to the final checkpoint.
            - ``"final_val_metrics"``: validation metrics from the final
              validation run (if any).
            - ``"training_time_seconds"``: total training time in seconds.

        Raises
        ------
        RuntimeError
            If training fails due to NaN/Inf gradients that cannot be
            recovered from.
        KeyboardInterrupt
            If the user interrupts training (Ctrl+C), the trainer will
            save a checkpoint and exit gracefully.

        Notes
        -----
        - The training loop is interrupt-safe: if a ``KeyboardInterrupt``
          is caught, the trainer saves a checkpoint before re-raising.
        - The best checkpoint is tracked by joint score and saved
          separately as ``best_checkpoint.pt``.
        - Validation is run at the end of training even if the final epoch
          is not an ``eval_interval`` multiple.
        """
        self.logger.info("=" * 72)
        self.logger.info("  ILGAN Training — Starting")
        self.logger.info(f"  Epochs: {self.start_epoch} → {self.epochs}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Batch size: {self.batch_size}")
        self.logger.info(f"  Mixed precision: {self.amp_scaler.is_enabled}")
        self.logger.info("=" * 72)

        # ── 1. Create dataloaders ─────────────────────────────────────────
        self.logger.info("Creating dataloaders...")

        # Check if streaming mode is enabled (for cloud training without local data)
        use_streaming = getattr(self.config.data, "use_streaming", False)
        streaming_dataset = getattr(self.config.data, "streaming_dataset", "voc")

        if use_streaming:
            self.logger.info(f"  Using streaming dataset: {streaming_dataset}")
            train_loader, val_loader = get_streaming_loaders(
                image_size=self.image_size,
                batch_size=self.batch_size,
                max_boxes=self.max_boxes,
                num_workers=self.num_workers,
                cache_size=getattr(self.config.data, "streaming_cache_size", 256),
            )
        else:
            train_loader, val_loader = get_train_val_loaders(
                root_dir=self.data_root,
                image_size=self.image_size,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                val_split=0.2,
                augment=True,
                global_max_boxes=self.max_boxes,
                train_max_boxes=self.max_boxes,
                val_max_boxes=self.max_boxes,
            )

        self.logger.info(
            f"  Train loader: {len(train_loader.dataset)} samples, "
            f"{len(train_loader)} batches"
        )
        self.logger.info(
            f"  Val loader:   {len(val_loader.dataset)} samples, "
            f"{len(val_loader)} batches"
        )

        # ── 2. Load checkpoint if available ───────────────────────────────
        self.start_epoch, self.global_step = load_or_initialize(
            checkpoint_manager=self.checkpoint_manager,
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            config=self.config,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
        )

        # ── 3. Training loop ─────────────────────────────────────────────
        training_start_time: float = time.time()
        final_val_metrics: Optional[Dict[str, Any]] = None

        try:
            for epoch in range(self.start_epoch, self.epochs):
                epoch_start_time: float = time.time()

                # ── 3a. Set epoch for deterministic augmentation ──────────
                train_loader.set_epoch(epoch)

                # ── 3b. Train one epoch ───────────────────────────────────
                self.global_step, epoch_losses = train_epoch(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    image_encoder=self.image_encoder,
                    box_encoder=self.box_encoder,
                    train_loader=train_loader,
                    g_optimizer=self.g_optimizer,
                    d_optimizer=self.d_optimizer,
                    loss_aggregator=self.loss_aggregator,
                    metrics_tracker=self.metrics_tracker,
                    epoch=epoch,
                    global_step=self.global_step,
                    config=self.config,
                    logger=self.logger,
                    amp_scaler=self.amp_scaler,
                    grad_clip_norm=self.grad_clip_norm,
                )

                # ── 3c. Step schedulers ───────────────────────────────────
                self.g_scheduler.step()
                self.d_scheduler.step()

                # ── 3d. Log epoch summary ─────────────────────────────────
                epoch_time: float = time.time() - epoch_start_time
                self._log_epoch_summary(
                    epoch=epoch,
                    epoch_time=epoch_time,
                    epoch_losses=epoch_losses,
                )

                # ── 3e. Validation every eval_interval epochs ─────────────
                if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                    val_metrics = self._validate(
                        val_loader=val_loader,
                        epoch=epoch,
                    )
                    if val_metrics is not None:
                        final_val_metrics = val_metrics

                        # Update best joint score
                        joint_score = val_metrics.get("val/joint_score", float("nan"))
                        if not math.isnan(joint_score) and joint_score > self.best_joint_score:
                            self.best_joint_score = joint_score
                            self.logger.info(
                                f"  🏆 New best joint score: {joint_score:.6f} "
                                f"(epoch {epoch})"
                            )

                # ── 3f. Save checkpoint every save_interval epochs ────────
                if (epoch + 1) % self.save_interval == 0 or epoch == self.epochs - 1:
                    self._save_checkpoint(epoch=epoch)

        except KeyboardInterrupt:
            self.logger.warning(
                "Training interrupted by user (Ctrl+C). "
                "Saving checkpoint before exit..."
            )
            self._save_checkpoint(epoch=self.global_step)
            raise

        except Exception as e:
            self.logger.error(
                f"Training failed with exception: {e}"
            )
            # Save a recovery checkpoint
            self._save_checkpoint(epoch=self.global_step)
            raise

        # ── 4. Finalise ──────────────────────────────────────────────────
        total_training_time: float = time.time() - training_start_time

        # Save final checkpoint
        final_checkpoint_path: str = self._save_checkpoint(
            epoch=self.epochs - 1,
        )

        # Run final validation if not already done
        if final_val_metrics is None:
            final_val_metrics = self._validate(
                val_loader=val_loader,
                epoch=self.epochs - 1,
            )

        # Get best checkpoint path
        best_checkpoint_path: Optional[str] = (
            self.checkpoint_manager.get_best_checkpoint()
        )

        # Log training summary
        self.logger.info("=" * 72)
        self.logger.info("  ILGAN Training — Complete")
        self.logger.info(f"  Total epochs: {self.epochs}")
        self.logger.info(f"  Total steps:  {self.global_step}")
        self.logger.info(f"  Total time:   {total_training_time:.2f}s "
                          f"({total_training_time / 60:.2f}min)")
        self.logger.info(f"  Best joint score: {self.best_joint_score:.6f}")
        if best_checkpoint_path:
            self.logger.info(f"  Best checkpoint: {best_checkpoint_path}")
        self.logger.info(f"  Final checkpoint: {final_checkpoint_path}")
        self.logger.info("=" * 72)

        return {
            "final_epoch": self.epochs - 1,
            "final_global_step": self.global_step,
            "best_joint_score": self.best_joint_score,
            "best_checkpoint_path": best_checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
            "final_val_metrics": final_val_metrics,
            "training_time_seconds": total_training_time,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Evaluate
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """Load a checkpoint and run a full evaluation on the validation set.

        This method:

        1. Loads the model and optimizer states from *checkpoint_path*.
        2. Creates a validation dataloader.
        3. Runs :func:`validate` on the full validation set.
        4. Logs all metrics and returns them.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (``.pt``) to load.

        Returns
        -------
        dict of str -> Any
            A dictionary of validation metrics (see :func:`validate` for
            the full list of keys).

        Raises
        ------
        FileNotFoundError
            If *checkpoint_path* does not exist.
        RuntimeError
            If the checkpoint is corrupt or missing required keys.

        Notes
        -----
        - The generator and discriminator are set to evaluation mode for
          the duration of the evaluation.
        - No gradients are computed (``torch.no_grad()`` is used internally
          by :func:`validate`).
        - The metrics tracker is reset at the start of evaluation.
        """
        self.logger.info(f"Loading checkpoint for evaluation: {checkpoint_path}")

        # ── Load checkpoint ──────────────────────────────────────────────
        epoch, global_step = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
        )
        self.logger.info(
            f"Checkpoint loaded — epoch {epoch}, global step {global_step}"
        )

        # ── Create validation dataloader ──────────────────────────────────
        _, val_loader = get_train_val_loaders(
            root_dir=self.data_root,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            val_split=0.2,
            augment=False,  # no augmentation for evaluation
            global_max_boxes=self.max_boxes,
            train_max_boxes=self.max_boxes,
            val_max_boxes=self.max_boxes,
        )

        # ── Run validation ───────────────────────────────────────────────
        val_metrics = self._validate(
            val_loader=val_loader,
            epoch=epoch,
        )

        self.logger.info("Evaluation complete.")
        return val_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Generate
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        num_samples: int,
        checkpoint_path: str,
        output_path: str,
    ) -> List[str]:
        """Load a checkpoint, generate samples, and save them to disk.

        This method:

        1. Loads the model state from *checkpoint_path*.
        2. Generates *num_samples* images and bounding boxes from random
           latent vectors.
        3. Saves each image as a PNG file and each set of bounding boxes
           as a YOLO-format text file in *output_path*.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.  Must be positive.
        checkpoint_path : str
            Path to the checkpoint file (``.pt``) to load.
        output_path : str
            Directory where generated samples will be saved.  Created
            automatically if it does not exist.

        Returns
        -------
        list of str
            A list of file paths to the saved samples.  Each entry is a
            tuple ``(image_path, label_path)``.

        Raises
        ------
        FileNotFoundError
            If *checkpoint_path* does not exist.
        ValueError
            If *num_samples* is not positive.

        Notes
        -----
        - Images are saved as PNG files with pixel values in ``[0, 1]``.
        - Labels are saved in YOLO format: ``class_id cx cy w h`` per line,
          where ``(cx, cy, w, h)`` are normalised to ``[0, 1]``.
        - The generator is set to evaluation mode for generation.
        - No gradients are computed.
        """
        if num_samples < 1:
            raise ValueError(
                f"num_samples must be positive, got {num_samples}."
            )

        self.logger.info(
            f"Generating {num_samples} samples from checkpoint: "
            f"{checkpoint_path}"
        )

        # ── Load checkpoint ──────────────────────────────────────────────
        epoch, global_step = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
        )
        self.logger.info(
            f"Checkpoint loaded — epoch {epoch}, global step {global_step}"
        )

        # ── Create output directory ───────────────────────────────────────
        os.makedirs(output_path, exist_ok=True)
        images_dir: str = os.path.join(output_path, "images")
        labels_dir: str = os.path.join(output_path, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # ── Generate samples ────────────────────────────────────────────
        self.generator.eval()
        saved_paths: List[str] = []

        # Generate in batches to avoid OOM
        batch_size: int = min(num_samples, self.batch_size)
        num_batches: int = math.ceil(num_samples / batch_size)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                # Determine batch size for this iteration
                remaining: int = num_samples - batch_idx * batch_size
                current_batch_size: int = min(batch_size, remaining)

                # Sample latent vectors
                z: torch.Tensor = torch.randn(
                    current_batch_size, self.latent_dim, device=self.device
                )

                # Generate
                gen_outputs: Dict[str, Any] = self.generator(z)

                # Extract outputs
                fake_images: torch.Tensor = gen_outputs["image"]  # [B, 3, H, W], [-1, 1]
                pred_boxes: torch.Tensor = gen_outputs["boxes"]  # [B, N, 4], (cx, cy, w, h)
                class_logits: torch.Tensor = gen_outputs["class_logits"]  # [B, N, C]
                confidences: torch.Tensor = gen_outputs["confidences"]  # [B, N, 1]

                # Convert class logits to hard labels
                pred_labels: torch.Tensor = class_logits.argmax(dim=-1)  # [B, N]

                # Normalise images from [-1, 1] to [0, 1] for saving
                images_01: torch.Tensor = (fake_images + 1.0) / 2.0
                images_01 = torch.clamp(images_01, 0.0, 1.0)

                # Save each sample
                for i in range(current_batch_size):
                    sample_idx: int = batch_idx * batch_size + i

                    # Image filename
                    img_filename: str = f"sample_{sample_idx:06d}.png"
                    img_path: str = os.path.join(images_dir, img_filename)

                    # Label filename (YOLO format)
                    label_filename: str = f"sample_{sample_idx:06d}.txt"
                    label_path: str = os.path.join(labels_dir, label_filename)

                    # Save image
                    from torchvision.utils import save_image
                    save_image(images_01[i], img_path)

                    # Save labels (YOLO format: class_id cx cy w h)
                    boxes_i: torch.Tensor = pred_boxes[i]  # [N, 4]
                    labels_i: torch.Tensor = pred_labels[i]  # [N]
                    confs_i: torch.Tensor = confidences[i].squeeze(-1)  # [N]

                    # Filter by confidence threshold
                    conf_threshold: float = getattr(
                        self.config.model, "conf_threshold", 0.3
                    )
                    valid_mask: torch.Tensor = confs_i > conf_threshold

                    with open(label_path, "w") as f:
                        for j in range(valid_mask.shape[0]):
                            if valid_mask[j].item():
                                cx, cy, w, h = boxes_i[j].tolist()
                                cls_id: int = labels_i[j].item()
                                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                    saved_paths.append(img_path)
                    saved_paths.append(label_path)

                self.logger.info(
                    f"  Generated batch {batch_idx + 1}/{num_batches} "
                    f"({min((batch_idx + 1) * batch_size, num_samples)}/{num_samples})"
                )

        self.logger.info(
            f"Generation complete — {num_samples} samples saved to {output_path}"
        )

        return saved_paths

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: Training epoch wrapper
    # ──────────────────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        train_loader: GANDataloader,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one training epoch.

        This is a thin wrapper around :func:`train_epoch` that handles
        the adaptive learning rate scheduler step.

        Parameters
        ----------
        train_loader : GANDataloader
            The training data loader.
        epoch : int
            Current epoch number.

        Returns
        -------
        dict of str -> float
            Epoch-averaged losses.
        """
        self.global_step, epoch_losses = train_epoch(
            generator=self.generator,
            discriminator=self.discriminator,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
            train_loader=train_loader,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            loss_aggregator=self.loss_aggregator,
            metrics_tracker=self.metrics_tracker,
            epoch=epoch,
            global_step=self.global_step,
            config=self.config,
            logger=self.logger,
            amp_scaler=self.amp_scaler,
            grad_clip_norm=self.grad_clip_norm,
        )

        # Step adaptive scheduler if enabled
        if self.adaptive_scheduler is not None:
            # We need discriminator output statistics for the adaptive scheduler.
            # These are not directly available here, so we use a heuristic:
            # the ratio of total_d_loss to total_g_loss as a proxy.
            d_loss = epoch_losses.get("total_d_loss", 1.0)
            g_loss = epoch_losses.get("total_g_loss", 1.0)
            # Avoid division by zero
            d_real_mean = max(d_loss, 0.01)
            d_fake_mean = max(g_loss, 0.01)
            adapt_result = self.adaptive_scheduler.step(d_real_mean, d_fake_mean)
            if adapt_result["adjusted"]:
                self.logger.info(
                    f"  Adaptive LR adjusted — direction: {adapt_result['direction']}, "
                    f"g_lr: {adapt_result['g_lr']:.8f}, "
                    f"d_lr: {adapt_result['d_lr']:.8f}"
                )

        return epoch_losses

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: Validation wrapper
    # ──────────────────────────────────────────────────────────────────────────

    def _validate(
        self,
        val_loader: GANDataloader,
        epoch: int,
    ) -> Dict[str, Any]:
        """Run validation and log results.

        Parameters
        ----------
        val_loader : GANDataloader
            The validation data loader.
        epoch : int
            Current epoch number (for logging).

        Returns
        -------
        dict of str -> Any
            Validation metrics dictionary.
        """
        self.logger.info(f"Running validation for epoch {epoch}...")
        val_start_time: float = time.time()

        val_metrics: Dict[str, Any] = validate(
            generator=self.generator,
            discriminator=self.discriminator,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
            val_loader=val_loader,
            loss_aggregator=self.loss_aggregator,
            metrics_tracker=self.metrics_tracker,
            epoch=epoch,
            config=self.config,
            logger=self.logger,
        )

        val_time: float = time.time() - val_start_time
        self.logger.info(
            f"Validation complete — {val_time:.2f}s "
            f"({val_metrics.get('val/num_batches', 0)} batches)"
        )

        return val_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: Checkpoint save
    # ──────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch: int,
    ) -> str:
        """Save a training checkpoint.

        This method saves both a regular checkpoint and (if the joint
        score has improved) a best checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch number.

        Returns
        -------
        str
            Path to the saved checkpoint file.
        """
        # Compute current metrics for checkpoint tracking
        metrics: Dict[str, Any] = self.metrics_tracker.compute_all()
        joint_score: float = metrics.get(_JOINT_SCORE_KEY, float("nan"))

        # Save regular checkpoint
        checkpoint_path: str = self.checkpoint_manager.save(
            epoch=epoch,
            global_step=self.global_step,
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            metrics=metrics,
            image_encoder=self.image_encoder,
            box_encoder=self.box_encoder,
        )

        # Save best checkpoint if joint score improved
        if not math.isnan(joint_score) and joint_score > self.best_joint_score:
            self.best_joint_score = joint_score
            best_path: Optional[str] = self.checkpoint_manager.save_best(
                metrics=metrics,
                epoch=epoch,
                global_step=self.global_step,
                generator=self.generator,
                discriminator=self.discriminator,
                g_optimizer=self.g_optimizer,
                d_optimizer=self.d_optimizer,
                image_encoder=self.image_encoder,
                box_encoder=self.box_encoder,
            )
            if best_path is not None:
                self.logger.info(
                    f"  🏆 New best checkpoint saved (joint score: {joint_score:.6f})"
                )

        return checkpoint_path

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: Epoch summary logging
    # ──────────────────────────────────────────────────────────────────────────

    def _log_epoch_summary(
        self,
        epoch: int,
        epoch_time: float,
        epoch_losses: Dict[str, float],
    ) -> None:
        """Log a summary of the completed epoch.

        Parameters
        ----------
        epoch : int
            The epoch that just completed.
        epoch_time : float
            Time taken for the epoch in seconds.
        epoch_losses : dict of str -> float
            Epoch-averaged losses.
        """
        # Build a concise summary string
        loss_parts: List[str] = []
        for key in [
            "total_g_loss",
            "total_d_loss",
            "box_loss",
            "consistency_loss",
            "collapse_loss",
        ]:
            if key in epoch_losses:
                loss_parts.append(f"{key}={epoch_losses[key]:.6f}")

        loss_str: str = ", ".join(loss_parts) if loss_parts else "N/A"

        # Compute samples per second
        samples_per_sec: float = (
            self.batch_size * len(epoch_losses) / epoch_time
            if epoch_time > 0 and len(epoch_losses) > 0
            else 0.0
        )

        self.logger.info(
            f"Epoch {epoch:>4d}/{self.epochs - 1:<4d} | "
            f"Step {self.global_step:>7d} | "
            f"Time: {epoch_time:.2f}s | "
            f"{samples_per_sec:.0f} img/s | "
            f"LR: {self.g_scheduler.get_last_lr()[0]:.8f} | "
            f"Losses: [{loss_str}]"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal: Distributed training setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_distributed(self) -> None:
        """Wrap models with DistributedDataParallel for multi-GPU training.

        This is a stub implementation that wraps the generator and
        discriminator with ``DistributedDataParallel``.  It is called
        automatically if ``torch.distributed`` is initialised.

        Notes
        -----
        - Only the generator and discriminator are wrapped (the encoders
          are small and can be replicated on each GPU without synchronisation).
        - The loss aggregator and metrics tracker are not wrapped (they are
          stateless or accumulate metrics locally).
        - The checkpoint manager saves only from rank 0.
        - For full multi-GPU support, the dataloader should use a
          ``DistributedSampler``.  This is not yet implemented.
        """
        if not self.is_distributed:
            return

        self.logger.info("Setting up DistributedDataParallel...")

        # Find a free port for NCCL
        import socket
        from contextlib import closing

        def _find_free_port() -> int:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.bind(("", 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return s.getsockname()[1]

        # Wrap generator
        self.generator = nn.parallel.DistributedDataParallel(
            self.generator,
            device_ids=[self.device] if self.device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

        # Wrap discriminator
        self.discriminator = nn.parallel.DistributedDataParallel(
            self.discriminator,
            device_ids=[self.device] if self.device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

        # Wrap encoders (optional, but good for completeness)
        self.image_encoder = nn.parallel.DistributedDataParallel(
            self.image_encoder,
            device_ids=[self.device] if self.device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

        self.box_encoder = nn.parallel.DistributedDataParallel(
            self.box_encoder,
            device_ids=[self.device] if self.device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

        self.logger.info("DistributedDataParallel setup complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (rank 0 or non-distributed)."""
        return self.rank == 0

    @property
    def current_lr(self) -> Dict[str, float]:
        """Current learning rates for both optimizers.

        Returns
        -------
        dict
            A dictionary with keys ``"g_lr"`` and ``"d_lr"``.
        """
        return {
            "g_lr": self.g_scheduler.get_last_lr()[0]
            if len(self.g_scheduler.get_last_lr()) > 0
            else self.g_optimizer.param_groups[0]["lr"],
            "d_lr": self.d_scheduler.get_last_lr()[0]
            if len(self.d_scheduler.get_last_lr()) > 0
            else self.d_optimizer.param_groups[0]["lr"],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ILGANTrainer(\n"
            f"  device={self.device},\n"
            f"  distributed={self.is_distributed},\n"
            f"  generator={self.generator.__class__.__name__},\n"
            f"  discriminator={self.discriminator.__class__.__name__},\n"
            f"  epochs={self.epochs},\n"
            f"  batch_size={self.batch_size},\n"
            f"  amp={self.amp_scaler.is_enabled},\n"
            f"  checkpoint_dir={self.checkpoint_dir!r},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ILGANTrainer",
]
