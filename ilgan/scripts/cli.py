"""
ILGAN Command-Line Interface (CLI) — Click-based entry point for training,
evaluation, generation, device inspection, loss analysis, memory profiling,
and dataset statistics.

This module provides the ``ilgan`` Click group with subcommands for the
full ILGAN workflow:

- ``ilgan train``: Run the full training loop with configurable
  hyperparameters, checkpointing, logging, and mixed-precision support.
- ``ilgan evaluate``: Load a checkpoint and run evaluation on the
  validation set.
- ``ilgan generate``: Load a checkpoint and generate samples (images +
  bounding boxes) to disk.
- ``ilgan list-devices``: Print available CUDA devices and their memory.
- ``ilgan analyze-losses``: Run a short training loop with gradient
  logging to debug training issues.
- ``ilgan profile-memory``: Run a single forward+backward pass and
  print CUDA memory usage per module.
- ``ilgan compute-statistics``: Analyze a dataset and print statistics
  (image count, boxes per image, class distribution, etc.).

All subcommands accept CLI options that override the YAML config file
and/or the default configuration, following the precedence:

    CLI options > user config file > default config

Usage examples
--------------

**Train with defaults**::

    python -m ilgan.scripts.cli train

**Train with custom config and overrides**::

    python -m ilgan.scripts.cli train \\
        --config my_config.yaml \\
        --data-root ./my_dataset \\
        --image-size 256 \\
        --batch-size 32 \\
        --epochs 1000 \\
        --lr 0.0001 \\
        --use-wandb

**Resume from checkpoint**::

    python -m ilgan.scripts.cli train \\
        --resume ./checkpoints/checkpoint_epoch_100_step_5000.pt

**Evaluate a trained model**::

    python -m ilgan.scripts.cli evaluate \\
        --checkpoint ./checkpoints/best_checkpoint.pt \\
        --data-root ./my_dataset

**Evaluate with custom sample count and output directory**::

    python -m ilgan.scripts.cli evaluate \\
        --checkpoint ./checkpoints/best_checkpoint.pt \\
        --data-root ./my_dataset \\
        --num-samples 500 \\
        --output-dir ./my_evaluation

**Generate samples**::

    python -m ilgan.scripts.cli generate \\
        --checkpoint ./checkpoints/best_checkpoint.pt \\
        --num-samples 64 \\
        --output-dir ./generated

**Generate samples with selective saving**::

    python -m ilgan.scripts.cli generate \\
        --checkpoint ./checkpoints/best_checkpoint.pt \\
        --num-samples 100 \\
        --output-dir ./generated \\
        --save-images \\
        --no-save-boxes \\
        --save-grid

**List available CUDA devices**::

    python -m ilgan.scripts.cli list-devices

**Analyze losses with gradient logging**::

    python -m ilgan.scripts.cli analyze-losses \\
        --data-root ./my_dataset \\
        --steps 15

**Profile memory usage**::

    python -m ilgan.scripts.cli profile-memory \\
        --data-root ./my_dataset

**Compute dataset statistics**::

    python -m ilgan.scripts.cli compute-statistics \\
        --data-root ./my_dataset

**Get help**::

    python -m ilgan.scripts.cli --help
    python -m ilgan.scripts.cli train --help
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from ilgan import __version__
from ilgan.training import ILGANTrainer, build_trainer
from ilgan.utils.config import Config
from ilgan.utils.logger import Logger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "default_config.yaml",
)

_EPILOG: str = (
    "ILGAN — One-shot Image and Bounding Box Generation via "
    "Generative Adversarial Networks.  "
    "For more information, see the project documentation."
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    This function configures:

    - Python's ``random`` module.
    - NumPy's random state.
    - PyTorch's CPU and CUDA random states.
    - CUDA deterministic mode (if available).

    Parameters
    ----------
    seed : int
        The random seed to use.  Must be a non-negative integer.

    Raises
    ------
    ValueError
        If *seed* is negative.

    Notes
    -----
    - Setting ``torch.backends.cudnn.deterministic = True`` and
      ``torch.backends.cudnn.benchmark = False`` ensures deterministic
      convolution algorithms, but may reduce performance.  For maximum
      performance, consider setting ``benchmark = True`` and accepting
      non-deterministic results.
    - Full reproducibility across different hardware (e.g., GPU models)
      is not guaranteed due to differences in floating-point arithmetic.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from ilgan.utils.device import get_device_info
    device_info = get_device_info()
    if device_info['cuda_available'] or device_info['mps_available']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Ensure deterministic behavior on GPU ops
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations may not support deterministic mode
            pass


def _build_config_from_cli(
    config_path: Optional[str],
    overrides: Dict[str, Any],
) -> Config:
    """Build a :class:`Config` object from a YAML file path and CLI overrides.

    Precedence (highest to lowest):

    1. CLI overrides (dictionary of dotted keys → values).
    2. User-provided YAML config file.
    3. Default config (``ilgan/configs/default_config.yaml``).

    Parameters
    ----------
    config_path : str or None
        Path to a user-provided YAML config file.  If ``None``, only the
        default config is used (plus CLI overrides).
    overrides : dict of str -> Any
        A flat dictionary of dotted config keys to override.  These are
        applied after the config file, so they take highest precedence.

    Returns
    -------
    Config
        A fully validated :class:`Config` instance.

    Raises
    ------
    FileNotFoundError
        If *config_path* is provided but does not exist.
    ValueError
        If the resulting config fails validation.
    """
    # Start with defaults, optionally override with user config file
    if config_path is not None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}.  "
                f"Please provide a valid path to a YAML configuration file."
            )
        cfg = Config(user_config=config_path)
    else:
        cfg = Config()

    # Apply CLI overrides on top
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    return cfg


def _validate_paths_for_training(config: Config) -> None:
    """Validate that all required paths exist or can be created for training.

    This function checks:

    - The data root directory exists.
    - The checkpoint directory exists (or can be created).
    - The log directory exists (or can be created).

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.

    Raises
    ------
    FileNotFoundError
        If the data root directory does not exist.
    OSError
        If the checkpoint or log directories cannot be created.
    """
    # Data root must exist
    data_root: str = str(config["paths.data_root"])
    if data_root:
        expanded: str = os.path.expanduser(data_root)
        if not os.path.exists(expanded):
            raise FileNotFoundError(
                f"Data root directory does not exist: {expanded}.  "
                f"Please ensure the dataset is available at the specified path."
            )

    # Checkpoint and log dirs — create if missing
    for key in ("paths.checkpoint_dir", "paths.log_dir"):
        path_val: str = str(config[key])
        if path_val:
            expanded = os.path.expanduser(path_val)
            os.makedirs(expanded, exist_ok=True)


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count into a human-readable string.

    Parameters
    ----------
    num_bytes : int
        Number of bytes.

    Returns
    -------
    str
        Human-readable string (e.g., ``"8.00 GiB"``, ``"512.00 MiB"``).
    """
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GiB"
    elif num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.2f} MiB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    else:
        return f"{num_bytes} B"


# ──────────────────────────────────────────────────────────────────────────────
# Click group
# ──────────────────────────────────────────────────────────────────────────────


@click.group(
    name="ilgan",
    help="ILGAN — One-shot Image and Bounding Box Generation via GANs.",
    epilog=_EPILOG,
    invoke_without_command=False,
)
@click.version_option(
    version=__version__,
    prog_name="ilgan",
    message="ILGAN version %(version)s",
)
def cli() -> None:
    """ILGAN CLI — One-shot Image and Bounding Box Generation.

    This is the top-level Click group.  Use subcommands to train, evaluate,
    generate samples, list available devices, analyze losses, profile memory,
    or compute dataset statistics with the ILGAN model.

    Examples
    --------
    .. code-block:: bash

        # Train with defaults
        python -m ilgan.scripts.cli train

        # Train with custom config
        python -m ilgan.scripts.cli train --config my_config.yaml

        # Evaluate
        python -m ilgan.scripts.cli evaluate --checkpoint model.pt

        # Generate samples
        python -m ilgan.scripts.cli generate --checkpoint model.pt --num-samples 100

        # List CUDA devices
        python -m ilgan.scripts.cli list-devices

        # Analyze losses with gradient logging
        python -m ilgan.scripts.cli analyze-losses --data-root ./my_dataset

        # Profile memory usage
        python -m ilgan.scripts.cli profile-memory --data-root ./my_dataset

        # Compute dataset statistics
        python -m ilgan.scripts.cli compute-statistics --data-root ./my_dataset
    """
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Train command
# ──────────────────────────────────────────────────────────────────────────────


@cli.command(
    name="train",
    help="Run the full ILGAN training loop.",
    epilog=(
        "The training loop supports checkpointing (resume with --resume), "
        "mixed-precision training (--mixed-precision), gradient checkpointing "
        "(--grad-checkpoint), and Weights & Biases logging (--use-wandb).  "
        "All hyperparameters can be overridden via CLI options."
    ),
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help=(
        "Path to a YAML configuration file.  If provided, its values are "
        "merged on top of the default config.  CLI options override both.  "
        "Default: use the built-in default config."
    ),
)
@click.option(
    "--data-root",
    "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help=(
        "Root directory of the dataset (containing images/ and labels/ "
        "subdirectories).  Overrides the config file and default config."
    ),
)
@click.option(
    "--image-size",
    "-s",
    type=click.IntRange(min=16, max=1024),
    default=128,
    show_default=True,
    help=(
        "Spatial size of training images in pixels (square).  Must be a "
        "power of 2 for the generator architecture.  Default: 128."
    ),
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1, max=1024),
    default=16,
    show_default=True,
    help=(
        "Number of samples per batch.  Larger values use more GPU memory "
        "but provide more stable gradients.  Default: 16."
    ),
)
@click.option(
    "--epochs",
    "-e",
    type=click.IntRange(min=1),
    default=500,
    show_default=True,
    help=(
        "Total number of training epochs.  One epoch = one full pass over "
        "the training dataset.  Default: 500."
    ),
)
@click.option(
    "--lr",
    "-l",
    "--learning-rate",
    type=click.FloatRange(min=1e-8, max=1.0),
    default=0.0002,
    show_default=True,
    help=(
        "Learning rate for both Adam optimizers (generator and "
        "discriminator).  Default: 0.0002."
    ),
)
@click.option(
    "--latent-dim",
    "-z",
    type=click.IntRange(min=16, max=4096),
    default=256,
    show_default=True,
    help=(
        "Dimensionality of the latent noise vector z sampled from "
        "N(0, I).  Default: 256."
    ),
)
@click.option(
    "--num-classes",
    "-n",
    type=click.IntRange(min=1, max=1000),
    default=80,
    show_default=True,
    help=(
        "Number of object classes for bounding box prediction.  "
        "COCO has 80 classes.  Default: 80."
    ),
)
@click.option(
    "--max-boxes",
    "-m",
    type=click.IntRange(min=1, max=200),
    default=20,
    show_default=True,
    help=(
        "Maximum number of bounding boxes predicted per image.  "
        "Default: 20."
    ),
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./checkpoints",
    show_default=True,
    help=(
        "Directory for saving model checkpoints.  Created automatically "
        "if it does not exist.  Default: ./checkpoints."
    ),
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./logs",
    show_default=True,
    help=(
        "Directory for saving log files.  Created automatically if it "
        "does not exist.  Default: ./logs."
    ),
)
@click.option(
    "--resume",
    "-r",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help=(
        "Path to a checkpoint file (.pt) to resume training from.  "
        "If provided, the model and optimizer states are loaded and "
        "training continues from the saved epoch.  Default: None "
        "(start from scratch)."
    ),
)
@click.option(
    "--use-wandb",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable Weights & Biases (wandb) logging for experiment tracking.  "
        "Requires the wandb package to be installed.  Default: False."
    ),
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    default=True,
    show_default=True,
    help=(
        "Enable Automatic Mixed Precision (AMP) training using PyTorch's "
        "native AMP.  Reduces GPU memory usage and accelerates training "
        "on compatible hardware.  Default: True."
    ),
)
@click.option(
    "--grad-checkpoint",
    is_flag=True,
    default=True,
    show_default=True,
    help=(
        "Enable gradient checkpointing (activation checkpointing) to "
        "reduce GPU memory usage during training.  Saves memory at the "
        "cost of a small amount of computation.  Default: True."
    ),
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0, max=64),
    default=4,
    show_default=True,
    help=(
        "Number of subprocess workers for data loading.  Set to 0 for "
        "in-process loading.  Default: 4."
    ),
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help=(
        "Random seed for reproducibility.  Sets Python, NumPy, and "
        "PyTorch random states.  Default: 42."
    ),
)
@click.option(
    "--n-critic",
    type=click.IntRange(min=1, max=20),
    default=None,
    help=(
        "Number of discriminator updates per generator update (WGAN-GP "
        "n_critic ratio).  Overrides the config file.  Default: use "
        "config value (typically 5)."
    ),
)
@click.option(
    "--gradient-accumulation",
    type=click.IntRange(min=1, max=64),
    default=None,
    help=(
        "Number of gradient accumulation steps.  Simulates a larger "
        "batch size without increasing GPU memory.  Overrides the config "
        "file.  Default: use config value (typically 1)."
    ),
)
@click.option(
    "--clip-grad-norm",
    type=click.FloatRange(min=0.0, max=100.0),
    default=None,
    help=(
        "Maximum gradient norm for gradient clipping.  Set to 0 to "
        "disable clipping.  Overrides the config file.  Default: use "
        "config value (typically 1.0)."
    ),
)
@click.option(
    "--log-interval",
    type=click.IntRange(min=1, max=10000),
    default=None,
    help=(
        "Number of training steps between console log messages.  "
        "Overrides the config file.  Default: use config value "
        "(typically 10)."
    ),
)
@click.option(
    "--save-interval",
    type=click.IntRange(min=1, max=1000),
    default=None,
    help=(
        "Number of epochs between checkpoint saves.  Overrides the "
        "config file.  Default: use config value (typically 50)."
    ),
)
@click.option(
    "--eval-interval",
    type=click.IntRange(min=1, max=1000),
    default=None,
    help=(
        "Number of epochs between validation runs.  Overrides the "
        "config file.  Default: use config value (typically 100)."
    ),
)
@click.option(
    "--gen-base-channels",
    type=click.IntRange(min=8, max=512),
    default=None,
    help=(
        "Base channel count for the generator (doubles per block).  "
        "Overrides the config file.  Default: use config value "
        "(typically 64)."
    ),
)
@click.option(
    "--disc-base-channels",
    type=click.IntRange(min=8, max=512),
    default=None,
    help=(
        "Base channel count for the discriminator.  Overrides the "
        "config file.  Default: use config value (typically 64)."
    ),
)
@click.option(
    "--num-attention-heads",
    type=click.IntRange(min=1, max=32),
    default=None,
    help=(
        "Number of self-attention heads in attention blocks.  "
        "Overrides the config file.  Default: use config value "
        "(typically 8)."
    ),
)
@click.option(
    "--adv-weight",
    type=click.FloatRange(min=0.0, max=100.0),
    default=None,
    help=(
        "Weight for the adversarial (GAN) loss term.  Overrides the "
        "config file.  Default: use config value (typically 1.0)."
    ),
)
@click.option(
    "--box-weight",
    type=click.FloatRange(min=0.0, max=1000.0),
    default=None,
    help=(
        "Weight for the bounding box regression loss term.  Overrides "
        "the config file.  Default: use config value (typically 5.0)."
    ),
)
@click.option(
    "--diversity-weight",
    type=click.FloatRange(min=0.0, max=100.0),
    default=None,
    help=(
        "Weight for the representation diversity / collapse penalty.  "
        "Overrides the config file.  Default: use config value "
        "(typically 0.1)."
    ),
)
@click.option(
    "--consistency-weight",
    type=click.FloatRange(min=0.0, max=100.0),
    default=None,
    help=(
        "Weight for the latent-image-box consistency constraint.  "
        "Overrides the config file.  Default: use config value "
        "(typically 0.5)."
    ),
)
@click.option(
    "--gp-weight",
    type=click.FloatRange(min=0.0, max=100.0),
    default=None,
    help=(
        "Weight for the gradient penalty (WGAN-GP).  Overrides the "
        "config file.  Default: use config value (typically 10.0)."
    ),
)
@click.option(
    "--beta1",
    type=click.FloatRange(min=0.0, max=0.999),
    default=None,
    help=(
        "Adam beta1 hyperparameter.  Set to 0.0 for WGAN-style "
        "training.  Overrides the config file.  Default: use config "
        "value (typically 0.0)."
    ),
)
@click.option(
    "--beta2",
    type=click.FloatRange(min=0.0, max=0.9999),
    default=None,
    help=(
        "Adam beta2 hyperparameter.  Overrides the config file.  "
        "Default: use config value (typically 0.9)."
    ),
)
@click.option(
    "--no-mixed-precision",
    is_flag=True,
    default=False,
    help=(
        "Disable mixed precision training.  Overrides --mixed-precision.  "
        "Use this flag if you encounter numerical instability with AMP."
    ),
)
@click.option(
    "--no-grad-checkpoint",
    is_flag=True,
    default=False,
    help=(
        "Disable gradient checkpointing.  Overrides --grad-checkpoint.  "
        "Use this flag if you have sufficient GPU memory and want to "
        "avoid the small computational overhead."
    ),
)
def train(
    config: Optional[str],
    data_root: Optional[str],
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    latent_dim: int,
    num_classes: int,
    max_boxes: int,
    checkpoint_dir: str,
    log_dir: str,
    resume: Optional[str],
    use_wandb: bool,
    mixed_precision: bool,
    grad_checkpoint: bool,
    num_workers: int,
    seed: int,
    n_critic: Optional[int],
    gradient_accumulation: Optional[int],
    clip_grad_norm: Optional[float],
    log_interval: Optional[int],
    save_interval: Optional[int],
    eval_interval: Optional[int],
    gen_base_channels: Optional[int],
    disc_base_channels: Optional[int],
    num_attention_heads: Optional[int],
    adv_weight: Optional[float],
    box_weight: Optional[float],
    diversity_weight: Optional[float],
    consistency_weight: Optional[float],
    gp_weight: Optional[float],
    beta1: Optional[float],
    beta2: Optional[float],
    no_mixed_precision: bool,
    no_grad_checkpoint: bool,
) -> None:
    """Run the full ILGAN training loop.

    This command orchestrates the complete training pipeline:

    1. **Configuration**: Builds a :class:`Config` object from the default
       config, optionally overridden by a user-provided YAML file, then
       overridden by CLI options.

    2. **Reproducibility**: Sets random seeds for Python, NumPy, and
       PyTorch.

    3. **Logging**: Creates a :class:`Logger` instance for console and
       file output.

    4. **Training**: Creates an :class:`ILGANTrainer` via the
       :func:`build_trainer` factory and calls :meth:`ILGANTrainer.train`.

    The training loop includes:

    - WGAN-GP adversarial training with configurable n_critic ratio.
    - Gradient accumulation for effective large-batch training.
    - Mixed precision (AMP) for reduced GPU memory and faster training.
    - Gradient checkpointing for further memory savings.
    - Automatic checkpointing with best-model tracking.
    - Periodic validation with comprehensive metrics (FID, mAP, joint score).
    - Weights & Biases logging (optional).

    \f
    Parameters
    ----------
    config : str or None
        Path to a YAML config file.
    data_root : str or None
        Dataset root directory.
    image_size : int
        Image size in pixels.
    batch_size : int
        Batch size.
    epochs : int
        Number of epochs.
    lr : float
        Learning rate.
    latent_dim : int
        Latent vector dimension.
    num_classes : int
        Number of object classes.
    max_boxes : int
        Maximum boxes per image.
    checkpoint_dir : str
        Checkpoint directory.
    log_dir : str
        Log directory.
    resume : str or None
        Path to checkpoint to resume from.
    use_wandb : bool
        Enable wandb logging.
    mixed_precision : bool
        Enable AMP.
    grad_checkpoint : bool
        Enable gradient checkpointing.
    num_workers : int
        Data loading workers.
    seed : int
        Random seed.
    n_critic : int or None
        Discriminator updates per generator update.
    gradient_accumulation : int or None
        Gradient accumulation steps.
    clip_grad_norm : float or None
        Gradient clipping norm.
    log_interval : int or None
        Logging interval in steps.
    save_interval : int or None
        Checkpoint save interval in epochs.
    eval_interval : int or None
        Validation interval in epochs.
    gen_base_channels : int or None
        Generator base channels.
    disc_base_channels : int or None
        Discriminator base channels.
    num_attention_heads : int or None
        Attention heads.
    adv_weight : float or None
        Adversarial loss weight.
    box_weight : float or None
        Box regression loss weight.
    diversity_weight : float or None
        Diversity loss weight.
    consistency_weight : float or None
        Consistency loss weight.
    gp_weight : float or None
        Gradient penalty weight.
    beta1 : float or None
        Adam beta1.
    beta2 : float or None
        Adam beta2.
    no_mixed_precision : bool
        Flag to disable mixed precision.
    no_grad_checkpoint : bool
        Flag to disable gradient checkpointing.
    """
    # ── Resolve conflicting flags ──────────────────────────────────────
    # If --no-mixed-precision is set, it overrides --mixed-precision
    if no_mixed_precision:
        mixed_precision = False
    # If --no-grad-checkpoint is set, it overrides --grad-checkpoint
    if no_grad_checkpoint:
        grad_checkpoint = False

    # ── Build overrides dictionary ──────────────────────────────────────
    overrides: Dict[str, Any] = {}

    # Map CLI parameter names to config dotted keys
    param_to_config_key: Dict[str, Tuple[str, Any]] = {
        "data_root": ("paths.data_root", data_root),
        "image_size": ("data.image_size", image_size),
        "batch_size": ("data.batch_size", batch_size),
        "num_workers": ("data.num_workers", num_workers),
        "latent_dim": ("model.latent_dim", latent_dim),
        "num_classes": ("model.num_classes", num_classes),
        "max_boxes": ("model.max_boxes", max_boxes),
        "epochs": ("training.epochs", epochs),
        "lr": ("training.learning_rate", lr),
        "mixed_precision": ("training.use_mixed_precision", mixed_precision),
        "grad_checkpoint": ("training.grad_checkpoint", grad_checkpoint),
        "use_wandb": ("logging.use_wandb", use_wandb),
        "checkpoint_dir": ("paths.checkpoint_dir", checkpoint_dir),
        "log_dir": ("paths.log_dir", log_dir),
        "n_critic": ("training.n_critic", n_critic),
        "gradient_accumulation": ("training.gradient_accumulation_steps", gradient_accumulation),
        "clip_grad_norm": ("training.clip_grad_norm", clip_grad_norm),
        "log_interval": ("logging.log_interval", log_interval),
        "save_interval": ("logging.save_interval", save_interval),
        "eval_interval": ("logging.eval_interval", eval_interval),
        "gen_base_channels": ("model.gen_base_channels", gen_base_channels),
        "disc_base_channels": ("model.disc_base_channels", disc_base_channels),
        "num_attention_heads": ("model.num_attention_heads", num_attention_heads),
        "adv_weight": ("loss.adv_weight", adv_weight),
        "box_weight": ("loss.box_weight", box_weight),
        "diversity_weight": ("loss.diversity_weight", diversity_weight),
        "consistency_weight": ("loss.consistency_weight", consistency_weight),
        "gp_weight": ("loss.gp_weight", gp_weight),
        "beta1": ("training.beta1", beta1),
        "beta2": ("training.beta2", beta2),
    }

    for param_name, (config_key, value) in param_to_config_key.items():
        if value is not None:
            overrides[config_key] = value

    # ── Build Config ────────────────────────────────────────────────────
    click.echo("=" * 72)
    click.echo("  ILGAN — One-shot Image and Bounding Box Generation")
    click.echo(f"  Version: {__version__}")
    click.echo("=" * 72)
    click.echo()

    click.echo("Building configuration...")
    try:
        cfg: Config = _build_config_from_cli(
            config_path=config,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error building configuration: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Config source: {config or 'default'}")
    click.echo(f"  Data root: {cfg['paths.data_root']}")
    click.echo(f"  Image size: {cfg['data.image_size']}")
    click.echo(f"  Batch size: {cfg['data.batch_size']}")
    click.echo(f"  Epochs: {cfg['training.epochs']}")
    click.echo(f"  Learning rate: {cfg['training.learning_rate']}")
    click.echo(f"  Latent dim: {cfg['model.latent_dim']}")
    click.echo(f"  Mixed precision: {cfg['training.use_mixed_precision']}")
    click.echo(f"  Gradient checkpointing: {cfg['training.grad_checkpoint']}")
    click.echo(f"  Wandb: {cfg['logging.use_wandb']}")
    click.echo()

    # ── Validate paths ──────────────────────────────────────────────────
    try:
        _validate_paths_for_training(cfg)
    except (FileNotFoundError, OSError) as e:
        click.echo(f"Error validating paths: {e}", err=True)
        sys.exit(1)

    # ── Set random seed ───────────────────────────────────────────────────
    click.echo(f"Setting random seed to {seed}...")
    try:
        _set_seed(seed)
    except ValueError as e:
        click.echo(f"Error setting seed: {e}", err=True)
        sys.exit(1)
    click.echo("  Seed set successfully.")
    click.echo()

    # ── Create Logger ───────────────────────────────────────────────────
    click.echo("Initialising logger...")
    logger: Logger = Logger(
        name="ilgan",
        log_dir=str(cfg["paths.log_dir"]),
        level="INFO",
    )
    click.echo(f"  Log directory: {cfg['paths.log_dir']}")
    click.echo(f"  Log file: {logger.log_file_path or 'console only'}")
    click.echo()

    # ── Log the configuration ────────────────────────────────────────────
    logger.info("ILGAN Training — Configuration")
    logger.info(f"  Config source: {config or 'default'}")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Data root: {cfg['paths.data_root']}")
    logger.info(f"  Checkpoint dir: {cfg['paths.checkpoint_dir']}")
    logger.info(f"  Log dir: {cfg['paths.log_dir']}")
    logger.info(f"  Image size: {cfg['data.image_size']}")
    logger.info(f"  Batch size: {cfg['data.batch_size']}")
    logger.info(f"  Epochs: {cfg['training.epochs']}")
    logger.info(f"  Learning rate: {cfg['training.learning_rate']}")
    logger.info(f"  Latent dim: {cfg['model.latent_dim']}")
    logger.info(f"  Num classes: {cfg['model.num_classes']}")
    logger.info(f"  Max boxes: {cfg['model.max_boxes']}")
    logger.info(f"  Mixed precision: {cfg['training.use_mixed_precision']}")
    logger.info(f"  Gradient checkpointing: {cfg['training.grad_checkpoint']}")
    logger.info(f"  Wandb: {cfg['logging.use_wandb']}")
    logger.info(f"  Num workers: {cfg['data.num_workers']}")

    # ── Handle resume ───────────────────────────────────────────────────
    if resume is not None:
        logger.info(f"Resume checkpoint provided: {resume}")
        # The trainer's load_or_initialize will handle the actual loading.
        # We store the resume path in the config so the trainer can use it.
        cfg["paths.resume_checkpoint"] = resume
        click.echo(f"  Will resume from: {resume}")
    else:
        click.echo("  Starting training from scratch (no resume checkpoint).")
    click.echo()

    # ── Build and run trainer ───────────────────────────────────────────
    click.echo("Building ILGANTrainer...")
    try:
        trainer: ILGANTrainer = build_trainer(config=cfg, logger=logger)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to build trainer: {e}")
        click.echo(f"Error building trainer: {e}", err=True)
        sys.exit(1)

    click.echo("  Trainer built successfully.")
    click.echo()

    # ── Run training ────────────────────────────────────────────────────
    click.echo("Starting training...")
    click.echo("=" * 72)
    click.echo()

    try:
        results: Dict[str, Any] = trainer.train()
    except KeyboardInterrupt:
        click.echo()
        click.echo("Training interrupted by user (Ctrl+C).  Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        click.echo(f"Training failed: {e}", err=True)
        sys.exit(1)

    # ── Print final summary ──────────────────────────────────────────────
    click.echo()
    click.echo("=" * 72)
    click.echo("  Training Complete — Summary")
    click.echo("=" * 72)
    click.echo(f"  Final epoch:      {results.get('final_epoch', 'N/A')}")
    click.echo(f"  Total steps:      {results.get('final_global_step', 'N/A')}")
    click.echo(f"  Best joint score: {results.get('best_joint_score', 'N/A'):.6f}")
    click.echo(f"  Training time:    {results.get('training_time_seconds', 0):.2f}s "
               f"({results.get('training_time_seconds', 0) / 60:.2f}min)")
    if results.get("best_checkpoint_path"):
        click.echo(f"  Best checkpoint:  {results['best_checkpoint_path']}")
    if results.get("final_checkpoint_path"):
        click.echo(f"  Final checkpoint: {results['final_checkpoint_path']}")
    click.echo("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate command
# ──────────────────────────────────────────────────────────────────────────────


@cli.command(
    name="evaluate",
    help="Load a checkpoint and run evaluation on the validation set.",
    epilog=(
        "The evaluation runs the full validation pipeline: image generation, "
        "bounding box prediction, and metric computation (FID, mAP, joint "
        "score, etc.).  Results are logged to console and file.  "
        "Evaluation results (sample grids, metrics JSON) are saved to the "
        "output directory."
    ),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a model checkpoint file (.pt) to load for evaluation.",
)
@click.option(
    "--data-root",
    "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=True,
    help=(
        "Root directory of the dataset (containing images/ and labels/ "
        "subdirectories).  Required for evaluation."
    ),
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help=(
        "Path to a YAML configuration file.  If not provided, the default "
        "config is used.  The checkpoint's own config is not used for "
        "evaluation (only model weights are loaded)."
    ),
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1, max=1024),
    default=16,
    show_default=True,
    help="Batch size for evaluation.  Default: 16.",
)
@click.option(
    "--num-samples",
    "-n",
    type=click.IntRange(min=1, max=1000000),
    default=1000,
    show_default=True,
    help=(
        "Number of validation samples to evaluate.  If the validation set "
        "has fewer samples, all available samples are used.  Default: 1000."
    ),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./evaluation",
    show_default=True,
    help=(
        "Directory where evaluation results will be saved (sample grids, "
        "metrics JSON, etc.).  Created automatically if it does not exist.  "
        "Default: ./evaluation."
    ),
)
@click.option(
    "--num-workers",
    type=click.IntRange(min=0, max=64),
    default=4,
    show_default=True,
    help="Data loading workers.  Default: 4.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help="Random seed for reproducibility.  Default: 42.",
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./logs",
    show_default=True,
    help="Directory for log output.  Default: ./logs.",
)
def evaluate(
    checkpoint: str,
    data_root: str,
    config: Optional[str],
    batch_size: int,
    num_samples: int,
    output_dir: str,
    num_workers: int,
    seed: int,
    log_dir: str,
) -> None:
    """Load a checkpoint and run evaluation on the validation set.

    This command:

    1. Builds a configuration from the default config (optionally
       overridden by a user config file and CLI options).
    2. Sets random seeds for reproducibility.
    3. Creates a logger.
    4. Builds an :class:`ILGANTrainer` and calls
       :meth:`ILGANTrainer.evaluate` with the given checkpoint.
    5. Saves evaluation results (metrics JSON, sample grid) to the
       output directory.

    \f
    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint file.
    data_root : str
        Dataset root directory.
    config : str or None
        Path to a YAML config file.
    batch_size : int
        Batch size.
    num_samples : int
        Number of validation samples to evaluate.
    output_dir : str
        Directory for evaluation results.
    num_workers : int
        Data loading workers.
    seed : int
        Random seed.
    log_dir : str
        Log directory.
    """
    # ── Build overrides ─────────────────────────────────────────────────
    overrides: Dict[str, Any] = {
        "paths.data_root": data_root,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "paths.log_dir": log_dir,
    }

    # ── Build Config ────────────────────────────────────────────────────
    click.echo("=" * 72)
    click.echo("  ILGAN — Evaluation")
    click.echo(f"  Version: {__version__}")
    click.echo("=" * 72)
    click.echo()

    click.echo("Building configuration for evaluation...")
    try:
        cfg: Config = _build_config_from_cli(
            config_path=config,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error building configuration: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Checkpoint: {checkpoint}")
    click.echo(f"  Data root:  {data_root}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  Samples:    {num_samples}")
    click.echo(f"  Output dir: {output_dir}")
    click.echo()

    # ── Set seed ────────────────────────────────────────────────────────
    _set_seed(seed)

    # ── Create Logger ───────────────────────────────────────────────────
    logger: Logger = Logger(
        name="ilgan",
        log_dir=str(cfg["paths.log_dir"]),
        level="INFO",
    )

    # ── Create output directory ──────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── Build trainer and evaluate ───────────────────────────────────────
    click.echo(f"Loading checkpoint: {checkpoint}")
    click.echo("Building trainer...")

    try:
        trainer: ILGANTrainer = build_trainer(config=cfg, logger=logger)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to build trainer: {e}")
        click.echo(f"Error building trainer: {e}", err=True)
        sys.exit(1)

    click.echo("Running evaluation...")
    try:
        metrics: Dict[str, Any] = trainer.evaluate(checkpoint_path=checkpoint)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo(f"Evaluation failed: {e}", err=True)
        sys.exit(1)

    # ── Save metrics to JSON ────────────────────────────────────────────
    metrics_path: str = os.path.join(output_dir, "evaluation_metrics.json")
    try:
        # Convert non-serialisable values (e.g., NaN, Inf) to strings
        serialisable_metrics: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                if math.isnan(value):
                    serialisable_metrics[key] = "NaN"
                elif math.isinf(value):
                    serialisable_metrics[key] = "Inf" if value > 0 else "-Inf"
                else:
                    serialisable_metrics[key] = round(value, 6)
            elif isinstance(value, (int, str, bool)):
                serialisable_metrics[key] = value
            elif isinstance(value, (list, tuple)):
                serialisable_metrics[key] = [str(v) for v in value]
            else:
                serialisable_metrics[key] = str(value)

        with open(metrics_path, "w") as f:
            json.dump(serialisable_metrics, f, indent=2, sort_keys=True)
        click.echo(f"  Metrics saved to: {metrics_path}")
    except Exception as e:
        logger.warning(f"Failed to save metrics JSON: {e}")

    # ── Print results ───────────────────────────────────────────────────
    click.echo()
    click.echo("=" * 72)
    click.echo("  Evaluation Results")
    click.echo("=" * 72)

    # Group metrics by category for cleaner display
    image_metrics = {k: v for k, v in metrics.items() if "image" in k or "fid" in k.lower()}
    box_metrics = {k: v for k, v in metrics.items() if "box" in k or "map" in k.lower() or "giou" in k.lower()}
    loss_metrics = {k: v for k, v in metrics.items() if "loss" in k or "val/" in k}
    joint_metrics = {k: v for k, v in metrics.items() if "joint" in k.lower()}
    other_metrics = {k: v for k, v in metrics.items()
                     if k not in image_metrics and k not in box_metrics
                     and k not in loss_metrics and k not in joint_metrics}

    if image_metrics:
        click.echo("  ┌─ Image Metrics")
        for key, value in image_metrics.items():
            if isinstance(value, float):
                click.echo(f"  │  {key:30s}: {value:.6f}")
            else:
                click.echo(f"  │  {key:30s}: {value}")

    if box_metrics:
        click.echo("  ├─ Box Metrics")
        for key, value in box_metrics.items():
            if isinstance(value, float):
                click.echo(f"  │  {key:30s}: {value:.6f}")
            else:
                click.echo(f"  │  {key:30s}: {value}")

    if loss_metrics:
        click.echo("  ├─ Loss Metrics")
        for key, value in loss_metrics.items():
            if isinstance(value, float):
                click.echo(f"  │  {key:30s}: {value:.6f}")
            else:
                click.echo(f"  │  {key:30s}: {value}")

    if joint_metrics:
        click.echo("  ├─ Joint Score")
        for key, value in joint_metrics.items():
            if isinstance(value, float):
                click.echo(f"  │  {key:30s}: {value:.6f}")
            else:
                click.echo(f"  │  {key:30s}: {value}")

    if other_metrics:
        click.echo("  └─ Other Metrics")
        for key, value in other_metrics.items():
            if isinstance(value, float):
                click.echo(f"     {key:30s}: {value:.6f}")
            else:
                click.echo(f"     {key:30s}: {value}")

    click.echo("=" * 72)
    click.echo(f"  Results saved to: {os.path.abspath(output_dir)}")
    click.echo("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Generate command
# ──────────────────────────────────────────────────────────────────────────────


@cli.command(
    name="generate",
    help="Load a checkpoint and generate samples (images + bounding boxes).",
    epilog=(
        "Generated images are saved as PNG files and bounding boxes as "
        "YOLO-format text files.  The output directory structure is:\n"
        "  output_dir/\n"
        "    images/  sample_000000.png, sample_000001.png, ...\n"
        "    labels/  sample_000000.txt, sample_000001.txt, ...\n"
        "    grid/    sample_grid.png (if --save-grid is enabled)\n"
        "Each label file contains one line per detected object:\n"
        "  class_id cx cy w h\n"
        "where (cx, cy, w, h) are normalised to [0, 1].\n"
        "Use --no-save-images, --no-save-boxes, or --no-save-grid to "
        "selectively disable saving of specific output types."
    ),
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a model checkpoint file (.pt) to load for generation.",
)
@click.option(
    "--num-samples",
    type=click.IntRange(min=1, max=100000),
    default=64,
    show_default=True,
    help="Number of samples to generate.  Default: 64.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./generated",
    show_default=True,
    help="Directory where generated samples will be saved.  Default: ./generated.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a YAML configuration file.  Default: use built-in default.",
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1, max=1024),
    default=16,
    show_default=True,
    help="Batch size for generation.  Default: 16.",
)
@click.option(
    "--save-images/--no-save-images",
    is_flag=True,
    default=True,
    show_default=True,
    help=(
        "Save individual generated images as PNG files.  "
        "Use --no-save-images to disable.  Default: True."
    ),
)
@click.option(
    "--save-boxes/--no-save-boxes",
    is_flag=True,
    default=True,
    show_default=True,
    help=(
        "Save YOLO-format label files for each generated sample.  "
        "Use --no-save-boxes to disable.  Default: True."
    ),
)
@click.option(
    "--save-grid/--no-save-grid",
    is_flag=True,
    default=True,
    show_default=True,
    help=(
        "Save a grid of generated images as a single PNG file.  "
        "Use --no-save-grid to disable.  Default: True."
    ),
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help="Random seed for reproducibility.  Default: 42.",
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./logs",
    show_default=True,
    help="Directory for log output.  Default: ./logs.",
)
def generate(
    checkpoint: str,
    num_samples: int,
    output_dir: str,
    config: Optional[str],
    batch_size: int,
    save_images: bool,
    save_boxes: bool,
    save_grid: bool,
    seed: int,
    log_dir: str,
) -> None:
    """Load a checkpoint and generate samples (images + bounding boxes).

    This command:

    1. Builds a configuration from the default config (optionally
       overridden by a user config file and CLI options).
    2. Sets random seeds for reproducibility.
    3. Creates a logger.
    4. Builds an :class:`ILGANTrainer` and calls
       :meth:`ILGANTrainer.generate` with the given checkpoint, number
       of samples, and output directory.
    5. Optionally saves individual images, YOLO-format label files, and/or
       a grid of generated images based on the ``--save-*`` flags.

    \f
    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint file.
    num_samples : int
        Number of samples to generate.
    output_dir : str
        Output directory for generated samples.
    config : str or None
        Path to a YAML config file.
    batch_size : int
        Batch size for generation.
    save_images : bool
        Whether to save individual image PNG files.
    save_boxes : bool
        Whether to save YOLO-format label files.
    save_grid : bool
        Whether to save a grid of generated images.
    seed : int
        Random seed.
    log_dir : str
        Log directory.
    """
    # ── Build overrides ─────────────────────────────────────────────────
    overrides: Dict[str, Any] = {
        "data.batch_size": batch_size,
        "paths.log_dir": log_dir,
    }

    # ── Build Config ────────────────────────────────────────────────────
    click.echo("=" * 72)
    click.echo("  ILGAN — Generation")
    click.echo(f"  Version: {__version__}")
    click.echo("=" * 72)
    click.echo()

    click.echo("Building configuration for generation...")
    try:
        cfg: Config = _build_config_from_cli(
            config_path=config,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error building configuration: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Checkpoint:   {checkpoint}")
    click.echo(f"  Samples:      {num_samples}")
    click.echo(f"  Batch size:   {batch_size}")
    click.echo(f"  Output dir:   {output_dir}")
    click.echo(f"  Save images:  {save_images}")
    click.echo(f"  Save boxes:   {save_boxes}")
    click.echo(f"  Save grid:    {save_grid}")
    click.echo()

    # ── Set seed ────────────────────────────────────────────────────────
    _set_seed(seed)

    # ── Create Logger ───────────────────────────────────────────────────
    logger: Logger = Logger(
        name="ilgan",
        log_dir=str(cfg["paths.log_dir"]),
        level="INFO",
    )

    # ── Build trainer ──────────────────────────────────────────────────
    click.echo(f"Loading checkpoint: {checkpoint}")
    click.echo("Building trainer...")

    try:
        trainer: ILGANTrainer = build_trainer(config=cfg, logger=logger)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to build trainer: {e}")
        click.echo(f"Error building trainer: {e}", err=True)
        sys.exit(1)

    # ── Load checkpoint ──────────────────────────────────────────────────
    click.echo("Loading model weights from checkpoint...")
    try:
        trainer.checkpoint_manager.load(
            checkpoint_path=checkpoint,
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            g_optimizer=trainer.g_optimizer,
            d_optimizer=trainer.d_optimizer,
            image_encoder=trainer.image_encoder,
            box_encoder=trainer.box_encoder,
        )
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        click.echo(f"Error loading checkpoint: {e}", err=True)
        sys.exit(1)

    # ── Create output directories ───────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    images_dir: str = os.path.join(output_dir, "images")
    labels_dir: str = os.path.join(output_dir, "labels")
    grid_dir: str = os.path.join(output_dir, "grid")

    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    if save_boxes:
        os.makedirs(labels_dir, exist_ok=True)
    if save_grid:
        os.makedirs(grid_dir, exist_ok=True)

    # ── Generate samples ────────────────────────────────────────────────
    trainer.generator.eval()
    saved_paths: List[str] = []
    latent_dim: int = int(cfg["model.latent_dim"])
    device: torch.device = trainer.device
    conf_threshold: float = float(getattr(cfg.model, "conf_threshold", 0.3))

    # Generate in batches to avoid OOM
    effective_batch_size: int = min(batch_size, num_samples)
    num_batches: int = math.ceil(num_samples / effective_batch_size)

    # Collect all generated images for grid
    all_fake_images: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Determine batch size for this iteration
            remaining: int = num_samples - batch_idx * effective_batch_size
            current_batch_size: int = min(effective_batch_size, remaining)

            # Sample latent vectors
            z: torch.Tensor = torch.randn(
                current_batch_size, latent_dim, device=device
            )

            # Generate
            gen_outputs: Dict[str, Any] = trainer.generator(z)

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

            # Collect for grid
            if save_grid:
                all_fake_images.append(images_01.cpu())

            # Save each sample
            for i in range(current_batch_size):
                sample_idx: int = batch_idx * effective_batch_size + i

                # Save individual image
                if save_images:
                    img_filename: str = f"sample_{sample_idx:06d}.png"
                    img_path: str = os.path.join(images_dir, img_filename)
                    save_image(images_01[i], img_path)
                    saved_paths.append(img_path)

                # Save YOLO-format label file
                if save_boxes:
                    label_filename: str = f"sample_{sample_idx:06d}.txt"
                    label_path: str = os.path.join(labels_dir, label_filename)

                    boxes_i: torch.Tensor = pred_boxes[i]  # [N, 4]
                    labels_i: torch.Tensor = pred_labels[i]  # [N]
                    confs_i: torch.Tensor = confidences[i].squeeze(-1)  # [N]

                    # Filter by confidence threshold
                    valid_mask: torch.Tensor = confs_i > conf_threshold

                    with open(label_path, "w") as f:
                        for j in range(valid_mask.shape[0]):
                            if valid_mask[j].item():
                                cx, cy, w, h = boxes_i[j].tolist()
                                cls_id: int = labels_i[j].item()
                                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                    saved_paths.append(label_path)

            # Log progress
            click.echo(
                f"  Generated batch {batch_idx + 1}/{num_batches} "
                f"({min((batch_idx + 1) * effective_batch_size, num_samples)}/{num_samples})"
            )

    # ── Save grid of generated images ────────────────────────────────────
    grid_path: Optional[str] = None
    if save_grid and all_fake_images:
        try:
            all_images: torch.Tensor = torch.cat(all_fake_images, dim=0)  # [N, 3, H, W]

            # Determine grid dimensions (roughly square)
            n_total: int = all_images.size(0)
            nrow: int = int(math.ceil(math.sqrt(n_total)))
            ncol: int = int(math.ceil(n_total / nrow))

            grid: torch.Tensor = make_grid(
                all_images,
                nrow=nrow,
                padding=2,
                normalize=False,
                pad_value=1.0,
            )

            grid_filename: str = "sample_grid.png"
            grid_path = os.path.join(grid_dir, grid_filename)
            save_image(grid, grid_path)
            saved_paths.append(grid_path)
            click.echo(f"  Sample grid saved to: {grid_path}")
        except Exception as e:
            logger.warning(f"Failed to save sample grid: {e}")

    # ── Print results ───────────────────────────────────────────────────
    num_images_saved: int = len([p for p in saved_paths if p.endswith(".png")])
    num_labels_saved: int = len([p for p in saved_paths if p.endswith(".txt")])

    click.echo()
    click.echo("=" * 72)
    click.echo("  Generation Complete")
    click.echo("=" * 72)
    click.echo(f"  Samples generated: {num_samples}")
    click.echo(f"  Images saved:     {num_images_saved}")
    click.echo(f"  Labels saved:     {num_labels_saved}")
    if grid_path:
        click.echo(f"  Grid saved:       {grid_path}")
    click.echo(f"  Output directory: {os.path.abspath(output_dir)}")
    click.echo("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# List Devices command
# ──────────────────────────────────────────────────────────────────────────────


@cli.command(
    name="list-devices",
    help="Print available CUDA devices and their memory capacity.",
    epilog=(
        "This command queries PyTorch's CUDA runtime to list all available "
        "GPU devices, their names, compute capabilities, total memory, "
        "and current memory usage.  It is useful for verifying GPU "
        "availability before running training or evaluation."
    ),
)
def list_devices() -> None:
    """Print available CUDA devices and their memory capacity.

    This command queries PyTorch's CUDA runtime to list all available
    GPU devices with the following information per device:

    - **Device index** (``cuda:0``, ``cuda:1``, etc.)
    - **Device name** (e.g., ``NVIDIA GeForce RTX 3090``)
    - **Compute capability** (e.g., ``8.6``)
    - **Total memory** (in GiB)
    - **Free memory** (in GiB, at the time of query)
    - **Used memory** (in GiB, at the time of query)
    - **Memory utilisation** (percentage used)

    If no CUDA devices are available, a message is printed indicating
    that only the CPU is available.

    The command also prints a summary of the PyTorch build (CUDA version,
    whether CUDA is available, number of devices).

    Examples
    --------
    .. code-block:: bash

        $ python -m ilgan.scripts.cli list-devices

        ╔══════════════════════════════════════════════════════════════════════╗
        ║  ILGAN — CUDA Device Information                                   ║
        ╚══════════════════════════════════════════════════════════════════════╝

        PyTorch CUDA available:  True
        CUDA version:            11.7
        Number of devices:       2

        ┌─ Device 0 ──────────────────────────────────────────────────────────┐
        │  Name:              NVIDIA GeForce RTX 3090                         │
        │  Compute Capability: 8.6                                           │
        │  Total Memory:      24.00 GiB                                      │
        │  Free Memory:       18.23 GiB                                      │
        │  Used Memory:        5.77 GiB                                      │
        │  Utilisation:       24.04%                                         │
        └─────────────────────────────────────────────────────────────────────┘

        ...
    """
    click.echo()
    click.echo("╔" + "═" * 70 + "╗")
    click.echo("║  ILGAN — CUDA Device Information" + " " * 37 + "║")
    click.echo("╚" + "═" * 70 + "╝")
    click.echo()

    # ── PyTorch build info ───────────────────────────────────────────────
    click.echo(f"  PyTorch version:      {torch.__version__}")
    click.echo(f"  CUDA available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"  CUDA version:         {torch.version.cuda}")
        click.echo(f"  Number of devices:    {torch.cuda.device_count()}")
    else:
        click.echo(f"  Number of devices:    0")
    click.echo()

    # ── No CUDA available ────────────────────────────────────────────────
    if not torch.cuda.is_available():
        click.echo("  ⚠  No CUDA-capable devices detected.")
        click.echo("  Training and evaluation will use the CPU.")
        click.echo("  For GPU acceleration, ensure:")
        click.echo("    - A compatible NVIDIA GPU is installed")
        click.echo("    - CUDA toolkit and drivers are properly configured")
        click.echo("    - PyTorch is installed with CUDA support")
        click.echo()
        click.echo("  CPU information:")
        import platform
        click.echo(f"    Processor: {platform.processor() or 'Unknown'}")
        click.echo(f"    Machine:   {platform.machine()}")
        click.echo(f"    System:    {platform.system()} {platform.release()}")
        click.echo()
        return

    # ── Enumerate CUDA devices ───────────────────────────────────────────
    for device_idx in range(torch.cuda.device_count()):
        device_props = torch.cuda.get_device_properties(device_idx)
        device_name: str = device_props.name
        compute_cap: str = f"{device_props.major}.{device_props.minor}"
        total_memory_bytes: int = device_props.total_memory
        total_memory_str: str = _format_bytes(total_memory_bytes)

        # Query current memory usage
        try:
            free_memory_bytes: int = torch.cuda.mem_get_info(device_idx)[0]
            used_memory_bytes: int = total_memory_bytes - free_memory_bytes
            free_memory_str: str = _format_bytes(free_memory_bytes)
            used_memory_str: str = _format_bytes(used_memory_bytes)
            utilisation_pct: float = (
                (used_memory_bytes / total_memory_bytes) * 100
                if total_memory_bytes > 0
                else 0.0
            )
        except (RuntimeError, AttributeError):
            # mem_get_info may not be available on older PyTorch versions
            free_memory_str = "N/A"
            used_memory_str = "N/A"
            utilisation_pct = 0.0

        # Additional device properties
        multi_processor_count: int = device_props.multi_processor_count
        max_threads_per_mp: int = device_props.max_threads_per_multi_processor

        click.echo(f"  ┌─ Device {device_idx} " + "─" * 48 + "┐")
        click.echo(f"  │  Name:                {device_name}")
        click.echo(f"  │  Compute Capability:  {compute_cap}")
        click.echo(f"  │  Total Memory:        {total_memory_str}")
        click.echo(f"  │  Free Memory:         {free_memory_str}")
        click.echo(f"  │  Used Memory:         {used_memory_str}")
        click.echo(f"  │  Memory Utilisation:  {utilisation_pct:.2f}%")
        click.echo(f"  │  Multiprocessors:     {multi_processor_count}")
        click.echo(f"  │  Max Threads/MP:      {max_threads_per_mp}")
        click.echo(f"  └" + "─" * 60 + "┘")
        click.echo()

    # ── Summary ─────────────────────────────────────────────────────────
    total_memory_all: int = sum(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    )
    try:
        total_free_all: int = sum(
            torch.cuda.mem_get_info(i)[0]
            for i in range(torch.cuda.device_count())
        )
        total_used_all: int = total_memory_all - total_free_all
        overall_utilisation: float = (
            (total_used_all / total_memory_all) * 100
            if total_memory_all > 0
            else 0.0
        )
    except (RuntimeError, AttributeError):
        total_free_all = 0
        total_used_all = 0
        overall_utilisation = 0.0

    click.echo("  ┌─ Summary " + "─" * 50 + "┐")
    click.echo(f"  │  Total devices:       {torch.cuda.device_count()}")
    click.echo(f"  │  Total memory:        {_format_bytes(total_memory_all)}")
    click.echo(f"  │  Total free:          {_format_bytes(total_free_all)}")
    click.echo(f"  │  Total used:          {_format_bytes(total_used_all)}")
    click.echo(f"  │  Overall utilisation: {overall_utilisation:.2f}%")
    click.echo(f"  │  Recommended batch size (128x128): {_estimate_batch_size(total_free_all)}")
    click.echo(f"  └" + "─" * 60 + "┘")
    click.echo()

    # ── Recommendation ──────────────────────────────────────────────────
    if torch.cuda.device_count() > 0:
        click.echo("  ✅  CUDA devices detected and ready for ILGAN training.")
        click.echo("  Use the --batch-size option to tune GPU memory usage.")
    click.echo()


def _estimate_batch_size(free_memory_bytes: int) -> str:
    """Estimate a safe batch size based on available GPU memory.

    This is a rough heuristic assuming 128x128 images with the ILGAN
    architecture.  The actual batch size depends on many factors
    (image size, model depth, mixed precision, etc.).

    Parameters
    ----------
    free_memory_bytes : int
        Available GPU memory in bytes.

    Returns
    -------
    str
        A recommended batch size range (e.g., ``"16-32"``).
    """
    # Rough estimate: each sample uses ~200 MB for 128x128 images
    # with the full ILGAN model (generator + discriminator + encoders)
    # This is a very rough heuristic.
    free_mb: float = free_memory_bytes / (1024 * 1024)

    if free_mb < 1000:
        return "1-4"
    elif free_mb < 2000:
        return "4-8"
    elif free_mb < 4000:
        return "8-16"
    elif free_mb < 8000:
        return "16-32"
    elif free_mb < 16000:
        return "32-64"
    else:
        return "64-128"


# ══════════════════════════════════════════════════════════════════════════════
# analyze-losses command
# ══════════════════════════════════════════════════════════════════════════════


@cli.command(
    name="analyze-losses",
    help="Run a short training loop with gradient logging enabled.",
    epilog=(
        "This command runs a brief training loop (default 15 steps) with "
        "detailed gradient statistics logged for each parameter group.  "
        "It is useful for debugging training issues such as vanishing or "
        "exploding gradients, NaN/Inf detection, and representation collapse.  "
        "No checkpoints are saved during this analysis run."
    ),
)
@click.option(
    "--data-root",
    "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=True,
    help="Root directory of the dataset (containing images/ and labels/ subdirectories).",
)
@click.option(
    "--steps",
    "-s",
    type=click.IntRange(min=1, max=100),
    default=15,
    show_default=True,
    help="Number of training steps to run for analysis.  Default: 15.",
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1, max=1024),
    default=4,
    show_default=True,
    help="Batch size for the analysis run.  Smaller values reduce memory.  Default: 4.",
)
@click.option(
    "--image-size",
    type=click.IntRange(min=16, max=1024),
    default=64,
    show_default=True,
    help="Image size for the analysis run.  Smaller values reduce memory.  Default: 64.",
)
@click.option(
    "--latent-dim",
    "-z",
    type=click.IntRange(min=16, max=4096),
    default=128,
    show_default=True,
    help="Latent dimension for the analysis run.  Default: 128.",
)
@click.option(
    "--num-classes",
    "-n",
    type=click.IntRange(min=1, max=1000),
    default=80,
    show_default=True,
    help="Number of object classes.  Default: 80.",
)
@click.option(
    "--max-boxes",
    "-m",
    type=click.IntRange(min=1, max=200),
    default=10,
    show_default=True,
    help="Maximum number of bounding boxes.  Default: 10.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a YAML configuration file.  Default: use built-in default.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help="Random seed for reproducibility.  Default: 42.",
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default="./logs",
    show_default=True,
    help="Directory for log output.  Default: ./logs.",
)
def analyze_losses(
    data_root: str,
    steps: int,
    batch_size: int,
    image_size: int,
    latent_dim: int,
    num_classes: int,
    max_boxes: int,
    config: Optional[str],
    seed: int,
    log_dir: str,
) -> None:
    """Run a short training loop with gradient logging enabled.

    This command builds a lightweight ILGAN model and runs a brief training
    loop (default 15 steps) with detailed gradient statistics logged for
    each parameter group.  It is designed for debugging training issues:

    - **Vanishing gradients**: gradient norms near zero indicate the
      discriminator may be too strong or the generator has collapsed.
    - **Exploding gradients**: gradient norms > 100 indicate instability.
    - **NaN/Inf gradients**: numerical divergence in attention or loss.
    - **Representation collapse**: zero gradient fractions > 0.5 indicate
      dead parameters.

    The command prints per-parameter-group gradient statistics (mean norm,
    max norm, fraction zero) at each step, plus a summary table at the end.

    \f
    Parameters
    ----------
    data_root : str
        Dataset root directory.
    steps : int
        Number of training steps to run.
    batch_size : int
        Batch size for the analysis run.
    image_size : int
        Image size for the analysis run.
    latent_dim : int
        Latent dimension.
    num_classes : int
        Number of object classes.
    max_boxes : int
        Maximum boxes per image.
    config : str or None
        Path to a YAML config file.
    seed : int
        Random seed.
    log_dir : str
        Log directory.
    """
    # ── Build overrides ─────────────────────────────────────────────────
    overrides: Dict[str, Any] = {
        "paths.data_root": data_root,
        "data.batch_size": batch_size,
        "data.image_size": image_size,
        "data.num_workers": 0,  # Use 0 workers for simplicity in analysis
        "model.latent_dim": latent_dim,
        "model.num_classes": num_classes,
        "model.max_boxes": max_boxes,
        "model.gen_base_channels": 32,  # Smaller model for analysis
        "model.disc_base_channels": 32,
        "model.num_attention_heads": 4,
        "training.epochs": 1,
        "training.learning_rate": 0.0002,
        "training.n_critic": 1,
        "training.gradient_accumulation_steps": 1,
        "training.use_mixed_precision": False,
        "training.grad_checkpoint": False,
        "training.clip_grad_norm": 0.0,  # No clipping during analysis
        "paths.log_dir": log_dir,
        "paths.checkpoint_dir": "./.tmp_analysis_checkpoints",
    }

    # ── Build Config ────────────────────────────────────────────────────
    click.echo("=" * 72)
    click.echo("  ILGAN — Loss Analysis")
    click.echo(f"  Version: {__version__}")
    click.echo("=" * 72)
    click.echo()

    click.echo("Building configuration for loss analysis...")
    try:
        cfg: Config = _build_config_from_cli(
            config_path=config,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error building configuration: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Data root:   {data_root}")
    click.echo(f"  Steps:       {steps}")
    click.echo(f"  Batch size:  {batch_size}")
    click.echo(f"  Image size:  {image_size}")
    click.echo(f"  Latent dim:  {latent_dim}")
    click.echo(f"  Num classes: {num_classes}")
    click.echo(f"  Max boxes:   {max_boxes}")
    click.echo()

    # ── Set seed ────────────────────────────────────────────────────────
    _set_seed(seed)

    # ── Create Logger ───────────────────────────────────────────────────
    logger: Logger = Logger(
        name="ilgan_analysis",
        log_dir=str(cfg["paths.log_dir"]),
        level="DEBUG",  # DEBUG level to capture all gradient stats
    )

    # ── Build trainer ──────────────────────────────────────────────────
    click.echo("Building ILGANTrainer for analysis...")
    try:
        trainer: ILGANTrainer = build_trainer(config=cfg, logger=logger)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to build trainer: {e}")
        click.echo(f"Error building trainer: {e}", err=True)
        sys.exit(1)

    # ── Create a minimal dataloader ─────────────────────────────────────
    from ilgan.data.dataloader import get_train_val_loaders

    click.echo("Creating dataloader...")
    try:
        train_loader, _ = get_train_val_loaders(
            root_dir=data_root,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=0,
            val_split=0.0,  # Use all data for training in analysis
            augment=False,  # No augmentation for analysis
            global_max_boxes=max_boxes,
            train_max_boxes=max_boxes,
            val_max_boxes=max_boxes,
        )
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        click.echo(f"Error creating dataloader: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Dataset size: {len(train_loader.dataset)} samples")
    click.echo()

    # ── Import gradient utilities ───────────────────────────────────────
    from ilgan.training.gradient_utils import (
        log_gradient_statistics,
        detect_nan_inf_gradients,
        clip_gradients,
        zero_gradients,
    )
    from ilgan.training.mixed_precision import autocast_context

    # ── Analysis loop ───────────────────────────────────────────────────
    click.echo("─" * 72)
    click.echo("  Running analysis loop...")
    click.echo("─" * 72)

    # Set models to train mode
    trainer.generator.train()
    trainer.discriminator.train()
    trainer.image_encoder.train()
    trainer.box_encoder.train()

    # Collect gradient statistics across steps
    all_gradient_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    nan_inf_events: List[int] = []

    # Get an iterator over the dataset
    data_iter = iter(train_loader)

    for step in range(steps):
        # ── Get a batch ─────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart the iterator if we exhaust the dataset
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Move batch to device
        batch = batch.to(trainer.device)

        # ── Sample latent vectors ───────────────────────────────────────
        z = torch.randn(batch.batch_size, latent_dim, device=trainer.device)

        # ── Zero gradients ──────────────────────────────────────────────
        zero_gradients([trainer.g_optimizer, trainer.d_optimizer])

        # ── Forward pass (generator) ────────────────────────────────────
        with autocast_context(trainer.amp_scaler.is_enabled):
            gen_outputs = trainer.generator(z)

            # ── Compute losses ──────────────────────────────────────────
            losses = trainer.loss_aggregator(
                generator_outputs=gen_outputs,
                batch={
                    "images": batch.images,
                    "boxes": batch.boxes,
                    "labels": batch.labels,
                    "valid_mask": batch.valid_mask,
                },
                discriminator=trainer.discriminator,
                image_encoder=trainer.image_encoder,
                box_encoder=trainer.box_encoder,
                z_batch=z,
            )

            # ── Discriminator loss ──────────────────────────────────────
            d_loss = losses["total_d_loss"]

        # ── Backward pass (discriminator) ──────────────────────────────
        if trainer.amp_scaler.is_enabled:
            trainer.amp_scaler.scale(d_loss).backward(retain_graph=True)
        else:
            d_loss.backward(retain_graph=True)

        # ── Log gradient statistics for discriminator ──────────────────
        d_grad_stats = log_gradient_statistics(
            model=trainer.discriminator,
            logger=logger,
            step=step,
        )
        all_gradient_stats["discriminator"].append(
            d_grad_stats.get("discriminator", {})
        )

        # ── Check for NaN/Inf in discriminator ─────────────────────────
        bad_d = detect_nan_inf_gradients(
            model=trainer.discriminator,
            logger=logger,
            step=step,
        )
        if bad_d:
            nan_inf_events.append(step)
            click.echo(f"  ⚠  Step {step}: NaN/Inf gradients in discriminator: {bad_d}")

        # ── Generator loss (every step for analysis) ───────────────────
        g_loss = losses["total_g_loss"]

        # ── Backward pass (generator) ──────────────────────────────────
        if trainer.amp_scaler.is_enabled:
            trainer.amp_scaler.scale(g_loss).backward()
        else:
            g_loss.backward()

        # ── Log gradient statistics for generator ──────────────────────
        g_grad_stats = log_gradient_statistics(
            model=trainer.generator,
            logger=logger,
            step=step,
        )
        all_gradient_stats["generator"].append(
            g_grad_stats.get("generator", {})
        )

        # ── Log gradient statistics for encoders ────────────────────────
        img_enc_stats = log_gradient_statistics(
            model=trainer.image_encoder,
            logger=logger,
            step=step,
        )
        all_gradient_stats["image_encoder"].append(
            img_enc_stats.get("image_encoder", {})
        )

        box_enc_stats = log_gradient_statistics(
            model=trainer.box_encoder,
            logger=logger,
            step=step,
        )
        all_gradient_stats["box_encoder"].append(
            box_enc_stats.get("box_encoder", {})
        )

        # ── Check for NaN/Inf in generator ─────────────────────────────
        bad_g = detect_nan_inf_gradients(
            model=trainer.generator,
            logger=logger,
            step=step,
        )
        if bad_g:
            nan_inf_events.append(step)
            click.echo(f"  ⚠  Step {step}: NaN/Inf gradients in generator: {bad_g}")

        # ── Optimizer steps ─────────────────────────────────────────────
        if trainer.amp_scaler.is_enabled:
            trainer.amp_scaler.step(trainer.d_optimizer)
            trainer.amp_scaler.step(trainer.g_optimizer)
            trainer.amp_scaler.update()
        else:
            trainer.d_optimizer.step()
            trainer.g_optimizer.step()

        # ── Print step summary ─────────────────────────────────────────
        click.echo(
            f"  Step {step:>3d}/{steps - 1:<3d} | "
            f"G_loss: {g_loss.item():.4f} | "
            f"D_loss: {d_loss.item():.4f} | "
            f"Box: {losses.get('box_loss', 0):.4f} | "
            f"Collapse: {losses.get('collapse_loss', 0):.4f} | "
            f"Consistency: {losses.get('consistency_loss', 0):.4f}"
        )

    # ── Print summary ──────────────────────────────────────────────────
    click.echo()
    click.echo("=" * 72)
    click.echo("  Analysis Complete — Gradient Statistics Summary")
    click.echo("=" * 72)

    for module_name, stats_list in all_gradient_stats.items():
        if not stats_list:
            continue

        # Filter out empty dicts
        valid_stats = [s for s in stats_list if s]

        if not valid_stats:
            click.echo(f"  {module_name:20s}: No gradient data collected.")
            continue

        # Compute aggregate statistics
        mean_norms = [s.get("mean_norm", 0.0) for s in valid_stats]
        max_norms = [s.get("max_norm", 0.0) for s in valid_stats]
        zero_fracs = [s.get("fraction_zero", 0.0) for s in valid_stats]

        avg_mean_norm = sum(mean_norms) / len(mean_norms)
        avg_max_norm = sum(max_norms) / len(max_norms)
        avg_zero_frac = sum(zero_fracs) / len(zero_fracs)
        max_mean_norm = max(mean_norms)
        min_mean_norm = min(mean_norms)

        click.echo(f"  ┌─ {module_name}")
        click.echo(f"  │  Avg mean gradient norm:  {avg_mean_norm:.6f}")
        click.echo(f"  │  Max mean gradient norm:  {max_mean_norm:.6f}")
        click.echo(f"  │  Min mean gradient norm:  {min_mean_norm:.6f}")
        click.echo(f"  │  Avg max gradient norm:   {avg_max_norm:.6f}")
        click.echo(f"  │  Avg zero fraction:       {avg_zero_frac:.4f}")

        # Diagnostic messages
        if avg_mean_norm < 1e-6:
            click.echo(f"  │  ⚠  WARNING: Vanishing gradients detected!")
        if avg_max_norm > 100.0:
            click.echo(f"  │  ⚠  WARNING: Exploding gradients detected!")
        if avg_zero_frac > 0.5:
            click.echo(f"  │  ⚠  WARNING: High fraction of zero gradients (>50%)")
        click.echo(f"  └" + "─" * 50)

    if nan_inf_events:
        click.echo()
        click.echo(f"  ⚠  NaN/Inf gradients detected at steps: {nan_inf_events}")
        click.echo("  This indicates numerical instability.  Consider:")
        click.echo("    - Reducing the learning rate")
        click.echo("    - Enabling gradient clipping (--clip-grad-norm)")
        click.echo("    - Using mixed precision (--mixed-precision)")
        click.echo("    - Reducing the model size")
    else:
        click.echo()
        click.echo("  ✅  No NaN/Inf gradients detected — numerical stability looks good.")

    click.echo()
    click.echo("=" * 72)

    # Clean up temporary checkpoint directory
    import shutil
    tmp_dir = "./.tmp_analysis_checkpoints"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# profile-memory command
# ══════════════════════════════════════════════════════════════════════════════


@cli.command(
    name="profile-memory",
    help="Run a single forward+backward pass and print CUDA memory usage.",
    epilog=(
        "This command builds the ILGAN model, runs one forward and backward "
        "pass on a dummy batch, and prints detailed CUDA memory usage for "
        "each module (generator, discriminator, image encoder, box encoder).  "
        "It is useful for estimating VRAM requirements before full training.  "
        "On CPU-only systems, it prints parameter counts and estimated memory "
        "instead."
    ),
)
@click.option(
    "--batch-size",
    "-b",
    type=click.IntRange(min=1, max=1024),
    default=4,
    show_default=True,
    help="Batch size for the memory profile.  Default: 4.",
)
@click.option(
    "--image-size",
    "-s",
    type=click.IntRange(min=16, max=1024),
    default=128,
    show_default=True,
    help="Image size for the memory profile.  Default: 128.",
)
@click.option(
    "--latent-dim",
    "-z",
    type=click.IntRange(min=16, max=4096),
    default=256,
    show_default=True,
    help="Latent dimension.  Default: 256.",
)
@click.option(
    "--num-classes",
    "-n",
    type=click.IntRange(min=1, max=1000),
    default=80,
    show_default=True,
    help="Number of object classes.  Default: 80.",
)
@click.option(
    "--max-boxes",
    "-m",
    type=click.IntRange(min=1, max=200),
    default=20,
    show_default=True,
    help="Maximum number of bounding boxes.  Default: 20.",
)
@click.option(
    "--gen-base-channels",
    type=click.IntRange(min=8, max=512),
    default=64,
    show_default=True,
    help="Generator base channels.  Default: 64.",
)
@click.option(
    "--disc-base-channels",
    type=click.IntRange(min=8, max=512),
    default=64,
    show_default=True,
    help="Discriminator base channels.  Default: 64.",
)
@click.option(
    "--num-attention-heads",
    type=click.IntRange(min=1, max=32),
    default=8,
    show_default=True,
    help="Number of attention heads.  Default: 8.",
)
@click.option(
    "--mixed-precision",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable mixed precision for the profile.  Default: False.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a YAML configuration file.  Default: use built-in default.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0),
    default=42,
    show_default=True,
    help="Random seed.  Default: 42.",
)
def profile_memory(
    batch_size: int,
    image_size: int,
    latent_dim: int,
    num_classes: int,
    max_boxes: int,
    gen_base_channels: int,
    disc_base_channels: int,
    num_attention_heads: int,
    mixed_precision: bool,
    config: Optional[str],
    seed: int,
) -> None:
    """Run a single forward+backward pass and print CUDA memory usage.

    This command builds the ILGAN model, runs one forward and backward
    pass on a dummy batch, and prints detailed CUDA memory usage for
    each module.  It is useful for:

    - Estimating VRAM requirements before full training.
    - Comparing memory usage across different model configurations.
    - Identifying memory bottlenecks in specific modules.
    - Validating that mixed precision reduces memory as expected.

    On CPU-only systems, the command prints parameter counts and estimated
    memory usage instead of actual CUDA measurements.

    \f
    Parameters
    ----------
    batch_size : int
        Batch size for the profile.
    image_size : int
        Image size.
    latent_dim : int
        Latent dimension.
    num_classes : int
        Number of object classes.
    max_boxes : int
        Maximum boxes per image.
    gen_base_channels : int
        Generator base channels.
    disc_base_channels : int
        Discriminator base channels.
    num_attention_heads : int
        Number of attention heads.
    mixed_precision : bool
        Whether to enable mixed precision.
    config : str or None
        Path to a YAML config file.
    seed : int
        Random seed.
    """
    # ── Build overrides ─────────────────────────────────────────────────
    overrides: Dict[str, Any] = {
        "data.batch_size": batch_size,
        "data.image_size": image_size,
        "data.num_workers": 0,
        "model.latent_dim": latent_dim,
        "model.num_classes": num_classes,
        "model.max_boxes": max_boxes,
        "model.gen_base_channels": gen_base_channels,
        "model.disc_base_channels": disc_base_channels,
        "model.num_attention_heads": num_attention_heads,
        "training.use_mixed_precision": mixed_precision,
        "training.grad_checkpoint": False,
        "training.epochs": 1,
        "training.learning_rate": 0.0002,
        "paths.data_root": "./data",
        "paths.checkpoint_dir": "./.tmp_profile_checkpoints",
        "paths.log_dir": "./logs",
    }

    # ── Build Config ────────────────────────────────────────────────────
    click.echo("=" * 72)
    click.echo("  ILGAN — Memory Profile")
    click.echo(f"  Version: {__version__}")
    click.echo("=" * 72)
    click.echo()

    click.echo("Building configuration for memory profiling...")
    try:
        cfg: Config = _build_config_from_cli(
            config_path=config,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error building configuration: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Batch size:           {batch_size}")
    click.echo(f"  Image size:           {image_size}")
    click.echo(f"  Latent dim:           {latent_dim}")
    click.echo(f"  Num classes:          {num_classes}")
    click.echo(f"  Max boxes:            {max_boxes}")
    click.echo(f"  Gen base channels:    {gen_base_channels}")
    click.echo(f"  Disc base channels:   {disc_base_channels}")
    click.echo(f"  Attention heads:      {num_attention_heads}")
    click.echo(f"  Mixed precision:      {mixed_precision}")
    click.echo()

    # ── Set seed ────────────────────────────────────────────────────────
    _set_seed(seed)

    # ── Create Logger ───────────────────────────────────────────────────
    logger: Logger = Logger(
        name="ilgan_profile",
        log_dir="./logs",
        level="INFO",
    )

    # ── Build trainer ──────────────────────────────────────────────────
    click.echo("Building ILGANTrainer for memory profiling...")
    try:
        trainer: ILGANTrainer = build_trainer(config=cfg, logger=logger)
    except (TypeError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to build trainer: {e}")
        click.echo(f"Error building trainer: {e}", err=True)
        sys.exit(1)

    # ── Determine device ────────────────────────────────────────────────
    device = trainer.device
    is_cuda = device.type == "cuda"

    if not is_cuda:
        click.echo("  ⚠  CUDA not available — running on CPU.")
        click.echo("  Memory estimates will be based on parameter counts only.")
        click.echo()

    # ── Print model parameter counts ────────────────────────────────────
    def count_params(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def count_trainable_params(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    modules = {
        "Generator": trainer.generator,
        "Discriminator": trainer.discriminator,
        "Image Encoder": trainer.image_encoder,
        "Box Encoder": trainer.box_encoder,
    }

    total_params = 0
    total_trainable = 0

    click.echo("  ┌─ Model Parameter Counts")
    click.echo("  │  " + "-" * 60)
    click.echo(f"  │  {'Module':<20s} {'Params':>12s} {'Trainable':>12s} {'Size':>12s}")
    click.echo("  │  " + "-" * 60)

    for name, model in modules.items():
        n_params = count_params(model)
        n_trainable = count_trainable_params(model)
        total_params += n_params
        total_trainable += n_trainable

        # Estimate memory: 4 bytes per param (float32) or 2 bytes (float16)
        bytes_per_param = 2 if mixed_precision else 4
        est_mb = (n_params * bytes_per_param) / (1024 * 1024)

        click.echo(
            f"  │  {name:<20s} {n_params:>12,} {n_trainable:>12,} {est_mb:>10.2f} MB"
        )

    click.echo("  │  " + "-" * 60)
    total_est_mb = (total_params * (2 if mixed_precision else 4)) / (1024 * 1024)
    click.echo(
        f"  │  {'TOTAL':<20s} {total_params:>12,} {total_trainable:>12,} {total_est_mb:>10.2f} MB"
    )
    click.echo("  └" + "─" * 60)
    click.echo()

    if not is_cuda:
        click.echo("  Estimated total model memory: " + _format_bytes(
            total_params * (2 if mixed_precision else 4)
        ))
        click.echo()
        click.echo("=" * 72)
        click.echo("  Profile complete (CPU mode — no CUDA memory data).")
        click.echo("=" * 72)
        return

    # ── CUDA memory profiling ───────────────────────────────────────────
    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Record memory before
    mem_before = torch.cuda.memory_allocated(device)

    # ── Create dummy inputs ────────────────────────────────────────────
    dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    dummy_boxes = torch.rand(batch_size, max_boxes, 4, device=device)
    dummy_labels = torch.randint(0, num_classes, (batch_size, max_boxes), device=device)
    dummy_valid = torch.ones(batch_size, max_boxes, dtype=torch.bool, device=device)
    dummy_z = torch.randn(batch_size, latent_dim, device=device)

    # ── Forward + backward pass ────────────────────────────────────────
    click.echo("  Running forward+backward pass...")

    # Set models to train mode
    trainer.generator.train()
    trainer.discriminator.train()
    trainer.image_encoder.train()
    trainer.box_encoder.train()

    # Record per-module memory snapshots
    module_memory: Dict[str, Dict[str, float]] = {}

    # Helper to capture memory delta
    def _capture_memory(label: str) -> None:
        current = torch.cuda.memory_allocated(device)
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        module_memory[label] = {
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
            "reserved_mb": reserved / (1024 * 1024),
            "delta_mb": (current - mem_before) / (1024 * 1024),
        }

    # Zero gradients
    trainer.g_optimizer.zero_grad(set_to_none=True)
    trainer.d_optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    _capture_memory("After zero_grad")

    # Generator forward
    gen_outputs = trainer.generator(dummy_z)
    torch.cuda.synchronize(device)
    _capture_memory("After generator forward")

    # Discriminator forward (real)
    real_scores = trainer.discriminator(dummy_images)
    torch.cuda.synchronize(device)
    _capture_memory("After discriminator forward (real)")

    # Discriminator forward (fake)
    fake_scores = trainer.discriminator(gen_outputs["image"])
    torch.cuda.synchronize(device)
    _capture_memory("After discriminator forward (fake)")

    # Encoder forward
    img_feat = trainer.image_encoder(dummy_images)
    box_feat = trainer.box_encoder(dummy_boxes, gen_outputs["confidences"], dummy_valid)
    torch.cuda.synchronize(device)
    _capture_memory("After encoder forward")

    # Compute loss
    losses = trainer.loss_aggregator(
        generator_outputs=gen_outputs,
        batch={
            "images": dummy_images,
            "boxes": dummy_boxes,
            "labels": dummy_labels,
            "valid_mask": dummy_valid,
        },
        discriminator=trainer.discriminator,
        image_encoder=trainer.image_encoder,
        box_encoder=trainer.box_encoder,
        z_batch=dummy_z,
    )

    # Backward pass
    losses["total_d_loss"].backward(retain_graph=True)
    losses["total_g_loss"].backward()
    torch.cuda.synchronize(device)
    _capture_memory("After backward pass")

    # ── Print memory profile ────────────────────────────────────────────
    click.echo()
    click.echo("  ┌─ CUDA Memory Profile (per stage)")
    click.echo("  │  " + "-" * 70)
    click.echo(
        f"  │  {'Stage':<30s} {'Current':>10s} {'Peak':>10s} {'Reserved':>10s}"
    )
    click.echo("  │  " + "-" * 70)

    for stage, mem in module_memory.items():
        click.echo(
            f"  │  {stage:<30s} "
            f"{mem['current_mb']:>8.2f} MB "
            f"{mem['peak_mb']:>8.2f} MB "
            f"{mem['reserved_mb']:>8.2f} MB"
        )

    click.echo("  │  " + "-" * 70)

    # Final memory
    mem_after = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    reserved_mem = torch.cuda.memory_reserved(device)

    click.echo(
        f"  │  {'Final allocated':<30s} {mem_after / (1024*1024):>8.2f} MB"
    )
    click.echo(
        f"  │  {'Peak allocated':<30s} {peak_mem / (1024*1024):>8.2f} MB"
    )
    click.echo(
        f"  │  {'Total reserved':<30s} {reserved_mem / (1024*1024):>8.2f} MB"
    )
    click.echo("  └" + "─" * 70)

    # ── Memory breakdown by tensor type ─────────────────────────────────
    click.echo()
    click.echo("  ┌─ Memory Breakdown by Component")
    click.echo("  │  " + "-" * 50)

    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in trainer.generator.parameters())
    param_mem += sum(p.numel() * p.element_size() for p in trainer.discriminator.parameters())
    param_mem += sum(p.numel() * p.element_size() for p in trainer.image_encoder.parameters())
    param_mem += sum(p.numel() * p.element_size() for p in trainer.box_encoder.parameters())

    # Gradient memory (same as param memory for most params)
    grad_mem = param_mem

    # Optimizer memory (2x param memory for Adam: exp_avg + exp_avg_sq)
    opt_mem = param_mem * 2

    # Activation memory (estimated as peak - param - grad - opt)
    activation_mem = max(0, peak_mem - param_mem - grad_mem - opt_mem)

    click.echo(f"  │  {'Parameters':<25s} {param_mem / (1024*1024):>10.2f} MB")
    click.echo(f"  │  {'Gradients':<25s} {grad_mem / (1024*1024):>10.2f} MB")
    click.echo(f"  │  {'Optimizer states':<25s} {opt_mem / (1024*1024):>10.2f} MB")
    click.echo(f"  │  {'Activations (est.)':<25s} {activation_mem / (1024*1024):>10.2f} MB")
    click.echo(f"  │  {'Total peak':<25s} {peak_mem / (1024*1024):>10.2f} MB")
    click.echo("  └" + "─" * 50)

    # ── Recommendations ─────────────────────────────────────────────────
    click.echo()
    click.echo("  ┌─ Recommendations")
    click.echo("  │  " + "-" * 50)

    # Estimate max batch size
    free_mem, _ = torch.cuda.mem_get_info(device)
    per_sample_mem = peak_mem / batch_size if batch_size > 0 else 0
    if per_sample_mem > 0:
        max_batch_estimate = int(free_mem * 0.8 / per_sample_mem)
        click.echo(f"  │  Estimated max batch size: {max(1, max_batch_estimate)}")
    else:
        click.echo("  │  Estimated max batch size: N/A")

    if mixed_precision:
        click.echo("  │  Mixed precision: ON — reduces activation memory by ~40%")
    else:
        click.echo("  │  Mixed precision: OFF — enable with --mixed-precision to save memory")

    click.echo("  │  Gradient checkpointing: OFF — enable in config to save memory")
    click.echo("  └" + "─" * 50)

    click.echo()
    click.echo("=" * 72)
    click.echo("  Memory profile complete.")
    click.echo("=" * 72)

    # Clean up temporary checkpoint directory
    import shutil
    tmp_dir = "./.tmp_profile_checkpoints"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# compute-statistics command
# ══════════════════════════════════════════════════════════════════════════════


@cli.command(
    name="compute-statistics",
    help="Analyze a dataset and print comprehensive statistics.",
    epilog=(
        "This command scans a dataset directory (images/ and labels/ "
        "subdirectories) and prints detailed statistics including:\n"
        "  - Total number of images\n"
        "  - Image size distribution (width, height, aspect ratio)\n"
        "  - Average, min, max boxes per image\n"
        "  - Class distribution (counts per class)\n"
        "  - Box size distribution (width, height, area)\n"
        "  - Empty images (images with no labels)\n"
        "  - Dataset balance metrics\n\n"
        "The statistics are printed to the console and optionally saved "
        "as a JSON file for further analysis."
    ),
)
@click.option(
    "--data-root",
    "-d",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=True,
    help="Root directory of the dataset (containing images/ and labels/ subdirectories).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, resolve_path=True),
    default=None,
    help=(
        "Optional path to save the statistics as a JSON file.  "
        "If not provided, statistics are only printed to the console."
    ),
)
@click.option(
    "--max-samples",
    type=click.IntRange(min=1, max=1000000),
    default=None,
    help=(
        "Maximum number of samples to analyze.  Useful for quick "
        "statistics on large datasets.  Default: analyze all samples."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print detailed per-sample information (first 10 samples).",
)
def compute_statistics(
    data_root: str,
    output: Optional[str],
    max_samples: Optional[int],
    verbose: bool,
) -> None:
    """Analyze a dataset and print comprehensive statistics.

    This command scans a dataset directory and prints detailed statistics
    about the images and their bounding box annotations.  It is useful for:

    - Understanding dataset composition before training.
    - Detecting data issues (missing labels, corrupt images, class imbalance).
    - Determining appropriate model hyperparameters (max_boxes, num_classes).
    - Validating data preprocessing and augmentation strategies.

    The command analyzes both the ``images/`` and ``labels/`` subdirectories
    under *data_root*.

    \f
    Parameters
    ----------
    data_root : str
        Root directory of the dataset.
    output : str or None
        Optional path to save statistics as JSON.
    max_samples : int or None
        Maximum number of samples to analyze.
    verbose : bool
        Whether to print detailed per-sample information.
    """
    click.echo("=" * 72)
    click.echo("  ILGAN — Dataset Statistics")
    click.echo(f"  Version: {__version__}")
    click.echo("=" * 72)
    click.echo()

    data_root_path = Path(data_root)
    images_dir = data_root_path / "images"
    labels_dir = data_root_path / "labels"

    # ── Validate directory structure ────────────────────────────────────
    if not images_dir.is_dir():
        click.echo(f"  Error: Images directory not found: {images_dir}", err=True)
        sys.exit(1)

    if not labels_dir.is_dir():
        click.echo(f"  Warning: Labels directory not found: {labels_dir}")
        click.echo("  Only image statistics will be computed.")
        has_labels = False
    else:
        has_labels = True

    # ── Discover image files ─────────────────────────────────────────────
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files: List[Path] = []
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() in supported_extensions:
            image_files.append(f)

    if not image_files:
        click.echo(f"  Error: No image files found in {images_dir}", err=True)
        sys.exit(1)

    # Apply max_samples limit
    if max_samples is not None and max_samples < len(image_files):
        click.echo(f"  Limiting analysis to {max_samples} samples (of {len(image_files)} total).")
        # Use deterministic sampling
        rng = random.Random(42)
        image_files = rng.sample(image_files, max_samples)
        image_files.sort(key=lambda p: p.name)

    total_images = len(image_files)
    click.echo(f"  Dataset root: {data_root}")
    click.echo(f"  Images found: {total_images}")
    click.echo()

    # ── Analyze images ──────────────────────────────────────────────────
    click.echo("  Analyzing images...")
    click.echo("  ┌─ Image Statistics")

    image_widths: List[int] = []
    image_heights: List[int] = []
    image_aspect_ratios: List[float] = []
    image_sizes_bytes: List[int] = []
    corrupt_images: List[str] = []

    from PIL import Image

    for img_file in image_files:
        try:
            with Image.open(img_file) as pil_img:
                w, h = pil_img.size
                image_widths.append(w)
                image_heights.append(h)
                image_aspect_ratios.append(w / h if h > 0 else 0.0)
                image_sizes_bytes.append(img_file.stat().st_size)
        except Exception as e:
            corrupt_images.append(str(img_file))
            click.echo(f"  ⚠  Corrupt image: {img_file.name} — {e}")

    # Image size statistics
    if image_widths:
        min_w, max_w = min(image_widths), max(image_widths)
        min_h, max_h = min(image_heights), max(image_heights)
        avg_w = sum(image_widths) / len(image_widths)
        avg_h = sum(image_heights) / len(image_heights)
        avg_ar = sum(image_aspect_ratios) / len(image_aspect_ratios)

        click.echo(f"  │  Width:   min={min_w}, max={max_w}, avg={avg_w:.1f}")
        click.echo(f"  │  Height:  min={min_h}, max={max_h}, avg={avg_h:.1f}")
        click.echo(f"  │  Aspect ratio: min={min(image_aspect_ratios):.3f}, "
                   f"max={max(image_aspect_ratios):.3f}, "
                   f"avg={avg_ar:.3f}")

        # Size distribution (buckets)
        size_buckets = Counter()
        for w, h in zip(image_widths, image_heights):
            max_dim = max(w, h)
            if max_dim <= 64:
                size_buckets["<=64"] += 1
            elif max_dim <= 128:
                size_buckets["64-128"] += 1
            elif max_dim <= 256:
                size_buckets["128-256"] += 1
            elif max_dim <= 512:
                size_buckets["256-512"] += 1
            elif max_dim <= 1024:
                size_buckets["512-1024"] += 1
            else:
                size_buckets[">1024"] += 1

        click.echo(f"  │  Size distribution:")
        for bucket in ["<=64", "64-128", "128-256", "256-512", "512-1024", ">1024"]:
            count = size_buckets.get(bucket, 0)
            pct = count / total_images * 100
            bar = "█" * int(pct / 2)
            click.echo(f"  │    {bucket:>10s}: {count:>6d} ({pct:5.1f}%) {bar}")

    if corrupt_images:
        click.echo(f"  │  Corrupt images: {len(corrupt_images)}")
    click.echo("  └" + "─" * 50)
    click.echo()

    # ── Analyze labels ──────────────────────────────────────────────────
    if not has_labels:
        click.echo("  No label directory found — skipping label statistics.")
        click.echo()
        click.echo("=" * 72)
        click.echo("  Statistics complete (images only).")
        click.echo("=" * 72)
        return

    click.echo("  Analyzing labels...")
    click.echo("  ┌─ Label Statistics")

    from ilgan.data.structures import parse_yolo_label

    boxes_per_image: List[int] = []
    class_counts: Counter = Counter()
    box_widths: List[float] = []
    box_heights: List[float] = []
    box_areas: List[float] = []
    empty_images: List[str] = []
    invalid_labels: List[str] = []
    per_image_class_counts: List[Counter] = []

    for img_file in image_files:
        stem = img_file.stem
        label_file = labels_dir / f"{stem}.txt"

        if not label_file.is_file():
            empty_images.append(stem)
            boxes_per_image.append(0)
            per_image_class_counts.append(Counter())
            continue

        try:
            boxes, labels, valid_mask = parse_yolo_label(str(label_file))
            n_boxes = valid_mask.sum().item()
            boxes_per_image.append(n_boxes)

            if n_boxes == 0:
                empty_images.append(stem)

            # Collect class and box statistics
            img_class_counter = Counter()
            for i in range(len(labels)):
                if valid_mask[i]:
                    cls_id = labels[i].item()
                    class_counts[cls_id] += 1
                    img_class_counter[cls_id] += 1

                    # Box dimensions (in normalised coordinates)
                    cx, cy, bw, bh = boxes[i].tolist()
                    box_widths.append(bw)
                    box_heights.append(bh)
                    box_areas.append(bw * bh)

            per_image_class_counts.append(img_class_counter)

        except Exception as e:
            invalid_labels.append(stem)
            boxes_per_image.append(0)
            per_image_class_counts.append(Counter())
            click.echo(f"  ⚠  Invalid label: {label_file.name} — {e}")

    # Box count statistics
    if boxes_per_image:
        min_boxes = min(boxes_per_image)
        max_boxes = max(boxes_per_image)
        avg_boxes = sum(boxes_per_image) / len(boxes_per_image)
        total_boxes = sum(boxes_per_image)
        non_empty = sum(1 for b in boxes_per_image if b > 0)

        click.echo(f"  │  Total boxes:          {total_boxes}")
        click.echo(f"  │  Boxes per image:      min={min_boxes}, max={max_boxes}, "
                   f"avg={avg_boxes:.2f}")
        click.echo(f"  │  Images with boxes:    {non_empty}/{total_images} "
                   f"({non_empty / total_images * 100:.1f}%)")
        click.echo(f"  │  Empty images:         {len(empty_images)} "
                   f"({len(empty_images) / total_images * 100:.1f}%)")

        # Box count distribution
        box_count_buckets = Counter()
        for n in boxes_per_image:
            if n == 0:
                box_count_buckets["0"] += 1
            elif n <= 5:
                box_count_buckets["1-5"] += 1
            elif n <= 10:
                box_count_buckets["6-10"] += 1
            elif n <= 20:
                box_count_buckets["11-20"] += 1
            elif n <= 50:
                box_count_buckets["21-50"] += 1
            else:
                box_count_buckets[">50"] += 1

        click.echo(f"  │  Box count distribution:")
        for bucket in ["0", "1-5", "6-10", "11-20", "21-50", ">50"]:
            count = box_count_buckets.get(bucket, 0)
            pct = count / total_images * 100
            bar = "█" * int(pct / 2)
            click.echo(f"  │    {bucket:>6s}: {count:>6d} ({pct:5.1f}%) {bar}")

    # Box size statistics
    if box_widths:
        avg_bw = sum(box_widths) / len(box_widths)
        avg_bh = sum(box_heights) / len(box_heights)
        avg_ba = sum(box_areas) / len(box_areas)
        min_bw, max_bw = min(box_widths), max(box_widths)
        min_bh, max_bh = min(box_heights), max(box_heights)

        click.echo(f"  │  Box width:            min={min_bw:.4f}, max={max_bw:.4f}, "
                   f"avg={avg_bw:.4f}")
        click.echo(f"  │  Box height:           min={min_bh:.4f}, max={max_bh:.4f}, "
                   f"avg={avg_bh:.4f}")
        click.echo(f"  │  Box area:             min={min(box_areas):.6f}, "
                   f"max={max(box_areas):.6f}, avg={avg_ba:.6f}")

        # Box size distribution (by area)
        area_buckets = Counter()
        for area in box_areas:
            if area <= 0.01:
                area_buckets["tiny (<=1%)"] += 1
            elif area <= 0.05:
                area_buckets["small (1-5%)"] += 1
            elif area <= 0.20:
                area_buckets["medium (5-20%)"] += 1
            elif area <= 0.50:
                area_buckets["large (20-50%)"] += 1
            else:
                area_buckets["huge (>50%)"] += 1

        click.echo(f"  │  Box area distribution:")
        for bucket in ["tiny (<=1%)", "small (1-5%)", "medium (5-20%)",
                       "large (20-50%)", "huge (>50%)"]:
            count = area_buckets.get(bucket, 0)
            pct = count / len(box_areas) * 100
            bar = "█" * int(pct / 2)
            click.echo(f"  │    {bucket:>18s}: {count:>6d} ({pct:5.1f}%) {bar}")

    # Class distribution
    if class_counts:
        num_classes_found = len(class_counts)
        total_class_instances = sum(class_counts.values())

        click.echo(f"  │  Classes found:        {num_classes_found}")
        click.echo(f"  │  Total class instances: {total_class_instances}")
        click.echo(f"  │  Class distribution:")

        # Sort by count descending
        sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])

        # Compute Gini coefficient for class imbalance
        sorted_counts = [c for _, c in sorted_classes]
        gini = _compute_gini(sorted_counts)
        click.echo(f"  │  Class imbalance (Gini): {gini:.4f} "
                   f"({'balanced' if gini < 0.3 else 'moderate' if gini < 0.6 else 'imbalanced'})")

        for cls_id, count in sorted_classes[:20]:  # Show top 20 classes
            pct = count / total_class_instances * 100
            bar = "█" * int(pct)
            click.echo(f"  │    Class {cls_id:>4d}: {count:>6d} ({pct:5.2f}%) {bar}")

        if len(sorted_classes) > 20:
            remaining = sum(c for _, c in sorted_classes[20:])
            remaining_pct = remaining / total_class_instances * 100
            click.echo(f"  │    ... and {len(sorted_classes) - 20} more classes "
                       f"({remaining} instances, {remaining_pct:.2f}%)")

    if invalid_labels:
        click.echo(f"  │  Invalid label files:  {len(invalid_labels)}")

    click.echo("  └" + "─" * 50)
    click.echo()

    # ── Summary ─────────────────────────────────────────────────────────
    click.echo("  ┌─ Summary")
    click.echo(f"  │  Total images:              {total_images}")
    click.echo(f"  │  Total boxes:               {total_boxes}")
    click.echo(f"  │  Average boxes per image:    {avg_boxes:.2f}")
    click.echo(f"  │  Images with boxes:          {non_empty}/{total_images}")
    click.echo(f"  │  Empty images:               {len(empty_images)}")
    click.echo(f"  │  Number of classes:          {num_classes_found}")
    click.echo(f"  │  Class imbalance (Gini):     {gini:.4f}")
    click.echo(f"  │  Recommended max_boxes:     {_recommend_max_boxes(boxes_per_image)}")
    click.echo(f"  │  Corrupt images:            {len(corrupt_images)}")
    click.echo(f"  │  Invalid label files:        {len(invalid_labels)}")
    click.echo("  └" + "─" * 50)
    click.echo()

    # ── Verbose: print first 10 samples ─────────────────────────────────
    if verbose:
        click.echo("  ┌─ First 10 Samples (detailed)")
        click.echo("  │  " + "-" * 70)
        click.echo(
            f"  │  {'#':>4s} {'Image':<30s} {'Size':>12s} {'Boxes':>6s} {'Classes':>8s}"
        )
        click.echo("  │  " + "-" * 70)

        for i, img_file in enumerate(image_files[:10]):
            stem = img_file.stem
            w = image_widths[i] if i < len(image_widths) else 0
            h = image_heights[i] if i < len(image_heights) else 0
            n_boxes = boxes_per_image[i] if i < len(boxes_per_image) else 0
            n_classes = len(per_image_class_counts[i]) if i < len(per_image_class_counts) else 0

            click.echo(
                f"  │  {i:>4d} {img_file.name:<30s} "
                f"{w}x{h:<9} {n_boxes:>6d} {n_classes:>8d}"
            )

        click.echo("  └" + "─" * 70)
        click.echo()

    # ── Save to JSON if requested ──────────────────────────────────────
    if output is not None:
        stats_dict = {
            "dataset_root": str(data_root),
            "total_images": total_images,
            "total_boxes": total_boxes,
            "average_boxes_per_image": round(avg_boxes, 4),
            "min_boxes_per_image": min_boxes,
            "max_boxes_per_image": max_boxes,
            "images_with_boxes": non_empty,
            "empty_images": len(empty_images),
            "num_classes": num_classes_found,
            "class_imbalance_gini": round(gini, 4),
            "recommended_max_boxes": _recommend_max_boxes(boxes_per_image),
            "image_size": {
                "width_min": min_w,
                "width_max": max_w,
                "width_avg": round(avg_w, 1),
                "height_min": min_h,
                "height_max": max_h,
                "height_avg": round(avg_h, 1),
                "aspect_ratio_avg": round(avg_ar, 3),
            },
            "box_size": {
                "width_min": round(min_bw, 4),
                "width_max": round(max_bw, 4),
                "width_avg": round(avg_bw, 4),
                "height_min": round(min_bh, 4),
                "height_max": round(max_bh, 4),
                "height_avg": round(avg_bh, 4),
                "area_avg": round(avg_ba, 6),
            },
            "class_distribution": {
                str(k): v for k, v in sorted_classes
            },
            "corrupt_images": corrupt_images,
            "invalid_labels": invalid_labels,
            "empty_images": empty_images,
        }

        try:
            with open(output, "w") as f:
                json.dump(stats_dict, f, indent=2, sort_keys=True)
            click.echo(f"  Statistics saved to: {output}")
        except Exception as e:
            click.echo(f"  ⚠  Failed to save statistics: {e}", err=True)

    click.echo()
    click.echo("=" * 72)
    click.echo("  Statistics complete.")
    click.echo("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────────────


def _compute_gini(sorted_counts: List[int]) -> float:
    """Compute the Gini coefficient for class imbalance.

    The Gini coefficient ranges from 0 (perfect equality — all classes
    have the same number of instances) to 1 (perfect inequality — one
    class has all instances).

    Parameters
    ----------
    sorted_counts : list of int
        Class instance counts, sorted in descending order.

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    if not sorted_counts:
        return 0.0

    n = len(sorted_counts)
    total = sum(sorted_counts)
    if total == 0:
        return 0.0

    # Gini = (2 * sum_i(i * c_i) / (n * sum(c_i))) - (n + 1) / n
    # where c_i are sorted in ascending order
    sorted_asc = sorted(sorted_counts)
    cumulative = 0
    for i, c in enumerate(sorted_asc):
        cumulative += (i + 1) * c

    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def _recommend_max_boxes(boxes_per_image: List[int]) -> int:
    """Recommend a value for ``max_boxes`` based on the dataset.

    Uses the 95th percentile of boxes per image, rounded up to the
    nearest multiple of 5, with a minimum of 5.

    Parameters
    ----------
    boxes_per_image : list of int
        Number of boxes per image across the dataset.

    Returns
    -------
    int
        Recommended ``max_boxes`` value.
    """
    if not boxes_per_image:
        return 20

    sorted_counts = sorted(boxes_per_image)
    p95_idx = int(len(sorted_counts) * 0.95)
    p95 = sorted_counts[min(p95_idx, len(sorted_counts) - 1)]

    # Round up to nearest multiple of 5, minimum 5
    recommended = max(5, ((p95 + 4) // 5) * 5)
    return recommended


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ILGAN CLI.

    This function is registered as a console_scripts entry point in
    ``setup.py`` (or ``pyproject.toml``) so that the CLI can be invoked
    as ``ilgan`` from the command line.

    It simply calls :func:`cli` (the Click group) with the command-line
    arguments.

    Examples
    --------
    .. code-block:: bash

        # Via python module
        python -m ilgan.scripts.cli train --help

        # Via installed package
        ilgan train --help
    """
    cli()


# ──────────────────────────────────────────────────────────────────────────────
# Script entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
