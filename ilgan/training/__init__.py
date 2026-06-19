"""
ILGAN training package — optimizers, schedulers, mixed precision, gradient
management, checkpointing, training loops, and top-level orchestration.

The ``ilgan.training`` package contains all components needed to train the
ILGAN dual-output GAN:

- :mod:`ilgan.training.optimizers`: custom optimizer setup, adaptive LR
  scheduler, and standard LR scheduler factories.
- :mod:`ilgan.training.mixed_precision`: mixed precision (AMP) training
  support with loss scaling, autocast context management, and CPU fallback.
- :mod:`ilgan.training.gradient_utils`: gradient clipping, statistics
  logging, NaN/Inf detection, and zero-grad convenience.
- :mod:`ilgan.training.checkpoint`: model checkpointing with save/load,
  best-checkpoint tracking, and automatic pruning.
- :mod:`ilgan.training.train_epoch`: single training epoch loop with
  WGAN-GP ``n_critic`` ratio, gradient accumulation, representation
  anchoring, and AMP support.
- :mod:`ilgan.training.val_epoch`: validation epoch loop with loss
  monitoring, metrics accumulation, realism gap computation, and sample
  grid generation.
- :mod:`ilgan.training.trainer`: top-level :class:`ILGANTrainer` that
  orchestrates the full training, evaluation, and generation pipeline.

Factory function
----------------
Use :func:`build_trainer` to construct a fully configured
:class:`ILGANTrainer` from a :class:`~ilgan.utils.config.Config` and
:class:`~ilgan.utils.logger.Logger`::

    from ilgan.utils.config import Config
    from ilgan.utils.logger import Logger
    from ilgan.training import build_trainer

    cfg = Config()
    logger = Logger(log_dir=cfg.paths.log_dir)
    trainer = build_trainer(cfg, logger)
    trainer.train()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ilgan.utils.config import Config
from ilgan.utils.logger import Logger

# ──────────────────────────────────────────────────────────────────────────────
# Re-exports from sub-modules
# ──────────────────────────────────────────────────────────────────────────────

# Optimizers
from ilgan.training.optimizers import (
    AdaptiveOptimizerScheduler,
    build_optimizers,
    build_scheduler,
)

# Mixed precision
from ilgan.training.mixed_precision import (
    AMPScaler,
    autocast_context,
    create_amp_scaler,
    should_use_amp,
)

# Gradient utilities
from ilgan.training.gradient_utils import (
    clip_gradients,
    detect_nan_inf_gradients,
    log_gradient_statistics,
    zero_gradients,
)

# Checkpointing
from ilgan.training.checkpoint import (
    CheckpointManager,
    load_or_initialize,
)

# Training loop
from ilgan.training.train_epoch import (
    RepresentationAnchor,
    train_epoch,
)

# Validation loop
from ilgan.training.val_epoch import (
    validate,
    _generate_sample_grid,
)

# Top-level trainer
from ilgan.training.trainer import (
    ILGANTrainer,
)

# ──────────────────────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────────────────────


def build_trainer(
    config: Config,
    logger: Logger,
    **kwargs: Any,
) -> ILGANTrainer:
    """Construct a fully configured :class:`ILGANTrainer` from a config and
    logger.

    This factory function is the recommended way to create an
    :class:`ILGANTrainer`.  It validates the config, creates the logger
    if needed, and returns a ready-to-use trainer instance.

    The factory performs the following steps:

    1. **Validates inputs**: checks that *config* is a :class:`Config`
       instance and *logger* is a :class:`Logger` instance.
    2. **Validates critical config keys**: ensures that all keys required
       for model construction and training are present in the config.
    3. **Constructs the trainer**: creates an :class:`ILGANTrainer` with
       the given config and logger.
    4. **Logs a summary**: logs the trainer configuration at INFO level.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.  Must contain at minimum the
        following keys (with valid values):

        - ``model.latent_dim`` (int, > 0)
        - ``model.gen_base_channels`` (int, > 0)
        - ``model.disc_base_channels`` (int, > 0)
        - ``model.num_attention_heads`` (int, > 0)
        - ``model.max_boxes`` (int, > 0)
        - ``model.num_classes`` (int, > 0)
        - ``data.image_size`` (int, > 0)
        - ``data.batch_size`` (int, > 0)
        - ``data.num_workers`` (int, >= 0)
        - ``training.epochs`` (int, > 0)
        - ``training.learning_rate`` (float, > 0)
        - ``training.beta1`` (float, in [0.0, 1.0))
        - ``training.beta2`` (float, in (0.0, 1.0])
        - ``training.n_critic`` (int, > 0)
        - ``training.gradient_accumulation_steps`` (int, > 0)
        - ``training.clip_grad_norm`` (float, >= 0)
        - ``paths.data_root`` (str, must exist)
        - ``paths.checkpoint_dir`` (str)
        - ``paths.log_dir`` (str)

        Additional keys are optional and will use sensible defaults.

    logger : Logger
        The ILGAN logger instance.  Must be an instance of
        :class:`~ilgan.utils.logger.Logger`.

    **kwargs
        Additional keyword arguments passed directly to the
        :class:`ILGANTrainer` constructor.  Currently supported:

        - ``eval_interval`` (int): override the logging eval_interval.
        - ``save_interval`` (int): override the logging save_interval.
        - ``log_interval`` (int): override the logging log_interval.

    Returns
    -------
    ILGANTrainer
        A fully configured :class:`ILGANTrainer` instance, ready to call
        :meth:`ILGANTrainer.train`, :meth:`ILGANTrainer.evaluate`, or
        :meth:`ILGANTrainer.generate`.

    Raises
    ------
    TypeError
        If *config* is not a :class:`Config` instance or *logger* is not
        a :class:`Logger` instance.
    ValueError
        If the config is missing required keys or has invalid values.

    Examples
    --------
    **Basic usage** (recommended)::

        from ilgan.utils.config import Config
        from ilgan.utils.logger import Logger
        from ilgan.training import build_trainer

        cfg = Config()
        logger = Logger(log_dir=cfg.paths.log_dir)
        trainer = build_trainer(cfg, logger)
        trainer.train()

    **With overrides**::

        cfg = Config(overrides={"training.epochs": 200})
        logger = Logger(log_dir="./logs")
        trainer = build_trainer(cfg, logger, eval_interval=10)
        trainer.train()

    **Evaluation only**::

        cfg = Config()
        logger = Logger(log_dir=cfg.paths.log_dir)
        trainer = build_trainer(cfg, logger)
        trainer.evaluate(checkpoint_path="./checkpoints/best_checkpoint.pt")

    **Generation only**::

        cfg = Config()
        logger = Logger(log_dir=cfg.paths.log_dir)
        trainer = build_trainer(cfg, logger)
        trainer.generate(
            num_samples=64,
            checkpoint_path="./checkpoints/best_checkpoint.pt",
            output_path="./generated_samples",
        )

    Notes
    -----
    **Config validation**

    The factory performs a lightweight validation of the config to catch
    common misconfigurations early.  It checks:

    - All required keys are present (not ``None``).
    - Numeric keys are within valid ranges (positive, non-negative, etc.).
    - The data root path exists on disk.

    If validation fails, a :class:`ValueError` is raised with a descriptive
    message indicating which key is problematic.

    **Logging**

    The factory logs a summary of the trainer configuration at INFO level,
    including the device, number of parameters, and key hyperparameters.
    This is useful for experiment tracking and reproducibility.

    **Performance considerations**

    The factory does **not** move models to the GPU or create optimizers
    — that is handled by the :class:`ILGANTrainer` constructor.  The factory
    is a lightweight wrapper that validates inputs and constructs the
    trainer.

    **Custom kwargs**

    The ``**kwargs`` are passed directly to the :class:`ILGANTrainer`
    constructor.  This allows overriding specific config values at
    construction time without modifying the config object.  For example,
    you can override the eval interval for a specific run::

        trainer = build_trainer(cfg, logger, eval_interval=1)

    This is equivalent to setting ``config.logging.eval_interval = 1``
    before constructing the trainer.
    """
    # ── 1. Validate inputs ──────────────────────────────────────────────
    if not isinstance(config, Config):
        raise TypeError(
            f"Expected 'config' to be a Config instance, "
            f"got {type(config).__name__}. "
            f"Please construct a Config object first."
        )

    if not isinstance(logger, Logger):
        raise TypeError(
            f"Expected 'logger' to be a Logger instance, "
            f"got {type(logger).__name__}. "
            f"Please construct a Logger object first."
        )

    # ── 2. Validate critical config keys ─────────────────────────────────
    _validate_config(config)

    # ── 3. Apply kwargs overrides to config ──────────────────────────────
    # If the user passed eval_interval, save_interval, or log_interval as
    # kwargs, we apply them to the config so the trainer picks them up.
    if "eval_interval" in kwargs:
        config["logging.eval_interval"] = kwargs["eval_interval"]
    if "save_interval" in kwargs:
        config["logging.save_interval"] = kwargs["save_interval"]
    if "log_interval" in kwargs:
        config["logging.log_interval"] = kwargs["log_interval"]

    # ── 4. Construct the trainer ─────────────────────────────────────────
    logger.info("Building ILGANTrainer via build_trainer factory...")
    logger.info(f"  Config path: {config._default_path}")
    from ilgan.utils.device import get_device_name
    logger.info(f"  Device: {get_device_name()}")
    logger.info(f"  Latent dim: {config.model.latent_dim}")
    logger.info(f"  Image size: {config.data.image_size}")
    logger.info(f"  Batch size: {config.data.batch_size}")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  n_critic: {config.training.n_critic}")
    logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Mixed precision: {config.training.use_mixed_precision}")
    logger.info(f"  Data root: {config.paths.data_root}")
    logger.info(f"  Checkpoint dir: {config.paths.checkpoint_dir}")
    logger.info(f"  Log dir: {config.paths.log_dir}")

    trainer = ILGANTrainer(config=config, logger=logger)

    logger.info("ILGANTrainer built successfully via build_trainer factory.")

    return trainer


# ──────────────────────────────────────────────────────────────────────────────
# Config validation
# ──────────────────────────────────────────────────────────────────────────────


def _validate_config(config: Config) -> None:
    """Validate that the config contains all required keys with valid values.

    This function checks:

    1. All required keys are present (not ``None``).
    2. Numeric keys are within valid ranges.
    3. The data root path exists on disk.

    Parameters
    ----------
    config : Config
        The ILGAN configuration object to validate.

    Raises
    ------
    ValueError
        If any required key is missing, has an invalid value, or the data
        root path does not exist.

    Notes
    -----
    This validation is intentionally lightweight — it catches the most
    common misconfigurations (missing keys, zero values, non-existent
    paths) without duplicating the full type-checking logic in the
    :class:`Config` class itself.  The :class:`Config` class already
    performs type validation on construction; this function adds
    semantic validation (e.g., "latent_dim must be positive") that is
    specific to the training pipeline.
    """
    import os

    # ── Define required keys and their validation rules ─────────────────
    # Each entry: (dotted_key, human-readable name, validation_fn or None)
    required_keys = [
        # Model
        ("model.latent_dim", "model.latent_dim", lambda v: v > 0),
        ("model.gen_base_channels", "model.gen_base_channels", lambda v: v > 0),
        ("model.disc_base_channels", "model.disc_base_channels", lambda v: v > 0),
        ("model.num_attention_heads", "model.num_attention_heads", lambda v: v > 0),
        ("model.max_boxes", "model.max_boxes", lambda v: v > 0),
        ("model.num_classes", "model.num_classes", lambda v: v > 0),
        # Data
        ("data.image_size", "data.image_size", lambda v: v > 0),
        ("data.batch_size", "data.batch_size", lambda v: v > 0),
        ("data.num_workers", "data.num_workers", lambda v: v >= 0),
        # Training
        ("training.epochs", "training.epochs", lambda v: v > 0),
        ("training.learning_rate", "training.learning_rate", lambda v: v > 0),
        ("training.beta1", "training.beta1", lambda v: 0.0 <= v < 1.0),
        ("training.beta2", "training.beta2", lambda v: 0.0 < v <= 1.0),
        ("training.n_critic", "training.n_critic", lambda v: v > 0),
        (
            "training.gradient_accumulation_steps",
            "training.gradient_accumulation_steps",
            lambda v: v > 0,
        ),
        ("training.clip_grad_norm", "training.clip_grad_norm", lambda v: v >= 0),
        # Paths
        ("paths.data_root", "paths.data_root", None),
        ("paths.checkpoint_dir", "paths.checkpoint_dir", None),
        ("paths.log_dir", "paths.log_dir", None),
    ]

    errors: list[str] = []

    for dotted_key, name, validator in required_keys:
        try:
            value = config[dotted_key]
        except (KeyError, TypeError):
            errors.append(
                f"Missing required config key '{dotted_key}' ({name}). "
                f"Please ensure your config file or overrides include this key."
            )
            continue

        if value is None:
            errors.append(
                f"Config key '{dotted_key}' ({name}) is None. "
                f"Please provide a valid value."
            )
            continue

        if validator is not None:
            try:
                if not validator(value):
                    errors.append(
                        f"Config key '{dotted_key}' ({name}) has invalid value "
                        f"{value!r}.  Please check your configuration."
                    )
            except (TypeError, ValueError) as e:
                errors.append(
                    f"Config key '{dotted_key}' ({name}) could not be validated: "
                    f"{e}.  Value: {value!r}"
                )

    # ── Validate data root path exists ──────────────────────────────────
    try:
        data_root = str(config["paths.data_root"])
        if data_root:
            expanded = os.path.expanduser(data_root)
            if not os.path.exists(expanded):
                errors.append(
                    f"Data root path '{data_root}' (expanded: '{expanded}') "
                    f"does not exist.  Please ensure the dataset directory exists."
                )
    except (KeyError, TypeError):
        errors.append(
            "Missing required config key 'paths.data_root'. "
            "Please ensure your config includes a data root path."
        )

    # ── Raise if any errors were found ──────────────────────────────────
    if errors:
        error_msg = (
            "Config validation failed with the following errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nPlease fix these issues and try again."
        )
        raise ValueError(error_msg)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Optimizers
    "build_optimizers",
    "AdaptiveOptimizerScheduler",
    "build_scheduler",
    # Mixed precision
    "AMPScaler",
    "should_use_amp",
    "autocast_context",
    "create_amp_scaler",
    # Gradient utilities
    "clip_gradients",
    "log_gradient_statistics",
    "detect_nan_inf_gradients",
    "zero_gradients",
    # Checkpointing
    "CheckpointManager",
    "load_or_initialize",
    # Training loop
    "RepresentationAnchor",
    "train_epoch",
    # Validation loop
    "validate",
    "_generate_sample_grid",
    # Top-level trainer
    "ILGANTrainer",
    # Factory
    "build_trainer",
]
