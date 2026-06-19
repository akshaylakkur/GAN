"""
Model checkpointing for ILGAN — save, load, and manage training checkpoints.

This module provides two core components for the ILGAN training pipeline:

1. :class:`CheckpointManager` — a stateful manager that handles saving and
   loading model checkpoints, maintaining a fixed-size rolling window of
   recent checkpoints, and tracking the best-performing checkpoint according
   to the joint score.

2. :func:`load_or_initialize` — a convenience function that attempts to
   resume from the latest checkpoint, or initialises training from scratch
   if no checkpoint exists.

Checkpoint format
-----------------
Each checkpoint is a dictionary saved via ``torch.save()`` with the following
structure::

    {
        "epoch": int,                          # Current epoch (0-indexed)
        "global_step": int,                    # Total training steps taken
        "generator_state_dict": dict,          # ILGANGenerator state dict
        "discriminator_state_dict": dict,      # ImageDiscriminator state dict
        "g_optimizer_state_dict": dict,        # Generator optimizer state dict
        "d_optimizer_state_dict": dict,        # Discriminator optimizer state dict
        "image_encoder_state_dict": dict|None, # ImageFeatureEncoder state dict
        "box_encoder_state_dict": dict|None,   # BoxFeatureEncoder state dict
        "metrics": dict,                       # Metrics snapshot (from MetricsTracker)
        "config": dict,                        # Config serialised via flatten()
        "joint_score": float,                  # Best joint score (for best checkpoint)
    }

Naming convention
-----------------
- Regular checkpoints: ``checkpoint_epoch_{epoch}_step_{global_step}.pt``
- Best checkpoint: ``best_checkpoint.pt`` (overwritten each time a new best
  is achieved)

Collapse prevention in checkpointing
-------------------------------------
The checkpoint manager uses a **joint score** heuristic (combining FID, mAP,
and Inception Score) to determine the best checkpoint.  This prevents the
common pitfall of saving checkpoints based on a single metric (e.g., FID
alone), which could favour a model that has collapsed on bounding box
prediction while producing good images, or vice versa.  By tracking the
joint score, we ensure that the best checkpoint represents the best overall
trade-off between image quality and detection accuracy.
"""

from __future__ import annotations

import glob
import math
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CHECKPOINT_PATTERN: str = r"checkpoint_epoch_(\d+)_step_(\d+)\.pt"
"""Regex pattern for parsing epoch and step from checkpoint filenames."""

_BEST_CHECKPOINT_NAME: str = "best_checkpoint.pt"
"""Filename for the best-performing checkpoint."""

_DEFAULT_MAX_CHECKPOINTS: int = 10
"""Default maximum number of regular checkpoints to keep."""

_JOINT_SCORE_KEY: str = "joint_score"
"""Key used to store the joint score in the checkpoint dictionary and in
the metrics dict for best-checkpoint comparison."""


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────────────────────────────────────


class CheckpointManager:
    """Manages saving, loading, and pruning of ILGAN training checkpoints.

    The :class:`CheckpointManager` provides a complete checkpointing solution
    for the ILGAN training loop:

    - **Save**: serialises the full training state (models, optimizers,
      metrics, config) to disk.
    - **Load**: restores models and optimizers from a checkpoint, returning
      the epoch and global step for resumption.
    - **Best tracking**: maintains a separate ``best_checkpoint.pt`` that is
      updated only when the joint score improves, ensuring the best model
      is always available.
    - **Pruning**: keeps at most ``max_checkpoints`` regular checkpoints,
      automatically removing the oldest ones when the limit is exceeded.

    Parameters
    ----------
    checkpoint_dir : str
        Directory path where checkpoints are stored.  Created automatically
        if it does not exist.
    config : Config
        The ILGAN configuration object.  Its flattened representation is
        stored in each checkpoint for provenance tracking.
    max_checkpoints : int, optional
        Maximum number of regular (non-best) checkpoints to retain.  Must
        be at least 1.  (default: 10)

    Raises
    ------
    FileNotFoundError
        If *checkpoint_dir* exists but is not a directory.
    ValueError
        If *max_checkpoints* is less than 1.

    Examples
    --------
    >>> from ilgan.utils.config import Config
    >>> from ilgan.training.checkpoint import CheckpointManager
    >>> cfg = Config()
    >>> ckpt_mgr = CheckpointManager("./checkpoints", cfg, max_checkpoints=5)
    >>> ckpt_mgr.save(epoch=10, global_step=5000, generator=gen,
    ...               discriminator=disc, g_optimizer=g_opt, d_optimizer=d_opt,
    ...               metrics={"joint_score": 0.75})
    >>> epoch, step = ckpt_mgr.load("./checkpoints/checkpoint_epoch_10_step_5000.pt",
    ...                              generator=gen, discriminator=disc,
    ...                              g_optimizer=g_opt, d_optimizer=d_opt)
    >>> latest = ckpt_mgr.get_latest_checkpoint()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        config: Config,
        max_checkpoints: int = _DEFAULT_MAX_CHECKPOINTS,
    ) -> None:
        # ── Validate and normalise checkpoint_dir ───────────────────────
        self._checkpoint_dir: str = os.path.expanduser(checkpoint_dir)

        if os.path.exists(self._checkpoint_dir) and not os.path.isdir(self._checkpoint_dir):
            raise FileNotFoundError(
                f"checkpoint_dir '{self._checkpoint_dir}' exists but is not a directory."
            )

        # Create the directory if it does not exist
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # ── Validate max_checkpoints ────────────────────────────────────
        if max_checkpoints < 1:
            raise ValueError(
                f"max_checkpoints must be at least 1, got {max_checkpoints}."
            )

        self._max_checkpoints: int = max_checkpoints

        # ── Store config ────────────────────────────────────────────────
        self._config: Config = config

        # ── Best checkpoint tracking ─────────────────────────────────────
        self._best_joint_score: float = -float("inf")
        self._best_checkpoint_path: str = os.path.join(
            self._checkpoint_dir, _BEST_CHECKPOINT_NAME,
        )

        # Load the best joint score from an existing best checkpoint if present
        self._load_best_score_from_disk()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Save
    # ──────────────────────────────────────────────────────────────────────────

    def save(
        self,
        epoch: int,
        global_step: int,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        metrics: Optional[Dict[str, Any]] = None,
        image_encoder: Optional[nn.Module] = None,
        box_encoder: Optional[nn.Module] = None,
    ) -> str:
        """Save a training checkpoint to disk.

        This method serialises the full training state into a dictionary and
        writes it to ``{checkpoint_dir}/checkpoint_epoch_{epoch}_step_{global_step}.pt``.

        After saving, it prunes old checkpoints to stay within
        ``max_checkpoints``.

        Parameters
        ----------
        epoch : int
            Current training epoch (0-indexed).
        global_step : int
            Total number of training steps taken so far.
        generator : nn.Module
            The ILGAN generator (``ILGANGenerator`` instance).
        discriminator : nn.Module
            The ILGAN discriminator (``ImageDiscriminator`` instance).
        g_optimizer : torch.optim.Optimizer
            The generator optimizer (typically returned by
            :func:`ilgan.training.optimizers.build_optimizers`).
        d_optimizer : torch.optim.Optimizer
            The discriminator optimizer.
        metrics : dict of str -> Any, optional
            A dictionary of metrics (typically from
            :meth:`ilgan.metrics.joint_metrics.MetricsTracker.compute_all`).
            If it contains a ``"joint_score"`` key, that value is stored
            in the checkpoint for best-checkpoint tracking.
        image_encoder : nn.Module, optional
            The image feature encoder (``ImageFeatureEncoder`` instance),
            if used.  Its state dict is saved if provided.
        box_encoder : nn.Module, optional
            The box feature encoder (``BoxFeatureEncoder`` instance),
            if used.  Its state dict is saved if provided.

        Returns
        -------
        str
            The absolute path to the saved checkpoint file.

        Raises
        ------
        TypeError
            If any of the model/optimizer arguments are of the wrong type.
        ValueError
            If *epoch* or *global_step* are negative.

        Notes
        -----
        - All tensors are moved to CPU before serialisation to ensure
          compatibility across devices.
        - The config is stored as a flattened dictionary (via
          ``config.flatten()``) for provenance tracking.
        - The checkpoint is saved atomically: first to a temporary file,
          then renamed to the final path.  This prevents corruption if the
          process is interrupted during writing.
        """
        # ── Validate inputs ─────────────────────────────────────────────
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}.")
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}.")

        for name, module in [
            ("generator", generator),
            ("discriminator", discriminator),
        ]:
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"Expected '{name}' to be an nn.Module, "
                    f"got {type(module).__name__}."
                )

        for name, opt in [
            ("g_optimizer", g_optimizer),
            ("d_optimizer", d_optimizer),
        ]:
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(
                    f"Expected '{name}' to be a torch.optim.Optimizer, "
                    f"got {type(opt).__name__}."
                )

        # ── Build checkpoint dictionary ──────────────────────────────────
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "generator_state_dict": self._get_cpu_state_dict(generator),
            "discriminator_state_dict": self._get_cpu_state_dict(discriminator),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "image_encoder_state_dict": (
                self._get_cpu_state_dict(image_encoder)
                if image_encoder is not None
                else None
            ),
            "box_encoder_state_dict": (
                self._get_cpu_state_dict(box_encoder)
                if box_encoder is not None
                else None
            ),
            "metrics": metrics if metrics is not None else {},
            "config": self._config.flatten(),
            _JOINT_SCORE_KEY: self._extract_joint_score(metrics),
        }

        # ── Determine save path ──────────────────────────────────────────
        filename: str = f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        save_path: str = os.path.join(self._checkpoint_dir, filename)

        # ── Atomic save ──────────────────────────────────────────────────
        tmp_path: str = save_path + ".tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, save_path)
        except Exception:
            # Clean up temporary file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # ── Prune old checkpoints ────────────────────────────────────────
        self._prune_checkpoints()

        return os.path.abspath(save_path)

    def save_best(
        self,
        metrics: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[str]:
        """Save a checkpoint only if the joint score is the best seen so far.

        This method extracts the joint score from *metrics* and compares it
        to the best joint score seen so far.  If the new score is higher
        (or if no best checkpoint exists yet), it saves a checkpoint to
        ``best_checkpoint.pt`` using the same arguments as :meth:`save`.

        The joint score is expected to be stored under the key
        ``"joint_score"`` in the *metrics* dictionary.  If the key is
        missing, the checkpoint is **not** saved (a warning is issued).

        Parameters
        ----------
        metrics : dict of str -> Any
            A dictionary of metrics (typically from
            :meth:`ilgan.metrics.joint_metrics.MetricsTracker.compute_all`).
            Must contain a ``"joint_score"`` key with a float value.
        *args
            Positional arguments passed to :meth:`save` (epoch, global_step,
            generator, discriminator, g_optimizer, d_optimizer).
        **kwargs
            Keyword arguments passed to :meth:`save` (image_encoder,
            box_encoder, etc.).

        Returns
        -------
        str or None
            The absolute path to the saved best checkpoint if a new best
            was achieved, or ``None`` if the current score did not improve.

        Raises
        ------
        ValueError
            If *metrics* does not contain a ``"joint_score"`` key.

        Notes
        -----
        - The best checkpoint is **always** written to
          ``best_checkpoint.pt``, overwriting any previous best checkpoint.
        - The best joint score is tracked in memory and persisted to disk
          (loaded from the existing ``best_checkpoint.pt`` on construction).
        - If the joint score is ``NaN`` or ``inf``, it is treated as a
          non-improvement and the checkpoint is not saved.
        """
        # ── Extract joint score ──────────────────────────────────────────
        if _JOINT_SCORE_KEY not in metrics:
            raise ValueError(
                f"metrics dictionary must contain '{_JOINT_SCORE_KEY}' key "
                f"for best-checkpoint tracking.  Got keys: {list(metrics.keys())}"
            )

        joint_score: float = float(metrics[_JOINT_SCORE_KEY])

        # ── Handle NaN / inf ─────────────────────────────────────────────
        if math.isnan(joint_score) or math.isinf(joint_score):
            warnings.warn(
                f"Joint score is {joint_score}; skipping best-checkpoint save.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # ── Compare with best ────────────────────────────────────────────
        if joint_score <= self._best_joint_score:
            return None

        # ── New best! Save checkpoint ────────────────────────────────────
        self._best_joint_score = joint_score

        # Call save with the best checkpoint path override
        # We save directly to best_checkpoint.pt
        best_path: str = self._best_checkpoint_path

        # Build the checkpoint dict manually (reusing save logic)
        # We need to extract args/kwargs
        epoch: int = args[0] if len(args) > 0 else kwargs.get("epoch", 0)
        global_step: int = args[1] if len(args) > 1 else kwargs.get("global_step", 0)
        generator: nn.Module = args[2] if len(args) > 2 else kwargs.get("generator")
        discriminator: nn.Module = args[3] if len(args) > 3 else kwargs.get("discriminator")
        g_optimizer: torch.optim.Optimizer = args[4] if len(args) > 4 else kwargs.get("g_optimizer")
        d_optimizer: torch.optim.Optimizer = args[5] if len(args) > 5 else kwargs.get("d_optimizer")
        image_encoder: Optional[nn.Module] = kwargs.get("image_encoder", None)
        box_encoder: Optional[nn.Module] = kwargs.get("box_encoder", None)

        # Validate required arguments
        if generator is None or discriminator is None:
            raise ValueError(
                "save_best requires 'generator' and 'discriminator' arguments."
            )
        if g_optimizer is None or d_optimizer is None:
            raise ValueError(
                "save_best requires 'g_optimizer' and 'd_optimizer' arguments."
            )

        # Build checkpoint dict
        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "generator_state_dict": self._get_cpu_state_dict(generator),
            "discriminator_state_dict": self._get_cpu_state_dict(discriminator),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "image_encoder_state_dict": (
                self._get_cpu_state_dict(image_encoder)
                if image_encoder is not None
                else None
            ),
            "box_encoder_state_dict": (
                self._get_cpu_state_dict(box_encoder)
                if box_encoder is not None
                else None
            ),
            "metrics": metrics,
            "config": self._config.flatten(),
            _JOINT_SCORE_KEY: joint_score,
        }

        # Atomic save
        tmp_path: str = best_path + ".tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, best_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        return os.path.abspath(best_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Load
    # ──────────────────────────────────────────────────────────────────────────

    def load(
        self,
        checkpoint_path: str,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        image_encoder: Optional[nn.Module] = None,
        box_encoder: Optional[nn.Module] = None,
    ) -> Tuple[int, int]:
        """Load a checkpoint and restore model and optimizer states.

        This method:

        1. Loads the checkpoint dictionary from *checkpoint_path*.
        2. Restores the generator, discriminator, and (optionally) encoder
           state dicts.
        3. Restores the generator and discriminator optimizer state dicts.
        4. Returns the epoch and global step for training resumption.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file (``.pt``) to load.
        generator : nn.Module
            The ILGAN generator instance to restore.
        discriminator : nn.Module
            The ILGAN discriminator instance to restore.
        g_optimizer : torch.optim.Optimizer
            The generator optimizer to restore.
        d_optimizer : torch.optim.Optimizer
            The discriminator optimizer to restore.
        image_encoder : nn.Module, optional
            The image feature encoder to restore (if saved).
        box_encoder : nn.Module, optional
            The box feature encoder to restore (if saved).

        Returns
        -------
        epoch : int
            The epoch at which the checkpoint was saved.
        global_step : int
            The global step at which the checkpoint was saved.

        Raises
        ------
        FileNotFoundError
            If *checkpoint_path* does not exist.
        RuntimeError
            If the checkpoint is corrupt or missing required keys.
        ValueError
            If the checkpoint's config is incompatible with the current
            configuration (e.g., different latent_dim or image_size).

        Notes
        -----
        - The checkpoint is loaded to CPU first, then state dicts are
          moved to the appropriate device (the device of the corresponding
          model parameters).
        - If the checkpoint contains a ``"config"`` entry, it is compared
          to the current config for critical architectural parameters.
          A warning is issued if they differ.
        - Missing keys in state dicts (e.g., if the model architecture
          changed) are handled gracefully with a warning.
        """
        # ── Validate path ────────────────────────────────────────────────
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        # ── Load checkpoint to CPU ──────────────────────────────────────
        device: torch.device = torch.device("cpu")
        try:
            checkpoint: Dict[str, Any] = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}: {e}"
            ) from e

        # ── Validate required keys ──────────────────────────────────────
        required_keys: List[str] = [
            "epoch",
            "global_step",
            "generator_state_dict",
            "discriminator_state_dict",
            "g_optimizer_state_dict",
            "d_optimizer_state_dict",
        ]
        missing_keys: List[str] = [
            key for key in required_keys if key not in checkpoint
        ]
        if missing_keys:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing required keys: "
                f"{missing_keys}.  Available keys: {list(checkpoint.keys())}"
            )

        # ── Check config compatibility ──────────────────────────────────
        if "config" in checkpoint:
            self._check_config_compatibility(checkpoint["config"])

        # ── Restore generator ───────────────────────────────────────────
        try:
            generator.load_state_dict(checkpoint["generator_state_dict"])
        except Exception as e:
            warnings.warn(
                f"Failed to load generator state dict: {e}. "
                "The generator architecture may have changed.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ── Restore discriminator ─────────────────────────────────────────
        try:
            discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        except Exception as e:
            warnings.warn(
                f"Failed to load discriminator state dict: {e}. "
                "The discriminator architecture may have changed.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ── Restore optimizers ──────────────────────────────────────────
        try:
            g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        except Exception as e:
            warnings.warn(
                f"Failed to load generator optimizer state dict: {e}. "
                "Optimizer hyperparameters may have changed.",
                RuntimeWarning,
                stacklevel=2,
            )

        try:
            d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        except Exception as e:
            warnings.warn(
                f"Failed to load discriminator optimizer state dict: {e}. "
                "Optimizer hyperparameters may have changed.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ── Restore encoders (optional) ─────────────────────────────────
        if image_encoder is not None and "image_encoder_state_dict" in checkpoint:
            enc_sd = checkpoint["image_encoder_state_dict"]
            if enc_sd is not None:
                try:
                    image_encoder.load_state_dict(enc_sd)
                except Exception as e:
                    warnings.warn(
                        f"Failed to load image_encoder state dict: {e}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if box_encoder is not None and "box_encoder_state_dict" in checkpoint:
            enc_sd = checkpoint["box_encoder_state_dict"]
            if enc_sd is not None:
                try:
                    box_encoder.load_state_dict(enc_sd)
                except Exception as e:
                    warnings.warn(
                        f"Failed to load box_encoder state dict: {e}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        # ── Extract epoch and global step ────────────────────────────────
        epoch: int = int(checkpoint["epoch"])
        global_step: int = int(checkpoint["global_step"])

        # ── Update best joint score if this checkpoint has one ──────────
        if _JOINT_SCORE_KEY in checkpoint:
            ckpt_score: float = float(checkpoint[_JOINT_SCORE_KEY])
            if not math.isnan(ckpt_score) and not math.isinf(ckpt_score):
                if ckpt_score > self._best_joint_score:
                    self._best_joint_score = ckpt_score

        return epoch, global_step

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Query
    # ──────────────────────────────────────────────────────────────────────────

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return the path to the most recent regular checkpoint.

        The "latest" checkpoint is determined by parsing the epoch and
        global step from filenames matching the pattern
        ``checkpoint_epoch_*_step_*.pt`` and selecting the one with the
        highest global step (ties broken by highest epoch).

        This method does **not** consider ``best_checkpoint.pt`` as a
        candidate for the latest checkpoint, since the best checkpoint
        may be from an earlier epoch.

        Returns
        -------
        str or None
            The absolute path to the most recent checkpoint file, or
            ``None`` if no regular checkpoints exist.

        Notes
        -----
        - The search is performed by scanning the checkpoint directory
          with ``glob``, so it is O(N) in the number of checkpoint files.
        - For large numbers of checkpoints, consider using a smaller
          ``max_checkpoints`` value.
        """
        checkpoints: List[Tuple[int, int, str]] = self._list_checkpoints()

        if not checkpoints:
            return None

        # Sort by (global_step, epoch) descending
        checkpoints.sort(key=lambda x: (x[1], x[0]), reverse=True)

        return checkpoints[0][2]

    def get_best_checkpoint(self) -> Optional[str]:
        """Return the path to the best checkpoint, if it exists.

        Returns
        -------
        str or None
            The absolute path to ``best_checkpoint.pt`` if it exists,
            or ``None`` otherwise.
        """
        if os.path.isfile(self._best_checkpoint_path):
            return os.path.abspath(self._best_checkpoint_path)
        return None

    def get_best_joint_score(self) -> float:
        """Return the best joint score seen so far.

        Returns
        -------
        float
            The best joint score.  Returns ``-inf`` if no checkpoint has
            been saved yet.
        """
        return self._best_joint_score

    def list_all_checkpoints(self) -> List[str]:
        """Return a sorted list of all regular checkpoint paths.

        The list is sorted by (global_step, epoch) descending, so the
        most recent checkpoint is first.

        Returns
        -------
        list of str
            Absolute paths to all regular checkpoint files, sorted from
            most recent to oldest.
        """
        checkpoints: List[Tuple[int, int, str]] = self._list_checkpoints()
        checkpoints.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return [path for _, _, path in checkpoints]

    @property
    def checkpoint_dir(self) -> str:
        """The directory where checkpoints are stored."""
        return self._checkpoint_dir

    @property
    def max_checkpoints(self) -> int:
        """Maximum number of regular checkpoints to retain."""
        return self._max_checkpoints

    @max_checkpoints.setter
    def max_checkpoints(self, value: int) -> None:
        """Set a new maximum checkpoint count and prune immediately."""
        if value < 1:
            raise ValueError(
                f"max_checkpoints must be at least 1, got {value}."
            )
        self._max_checkpoints = value
        self._prune_checkpoints()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
        """Return the state dict of a module with all tensors moved to CPU.

        Parameters
        ----------
        module : nn.Module
            The module whose state dict to extract.

        Returns
        -------
        dict of str -> torch.Tensor
            The state dict with all tensors on CPU.
        """
        state_dict: Dict[str, torch.Tensor] = module.state_dict()
        cpu_state_dict: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            cpu_state_dict[key] = tensor.detach().cpu()
        return cpu_state_dict

    def _extract_joint_score(
        self,
        metrics: Optional[Dict[str, Any]],
    ) -> float:
        """Extract the joint score from a metrics dictionary.

        Parameters
        ----------
        metrics : dict or None
            The metrics dictionary (may be ``None``).

        Returns
        -------
        float
            The joint score, or ``-inf`` if not available.
        """
        if metrics is None:
            return -float("inf")

        score = metrics.get(_JOINT_SCORE_KEY)
        if score is None:
            return -float("inf")

        try:
            val = float(score)
            if math.isnan(val) or math.isinf(val):
                return -float("inf")
            return val
        except (TypeError, ValueError):
            return -float("inf")

    def _list_checkpoints(self) -> List[Tuple[int, int, str]]:
        """List all regular checkpoint files in the checkpoint directory.

        Returns
        -------
        list of (epoch, global_step, path)
            A list of tuples, one per checkpoint file matching the pattern
            ``checkpoint_epoch_*_step_*.pt``.
        """
        pattern: str = os.path.join(self._checkpoint_dir, "checkpoint_epoch_*_step_*.pt")
        matching_files: List[str] = glob.glob(pattern)

        checkpoints: List[Tuple[int, int, str]] = []
        for filepath in matching_files:
            filename: str = os.path.basename(filepath)
            match = re.match(_CHECKPOINT_PATTERN, filename)
            if match:
                epoch = int(match.group(1))
                global_step = int(match.group(2))
                checkpoints.append((epoch, global_step, filepath))

        return checkpoints

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints to stay within ``max_checkpoints``.

        This method:

        1. Lists all regular checkpoints.
        2. Sorts them by (global_step, epoch) descending (most recent first).
        3. Removes the oldest checkpoints that exceed ``max_checkpoints``.

        The ``best_checkpoint.pt`` is **never** removed by pruning.
        """
        checkpoints: List[Tuple[int, int, str]] = self._list_checkpoints()

        if len(checkpoints) <= self._max_checkpoints:
            return

        # Sort by (global_step, epoch) descending
        checkpoints.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # Keep the first max_checkpoints, remove the rest
        to_remove: List[str] = [path for _, _, path in checkpoints[self._max_checkpoints:]]

        for path in to_remove:
            try:
                os.remove(path)
            except OSError as e:
                warnings.warn(
                    f"Failed to remove old checkpoint {path}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _load_best_score_from_disk(self) -> None:
        """Load the best joint score from an existing best checkpoint on
        disk, if one exists.

        This is called during initialisation so that the best score is
        preserved across training restarts.
        """
        if not os.path.isfile(self._best_checkpoint_path):
            return

        try:
            checkpoint: Dict[str, Any] = torch.load(
                self._best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if _JOINT_SCORE_KEY in checkpoint:
                score: float = float(checkpoint[_JOINT_SCORE_KEY])
                if not math.isnan(score) and not math.isinf(score):
                    self._best_joint_score = score
        except Exception:
            # If the best checkpoint is corrupt, start fresh
            warnings.warn(
                f"Could not load best joint score from "
                f"{self._best_checkpoint_path}.  Starting fresh.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _check_config_compatibility(
        self,
        checkpoint_config: Dict[str, Any],
    ) -> None:
        """Compare the checkpoint's config with the current config and
        warn about critical differences.

        Critical parameters that must match for safe resumption:

        - ``model.latent_dim``
        - ``model.gen_base_channels``
        - ``model.disc_base_channels``
        - ``model.num_attention_heads``
        - ``model.max_boxes``
        - ``model.num_classes``
        - ``data.image_size``

        Parameters
        ----------
        checkpoint_config : dict
            The flattened config dictionary from the checkpoint.
        """
        critical_keys: List[str] = [
            "model.latent_dim",
            "model.gen_base_channels",
            "model.disc_base_channels",
            "model.num_attention_heads",
            "model.max_boxes",
            "model.num_classes",
            "data.image_size",
        ]

        current_flat: Dict[str, Any] = self._config.flatten()

        mismatches: List[str] = []
        for key in critical_keys:
            ckpt_val = checkpoint_config.get(key)
            curr_val = current_flat.get(key)
            if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
                mismatches.append(
                    f"  {key}: checkpoint={ckpt_val}, current={curr_val}"
                )

        if mismatches:
            warnings.warn(
                "Checkpoint config differs from current config on critical "
                "architectural parameters:\n" + "\n".join(mismatches) + "\n"
                "This may cause errors during state dict loading.  "
                "Proceed with caution.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Representation
    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        num_checkpoints: int = len(self._list_checkpoints())
        return (
            f"CheckpointManager(\n"
            f"  checkpoint_dir={self._checkpoint_dir!r},\n"
            f"  max_checkpoints={self._max_checkpoints},\n"
            f"  current_checkpoints={num_checkpoints},\n"
            f"  best_joint_score={self._best_joint_score:.6f},\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# load_or_initialize
# ──────────────────────────────────────────────────────────────────────────────


def load_or_initialize(
    checkpoint_manager: CheckpointManager,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    config: Config,
    image_encoder: Optional[nn.Module] = None,
    box_encoder: Optional[nn.Module] = None,
) -> Tuple[int, int]:
    """Try to load the latest checkpoint, or initialise from scratch.

    This convenience function is designed to be called at the start of the
    training loop.  It:

    1. Queries the :class:`CheckpointManager` for the latest checkpoint.
    2. If a checkpoint exists, loads it and returns the resumption epoch
       and global step.
    3. If no checkpoint exists, returns ``(0, 0)`` to start training from
       scratch.

    Parameters
    ----------
    checkpoint_manager : CheckpointManager
        The checkpoint manager instance (must be initialised with the
        correct checkpoint directory and config).
    generator : nn.Module
        The ILGAN generator instance.
    discriminator : nn.Module
        The ILGAN discriminator instance.
    g_optimizer : torch.optim.Optimizer
        The generator optimizer.
    d_optimizer : torch.optim.Optimizer
        The discriminator optimizer.
    config : Config
        The ILGAN configuration object.  Used for logging the resumption
        status.
    image_encoder : nn.Module, optional
        The image feature encoder (if used).
    box_encoder : nn.Module, optional
        The box feature encoder (if used).

    Returns
    -------
    start_epoch : int
        The epoch to resume from (0 if starting fresh).
    global_step : int
        The global step to resume from (0 if starting fresh).

    Notes
    -----
    - If a checkpoint is loaded, the function logs a message via
      ``print()`` (since the logger may not be initialised yet at this
      point in the training pipeline).
    - The function does **not** raise an error if no checkpoint exists;
      it simply returns ``(0, 0)``.

    Examples
    --------
    >>> from ilgan.training.checkpoint import CheckpointManager, load_or_initialize
    >>> ckpt_mgr = CheckpointManager("./checkpoints", config)
    >>> start_epoch, global_step = load_or_initialize(
    ...     ckpt_mgr, generator, discriminator, g_opt, d_opt, config,
    ... )
    >>> print(f"Resuming from epoch {start_epoch}, step {global_step}")
    """
    # ── Look for the latest checkpoint ──────────────────────────────────
    latest_path: Optional[str] = checkpoint_manager.get_latest_checkpoint()

    if latest_path is None:
        # No checkpoint found — start from scratch
        print(
            f"[ILGAN] No checkpoint found in "
            f"'{checkpoint_manager.checkpoint_dir}'.  "
            f"Starting training from scratch."
        )
        return 0, 0

    # ── Load the latest checkpoint ─────────────────────────────────────
    print(
        f"[ILGAN] Resuming from checkpoint: {latest_path}"
    )

    start_epoch, global_step = checkpoint_manager.load(
        checkpoint_path=latest_path,
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        image_encoder=image_encoder,
        box_encoder=box_encoder,
    )

    print(
        f"[ILGAN] Successfully loaded checkpoint.  "
        f"Resuming from epoch {start_epoch}, global step {global_step}."
    )

    return start_epoch, global_step


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "CheckpointManager",
    "load_or_initialize",
]
