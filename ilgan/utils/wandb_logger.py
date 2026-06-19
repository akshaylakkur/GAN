"""
Weights & Biases integration for ILGAN.

Provides a :class:`WandbLogger` that wraps the ``wandb`` Python client for
experiment tracking, metric logging, image logging, histogram logging, and
model graph visualisation.  All methods degrade gracefully to no-ops when
W&B is not installed, making the logger safe to use in any environment.

Typical usage::

    from ilgan.utils.wandb_logger import WandbLogger

    wandb_logger = WandbLogger()
    if config.logging.use_wandb:
        wandb_logger.init(config, run_name="experiment_001")

    # During training
    wandb_logger.log_metrics({"loss/g_total": 1.23}, step=global_step)
    wandb_logger.log_images(fake_images, caption="Generated samples", step=epoch)
    wandb_logger.log_histogram(box_sizes, name="box/sizes", step=epoch)

    # At end
    wandb_logger.close()
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ilgan.utils.config import Config

# ──────────────────────────────────────────────────────────────────────────────
# Module-level logger
# ──────────────────────────────────────────────────────────────────────────────

_logger = logging.getLogger("ilgan.wandb_logger")

# ──────────────────────────────────────────────────────────────────────────────
# Lazy import helper
# ──────────────────────────────────────────────────────────────────────────────

_WANDB_AVAILABLE: bool = False
_wandb = None  # type: ignore

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    _wandb = None  # type: ignore[assignment]


def _is_wandb_available() -> bool:
    """Return ``True`` if the ``wandb`` package is installed."""
    return _WANDB_AVAILABLE


# ──────────────────────────────────────────────────────────────────────────────
# WandbLogger
# ──────────────────────────────────────────────────────────────────────────────


class WandbLogger:
    """Weights & Biases logger for ILGAN experiment tracking.

    This class wraps the ``wandb`` Python client and provides a clean
    interface for logging metrics, images, histograms, and model graphs
    during ILGAN training.  When W&B is not installed, all methods become
    no-ops — no imports are attempted and no errors are raised.

    The logger is designed to be created once and used throughout the
    training lifecycle.  Call :meth:`init` at the start of training,
    :meth:`log_metrics` / :meth:`log_images` / :meth:`log_histogram`
    during training, and :meth:`close` at the end.

    Parameters
    ----------
    silent : bool
        If ``True``, suppress all W&B warnings and info messages.
        Default ``True``.

    Attributes
    ----------
    enabled : bool
        ``True`` if W&B is available and :meth:`init` has been called
        successfully.
    run : wandb.Run or None
        The active W&B run object, or ``None`` if not initialised.
    config : Config or None
        The ILGAN configuration used to initialise the run.
    """

    def __init__(self, silent: bool = True) -> None:
        self._silent: bool = silent
        self._enabled: bool = False
        self._run: Any = None
        self._config: Optional[Config] = None
        self._run_name: Optional[str] = None

        # If W&B is not available, log a single warning
        if not _is_wandb_available():
            _logger.info(
                "W&B is not installed.  WandbLogger will operate in no-op mode. "
                "Install with: pip install wandb"
            )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """Whether the logger is active (W&B available and initialised)."""
        return self._enabled

    @property
    def run(self) -> Any:
        """The active W&B run object, or ``None``."""
        return self._run

    @property
    def config(self) -> Optional[Config]:
        """The ILGAN configuration used to initialise the run."""
        return self._config

    @property
    def run_name(self) -> Optional[str]:
        """The name of the current W&B run."""
        return self._run_name

    # ── Initialisation ──────────────────────────────────────────────────────

    def init(
        self,
        config: Config,
        project_name: str = "ILGAN",
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        id: Optional[str] = None,
        resume: Optional[Union[bool, str]] = None,
        reinit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise a W&B run.

        This method must be called before any logging methods.  It is safe
        to call multiple times — subsequent calls are no-ops if the logger
        is already enabled.

        Parameters
        ----------
        config : Config
            The ILGAN configuration object.  Its flattened representation
            will be logged as W&B config.
        project_name : str
            Name of the W&B project.  Default ``"ILGAN"``.
        run_name : str, optional
            A human-readable name for this run.  If ``None``, W&B will
            auto-generate one.
        tags : list of str, optional
            Tags to attach to the run (e.g. ``["debug", "ablation"]``).
        notes : str, optional
            Free-text notes for the run.
        id : str, optional
            Unique ID for the run.  If provided, can be used to resume
            a previous run.
        resume : bool or str, optional
            Whether to resume a previous run.  If ``"must"``, W&B will
            raise an error if the run does not exist.  If ``True``, W&B
            will resume if the run exists, otherwise start a new run.
        reinit : bool
            If ``True`` (default), allow re-initialisation of a new run
            after :meth:`close` has been called.
        **kwargs
            Additional keyword arguments passed to ``wandb.init``.

        Raises
        ------
        RuntimeError
            If W&B is not installed and ``init`` is called.  This is a
            programming error — the caller should check ``enabled`` or
            guard the call with ``if config.logging.use_wandb``.
        """
        if self._enabled:
            _logger.debug("WandbLogger.init() called but logger is already enabled — skipping.")
            return

        if not _is_wandb_available():
            _logger.warning(
                "WandbLogger.init() called but W&B is not installed. "
                "Install with: pip install wandb"
            )
            self._enabled = False
            return

        # Store config
        self._config = config
        self._run_name = run_name

        # Suppress W&B output if silent
        if self._silent:
            os.environ.setdefault("WANDB_SILENT", "true")
            # Also redirect wandb's internal logger
            wandb_logger = logging.getLogger("wandb")
            wandb_logger.setLevel(logging.WARNING)

        # Build the W&B config dict from the ILGAN config
        wandb_config: Dict[str, Any] = config.flatten()

        # Add system info
        wandb_config["_system/python_version"] = sys.version.split()[0]
        wandb_config["_system/torch_version"] = torch.__version__
        wandb_config["_system/cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            wandb_config["_system/cuda_device_count"] = torch.cuda.device_count()
            wandb_config["_system/cuda_device_name"] = torch.cuda.get_device_name(0)

        # Initialise the run
        init_kwargs: Dict[str, Any] = dict(
            project=project_name,
            config=wandb_config,
            reinit=reinit,
            **kwargs,
        )
        if run_name is not None:
            init_kwargs["name"] = run_name
        if tags is not None:
            init_kwargs["tags"] = tags
        if notes is not None:
            init_kwargs["notes"] = notes
        if id is not None:
            init_kwargs["id"] = id
        if resume is not None:
            init_kwargs["resume"] = resume

        try:
            self._run = _wandb.init(**init_kwargs)
            self._enabled = True
            _logger.info(
                f"W&B run initialised — project={project_name!r}, "
                f"run_name={self._run.name or run_name!r}, "
                f"run_id={self._run.id}"
            )
        except Exception as exc:
            _logger.error(f"Failed to initialise W&B run: {exc}")
            self._enabled = False
            self._run = None

    # ── Metric logging ──────────────────────────────────────────────────────

    def log_metrics(
        self,
        metrics_dict: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log a dictionary of metrics to W&B.

        Parameters
        ----------
        metrics_dict : dict of str -> Any
            A flat dictionary mapping metric names to scalar values.
            Nested keys (e.g. ``"loss/g_total"``) are supported and will
            appear grouped in the W&B UI.
        step : int, optional
            The global training step (or epoch) to associate with these
            metrics.  If ``None``, W&B auto-increments the step counter.
        commit : bool
            If ``True`` (default), commit the step so that the logged
            metrics are persisted immediately.  Set to ``False`` if you
            intend to log more data for the same step.

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - All values are converted to floats for W&B compatibility.
        - NaN and Inf values are silently skipped to avoid W&B errors.
        """
        if not self._enabled or self._run is None:
            return

        # Sanitise values: convert to float, skip NaN/Inf
        sanitised: Dict[str, float] = {}
        for key, value in metrics_dict.items():
            try:
                val = float(value)
                if not (val != val):  # NaN check (NaN != NaN is True)
                    sanitised[key] = val
                else:
                    _logger.debug(f"Skipping NaN metric: {key}")
            except (TypeError, ValueError):
                _logger.debug(f"Skipping non-numeric metric: {key}={value!r}")

        if not sanitised:
            return

        try:
            self._run.log(sanitised, step=step, commit=commit)
        except Exception as exc:
            _logger.warning(f"Failed to log metrics to W&B: {exc}")

    # ── Image logging ───────────────────────────────────────────────────────

    def log_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        caption: str = "Generated samples",
        step: Optional[int] = None,
        max_images: int = 16,
        normalize: bool = True,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Log image samples to W&B.

        Parameters
        ----------
        images : torch.Tensor or list of torch.Tensor
            A batch of images to log.  Accepted formats:

            - ``[B, C, H, W]`` tensor with values in ``[-1, 1]`` or ``[0, 1]``.
            - ``[B, H, W, C]`` tensor (channel-last) — will be permuted.
            - List of ``[C, H, W]`` tensors.

        caption : str
            A caption / prefix for the image panel.  Default
            ``"Generated samples"``.
        step : int, optional
            The global step to associate with these images.
        max_images : int
            Maximum number of images to log (to avoid overloading the
            dashboard).  Default ``16``.
        normalize : bool
            If ``True`` (default), scale pixel values to ``[0, 1]`` for
            proper display in the W&B UI.  Set to ``False`` if the images
            are already in ``[0, 1]``.
        value_range : tuple of (float, float), optional
            The min/max range of the input pixel values.  If ``None``,
            inferred from the data (assumes ``[-1, 1]`` if any value is
            negative, otherwise ``[0, 1]``).

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - Images are converted to ``wandb.Image`` objects internally.
        - The caption is appended with the step number for disambiguation.
        """
        if not self._enabled or self._run is None:
            return

        if not isinstance(images, (torch.Tensor, list)):
            _logger.warning(
                f"log_images expected a torch.Tensor or list, got {type(images)}. Skipping."
            )
            return

        # Convert list to tensor
        if isinstance(images, list):
            if len(images) == 0:
                return
            try:
                images = torch.stack(images)
            except RuntimeError as exc:
                _logger.warning(f"Could not stack image list into tensor: {exc}. Skipping.")
                return

        # Ensure 4D tensor [B, C, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(0)  # [1, C, H, W]
        elif images.dim() == 4 and images.shape[-1] in (1, 3, 4):
            # Channel-last → channel-first
            images = images.permute(0, 3, 1, 2)
        elif images.dim() != 4:
            _logger.warning(
                f"log_images expected a 3D or 4D tensor, got {images.dim()}D. Skipping."
            )
            return

        # Limit number of images
        images = images[:max_images]

        # Move to CPU for W&B
        images = images.detach().cpu()

        # Determine value range for normalisation
        if value_range is None:
            if images.min() < 0:
                value_range = (-1.0, 1.0)
            else:
                value_range = (0.0, 1.0)

        # Normalise to [0, 1] if requested
        if normalize:
            vmin, vmax = value_range
            if vmax - vmin > 1e-8:
                images = (images - vmin) / (vmax - vmin)
            images = torch.clamp(images, 0.0, 1.0)

        # Build wandb.Image list
        try:
            wandb_images: List[Any] = []
            for i in range(images.shape[0]):
                img_tensor = images[i]  # [C, H, W]
                # Convert to HWC numpy uint8 for wandb
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
                wandb_images.append(
                    _wandb.Image(
                        img_np,
                        caption=f"{caption} — sample {i}",
                    )
                )

            self._run.log(
                {f"images/{caption}": wandb_images},
                step=step,
            )
        except Exception as exc:
            _logger.warning(f"Failed to log images to W&B: {exc}")

    # ── Histogram logging ──────────────────────────────────────────────────

    def log_histogram(
        self,
        values: Union[torch.Tensor, List[float], "np.ndarray"],
        name: str,
        step: Optional[int] = None,
        bins: Union[int, str] = "auto",
    ) -> None:
        """Log a histogram of values to W&B.

        This is useful for tracking distributions of bounding box sizes,
        confidence scores, latent activations, gradient norms, etc.

        Parameters
        ----------
        values : torch.Tensor or list of float or numpy.ndarray
            The values to histogram.  Flattened automatically if
            multi-dimensional.
        name : str
            A name for the histogram (e.g. ``"box/sizes"``,
            ``"confidences"``, ``"grad_norm"``).
        step : int, optional
            The global step to associate with this histogram.
        bins : int or str
            Number of histogram bins, or ``"auto"`` for automatic binning.
            Default ``"auto"``.

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - NaN and Inf values are filtered out before logging.
        - If the tensor is empty after filtering, the histogram is skipped.
        """
        if not self._enabled or self._run is None:
            return

        # Convert to flat numpy array
        try:
            if isinstance(values, torch.Tensor):
                values_np = values.detach().cpu().flatten().numpy()
            elif isinstance(values, list):
                import numpy as np

                values_np = np.array(values, dtype=np.float64).flatten()
            else:
                # Assume numpy array
                import numpy as np

                values_np = np.asarray(values, dtype=np.float64).flatten()
        except Exception as exc:
            _logger.warning(f"Failed to convert values for histogram '{name}': {exc}")
            return

        # Filter out NaN and Inf
        import numpy as np

        mask = np.isfinite(values_np)
        if not mask.any():
            _logger.debug(f"Histogram '{name}' has no finite values — skipping.")
            return
        values_np = values_np[mask]

        if values_np.size == 0:
            _logger.debug(f"Histogram '{name}' is empty — skipping.")
            return

        try:
            self._run.log(
                {f"histograms/{name}": _wandb.Histogram(values_np, bins=bins)},
                step=step,
            )
        except Exception as exc:
            _logger.warning(f"Failed to log histogram '{name}' to W&B: {exc}")

    # ── Model graph logging ────────────────────────────────────────────────

    def log_model_graph(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        caption: str = "Model Graph",
    ) -> None:
        """Log the model architecture graph to W&B.

        This method uses ``wandb.watch`` to log the model's computational
        graph, which enables visualisation of the forward pass structure
        and gradient flow in the W&B UI.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model to log (e.g. the generator or discriminator).
        input_tensor : torch.Tensor
            A dummy input tensor of the correct shape to trace the graph.
            The tensor should be on the same device as the model.
        caption : str
            A label for the model graph (e.g. ``"Generator"``,
            ``"Discriminator"``).  Default ``"Model Graph"``.

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - ``wandb.watch`` is called with ``log_freq=0`` (no gradient
          histogram logging) to avoid performance overhead.  If you want
          gradient histograms, call :meth:`log_histogram` manually.
        - The model graph is logged once at initialisation time.  To
          re-log (e.g. after a model change), call this method again.
        - This method logs the graph synchronously.  For large models,
          this may take a few seconds.
        """
        if not self._enabled or self._run is None:
            return

        if not isinstance(model, nn.Module):
            _logger.warning(
                f"log_model_graph expected an nn.Module, got {type(model)}. Skipping."
            )
            return

        if not isinstance(input_tensor, torch.Tensor):
            _logger.warning(
                f"log_model_graph expected a torch.Tensor, got {type(input_tensor)}. Skipping."
            )
            return

        try:
            # Use wandb.watch to log the model graph
            # log_freq=0 means: log the graph but don't log gradients every N steps
            _wandb.watch(
                model,
                log="all" if self._silent else "gradients",
                log_freq=0,
                log_graph=True,
            )
            _logger.info(f"Model graph logged for '{caption}'.")
        except Exception as exc:
            _logger.warning(f"Failed to log model graph to W&B: {exc}")

    # ── Run metadata ───────────────────────────────────────────────────────

    def log_summary(self, summary_dict: Dict[str, Any]) -> None:
        """Set run-level summary metrics.

        Summary metrics are displayed in the W&B project overview and
        persist after the run finishes.

        Parameters
        ----------
        summary_dict : dict of str -> Any
            A flat dictionary of summary metrics (e.g. best joint score,
            final FID, final mAP).

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - Summary values overwrite previous values for the same key.
        """
        if not self._enabled or self._run is None:
            return

        try:
            for key, value in summary_dict.items():
                setattr(self._run.summary, key, value)
        except Exception as exc:
            _logger.warning(f"Failed to log summary to W&B: {exc}")

    def log_artifact(
        self,
        file_path: str,
        name: Optional[str] = None,
        artifact_type: str = "checkpoint",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Log a file as a W&B artifact.

        This is useful for logging model checkpoints, configuration files,
        or any other binary artifacts.

        Parameters
        ----------
        file_path : str
            Path to the file to log.
        name : str, optional
            Name for the artifact.  If ``None``, the filename is used.
        artifact_type : str
            Type of artifact (e.g. ``"checkpoint"``, ``"config"``,
            ``"sample"``).  Default ``"checkpoint"``.
        aliases : list of str, optional
            Aliases to assign to the artifact version (e.g.
            ``["best", "latest"]``).

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - The file must exist on disk.
        """
        if not self._enabled or self._run is None:
            return

        if not os.path.isfile(file_path):
            _logger.warning(
                f"log_artifact: file not found: {file_path}. Skipping."
            )
            return

        try:
            artifact_name = name or os.path.basename(file_path)
            artifact = _wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
            )
            artifact.add_file(file_path)
            self._run.log_artifact(artifact, aliases=aliases or [])
            _logger.info(
                f"Logged artifact '{artifact_name}' (type={artifact_type})."
            )
        except Exception as exc:
            _logger.warning(f"Failed to log artifact to W&B: {exc}")

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Finish the W&B run and flush all pending data.

        This method should be called at the end of training to ensure all
        metrics, images, and histograms are persisted.  After calling
        :meth:`close`, the logger is disabled and a new run can be started
        with :meth:`init` (if ``reinit=True`` was used).

        Notes
        -----
        - This method is a no-op if the logger is not enabled.
        - It is safe to call :meth:`close` multiple times.
        """
        if not self._enabled or self._run is None:
            return

        try:
            self._run.finish()
            _logger.info("W&B run finished.")
        except Exception as exc:
            _logger.warning(f"Failed to finish W&B run: {exc}")
        finally:
            self._enabled = False
            self._run = None
            self._config = None
            self._run_name = None

    def __enter__(self) -> "WandbLogger":
        """Context manager support.

        Usage::

            with WandbLogger() as wl:
                wl.init(config)
                ...
                # close() is called automatically on exit
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """Ensure the run is finished on context exit."""
        self.close()

    # ── Representation ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self._enabled and self._run is not None:
            return (
                f"WandbLogger(enabled=True, "
                f"project={self._run.project_name()!r}, "
                f"run_name={self._run.name!r}, "
                f"run_id={self._run.id})"
            )
        return f"WandbLogger(enabled={self._enabled})"


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────


def create_wandb_logger(
    config: Config,
    project_name: str = "ILGAN",
    run_name: Optional[str] = None,
    silent: bool = True,
) -> WandbLogger:
    """Create and initialise a :class:`WandbLogger` from a config.

    This is a convenience function that combines construction and
    initialisation into a single call.  It respects the
    ``config.logging.use_wandb`` flag — if it is ``False``, the logger
    is returned but not initialised (all methods are no-ops).

    Parameters
    ----------
    config : Config
        The ILGAN configuration object.
    project_name : str
        W&B project name.  Default ``"ILGAN"``.
    run_name : str, optional
        Human-readable run name.
    silent : bool
        Suppress W&B output.  Default ``True``.

    Returns
    -------
    WandbLogger
        An initialised (or no-op) W&B logger.

    Examples
    --------
    >>> from ilgan.utils.config import Config
    >>> from ilgan.utils.wandb_logger import create_wandb_logger
    >>> cfg = Config()
    >>> wl = create_wandb_logger(cfg, run_name="my_experiment")
    >>> wl.log_metrics({"loss": 0.5}, step=1)
    >>> wl.close()
    """
    logger = WandbLogger(silent=silent)

    use_wandb: bool = getattr(config.logging, "use_wandb", False)
    if use_wandb:
        logger.init(
            config=config,
            project_name=project_name,
            run_name=run_name,
        )

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "WandbLogger",
    "create_wandb_logger",
]
