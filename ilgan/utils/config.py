"""
Configuration management for ILGAN.

Provides a `Config` class that loads YAML defaults, accepts user overrides,
exposes values via attribute and dictionary access, validates types, and
verifies critical paths exist at construction time.
"""

from __future__ import annotations

import os
import copy
import numbers
import pathlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "default_config.yaml",
)

# Schema: maps dotted key → (expected_type, constraint_fn_or_None)
# Keys marked with "optional" are not required to be present (they have defaults
# in the YAML or are read dynamically).  The schema validates them only if they
# exist in the loaded config.
_TYPE_SCHEMA: Dict[str, Tuple[type, Optional[str]]] = {
    # ── data ───────────────────────────────────────────────────────────────────
    "data.image_size": (int, "must be positive"),
    "data.batch_size": (int, "must be positive"),
    "data.num_workers": (int, "must be non-negative"),
    "data.augment_prob": (float, "must be in [0.0, 1.0]"),
    "data.yolo_format": (bool, None),
    # ── model ──────────────────────────────────────────────────────────────────
    "model.latent_dim": (int, "must be positive"),
    "model.gen_base_channels": (int, "must be positive"),
    "model.disc_base_channels": (int, "must be positive"),
    "model.num_attention_heads": (int, "must be positive"),
    "model.max_boxes": (int, "must be positive"),
    "model.num_classes": (int, "must be positive"),
    # ── loss ───────────────────────────────────────────────────────────────────
    "loss.adv_weight": (float, "must be non-negative"),
    "loss.box_weight": (float, "must be non-negative"),
    "loss.class_weight": (float, "must be non-negative"),
    "loss.confidence_weight": (float, "must be non-negative"),
    "loss.diversity_weight": (float, "must be non-negative"),
    "loss.consistency_weight": (float, "must be non-negative"),
    "loss.entropy_weight": (float, "must be non-negative"),
    "loss.repulsion_weight": (float, "must be non-negative"),
    "loss.feature_diversity_weight": (float, "must be non-negative"),
    "loss.latent_diversity_weight": (float, "must be non-negative"),
    "loss.spectral_reg_weight": (float, "must be non-negative"),
    "loss.gp_weight": (float, "must be non-negative"),
    "loss.w_global": (float, "must be in [0.0, 1.0]"),
    "loss.repulsion_threshold": (float, "must be in (0.0, 1.0]"),
    "loss.noise_schedule_initial": (float, "must be positive"),
    "loss.noise_schedule_final": (float, "must be non-negative"),
    # ── training ───────────────────────────────────────────────────────────────
    "training.epochs": (int, "must be positive"),
    "training.learning_rate": (float, "must be positive"),
    "training.beta1": (float, "must be in [0.0, 1.0)"),
    "training.beta2": (float, "must be in (0.0, 1.0]"),
    "training.n_critic": (int, "must be positive"),
    "training.gradient_accumulation_steps": (int, "must be positive"),
    "training.use_mixed_precision": (bool, None),
    "training.grad_checkpoint": (bool, None),
    "training.clip_grad_norm": (float, "must be non-negative"),
    "training.adaptive_lr": (bool, None),
    "training.gradient_balance": (bool, None),
    "training.representation_anchor_frequency": (int, "must be positive"),
    "training.representation_anchor_weight": (float, "must be non-negative"),
    # ── training.noise_schedule (sub-section) ─────────────────────────────────
    "training.noise_schedule.initial_noise_std": (float, "must be positive"),
    "training.noise_schedule.min_noise_std": (float, "must be non-negative"),
    "training.noise_schedule.warmup_steps": (int, "must be non-negative"),
    # total_steps is optional (can be null for auto-compute)
    # ── logging ────────────────────────────────────────────────────────────────
    "logging.log_interval": (int, "must be positive"),
    "logging.save_interval": (int, "must be positive"),
    "logging.eval_interval": (int, "must be positive"),
    "logging.use_wandb": (bool, None),
    # ── metrics ────────────────────────────────────────────────────────────────
    "metrics.fid_sample_size": (int, "must be non-negative"),
    "metrics.map_iou_threshold": (float, "must be in (0.0, 1.0]"),
    "metrics.eval_sample_size": (int, "must be positive"),
    "metrics.compute_fid": (bool, None),
    "metrics.compute_is": (bool, None),
    "metrics.compute_map": (bool, None),
    "metrics.num_fid_splits": (int, "must be positive"),
    # ── augmentation ───────────────────────────────────────────────────────────
    "augmentation.use_mosaic": (bool, None),
    "augmentation.use_mixup": (bool, None),
    "augmentation.mosaic_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.mixup_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.mixup_alpha": (float, "must be positive"),
    "augmentation.random_erasing_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.random_erasing_fill": (str, None),
    "augmentation.random_erasing_max": (int, "must be positive"),
    "augmentation.hflip_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.color_jitter_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.color_jitter_brightness": (float, "must be non-negative"),
    "augmentation.color_jitter_contrast": (float, "must be non-negative"),
    "augmentation.color_jitter_saturation": (float, "must be non-negative"),
    "augmentation.color_jitter_hue": (float, "must be non-negative"),
    "augmentation.affine_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.affine_degrees": (float, "must be non-negative"),
    "augmentation.affine_translate": (float, "must be in [0.0, 1.0)"),
    "augmentation.cutout_prob": (float, "must be in [0.0, 1.0]"),
    "augmentation.cutout_max_holes": (int, "must be positive"),
    "augmentation.cutout_max_size": (float, "must be in (0.0, 1.0)"),
    # ── profiling ────────────────────────────────────────────────────────────
    "profiling.profile_memory": (bool, None),
    "profiling.log_gradients": (bool, None),
    "profiling.detect_anomalies": (bool, None),
    "profiling.log_spectral_norms": (bool, None),
    "profiling.log_parameter_stats": (bool, None),
    "profiling.profile_frequency": (int, "must be positive"),
    "profiling.track_ema_ratio": (bool, None),
    # ── paths ──────────────────────────────────────────────────────────────────
    "paths.data_root": (str, None),
    "paths.checkpoint_dir": (str, None),
    "paths.log_dir": (str, None),
}

# Constraint predicates (return True if valid)
_CONSTRAINT_CHECKS: Dict[str, Any] = {
    "must be positive": lambda v: v > 0,
    "must be non-negative": lambda v: v >= 0,
    "must be in [0.0, 1.0]": lambda v: 0.0 <= v <= 1.0,
    "must be in [0.0, 1.0)": lambda v: 0.0 <= v < 1.0,
    "must be in (0.0, 1.0]": lambda v: 0.0 < v <= 1.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dotted key path."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _get_nested(d: Dict[str, Any], key: str) -> Any:
    """Get a value from a nested dictionary using a dotted key path."""
    parts = key.split(".")
    for part in parts:
        d = d[part]
    return d


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Recursively flatten a nested dictionary into dotted keys."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        dotted = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, prefix=dotted))
        else:
            flat[dotted] = v
    return flat


def _unflatten_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a flat dotted-key dict back into a nested dict."""
    nested: Dict[str, Any] = {}
    for key, value in flat.items():
        _set_nested(nested, key, value)
    return nested


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load and return a YAML file as a dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Config class
# ──────────────────────────────────────────────────────────────────────────────


class Config:
    """ILGAN configuration with attribute + dict access, validation, and
    path verification.

    Usage::

        cfg = Config()                          # loads defaults only
        cfg = Config(user_config="my_cfg.yaml") # overrides defaults
        cfg = Config(overrides={"training.epochs": 100})  # programmatic overrides
    """

    def __init__(
        self,
        user_config: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        # 1. Load defaults
        self._default_path: str = _DEFAULT_CONFIG_PATH
        if not os.path.isfile(self._default_path):
            raise FileNotFoundError(
                f"Default config not found at {self._default_path}"
            )

        raw: Dict[str, Any] = _load_yaml(self._default_path)

        # 2. Override with user-provided config file
        if user_config is not None:
            if not os.path.isfile(user_config):
                raise FileNotFoundError(
                    f"User config file not found: {user_config}"
                )
            user_raw: Dict[str, Any] = _load_yaml(user_config)
            self._deep_merge(raw, user_raw)

        # 3. Override with programmatic overrides
        if overrides is not None:
            for key, value in overrides.items():
                _set_nested(raw, key, value)

        # 4. Store the nested dict (validated later)
        self._data: Dict[str, Any] = raw

        # 5. Convert to flat representation and type-validate
        self._flat: Dict[str, Any] = _flatten_dict(self._data)
        self._validate_types()

        # 6. Verify critical paths exist
        self._validate_paths()

    # ── public helpers ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the nested configuration dictionary."""
        return copy.deepcopy(self._data)

    def flatten(self) -> Dict[str, Any]:
        """Return a flat representation with dotted keys."""
        return dict(self._flat)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Construct a Config directly from a dictionary (skips file I/O)."""
        cfg = cls.__new__(cls)
        cfg._default_path = _DEFAULT_CONFIG_PATH
        cfg._data = d
        cfg._flat = _flatten_dict(d)
        cfg._validate_types()
        cfg._validate_paths()
        return cfg

    # ── private validation ─────────────────────────────────────────────────

    def _validate_types(self) -> None:
        """Check every known key has the correct type and satisfies constraints."""
        for dotted_key, (expected_type, constraint_msg) in _TYPE_SCHEMA.items():
            value = self._flat.get(dotted_key)
            if value is None:
                # Allow None for optional keys (e.g., total_steps can be null)
                # Skip validation for keys that are not present
                continue

            # Type check (allow int → float promotion for convenience)
            if expected_type is float and isinstance(value, numbers.Integral):
                value = float(value)
                _set_nested(self._data, dotted_key, value)
                self._flat[dotted_key] = value
            elif not isinstance(value, expected_type):
                raise TypeError(
                    f"Config key '{dotted_key}' expected {expected_type.__name__}, "
                    f"got {type(value).__name__} (value={value!r})."
                )

            # Constraint check
            if constraint_msg is not None:
                check_fn = _CONSTRAINT_CHECKS.get(constraint_msg)
                if check_fn is not None and not check_fn(value):
                    raise ValueError(
                        f"Config key '{dotted_key}' = {value} {constraint_msg}."
                    )

    def _validate_paths(self) -> None:
        """Ensure critical directory paths exist (create if missing for
        output directories)."""
        # Input path (data_root) must exist
        data_root = self._flat.get("paths.data_root", "")
        if data_root:
            expanded = os.path.expanduser(data_root)
            if not os.path.exists(expanded):
                raise FileNotFoundError(
                    f"Data root path does not exist: {expanded}. "
                    "Please ensure the dataset directory exists."
                )

        # Output paths — create if missing
        for key in ("checkpoint_dir", "log_dir"):
            path_val = self._flat.get(f"paths.{key}", "")
            if path_val:
                expanded = os.path.expanduser(path_val)
                os.makedirs(expanded, exist_ok=True)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge *override* into *base* (in-place)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    # ── attribute access ───────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Support ``cfg.section.key`` style access via proxying."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            # Return a section proxy
            return _SectionProxy(self._data[name], prefix=name, root=self)
        raise AttributeError(
            f"Config has no attribute '{name}'. "
            f"Available top-level sections: {list(self._data.keys())}"
        )

    def __getitem__(self, key: str) -> Any:
        """Support ``cfg["section.key"]`` access."""
        if "." in key:
            return _get_nested(self._data, key)
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Support ``cfg["section.key"] = value``."""
        _set_nested(self._data, key, value)
        # Recompute flat representation
        self._flat = _flatten_dict(self._data)

    def __contains__(self, key: str) -> bool:
        """Support ``"key" in cfg``."""
        try:
            _get_nested(self._data, key)
            return True
        except (KeyError, TypeError):
            return False

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def __str__(self) -> str:
        return yaml.dump(self._data, default_flow_style=False).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Section proxy
# ──────────────────────────────────────────────────────────────────────────────


class _SectionProxy:
    """Provides ``cfg.data.image_size`` style access by proxying into a
    sub-dict of the config."""

    def __init__(self, data: Any, prefix: str, root: Config) -> None:
        self._data = data
        self._prefix = prefix
        self._root = root

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if isinstance(self._data, dict) and name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return _SectionProxy(value, prefix=f"{self._prefix}.{name}", root=self._root)
            return value
        raise AttributeError(
            f"Config '{self._prefix}' has no attribute '{name}'. "
            f"Available keys: {list(self._data.keys()) if isinstance(self._data, dict) else 'N/A'}"
        )

    def __getitem__(self, key: str) -> Any:
        if isinstance(self._data, dict):
            return self._data[key]
        raise TypeError("Section is not a dictionary.")

    def __repr__(self) -> str:
        return f"ConfigSection({self._prefix}): {self._data!r}"
