"""
Device detection and management for ILGAN.

Provides a single source of truth for device selection across the entire
codebase.  Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU
fallback.

Priority order:
1. CUDA (NVIDIA GPU) — fastest, best supported
2. MPS (Apple Silicon / Metal) — good for development and testing
3. CPU — universal fallback

Usage
-----
::

    from ilgan.utils.device import get_device, get_device_name, DEVICE

    device = get_device()           # auto-detect best available
    device = get_device(prefer="mps")  # force MPS if available
    print(get_device_name())        # "cuda:0", "mps", or "cpu"
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_CUDA_AVAILABLE: bool = torch.cuda.is_available()
"""Whether CUDA-capable GPUs are available."""

_MPS_AVAILABLE: bool = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
    and torch.backends.mps.is_built()
)
"""Whether MPS (Apple Silicon / Metal) is available."""

_DEVICE_OVERRIDE: Optional[str] = os.environ.get("ILGAN_DEVICE", None)
"""Optional override via ``ILGAN_DEVICE`` environment variable.
Set to ``"cuda"``, ``"mps"``, or ``"cpu"`` to force a specific device."""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available device for ILGAN.

    Priority: CUDA > MPS > CPU, unless overridden by the ``ILGAN_DEVICE``
    environment variable or the *prefer* parameter.

    Parameters
    ----------
    prefer : str, optional
        If set to ``"cuda"``, ``"mps"``, or ``"cpu"``, prefer that device
        if available.  Falls back to the next best if unavailable.

    Returns
    -------
    torch.device
        The selected device.

    Examples
    --------
    >>> from ilgan.utils.device import get_device
    >>> device = get_device()           # auto-detect
    >>> device = get_device(prefer="mps")  # prefer MPS
    """
    # 1. Environment variable override takes highest priority
    if _DEVICE_OVERRIDE is not None:
        override = _DEVICE_OVERRIDE.lower().strip()
        if override == "cuda" and _CUDA_AVAILABLE:
            return torch.device("cuda:0")
        elif override == "mps" and _MPS_AVAILABLE:
            return torch.device("mps")
        elif override == "cpu":
            return torch.device("cpu")
        # If override is set but unavailable, fall through to auto-detect

    # 2. Preference parameter
    if prefer is not None:
        p = prefer.lower().strip()
        if p == "cuda" and _CUDA_AVAILABLE:
            return torch.device("cuda:0")
        elif p == "mps" and _MPS_AVAILABLE:
            return torch.device("mps")
        elif p == "cpu":
            return torch.device("cpu")
        # If preference is set but unavailable, fall through

    # 3. Auto-detect: CUDA > MPS > CPU
    if _CUDA_AVAILABLE:
        return torch.device("cuda:0")
    elif _MPS_AVAILABLE:
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """Return a human-readable name for the current device.

    Returns
    -------
    str
        ``"cuda:0"``, ``"mps"``, or ``"cpu"``.
    """
    return str(get_device())


def get_device_info() -> dict:
    """Return detailed information about the current device.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``"device"``: the selected device string.
        - ``"cuda_available"``: whether CUDA is available.
        - ``"cuda_device_count"``: number of CUDA devices (0 if none).
        - ``"cuda_device_name"``: name of the first CUDA device (or ``None``).
        - ``"mps_available"``: whether MPS is available.
        - ``"mps_device_name"``: ``"Apple Silicon"`` if MPS available.
        - ``"cpu_count"``: number of CPU cores.
    """
    info: dict = {
        "device": get_device_name(),
        "cuda_available": _CUDA_AVAILABLE,
        "cuda_device_count": torch.cuda.device_count() if _CUDA_AVAILABLE else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None,
        "mps_available": _MPS_AVAILABLE,
        "mps_device_name": "Apple Silicon" if _MPS_AVAILABLE else None,
        "cpu_count": os.cpu_count(),
    }
    return info


def supports_amp(device: Optional[torch.device] = None) -> bool:
    """Check whether AMP (Automatic Mixed Precision) is supported on the
    given device.

    Parameters
    ----------
    device : torch.device, optional
        Device to check.  If ``None``, uses the current device from
        :func:`get_device`.

    Returns
    -------
    bool
        ``True`` if AMP is supported on this device.
    """
    if device is None:
        device = get_device()
    dtype = device.type
    # CUDA supports AMP via torch.amp.autocast("cuda")
    # MPS supports AMP via torch.amp.autocast("mps") (PyTorch >= 2.0)
    # CPU does not support AMP
    return dtype in ("cuda", "mps")


def get_amp_device_type(device: Optional[torch.device] = None) -> str:
    """Return the device type string for ``torch.amp.autocast()``.

    Parameters
    ----------
    device : torch.device, optional
        Device to use.  If ``None``, uses the current device.

    Returns
    -------
    str
        ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if device is None:
        device = get_device()
    return device.type


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton for convenience
# ──────────────────────────────────────────────────────────────────────────────

DEVICE: torch.device = get_device()
"""Module-level device singleton.  Import this for convenience:

>>> from ilgan.utils.device import DEVICE
>>> model.to(DEVICE)
"""

__all__ = [
    "get_device",
    "get_device_name",
    "get_device_info",
    "supports_amp",
    "get_amp_device_type",
    "DEVICE",
    "_CUDA_AVAILABLE",
    "_MPS_AVAILABLE",
]
