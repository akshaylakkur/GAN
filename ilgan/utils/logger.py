"""
Logging utilities for ILGAN.

Provides a `Logger` class (wrapping Python's logging module) that supports
console output and rotating file handlers, plus a `ProgressLogger` subclass
for tqdm-compatible progress bar integration.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, TextIO

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_LOG_FORMAT = (
    "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB per log file
_BACKUP_COUNT = 5


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────


class Logger:
    """ILGAN logger with dual output: console (stdout/stderr) and a rotating
    file handler in the configured log directory.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    log_dir : str, optional
        Directory for log files. If None, file logging is disabled.
    level : int or str
        Logging threshold. Default ``logging.INFO``.
    console_stream : TextIO, optional
        Stream for console output (default: ``sys.stdout``).
    """

    def __init__(
        self,
        name: str = "ilgan",
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        console_stream: Optional[TextIO] = None,
    ) -> None:
        self._name = name
        self._log_dir = log_dir

        # Resolve level
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        # Create Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()  # avoid duplicate handlers on re-init
        self._logger.propagate = False

        # ── Console handler ──────────────────────────────────────────────
        ch = logging.StreamHandler(stream=console_stream or sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(self._build_formatter())
        self._logger.addHandler(ch)

        # ── File handler (rotating) ──────────────────────────────────────
        self._file_handler: Optional[RotatingFileHandler] = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{name}.log")
            fh = RotatingFileHandler(
                log_path,
                maxBytes=_MAX_LOG_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(self._build_formatter())
            self._logger.addHandler(fh)
            self._file_handler = fh

    # ── public logging methods ───────────────────────────────────────────

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an INFO-level message."""
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a DEBUG-level message."""
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a WARNING-level message."""
        self._logger.warning(msg, *args, **kwargs)

    def warn(self, msg: str, *args, **kwargs) -> None:
        """Alias for ``warning``."""
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an ERROR-level message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a CRITICAL-level message."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log an exception traceback at ERROR level."""
        self._logger.exception(msg, *args, **kwargs)

    # ── context / lifecycle ──────────────────────────────────────────────

    @property
    def level(self) -> int:
        """Current logging level."""
        return self._logger.level

    @level.setter
    def level(self, level: int) -> None:
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)

    @property
    def log_file_path(self) -> Optional[str]:
        """Path to the active log file, if file logging is enabled."""
        if self._file_handler is not None:
            return self._file_handler.baseFilename
        return None

    def close(self) -> None:
        """Close all handlers (clean up file handles)."""
        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_formatter() -> logging.Formatter:
        return logging.Formatter(
            fmt=_DEFAULT_LOG_FORMAT,
            datefmt=_DEFAULT_DATE_FORMAT,
        )


# ──────────────────────────────────────────────────────────────────────────────
# ProgressLogger (tqdm wrapper)
# ──────────────────────────────────────────────────────────────────────────────


class ProgressLogger(Logger):
    """Logger that wraps tqdm progress bars and integrates with console/file
    logging.

    Usage::

        plog = ProgressLogger(log_dir="./logs")
        for epoch in plog.progress(range(epochs), desc="Training"):
            ...
    """

    def __init__(
        self,
        name: str = "ilgan",
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        **tqdm_kwargs,
    ) -> None:
        super().__init__(name=name, log_dir=log_dir, level=level)
        self._tqdm_kwargs = tqdm_kwargs

    def progress(self, iterable, desc: str = "", total: Optional[int] = None, **kwargs):
        """Wrap an iterable with a tqdm progress bar.

        Parameters
        ----------
        iterable : iterable
            The iterable to wrap.
        desc : str
            Description shown in the progress bar.
        total : int, optional
            Total number of items (inferred from iterable if possible).
        **kwargs
            Additional tqdm keyword arguments.

        Yields
        ------
        items from *iterable* with a progress bar.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            self.warning("tqdm not installed; falling back to simple iteration.")
            yield from iterable
            return

        merged_kw = {**self._tqdm_kwargs, **kwargs}
        merged_kw.setdefault("desc", desc)
        if total is not None:
            merged_kw["total"] = total

        # Use tqdm.write for logging so progress bar stays clean
        original_info = self.info
        original_debug = self.debug
        original_warning = self.warning
        original_error = self.error

        def _tqdm_info(msg, *a, **kw):
            tqdm.write(msg)

        self.info = _tqdm_info  # type: ignore[assignment]
        self.debug = _tqdm_debug  # type: ignore[assignment]
        self.warning = _tqdm_warning  # type: ignore[assignment]
        self.error = _tqdm_error  # type: ignore[assignment]

        try:
            yield from tqdm(iterable, **merged_kw)
        finally:
            self.info = original_info
            self.debug = original_debug
            self.warning = original_warning
            self.error = original_error


def _tqdm_debug(msg, *a, **kw):
    from tqdm import tqdm as _tqdm
    _tqdm.write(msg)


def _tqdm_warning(msg, *a, **kw):
    from tqdm import tqdm as _tqdm
    _tqdm.write(msg)


def _tqdm_error(msg, *a, **kw):
    from tqdm import tqdm as _tqdm
    _tqdm.write(msg)