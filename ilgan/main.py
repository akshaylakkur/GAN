#!/usr/bin/env python3
"""ILGAN — One-shot Image and Bounding Box Generation via GANs.

This module serves as the top-level entry point for the ILGAN package.
It re-exports the Click-based CLI group from :mod:`ilgan.scripts.cli`
so that a ``console_scripts`` entry point (defined in ``setup.py`` or
``pyproject.toml``) can point to ``ilgan.main:cli``.

Usage
-----
Once installed (``pip install -e .``), the ``ilgan`` command is available::

    ilgan --help
    ilgan train --help
    ilgan evaluate --help
    ilgan generate --help
    ilgan list-devices
    ilgan analyze-losses --data-root ./data
    ilgan profile-memory
    ilgan compute-statistics --data-root ./data

Without installation, the CLI can be invoked via Python::

    python -m ilgan train
    python -m ilgan evaluate --checkpoint model.pt
    python -m ilgan generate --checkpoint model.pt --num-samples 64

Or equivalently::

    python -c "from ilgan.main import cli; cli()"
"""

from __future__ import annotations

import sys
from typing import NoReturn

from ilgan.scripts.cli import cli


def main() -> NoReturn:
    """Execute the ILGAN CLI.

    This is the primary entry point for the ``ilgan`` command-line interface.
    It delegates to the :func:`cli` Click group, which parses command-line
    arguments and dispatches to the appropriate subcommand (train, evaluate,
    generate, list-devices, analyze-losses, profile-memory,
    compute-statistics).

    This function is designed to be used as a ``console_scripts`` entry point
    in ``setup.py`` / ``pyproject.toml``::

        [project.scripts]
        ilgan = "ilgan.main:main"

    It can also be called programmatically::

        from ilgan.main import main
        main()

    The function calls :func:`cli()` and exits with the appropriate system
    exit code.  It does **not** return.

    Raises
    ------
    SystemExit
        Always exits with the return code from the Click CLI.
    """
    sys.exit(cli())


# Re-export the Click group so it can be imported directly:
#   from ilgan.main import cli
#   from ilgan.main import main
__all__ = ["cli", "main"]
