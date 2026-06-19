"""
ILGAN Scripts — CLI entry point and subcommand definitions.

This package provides the main ``ilgan`` Click-based command-line interface.
The primary entry point is the :func:`cli` Click group, which exposes
subcommands for training, evaluation, generation, device inspection, loss
analysis, memory profiling, and dataset statistics.

Usage
-----
The CLI can be invoked directly via Python::

    python -m ilgan.scripts.cli train
    python -m ilgan.scripts.cli evaluate --checkpoint model.pt
    python -m ilgan.scripts.cli generate --checkpoint model.pt --num-samples 64
    python -m ilgan.scripts.cli list-devices
    python -m ilgan.scripts.cli analyze-losses --data-root ./data
    python -m ilgan.scripts.cli profile-memory
    python -m ilgan.scripts.cli compute-statistics --data-root ./data

Or via the :func:`main` convenience function::

    from ilgan.scripts import main
    main()  # equivalent to calling cli()

Or via a console_scripts entry point (if installed)::

    ilgan train
    ilgan evaluate --checkpoint model.pt
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
        ilgan = "ilgan.scripts:main"

    It can also be called programmatically::

        from ilgan.scripts import main
        main()

    The function calls :func:`cli()` and exits with the appropriate system
    exit code.  It does **not** return.

    Raises
    ------
    SystemExit
        Always exits with the return code from the Click CLI.
    """
    # Click's cli() will handle argument parsing, dispatch, and sys.exit
    sys.exit(cli())


# Re-export the Click group so it can be imported directly:
#   from ilgan.scripts import cli
#   from ilgan.scripts.cli import cli
__all__ = ["cli", "main"]
