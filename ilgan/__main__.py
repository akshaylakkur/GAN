#!/usr/bin/env python3
"""ILGAN — ``python -m ilgan`` entry point.

This module allows the ILGAN CLI to be invoked directly via::

    python -m ilgan train
    python -m ilgan evaluate --checkpoint model.pt
    python -m ilgan generate --checkpoint model.pt --num-samples 64

It simply delegates to :func:`ilgan.main.main`.
"""

from __future__ import annotations

from ilgan.main import main

if __name__ == "__main__":
    main()
