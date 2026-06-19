#!/usr/bin/env python3
"""Minimal setup.py for ILGAN — fallback for pip < 21.3 or editable installs.

This file exists because Lambda Cloud's Ubuntu 20.04 ships pip 20.x which
doesn't fully support PEP 660 (editable installs via pyproject.toml).

If you see:
    Cannot import 'setuptools.backends._legacy'

It means pip is reading pyproject.toml but the build-backend is wrong.
Run this to fix it:
    sed -i 's/setuptools.backends._legacy/setuptools.build_meta/' pyproject.toml
    pip install -e .

Or use the fallback below.
"""

from setuptools import setup

setup()