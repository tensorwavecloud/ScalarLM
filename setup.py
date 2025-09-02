#!/usr/bin/env python
"""
Minimal setup.py for compatibility with older pip versions.
This reads configuration from pyproject.toml.
"""

from setuptools import setup

# For compatibility with older pip versions that don't support 
# editable installs with pyproject.toml only
setup()