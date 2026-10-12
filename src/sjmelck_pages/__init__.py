"""Tooling for the Sjmelck blog.

The marimo post workflow lives in :mod:`sjmelck_pages.marimo_cli`; ``main`` is
re-exported here because it is the ``sjmelck-pages`` entry point.
"""

from sjmelck_pages.marimo_cli import main

__all__ = ["main"]
