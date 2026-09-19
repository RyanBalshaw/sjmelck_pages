"""Tooling for the Sjmelck blog.

Write a post as a marimo notebook, then convert it into a Hugo post:

    sjmelck-pages new "An introduction to Kalman filters"
    sjmelck-pages convert assets/images/<slug>/<slug>.py
"""

from sjmelck_pages.cli import main

__all__ = ["main"]
