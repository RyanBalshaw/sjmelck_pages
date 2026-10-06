"""Write Sjmelck posts as marimo notebooks and convert them into Hugo posts.

    sjmelck-pages new "An introduction to Kalman filters"
    sjmelck-pages convert assets/images/<slug>/<slug>.py

The notebook lives at ``assets/images/<slug>/<slug>.py``, beside the figures
it generates, and is the source of truth for ``content/blog/<slug>.md``.
"""

from sjmelck_pages.marimo_cli.cli import build_parser, main
from sjmelck_pages.marimo_cli.convert import ConversionResult, Note, convert_notebook
from sjmelck_pages.marimo_cli.repo import RepoError
from sjmelck_pages.marimo_cli.scaffold import create_post

__all__ = [
    "ConversionResult",
    "Note",
    "RepoError",
    "build_parser",
    "convert_notebook",
    "create_post",
    "main",
]
