"""Repository layout helpers for the Sjmelck blog.

The slug contract is central to this repo: a post lives at
``content/blog/<slug>.md`` and its figures live in ``assets/images/<slug>/``.
Marimo notebooks live beside the figures they generate, at
``assets/images/<slug>/<slug>.py``.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

# South African Standard Time. A fixed offset is used rather than zoneinfo
# because zoneinfo needs the tzdata package on Windows, and SAST has no DST.
SAST = timezone(timedelta(hours=2))

_SLUG_STRIP = re.compile(r"[^\w\s-]")
_SLUG_SPACES = re.compile(r"[-\s]+")


class RepoError(Exception):
    """Raised when the repository layout is not what we expect."""


def sast_now() -> datetime:
    """Return the current time in SAST, to the second."""
    return datetime.now(SAST).replace(microsecond=0)


def format_publishdate(moment: datetime) -> str:
    """Format a datetime as Hugo expects it in front matter."""
    return moment.isoformat()


def slugify(text: str) -> str:
    """Convert a post title into a URL slug.

    Lowercase, hyphen separated, ASCII only. This is the same shape that
    ``hugo new blog/<slug>.md`` expects.
    """
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    cleaned = _SLUG_STRIP.sub("", ascii_only).strip().lower()
    slug = _SLUG_SPACES.sub("-", cleaned).strip("-")
    if not slug:
        raise RepoError(f"Could not derive a slug from {text!r}")
    return slug


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards looking for the Sjmelck repository root.

    The root is identified by holding both ``config.yaml`` and
    ``content/blog``, which together are unique to this site.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config.yaml").is_file() and (
            candidate / "content" / "blog"
        ).is_dir():
            return candidate
    raise RepoError(
        "Could not find the Sjmelck repository root (looked for config.yaml "
        f"and content/blog above {current}). Run this from inside the repo."
    )


def image_dir(root: Path, slug: str) -> Path:
    """Directory holding a post's figures and its notebook."""
    return root / "assets" / "images" / slug


def notebook_path(root: Path, slug: str) -> Path:
    """The marimo notebook that generates a post."""
    return image_dir(root, slug) / f"{slug}.py"


def post_path(root: Path, slug: str) -> Path:
    """The generated Hugo post."""
    return root / "content" / "blog" / f"{slug}.md"


def relative_to_root(root: Path, path: Path) -> str:
    """Display a path relative to the repo root, using forward slashes."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
