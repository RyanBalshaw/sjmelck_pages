"""Create a starter marimo notebook for a new blog post.

The notebook is the source of truth for a post: ``sjmelck-pages convert``
turns it into ``content/blog/<slug>.md``. It lives beside the figures it
generates, at ``assets/images/<slug>/<slug>.py``, which means the notebook
can write figures to its own directory without any path juggling.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sjmelck_pages.repo import (
    RepoError,
    format_publishdate,
    image_dir,
    notebook_path,
    sast_now,
    slugify,
)

# Placeholders are substituted rather than formatted, because the template is
# full of braces that str.format would choke on.
_TEMPLATE = '''import marimo

app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    title: "__TITLE__"
    publishdate: __PUBLISHDATE__
    author: __AUTHOR__
    description: One or two plain sentences, 120 to 160 characters, that make sense without the title.
    draft: true
    toc: true
    math: true
    hasMermaid: false
    OverviewFig: "overview.png"
    tags: ["Tag one", "Tag two", "Tag three"]
    categories: ["Machine learning"]
    build:
      list: always
      publishResources: true
      render: always
    ---
    """
    )
    return


@app.cell
def _():
    # sjmelck: hide
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # Figures are written next to this notebook, which is exactly the folder
    # the Hugo theme resolves image names against.
    IMAGE_DIR = Path(__file__).resolve().parent
    return IMAGE_DIR, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction

    Write the post here. Headings start at `##`, because the title is the `h1`.

    Inline maths is written with single dollars, like $x^2$, and display maths
    with double dollars:

    $$
    f(x) = \\sum_{i=1}^{n} w_i \\phi(\\|x - c_i\\|)
    $$

    `sjmelck-pages convert` rewrites the inline spans to `\\\\(...\\\\)` before
    Hugo sees them, because Goldmark eats backslash escapes inside `$...$`.
    Write whichever you prefer; both survive.
    """
    )
    return


@app.cell
def _(IMAGE_DIR, np, plt):
    x = np.linspace(0, 10, 200)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, np.sin(x))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()

    # Exporting drops cell outputs, so every figure must be saved to disk.
    # 8 x 4.5 inches at 150 dpi is 1200 px wide, which is what the theme wants.
    fig.savefig(IMAGE_DIR / "overview.png", dpi=150, bbox_inches="tight")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Images are referenced by bare filename. The theme resolves the path,
    resizes the image, and uses the alt text as the caption.

    ![A sine wave plotted against time, used here as a placeholder figure](overview.png)
    """
    )
    return


@app.cell
def _():
    # sjmelck: hide
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
'''


def render_notebook(
    *,
    slug: str,
    title: str,
    author: str,
    publishdate: str,
) -> str:
    """Render the starter notebook for a post."""
    return (
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__PUBLISHDATE__", publishdate)
        .replace("__AUTHOR__", author)
        .replace("__SLUG__", slug)
    )


def create_post(
    root: Path,
    title: str,
    *,
    author: str,
    slug: str | None = None,
    force: bool = False,
    now: datetime | None = None,
) -> tuple[str, Path]:
    """Scaffold the notebook and its image folder.

    Returns the slug and the notebook path.
    """
    resolved_slug = slug or slugify(title)
    notebook = notebook_path(root, resolved_slug)

    if notebook.exists() and not force:
        raise RepoError(f"{notebook} already exists. Pass --force to overwrite it.")

    folder = image_dir(root, resolved_slug)
    folder.mkdir(parents=True, exist_ok=True)

    content = render_notebook(
        slug=resolved_slug,
        title=title,
        author=author,
        publishdate=format_publishdate(now or sast_now()),
    )
    notebook.write_text(content, encoding="utf-8", newline="\n")

    return resolved_slug, notebook
