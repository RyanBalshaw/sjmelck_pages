r"""Tests for the marimo to Hugo post pipeline.

Equations are not rewritten: authors write ``\(...\)`` and ``$$...$$``
themselves, as AGENTS.md has always asked. The converter only checks that
``math: true`` is set when a post contains equations.

``\(`` in a raw string below is two characters, a backslash and ``(``.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sjmelck_pages.marimo_cli.convert import (
    ERROR,
    check_post,
    convert,
    split_front_matter,
)
from sjmelck_pages.marimo_cli.mdfix import has_maths, strip_hidden_cells
from sjmelck_pages.marimo_cli.repo import SAST, RepoError, slugify
from sjmelck_pages.marimo_cli.scaffold import create_post, render_notebook

NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=SAST)

MARIMO_BLOCK = "---\ntitle: Demo\nmarimo-version: 0.24.2\n---\n"
FRONT = (
    "---\n"
    'title: "Demo post"\n'
    "publishdate: 2026-09-18T10:00:00+02:00\n"
    "author: Johann Bouwer\n"
    "draft: false\n"
    "math: true\n"
    "---"
)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def test_front_matter_is_promoted_over_marimos_own() -> None:
    front, body = split_front_matter(MARIMO_BLOCK + "\n" + FRONT + "\n\ntext\n")
    assert front is not None
    assert "Demo post" in front
    assert "marimo-version" not in front
    assert body.strip() == "text"


def test_a_horizontal_rule_in_prose_is_not_front_matter() -> None:
    front, body = split_front_matter(MARIMO_BLOCK + "\nprose\n\n---\n\nmore\n")
    assert front is None
    assert "prose" in body


def test_a_notebook_without_front_matter_is_refused(tmp_path: Path) -> None:
    """A generated placeholder would land in the post, where edits are lost."""
    with pytest.raises(RepoError, match="front matter"):
        convert(
            MARIMO_BLOCK + "\nprose\n",
            slug="demo",
            folder=tmp_path,
            now=NOW,
            source="demo.py",
        )


def test_future_publishdate_is_an_error(tmp_path: Path) -> None:
    """Hugo drops a future-dated post from the build without saying anything."""
    front = FRONT.replace(
        "2026-09-18T10:00:00+02:00", (NOW + timedelta(days=1)).isoformat()
    )
    notes = check_post(
        front,
        "",
        slug="demo",
        folder=tmp_path,
        referenced=[],
        now=NOW,
    )
    assert any(n.level == ERROR and "future" in n.message for n in notes)


def test_equations_without_math_true_is_an_error(tmp_path: Path) -> None:
    """Without `math: true` MathJax never loads and the equation renders raw."""
    front = FRONT.replace("math: true", "math: false")
    notes = check_post(
        front,
        r"Some prose with \\(x^2\\) in it.",
        slug="demo",
        folder=tmp_path,
        referenced=[],
        now=NOW,
    )
    assert any(n.level == ERROR and "math: true" in n.message for n in notes)


def test_maths_delimiters_inside_a_code_fence_do_not_count(tmp_path: Path) -> None:
    front = FRONT.replace("math: true", "math: false")
    notes = check_post(
        front,
        '```python\nax.set(label="$$x$$")\n```\n',
        slug="demo",
        folder=tmp_path,
        referenced=[],
        now=NOW,
    )
    assert not [n for n in notes if "math: true" in n.message]


def test_has_maths_detects_the_documented_delimiters() -> None:
    assert has_maths(r"inline \\(x^2\\) here")
    assert has_maths("$$\nE = mc^2\n$$")
    assert has_maths(r"\[ x \]")
    # Single dollars are no longer special-cased.
    assert not has_maths("costs $5 and $10")


def test_hidden_cells_are_dropped() -> None:
    text = "```python\n# sjmelck: hide\nimport marimo as mo\n```\n\n```python\nkeep = 1\n```\n"
    result, removed = strip_hidden_cells(text)
    assert removed == 1
    assert "import marimo" not in result
    assert "keep = 1" in result


def test_missing_figure_is_an_error(tmp_path: Path) -> None:
    """Export drops cell outputs, so a figure only exists if it was saved."""
    notes = check_post(
        FRONT,
        "",
        slug="demo",
        folder=tmp_path,
        referenced=["gone.png"],
        now=NOW,
    )
    assert any(n.level == ERROR and "gone.png" in n.message for n in notes)


def test_convert_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "fig.png").write_bytes(b"")
    exported = (
        MARIMO_BLOCK
        + "\n"
        + FRONT
        + "\n\nProse with "
        + r"\\(x^2\\)"
        + " maths.\n\n"
        + "```python {.marimo}\n# sjmelck: hide\nimport marimo as mo\n```\n\n"
        + "```python {.marimo}\nkeep = 1\n```\n\n"
        + "![A figure](../assets/images/demo/fig.png)\n"
    )
    result = convert(exported, slug="demo", folder=tmp_path, now=NOW, source="demo.py")

    assert result.hidden_cells == 1
    assert result.referenced_images == ["fig.png"]
    assert "marimo-version" not in result.text
    assert "import marimo" not in result.text  # hidden cell dropped
    assert "```python\nkeep = 1" in result.text  # {.marimo} stripped
    assert r"\\(x^2\\)" in result.text  # equations pass through untouched
    assert "![A figure](fig.png)" in result.text  # path shortened
    assert result.text.endswith(")\n")  # exactly one trailing newline
    assert not result.errors


# --------------------------------------------------------------------------
# The scaffold
# --------------------------------------------------------------------------


def test_scaffolded_notebook_is_valid_python_and_complete() -> None:
    source = render_notebook(
        slug="demo",
        title="Demo post",
        author="Johann Bouwer",
        publishdate="2026-09-18T10:00:00+02:00",
    )
    ast.parse(source)
    assert 'title: "Demo post"' in source
    for field in ("draft:", "toc:", "math:", "hasMermaid:", "OverviewFig:", "build:"):
        assert field in source
    assert source.count("# sjmelck: hide") == 2  # plumbing stays out of the post
    assert "fig.savefig(" in source  # export drops outputs


def test_create_post_writes_the_notebook_beside_its_figures(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("title: Sjmelck\n", encoding="utf-8")
    (tmp_path / "content" / "blog").mkdir(parents=True)

    slug, notebook = create_post(tmp_path, "Demo post", author="x", now=NOW)

    assert slug == slugify("Demo post") == "demo-post"
    assert notebook == tmp_path / "assets" / "images" / "demo-post" / "demo-post.py"
    assert notebook.is_file()
