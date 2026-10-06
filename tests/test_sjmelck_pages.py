r"""Tests for the marimo to Hugo post pipeline.

The focus is the inline maths converter, because that is the part that can
silently corrupt a post. The theme enables Goldmark's passthrough extension
for ``\(...\)``, ``\[...\]`` and ``$$...$$`` but not for single dollar
``$...$``, so Goldmark eats backslash escapes before MathJax sees them.
Simple spans survive, which is what makes it dangerous: it fails only on
LaTeX containing escapes, and it fails after a green build.

``\\(`` in a raw string below is two characters, a backslash and ``(``.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sjmelck_pages.convert import ERROR, check_post, convert, split_front_matter
from sjmelck_pages.mdfix import convert_inline_math, iter_code_fences
from sjmelck_pages.repo import SAST, RepoError, slugify
from sjmelck_pages.scaffold import create_post, render_notebook

NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=SAST)
BLOG = Path(__file__).resolve().parents[1] / "content" / "blog"
POSTS = sorted(BLOG.glob("*.md"))

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
# The maths converter
# --------------------------------------------------------------------------

CASES = [
    pytest.param(r"$x^2$", r"\\(x^2\\)", id="simple"),
    pytest.param(r"$\{x\} \| y \|$", r"\\(\{x\} \| y \|\\)", id="escapes-survive"),
    pytest.param(r"$1 - \sigma(u)$", r"\\(1 - \sigma(u)\\)", id="starts-with-digit"),
    pytest.param("costs $5 and $10", "costs $5 and $10", id="currency"),
    pytest.param(r'`price = "$5"`', r'`price = "$5"`', id="code-span"),
    pytest.param(r"\$5", r"\$5", id="escaped-dollar"),
    pytest.param(r"\\(already\\)", r"\\(already\\)", id="already-converted"),
    pytest.param("$$E=mc^2$$", "$$E=mc^2$$", id="display-maths"),
]


@pytest.mark.parametrize("source, expected", CASES)
def test_inline_maths_conversion(source: str, expected: str) -> None:
    assert convert_inline_math(source).text == expected


@pytest.mark.parametrize("source, _expected", CASES)
def test_conversion_is_idempotent(source: str, _expected: str) -> None:
    once = convert_inline_math(source).text
    assert convert_inline_math(once).text == once


def test_code_fences_are_untouched() -> None:
    """A regex would rewrite the matplotlib label and break the example."""
    text = '```python {.marimo}\nax.set(label="$z = b_0$")\n```\n'
    assert convert_inline_math(text).text == text


def test_display_block_is_untouched_but_prose_after_it_converts() -> None:
    text = "$$\nf(x) = \\sum_{i=1}^{n} w_i\n$$\n\nafter $x$\n"
    result = convert_inline_math(text)
    assert "$$\nf(x) = \\sum_{i=1}^{n} w_i\n$$" in result.text
    assert result.text.endswith("after \\\\(x\\\\)\n")


def test_raw_html_is_untouched() -> None:
    """Goldmark does not process escapes inside an HTML block, so $ is already
    safe there. Rewriting it would break an embedded chart's axis config."""
    text = (
        '<script>Plotly.newPlot("c",[{"y":[1,2]}],'
        '{"yaxis":{"tickprefix":"$","ticksuffix":"$"}});</script>\n'
    )
    assert convert_inline_math(text).text == text


# --------------------------------------------------------------------------
# The published posts, used as a regression corpus
# --------------------------------------------------------------------------


def _fences(text: str) -> list[str]:
    lines = text.splitlines()
    return ["\n".join(lines[s : e + 1]) for s, e, _ in iter_code_fences(text)]


@pytest.mark.parametrize("path", POSTS, ids=lambda p: p.name)
def test_existing_posts_survive_conversion(path: Path) -> None:
    """Every code fence and every $$ region must come out byte-identical.

    pretty-plotting-in-python.md holds matplotlib label strings full of
    dollars inside code fences; logistic-regression.md has 143 inline spans
    in prose. Between them they are a genuinely adversarial corpus.
    """
    original = path.read_text(encoding="utf-8")
    converted = convert_inline_math(original).text

    assert _fences(converted) == _fences(original)

    display = re.compile(r"\$\$.*?\$\$", re.DOTALL)
    assert display.findall(converted) == display.findall(original)

    assert convert_inline_math(converted).text == converted


def test_the_corpus_actually_exercises_the_converter() -> None:
    """Guard against the test above passing because nothing was converted."""
    total = sum(
        convert_inline_math(p.read_text(encoding="utf-8")).converted for p in POSTS
    )
    assert total > 100


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
        converted_math=0,
        now=NOW,
    )
    assert any(n.level == ERROR and "future" in n.message for n in notes)


def test_missing_figure_is_an_error(tmp_path: Path) -> None:
    """Export drops cell outputs, so a figure only exists if it was saved."""
    notes = check_post(
        FRONT,
        "",
        slug="demo",
        folder=tmp_path,
        referenced=["gone.png"],
        converted_math=0,
        now=NOW,
    )
    assert any(n.level == ERROR and "gone.png" in n.message for n in notes)


def test_convert_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "fig.png").write_bytes(b"")
    exported = (
        MARIMO_BLOCK
        + "\n"
        + FRONT
        + "\n\nProse with $x^2$ maths.\n\n"
        + "```python {.marimo}\n# sjmelck: hide\nimport marimo as mo\n```\n\n"
        + "```python {.marimo}\nkeep = 1\n```\n\n"
        + "![A figure](../assets/images/demo/fig.png)\n"
    )
    result = convert(exported, slug="demo", folder=tmp_path, now=NOW, source="demo.py")

    assert result.converted_math == 1
    assert result.hidden_cells == 1
    assert result.referenced_images == ["fig.png"]
    assert "marimo-version" not in result.text
    assert "import marimo" not in result.text  # hidden cell dropped
    assert "```python\nkeep = 1" in result.text  # {.marimo} stripped
    assert r"\\(x^2\\)" in result.text
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
