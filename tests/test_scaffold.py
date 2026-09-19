"""Tests for the notebook scaffold."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import pytest

from sjmelck_pages.repo import SAST, RepoError, slugify
from sjmelck_pages.scaffold import create_post, render_notebook

NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=SAST)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text("title: Sjmelck\n", encoding="utf-8")
    (tmp_path / "content" / "blog").mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize(
    "title, expected",
    [
        ("An introduction to Kalman filters", "an-introduction-to-kalman-filters"),
        ("Pretty plotting in Python!", "pretty-plotting-in-python"),
        ("  Spaced   out  ", "spaced-out"),
        ("Already-hyphenated", "already-hyphenated"),
        ("Unicode: naïve café", "unicode-naive-cafe"),
    ],
)
def test_slugify(title: str, expected: str) -> None:
    assert slugify(title) == expected


def test_slugify_rejects_empty_input() -> None:
    with pytest.raises(RepoError):
        slugify("!!!")


def test_rendered_notebook_is_valid_python() -> None:
    source = render_notebook(
        slug="demo",
        title="Demo post",
        author="Johann Bouwer",
        publishdate="2026-09-18T10:00:00+02:00",
    )
    ast.parse(source)


def test_rendered_notebook_has_the_expected_parts() -> None:
    source = render_notebook(
        slug="demo",
        title="Demo post",
        author="Johann Bouwer",
        publishdate="2026-09-18T10:00:00+02:00",
    )
    assert source.startswith("import marimo\n")
    assert source.rstrip().endswith("app.run()")
    assert 'title: "Demo post"' in source
    assert "author: Johann Bouwer" in source
    assert "publishdate: 2026-09-18T10:00:00+02:00" in source
    # Front matter must carry every field the theme needs.
    for field in ("draft:", "toc:", "math:", "hasMermaid:", "OverviewFig:", "build:"):
        assert field in source
    # Two plumbing cells must be hidden, or the post opens with imports.
    assert source.count("# sjmelck: hide") == 2
    # The figure must be saved, because export drops cell outputs.
    assert "fig.savefig(" in source
    assert "dpi=150" in source
    # Inline maths is taught with single dollars.
    assert "$x^2$" in source


def test_create_post_writes_notebook_and_folder(fake_repo: Path) -> None:
    slug, notebook = create_post(
        fake_repo, "Demo post", author="Johann Bouwer", now=NOW
    )
    assert slug == "demo-post"
    assert notebook == fake_repo / "assets" / "images" / "demo-post" / "demo-post.py"
    assert notebook.is_file()
    assert notebook.parent.is_dir()
    assert "Demo post" in notebook.read_text(encoding="utf-8")


def test_create_post_honours_an_explicit_slug(fake_repo: Path) -> None:
    slug, notebook = create_post(
        fake_repo, "Demo post", author="x", slug="custom", now=NOW
    )
    assert slug == "custom"
    assert notebook.name == "custom.py"


def test_create_post_refuses_to_clobber(fake_repo: Path) -> None:
    create_post(fake_repo, "Demo post", author="x", now=NOW)
    with pytest.raises(RepoError, match="already exists"):
        create_post(fake_repo, "Demo post", author="x", now=NOW)


def test_create_post_force_overwrites(fake_repo: Path) -> None:
    _slug, notebook = create_post(fake_repo, "Demo post", author="x", now=NOW)
    notebook.write_text("scribbled over", encoding="utf-8")
    create_post(fake_repo, "Demo post", author="x", force=True, now=NOW)
    assert "import marimo" in notebook.read_text(encoding="utf-8")


def test_notebook_is_written_with_unix_newlines(fake_repo: Path) -> None:
    _slug, notebook = create_post(fake_repo, "Demo post", author="x", now=NOW)
    assert b"\r\n" not in notebook.read_bytes()
