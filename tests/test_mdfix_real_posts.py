"""Run the maths converter over every real post and assert it changes nothing
it should not.

These posts are a genuinely adversarial corpus that we already own:
``logistic-regression.md`` has well over a hundred inline spans, and
``pretty-plotting-in-python.md`` contains matplotlib label strings such as
``"$\\mathdefault{-}$"`` *inside code fences*, which a naive regex would
happily corrupt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sjmelck_pages.mdfix import convert_inline_math, iter_code_fences

BLOG = Path(__file__).resolve().parents[1] / "content" / "blog"
POSTS = sorted(BLOG.glob("*.md"))


def _fence_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    return [
        "\n".join(lines[start : end + 1]) for start, end, _ in iter_code_fences(text)
    ]


def _display_blocks(text: str) -> list[str]:
    return re.findall(r"\$\$.*?\$\$", text, flags=re.DOTALL)


def test_there_are_posts_to_check() -> None:
    assert POSTS, f"no posts found in {BLOG}"


@pytest.mark.parametrize("path", POSTS, ids=lambda p: p.name)
def test_code_fences_are_byte_identical(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    converted = convert_inline_math(original).text
    assert _fence_blocks(converted) == _fence_blocks(original)


@pytest.mark.parametrize("path", POSTS, ids=lambda p: p.name)
def test_display_maths_is_byte_identical(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    converted = convert_inline_math(original).text
    assert _display_blocks(converted) == _display_blocks(original)


@pytest.mark.parametrize("path", POSTS, ids=lambda p: p.name)
def test_conversion_is_idempotent(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    once = convert_inline_math(original).text
    assert convert_inline_math(once).text == once


def test_the_corpus_actually_exercises_the_converter() -> None:
    """Guard against the suite passing because nothing was converted."""
    total = sum(
        convert_inline_math(path.read_text(encoding="utf-8")).converted
        for path in POSTS
    )
    assert total > 100
