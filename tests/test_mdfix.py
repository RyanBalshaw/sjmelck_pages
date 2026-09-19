r"""Tests for the markdown fixups, especially the inline maths converter.

The maths converter is the part that can silently corrupt a post, so the
table below is deliberately adversarial. ``\\(`` in a raw string is two
characters: a backslash and ``(``.
"""

from __future__ import annotations

import pytest

from sjmelck_pages.mdfix import (
    convert_inline_math,
    has_mermaid_fence,
    iter_code_fences,
    normalise_whitespace,
    normalize_image_links,
    strip_fence_attributes,
    strip_hidden_cells,
)

CONVERSIONS = [
    pytest.param(r"$x^2$", r"\\(x^2\\)", id="simple"),
    pytest.param(
        r"$\{x\} \| y \|$",
        r"\\(\{x\} \| y \|\\)",
        id="backslash-escapes-survive",
    ),
    pytest.param(r"$a$ and $b$", r"\\(a\\) and \\(b\\)", id="two-spans"),
    pytest.param(
        r"$1 - \sigma(u)$",
        r"\\(1 - \sigma(u)\\)",
        id="maths-may-start-with-a-digit",
    ),
    pytest.param("costs $5 and $10", "costs $5 and $10", id="currency-pair"),
    pytest.param("it cost $5.", "it cost $5.", id="currency-single"),
    pytest.param(r'`price = "$5"`', r'`price = "$5"`', id="code-span"),
    pytest.param(r"\$5", r"\$5", id="escaped-dollar"),
    pytest.param("$ x $", "$ x $", id="space-after-opener"),
    pytest.param("$x $", "$x $", id="space-before-closer"),
    pytest.param(r"\\(already\\)", r"\\(already\\)", id="already-converted"),
    pytest.param(r"\(single\)", r"\(single\)", id="single-backslash-span"),
    pytest.param(r"\[ x \]", r"\[ x \]", id="display-brackets"),
    pytest.param("$$E=mc^2$$", "$$E=mc^2$$", id="inline-display"),
    pytest.param("empty $$ here", "empty $$ here", id="bare-double-dollar"),
    pytest.param("a $ b", "a $ b", id="lonely-dollar"),
]


@pytest.mark.parametrize("source, expected", CONVERSIONS)
def test_convert_inline_math(source: str, expected: str) -> None:
    assert convert_inline_math(source).text == expected


@pytest.mark.parametrize("source, _expected", CONVERSIONS)
def test_conversion_is_idempotent(source: str, _expected: str) -> None:
    once = convert_inline_math(source).text
    assert convert_inline_math(once).text == once


def test_display_block_is_untouched() -> None:
    text = "before\n\n$$\nf(x) = \\sum_{i=1}^{n} w_i\n$$\n\nafter $x$\n"
    result = convert_inline_math(text)
    assert "$$\nf(x) = \\sum_{i=1}^{n} w_i\n$$" in result.text
    assert result.text.endswith("after \\\\(x\\\\)\n")


def test_dollar_inside_display_block_is_untouched() -> None:
    text = "$$\na $x$ b\n$$\n"
    assert convert_inline_math(text).text == text


def test_code_fence_contents_are_untouched() -> None:
    text = '```python {.marimo}\nax.set(label="$z = b_0$")\n```\n'
    assert convert_inline_math(text).text == text


def test_tilde_fence_is_recognised() -> None:
    text = "~~~\n$x$\n~~~\n"
    assert convert_inline_math(text).text == text


def test_nested_fence_does_not_close_early() -> None:
    text = "````md\n```\n$x$\n```\n````\n\nafter $y$\n"
    result = convert_inline_math(text).text
    assert "$x$" in result
    assert result.endswith("after \\\\(y\\\\)\n")


def test_unclosed_fence_swallows_rest_of_document() -> None:
    text = "```python\n$x$\nstill code $y$\n"
    assert convert_inline_math(text).text == text


class TestRawHtmlIsLeftAlone:
    r"""The theme sets ``unsafe: true``, so raw HTML reaches the page verbatim.

    Goldmark does not process backslash escapes inside an HTML block, so
    ``$...$`` there is already safe for MathJax. Rewriting it would corrupt
    embedded JavaScript, which is how an interactive figure is embedded.
    """

    def test_script_with_currency_axis_is_untouched(self) -> None:
        text = (
            '<script>Plotly.newPlot("c",[{"y":[1,2]}],'
            '{"yaxis":{"tickprefix":"$","ticksuffix":"$"}});</script>\n'
        )
        assert convert_inline_math(text).text == text

    def test_script_with_latex_strings_is_untouched(self) -> None:
        text = '<script>render({"title":"$\\alpha$ vs $\\beta$"});</script>\n'
        assert convert_inline_math(text).text == text

    def test_multiline_script_is_untouched(self) -> None:
        text = '<script>\n  const a = "$x$";\n\n  const b = "$y$";\n</script>\n'
        assert convert_inline_math(text).text == text

    def test_html_attribute_is_untouched(self) -> None:
        text = '<div style="height:300px" data-x="$a$ and $b$"></div>\n'
        assert convert_inline_math(text).text == text

    def test_html_block_ends_at_a_blank_line(self) -> None:
        text = '<div id="chart"></div>\n\nProse after with $x$.\n'
        result = convert_inline_math(text).text
        assert '<div id="chart"></div>' in result
        assert result.endswith("Prose after with \\\\(x\\\\).\n")

    def test_prose_after_a_script_is_still_converted(self) -> None:
        text = "<script>\nvar a = 1;\n</script>\n\nThen $y$ follows.\n"
        result = convert_inline_math(text)
        assert result.converted == 1
        assert result.text.endswith("Then \\\\(y\\\\) follows.\n")

    def test_inline_html_mid_paragraph_still_converts(self) -> None:
        # Only a tag at the start of a line opens an HTML block.
        text = "Some <em>emphasis</em> and $x$ maths.\n"
        assert convert_inline_math(text).converted == 1

    def test_less_than_in_prose_is_not_an_html_block(self) -> None:
        text = "when $a$ < $b$ holds\n"
        assert convert_inline_math(text).converted == 2


def test_unmatched_dollar_is_reported() -> None:
    result = convert_inline_math("a lonely $ dollar\n")
    assert result.unmatched_dollars == [1]
    assert result.converted == 0


def test_converted_count() -> None:
    assert convert_inline_math("$a$ $b$ $c$").converted == 3


def test_iter_code_fences_reports_bounds_and_info() -> None:
    text = "intro\n```python {.marimo}\ncode\n```\ntail\n"
    assert iter_code_fences(text) == [(1, 3, "python {.marimo}")]


def test_strip_hidden_cells_removes_the_whole_block() -> None:
    text = (
        "intro\n\n"
        "```python\n# sjmelck: hide\nimport marimo as mo\n```\n\n"
        "```python\nkeep = 1\n```\n"
    )
    result, removed = strip_hidden_cells(text)
    assert removed == 1
    assert "import marimo" not in result
    assert "keep = 1" in result


def test_strip_hidden_cells_ignores_a_later_sentinel() -> None:
    text = "```python\nx = 1\n# sjmelck: hide\n```\n"
    result, removed = strip_hidden_cells(text)
    assert removed == 0
    assert result == text


def test_strip_fence_attributes() -> None:
    text = "```python {.marimo}\ncode\n```\n"
    assert strip_fence_attributes(text) == "```python\ncode\n```\n"


def test_strip_fence_attributes_leaves_plain_fences() -> None:
    text = "```python\ncode\n```\n"
    assert strip_fence_attributes(text) == text


def test_normalize_image_links_shortens_paths() -> None:
    text = "![alt](../assets/images/my-post/fig.png)\n"
    result, referenced = normalize_image_links(text, "my-post")
    assert result == "![alt](fig.png)\n"
    assert referenced == ["fig.png"]


def test_normalize_image_links_keeps_titles() -> None:
    text = '![alt](fig.png "A caption")\n'
    result, referenced = normalize_image_links(text, "my-post")
    assert result == text
    assert referenced == ["fig.png"]


def test_normalize_image_links_ignores_fenced_code() -> None:
    text = '```python\nprint("![alt](assets/images/my-post/fig.png)")\n```\n'
    result, referenced = normalize_image_links(text, "my-post")
    assert result == text
    assert referenced == []


def test_normalize_image_links_leaves_foreign_paths() -> None:
    text = "![alt](https://example.com/fig.png)\n"
    result, referenced = normalize_image_links(text, "my-post")
    assert result == text
    assert referenced == []


def test_has_mermaid_fence() -> None:
    assert has_mermaid_fence("```mermaid\ngraph LR;\n```\n")
    assert not has_mermaid_fence("```python\nx = 1\n```\n")


def test_normalise_whitespace() -> None:
    assert normalise_whitespace("a  \r\nb\n\n\n") == "a\nb\n"
