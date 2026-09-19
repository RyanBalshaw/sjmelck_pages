"""Tests for front-matter splicing, the checks, and the whole pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from sjmelck_pages.convert import (
    ERROR,
    INFO,
    WARNING,
    check_post,
    convert,
    default_front_matter,
    split_front_matter,
)
from sjmelck_pages.mdfix import convert_inline_math
from sjmelck_pages.repo import SAST

NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=SAST)

MARIMO_BLOCK = "---\ntitle: Demo\nmarimo-version: 0.24.2\n---\n"
SJMELCK_BLOCK = '---\ntitle: "Demo post"\npublishdate: 2026-09-18T10:00:00+02:00\n---\n'


def _front(**overrides: str) -> str:
    fields = {
        "title": '"Demo post"',
        "publishdate": "2026-09-18T10:00:00+02:00",
        "author": "Johann Bouwer",
        "draft": "false",
        "toc": "true",
        "math": "true",
        "hasMermaid": "false",
    }
    fields.update(overrides)
    body = "\n".join(f"{key}: {value}" for key, value in fields.items())
    return f"---\n{body}\n---"


class TestSplitFrontMatter:
    def test_both_blocks_present(self) -> None:
        front, body = split_front_matter(MARIMO_BLOCK + "\n" + SJMELCK_BLOCK + "\ntext")
        assert front is not None
        assert "Demo post" in front
        assert "marimo-version" not in front
        assert body.strip() == "text"

    def test_only_marimo_block(self) -> None:
        front, body = split_front_matter(MARIMO_BLOCK + "\njust text\n")
        assert front is None
        assert body.strip() == "just text"

    def test_no_front_matter_at_all(self) -> None:
        front, body = split_front_matter("plain text\n")
        assert front is None
        assert body == "plain text\n"

    def test_horizontal_rule_later_in_prose_is_not_front_matter(self) -> None:
        text = MARIMO_BLOCK + "\nsome prose\n\n---\n\nmore prose\n"
        front, body = split_front_matter(text)
        assert front is None
        assert "---" in body
        assert "some prose" in body

    def test_unterminated_second_block(self) -> None:
        text = MARIMO_BLOCK + '\n---\ntitle: "Oops"\n'
        front, body = split_front_matter(text)
        assert front is None
        assert "Oops" in body


class TestDefaultFrontMatter:
    def test_shape(self) -> None:
        front = default_front_matter("my-post", now=NOW)
        assert front.startswith("---\n")
        assert front.endswith("\n---")
        assert 'title: "My post"' in front
        assert "publishdate: 2026-09-19T12:00:00+02:00" in front
        assert "  render: always" in front


class TestChecks:
    def _run(self, front: str, body: str = "", *, folder: Path, **kwargs: object):
        defaults = {
            "slug": "demo",
            "folder": folder,
            "referenced": [],
            "converted_math": 0,
            "now": NOW,
        }
        defaults.update(kwargs)
        return check_post(front, body, **defaults)  # type: ignore[arg-type]

    def test_future_publishdate_is_an_error(self, tmp_path: Path) -> None:
        future = (NOW + timedelta(days=1)).isoformat()
        notes = self._run(_front(publishdate=future), folder=tmp_path)
        assert any(n.level == ERROR and "future" in n.message for n in notes)

    def test_past_publishdate_is_fine(self, tmp_path: Path) -> None:
        notes = self._run(_front(), folder=tmp_path)
        assert not [n for n in notes if n.level == ERROR]

    def test_missing_publishdate_is_an_error(self, tmp_path: Path) -> None:
        notes = self._run("---\ntitle: x\n---", folder=tmp_path)
        assert any(n.level == ERROR and "publishdate" in n.message for n in notes)

    def test_unparseable_publishdate_is_an_error(self, tmp_path: Path) -> None:
        notes = self._run(_front(publishdate="last tuesday"), folder=tmp_path)
        assert any(n.level == ERROR and "ISO 8601" in n.message for n in notes)

    def test_missing_image_is_an_error(self, tmp_path: Path) -> None:
        notes = self._run(_front(), folder=tmp_path, referenced=["gone.png"])
        assert any(n.level == ERROR and "gone.png" in n.message for n in notes)

    def test_present_image_is_not_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "there.png").write_bytes(b"")
        notes = self._run(_front(), folder=tmp_path, referenced=["there.png"])
        assert not [n for n in notes if n.level == ERROR]

    def test_converted_maths_without_math_flag_is_an_error(
        self, tmp_path: Path
    ) -> None:
        notes = self._run(_front(math="false"), folder=tmp_path, converted_math=3)
        assert any(n.level == ERROR and "math: true" in n.message for n in notes)

    def test_mermaid_without_flag_is_an_error(self, tmp_path: Path) -> None:
        notes = self._run(_front(), "```mermaid\ngraph LR;\n```\n", folder=tmp_path)
        assert any(n.level == ERROR and "hasMermaid" in n.message for n in notes)

    def test_placeholders_warn(self, tmp_path: Path) -> None:
        notes = self._run(_front(author="dummy-name"), folder=tmp_path)
        assert any(n.level == WARNING and "dummy-name" in n.message for n in notes)

    def test_short_description_warns(self, tmp_path: Path) -> None:
        notes = self._run(_front(description="too short"), folder=tmp_path)
        assert any(n.level == WARNING and "characters" in n.message for n in notes)

    def test_missing_overview_fig_warns(self, tmp_path: Path) -> None:
        notes = self._run(_front(), folder=tmp_path)
        assert any(n.level == WARNING and "OverviewFig" in n.message for n in notes)

    def test_mo_ui_warns(self, tmp_path: Path) -> None:
        notes = self._run(
            _front(), "```python\nslider = mo.ui.slider(1, 10)\n```\n", folder=tmp_path
        )
        assert any(n.level == WARNING and "mo.ui" in n.message for n in notes)

    def test_h1_heading_warns(self, tmp_path: Path) -> None:
        notes = self._run(_front(), "# Top level\n", folder=tmp_path)
        assert any(n.level == WARNING and "##" in n.message for n in notes)

    def test_h1_inside_a_fence_does_not_warn(self, tmp_path: Path) -> None:
        notes = self._run(_front(), "```python\n# Top level\n```\n", folder=tmp_path)
        assert not [n for n in notes if "##" in n.message]

    def test_unreferenced_image_is_info(self, tmp_path: Path) -> None:
        (tmp_path / "stray.png").write_bytes(b"")
        notes = self._run(_front(), folder=tmp_path)
        assert any(n.level == INFO and "stray.png" in n.message for n in notes)

    def test_draft_is_info(self, tmp_path: Path) -> None:
        notes = self._run(_front(draft="true"), folder=tmp_path)
        assert any(n.level == INFO and "draft" in n.message for n in notes)


class TestConvert:
    def _convert(self, exported: str, tmp_path: Path):
        return convert(
            exported,
            slug="demo",
            folder=tmp_path,
            now=NOW,
            source="assets/images/demo/demo.py",
        )

    def test_end_to_end(self, tmp_path: Path) -> None:
        (tmp_path / "fig.png").write_bytes(b"")
        exported = (
            MARIMO_BLOCK
            + "\n"
            + _front(OverviewFig='"fig.png"')
            + "\n\n"
            + "Some prose with $x^2$ maths.\n\n"
            + "```python {.marimo}\n# sjmelck: hide\nimport marimo as mo\n```\n\n"
            + "```python {.marimo}\nkeep = 1\n```\n\n"
            + "![A figure](../assets/images/demo/fig.png)\n"
        )
        result = self._convert(exported, tmp_path)

        assert result.converted_math == 1
        assert result.hidden_cells == 1
        assert result.referenced_images == ["fig.png"]
        assert "marimo-version" not in result.text
        assert "import marimo" not in result.text
        assert "```python\nkeep = 1" in result.text
        assert r"\\(x^2\\)" in result.text
        assert "![A figure](fig.png)" in result.text
        assert "Generated by" in result.text
        assert not result.errors

    def test_converted_output_is_stable_under_another_maths_pass(
        self, tmp_path: Path
    ) -> None:
        exported = MARIMO_BLOCK + "\n" + _front() + "\n\nText with $y$.\n"
        first = self._convert(exported, tmp_path).text
        assert convert_inline_math(first).text == first

    def test_missing_front_matter_is_synthesised_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        result = self._convert(MARIMO_BLOCK + "\nJust prose.\n", tmp_path)
        assert any("No Sjmelck front matter" in n.message for n in result.warnings)
        assert "publishdate:" in result.text

    def test_generated_stamp_names_the_source(self, tmp_path: Path) -> None:
        result = self._convert(MARIMO_BLOCK + "\n" + _front() + "\n\nx\n", tmp_path)
        assert "assets/images/demo/demo.py" in result.text

    def test_keep_attrs_leaves_the_fence_alone(self, tmp_path: Path) -> None:
        exported = (
            MARIMO_BLOCK + "\n" + _front() + "\n\n```python {.marimo}\nx = 1\n```\n"
        )
        result = convert(
            exported,
            slug="demo",
            folder=tmp_path,
            now=NOW,
            source="demo.py",
            keep_attrs=True,
        )
        assert "{.marimo}" in result.text

    def test_text_ends_with_exactly_one_newline(self, tmp_path: Path) -> None:
        result = self._convert(MARIMO_BLOCK + "\n" + _front() + "\n\nx\n\n\n", tmp_path)
        assert result.text.endswith("x\n")
        assert not result.text.endswith("x\n\n")
