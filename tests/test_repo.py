"""Tests for repository layout discovery."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from sjmelck_pages.repo import (
    SAST,
    RepoError,
    find_repo_root,
    format_publishdate,
    image_dir,
    notebook_path,
    post_path,
    relative_to_root,
    sast_now,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "content" / "blog").mkdir(parents=True)
    (root / "config.yaml").write_text("title: Sjmelck\n", encoding="utf-8")
    return root


def test_find_repo_root_from_the_root(fake_repo: Path) -> None:
    assert find_repo_root(fake_repo) == fake_repo.resolve()


def test_find_repo_root_from_a_nested_directory(fake_repo: Path) -> None:
    nested = fake_repo / "assets" / "images" / "a-post"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == fake_repo.resolve()


def test_find_repo_root_raises_outside_a_repo(tmp_path: Path) -> None:
    lonely = tmp_path / "elsewhere"
    lonely.mkdir()
    with pytest.raises(RepoError, match="repository root"):
        find_repo_root(lonely)


def test_config_without_content_blog_is_not_the_root(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    with pytest.raises(RepoError):
        find_repo_root(tmp_path)


def test_paths_follow_the_slug_contract(fake_repo: Path) -> None:
    assert image_dir(fake_repo, "a-post") == fake_repo / "assets" / "images" / "a-post"
    assert notebook_path(fake_repo, "a-post").name == "a-post.py"
    assert notebook_path(fake_repo, "a-post").parent == image_dir(fake_repo, "a-post")
    assert post_path(fake_repo, "a-post") == (
        fake_repo / "content" / "blog" / "a-post.md"
    )


def test_relative_to_root_uses_forward_slashes(fake_repo: Path) -> None:
    path = notebook_path(fake_repo, "a-post")
    assert relative_to_root(fake_repo, path) == "assets/images/a-post/a-post.py"


def test_relative_to_root_falls_back_for_outside_paths(
    fake_repo: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "somewhere" / "else.py"
    assert relative_to_root(fake_repo, outside).endswith("else.py")


def test_sast_now_is_offset_aware() -> None:
    now = sast_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == SAST.utcoffset(None)
    assert now.microsecond == 0


def test_format_publishdate_is_iso_with_offset() -> None:
    moment = datetime(2026, 9, 19, 12, 0, 0, tzinfo=SAST)
    assert format_publishdate(moment) == "2026-09-19T12:00:00+02:00"
