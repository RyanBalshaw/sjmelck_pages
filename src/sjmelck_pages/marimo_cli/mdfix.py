"""Pure text transformations over marimo's exported markdown.

Nothing in this module touches the filesystem or runs a subprocess, which
keeps it directly testable.

Equations are not rewritten here. Authors write inline maths as ``\\(...\\)``
and display maths as ``$$...$$``, which is what AGENTS.md has always asked
for and what the theme's Goldmark passthrough is configured to handle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FENCE_OPEN = re.compile(r"^(?P<indent>\s{0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
HIDE_SENTINEL = re.compile(r"^#\s*sjmelck:\s*hide\b")
IMAGE_LINK = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
MATHS_DELIMITER = re.compile(r"\\\\\(|\\\(|\$\$|\\\[")


@dataclass
class _Fence:
    char: str
    length: int


def _split_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\n", "\r")):
        return line[:-1], line[-1]
    return line, ""


def _closes_fence(text: str, fence: _Fence) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] != fence.char:
        return False
    run = len(stripped) - len(stripped.lstrip(fence.char))
    return run >= fence.length and not stripped[run:].strip()


def iter_code_fences(markdown: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, info)`` line indices for every fenced block.

    ``start`` is the opening delimiter, ``end`` the closing one (or the last
    line of the document for an unclosed fence). Both are 0-based.
    """
    fences: list[tuple[int, int, str]] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        match = FENCE_OPEN.match(lines[index])
        if not match:
            index += 1
            continue
        marker = match.group("fence")
        fence = _Fence(char=marker[0], length=len(marker))
        info = match.group("info").strip()
        end = len(lines) - 1
        cursor = index + 1
        while cursor < len(lines):
            if _closes_fence(lines[cursor], fence):
                end = cursor
                break
            cursor += 1
        fences.append((index, end, info))
        index = end + 1
    return fences


def strip_hidden_cells(markdown: str) -> tuple[str, int]:
    """Drop fenced blocks whose first body line is ``# sjmelck: hide``.

    Without this every exported post opens with an ``import marimo as mo``
    block. Returns the new text and how many blocks were removed.
    """
    lines = markdown.splitlines(keepends=True)
    drop: set[int] = set()
    removed = 0

    for start, end, _info in iter_code_fences(markdown):
        body = start + 1
        if (
            body <= end
            and body < len(lines)
            and HIDE_SENTINEL.match(lines[body].strip())
        ):
            drop.update(range(start, end + 1))
            removed += 1
            # Swallow one trailing blank line so paragraphs stay tidy.
            if end + 1 < len(lines) and not lines[end + 1].strip():
                drop.add(end + 1)

    if not removed:
        return markdown, 0
    return "".join(line for i, line in enumerate(lines) if i not in drop), removed


def strip_fence_attributes(markdown: str) -> str:
    """Rewrite a ``python {.marimo}`` info string down to ``python``.

    Goldmark consumes the attribute and highlighting works either way, but
    the committed markdown is what a human reads, and dropping it removes a
    silent dependency on ``parser.attribute.block`` staying enabled.
    """
    lines = markdown.splitlines(keepends=True)
    for start, _end, _info in iter_code_fences(markdown):
        text, newline = _split_newline(lines[start])
        match = FENCE_OPEN.match(text)
        if not match:
            continue
        info = match.group("info")
        cleaned = re.sub(r"\s*\{[^}]*\}\s*$", "", info).rstrip()
        if cleaned != info:
            lines[start] = (
                match.group("indent") + match.group("fence") + cleaned + newline
            )
    return "".join(lines)


def normalize_image_links(markdown: str, slug: str) -> tuple[str, list[str]]:
    """Reduce image targets to bare filenames and collect what is referenced.

    The theme resolves ``![alt](figure.png)`` against ``assets/images/<slug>/``
    itself, so a longer path is redundant. Code fences are left alone.
    """
    referenced: list[str] = []
    lines = markdown.splitlines(keepends=True)
    fenced: set[int] = set()
    for start, end, _info in iter_code_fences(markdown):
        fenced.update(range(start, end + 1))

    marker = f"assets/images/{slug}/"

    def _replace(match: re.Match[str]) -> str:
        normalised = match.group(2).replace("\\", "/")
        if marker in normalised:
            name = normalised.split(marker, 1)[1]
        elif "/" not in normalised:
            name = normalised
        else:
            return match.group(0)
        if name and name not in referenced:
            referenced.append(name)
        return f"{match.group(1)}{name}{match.group(3)}"

    for index, line in enumerate(lines):
        if index not in fenced:
            lines[index] = IMAGE_LINK.sub(_replace, line)

    return "".join(lines), referenced


def has_maths(markdown: str) -> bool:
    """True if the body uses a maths delimiter outside a code fence.

    Used only to check that ``math: true`` is set, which loads MathJax.
    """
    lines = markdown.splitlines()
    fenced: set[int] = set()
    for start, end, _info in iter_code_fences(markdown):
        fenced.update(range(start, end + 1))
    return any(
        MATHS_DELIMITER.search(line)
        for index, line in enumerate(lines)
        if index not in fenced
    )


def has_mermaid_fence(markdown: str) -> bool:
    """True if the document contains a mermaid block."""
    for _start, _end, info in iter_code_fences(markdown):
        parts = info.split()
        if parts and parts[0].lower() == "mermaid":
            return True
    return False


def normalise_whitespace(text: str) -> str:
    """LF endings, no trailing whitespace, exactly one final newline.

    Doing this ourselves means the ``trailing-whitespace``,
    ``end-of-file-fixer`` and ``mixed-line-ending`` hooks never rewrite the
    generated file afterwards, so re-running the converter is byte-stable.
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).rstrip("\n") + "\n"
