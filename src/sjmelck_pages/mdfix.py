r"""Pure text transformations over marimo's exported markdown.

Nothing in this module touches the filesystem or runs a subprocess, which
keeps the awkward parts directly testable.

The important one is :func:`convert_inline_math`. The Sjmelck theme enables
Goldmark's passthrough extension for ``\(...\)``, ``\[...\]`` and ``$$...$$``
but *not* for single dollar ``$...$``. MathJax accepts ``$...$`` at runtime,
so simple spans appear to work, but Goldmark has already eaten any backslash
escapes by then: ``$\{x\} \| y \|$`` reaches the browser as ``${x} | y |$``.
Rewriting inline spans to ``\\(...\\)`` is what makes escapes survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

FENCE_OPEN = re.compile(r"^(?P<indent>\s{0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
HIDE_SENTINEL = re.compile(r"^#\s*sjmelck:\s*hide\b")
IMAGE_LINK = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")

# CommonMark HTML blocks. Type 1 runs to its own closing tag and may contain
# blank lines; every other kind ends at a blank line. The theme sets
# `unsafe: true`, so raw HTML reaches the page verbatim - and Goldmark does
# not process backslash escapes inside it, which means `$...$` there is
# already safe for MathJax and must be left alone. Rewriting it would corrupt
# things like a Plotly axis with `"tickprefix": "$"`.
HTML_RAW_OPEN = re.compile(r"^\s{0,3}<(script|pre|style|textarea)\b", re.IGNORECASE)
HTML_RAW_CLOSE = re.compile(r"</(script|pre|style|textarea)\s*>", re.IGNORECASE)
HTML_BLOCK_OPEN = re.compile(r"^\s{0,3}</?[a-zA-Z][a-zA-Z0-9-]*(\s|/?>|$)")


class _State(Enum):
    NORMAL = auto()
    FENCE = auto()
    DISPLAY_DOLLAR = auto()
    DISPLAY_BRACKET = auto()
    HTML_RAW = auto()
    HTML_BLOCK = auto()


@dataclass
class MathResult:
    """Outcome of an inline-math conversion pass."""

    text: str
    converted: int = 0
    unmatched_dollars: list[int] = field(default_factory=list)


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


def _is_valid_opener(line: str, pos: int) -> bool:
    """A ``$`` opens an inline span only if what follows looks like maths.

    A digit is deliberately allowed here: ``$1 - \\sigma(u)$`` is ordinary
    maths. Currency is rejected by the closer rules instead, because in
    ``costs $5 and $10`` the candidate closer is preceded by whitespace.
    """
    if pos + 1 >= len(line):
        return False
    nxt = line[pos + 1]
    return not (nxt.isspace() or nxt == "$")


def _find_closer(line: str, start: int) -> int:
    r"""Find the closing ``$`` of an inline span, or -1.

    Aborts on a backtick or ``\(`` so a code span or an already-converted
    span cannot be swallowed.
    """
    j = start
    while j < len(line):
        ch = line[j]
        if ch == "`":
            return -1
        if ch == "\\":
            if line[j : j + 2] == "\\(":
                return -1
            j += 2
            continue
        if ch == "$":
            if line[j - 1] == "\\" or line[j - 1].isspace():
                return -1
            if j + 1 < len(line) and (line[j + 1] == "$" or line[j + 1].isdigit()):
                return -1
            return j
        j += 1
    return -1


def _scan_line(line: str) -> tuple[str, _State, int, bool]:
    """Convert inline maths on one line.

    Returns the rewritten line, the state the next line starts in, how many
    spans were converted, and whether a stray ``$`` was left behind.
    """
    out: list[str] = []
    pos = 0
    converted = 0
    stray = False

    while pos < len(line):
        ch = line[pos]

        if ch == "\\":
            pair = line[pos : pos + 2]
            if pair == "\\(":
                end = line.find("\\)", pos + 2)
                if end == -1:
                    out.append(line[pos:])
                    return "".join(out), _State.NORMAL, converted, stray
                out.append(line[pos : end + 2])
                pos = end + 2
                continue
            if pair == "\\[":
                end = line.find("\\]", pos + 2)
                if end == -1:
                    out.append(line[pos:])
                    return "".join(out), _State.DISPLAY_BRACKET, converted, stray
                out.append(line[pos : end + 2])
                pos = end + 2
                continue
            # Any other escape, including \$ and \\, is copied as a pair. This
            # is also what makes an existing \\(...\\) span idempotent.
            out.append(line[pos : pos + 2])
            pos += 2
            continue

        if ch == "`":
            run = len(line[pos:]) - len(line[pos:].lstrip("`"))
            ticks = "`" * run
            end = line.find(ticks, pos + run)
            while end != -1 and line[end : end + run + 1] == ticks + "`":
                end = line.find(ticks, end + 1)
            if end == -1:
                out.append(ticks)
                pos += run
                continue
            out.append(line[pos : end + run])
            pos = end + run
            continue

        if line[pos : pos + 2] == "$$":
            end = line.find("$$", pos + 2)
            if end == -1:
                out.append(line[pos:])
                return "".join(out), _State.DISPLAY_DOLLAR, converted, stray
            out.append(line[pos : end + 2])
            pos = end + 2
            continue

        if ch == "$":
            if _is_valid_opener(line, pos):
                closer = _find_closer(line, pos + 1)
                if closer != -1:
                    content = line[pos + 1 : closer]
                    if content and "$" not in content:
                        out.append("\\\\(" + content + "\\\\)")
                        converted += 1
                        pos = closer + 1
                        continue
            stray = True
            out.append("$")
            pos += 1
            continue

        out.append(ch)
        pos += 1

    return "".join(out), _State.NORMAL, converted, stray


def convert_inline_math(markdown: str) -> MathResult:
    r"""Rewrite single-dollar inline maths to the Goldmark-safe form.

    Code fences, code spans, ``$$...$$`` display maths and existing
    ``\\(...\\)`` spans are left byte-identical. The pass is idempotent.
    """
    out: list[str] = []
    state = _State.NORMAL
    fence: _Fence | None = None
    converted = 0
    stray_lines: list[int] = []

    for number, raw in enumerate(markdown.splitlines(keepends=True), start=1):
        text, newline = _split_newline(raw)

        if state is _State.FENCE:
            out.append(raw)
            if fence is not None and _closes_fence(text, fence):
                state, fence = _State.NORMAL, None
            continue

        if state is _State.HTML_RAW:
            out.append(raw)
            if HTML_RAW_CLOSE.search(text):
                state = _State.NORMAL
            continue

        if state is _State.HTML_BLOCK:
            out.append(raw)
            if not text.strip():
                state = _State.NORMAL
            continue

        if state in (_State.DISPLAY_DOLLAR, _State.DISPLAY_BRACKET):
            closer = "$$" if state is _State.DISPLAY_DOLLAR else "\\]"
            index = text.find(closer)
            if index == -1:
                out.append(raw)
                continue
            head = text[: index + len(closer)]
            tail, state, count, stray = _scan_line(text[index + len(closer) :])
            converted += count
            if stray:
                stray_lines.append(number)
            out.append(head + tail + newline)
            continue

        match = FENCE_OPEN.match(text)
        if match:
            marker = match.group("fence")
            fence = _Fence(char=marker[0], length=len(marker))
            state = _State.FENCE
            out.append(raw)
            continue

        if HTML_RAW_OPEN.match(text):
            out.append(raw)
            if not HTML_RAW_CLOSE.search(text):
                state = _State.HTML_RAW
            continue

        if HTML_BLOCK_OPEN.match(text):
            out.append(raw)
            if text.strip():
                state = _State.HTML_BLOCK
            continue

        converted_text, state, count, stray = _scan_line(text)
        converted += count
        if stray:
            stray_lines.append(number)
        out.append(converted_text + newline)

    return MathResult("".join(out), converted, stray_lines)


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
