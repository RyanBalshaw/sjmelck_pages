"""Command line interface for the Sjmelck post tooling."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sjmelck_pages import convert as convert_module
from sjmelck_pages.repo import (
    RepoError,
    find_repo_root,
    image_dir,
    notebook_path,
    post_path,
    relative_to_root,
    sast_now,
)
from sjmelck_pages.scaffold import create_post

LEVEL_PREFIX = {"error": "error  ", "warning": "warning", "info": "info   "}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sjmelck-pages",
        description="Write Sjmelck blog posts as marimo notebooks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="scaffold a notebook for a new post")
    new.add_argument("title", help="post title, in sentence case")
    new.add_argument("--author", default="dummy-name", help="post author")
    new.add_argument("--slug", help="override the slug derived from the title")
    new.add_argument(
        "--force", action="store_true", help="overwrite an existing notebook"
    )

    convert = sub.add_parser(
        "convert", aliases=["import"], help="turn a notebook into a Hugo post"
    )
    convert.add_argument("notebook", type=Path, help="path to the marimo notebook")
    convert.add_argument("--slug", help="override the slug derived from the filename")
    convert.add_argument(
        "--dry-run", action="store_true", help="print the post instead of writing it"
    )
    convert.add_argument(
        "--strict", action="store_true", help="exit non-zero on warnings too"
    )
    convert.add_argument(
        "--run-notebook",
        action="store_true",
        help="execute the notebook first so its figures are regenerated",
    )
    convert.add_argument(
        "--from-markdown",
        type=Path,
        help="convert this already-exported markdown instead of running marimo",
    )
    convert.add_argument(
        "--marimo-cmd", nargs="+", help="command used to invoke marimo"
    )
    convert.add_argument(
        "--keep-marimo-attrs",
        action="store_true",
        help="leave the {.marimo} attribute on code fences",
    )

    return parser


def _report(notes: list[convert_module.Note]) -> None:
    for note in notes:
        print(f"  {LEVEL_PREFIX[note.level]}  {note.message}", file=sys.stderr)


def _cmd_new(args: argparse.Namespace) -> int:
    root = find_repo_root()
    slug, notebook = create_post(
        root,
        args.title,
        author=args.author,
        slug=args.slug,
        force=args.force,
    )
    folder = image_dir(root, slug)
    print(f"Created {relative_to_root(root, notebook)}")
    print(f"Figures go in {relative_to_root(root, folder)}/")
    print()
    print("Next:")
    print(f"  uvx marimo edit {relative_to_root(root, notebook)}")
    print(f"  uv run sjmelck-pages convert {relative_to_root(root, notebook)}")
    print("  hugo server -D")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    root = find_repo_root()
    notebook = args.notebook.resolve()
    slug = args.slug or notebook.stem

    result = convert_module.convert_notebook(
        root,
        notebook,
        slug=slug,
        now=sast_now(),
        from_markdown=args.from_markdown,
        marimo_cmd=args.marimo_cmd,
        execute=args.run_notebook,
        keep_attrs=args.keep_marimo_attrs,
    )

    if args.dry_run:
        print(result.text, end="")
    else:
        target = post_path(root, slug)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.text, encoding="utf-8", newline="\n")
        image_dir(root, slug).mkdir(parents=True, exist_ok=True)
        print(f"Wrote {relative_to_root(root, target)}")

    summary = (
        f"{result.converted_math} inline maths span(s) converted, "
        f"{result.hidden_cells} hidden cell(s) removed, "
        f"{len(result.referenced_images)} image(s) referenced"
    )
    print(summary, file=sys.stderr)
    _report(result.notes)

    if result.errors:
        return 1
    if args.strict and result.warnings:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "new": _cmd_new,
        "convert": _cmd_convert,
        "import": _cmd_convert,
    }

    try:
        return handlers[args.command](args)
    except RepoError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _console_main() -> None:
    raise SystemExit(main())


__all__ = ["build_parser", "main", "notebook_path"]
