"""Render the Mermaid diagrams in ``diagrams/`` to light and dark PNGs for the docs.

Run from anywhere inside the repository::

    python scripts/render_diagrams.py                      # render every diagram
    python scripts/render_diagrams.py --only architecture  # render selected diagrams
    python scripts/render_diagrams.py --check              # verify without rendering

Rendering runs the pinned ``@mermaid-js/mermaid-cli`` through ``npx`` (Node.js
22.13 or later). Puppeteer downloads its own browser on first use; to render with
an installed Chrome or Edge instead, set ``PUPPETEER_EXECUTABLE_PATH`` to the
browser executable.

``--check`` fails when a source or config changed since the last render, a PNG is
missing, or ``docs/assets/diagrams`` contains files that no source produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "diagrams"
OUTPUT = ROOT / "docs" / "assets" / "diagrams"
MANIFEST = SOURCES / "manifest.json"
MERMAID_CLI_VERSION = "12.0.0"
MERMAID_CLI = f"@mermaid-js/mermaid-cli@{MERMAID_CLI_VERSION}"
SCALE = 2
BROWSER_VARIABLE = "PUPPETEER_EXECUTABLE_PATH"
# Diagrams wider than this (in CSS pixels) shrink noticeably in the docs column.
WIDE_DIAGRAM = 900
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OUTPUT_NAME = re.compile(r"(?P<name>.+)-(?P<theme>light|dark)\.png")
DESCRIPTION = "Render the Mermaid diagrams to light and dark PNGs for the docs."


@dataclass(frozen=True)
class Theme:
    """One rendered variant of every diagram."""

    name: str
    config: str
    background: str


THEMES = (
    Theme("light", "mermaid-light.json", "white"),
    Theme("dark", "mermaid-dark.json", "transparent"),
)


class RenderError(Exception):
    """Rendering could not start, or a diagram failed to render."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit status."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="render only these diagrams (source names without .mmd)",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="verify sources, configs, manifest, and PNGs without rendering",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="number of PNGs rendered in parallel (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.check:
        return check()
    try:
        render(args.only, jobs=args.jobs)
    except RenderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def check() -> int:
    """Report every reason the rendered diagrams are out of date."""
    problems = find_problems()
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        print(
            "Render the diagrams with `python scripts/render_diagrams.py`.",
            file=sys.stderr,
        )
        return 1
    print(f"{len(sources())} diagrams are up to date.")
    return 0


def find_problems() -> list[str]:
    """Return the reasons the rendered diagrams do not match their sources."""
    problems = [
        f"{relative(SOURCES / theme.config)} is missing"
        for theme in THEMES
        if not (SOURCES / theme.config).is_file()
    ]
    if problems:
        return problems
    try:
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"{relative(MANIFEST)} is missing"]
    except json.JSONDecodeError as error:
        return [f"{relative(MANIFEST)} is not valid JSON: {error}"]
    available = sources()
    expected = manifest(available)
    if recorded.get("mermaid_cli") != MERMAID_CLI_VERSION:
        problems.append(
            f"the diagrams were rendered with mermaid-cli "
            f"{recorded.get('mermaid_cli')}, not {MERMAID_CLI_VERSION}"
        )
    if recorded.get("scale") != SCALE:
        problems.append(f"the diagrams were not rendered at scale {SCALE}")
    recorded_configs = recorded.get("configs") or {}
    problems.extend(
        f"{relative(SOURCES / config)} changed since the last render"
        for config, checksum in expected["configs"].items()
        if recorded_configs.get(config) != checksum
    )
    recorded_diagrams = recorded.get("diagrams") or {}
    for name, checksum in expected["diagrams"].items():
        source = relative(available[name])
        if name not in recorded_diagrams:
            problems.append(f"{source} has not been rendered")
        elif recorded_diagrams[name] != checksum:
            problems.append(f"{source} changed since the last render")
        problems.extend(output_problems(name))
    problems.extend(
        f"{relative(MANIFEST)} lists {name}, which has no source"
        for name in sorted(recorded_diagrams.keys() - available.keys())
    )
    problems.extend(describe_stray(path) for path in stray_files(available))
    return problems


def output_problems(name: str) -> list[str]:
    """Describe missing or invalid PNGs of one diagram."""
    problems = []
    for path in outputs(name):
        if not path.is_file():
            problems.append(f"{relative(path)} is missing")
        elif not is_png(path):
            problems.append(f"{relative(path)} is not a PNG image")
    return problems


def stray_files(names: Iterable[str]) -> list[Path]:
    """Return the entries of the output directory that no source produces."""
    if not OUTPUT.is_dir():
        return []
    produced = {path.name for name in names for path in outputs(name)}
    return [path for path in sorted(OUTPUT.iterdir()) if path.name not in produced]


def describe_stray(path: Path) -> str:
    """Explain why one output entry is unexpected."""
    match = OUTPUT_NAME.fullmatch(path.name)
    if match is None:
        return f"{relative(path)} is not produced from a diagram source"
    source = relative(SOURCES / f"{match['name']}.mmd")
    return f"{relative(path)} has no source {source}"


def render(only: Sequence[str] | None, *, jobs: int) -> None:
    """Render diagrams and update the manifest.

    Without ``only``, every diagram is rendered, files in the output directory
    that no source produces are removed, and the manifest is rewritten.

    Raises:
        RenderError: If a name is unknown, the tooling is unavailable, or a
            diagram fails to render.
    """
    available = sources()
    if not available:
        raise RenderError(f"no diagram sources found in {relative(SOURCES)}")
    missing = [
        theme.config for theme in THEMES if not (SOURCES / theme.config).is_file()
    ]
    if missing:
        raise RenderError(f"missing Mermaid configs in diagrams/: {', '.join(missing)}")
    expected = manifest(available)
    if only:
        unknown = sorted(set(only) - available.keys())
        if unknown:
            raise RenderError(
                f"unknown diagrams: {', '.join(unknown)}; "
                f"available: {', '.join(available)}"
            )
        recorded = partial_baseline(expected)
        names = sorted(set(only))
    else:
        recorded = {**expected, "diagrams": {}}
        names = list(available)
    npx = find_npx()
    run_cli(npx, ["--version"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tasks = [(name, theme) for name in names for theme in THEMES]
    with tempfile.TemporaryDirectory(prefix="agloom-diagrams-") as scratch:
        workdir = Path(scratch)
        options = browser_options(workdir)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(
                pool.map(lambda task: render_one(npx, *task, workdir, options), tasks)
            )
    failures = [result for result in results if isinstance(result, str)]
    for result in results:
        if isinstance(result, Path):
            report(result)
    if failures:
        raise RenderError("\n".join(failures))
    if not only:
        for path in stray_files(available):
            if path.is_file():
                path.unlink()
                print(f"removed {relative(path)}")
    diagrams = dict(recorded["diagrams"])
    diagrams.update({name: expected["diagrams"][name] for name in names})
    recorded["diagrams"] = dict(sorted(diagrams.items()))
    MANIFEST.write_text(
        json.dumps(recorded, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"updated {relative(MANIFEST)}")


def partial_baseline(expected: dict[str, Any]) -> dict[str, Any]:
    """Return the recorded manifest that a partial render extends.

    Raises:
        RenderError: If there is no valid manifest, or the configs or the
            renderer changed, in which case every diagram must be rendered.
    """
    try:
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RenderError(
            f"{relative(MANIFEST)} is missing or invalid; render every diagram"
        ) from error
    keys = ("mermaid_cli", "scale", "configs")
    if any(recorded.get(key) != expected[key] for key in keys):
        raise RenderError(
            "the Mermaid configs or the renderer changed since the last render; "
            "render every diagram (omit --only)"
        )
    recorded.setdefault("diagrams", {})
    return recorded


def render_one(
    npx: str, name: str, theme: Theme, workdir: Path, options: Sequence[str]
) -> Path | str:
    """Render one diagram variant; return its path or an error description."""
    target = OUTPUT / f"{name}-{theme.name}.png"
    scratch = workdir / target.name
    arguments = [
        "--quiet",
        "--input",
        str(SOURCES / f"{name}.mmd"),
        "--output",
        str(scratch),
        "--configFile",
        str(SOURCES / theme.config),
        "--backgroundColor",
        theme.background,
        "--scale",
        str(SCALE),
        *options,
    ]
    try:
        run_cli(npx, arguments)
    except RenderError as error:
        return f"{name} ({theme.name}): {error}"
    if not scratch.is_file() or not is_png(scratch):
        return f"{name} ({theme.name}): mermaid-cli did not write a PNG"
    shutil.move(scratch, target)
    return target


def run_cli(npx: str, arguments: Sequence[str]) -> None:
    """Run the pinned mermaid-cli through npx.

    Raises:
        RenderError: If npx cannot be started or mermaid-cli fails.
    """
    command = [npx, "--yes", MERMAID_CLI, *arguments]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", check=False
        )
    except OSError as error:
        raise RenderError(f"cannot run {npx}: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RenderError(detail or f"mermaid-cli exited with {completed.returncode}")


def find_npx() -> str:
    """Return the npx executable (``npx.cmd`` on Windows).

    Raises:
        RenderError: If npx is not on PATH.
    """
    candidates = ("npx.cmd", "npx") if os.name == "nt" else ("npx",)
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    raise RenderError("npx was not found; install Node.js 22.13 or later")


def browser_options(workdir: Path) -> list[str]:
    """Return mermaid-cli options selecting an installed browser, if configured."""
    browser = os.environ.get(BROWSER_VARIABLE, "").strip()
    if not browser:
        return []
    config = workdir / "puppeteer.json"
    # Installed Chrome and Edge builds support only the new headless mode.
    config.write_text(
        json.dumps({"executablePath": browser, "headless": True}), encoding="utf-8"
    )
    return ["--puppeteerConfigFile", str(config)]


def manifest(available: dict[str, Path]) -> dict[str, Any]:
    """Return the manifest describing the current sources and configs."""
    return {
        "mermaid_cli": MERMAID_CLI_VERSION,
        "scale": SCALE,
        "configs": {
            config: checksum(SOURCES / config)
            for config in sorted(theme.config for theme in THEMES)
        },
        "diagrams": {name: checksum(path) for name, path in available.items()},
    }


def sources() -> dict[str, Path]:
    """Return the diagram sources by name, in name order."""
    return {path.stem: path for path in sorted(SOURCES.glob("*.mmd"))}


def outputs(name: str) -> list[Path]:
    """Return the PNG files rendered from one source."""
    return [OUTPUT / f"{name}-{theme.name}.png" for theme in THEMES]


def checksum(path: Path) -> str:
    """Return the SHA-256 of a text file with line endings normalized to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def is_png(path: Path) -> bool:
    """Return whether a file starts with the PNG signature."""
    with path.open("rb") as handle:
        return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


def report(path: Path) -> None:
    """Print the size of a rendered PNG in CSS pixels."""
    with path.open("rb") as handle:
        header = handle.read(24)
    width, height = (value // SCALE for value in struct.unpack(">II", header[16:24]))
    wide = " (wide: text shrinks in the docs column)" if width > WIDE_DIAGRAM else ""
    print(f"rendered {relative(path)}: {width}x{height} px{wide}")


def relative(path: Path) -> str:
    """Return a repository-relative path with forward slashes."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
