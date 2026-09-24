"""Check the project's incremental Python size and Ruff limits.

Run ``python scripts/check_code_standard.py`` from any directory. The checked-in
baseline records existing debt; new files and functions have no exemption.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import shutil
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "code_standard_baseline.json"
FILE_LIMIT = 400
FUNCTION_LIMIT = 100


def tracked_python_files() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to list project Python files")
    # The executable is resolved explicitly and arguments do not use a shell.
    result = subprocess.run(  # noqa: S603
        [
            git,
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def function_lines(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    logical_lines = {
        token.end[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NEWLINE
    }
    lengths: dict[str, int] = {}

    def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, child.name))
                lengths[name] = sum(
                    child.lineno <= line <= child.end_lineno for line in logical_lines
                )
                visit(child, (*parents, child.name))
            elif isinstance(child, ast.ClassDef):
                visit(child, (*parents, child.name))
            else:
                visit(child, parents)

    visit(tree)
    return lengths


def measure_files(
    paths: list[Path],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    files: dict[str, int] = {}
    functions: dict[str, dict[str, int]] = {}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        files[name] = len(source.splitlines())
        functions[name] = function_lines(source)
    return files, functions


def ruff_findings(paths: list[Path]) -> dict[str, dict[str, int]]:
    # Paths come from git and are passed as arguments, never through a shell.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or result.stdout)
    counts: dict[str, Counter[str]] = {}
    for item in json.loads(result.stdout):
        name = Path(item["filename"]).relative_to(ROOT).as_posix()
        counts.setdefault(name, Counter())[item["code"]] += 1
    return {name: dict(sorted(rules.items())) for name, rules in sorted(counts.items())}


def create_baseline(paths: list[Path]) -> dict[str, object]:
    files, functions = measure_files(paths)
    return {
        "version": 1,
        "legacy_files": {
            name: count for name, count in files.items() if count > FILE_LIMIT
        },
        "legacy_functions": {
            name: {
                func: count for func, count in values.items() if count > FUNCTION_LIMIT
            }
            for name, values in functions.items()
            if any(count > FUNCTION_LIMIT for count in values.values())
        },
        "ruff_findings": ruff_findings(paths),
    }


def check_baseline(baseline: dict[str, object], paths: list[Path]) -> list[str]:
    files, functions = measure_files(paths)
    findings = ruff_findings(paths)
    legacy_files = baseline["legacy_files"]
    legacy_functions = baseline["legacy_functions"]
    legacy_findings = baseline["ruff_findings"]
    errors: list[str] = []

    for name, count in files.items():
        allowed = max(FILE_LIMIT, legacy_files.get(name, 0))
        if count > allowed:
            errors.append(f"{name}: {count} physical lines exceeds {allowed}")
        if name in legacy_files and count <= FILE_LIMIT:
            errors.append(f"{name}: remove obsolete file-size baseline entry")
        for func, length in functions[name].items():
            allowed = max(FUNCTION_LIMIT, legacy_functions.get(name, {}).get(func, 0))
            if length > allowed:
                errors.append(
                    f"{name}:{func}: {length} logical lines exceeds {allowed}"
                )
            if func in legacy_functions.get(name, {}) and length <= FUNCTION_LIMIT:
                errors.append(
                    f"{name}:{func}: remove obsolete function-size baseline entry"
                )
        for rule, count in findings.get(name, {}).items():
            allowed = legacy_findings.get(name, {}).get(rule, 0)
            if count > allowed:
                errors.append(
                    f"{name}: {rule} findings increased from {allowed} to {count}"
                )
    for name in legacy_files:
        if name not in files:
            errors.append(f"{name}: remove baseline entry for deleted file")
    for name, values in legacy_functions.items():
        for func in values:
            if func not in functions.get(name, {}):
                errors.append(
                    f"{name}:{func}: remove baseline entry for deleted function"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="write a new baseline; inspect the diff before committing",
    )
    args = parser.parse_args()
    paths = tracked_python_files()
    if args.create_baseline:
        BASELINE.write_text(
            json.dumps(create_baseline(paths), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {BASELINE.relative_to(ROOT)}")
        return 0
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline.get("version") != 1:
        parser.error("unsupported code standard baseline version")
    errors = check_baseline(baseline, paths)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Code standard passed for {len(paths)} tracked Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
