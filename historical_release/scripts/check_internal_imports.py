#!/usr/bin/env python3
"""Verify that every internal pipeline import resolves inside the release."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINES = ROOT / "pipelines"
PACKAGE = "card10_energy_baselines"


def main() -> None:
    files = sorted(PIPELINES.rglob("*.py"))
    available: set[str] = set()
    metadata: dict[Path, tuple[str, bool]] = {}
    for path in files:
        parts = list(path.relative_to(PIPELINES).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        module = ".".join(parts)
        available.add(module)
        metadata[path] = (module, is_package)

    missing: list[str] = []
    for path, (module, is_package) in metadata.items():
        package = module if is_package else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                relative = "." * node.level + (node.module or "")
                resolved = importlib.util.resolve_name(relative, package)
            else:
                resolved = node.module or ""
            if resolved.startswith(PACKAGE) and resolved not in available:
                relative_path = path.relative_to(ROOT).as_posix()
                missing.append(f"{relative_path}:{node.lineno}: {resolved}")

    if missing:
        raise SystemExit("missing internal imports:\n" + "\n".join(missing))
    print(f"internal import check passed for {len(files)} Python files")


if __name__ == "__main__":
    main()
