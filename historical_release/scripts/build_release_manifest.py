#!/usr/bin/env python3
"""Create or verify a SHA-256 inventory of the candidate release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.json"
EXCLUDED_PARTS = {".git", ".pytest_cache", ".venv", "venv", "__pycache__", "derived"}


def iter_release_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == OUTPUT:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory() -> dict:
    files = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in iter_release_files()
    ]
    return {
        "schema": "anonymous-reproducibility-release-manifest-v1",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    current = inventory()
    if args.verify:
        expected = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if current != expected:
            raise SystemExit("release manifest verification failed")
        print(f"verified {current['file_count']} release files")
        return
    OUTPUT.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print(f"wrote manifest for {current['file_count']} release files")


if __name__ == "__main__":
    main()
