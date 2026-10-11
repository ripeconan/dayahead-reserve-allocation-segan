#!/usr/bin/env python3
"""Verify prepared data files against the public acquisition manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "acquisition_manifest.json"


def digest(path: Path, compressed: bool = False) -> tuple[int, str]:
    value = hashlib.sha256()
    size = 0
    opener = gzip.open if compressed else Path.open
    if compressed:
        handle = opener(path, "rb")
    else:
        handle = opener(path, "rb")
    with handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            value.update(block)
    return size, value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]
    checked = 0
    missing: list[str] = []
    for record in sources.values():
        identity = record.get("prepared_artifact")
        path_value = identity.get("path") if identity else record.get("local_path")
        if not path_value:
            continue
        path = ROOT / path_value
        if not path.is_file():
            missing.append(path_value)
            continue
        expected = identity or record["source_artifact"]
        size, sha256 = digest(path)
        if size != expected["bytes"] or sha256 != expected["sha256"]:
            raise SystemExit(f"data identity mismatch: {path_value}")
        if path.name.endswith(".gz") and "uncompressed_sha256" in expected:
            content_size, content_sha256 = digest(path, compressed=True)
            if (
                content_size != expected["uncompressed_bytes"]
                or content_sha256 != expected["uncompressed_sha256"]
            ):
                raise SystemExit(f"data content mismatch: {path_value}")
        checked += 1
    if args.require_all and missing:
        raise SystemExit("missing prepared data:\n" + "\n".join(missing))
    print(f"verified {checked} prepared data files; {len(missing)} not present")


if __name__ == "__main__":
    main()
