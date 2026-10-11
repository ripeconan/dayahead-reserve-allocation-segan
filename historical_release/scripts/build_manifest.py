#!/usr/bin/env python3
"""Create or verify the SHA-256 manifest for frozen scientific artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MANIFEST = ARTIFACTS / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_entries() -> list[dict[str, object]]:
    entries = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if path.is_file() and path != MANIFEST:
            entries.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return entries


def verify() -> None:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = {item["path"]: item for item in current_entries()}
    declared = {item["path"]: item for item in expected["artifacts"]}
    if actual != declared:
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        changed = sorted(
            path
            for path in set(actual) & set(declared)
            if actual[path] != declared[path]
        )
        raise SystemExit(
            f"artifact manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    print(f"verified {len(actual)} frozen artifacts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    payload = {"schema": "artifact-manifest-v1", "artifacts": current_entries()}
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(payload['artifacts'])} entries")


if __name__ == "__main__":
    main()

