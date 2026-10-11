#!/usr/bin/env python3
"""Download and verify the fixed GEFCom2014 and CAISO source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "acquisition_manifest.json"


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, target: Path, expected_bytes: int, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "energy-uq-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    observed_bytes = temporary.stat().st_size
    observed_sha256 = sha256_file(temporary)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"download identity mismatch: bytes={observed_bytes}, sha256={observed_sha256}"
        )
    temporary.replace(target)


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
    archive.extractall(destination)


def acquire_gefcom(record: dict) -> None:
    directory = ROOT / "data" / "downloads" / "gefcom2014"
    outer_path = directory / record["source_archive"]["name"]
    download(
        record["download_url"],
        outer_path,
        record["source_archive"]["bytes"],
        record["source_archive"]["sha256"],
    )
    with zipfile.ZipFile(outer_path) as outer:
        if outer.testzip() is not None:
            raise RuntimeError("GEFCom2014 source archive failed CRC validation")
        nested = outer.read(record["load_track_archive"]["member"])
    if len(nested) != record["load_track_archive"]["bytes"]:
        raise RuntimeError("GEFCom2014 Load Track archive size mismatch")
    if hashlib.sha256(nested).hexdigest() != record["load_track_archive"]["sha256"]:
        raise RuntimeError("GEFCom2014 Load Track archive hash mismatch")
    nested_path = directory / "GEFCom2014-L_V2.zip"
    nested_path.write_bytes(nested)
    with zipfile.ZipFile(nested_path) as load_archive:
        if load_archive.testzip() is not None:
            raise RuntimeError("GEFCom2014 Load Track archive failed CRC validation")
        safe_extract(load_archive, directory)
    print(f"verified and extracted {outer_path.relative_to(ROOT)}")


def acquire_caiso(record: dict) -> None:
    target = ROOT / record["local_path"]
    source = record["source_artifact"]
    download(record["download_url"], target, source["bytes"], source["sha256"])
    print(f"verified {target.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gefcom2014", "caiso", "all"), default="all")
    args = parser.parse_args()
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"]
    if args.dataset in {"gefcom2014", "all"}:
        acquire_gefcom(sources["gefcom2014_load"])
    if args.dataset in {"caiso", "all"}:
        acquire_caiso(sources["caiso_zenodo_15034924"])


if __name__ == "__main__":
    main()
