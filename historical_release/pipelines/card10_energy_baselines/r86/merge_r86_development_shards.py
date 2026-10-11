#!/usr/bin/env python3
"""Fail-closed merge of isolated R86 candidate shards into one model package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, obj: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".merge_tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge R86 development shards")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    result_path = destination / "development_results.json"
    merged = json.loads(result_path.read_text(encoding="utf-8"))
    merged_ids = []
    for shard in [path.resolve() for path in args.shards]:
        shard_result_path = shard / "development_results.json"
        shard_result = json.loads(shard_result_path.read_text(encoding="utf-8"))
        for binding in (
            "candidate_grid_file_sha256",
            "candidate_grid_canonical_sha256",
            "input_sha256",
            "split",
            "access_flags",
        ):
            if shard_result[binding] != merged[binding]:
                raise RuntimeError(f"shard binding mismatch for {binding}: {shard}")
        if len(shard_result["candidates"]) != 1:
            raise RuntimeError(f"shard must contain exactly one candidate: {shard}")
        candidate_id, record = next(iter(shard_result["candidates"].items()))
        if record.get("status") == "PASS":
            bindings = (
                (
                    "development_prediction",
                    "development_prediction_sha256",
                    "development_predictions",
                ),
                ("checkpoint", "checkpoint_sha256", "development_checkpoints"),
            )
        elif record.get("status") == "FAIL":
            bindings = (("failure_log", "failure_log_sha256", "failure_logs"),)
        else:
            raise RuntimeError(f"refuse non-terminal shard: {candidate_id}")
        for record_key, hash_key, subdir in bindings:
            source = shard / record[record_key]
            expected_hash = record[hash_key]
            if sha256_file(source) != expected_hash:
                raise RuntimeError(f"source hash mismatch: {source}")
            target = destination / subdir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256_file(target) != expected_hash:
                raise RuntimeError(f"destination collision: {target}")
            if not target.exists():
                shutil.copy2(source, target)
            if sha256_file(target) != expected_hash:
                raise RuntimeError(f"post-copy hash mismatch: {target}")
        prior = merged["candidates"].get(candidate_id)
        if prior is not None and prior != record:
            raise RuntimeError(f"candidate collision with different record: {candidate_id}")
        merged["candidates"][candidate_id] = record
        merged_ids.append(candidate_id)
    write_json_atomic(result_path, merged)
    manifest = {
        "schema": "card10_r86_development_shard_merge_v1",
        "status": "PASS",
        "merged_candidates": merged_ids,
        "development_results_sha256": sha256_file(result_path),
        "source_shards": [
            {
                "path": str(path.resolve()),
                "development_results_sha256": sha256_file(path / "development_results.json"),
            }
            for path in args.shards
        ],
    }
    write_json_atomic(destination / "shard_merge_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
