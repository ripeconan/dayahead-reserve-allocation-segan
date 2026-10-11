"""Hash helpers for R78 artifact binding."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    import json

    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(encoded)


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.to_csv(index=True, lineterminator="\n", float_format="%.17g")
    return sha256_text(normalized)


def array_sha256(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()
