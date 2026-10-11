"""Load frozen JSON contracts from the package contracts directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACTS_DIR = Path(__file__).resolve().parent / "contracts"


def load_json_contract(name: str) -> dict[str, Any]:
    path = CONTRACTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"contract missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_candidate_grid() -> dict[str, Any]:
    return load_json_contract("candidate_grid_v1.json")


def load_split_contract() -> dict[str, Any]:
    return load_json_contract("split_and_information_contract_v1.json")


def load_model_registry() -> dict[str, Any]:
    return load_json_contract("model_registry_v1.json")
