#!/usr/bin/env python3
"""Verify the frozen robustness analyses and replay configurable controllers."""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "artifacts" / "robustness"
sys.path.insert(0, str(ROOT / "src"))

from energy_uq.reliability_shield import (  # noqa: E402
    StrictLag2ReliabilityShield,
    rolling_worst_service,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_close(actual: float, expected: float, *, atol: float = 5e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"expected {expected}, found {actual}")


def require_pass(document: dict, name: str) -> None:
    if document.get("status") != "PASS":
        raise AssertionError(f"{name}: frozen verification status is not PASS")


def verify_frozen_objects() -> dict[str, int]:
    """Check shapes, statuses, and the numerical anchors reported in the paper."""
    expected_rows = {
        "production_transfer/aggregate_results.csv.gz": 12,
        "production_transfer/daily_trace.csv.gz": 8376,
        "production_transfer/paired_differences.csv.gz": 299,
        "production_transfer/paired_inference.csv.gz": 39,
        "production_transfer/per_origin_results.csv.gz": 92,
        "partial_effectiveness/aggregate_results.csv.gz": 72,
        "partial_effectiveness/paired_inference.csv.gz": 36,
        "partial_effectiveness/per_cell_results.csv.gz": 444,
        "action_cost/descriptive_uncertainty.csv.gz": 10,
        "action_cost/partial_response_break_even.csv.gz": 36,
        "action_cost/primary_break_even.csv.gz": 10,
        "action_cost/primary_units.csv.gz": 72,
        "inference_robustness/block_length_robustness.csv": 32,
        "inference_robustness/inference_contracts.csv": 7,
        "inference_robustness/leave_one_origin_out.csv": 40,
        "inference_robustness/recovery_stress.csv": 3,
        "hyperparameter_sensitivity/aggregate_grid.csv": 168,
        "hyperparameter_sensitivity/local_neighborhood.csv": 27,
        "hyperparameter_sensitivity/per_origin_grid.csv": 1176,
    }
    observed = {
        relative: len(read_csv(ROBUSTNESS / relative))
        for relative in expected_rows
    }
    if observed != expected_rows:
        raise AssertionError(f"robustness table shape mismatch: {observed}")

    floor = read_json(ROBUSTNESS / "construction_floor/verification.json")
    require_pass(floor, "construction floor")
    if floor["grid"]["case_count"] != 33 or floor["grid"]["mask_count"] != 3044:
        raise AssertionError("unexpected construction-floor grid")
    if floor["independent_finite_grid"]["trajectory_count"] != 5103:
        raise AssertionError("unexpected finite-grid trajectory count")
    cases = {row["case_id"]: row for row in floor["grid"]["cases"]}
    require_close(cases["production_L2_M6_W7_theta1"]["service_infimum"], 4.0 / 7.0)
    require_close(cases["lag1_L1_M6_W7_theta1"]["service_infimum"], 5.0 / 7.0)

    transfer = read_json(ROBUSTNESS / "production_transfer/verification.json")
    require_pass(transfer, "production transfer")
    if transfer["input_schema"]["origin_count"] != 23:
        raise AssertionError("unexpected transfer origin count")
    if transfer["controller"]["three_way_identity_origins"] != 23:
        raise AssertionError("controller identity did not cover every origin")
    blocked = sum(
        item["production_blocked_triggers"] for item in transfer["summary"].values()
    )
    if blocked != 498:
        raise AssertionError(f"expected 498 blocked triggers, found {blocked}")

    partial = read_json(ROBUSTNESS / "partial_effectiveness/verification.json")
    require_pass(partial, "partial effectiveness")
    if partial["grid"]["cell_count"] != 37:
        raise AssertionError("unexpected partial-effectiveness cell count")
    identities = partial["identity_checks"]
    if set(identities.values()) != {37}:
        raise AssertionError("partial-effectiveness endpoint identities failed")
    for base in partial["critical_eta"].values():
        for mode in base.values():
            require_close(mode["eta_for_worst7_gain_ge_0p01"], 0.25)
            if mode["eta075_worst7_gain"] <= 0.0 or mode["eta075_L1_delta"] <= 0.0:
                raise AssertionError("unexpected partial-effectiveness direction")

    cost = read_json(ROBUSTNESS / "action_cost/verification.json")
    require_pass(cost, "action cost")
    primary = read_csv(ROBUSTNESS / "action_cost/primary_break_even.csv.gz")
    lambdas = [float(row["lambda_star"]) for row in primary]
    if not (0.0178 < min(lambdas) < 0.0180 and 0.0842 < max(lambdas) < 0.0844):
        raise AssertionError("unexpected break-even cost range")
    if not all(float(row["delta_L_lambda_0.25"]) > 0.0 for row in primary):
        raise AssertionError("a primary stratum is favorable at lambda=0.25")

    comparator = read_json(
        ROBUSTNESS / "comparator_formalization/verification.json"
    )
    require_pass(comparator, "comparator formalization")
    cocd = comparator["r45_budget_conditioned_cocd"]
    astar = comparator["recovery_astar_registered_horizon"]
    if cocd["candidate_actions"] != 280 or cocd["contiguous_segments"] != 8:
        raise AssertionError("unexpected comparator action accounting")
    if cocd["deployable_without_budget_or_end_oracle"]:
        raise AssertionError("budget-conditioned comparator was misclassified")
    if astar["candidate_actions"] != 27 or not astar["zero_final_bank"]:
        raise AssertionError("unexpected registered-horizon accounting")

    inference = read_json(ROBUSTNESS / "inference_robustness/summary.json")
    require_pass(inference, "inference robustness")
    if not inference["block_length"]["all_32_relation_stable"]:
        raise AssertionError("a block-length relation changed")
    if inference["small_cluster"]["loo_sign_stable_rows"] != 34:
        raise AssertionError("unexpected leave-one-origin-out result")
    require_close(inference["recovery"]["c1_block2_holm_p"], 0.00704964751762412)
    require_close(inference["recovery"]["c1_block3_holm_p"], 0.0010499475026248687)
    require_close(inference["recovery"]["c1_signflip_holm_p"], 0.09375)

    sensitivity = read_json(
        ROBUSTNESS / "hyperparameter_sensitivity/summary.json"
    )
    require_pass(sensitivity, "hyperparameter sensitivity")
    full = sensitivity["full_grid_ranges"]
    local = sensitivity["local_neighborhood_ranges"]
    if (full["inactive_configs"], full["cap_pressured_configs"], full["cost_adverse_configs"]) != (52, 89, 168):
        raise AssertionError("unexpected full-grid regime counts")
    if (local["inactive_configs"], local["cap_pressured_configs"], local["cost_adverse_configs"]) != (1, 15, 27):
        raise AssertionError("unexpected local-grid regime counts")
    production = sensitivity["production_reference"]
    if (production["memory"], production["theta"], production["budget_frac"]) != (6, 1.0, 0.15):
        raise AssertionError("production configuration is not the frozen reference")
    require_close(production["worst7"], 0.7304421768707483)
    return observed


def verify_transfer_trace() -> dict[str, int]:
    """Replay the production controller on all exposed transfer origins."""
    rows = read_csv(ROBUSTNESS / "production_transfer/daily_trace.csv.gz")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["cohort"], row["origin"])].append(row)

    checked_days = 0
    checked_actions = 0
    for key, group in grouped.items():
        base = sorted(
            (row for row in group if row["policy"] == "no_fallback"),
            key=lambda row: int(row["decision_index"]),
        )
        target = sorted(
            (
                row
                for row in group
                if row["policy"] == "production_shortfall_sum_lag2"
            ),
            key=lambda row: int(row["decision_index"]),
        )
        if len(base) != len(target):
            raise AssertionError(f"transfer policy length mismatch for {key}")
        replay = StrictLag2ReliabilityShield(len(base)).replay(
            [float(row["base_covered"]) for row in base],
            [row["segment_id"] for row in base],
        )
        for result, expected in zip(replay, target):
            action = expected["action"] == "1"
            if result.action != action:
                raise AssertionError(
                    f"transfer action mismatch for {key} at index {result.index}"
                )
            require_close(result.service, float(expected["service"]), atol=1e-12)
            require_close(result.eligible_shortfall_sum, float(expected["state"]), atol=1e-12)
            if result.actions_used != int(expected["used_after"]):
                raise AssertionError("transfer action counter mismatch")
            checked_days += 1
            checked_actions += int(result.action)
    if len(grouped) != 23 or checked_days != 2094:
        raise AssertionError("transfer trace did not cover the frozen cohort")
    return {"origins": len(grouped), "days": checked_days, "actions": checked_actions}


def verify_hyperparameter_reference() -> dict[str, int]:
    """Recompute the seven production-reference rows from the base trace."""
    trace = read_csv(ROOT / "artifacts/verified_saocp/daily_policy_trace.csv.gz")
    base_by_origin: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trace:
        if row["layer"] == "base_no_fallback":
            base_by_origin[row["origin"]].append(row)
    for rows in base_by_origin.values():
        rows.sort(key=lambda row: row["delivery_date"])

    expected_rows = [
        row
        for row in read_csv(
            ROBUSTNESS / "hyperparameter_sensitivity/per_origin_grid.csv"
        )
        if (int(row["memory"]), float(row["theta"]), float(row["budget_frac"]))
        == (6, 1.0, 0.15)
    ]
    checked_actions = 0
    for expected in expected_rows:
        origin = expected["origin"]
        source = base_by_origin[origin]
        coverage = [float(row["base_daily_coverage"]) for row in source]
        segments = [row["segment_id"] for row in source]
        controller = StrictLag2ReliabilityShield(
            len(source),
            budget_fraction=float(expected["budget_frac"]),
            lookback=int(expected["memory"]),
            shortfall_threshold=float(expected["theta"]),
        )
        replay = controller.replay(coverage, segments)
        service = [row.service for row in replay]
        actions = [row.action for row in replay]
        action_total = sum(actions)
        blocked = sum(
            row.eligible_shortfall_sum >= controller.shortfall_threshold
            and not row.action
            and row.actions_used >= row.action_cap
            for row in replay
        )
        rescued = sum((1.0 - value) for value, action in zip(coverage, actions) if action)
        rescue_precision = (
            sum(value < 1.0 for value, action in zip(coverage, actions) if action)
            / action_total
            if action_total
            else 0.0
        )
        base_mean = sum(coverage) / len(coverage)
        mean_service = sum(service) / len(service)
        action_rate = action_total / len(service)
        metrics = {
            "base_mean_service": base_mean,
            "mean_service": mean_service,
            "mean_service_gain": mean_service - base_mean,
            "action_total": float(action_total),
            "action_rate": action_rate,
            "action_cap": float(controller.cap),
            "cap_utilization": action_total / controller.cap if controller.cap else 0.0,
            "blocked_triggers": float(blocked),
            "rescue_precision": rescue_precision,
            "rescued_shortfall": rescued,
            "base_L1": 1.0 - base_mean,
            "L1": 1.0 - mean_service + action_rate,
            "delta_L1": (1.0 - mean_service + action_rate) - (1.0 - base_mean),
        }
        for window in (3, 7, 14):
            base_worst = rolling_worst_service(coverage, segments, window)
            served_worst = rolling_worst_service(service, segments, window)
            metrics[f"base_worst{window}"] = base_worst
            metrics[f"worst{window}"] = served_worst
            metrics[f"worst{window}_gain"] = served_worst - base_worst
        for field, actual in metrics.items():
            if expected[field] == "":
                if field != "rescue_precision" or action_total != 0:
                    raise AssertionError(f"unexpected missing value in {field}")
                continue
            try:
                require_close(actual, float(expected[field]), atol=5e-12)
            except AssertionError as error:
                setting = (
                    origin,
                    expected["memory"],
                    expected["theta"],
                    expected["budget_frac"],
                )
                raise AssertionError(f"{field} mismatch for {setting}: {error}") from error
        checked_actions += action_total
    if len(base_by_origin) != 7 or len(expected_rows) != 7:
        raise AssertionError("reference replay did not cover all seven origins")
    return {
        "origins": len(base_by_origin),
        "configurations": len(expected_rows),
        "actions": checked_actions,
    }


def reproduce() -> dict[str, object]:
    return {
        "status": "PASS",
        "frozen_table_shapes": verify_frozen_objects(),
        "transfer_trace": verify_transfer_trace(),
        "hyperparameter_reference": verify_hyperparameter_reference(),
    }


def main() -> None:
    print(json.dumps(reproduce(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
