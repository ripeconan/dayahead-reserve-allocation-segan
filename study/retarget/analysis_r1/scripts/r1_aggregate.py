"""R-A aggregation: macro tables, paired contrasts, bootstrap, LOO, context rows.

Descriptive exposed-data inference only: paired origin differences, IID-over-
origin bootstrap (fixed seed), leave-one-origin-out means.  No Holm family, no
confirmatory or equivalence language.  Context rows (random, COCD, oracle) are
labeled by their information contracts and never treated as deployable across
the board.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import r1_common as C  # noqa: E402
import r1_policies as P  # noqa: E402

POLICY_ORDER = ("P", "F1", "F2", "F3", "F4", "F5")
CONTRAST_KEYS = ("service_worst7", "mean_service", "L_lambda_1", "action_total")


def _load_results(out_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(out_root.glob("*.json")):
        if path.name.endswith("__fingerprint.json") or path.name == "allocation_run_meta.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload["meta"]
        for policy, metrics in payload["policies"].items():
            row = {"cohort": meta["cohort"], "origin": meta["origin"], "policy": policy, **metrics}
            rows.append(row)
    return pd.DataFrame(rows)


def _cocd_context(root: Path, nyiso: pd.DataFrame) -> pd.DataFrame:
    ledger = C.read_gzip_csv(root / C.R45_LEDGER)
    ledger["delivery_date"] = pd.to_datetime(ledger["delivery_date"], errors="raise")
    ledger = ledger.rename(columns={"candidate_action": "action"})
    merged = nyiso.merge(ledger[["origin", "delivery_date", "action", "segment_id"]],
                         on=["origin", "delivery_date"], suffixes=("", "_cocd"), how="inner")
    rows = []
    for origin, group in merged.groupby("origin", sort=True):
        group = group.sort_values("delivery_date", kind="mergesort").reset_index(drop=True)
        q = group["q"].to_numpy(float)
        actions = group["action"].to_numpy(bool)
        segments = P.normalize_segments(group["segment_id_cocd"].to_numpy(int))
        metrics = C.policy_metrics(q, actions, segments)
        rows.append({"origin": str(origin), **metrics})
    frame = pd.DataFrame(rows)
    frame["macro"] = False
    macro = {"origin": "MACRO", **C.macrorow(rows), "macro": True}
    macro["mean_actions_per_origin"] = float(np.mean([r["action_total"] for r in rows]))
    macro["action_total_sum"] = int(sum(int(r["action_total"]) for r in rows))
    macro["action_rate_macro"] = float(np.mean([r["action_rate"] for r in rows]))
    return pd.concat([frame, pd.DataFrame([macro])], ignore_index=True)


def _random_context(root: Path) -> dict:
    draws = C.read_gzip_csv(root / C.R45_RANDOM)
    per_origin = draws.groupby("origin")["service_worst7"].mean()
    return {
        "random_realized_count_mean": {
            "identity": "R45 count-conditioned random placement (non-deployable: conditioned on the realized DRB count)",
            "mean_over_draws_of_origin_macro_worst7": float(per_origin.mean()),
            "n_draws_per_origin": int(draws.groupby("origin").size().iloc[0]),
            "source": C.R45_RANDOM,
        }
    }


def run(root: Path, out_root: Path, report_out: Path) -> dict:
    nyiso = C.load_nyiso_daily(root)
    results = _load_results(out_root)
    payload: dict = {"cohorts": {}, "contrasts": {}}

    for cohort, group in results.groupby("cohort", sort=True):
        macro_rows: list[dict] = []
        macro_by_policy: dict[str, dict] = {}
        for policy in POLICY_ORDER:
            sub = group[group["policy"] == policy].sort_values("origin")
            if sub.empty:
                continue
            metrics = sub.drop(columns=["cohort", "origin", "policy"]).to_dict("records")
            macro = C.macrorow(metrics)
            # Unambiguous action fields (item 10): the macro row previously
            # carried an averaged "action_total"; keep both distinct estimands.
            mean_actions = macro.pop("action_total", None)
            macro["mean_actions_per_origin"] = float(mean_actions) if mean_actions is not None else None
            macro["action_total_sum"] = int(sum(int(row["action_total"]) for row in metrics))
            macro["action_rate_macro"] = float(np.mean([row["action_rate"] for row in metrics]))
            for lam in C.LAMBDA_FINE:
                macro[f"L_lambda_fine_{lam:g}"] = float(macro["mean_shortfall"] + lam * macro["action_rate_macro"])
            for key in ("blocked_trigger_count", "cap_skipped_scheduled_days",
                        "credit_dropped_on_segment_resets", "bank_open_at_trace_end",
                        "all_windows_have_action"):
                values = [row.get(key) for row in metrics]
                values = [v for v in values
                          if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if not values:
                    continue
                if all(isinstance(v, bool) for v in values):
                    macro[key] = values if key == "bank_open_at_trace_end" else bool(all(values))
                else:
                    macro[key] = int(sum(int(v) for v in values))
            macro_rows.append({"cohort": cohort, "policy": policy, **macro})
            macro_by_policy[policy] = macro
        payload["cohorts"][cohort] = {"macro": macro_rows}

        # Frozen fine-grid pairwise crossings (item 9): DL(l) = d_short + l*d_rate.
        crossings = []
        if "P" in macro_by_policy:
            base = macro_by_policy["P"]
            for policy in POLICY_ORDER[1:]:
                if policy not in macro_by_policy:
                    continue
                other = macro_by_policy[policy]
                d_short = float(base["mean_shortfall"] - other["mean_shortfall"])
                d_rate = float(base["action_rate_macro"] - other["action_rate_macro"])
                crossings.append({"contrast": f"P_minus_{policy}",
                                  "delta_mean_shortfall": d_short, "delta_action_rate": d_rate,
                                  **C.crossing_lambda(d_short, d_rate)})
        payload["cohorts"][cohort]["fine_grid_crossings"] = crossings

        # Paired contrasts P - F_k
        if cohort == "nyiso_exposed":
            cocd = _cocd_context(root, nyiso)
            payload["cohorts"][cohort]["context_cocd_macro"] = {
                k: v for k, v in cocd[cocd["macro"]].iloc[0].items() if k != "macro"
            }
            payload["cohorts"][cohort]["context_random"] = _random_context(root)
        contrasts = []
        for key in CONTRAST_KEYS:
            pivot = group.pivot_table(index="origin", columns="policy", values=key)
            for policy in POLICY_ORDER[1:]:
                if policy not in pivot or "P" not in pivot:
                    continue
                diff = (pivot["P"] - pivot[policy]).to_numpy(float)
                stat = C.paired_origin_bootstrap(diff)
                contrasts.append({
                    "cohort": cohort, "endpoint": key, "contrast": f"P_minus_{policy}",
                    **stat,
                    "delta_material": C.DELTA_MATERIAL,
                    "lo_o": C.leave_one_out(diff, pivot.index.tolist()),
                })
        payload["contrasts"][cohort] = contrasts

    C.write_json(report_out, payload)

    # Markdown summary
    lines = ["# R-A allocation-value results (descriptive, exposed replay)", ""]
    for cohort, block in payload["cohorts"].items():
        lines.append(f"## Cohort: {cohort}")
        lines.append("")
        lines.append("| policy | W3 | W7 | W14 | mean | mean_act/origin | act_sum | rate | rescue prec | L1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in block["macro"]:
            lines.append("| {policy} | {w3:.4f} | {w7:.4f} | {w14:.4f} | {mean:.4f} | {act} | {actsum} | {rate:.4f} | {rp:.4f} | {l1:.4f} |".format(
                policy=row["policy"], w3=row["service_worst3"], w7=row["service_worst7"],
                w14=row["service_worst14"], mean=row["mean_service"],
                act=row.get("mean_actions_per_origin", ""), actsum=row.get("action_total_sum", ""),
                rate=row.get("action_rate_macro", float("nan")),
                rp=row.get("rescue_precision", float("nan")), l1=row.get("L_lambda_1", float("nan"))))
        lines.append("")
        for row in block.get("fine_grid_crossings", []):
            if row["equal_action_rate"]:
                lines.append(f"- crossing {row['contrast']}: equal action rate; DL constant "
                             f"{row['delta_L_constant']:+.5f} ({row['sign']})")
            else:
                lines.append(f"- crossing {row['contrast']}: lambda* = {row['crossing_lambda']:.4f} "
                             f"(d_shortfall {row['delta_mean_shortfall']:+.5f}, d_rate {row['delta_action_rate']:+.5f})")
        lines.append("")
        for row in payload["contrasts"].get(cohort, []):
            lines.append(f"- {row['contrast']} on {row['endpoint']}: effect {row['effect']:+.5f}, "
                         f"descriptive CI [{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}] "
                         f"(positive-share {row['share_positive']:.3f})")
        lines.append("")
    Path(report_out).with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.EVIDENCE_ROOT)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "allocation_value")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "summary" / "r1_summary.json")
    args = parser.parse_args()
    payload = run(args.root, args.results, args.out)
    for cohort, block in payload["cohorts"].items():
        print(f"== {cohort} ==")
        for row in block["macro"]:
            print(f"  {row['policy']}: W7={row['service_worst7']:.4f} mean={row['mean_service']:.4f} "
                  f"actions={row.get('action_total')} rate={row.get('action_rate'):.4f} L1={row.get('L_lambda_1'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
