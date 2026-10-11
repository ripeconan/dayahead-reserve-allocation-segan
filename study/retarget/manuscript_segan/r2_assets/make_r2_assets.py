"""Generate the R2 manuscript tables and figures from the admitted artifacts.

Inputs (READ-ONLY):
  analysis_r2/outputs/formal_r2/aggregate_summary.json
  analysis_r2/outputs/formal_r2/origin_YYYY/origin.json

Outputs (written next to this file / into the manuscript tree):
  tables/r2_tab1_service.tex ... tables/r2_tab5_cost.tex
  figures/fig12_service_w7.pdf, fig13_physical_response.pdf, fig14_cost_curves.pdf
  r2_assets/r2_input_manifest.json      (sha256 of every input read)
  r2_assets/r2_derived_values.json      (every derived number used in the paper)

No number is typed by hand: every table and figure cell is computed from the
admitted artifacts.  The script is deterministic and fails closed on a
fingerprint mismatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MS = HERE.parent                      # manuscript_segan
WORKSPACE = MS.parent                 # 11_SEGAN_retarget_2026-10-10
FORMAL = WORKSPACE / "analysis_r2" / "outputs" / "formal_r2"
AGG = FORMAL / "aggregate_summary.json"

EXPECTED_FINGERPRINT = "06d84da18dcd5355834f4844a104f2ee8936177e381d8f7894e75fa22c6ea322"
POLICIES = ["none", "P", "F1", "F2", "F3", "F4", "F5"]
COMPARATORS = ["none", "F1", "F2", "F3", "F4", "F5"]
KAPPA_GRID = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
ETA_GRID = [0.0, 0.5, 1.0, 2.0]

C_NONE = "#4d4d4d"
C_P = "#1f5fa8"
C_F = "#8aa9c9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load() -> tuple[dict, dict[int, dict], dict]:
    agg = json.loads(AGG.read_text(encoding="utf-8"))
    if agg["fingerprint"] != EXPECTED_FINGERPRINT:
        raise SystemExit(f"fingerprint mismatch: {agg['fingerprint']}")
    if not agg.get("grid_complete"):
        raise SystemExit("aggregate grid incomplete")
    origins = {}
    manifest = {"aggregate_summary.json": sha256(AGG)}
    for year in range(2019, 2026):
        path = FORMAL / f"origin_{year}" / "origin.json"
        origins[year] = json.loads(path.read_text(encoding="utf-8"))
        manifest[f"origin_{year}/origin.json"] = sha256(path)
    return agg, origins, manifest


def macro_map(agg: dict) -> dict[str, dict]:
    return {row["policy"]: row for row in agg["macro"]}


def contrast(agg: dict, name: str, endpoint: str, eta=None, kappa=None) -> dict:
    pool = agg["physical_contrasts"] if eta is not None else agg["contrasts"]
    for row in pool:
        if row["contrast"] != name or row["endpoint"] != endpoint:
            continue
        if eta is not None and float(row["eta"]) != float(eta):
            continue
        if kappa is not None and float(row["kappa"]) != float(kappa):
            continue
        return row
    raise KeyError(f"{name}/{endpoint}/eta={eta}/kappa={kappa}")


def phys_macro(agg: dict, policy: str, eta: float, kappa: float = 0.0) -> dict:
    for row in agg["physical_macro"]:
        if (row["policy"] == policy and float(row["eta"]) == float(eta)
                and float(row["kappa"]) == float(kappa)):
            return row
    raise KeyError(f"{policy}/eta={eta}/kappa={kappa}")


def f(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------- tables

def tab_service(mac: dict) -> str:
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Modeled coverage service on the seven exposed NYISO origins "
        r"(2019--2025) for every declared policy. $W_3$ and $W_7$ are the lowest "
        r"3- and 7-day mean service inside one segment; actions and blocked days "
        r"are annual means; the horizon cap is $B=55$ committed days per origin.}",
        r"\label{tab:r2-service}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Policy & Mean & $W_3$ & $W_7$ & Actions & Rate (\%) & Blocked & "
        r"Rescue & $L_1$ \\",
        r"\midrule",
    ]
    label = {"none": "none", "P": "P (production)", "F1": "F1 pace-7",
             "F2": "F2 bank-pace", "F3": "F3 two-low", "F4": "F4 persistence",
             "F5": "F5 declared-end bank"}
    for pol in POLICIES:
        row = mac[pol]
        rescue = row["macro_rescue_precision"]
        rescue_s = "--" if rescue != rescue else f(rescue, 4)
        lines.append(
            f"{label[pol]} & {f(row['macro_mean_service'])} & "
            f"{f(row['macro_service_worst3'])} & {f(row['macro_service_worst7'], 6)} & "
            f"{f(row['macro_action_total_all_declared'], 2)} & "
            f"{100 * row['macro_action_rate_all_declared']:.2f} & "
            f"{f(row['macro_blocked_trigger_days'], 2)} & {rescue_s} & "
            f"{f(row['macro_L_lambda_1'], 4)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\footnotesize Service is the modeled coverage score of "
        r"Eq.~\eqref{eq:service}; actions are committed days counted over the "
        r"declared calendar. ``--'' marks an undefined ratio (the no-commitment "
        r"policy has no committed day).",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def annual_counts(origins: dict[int, dict], comparator: str, endpoint: str) -> tuple[int, int, int]:
    """Strictly positive / tie / negative counts of the per-origin annual
    contrast P minus `comparator` on `endpoint`, read from the admitted
    per-origin payloads."""
    diffs = [origins[y]["allocator_metrics"]["P"][endpoint]
             - origins[y]["allocator_metrics"][comparator][endpoint]
             for y in sorted(origins)]
    pos = sum(1 for d in diffs if d > 1e-12)
    neg = sum(1 for d in diffs if d < -1e-12)
    return pos, len(diffs) - pos - neg, neg


def tab_contrasts(agg: dict, origins: dict[int, dict]) -> str:
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Paired annual $W_7$ contrasts of the production rule against "
        r"every comparator, seven exposed origins. Effects are candidate minus "
        r"comparator; intervals are 10{,}000-draw IID-over-origin bootstrap "
        r"percentiles (seed 20261011), LOO is the leave-one-origin-out range of "
        r"the mean, and the last two columns count strictly positive annual "
        r"effects and flag whether the mean effect reaches the predeclared "
        r"$0.01$ material-effect benchmark.}",
        r"\label{tab:r2-contrasts}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Contrast & Mean effect & 95\% interval & LOO range & Annual "
        r"$>0$ & Above $0.01$ \\",
        r"\midrule",
    ]
    for comp in COMPARATORS:
        row = contrast(agg, f"P_minus_{comp}", "service_worst7")
        loo = row["loo"]
        pos, ties, neg = annual_counts(origins, comp, "service_worst7")
        counts = f"{pos}/7" + (f" ({ties} tie)" if ties == 1 else
                               (f" ({ties} ties)" if ties else ""))
        above = "yes" if row["effect"] >= row["delta_material"] else "no"
        lines.append(
            f"P $-$ {comp} & {f(row['effect'])} & "
            f"[{f(row['ci_lo'])}, {f(row['ci_hi'])}] & "
            f"[{f(min(loo.values()))}, {f(max(loo.values()))}] & "
            f"{counts} & {above} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\footnotesize The annual column counts origins with a strictly "
        r"positive paired annual $W_7$ difference, not positive means or "
        r"intervals, and ties are exact zero differences; there is one negative "
        r"annual effect for pacing and one for bank pacing and none for the "
        r"other comparators. Intervals and leave-one-origin-out ranges are "
        r"descriptive precision statements for exposed data under the contract "
        r"of Section~\ref{sec:protocol}.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def tab_physical(mac: dict, agg: dict) -> str:
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Finite zonal reserve response at the declared multiplier "
        r"$\eta=1$ (dimensionless; it multiplies the dimensional reference "
        r"headroom $H_z$ in MW). Annual means over the seven exposed origins; "
        r"booked capacity is capacity-time, not generated electricity.}",
        r"\label{tab:r2-physical}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Uncovered & $W_7$ deficit & $W_7$ adequacy & "
        r"Extra booked & Committed \\",
        r"& (MWh/yr) & (MWh) & (fraction) & ($10^6$ MW$\cdot$h/yr) & days/yr \\",
        r"\midrule",
    ]
    label = {"none": "none", "P": "P (production)", "F1": "F1 pace-7",
             "F2": "F2 bank-pace", "F3": "F3 two-low", "F4": "F4 persistence",
             "F5": "F5 declared-end bank"}
    for pol in POLICIES:
        row = phys_macro(agg, pol, 1.0)
        lines.append(
            f"{label[pol]} & {f(row['macro_uncovered_MWh_finite_hours'], 1)} & "
            f"{f(row['macro_worst7_deficit_MWh'], 1)} & "
            f"{f(row['macro_worst7_adequacy_fraction'], 4)} & "
            f"{f(row['macro_extra_committed_MWh'] / 1e6, 4)} & "
            f"{f(mac[pol]['macro_action_total_all_declared'], 2)} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\footnotesize Uncovered energy integrates all finite issued "
        r"zone-hour rows; the worst-window endpoints use complete observed days. "
        r"Lower uncovered energy and lower deficit are better, higher adequacy is "
        r"better. The no-commitment row books none of the extra capacity.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def tab_phys_contrasts(agg: dict) -> str:
    rows = [
        ("P_minus_none", "P $-$ none", "uncovered_MWh_finite_hours",
         "Uncovered", "MWh/yr", 1.0, 2),
        ("P_minus_none", "P $-$ none", "worst7_deficit_MWh_complete_days",
         "$W_7$ deficit", "MWh", 1.0, 2),
        ("P_minus_none", "P $-$ none", "worst7_adequacy_fraction_complete_days",
         "$W_7$ adequacy", "fraction", 1.0, 4),
        ("P_minus_none", "P $-$ none", "extra_committed_MWh",
         "Extra booked", "$10^6$ MW$\\cdot$h/yr", 1e-6, 4),
        ("P_minus_F2", "P $-$ F2", "worst7_deficit_MWh_complete_days",
         "$W_7$ deficit", "MWh", 1.0, 2),
        ("P_minus_F2", "P $-$ F2", "uncovered_MWh_finite_hours",
         "Uncovered", "MWh/yr", 1.0, 2),
    ]
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Paired annual physical contrasts at $\eta=1$. Effects are "
        r"candidate minus comparator in the unit shown; intervals and the LOO "
        r"range are computed as in Table~\ref{tab:r2-contrasts}.}",
        r"\label{tab:r2-physical-contrasts}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Contrast & Endpoint & Unit & Effect & 95\% interval & LOO range \\",
        r"\midrule",
    ]
    for key, name, endpoint, label, unit, scale, digits in rows:
        row = contrast(agg, key, endpoint, eta=1.0, kappa=0.0)
        loo = row["loo"]
        lines.append(
            f"{name} & {label} & {unit} & {f(row['effect'] * scale, digits)} & "
            f"[{f(row['ci_lo'] * scale, digits)}, {f(row['ci_hi'] * scale, digits)}] & "
            f"[{f(min(loo.values()) * scale, digits)}, "
            f"{f(max(loo.values()) * scale, digits)}] \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\footnotesize A negative uncovered or deficit effect is better; a "
        r"positive adequacy effect is better. The $W_7$ deficit and adequacy "
        r"intervals touch zero on their favorable side, while the annual "
        r"uncovered reduction excludes zero; the $(P-F2)$ rows keep the same "
        r"orientation and show that the production rule leaves a larger $W_7$ "
        r"deficit than bank pacing. Inference follows Section~\ref{sec:protocol}.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def tab_cost(agg: dict) -> str:
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Normalized annual cost $J/c_D=\kappa\,(\text{base}+\text{extra "
        r"booked})+\text{uncovered}$ in MWh-equivalent units at $\eta=1$, over the "
        r"complete declared price grid. $\kappa=c_R/c_D$ is the capacity price per "
        r"MW$\cdot$h relative to the uncovered-energy price per MWh.}",
        r"\label{tab:r2-cost}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"$\kappa$ & none & P (production) & $P-$none \\",
        r"\midrule",
    ]
    for kappa in KAPPA_GRID:
        a = phys_macro(agg, "none", 1.0, kappa)
        b = phys_macro(agg, "P", 1.0, kappa)
        ca = a["macro_full_cost_MWh_equivalent"]
        cb = b["macro_full_cost_MWh_equivalent"]
        lines.append(f"{kappa:g} & {ca:,.0f} & {cb:,.0f} & {cb - ca:+,.0f} \\\\")
    crossing = {}
    for eta in (0.5, 1.0, 2.0):
        saved = (phys_macro(agg, "none", eta)["macro_uncovered_MWh_finite_hours"]
                 - phys_macro(agg, "P", eta)["macro_uncovered_MWh_finite_hours"])
        extra = phys_macro(agg, "P", eta)["macro_extra_committed_MWh"]
        crossing[eta] = saved / extra
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\footnotesize The crossing $\kappa^\star$ that equates the two "
        r"policies solves $\kappa^\star=\Delta\text{uncovered}/\Delta\text{booked}$: "
        f"${crossing[0.5]:.10f}$ at $\\eta=0.5$, ${crossing[1.0]:.10f}$ at "
        f"$\\eta=1$ and ${crossing[2.0]:.10f}$ at $\\eta=2$. It is a post-run "
        r"algebraic diagnostic of the fixed cost model, not a newly selected grid "
        r"cell: every declared positive price in the grid above favors the "
        r"no-extra-commitment comparator in this normalized objective.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- figures

def fig_service(agg: dict, mac: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    order = ["none", "F3", "F2", "F1", "F4", "F5", "P"]
    vals = [mac[p]["macro_service_worst7"] for p in order]
    colors = [C_NONE if p == "none" else (C_P if p == "P" else C_F) for p in order]
    ax.bar(range(len(order)), vals, color=colors, width=0.68)
    ax.axhline(mac["none"]["macro_service_worst7"], color=C_NONE, lw=0.8, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["none", "F3", "F2", "F1", "F4", "F5", "P"], fontsize=8)
    ax.set_ylabel(r"$W_7$: lowest 7-day mean service", fontsize=8)
    ax.set_ylim(0.5, 0.76)
    ax.tick_params(labelsize=8)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=7)
    ax.set_title("(a) worst-window modeled service by policy", fontsize=8.5)

    ax = axes[1]
    comps = ["none", "F1", "F2", "F3", "F4", "F5"]
    ys = np.arange(len(comps))[::-1]
    for y, comp in zip(ys, comps):
        row = contrast(agg, f"P_minus_{comp}", "service_worst7")
        ax.plot([row["ci_lo"], row["ci_hi"]], [y, y], color=C_P, lw=1.4)
        ax.plot([row["effect"]], [y], "o", color=C_P, ms=4)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(0.01, color="#999999", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"P $-$ {c}" for c in comps], fontsize=8)
    ax.set_xlabel(r"paired annual $W_7$ effect (95\% bootstrap interval)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("(b) production minus each comparator", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_physical(agg: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    order = ["none", "F3", "F1", "F4", "F5", "P", "F2"]
    labels = ["none", "F3", "F1", "F4", "F5", "P", "F2"]
    colors = [C_NONE if p == "none" else (C_P if p == "P" else C_F) for p in order]
    for ax, key, title in (
            (axes[0], "macro_uncovered_MWh_finite_hours",
             "(a) uncovered energy, $\\eta=1$"),
            (axes[1], "macro_worst7_deficit_MWh",
             "(b) worst-7 deficit, $\\eta=1$")):
        vals = [phys_macro(agg, p, 1.0)[key] for p in order]
        ax.bar(range(len(order)), vals, color=colors, width=0.68)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("MWh per year" if key.endswith("hours") else "MWh",
                      fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=8.5)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:,.0f}", (i, v), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_cost(agg: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axes[0]
    ks = np.array(KAPPA_GRID)
    none = np.array([phys_macro(agg, "none", 1.0, k)["macro_full_cost_MWh_equivalent"]
                     for k in KAPPA_GRID])
    prod = np.array([phys_macro(agg, "P", 1.0, k)["macro_full_cost_MWh_equivalent"]
                     for k in KAPPA_GRID])
    ax.plot(ks, none, "o-", color=C_NONE, ms=3.5, lw=1.2, label="none")
    ax.plot(ks, prod, "s-", color=C_P, ms=3.5, lw=1.2, label="P (production)")
    ax.set_yscale("log")
    ax.set_xlabel(r"capacity-to-deficit price ratio $\kappa=c_R/c_D$", fontsize=8)
    ax.set_ylabel(r"annual $J/c_D$ (MWh-equivalent)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("(a) complete declared price grid", fontsize=8.5)

    # exact affine cost difference: J/c_D = uncovered + kappa * booked capacity,
    # with the common base-booking term cancelling in the comparison.
    zero_none = phys_macro(agg, "none", 1.0, 0.0)
    zero_p = phys_macro(agg, "P", 1.0, 0.0)
    u_none = zero_none["macro_uncovered_MWh_finite_hours"]
    u_p = zero_p["macro_uncovered_MWh_finite_hours"]
    extra = zero_p["macro_extra_committed_MWh"]
    kstar = (u_none - u_p) / extra
    ax = axes[1]
    kk = np.linspace(0.0, 0.0015, 120)
    delta = (u_p - u_none) + kk * extra
    ax.plot(kk, delta, color="#333333", lw=1.3, label="P $-$ none")
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.axvline(kstar, color="#b03030", lw=0.9, ls="--")
    ax.axvline(0.001, color="#3a7d3a", lw=0.9, ls="-.")
    ax.annotate(rf"$\kappa^\star={kstar:.2e}$", (kstar, delta.max() * 0.15),
                fontsize=7, xytext=(4, 0), textcoords="offset points",
                color="#b03030")
    ax.annotate("smallest declared\npositive price 0.001", (0.001, delta.max() * 0.55),
                fontsize=7, xytext=(-6, 0), textcoords="offset points",
                color="#3a7d3a", ha="right")
    ax.set_xlim(0, 0.0015)
    ax.set_ylim(delta.min() * 1.15, delta.max() * 1.15)
    ax.set_xlabel(r"price ratio $\kappa$ (near-zero range)", fontsize=8)
    ax.set_ylabel(r"annual $(P-\text{none})$ in $J/c_D$", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("(b) cost difference and its zero, $\\eta=1$", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def sup_per_origin(origins: dict[int, dict]) -> str:
    label = {"none": "none", "P": "P", "F1": "F1", "F2": "F2", "F3": "F3",
             "F4": "F4", "F5": "F5"}
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Per-origin modeled endpoints for all seven declared policies.}"
        r"\label{sup:r2-perorigin}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Origin & Policy & Mean & $W_3$ & $W_7$ & Actions & Blocked \\",
        r"\midrule",
    ]
    for year in sorted(origins):
        metrics = origins[year]["allocator_metrics"]
        for pol in POLICIES:
            row = metrics[pol]
            lines.append(
                f"{year} & {label[pol]} & {f(row['mean_service'])} & "
                f"{f(row['service_worst3'])} & {f(row['service_worst7'], 6)} & "
                f"{int(row['action_total_all_declared'])} & "
                f"{int(row['blocked_trigger_days'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\footnotesize Higher mean, $W_3$ and $W_7$ are better. Actions and "
        r"blocked triggers are counts over that origin's declared calendar; the "
        r"cap is $B=55$ committed days.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def sup_contrasts(agg: dict, origins: dict[int, dict]) -> str:
    endpoints = [("mean_service", "mean"), ("service_worst3", "$W_3$"),
                 ("service_worst7", "$W_7$"), ("service_worst14", "$W_{14}$"),
                 ("L_lambda_1", "$L_1$")]
    lines = [
        r"{\scriptsize",
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\caption{Complete paired annual contrast set for modeled endpoints: "
        r"candidate minus comparator, 10{,}000-draw IID-over-origin bootstrap "
        r"(seed 20261011), leave-one-origin-out range of the mean, and the count "
        r"of strictly positive per-origin annual effects.}"
        r"\label{sup:r2-contrasts}\\",
        r"\toprule",
        r"Contrast & Endpoint & Mean effect & 95\% interval & LOO range & "
        r"Annual $>0$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Contrast & Endpoint & Mean effect & 95\% interval & LOO range & "
        r"Annual $>0$ \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for comp in COMPARATORS:
        for endpoint, name in endpoints:
            row = contrast(agg, f"P_minus_{comp}", endpoint)
            loo = row["loo"]
            pos, ties, neg = annual_counts(origins, comp, endpoint)
            counts = f"{pos}/7" + (f" ({ties} tie)" if ties == 1 else
                                   (f" ({ties} ties)" if ties else ""))
            lines.append(
                f"P $-$ {comp} & {name} & {f(row['effect'])} & "
                f"[{f(row['ci_lo'])}, {f(row['ci_hi'])}] & "
                f"[{f(min(loo.values()))}, {f(max(loo.values()))}] & "
                f"{counts} \\\\")
    lines += [
        r"\end{longtable}",
        r"}",
        r"\par\footnotesize Positive effects favor the production rule on mean, "
        r"$W_3$, $W_7$ and $W_{14}$; for $L_1$, which adds the abstract penalty "
        r"per committed day, a negative effect favors the production rule. The "
        r"annual column counts origins with a strictly positive paired annual "
        r"difference on that endpoint, so a positive mean, interval or "
        r"leave-one-origin-out range must not be read as every origin being "
        r"positive.",
    ]
    return "\n".join(lines) + "\n"


def sup_physical(agg: dict) -> str:
    lines = [
        r"{\scriptsize",
        r"\begin{longtable}{@{}llrrrr@{}}",
        r"\caption{Complete finite zonal reserve response grid by policy and "
        r"declared capacity multiplier $\eta$: annual means over seven exposed "
        r"origins.}"
        r"\label{sup:r2-physical-grid}\\",
        r"\toprule",
        r"Policy & $\eta$ & Uncovered & $W_7$ deficit & "
        r"$W_7$ adequacy & Extra booked \\",
        r"& & (MWh/yr) & (MWh) & (fraction) & ($10^6$ MW$\cdot$h/yr) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Policy & $\eta$ & Uncovered & $W_7$ deficit & "
        r"$W_7$ adequacy & Extra booked \\",
        r"& & (MWh/yr) & (MWh) & (fraction) & ($10^6$ MW$\cdot$h/yr) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for pol in POLICIES:
        for eta in ETA_GRID:
            row = phys_macro(agg, pol, eta)
            lines.append(
                f"{pol} & {eta:g} & "
                f"{f(row['macro_uncovered_MWh_finite_hours'], 1)} & "
                f"{f(row['macro_worst7_deficit_MWh'], 1)} & "
                f"{f(row['macro_worst7_adequacy_fraction'], 4)} & "
                f"{f(row['macro_extra_committed_MWh'] / 1e6, 4)} \\\\")
    lines += [
        r"\end{longtable}",
        r"}",
        r"\par\footnotesize Lower uncovered energy and lower $W_7$ deficit are "
        r"better; higher $W_7$ adequacy is better. The $\eta=0$ row of each "
        r"policy is identical by construction, because a policy books no extra "
        r"capacity without a commitment, so those rows restate the shared "
        r"no-commitment physical baseline. The table covers all 28 declared "
        r"policy and multiplier combinations; together with the nine declared "
        r"price ratios these are the 1{,}764 cells of the admitted "
        r"machine-readable grid.",
    ]
    return "\n".join(lines) + "\n"


def sup_cost(agg: dict) -> str:
    lines = [
        r"{\scriptsize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{none} & \multicolumn{2}{c}{P} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$\kappa$ & $\eta=1$ & $\eta=2$ & $\eta=1$ & $\eta=2$ \\",
        r"\midrule",
    ]
    for kappa in KAPPA_GRID:
        cells = []
        for pol in ("none", "P"):
            for eta in (1.0, 2.0):
                cells.append(phys_macro(agg, pol, eta, kappa)[
                    "macro_full_cost_MWh_equivalent"])
        lines.append(
            f"{kappa:g} & {cells[0]:,.0f} & {cells[1]:,.0f} & "
            f"{cells[2]:,.0f} & {cells[3]:,.0f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par\footnotesize Normalized annual cost $J/c_D$ in MWh-equivalent "
        r"units at the declared price ratios, for the two policies and the two "
        r"largest declared response multipliers; the $\eta=0.5$ and $\eta=0$ "
        r"columns and the remaining policies are in the machine-readable "
        r"aggregate. Higher $\kappa$ makes booked capacity relatively more "
        r"expensive.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    agg, origins, manifest = load()
    mac = macro_map(agg)
    tables = MS / "tables"
    figures = MS / "figures"

    outputs = {
        "tables/r2_tab1_service.tex": tab_service(mac),
        "tables/r2_tab2_contrasts.tex": tab_contrasts(agg, origins),
        "tables/r2_tab3_physical.tex": tab_physical(mac, agg),
        "tables/r2_tab4_physical_contrasts.tex": tab_phys_contrasts(agg),
        "tables/r2_tab5_cost.tex": tab_cost(agg),
        "tables/r2_sup_perorigin.tex": sup_per_origin(origins),
        "tables/r2_sup_contrasts.tex": sup_contrasts(agg, origins),
        "tables/r2_sup_physical.tex": sup_physical(agg),
        "tables/r2_sup_cost.tex": sup_cost(agg),
    }
    for rel, text in outputs.items():
        path = MS / rel
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(MS)}")

    fig_service(agg, mac, figures / "fig12_service_w7.pdf")
    fig_physical(agg, figures / "fig13_physical_response.pdf")
    fig_cost(agg, figures / "fig14_cost_curves.pdf")
    print("wrote figures fig12/fig13/fig14")

    derived = {
        "fingerprint": agg["fingerprint"],
        "macro_service": {p: {k: mac[p][k] for k in (
            "macro_mean_service", "macro_service_worst3", "macro_service_worst7",
            "macro_action_total_all_declared", "macro_action_rate_all_declared",
            "macro_blocked_trigger_days", "macro_rescue_precision",
            "macro_L_lambda_1")} for p in POLICIES},
        "physical_eta1": {p: {k: phys_macro(agg, p, 1.0)[k] for k in (
            "macro_uncovered_MWh_finite_hours", "macro_worst7_deficit_MWh",
            "macro_worst7_adequacy_fraction", "macro_extra_committed_MWh")}
            for p in POLICIES},
        "kappa_star": {str(eta): (
            phys_macro(agg, "none", eta)["macro_uncovered_MWh_finite_hours"]
            - phys_macro(agg, "P", eta)["macro_uncovered_MWh_finite_hours"])
            / phys_macro(agg, "P", eta)["macro_extra_committed_MWh"]
            for eta in (0.5, 1.0, 2.0)},
        "per_origin_w7": {str(y): {p: origins[y]["allocator_metrics"][p]["service_worst7"]
                                   for p in POLICIES} for y in origins},
        "per_origin_actions": {str(y): {p: origins[y]["allocator_metrics"][p]["action_total_all_declared"]
                                        for p in POLICIES} for y in origins},
    }
    derived_path = HERE / "r2_derived_values.json"
    derived_path.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8", newline="\n")

    # bind the generator itself and every generated artifact by SHA-256; the
    # manifest's own hash is excluded so the record has no self-reference.
    outputs_rel = list(outputs) + [
        "figures/fig12_service_w7.pdf",
        "figures/fig13_physical_response.pdf",
        "figures/fig14_cost_curves.pdf",
        "r2_assets/r2_derived_values.json",
    ]
    generated = {}
    for rel in outputs_rel:
        path = MS / rel
        generated[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest["generator"] = {
        "path": "r2_assets/make_r2_assets.py",
        "sha256": sha256(Path(__file__).resolve()),
    }
    manifest["generated_artifacts"] = generated
    manifest["fingerprint"] = agg["fingerprint"]
    (HERE / "r2_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    print(f"wrote r2_input_manifest.json ({len(generated)} artifact hashes) "
          f"and r2_derived_values.json")


if __name__ == "__main__":
    main()
