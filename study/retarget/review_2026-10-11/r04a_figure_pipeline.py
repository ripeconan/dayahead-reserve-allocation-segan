"""R04A figure helper: end-to-end study-design schematic (vector PDF).

Three-column overview of the frozen R2 study design:

  (1) date-available forecast/region construction at the day-ahead cutoff,
  (2) delayed feedback and the declared-budget commitment menu,
  (3) the two evaluation layers: modeled coverage service and the finite
      zonal balancing-reserve simulator.

The figure is a design schematic: it carries no result and no numeric
outcome.  Output: manuscript_segan/figures/fig11_endtoend_design.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "manuscript_segan" / "figures" / "fig11_endtoend_design.pdf"

W, H = 7.6, 3.1
FS = 5.9          # box text
FS_T = 6.8        # column titles
FS_S = 5.4        # annotations

C_FORECAST = "#d7e5f7"
C_REGION = "#dcefdc"
C_ALLOC = "#fbe6cd"
C_EVAL = "#e8e2f3"
C_EDGE = "#333333"

X1, X2, X3 = 0.5, 34.0, 67.5
CW = 32.0


def box(ax, x, y, w, h, text, face, fs=FS):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=0.7, edgecolor=C_EDGE, facecolor=face))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.45)


def arrow(ax, p, q, rad=0.0, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5,
                                 linewidth=lw, color=C_EDGE, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.0, shrinkB=1.0))


def main() -> None:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    # ------------------------------------------------------------- column 1
    ax.text(X1, 39.6, "1   Decision on $t-1$ after the forecast release",
            fontsize=FS_T, weight="bold", ha="left", va="center")
    box(ax, X1, 32.0, CW, 5.6,
        "official day-ahead forecast (published on $t-1$)\n$+$ $72$ h / $168$ h realized-error lags\n$+$ local-clock terms", C_FORECAST)
    box(ax, X1, 23.4, CW, 6.4,
        "ridge correction, penalty $10$\ntraining labels stop at midnight of $t-1$\n"
        "$\\Rightarrow$ issued point forecast", C_FORECAST)
    box(ax, X1, 14.8, CW, 6.4,
        "reference year $Y{-}1$: OAS covariance $\\Sigma$,\n"
        "reference score prefix $\\Rightarrow$ slot thresholds\n(calendar-batched verified SAOCP kernel)", C_REGION)
    box(ax, X1, 6.6, CW, 5.6,
        "frozen hourly joint regions\n$C_{t,h}\\subset\\mathbb{R}^{11}$", C_REGION)
    for y0 in (32.0, 23.4, 14.8):
        arrow(ax, (X1 + CW / 2, y0), (X1 + CW / 2, y0 - 2.2))

    # ------------------------------------------------------------- column 2
    ax.text(X2, 39.6, "2   Delayed feedback, declared budget",
            fontsize=FS_T, weight="bold", ha="left", va="center")
    box(ax, X2, 30.2, CW, 7.4,
        "outcome of delivery day $u$ completes at its end\n"
        "$\\Downarrow$ released to the decision for delivery $u+2$\n"
        "eligible post-action service history", C_FORECAST)
    box(ax, X2, 15.4, CW, 11.6,
        "commitment menu, method-owned histories\n"
        "none   $\\cdot$   P: shortfall sum, $\\theta{=}1$, memory $6$\n"
        "F1 pace-7   $\\cdot$   F2 bank pace\n"
        "F3 two-low   $\\cdot$   F4 persistence\n"
        "F5 declared-end bank", C_ALLOC)
    box(ax, X2, 6.6, CW, 6.2,
        "declared cap $B=\\mathrm{round}(0.15\\,T_{\\mathrm{planned}})=55$\n"
        "nonreplenishable; blocked triggers recorded", C_ALLOC)
    arrow(ax, (X2 + CW / 2, 30.2), (X2 + CW / 2, 27.6))
    arrow(ax, (X2 + CW / 2, 15.4), (X2 + CW / 2, 13.2))
    arrow(ax, (X1 + CW + 0.5, 9.4), (X2 - 0.5, 9.4))

    # ------------------------------------------------------------- column 3
    ax.text(X3, 39.6, "3   Evaluation on the declared calendar",
            fontsize=FS_T, weight="bold", ha="left", va="center")
    box(ax, X3, 24.6, 32.0, 12.0,
        "modeled coverage service\n"
        "$s_t=a_t+(1-a_t)\\,q_t$,\n"
        "$q_t=\\mathrm{mean}_h\\,\\mathbf{1}\\{y_{t,h}\\!\\in\\! C_{t,h}\\}$\n"
        "mean, $W_3/W_7/W_{14}$, actions, rescue, $L_\\lambda$", C_EVAL)
    box(ax, X3, 6.6, 32.0, 15.4,
        "finite zonal reserve simulator\n"
        "$r^{0}_{z}=\\max(0,\\,\\mu_z+\\sqrt{R\\,\\Sigma_{zz}})$ MW\n"
        "$+\\,\\eta H_z$ on committed days\n"
        "uncovered deficit (MWh), adequacy\n"
        "worst-7 deficit, cost ratio $\\kappa$", C_EVAL)
    arrow(ax, (X2 + CW + 0.5, 30.6), (X3 - 0.5, 30.6))
    arrow(ax, (X2 + CW + 0.5, 14.0), (X3 - 0.5, 14.0))
    ax.text((X2 + CW + X3) / 2, 31.4, "actions", fontsize=FS_S, ha="center", va="bottom")
    ax.text((X2 + CW + X3) / 2, 14.8, "actions", fontsize=FS_S, ha="center", va="bottom")

    ax.text(X1, 2.0,
            "clock:  decision for delivery day $t$ on $t-1$, after the forecast release and before delivery   $\\rightarrow$   "
            "outcome of $t$ completes at the end of $t$   $\\rightarrow$   first eligible for the decision for delivery $t+2$",
            fontsize=FS_S, ha="left", va="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
