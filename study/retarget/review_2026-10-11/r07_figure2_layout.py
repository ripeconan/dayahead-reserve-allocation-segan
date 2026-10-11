"""R07 versioned generator: Figure 2 = end-to-end study-design schematic.

Output (single artefact):
    manuscript_segan/figures/fig11_endtoend_design.pdf

Why this generator exists (defect it repairs)
---------------------------------------------
The round-04a generator (``review_2026-10-11/r04a_figure_pipeline.py``, kept
untouched as the historical source) laid the schematic out in a 100 x 42 data
space and placed the three columns at x = 0.5..32.5, 34..66 and 67.5..99.5.
Two defects followed from that geometry:

1.  The horizontal connector corridors were 1.5 data units wide.  On the
    550.08 pt page that is 8.2 pt, but both rightward connectors were drawn
    across only 0.5 units (2.7 pt) *and* ``FancyArrowPatch`` was given
    ``shrinkA = shrinkB = 1.0`` pt, so the shaft was consumed entirely: the PDF
    contains two 0.40 pt shafts carrying 2.6 pt heads.  The figure is included
    with ``width=\\linewidth`` (388.54 pt), i.e. scaled by 0.7065, so on the
    page the connectors were a 0.28 pt stub with a 1.8 pt head.
2.  The two ``actions`` annotations were centred in the same 1.5-unit gap at
    x = 66.75 with a rendered width of 19.4 pt (3.3 data units), so each
    annotation overlaid the right edge of the commitment-menu box *and* the
    left edge of an evaluation box.

Design rules used by this rebuild
---------------------------------
*   The page is authored at exactly ``\\linewidth`` = 390.0 pt, so the
    ``\\includegraphics[width=\\linewidth]`` scale factor is 1.0 and every point
    measured here is a point on the printed page (no hidden shrink).
*   One data unit is one PDF point (72 units per inch on both axes), so box
    coordinates, text bounding boxes and clearances are directly comparable.
*   The columns are separated by explicit connector corridors: 24 pt on the
    left, 52 pt on the right.  The right corridor carries the action branch
    (trunk + vertical bus + two horizontal arms) and the two ``actions``
    labels in clear whitespace.
*   Text never overflows its box: every paragraph is wrapped against a
    renderer measurement, then re-measured after drawing.
*   After drawing, every text bounding box, box rectangle and connector is
    re-measured through the Agg renderer and checked against explicit
    clearance thresholds; the measurement is written to
    ``review_2026-10-11/figure2_fix/geometry_r07.json``.  Any violation raises.

The schematic carries no result: it is a design diagram and the scientific
content (boxes, headings, clock note) is the content of the round-04a figure,
re-wrapped only.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG_OUT = ROOT / "manuscript_segan" / "figures" / "fig11_endtoend_design.pdf"
FIX_DIR = HERE / "figure2_fix"
GEOM_OUT = FIX_DIR / "geometry_r07.json"
PNG_DIR = FIX_DIR / "after_png"

# ---------------------------------------------------------------- page geometry
# 390.0 pt is \linewidth of manuscript_segan/manuscript.tex (elsarticle preprint
# 12pt, one column), probed with figure2_fix/float_probe/probe.tex and consistent
# with the ink width of the full-width rules measured in the compiled PDF
# (388.54 pt of ink = 390.0 pt box minus the rule width).  Authoring at exactly
# \linewidth makes \includegraphics[width=\linewidth] a 1:1 inclusion, so every
# point below is a point on the printed page.
PAGE_W = 390.0
MARGIN_X = 1.6
CORRIDOR_L = 24.0          # column 1 -> column 2 connector corridor
CORRIDOR_R = 52.0          # column 2 -> column 3 action-branch corridor
TRUNK_LEN = 15.0           # menu box -> branch bus, measured from the corridor
                           # left edge; the remaining corridor carries the two
                           # horizontal arms and the two "actions" labels
# "actions" label baseline -> its arm line.  The label carries no descenders,
# so the ink gap equals LABEL_GAP; the text *metric* box that a PDF reader
# reports extends one descent below the baseline (0.236 em for DejaVu Sans), so
# LABEL_GAP >= LABEL_CLEAR + 0.236 * FS_LABEL keeps the reported box clear too.
LABEL_GAP = 5.0
MARGIN_Y = 2.0
COL_W = (PAGE_W - 2 * MARGIN_X - CORRIDOR_L - CORRIDOR_R) / 3.0
X1 = MARGIN_X
X2 = X1 + COL_W + CORRIDOR_L
X3 = X2 + COL_W + CORRIDOR_R

# ------------------------------------------------------------------- typography
FS_BOX = 6.6               # box body text (printed size; scale is 1.0)
LS_BOX = 1.42
PITCH_BOX = FS_BOX * LS_BOX
PAD_X = 3.2                # text inset inside a box
PAD_Y = 3.6
FS_TITLE = 7.2             # column headings (bold, two lines each)
PITCH_TITLE = 10.0
FS_LABEL = 6.4             # "actions" corridor labels
FS_NOTE = 6.2              # bottom clock note
PITCH_NOTE = 9.0

GAP_BOX = 12.0             # minimum vertical gap between stacked boxes
GAP_TITLE = 5.0            # heading block -> first box
GAP_NOTE = 11.0            # column block -> bottom note

# ------------------------------------------------------------------- connectors
LW = 0.9
MS = 8.5                   # mutation scale: head length = 0.4 * MS = 3.4 pt
SHRINK = 1.2
HEAD_LEN = 0.4 * MS

# ---------------------------------------------------------------------- colours
C_FORECAST = "#d7e5f7"
C_REGION = "#dcefdc"
C_ALLOC = "#fbe6cd"
C_EVAL = "#e8e2f3"
C_EDGE = "#333333"

# --------------------------------------------------------------------- thresholds
TEXT_IN_BOX_CLEAR = 1.0    # pt: text bbox must stay this far inside its box
LABEL_CLEAR = 2.0          # pt: "actions" label -> every box and connector
TITLE_CLEAR = 1.5          # pt: heading text -> neighbouring corridor rectangle
SHAFT_MIN = 5.0            # pt: visible connector shaft outside the arrow head
PAGE_CLEAR = 0.4           # pt: every drawn item inside the page

NBSP = " "            # glue: keeps a phrase on one line


def r3(v) -> float:
    """Round to 3 decimals as a plain float (matplotlib bboxes are numpy)."""
    return round(float(v), 3)


# ------------------------------------------------------------------------- data
@dataclass
class BoxSpec:
    """One rectangle of the schematic."""

    box_id: str
    face: str
    paras: list[str]       # paragraphs; each is wrapped independently
    lines: list[str] = None       # filled by the wrapper


@dataclass
class ColumnSpec:
    col_id: str
    title: str             # "1   Decision on $t-1$ after\nthe forecast release"
    x: float
    boxes: list[BoxSpec]


def build_columns() -> list[ColumnSpec]:
    """The scientific content of the round-04a figure, re-wrapped only.

    Every paragraph below is a line of the round-04a figure with the same
    wording; ``NBSP`` only glues a phrase that must not be split.
    """
    c1 = ColumnSpec(
        "col1",
        "1   Decision on $t-1$ after\nthe forecast release",
        X1,
        [
            BoxSpec("c1_forecast", C_FORECAST, [
                "official day-ahead forecast (published on $t-1$)",
                f"$+$ $72$ h / $168$ h realized-error{NBSP}lags",
                f"$+$ local-clock{NBSP}terms",
            ]),
            BoxSpec("c1_ridge", C_FORECAST, [
                "ridge correction, penalty $10$",
                f"training labels stop at midnight of $t-1$",
                "$\\Rightarrow$ issued point forecast",
            ]),
            BoxSpec("c1_thresholds", C_REGION, [
                f"reference year $Y{{-}}1$: OAS covariance $\\Sigma$,",
                "reference score prefix $\\Rightarrow$ slot thresholds",
                "(calendar-batched verified SAOCP kernel)",
            ]),
            BoxSpec("c1_regions", C_REGION, [
                f"frozen hourly joint{NBSP}regions",
                "$C_{t,h}\\subset\\mathbb{R}^{11}$",
            ]),
        ],
    )
    c2 = ColumnSpec(
        "col2",
        "2   Delayed feedback,\ndeclared budget",
        X2,
        [
            BoxSpec("c2_feedback", C_FORECAST, [
                "outcome of delivery day $u$ completes at its end",
                "$\\Downarrow$ released to the decision for delivery $u+2$",
                f"eligible post-action service{NBSP}history",
            ]),
            BoxSpec("c2_menu", C_ALLOC, [
                "commitment menu, method-owned histories",
                f"none{NBSP}{NBSP}$\\cdot${NBSP}{NBSP}P: shortfall sum, "
                f"$\\theta{{=}}1$, memory $6$",
                f"F1 pace-7{NBSP}{NBSP}$\\cdot${NBSP}{NBSP}F2 bank pace",
                f"F3 two-low{NBSP}{NBSP}$\\cdot${NBSP}{NBSP}F4 persistence",
                "F5 declared-end bank",
            ]),
            BoxSpec("c2_cap", C_ALLOC, [
                "declared cap $B=\\mathrm{round}(0.15\\,T_{\\mathrm{planned}})=55$",
                "nonreplenishable; blocked triggers recorded",
            ]),
        ],
    )
    c3 = ColumnSpec(
        "col3",
        "3   Evaluation on the\ndeclared calendar",
        X3,
        [
            BoxSpec("c3_coverage", C_EVAL, [
                f"modeled coverage{NBSP}service",
                "$s_t=a_t+(1-a_t)\\,q_t$,",
                # no space inside the math: a space would split the atom and
                # leave an unbalanced "$" that mathtext would render literally
                "$q_t=\\mathrm{mean}_h\\,\\mathbf{1}\\{y_{t,h}\\!\\in\\!C_{t,h}\\}$",
                f"mean, $W_3/W_7/W_{{14}}$, actions, rescue, $L_\\lambda$",
            ]),
            BoxSpec("c3_reserve", C_EVAL, [
                f"finite zonal reserve{NBSP}simulator",
                "$r^{0}_{z}=\\max(0,\\,\\mu_z+\\sqrt{R\\,\\Sigma_{zz}})$ MW",
                f"$+\\,\\eta H_z$ on committed{NBSP}days",
                "uncovered deficit (MWh), adequacy",
                f"worst-7 deficit, cost ratio $\\kappa$",
            ]),
        ],
    )
    return [c1, c2, c3]


NOTE_LINES = [
    "clock:  decision for delivery day $t$ on $t-1$, after the forecast release "
    "and before delivery   $\\rightarrow$",
    "outcome of $t$ completes at the end of $t$   $\\rightarrow$   "
    "first eligible for the decision for delivery $t+2$",
]


# ------------------------------------------------------------------- renderer
class Canvas:
    """A matplotlib figure whose data coordinates are PDF points.

    ``figsize = (PAGE_W / 72, height_in)`` with ``xlim = (0, PAGE_W)`` and
    ``ylim = (0, height_in * 72)`` makes one data unit exactly one point on the
    produced PDF page, in both axes.
    """

    def __init__(self, height_in: float):
        self.fig = plt.figure(figsize=(PAGE_W / 72.0, height_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(0, height_in * 72.0)
        self.ax.axis("off")
        self.fig.canvas.draw()
        self.renderer = self.fig.canvas.get_renderer()

    @property
    def height(self) -> float:
        return self.fig.get_size_inches()[1] * 72.0

    def close(self) -> None:
        plt.close(self.fig)

    # -- measurement ------------------------------------------------------
    def text_bbox(self, artist) -> Bbox:
        """Bounding box of a Text artist in PDF points (data coordinates)."""
        disp = artist.get_window_extent(renderer=self.renderer)
        return Bbox(self.ax.transData.inverted().transform_bbox(disp))

    def probe(self, s: str, fontsize: float, weight: str = "normal") -> tuple[float, float]:
        """Width/height of ``s`` in points without keeping the artist."""
        t = self.ax.text(0, 0, s, fontsize=fontsize, weight=weight)
        bb = self.text_bbox(t)
        t.remove()
        return bb.width, bb.height


def wrap_paragraph(canvas: Canvas, para: str, max_w: float) -> tuple[list[str], list[str]]:
    """Greedy wrap of one paragraph against renderer-measured widths.

    Atoms are separated by ordinary spaces; ``NBSP`` glues a phrase together.
    Returns (lines, overflows) where ``overflows`` lists single atoms that are
    wider than the available width (they are reported, never silently clipped).
    """
    atoms = para.split(" ")
    lines: list[str] = []
    overflows: list[str] = []
    cur = ""
    for atom in atoms:
        trial = atom if not cur else f"{cur} {atom}"
        if canvas.probe(trial, FS_BOX)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = atom
            if canvas.probe(atom, FS_BOX)[0] > max_w:
                overflows.append(atom)
    if cur:
        lines.append(cur)
    return lines, overflows


# ---------------------------------------------------------------------- layout
def run() -> None:
    columns = build_columns()
    inner_w = COL_W - 2 * PAD_X

    # ---- pass 1: wrap every paragraph and size the boxes -----------------
    probe = Canvas(4.5)
    overflows: list[dict] = []
    for col in columns:
        for box in col.boxes:
            lines: list[str] = []
            for para in box.paras:
                got, over = wrap_paragraph(probe, para, inner_w)
                lines.extend(got)
                for atom in over:
                    overflows.append({"box": box.box_id, "atom": atom,
                                      "width": r3(probe.probe(atom, FS_BOX)[0]),
                                      "avail": r3(inner_w)})
            box.lines = lines
    probe.close()

    box_h = {b.box_id: len(b.lines) * PITCH_BOX + 2 * PAD_Y
             for c in columns for b in c.boxes}

    # Column height: sum of boxes plus the minimum gaps.  The tallest column
    # sets the common block height; the slack of a shorter column is shared by
    # its gaps so that all three columns start and end on the same lines.
    col_sum = {c.col_id: sum(box_h[b.box_id] for b in c.boxes) for c in columns}
    # The tallest column (at the minimum gap) sets the common block height; the
    # other columns share their slack between their own gaps, so all three
    # columns start and end on the same two lines and every gap is >= GAP_BOX.
    block_h = max(col_sum[c.col_id] + (len(c.boxes) - 1) * GAP_BOX for c in columns)
    gap_of = {c.col_id: (block_h - col_sum[c.col_id]) / (len(c.boxes) - 1)
              for c in columns}

    note_h = len(NOTE_LINES) * PITCH_NOTE
    page_h = (MARGIN_Y + 2 * PITCH_TITLE + GAP_TITLE + block_h
              + GAP_NOTE + note_h + MARGIN_Y)

    # ---- pass 2: draw at the final page height ---------------------------
    canvas = Canvas(page_h / 72.0)
    ax = canvas.ax
    rects: dict[str, tuple[float, float, float, float]] = {}
    text_artists: list[dict] = []
    title_artists: list[dict] = []

    y_top = page_h - MARGIN_Y - 2 * PITCH_TITLE - GAP_TITLE

    for col in columns:
        # heading: two lines, block top at page_h - MARGIN_Y
        head_top = page_h - MARGIN_Y
        for i, line in enumerate(col.title.split("\n")):
            t = ax.text(col.x, head_top - (i + 0.5) * PITCH_TITLE, line,
                        fontsize=FS_TITLE, weight="bold", ha="left", va="center")
            title_artists.append({"col": col.col_id, "artist": t, "line": line})

        y = y_top
        for box in col.boxes:
            h = box_h[box.box_id]
            rect = (col.x, y - h, COL_W, h)
            rects[box.box_id] = rect
            ax.add_patch(FancyBboxPatch(
                (rect[0], rect[1]), rect[2], rect[3],
                boxstyle="round,pad=0,rounding_size=3.0",
                linewidth=0.7, edgecolor=C_EDGE, facecolor=box.face))
            t = ax.text(rect[0] + COL_W / 2, rect[1] + h / 2, "\n".join(box.lines),
                        ha="center", va="center", fontsize=FS_BOX,
                        linespacing=LS_BOX)
            text_artists.append({"id": box.box_id, "artist": t, "box": box.box_id})
            y -= h + gap_of[col.col_id]

    # ---- bottom clock note ----------------------------------------------
    note_top = MARGIN_Y + note_h
    for i, line in enumerate(NOTE_LINES):
        ax.text(MARGIN_X, note_top - (i + 0.5) * PITCH_NOTE - (PITCH_NOTE - FS_NOTE) / 2,
                line, fontsize=FS_NOTE, ha="left", va="center")

    # ---- connectors ------------------------------------------------------
    arrows: list[dict] = []
    segments: list[dict] = []          # axis-aligned connector segments

    def add_arrow(p, q, name, head=True):
        ax.add_patch(FancyArrowPatch(
            p, q, arrowstyle="-|>" if head else "-",
            mutation_scale=MS, linewidth=LW, color=C_EDGE,
            shrinkA=SHRINK, shrinkB=SHRINK, capstyle="butt"))
        length = math.dist(p, q)
        head_len = HEAD_LEN if head else 0.0
        arrows.append({
            "name": name, "p": [r3(v) for v in p], "q": [r3(v) for v in q],
            "length": r3(length), "head_len": r3(head_len),
            "shaft_len": r3(length - 2 * SHRINK - head_len),
            "head_width": r3(0.4 * MS if head else 0.0),
        })
        segments.append({"name": name, "p": [r3(v) for v in p],
                         "q": [r3(v) for v in q]})

    # vertical chain inside column 1 and column 2
    for col in columns[:2]:
        for upper, lower in zip(col.boxes, col.boxes[1:]):
            ux, uy, uw, uh = rects[upper.box_id]
            lx, ly, lw_, lh = rects[lower.box_id]
            cx = ux + uw / 2
            add_arrow((cx, uy), (cx, ly + lh), f"{upper.box_id}->{lower.box_id}")

    # column 1 -> column 2, entering the declared-budget box (round-04a endpoint)
    r1 = rects[columns[0].boxes[-1].box_id]
    r2 = rects[columns[1].boxes[-1].box_id]
    ov_lo, ov_hi = max(r1[1], r2[1]), min(r1[1] + r1[3], r2[1] + r2[3])
    y_link = 0.5 * (ov_lo + ov_hi)
    add_arrow((r1[0] + r1[2], y_link), (r2[0], y_link), "c1_regions->c2_cap")

    # action branch: commitment menu -> both evaluation boxes.
    menu = rects["c2_menu"]
    eval_top = rects["c3_coverage"]
    eval_bot = rects["c3_reserve"]
    flank_top = rects["c2_feedback"]
    flank_bot = rects["c2_cap"]

    def arm_y(target, flank):
        """Height of a horizontal arm: inside its target box and inside the
        column-2 box that flanks the corridor at that height."""
        lo = max(target[1] + 10.0, flank[1] + 10.0)
        hi = min(target[1] + target[3] - 10.0, flank[1] + flank[3] - 10.0)
        if lo <= hi:
            return 0.5 * (lo + hi)
        return target[1] + target[3] / 2

    y_arm_top = arm_y(eval_top, flank_top)
    y_arm_bot = arm_y(eval_bot, flank_bot)
    spine_x = X2 + COL_W + TRUNK_LEN
    menu_cy = menu[1] + menu[3] / 2

    add_arrow((menu[0] + menu[2], menu_cy), (spine_x, menu_cy), "menu->bus", head=False)
    add_arrow((spine_x, y_arm_top), (spine_x, y_arm_bot), "bus", head=False)
    add_arrow((spine_x, y_arm_top), (eval_top[0], y_arm_top), "bus->c3_coverage")
    add_arrow((spine_x, y_arm_bot), (eval_bot[0], y_arm_bot), "bus->c3_reserve")

    # ---- "actions" labels, in the corridor above each arm ----------------
    labels: list[dict] = []
    for name, y_arm, target in (("actions_top", y_arm_top, eval_top),
                                ("actions_bot", y_arm_bot, eval_bot)):
        lw_pt = canvas.probe("actions", FS_LABEL)[0]
        lx = spine_x + (target[0] - spine_x - lw_pt) / 2      # centred on the arm
        t = ax.text(lx, y_arm + LABEL_GAP, "actions", fontsize=FS_LABEL,
                    ha="left", va="bottom")
        labels.append({"id": name, "artist": t, "arm_y": y_arm})

    # ---- measurement pass ------------------------------------------------
    canvas.fig.canvas.draw()
    text_records = []
    for rec in text_artists:
        bb = canvas.text_bbox(rec["artist"])
        bx, by, bw, bh = rects[rec["box"]]
        text_records.append({
            "box": rec["box"],
            "bbox": [r3(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1)],
            "clearance_pt": {
                "left": r3(bb.x0 - bx),
                "right": r3(bx + bw - bb.x1),
                "bottom": r3(bb.y0 - by),
                "top": r3(by + bh - bb.y1),
            },
        })

    label_records = []
    for rec in labels:
        bb = canvas.text_bbox(rec["artist"])
        near = []
        for bid, (bx, by, bw, bh) in rects.items():
            near.append({"box": bid, "dist_pt": r3(bbox_rect_dist(bb, (bx, by, bw, bh)))})
        near.sort(key=lambda d: d["dist_pt"])
        seg_near = []
        for seg in segments:
            seg_near.append({"seg": seg["name"],
                             "dist_pt": r3(bbox_seg_dist(bb, seg["p"], seg["q"]))})
        seg_near.sort(key=lambda d: d["dist_pt"])
        label_records.append({
            "id": rec["id"],
            "text": "actions",
            "fontsize_pt": FS_LABEL,
            "bbox": [r3(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1)],
            "arm_y": r3(rec["arm_y"]),
            "nearest_boxes": near[:4],
            "nearest_connectors": seg_near[:4],
        })

    col_of = {c.col_id: c for c in columns}
    title_records = []
    for rec in title_artists:
        bb = canvas.text_bbox(rec["artist"])
        col = col_of[rec["col"]]
        corridor = CORRIDOR_L if rec["col"] == "col1" else CORRIDOR_R
        # aligned with the left edge of its own column, and clear of both the
        # page edge and the heading of the next column
        right_limit = min(col.x + COL_W + corridor, PAGE_W - MARGIN_X)
        title_records.append({
            "col": rec["col"],
            "text": rec["line"],
            "fontsize_pt": FS_TITLE,
            "bbox": [r3(v) for v in (bb.x0, bb.y0, bb.x1, bb.y1)],
            "align_left_pt": r3(bb.x0 - col.x),
            "room_right_pt": r3(right_limit - bb.x1),
        })
    heading_gaps = []
    for i, a in enumerate(title_records):
        for b in title_records[i + 1:]:
            if b["bbox"][1] < a["bbox"][3] and a["bbox"][1] < b["bbox"][3]:
                heading_gaps.append({"left": a, "right": b,
                                     "gap_pt": r3(b["bbox"][0] - a["bbox"][2])})

    # ---- checks ----------------------------------------------------------
    violations: list[str] = []
    for ov in overflows:
        violations.append(f"atom wider than the box ({ov['box']}): {ov['atom']!r} "
                          f"{ov['width']} pt > {ov['avail']} pt")
    for tr in text_records:
        worst = min(tr["clearance_pt"].values())
        if worst < TEXT_IN_BOX_CLEAR:
            violations.append(f"text outside its box ({tr['box']}): clearance {worst} pt")
    # The renderer's text box stops one descent above the baseline, but a PDF
    # reader reports the full metric box, so the horizontal box clearance is
    # checked against the measured bbox while the vertical arm clearance is
    # checked against LABEL_GAP minus the font descent (the conservative value).
    label_arm_gap = LABEL_GAP - 0.236 * FS_LABEL
    if label_arm_gap < LABEL_CLEAR:
        violations.append(f"LABEL_GAP {LABEL_GAP} pt leaves only {label_arm_gap} pt "
                          f"above an arm after the font descent")
    for lr in label_records:
        for nb in lr["nearest_boxes"][:2]:
            if nb["dist_pt"] < LABEL_CLEAR:
                violations.append(f"label {lr['id']} too close to {nb['box']}: "
                                  f"{nb['dist_pt']} pt")
        for nc in lr["nearest_connectors"]:
            if nc["dist_pt"] < LABEL_CLEAR:
                violations.append(f"label {lr['id']} too close to connector "
                                  f"{nc['seg']}: {nc['dist_pt']} pt")
    for lr in label_records:
        lr["conservative_arm_clearance_pt"] = r3(label_arm_gap)
    for tr in title_records:
        if tr["align_left_pt"] < -0.6 or tr["room_right_pt"] < TITLE_CLEAR:
            violations.append(f"heading {tr['text']!r} does not fit its column "
                              f"(left {tr['align_left_pt']}, "
                              f"right {tr['room_right_pt']} pt)")
    for hg in heading_gaps:
        if hg["gap_pt"] < TITLE_CLEAR:
            violations.append(f"headings {hg['left']['text']!r} / "
                              f"{hg['right']['text']!r} only {hg['gap_pt']} pt apart")
    for line in NOTE_LINES:
        if canvas.probe(line, FS_NOTE)[0] > PAGE_W - 2 * MARGIN_X:
            violations.append(f"note line wider than the page: {line[:40]!r}")
    for ar in arrows:
        if ar["shaft_len"] < SHAFT_MIN:
            violations.append(f"connector {ar['name']} shaft only {ar['shaft_len']} pt")
    for bid, (bx, by, bw, bh) in rects.items():
        if min(bx, by, PAGE_W - (bx + bw), page_h - (by + bh)) < PAGE_CLEAR:
            violations.append(f"box {bid} touches the page edge")

    # ---- artefacts -------------------------------------------------------
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    canvas.fig.savefig(FIG_OUT, format="pdf")
    canvas.fig.savefig(PNG_DIR / "fig11_endtoend_design_x3.png", dpi=216)
    canvas.fig.savefig(PNG_DIR / "fig11_endtoend_design_x2.png", dpi=144)

    geometry = {
        "generator": "review_2026-10-11/r07_figure2_layout.py",
        "supersedes": "review_2026-10-11/r04a_figure_pipeline.py",
        "figure": str(FIG_OUT),
        "page_pt": [r3(PAGE_W), r3(page_h)],
        "page_in": [round(PAGE_W / 72, 4), round(page_h / 72, 4)],
        "manuscript_linewidth_pt": PAGE_W,
        "inclusion_scale": 1.0,
        "columns": [{"id": c.col_id, "x": r3(c.x), "w": r3(COL_W)}
                    for c in columns],
        "corridors_pt": {"left": CORRIDOR_L, "right": CORRIDOR_R},
        "fonts_pt": {"box": FS_BOX, "title": FS_TITLE, "label": FS_LABEL,
                     "note": FS_NOTE},
        "boxes": {bid: [r3(v) for v in r] for bid, r in rects.items()},
        "box_lines": {b.box_id: b.lines for c in columns for b in c.boxes},
        "text_bboxes": text_records,
        "headings": title_records,
        "labels": label_records,
        "connectors": arrows,
        "note_lines": NOTE_LINES,
        "violations": violations,
    }
    GEOM_OUT.write_text(json.dumps(geometry, indent=1), encoding="utf-8")
    canvas.close()

    # ---- console summary -------------------------------------------------
    print(f"figure   : {FIG_OUT}")
    print(f"page     : {PAGE_W:.2f} x {page_h:.2f} pt "
          f"({PAGE_W/72:.3f} x {page_h/72:.3f} in), inclusion scale 1.0")
    print(f"columns  : w={COL_W:.2f} pt, corridors {CORRIDOR_L} / {CORRIDOR_R} pt")
    print(f"fonts    : box {FS_BOX} pt, title {FS_TITLE} pt, label {FS_LABEL} pt, "
          f"note {FS_NOTE} pt (printed size)")
    for c in columns:
        print(f"  {c.col_id}: {len(c.boxes)} boxes, gap {gap_of[c.col_id]:.2f} pt, "
              f"lines {[len(b.lines) for b in c.boxes]}")
    for ar in arrows:
        print(f"  connector {ar['name']:28s} len {ar['length']:6.2f} "
              f"shaft {ar['shaft_len']:6.2f} head {ar['head_len']:.2f}")
    for lr in label_records:
        nb = ", ".join(f"{d['box']}={d['dist_pt']}" for d in lr["nearest_boxes"][:2])
        nc = ", ".join(f"{d['seg']}={d['dist_pt']}" for d in lr["nearest_connectors"][:2])
        print(f"  label {lr['id']:12s} bbox {lr['bbox']} | boxes: {nb} | lines: {nc}")
    if violations:
        print("\nVIOLATIONS")
        for v in violations:
            print("  !", v)
        sys.exit(1)
    print("\nall geometry checks passed")


# ----------------------------------------------------------------- geometry aid
def bbox_rect_dist(bb: Bbox, rect: tuple[float, float, float, float]) -> float:
    """Euclidean distance between a text bbox and a rectangle (0 if overlapping)."""
    x, y, w, h = rect
    dx = max(0.0, max(x - bb.x1, bb.x0 - (x + w)))
    dy = max(0.0, max(y - bb.y1, bb.y0 - (y + h)))
    return math.hypot(dx, dy)


def bbox_seg_dist(bb: Bbox, p, q) -> float:
    """Distance between a text bbox and an axis-aligned segment (0 if crossing)."""
    (px, py), (qx, qy) = p, q
    if abs(px - qx) < 1e-6:                       # vertical segment
        sx0, sx1, sy0, sy1 = px, px, min(py, qy), max(py, qy)
    elif abs(py - qy) < 1e-6:                     # horizontal segment
        sx0, sx1, sy0, sy1 = min(px, qx), max(px, qx), py, py
    else:                                         # not used by this layout
        return 0.0
    dx = max(0.0, max(sx0 - bb.x1, bb.x0 - sx1))
    dy = max(0.0, max(sy0 - bb.y1, bb.y0 - sy1))
    return math.hypot(dx, dy)


if __name__ == "__main__":
    run()
