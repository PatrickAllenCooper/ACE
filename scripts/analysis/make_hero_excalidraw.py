#!/usr/bin/env python3
"""
Generate an Excalidraw hero figure for the ACE paper.

Layout (modeled on the LatentMAS-style two-tier reference):

  Top tier (no container):
    Tiny "ACE" label in top-left, then a horizontal outer-loop strip
    showing how each step flows into the next:
       state_t -> step -> state_{t+1} -> step -> state_{t+2} -> ...

  Bottom tier (dashed-bordered container):
    "One ACE step" label in top-left of the container, then the inner
    mechanism of a single step. All blocks are placed on a clean grid;
    arrows are routed to avoid crossing any node.

Output: paper/figs/ace_hero.excalidraw
"""

import json
import os
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Style: solid fills, slight hand-drawn roughness; lifted from the LatentMAS
# reference rather than the default Excalidraw "hachure".
# ---------------------------------------------------------------------------
ROUGHNESS = 1
FILL_STYLE = "solid"

# Soft pastel palette (LatentMAS-style):
COL_STATE     = ("#e2e8f0", "#475569")   # slate gray
COL_POLICY    = ("#c6f6d5", "#22863a")   # leaf green
COL_CAND      = ("#ffd6e0", "#c01048")   # rose
COL_LOOK      = ("#fed7aa", "#c2410c")   # amber
COL_BEST      = ("#fef3bd", "#a16207")   # gold
COL_WORST     = ("#fecdd3", "#be123c")   # ruby
COL_ENV       = ("#bee3f8", "#1e40af")   # sky blue
COL_LEARNER   = ("#fde68a", "#a16207")   # honey
COL_DPO       = ("#e9d8fd", "#6b21a8")   # violet
COL_RUNG_HI   = ("#ffd6e0", "#c01048")
COL_RUNG_LO   = ("#e2e8f0", "#64748b")
COL_DARK      = "#1f2937"
COL_MUTED     = "#64748b"


# ---------------------------------------------------------------------------
# Element factories
# ---------------------------------------------------------------------------
def _rand():
    return random.randint(1, 2_000_000_000)

def _now_ms():
    return int(time.time() * 1000)

def _id():
    return f"el-{_rand():010d}"

def _base(elem_type, x, y, w, h, stroke=COL_DARK, fill="transparent",
          stroke_width=2, fill_style=FILL_STYLE, roughness=ROUGHNESS,
          stroke_style="solid"):
    return {
        "id": _id(),
        "type": elem_type,
        "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": stroke, "backgroundColor": fill,
        "fillStyle": fill_style, "strokeWidth": stroke_width,
        "strokeStyle": stroke_style, "roughness": roughness, "opacity": 100,
        "groupIds": [], "frameId": None,
        "roundness": {"type": 3} if elem_type == "rectangle" else None,
        "seed": _rand(), "version": 1, "versionNonce": _rand(),
        "isDeleted": False, "boundElements": [],
        "updated": _now_ms(), "link": None, "locked": False,
    }

def box(x, y, w, h, fill_stroke, stroke_width=2, stroke_style="solid",
        rounded=True):
    fill, stroke = fill_stroke
    e = _base("rectangle", x, y, w, h, stroke=stroke, fill=fill,
              stroke_width=stroke_width, stroke_style=stroke_style)
    if not rounded:
        e["roundness"] = None
    return e

def container(x, y, w, h, stroke="#94a3b8", stroke_width=1.5):
    """Dashed bordered container with no fill (like the reference's outer box)."""
    e = _base("rectangle", x, y, w, h, stroke=stroke, fill="transparent",
              stroke_width=stroke_width, stroke_style="dashed")
    e["roundness"] = {"type": 3}
    return e

def text(x, y, w, h, content, size=18, color=COL_DARK, bold=False,
         align="center"):
    e = _base("text", x, y, w, h, stroke=color, fill="transparent",
              stroke_width=1)
    e["roundness"] = None
    e["text"] = content
    e["fontSize"] = size
    e["fontFamily"] = 2  # 2 = Helvetica (clean) like the reference
    e["textAlign"] = align
    e["verticalAlign"] = "middle"
    e["containerId"] = None
    e["originalText"] = content
    e["lineHeight"] = 1.25
    e["baseline"] = int(size * 0.85)
    return e

def line(x1, y1, x2, y2, color=COL_DARK, stroke_width=2, dashed=False,
         arrow_end=True, arrow_start=False):
    """Single-segment arrow / line."""
    a = _base("arrow", min(x1, x2), min(y1, y2),
              max(abs(x2 - x1), 1), max(abs(y2 - y1), 1),
              stroke=color, fill="transparent", stroke_width=stroke_width,
              stroke_style="dashed" if dashed else "solid")
    a["roundness"] = {"type": 2}
    a["points"] = [[0, 0], [x2 - x1, y2 - y1]]
    a["lastCommittedPoint"] = None
    a["startBinding"] = None
    a["endBinding"] = None
    a["startArrowhead"] = "arrow" if arrow_start else None
    a["endArrowhead"] = "arrow" if arrow_end else None
    a["x"] = x1
    a["y"] = y1
    a["width"] = x2 - x1
    a["height"] = y2 - y1
    return a

def polyline(points, color=COL_DARK, stroke_width=2, dashed=False,
             arrow_end=True):
    """Multi-segment orthogonal-friendly arrow."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = xs[0], ys[0]
    a = _base("arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
              stroke=color, fill="transparent", stroke_width=stroke_width,
              stroke_style="dashed" if dashed else "solid")
    a["roundness"] = {"type": 2}
    a["points"] = [[x - x0, y - y0] for x, y in points]
    a["lastCommittedPoint"] = None
    a["startBinding"] = None
    a["endBinding"] = None
    a["startArrowhead"] = None
    a["endArrowhead"] = "arrow" if arrow_end else None
    return a

def label_at(x, y, content, size=12, color=COL_MUTED, w=200, align="center"):
    return text(x - w / 2, y - 9, w, 18, content, size=size, color=color,
                align=align)


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
def build():
    elements = []

    # ============ TOP TIER: outer loop summary ==============================
    elements.append(text(40, 24, 160, 24, "ACE", size=18, bold=True,
                         color=COL_DARK, align="left"))

    # 5 small boxes in a row showing the loop: s_t -> step -> s_{t+1} -> step -> s_{t+2}
    top_y = 90
    top_h = 56
    state_w = 90
    step_w = 130
    gap = 36
    flow = []  # (label_text, color, x, w)
    cursor = 200
    items = [
        ("$s_t$",       state_w,  COL_STATE),
        ("ACE step",    step_w,   COL_POLICY),
        ("$s_{t+1}$",   state_w,  COL_STATE),
        ("ACE step",    step_w,   COL_POLICY),
        ("$s_{t+2}$",   state_w,  COL_STATE),
    ]
    for label, w, c in items:
        elements.append(box(cursor, top_y, w, top_h, c, stroke_width=1.8))
        elements.append(text(cursor, top_y, w, top_h, label, size=15))
        flow.append((cursor, w))
        cursor += w + gap

    # arrows between
    for (x_a, w_a), (x_b, _) in zip(flow[:-1], flow[1:]):
        elements.append(line(x_a + w_a + 2, top_y + top_h / 2,
                             x_b - 4, top_y + top_h / 2,
                             stroke_width=1.6, color=COL_MUTED))

    # ellipsis
    elements.append(text(cursor + 10, top_y, 80, top_h, "…", size=22,
                         color=COL_MUTED))

    elements.append(text(200, top_y + top_h + 12, 660, 18,
                         "outer loop: state evolves as the policy executes interventions",
                         size=12, color=COL_MUTED, align="left"))

    # ============ BOTTOM TIER: inside one ACE step ==========================
    # Outer dashed container
    cont_x, cont_y, cont_w, cont_h = 40, 220, 1500, 720
    elements.append(container(cont_x, cont_y, cont_w, cont_h))
    elements.append(text(cont_x + 16, cont_y + 14, 220, 22,
                         "Inside one ACE step", size=15, bold=True,
                         color=COL_DARK, align="left"))

    # ----------- (1) STATE block (top-left of container) -------------------
    s_x, s_y, s_w, s_h = 80, 290, 240, 150
    elements.append(box(s_x, s_y, s_w, s_h, COL_STATE, stroke_width=1.8))
    elements.append(text(s_x, s_y + 8, s_w, 24, "State  $s_t$",
                         size=16, bold=True))
    elements.append(text(s_x + 16, s_y + 38, s_w - 32, 22,
                         "• graph  G", size=13, align="left"))
    elements.append(text(s_x + 16, s_y + 64, s_w - 32, 22,
                         "• per-node losses  $\\{L_i\\}$",
                         size=13, align="left"))
    elements.append(text(s_x + 16, s_y + 90, s_w - 32, 22,
                         "• recent intervention history",
                         size=13, align="left"))
    elements.append(text(s_x + 16, s_y + 116, s_w - 32, 22,
                         "(formatted as a text prompt)",
                         size=11, align="left", color=COL_MUTED))

    # ----------- (2) POLICY block (right of state) -------------------------
    p_x, p_y, p_w, p_h = 380, 290, 280, 150
    elements.append(box(p_x, p_y, p_w, p_h, COL_POLICY, stroke_width=2.2))
    elements.append(text(p_x, p_y + 8, p_w, 24, "Policy  $\\pi_\\phi$",
                         size=16, bold=True))
    elements.append(text(p_x, p_y + 38, p_w, 22,
                         "Qwen2.5-1.5B (LM)", size=14))
    elements.append(text(p_x, p_y + 62, p_w, 22,
                         "+ DPO fine-tuning",
                         size=14, color=COL_DPO[1]))
    elements.append(text(p_x, p_y + 92, p_w, 22,
                         "generates $K=4$ candidates",
                         size=12, color=COL_MUTED))
    elements.append(text(p_x, p_y + 114, p_w, 22,
                         "as text:  DO  $V_i = \\nu$",
                         size=12, color=COL_MUTED))

    # State -> Policy
    elements.append(line(s_x + s_w + 2, s_y + s_h / 2,
                         p_x - 4, p_y + p_h / 2,
                         stroke_width=2))
    elements.append(text(s_x + s_w, s_y + s_h / 2 - 22,
                         p_x - (s_x + s_w), 18,
                         "prompt", size=12, color=COL_MUTED))

    # ----------- (3) K=4 CANDIDATES (right of policy) ----------------------
    cand_y = 300
    cand_box_h = 28
    cand_w = 110
    cand_xs = []
    base_cx = 730
    for i in range(4):
        cy = cand_y + i * (cand_box_h + 8)
        elements.append(box(base_cx, cy, cand_w, cand_box_h, COL_CAND,
                            stroke_width=1.5))
        elements.append(text(base_cx, cy, cand_w, cand_box_h,
                             f"$c_{{{i+1}}}$:  DO X={i}", size=12))
        cand_xs.append((base_cx, cy))
    # Policy -> candidates (single arrow, label "K=4")
    elements.append(line(p_x + p_w + 2, p_y + p_h / 2,
                         base_cx - 4, cand_y + (cand_box_h * 4 + 24) / 2,
                         stroke_width=2))
    elements.append(text(p_x + p_w + 4, p_y + p_h / 2 - 22, 80, 18,
                         "K=4", size=12, color=COL_MUTED))

    # ----------- (4) LOOKAHEAD block (right of candidates) -----------------
    l_x, l_y, l_w, l_h = 900, 290, 320, 150
    elements.append(box(l_x, l_y, l_w, l_h, COL_LOOK, stroke_width=2.2))
    elements.append(text(l_x, l_y + 8, l_w, 24,
                         "Lookahead", size=16, bold=True))
    elements.append(text(l_x, l_y + 36, l_w, 22,
                         "for each $c_k$:  clone learner,  execute,",
                         size=12))
    elements.append(text(l_x, l_y + 56, l_w, 22,
                         "compute  $\\Delta L(c_k) = L_{before} - L_{after}$",
                         size=12))
    elements.append(text(l_x, l_y + 86, l_w, 22,
                         "reward  $R = \\Delta L + \\alpha\\, w + \\gamma\\, D$",
                         size=13, color=COL_DARK))
    elements.append(text(l_x, l_y + 110, l_w, 22,
                         "(IG, node-importance, diversity)",
                         size=11, color=COL_MUTED))

    # candidates -> lookahead (single horizontal arrow, no overlap)
    cand_right_x = base_cx + cand_w + 2
    cand_mid_y = cand_y + (cand_box_h * 4 + 24) / 2
    elements.append(line(cand_right_x, cand_mid_y, l_x - 4, l_y + l_h / 2,
                         stroke_width=1.5, color=COL_MUTED))

    # ----------- (5) BEST / WORST below lookahead --------------------------
    bw_y = 480
    best_x, best_w = 920, 130
    worst_x, worst_w = 1080, 130
    elements.append(box(best_x, bw_y, best_w, 56, COL_BEST, stroke_width=2.4))
    elements.append(text(best_x, bw_y, best_w, 56, "BEST  $c^*$", size=16,
                         bold=True))
    elements.append(box(worst_x, bw_y, worst_w, 56, COL_WORST,
                        stroke_width=1.8))
    elements.append(text(worst_x, bw_y, worst_w, 56, "WORST  $c^-$",
                         size=16, bold=True))
    # lookahead -> best/worst
    elements.append(line(best_x + best_w / 2, l_y + l_h + 2,
                         best_x + best_w / 2, bw_y - 4,
                         stroke_width=1.8))
    elements.append(line(worst_x + worst_w / 2, l_y + l_h + 2,
                         worst_x + worst_w / 2, bw_y - 4,
                         stroke_width=1.8))
    # arg labels
    elements.append(text(best_x - 90, l_y + l_h + 14, 80, 18,
                         "argmax R", size=11, color=COL_MUTED, align="right"))
    elements.append(text(worst_x + worst_w + 8, l_y + l_h + 14, 90, 18,
                         "argmin R", size=11, color=COL_MUTED, align="left"))

    # ----------- (6) ENVIRONMENT and LEARNER (right edge column) -----------
    env_x, env_y, env_w, env_h = 1280, 290, 220, 100
    elements.append(box(env_x, env_y, env_w, env_h, COL_ENV, stroke_width=2.4))
    elements.append(text(env_x, env_y + 6, env_w, 22,
                         "Environment $M^*$", size=15, bold=True))
    elements.append(text(env_x, env_y + 32, env_w, 22,
                         "(ground truth SCM)", size=11, color=COL_MUTED))
    elements.append(text(env_x, env_y + 56, env_w, 22,
                         "do$(V_i = \\nu)$  →  samples", size=12))

    learn_x, learn_y, learn_w, learn_h = 1280, 470, 220, 100
    elements.append(box(learn_x, learn_y, learn_w, learn_h, COL_LEARNER,
                        stroke_width=2.4))
    elements.append(text(learn_x, learn_y + 6, learn_w, 22,
                         "Learner $M_\\theta$", size=15, bold=True))
    elements.append(text(learn_x, learn_y + 32, learn_w, 22,
                         "(student SCM)", size=11, color=COL_MUTED))
    elements.append(text(learn_x, learn_y + 56, learn_w, 22,
                         "trains on D", size=12))

    # BEST -> Environment (right-angle to avoid candidate boxes)
    elements.append(polyline(
        [(best_x + best_w / 2, bw_y + 56),
         (best_x + best_w / 2, env_y + env_h / 2),
         (env_x - 4, env_y + env_h / 2)],
        color=COL_BEST[1], stroke_width=2.4))
    elements.append(text(best_x + best_w / 2 + 6, bw_y + 70, 130, 18,
                         "execute c*", size=11, color=COL_BEST[1], align="left"))

    # Environment -> Learner (down)
    elements.append(line(env_x + env_w / 2, env_y + env_h + 2,
                         learn_x + learn_w / 2, learn_y - 4,
                         stroke_width=2))
    elements.append(text(env_x + env_w + 6, env_y + env_h + 22, 80, 18,
                         "data D", size=11, color=COL_MUTED, align="left"))

    # ----------- (7) DPO update block (bottom row) -------------------------
    d_x, d_y, d_w, d_h = 380, 620, 380, 130
    elements.append(box(d_x, d_y, d_w, d_h, COL_DPO, stroke_width=2.4))
    elements.append(text(d_x, d_y + 8, d_w, 24, "DPO Update", size=16,
                         bold=True, color=COL_DPO[1]))
    elements.append(text(d_x, d_y + 38, d_w, 22,
                         "preference pair: $(c^*, c^-)$",
                         size=13))
    elements.append(text(d_x, d_y + 64, d_w, 22,
                         "loss = $-\\log\\sigma(\\beta\\,[\\log\\pi_\\phi/\\pi_{ref}])$",
                         size=12))
    elements.append(text(d_x, d_y + 92, d_w, 22,
                         "$\\nabla_\\phi$ updates the policy",
                         size=12, color=COL_MUTED))

    # BEST -> DPO (curved down-left)
    elements.append(polyline(
        [(best_x + 20, bw_y + 56),
         (best_x + 20, d_y - 12),
         (d_x + 100, d_y - 12),
         (d_x + 100, d_y - 4)],
        color=COL_DPO[1], stroke_width=1.6))
    # WORST -> DPO
    elements.append(polyline(
        [(worst_x + worst_w - 20, bw_y + 56),
         (worst_x + worst_w - 20, d_y - 24),
         (d_x + 280, d_y - 24),
         (d_x + 280, d_y - 4)],
        color=COL_DPO[1], stroke_width=1.6))

    # DPO -> Policy (curved feedback to the left, dashed)
    elements.append(polyline(
        [(d_x + d_w / 2, d_y + d_h + 2),
         (d_x + d_w / 2, d_y + d_h + 26),
         (220, d_y + d_h + 26),
         (220, p_y + p_h + 32),
         (p_x + p_w / 2, p_y + p_h + 32),
         (p_x + p_w / 2, p_y + p_h + 4)],
        color=COL_DPO[1], stroke_width=2, dashed=True))
    elements.append(text(225, d_y + d_h + 8, 220, 18,
                         "policy gradient (dashed)",
                         size=11, color=COL_DPO[1], align="left"))

    # Learner -> State (close the data loop, curved up the left side)
    elements.append(polyline(
        [(learn_x + learn_w / 2, learn_y + learn_h + 2),
         (learn_x + learn_w / 2, learn_y + learn_h + 36),
         (cont_x + cont_w - 30, learn_y + learn_h + 36),
         (cont_x + cont_w - 30, s_y + s_h / 2),
         (s_x + s_w + 4, s_y + s_h / 2)],
        color=COL_MUTED, stroke_width=1.6, dashed=True))
    elements.append(text(cont_x + cont_w - 290, learn_y + learn_h + 18, 250, 18,
                         "next step (state advances)", size=11,
                         color=COL_MUTED, align="left"))

    # ============ Bottom note (outside container) ===========================
    elements.append(text(40, 960, 1500, 22,
                         "Key insight: $\\mathbb{E}[\\Delta L]$ collapses ~500x over training while candidate-rank Spearman $\\rho$ stays > 0.85; DPO depends only on rank.",
                         size=13, color=COL_MUTED, align="left"))

    return elements


def main():
    out_dir = Path("paper/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ace_hero.excalidraw"

    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": build(),
        "appState": {
            "gridSize": 20,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Elements: {len(payload['elements'])}")
    print()
    print("Open at https://excalidraw.com and drag the file onto the canvas.")
    print("File -> Export image -> PNG/SVG into paper/figs/ for use in paper.tex.")


if __name__ == "__main__":
    random.seed(42)
    main()
