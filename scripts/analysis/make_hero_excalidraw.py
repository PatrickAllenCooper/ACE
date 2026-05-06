#!/usr/bin/env python3
"""
Generate an Excalidraw hero figure for the ACE paper.

Output: paper/figs/ace_hero.excalidraw  (importable at excalidraw.com)

The figure tells the ACE story end-to-end in one sketch:
    Pearl Rung-2 framing  ->  State  ->  LM policy  ->  K=4 candidates
                          ->  lookahead on cloned learner  ->  best/worst pair
                          ->  execute best on environment  ->  update learner
                          (and DPO update closes the policy loop)

Usage:
    python scripts/analysis/make_hero_excalidraw.py
    # then drag the output file onto excalidraw.com
"""

import json
import os
import random
import time
from pathlib import Path

# --- color palette (matches paper figures) ----------------------------------
COL_ENV       = ("#a5d8ff", "#1864ab")   # blue
COL_LEARNER   = ("#ffec99", "#e67700")   # amber
COL_POLICY    = ("#b2f2bb", "#2b8a3e")   # green
COL_CAND      = ("#ffc9c9", "#c92a2a")   # red
COL_LOOKAHEAD = ("#fcc2d7", "#a61e4d")   # pink
COL_DPO       = ("#d0bfff", "#5f3dc4")   # violet
COL_RUNG      = ("#dee2e6", "#495057")   # gray

# --- helpers ---------------------------------------------------------------
def _rand():
    return random.randint(1, 2_000_000_000)

def _now_ms():
    return int(time.time() * 1000)

def _id():
    return f"el-{_rand():010d}"

def _base(elem_type, x, y, w, h, stroke="#1e1e1e", fill="transparent",
          stroke_width=2, fill_style="hachure", roughness=1):
    return {
        "id": _id(),
        "type": elem_type,
        "x": x, "y": y,
        "width": w, "height": h,
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": fill_style,
        "strokeWidth": stroke_width,
        "strokeStyle": "solid",
        "roughness": roughness,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if elem_type == "rectangle" else None,
        "seed": _rand(),
        "version": 1,
        "versionNonce": _rand(),
        "isDeleted": False,
        "boundElements": [],
        "updated": _now_ms(),
        "link": None,
        "locked": False,
    }

def box(x, y, w, h, fill_stroke, stroke_width=2):
    fill, stroke = fill_stroke
    return _base("rectangle", x, y, w, h, stroke=stroke, fill=fill,
                 stroke_width=stroke_width)

def diamond(x, y, w, h, fill_stroke):
    fill, stroke = fill_stroke
    e = _base("diamond", x, y, w, h, stroke=stroke, fill=fill)
    e["roundness"] = None
    return e

def text(x, y, w, h, content, size=20, color="#1e1e1e", bold=False, align="center"):
    e = _base("text", x, y, w, h, stroke=color, fill="transparent")
    e["roundness"] = None
    e["text"] = content
    e["fontSize"] = size
    # 1=Virgil (hand-drawn), 2=Helvetica, 3=Cascadia (mono)
    e["fontFamily"] = 1
    e["textAlign"] = align
    e["verticalAlign"] = "middle"
    e["containerId"] = None
    e["originalText"] = content
    e["lineHeight"] = 1.25
    e["baseline"] = int(size * 0.85)
    e["strokeWidth"] = 1
    if bold:
        e["fontFamily"] = 1  # Virgil already has weight via roughness
        e["fontSize"] = int(size * 1.05)
    return e

def arrow(x1, y1, x2, y2, color="#1e1e1e", stroke_width=2, dashed=False, label=None):
    elements = []
    a = _base("arrow", min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1),
              stroke=color, fill="transparent", stroke_width=stroke_width)
    a["roundness"] = {"type": 2}  # for arrows: 2 = round/curved
    a["points"] = [[0, 0], [x2 - x1, y2 - y1]]
    a["lastCommittedPoint"] = None
    a["startBinding"] = None
    a["endBinding"] = None
    a["startArrowhead"] = None
    a["endArrowhead"] = "arrow"
    if dashed:
        a["strokeStyle"] = "dashed"
    a["x"] = x1
    a["y"] = y1
    a["width"] = x2 - x1
    a["height"] = y2 - y1
    elements.append(a)

    if label:
        # midpoint, slightly offset upward
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 14
        elements.append(text(mx - 80, my, 160, 20, label, size=14,
                             color="#495057"))
    return elements

def curved_arrow(points, color="#1e1e1e", stroke_width=2, dashed=False, label=None):
    """Multi-point curved arrow."""
    elements = []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = xs[0], ys[0]
    a = _base("arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys),
              stroke=color, fill="transparent", stroke_width=stroke_width)
    a["roundness"] = {"type": 2}
    a["points"] = [[x - x0, y - y0] for x, y in points]
    a["lastCommittedPoint"] = None
    a["startBinding"] = None
    a["endBinding"] = None
    a["startArrowhead"] = None
    a["endArrowhead"] = "arrow"
    if dashed:
        a["strokeStyle"] = "dashed"
    elements.append(a)

    if label:
        mid = points[len(points) // 2]
        elements.append(text(mid[0] - 80, mid[1] - 14, 160, 20, label, size=14,
                             color="#495057"))
    return elements


# --- build the figure -------------------------------------------------------
def build():
    elements = []

    # =========================================================================
    # Title and tagline
    # =========================================================================
    elements.append(text(620, 30, 760, 36,
                         "ACE: Active Causal Experimentalist",
                         size=28, bold=True))
    elements.append(text(620, 70, 760, 22,
                         "Each step: propose K=4, lookahead, execute the best, "
                         "DPO on (best, worst).",
                         size=14, color="#495057"))

    # =========================================================================
    # (1) Pearl rung sidebar (top-left): brief context box
    # =========================================================================
    rung_x, rung_y = 60, 130
    elements.append(text(rung_x, rung_y - 24, 200, 18, "Pearl's hierarchy",
                         size=12, color="#495057"))
    # three rungs stacked
    elements.append(box(rung_x, rung_y, 200, 36, COL_RUNG))
    elements.append(text(rung_x, rung_y, 200, 36, "3  Counterfactual",
                         size=14))
    elements.append(box(rung_x, rung_y + 44, 200, 36, COL_LOOKAHEAD,
                        stroke_width=3))
    elements.append(text(rung_x, rung_y + 44, 200, 36, "2  Intervention  <<",
                         size=14, bold=True))
    elements.append(box(rung_x, rung_y + 88, 200, 36, COL_RUNG))
    elements.append(text(rung_x, rung_y + 88, 200, 36, "1  Association",
                         size=14))
    elements.append(text(rung_x, rung_y + 134, 200, 18,
                         "ACE operates here", size=11,
                         color=COL_LOOKAHEAD[1]))

    # =========================================================================
    # (2) State box (center-left)
    # =========================================================================
    state_x, state_y, state_w, state_h = 320, 160, 280, 130
    elements.append(box(state_x, state_y, state_w, state_h, COL_RUNG,
                        stroke_width=2))
    elements.append(text(state_x, state_y + 6, state_w, 22,
                         "State  s_t", size=18, bold=True))
    elements.append(text(state_x + 12, state_y + 36, state_w - 24, 22,
                         "* graph G", size=14, align="left"))
    elements.append(text(state_x + 12, state_y + 60, state_w - 24, 22,
                         "* per-node losses {L_i}", size=14, align="left"))
    elements.append(text(state_x + 12, state_y + 84, state_w - 24, 22,
                         "* recent intervention history", size=14, align="left"))

    # =========================================================================
    # (3) Policy LM box (center-right of state)
    # =========================================================================
    pol_x, pol_y, pol_w, pol_h = 660, 160, 280, 130
    elements.append(box(pol_x, pol_y, pol_w, pol_h, COL_POLICY,
                        stroke_width=3))
    elements.append(text(pol_x, pol_y + 8, pol_w, 24,
                         "LM Policy  π_φ", size=18, bold=True))
    elements.append(text(pol_x + 10, pol_y + 38, pol_w - 20, 22,
                         "Qwen2.5-1.5B", size=14, align="center"))
    elements.append(text(pol_x + 10, pol_y + 62, pol_w - 20, 22,
                         "(pretrained world prior)", size=12,
                         color="#495057", align="center"))
    elements.append(text(pol_x + 10, pol_y + 90, pol_w - 20, 22,
                         "+ DPO fine-tuning", size=14, align="center",
                         color=COL_DPO[1]))

    # State -> Policy arrow
    elements += arrow(state_x + state_w, state_y + state_h / 2,
                      pol_x, pol_y + pol_h / 2,
                      label="prompt")

    # =========================================================================
    # (4) K=4 candidate boxes (below policy)
    # =========================================================================
    cand_y = 360
    cand_box_w, cand_box_h = 130, 60
    cand_xs = [330, 480, 630, 780]
    for i, cx in enumerate(cand_xs):
        elements.append(box(cx, cand_y, cand_box_w, cand_box_h, COL_CAND))
        elements.append(text(cx, cand_y + 8, cand_box_w, 22,
                             f"c_{i+1}", size=16, bold=True))
        elements.append(text(cx, cand_y + 30, cand_box_w, 22,
                             "DO X_i = v", size=11,
                             color="#495057"))

    # Policy -> candidates (4 fan-out arrows)
    pol_bottom_x = pol_x + pol_w / 2
    pol_bottom_y = pol_y + pol_h
    for cx in cand_xs:
        elements += arrow(pol_bottom_x, pol_bottom_y,
                          cx + cand_box_w / 2, cand_y,
                          color="#888888", stroke_width=1.5)
    elements.append(text(800, 320, 200, 20,
                         "K=4 candidates", size=13, color="#495057"))

    # =========================================================================
    # (5) Lookahead evaluation block (below candidates)
    # =========================================================================
    look_x, look_y, look_w, look_h = 280, 480, 730, 100
    elements.append(box(look_x, look_y, look_w, look_h, COL_LOOKAHEAD))
    elements.append(text(look_x, look_y + 8, look_w, 24,
                         "Lookahead: simulate each c_k on a CLONED learner",
                         size=18, bold=True))
    elements.append(text(look_x, look_y + 38, look_w, 22,
                         "ΔL(c_k) = L_before  -  L_after",
                         size=16, align="center"))
    elements.append(text(look_x, look_y + 64, look_w, 22,
                         "(reward = ΔL + α · node-importance + γ · diversity)",
                         size=12, color="#495057", align="center"))

    # candidates -> lookahead (group arrow)
    for cx in cand_xs:
        elements += arrow(cx + cand_box_w / 2, cand_y + cand_box_h,
                          cx + cand_box_w / 2 + 0, look_y,
                          color="#aaaaaa", stroke_width=1)

    # =========================================================================
    # (6) Best / Worst split below lookahead
    # =========================================================================
    best_x, best_y = 360, 640
    worst_x, worst_y = 720, 640
    elements.append(box(best_x, best_y, 200, 70, COL_POLICY, stroke_width=3))
    elements.append(text(best_x, best_y + 8, 200, 22,
                         "BEST  c*", size=18, bold=True,
                         color=COL_POLICY[1]))
    elements.append(text(best_x, best_y + 36, 200, 22,
                         "execute on environment", size=12,
                         color="#495057"))

    elements.append(box(worst_x, worst_y, 200, 70, COL_CAND, stroke_width=2))
    elements.append(text(worst_x, worst_y + 8, 200, 22,
                         "WORST  c-", size=18, bold=True,
                         color=COL_CAND[1]))
    elements.append(text(worst_x, worst_y + 36, 200, 22,
                         "preference loser", size=12,
                         color="#495057")) 

    elements += arrow(look_x + look_w / 4, look_y + look_h,
                      best_x + 100, best_y, label="argmax")
    elements += arrow(look_x + 3 * look_w / 4, look_y + look_h,
                      worst_x + 100, worst_y, label="argmin")

    # =========================================================================
    # (7) Environment + Learner column (right side)
    # =========================================================================
    env_x, env_y, env_w, env_h = 1080, 220, 260, 110
    elements.append(box(env_x, env_y, env_w, env_h, COL_ENV, stroke_width=3))
    elements.append(text(env_x, env_y + 8, env_w, 26,
                         "Environment  M*", size=18, bold=True))
    elements.append(text(env_x, env_y + 38, env_w, 22,
                         "(ground truth SCM)", size=12, color="#495057"))
    elements.append(text(env_x, env_y + 62, env_w, 22,
                         "do(V_i = ν)  →  samples", size=14))

    learn_x, learn_y, learn_w, learn_h = 1080, 380, 260, 110
    elements.append(box(learn_x, learn_y, learn_w, learn_h, COL_LEARNER,
                        stroke_width=3))
    elements.append(text(learn_x, learn_y + 8, learn_w, 26,
                         "Learner  M_θ", size=18, bold=True))
    elements.append(text(learn_x, learn_y + 38, learn_w, 22,
                         "(student SCM, MLP/node)", size=12,
                         color="#495057"))
    elements.append(text(learn_x, learn_y + 62, learn_w, 22,
                         "trained on intervention data", size=12,
                         color="#495057"))

    # Best -> Environment
    elements += arrow(best_x + 200, best_y + 35,
                      env_x, env_y + env_h / 2,
                      color=COL_POLICY[1], stroke_width=2.5,
                      label="execute c*")
    # Environment -> Learner
    elements += arrow(env_x + env_w / 2, env_y + env_h,
                      learn_x + learn_w / 2, learn_y,
                      label="data D")

    # Learner -> State (closing the loop, dashed)
    elements += curved_arrow(
        [(learn_x + learn_w / 2, learn_y + learn_h),
         (learn_x + learn_w / 2, 720),
         (state_x + state_w / 2, 720),
         (state_x + state_w / 2, state_y + state_h)],
        color="#888888", dashed=True, label="next step (updated state)"
    )

    # =========================================================================
    # (8) DPO update (right of worst, then arrow back to policy)
    # =========================================================================
    dpo_x, dpo_y, dpo_w, dpo_h = 1080, 580, 260, 100
    elements.append(box(dpo_x, dpo_y, dpo_w, dpo_h, COL_DPO, stroke_width=3))
    elements.append(text(dpo_x, dpo_y + 8, dpo_w, 24,
                         "DPO Update", size=18, bold=True,
                         color=COL_DPO[1]))
    elements.append(text(dpo_x + 8, dpo_y + 36, dpo_w - 16, 22,
                         "preference: (c*, c-)", size=14, align="center"))
    elements.append(text(dpo_x + 8, dpo_y + 62, dpo_w - 16, 22,
                         "loss = -log σ(β · log π/π_ref)", size=11,
                         color="#495057", align="center"))

    # best & worst -> DPO
    elements += arrow(best_x + 200, best_y + 35,
                      dpo_x, dpo_y + 30,
                      color=COL_DPO[1], stroke_width=1.5, dashed=True)
    elements += arrow(worst_x + 200, worst_y + 35,
                      dpo_x, dpo_y + 70,
                      color=COL_DPO[1], stroke_width=1.5, dashed=True)

    # DPO -> Policy (gradient update, dashed curved arrow)
    elements += curved_arrow(
        [(dpo_x, dpo_y + dpo_h / 2),
         (1000, dpo_y + dpo_h / 2),
         (1000, pol_y + pol_h + 30),
         (pol_x + pol_w / 2, pol_y + pol_h + 30),
         (pol_x + pol_w / 2, pol_y + pol_h)],
        color=COL_DPO[1], dashed=True, stroke_width=2,
        label="∇φ  policy update"
    )

    # =========================================================================
    # (9) Bottom note: key insight
    # =========================================================================
    elements.append(text(60, 770, 1300, 22,
                         "Key insight: as training proceeds, ΔL collapses ~500x "
                         "but the relative ranking of candidates stays stable. "
                         "DPO learns from ranking only, so it inherits stability.",
                         size=14, color="#495057"))

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
    print("To view / edit:")
    print(f"  1. Open https://excalidraw.com")
    print(f"  2. Drag {out_path} onto the canvas")
    print()
    print("To export as PNG / SVG for the paper:")
    print(f"  - Open in Excalidraw, then File -> Export image")
    print(f"  - Save into paper/figs/ace_hero.png (or .svg) and \\includegraphics in paper.tex")


if __name__ == "__main__":
    random.seed(42)  # deterministic IDs across runs
    main()
