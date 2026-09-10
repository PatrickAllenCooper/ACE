"""
Aggregate a "one condition per subdirectory" ACE suite into a comparison table.

Covers the two suites that share this layout and that the AISTATS 2027 paper
still promises but could not yet report when it was drafted:

  results/curc_dpo_alternatives/{dpo,sft_best,ranking}/seed_*/...
      -> the calibration-rule comparison (Section "Reward, Diversity, and
         Calibration Rules"; Table "tab:calibration-rules" in the supplement)

  results/curc_node_importance_ablation/{full,no_node_importance}/seed_*/...
      -> the final row of the component-ablation table ("tab:ablations")

Both write the standard ace_experiments.py output (node_losses.csv with
episode/total_loss/loss_* columns), so the same summarize_seed_dir() used by
aggregate_budget_fairness.py applies unchanged. Reported metric is total
mechanism MSE (per-node best x n_nodes), matching how the 5-node benchmark
is reported in the paper; per-node best is also emitted.

Cells that did not reach the 200-episode cap (max episode < 199) are
reported but flagged, so a wall-time-truncated run cannot silently pass as
a completed one -- the same convention the submit scripts' SKIP_COMPLETED
guard uses.

Usage:
    python scripts/analysis/aggregate_policy_update_suite.py \
        --root results/curc_dpo_alternatives
    python scripts/analysis/aggregate_policy_update_suite.py \
        --root results/curc_node_importance_ablation --min-episode 199 \
        --out results/curc_node_importance_ablation/aggregate.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from scaling_common import summarize_seed_dir  # noqa: E402


def max_episode(seed_dir: str) -> int | None:
    files = sorted(glob.glob(os.path.join(seed_dir, "**", "node_losses.csv"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        return None
    df = pd.read_csv(files[-1])
    return int(df["episode"].max()) if "episode" in df.columns else None


def collect(root: str, min_episode: int) -> list[dict]:
    rows = []
    for cond_dir in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
        cond = os.path.basename(cond_dir)
        if cond == "logs":
            continue
        for seed_dir in sorted(glob.glob(os.path.join(cond_dir, "seed_*"))):
            seed = os.path.basename(seed_dir).split("_", 1)[-1]
            summary = summarize_seed_dir(seed_dir)
            if summary is None:
                print(f"  WARNING: no output under {seed_dir}", file=sys.stderr)
                continue
            ep = max_episode(seed_dir)
            rows.append({
                "condition": cond,
                "seed": seed,
                "total_best": summary["per_node_best"] * summary["n_nodes"],
                "total_final": summary["per_node_final"] * summary["n_nodes"],
                "per_node_best": summary["per_node_best"],
                "n_nodes": summary["n_nodes"],
                "max_episode": ep,
                "complete": (ep is not None and ep >= min_episode),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-episode", type=int, default=199,
                    help="a cell counts as complete only if node_losses.csv reaches this episode")
    args = ap.parse_args()

    rows = collect(args.root, args.min_episode)
    if not rows:
        sys.exit(f"no results found under {args.root}")
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} rows to {args.out}")

    print(f"\n=== {os.path.basename(args.root)}: total best MSE (per-node best x n_nodes) ===")
    print(f"{'condition':<20}{'n':>3}{'complete':>9}{'mean_best':>11}{'std':>8}{'median':>9}{'mean_final':>12}")
    print("-" * 72)
    for cond, g in df.groupby("condition", sort=False):
        comp = g[g.complete]
        use = comp if len(comp) else g
        tag = "" if len(comp) == len(g) else f"  ({len(g) - len(comp)} INCOMPLETE, excluded)"
        mean = st.mean(use.total_best)
        sd = st.stdev(use.total_best) if len(use) > 1 else float("nan")
        med = st.median(use.total_best)
        fin = st.mean(use.total_final)
        print(f"{cond:<20}{len(use):>3}{len(comp):>9}{mean:>11.3f}{sd:>8.3f}{med:>9.3f}{fin:>12.3f}{tag}")
    incomplete = df[~df.complete]
    if len(incomplete):
        print("\nIncomplete cells (not at episode "
              f"{args.min_episode}; do NOT report these as finished):")
        for _, r in incomplete.iterrows():
            print(f"  {r.condition:<18} seed {r.seed:<6} max_episode={r.max_episode}")


if __name__ == "__main__":
    main()
