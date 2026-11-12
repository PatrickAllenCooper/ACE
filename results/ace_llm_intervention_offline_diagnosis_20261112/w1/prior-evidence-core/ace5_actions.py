"""Read-only: action distribution in saved Jan-2026 5-node ACE multi-seed runs (paper Table 1 era)."""
import sys, glob, os, collections
import pandas as pd
root = sys.argv[1]
tot = collections.Counter(); vals = collections.Counter(); n = 0
for f in sorted(glob.glob(os.path.join(root, "seed_*", "run_*", "metrics.csv"))):
    df = pd.read_csv(f)
    c = df["target"].value_counts()
    m = len(df)
    x2 = df[df["target"] == "X2"]["value"].round(2)
    x1 = df[df["target"] == "X1"]["value"].round(2)
    print(os.path.basename(os.path.dirname(f)), "rows", m, {k: int(v) for k, v in c.items()},
          "X2 top values", [(float(k), int(v)) for k, v in x2.value_counts().head(3).items()],
          "X1 top values", [(float(k), int(v)) for k, v in x1.value_counts().head(3).items()],
          "episodes", int(df["episode"].max()) + 1,
          "parse_rate", float(df["train_candidate_parse_rate"].iloc[-1]),
          "teacher_fallback_rate", float(df["train_teacher_fallback_rate"].iloc[-1]),
          "dpo_loss_mean_last200", round(float(df["dpo_loss"].tail(200).mean()), 4))
    tot.update({k: int(v) for k, v in c.items()}); n += m
    vals.update({("X2", float(k)): int(v) for k, v in x2.value_counts().items()})
print("TOTAL", n, dict(tot), {k: round(v / n, 4) for k, v in tot.items()})
print("X2 value top", vals.most_common(5), "X2 total", sum(v for (t, _), v in vals.items()))
