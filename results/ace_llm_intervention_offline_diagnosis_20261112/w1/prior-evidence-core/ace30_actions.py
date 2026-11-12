"""Read-only: target/value concentration in saved June-2026 30-node ACE scaling runs (DPO arm)."""
import sys, glob, os, collections
import pandas as pd
for root in sys.argv[1:]:
    tot = collections.Counter(); n = 0; x2_15 = 0; x2 = 0
    for f in sorted(glob.glob(os.path.join(root, "seed_*", "run_*", "metrics.csv"))):
        df = pd.read_csv(f)
        if "target" not in df.columns or len(df) == 0:
            print(f, "no target col or empty"); continue
        c = df["target"].value_counts()
        m = len(df)
        vx2 = df[df["target"] == "X2"]["value"]
        e15 = int((vx2 == 1.5).sum())
        top = [(k, int(v), round(v / m, 3)) for k, v in c.head(4).items()]
        print(os.path.relpath(f, root).split("/")[0], "rows", m, "distinct targets", c.shape[0], "top", top, "X2==1.5:", e15, "/", int(len(vx2)),
              "parse_rate", float(df["train_candidate_parse_rate"].iloc[-1]) if "train_candidate_parse_rate" in df else None)
        tot.update({k: int(v) for k, v in c.items()}); n += m; x2_15 += e15; x2 += len(vx2)
    if n:
        print("TOTAL", root, "rows", n, "top", [(k, v, round(v / n, 3)) for k, v in tot.most_common(5)], "X2==1.5", x2_15, "of X2", x2, "share of all rows", round(x2_15 / n, 4))
    print()
