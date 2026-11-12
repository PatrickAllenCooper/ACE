"""Supervised-target distributions from the saved records. Pure offline counting."""
import pickle
import sys
from collections import Counter, defaultdict

import pandas as pd

D = pickle.load(open(sys.argv[1], "rb"))
R = D["records"]
ACTIONS = ["observe_all", "set_x_neg", "set_x_zero", "set_x_pos",
           "set_y_neg", "set_y_zero", "set_y_pos", "set_xy", "STOP"]


def tgt(r):
    return "STOP" if r["supervised_action_id"] is None else r["supervised_action_id"]


df = pd.DataFrame([{**{k: r[k] for k in ("record_id", "world_index", "snapshot", "regime", "graph",
                                          "cost_menu", "noise_variance", "behavior_method",
                                          "teacher_method", "milestone", "remaining_units",
                                          "spent_units", "terminal", "strict_preferences",
                                          "n_preferences", "actual_acquisition_cost_units")},
                    "target": tgt(r)} for r in R])
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

print("TOTAL supervised examples:", len(df))
print("\n== overall target distribution")
c = df["target"].value_counts().reindex(ACTIONS, fill_value=0)
print(pd.DataFrame({"n": c, "frac_of_160": (c / len(df)).round(4)}))
nt = df[df["target"] != "STOP"]
print("\nnonterminal examples:", len(nt), " stop examples:", (df["target"] == "STOP").sum())
c2 = nt["target"].value_counts().reindex(ACTIONS[:-1], fill_value=0)
print(pd.DataFrame({"n": c2, "frac_of_128": (c2 / len(nt)).round(4)}))

print("\n== sanity: are all stop targets exactly the snapshot-4 terminal records?")
print(pd.crosstab(df["snapshot"], df["target"] == "STOP"))
print("stop records: terminal flag values", df[df["target"] == "STOP"]["terminal"].value_counts().to_dict(),
      "remaining_units", df[df["target"] == "STOP"]["remaining_units"].value_counts().to_dict())
print("nonterminal records: remaining_units dist", df[df["target"] != "STOP"]["remaining_units"].value_counts().sort_index().to_dict())

for col in ("teacher_method", "graph", "cost_menu", "snapshot", "regime", "behavior_method"):
    print(f"\n== target by {col} (all 160)")
    print(pd.crosstab(df[col], df["target"]).reindex(columns=ACTIONS, fill_value=0).assign(total=lambda x: x.sum(axis=1)))

print("\n== target by teacher_method, nonterminal only (128)")
print(pd.crosstab(nt["teacher_method"], nt["target"]).reindex(columns=ACTIONS[:-1], fill_value=0).assign(total=lambda x: x.sum(axis=1)))
print("\n== target by teacher x graph, nonterminal only")
print(pd.crosstab([nt["teacher_method"], nt["graph"]], nt["target"]).reindex(columns=ACTIONS[:-1], fill_value=0).assign(total=lambda x: x.sum(axis=1)))
print("\n== target by teacher x cost menu, nonterminal only")
print(pd.crosstab([nt["teacher_method"], nt["cost_menu"]], nt["target"]).reindex(columns=ACTIONS[:-1], fill_value=0).assign(total=lambda x: x.sum(axis=1)))
print("\n== target by graph x cost menu (all 160)")
print(pd.crosstab([df["graph"], df["cost_menu"]], df["target"]).reindex(columns=ACTIONS, fill_value=0).assign(total=lambda x: x.sum(axis=1)))


def majority_accuracy(frame, keys):
    if not keys:
        top = frame["target"].value_counts()
        return top.iloc[0], len(frame), top.index[0]
    correct = 0
    for _, g in frame.groupby(keys):
        correct += g["target"].value_counts().iloc[0]
    return correct, len(frame), None


print("\n== majority-class accuracy (in-sample; a lookup that outputs the modal target of each cell)")
for label, frame in (("all 160", df), ("nonterminal 128", nt)):
    for keys in ([], ["graph"], ["cost_menu"], ["graph", "cost_menu"], ["snapshot"], ["teacher_method"],
                 ["graph", "cost_menu", "snapshot"], ["graph", "cost_menu", "teacher_method"],
                 ["regime"], ["graph", "cost_menu", "regime"]):
        k, n, top = majority_accuracy(frame, keys)
        print(f"{label:16s} cells={'+'.join(keys) if keys else '(none)':38s} correct={k:3d}/{n} = {k/n:.4f}"
              + (f"  modal={top}" if top else ""))

print("\n== graph x cost cells: modal target and its count (all 160, then nonterminal)")
for label, frame in (("all", df), ("nonterminal", nt)):
    rows = []
    for (g, cm), sub in frame.groupby(["graph", "cost_menu"]):
        vc = sub["target"].value_counts()
        rows.append({"graph": g, "cost_menu": cm, "n": len(sub), "modal": vc.index[0], "modal_n": int(vc.iloc[0]),
                     "modal_frac": round(vc.iloc[0] / len(sub), 4),
                     "set_x_neg_n": int((sub["target"] == "set_x_neg").sum()),
                     "dist": dict(vc)})
    print(label)
    print(pd.DataFrame(rows).to_string())

print("\n== leave-one-world-out majority accuracy (honest generalization of a constant/cell-modal rule)")
for label, frame in (("all 160", df), ("nonterminal 128", nt)):
    for keys in ([], ["graph", "cost_menu"], ["snapshot"], ["teacher_method"]):
        correct = 0
        for w in sorted(frame["world_index"].unique()):
            train, test = frame[frame["world_index"] != w], frame[frame["world_index"] == w]
            for _, row in test.iterrows():
                sub = train
                for k in keys:
                    sub = sub[sub[k] == row[k]]
                if len(sub) == 0:
                    sub = train
                vc = sub["target"].value_counts()
                # deterministic tie break: lexical
                best = sorted(vc.index, key=lambda a: (-vc[a], a))[0]
                correct += int(best == row["target"])
        print(f"{label:16s} cells={'+'.join(keys) if keys else '(none)':24s} LOWO correct={correct}/{len(frame)} = {correct/len(frame):.4f}")

df.to_pickle(sys.argv[2])
