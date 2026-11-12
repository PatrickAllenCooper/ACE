"""(6) Training logs: loss/gradient trajectories, saved command scores, saved generations. Offline JSON only.

No model weights are loaded: only *.json files are read; *.pt files are never opened.
collection.pkl is my own scratch pickle from load_collection.py (trusted).
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

FIT = Path("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/"
           "remote/launch/training/session/worker/artifacts/fit")
TR = FIT / "training"
D = pickle.load(open(sys.argv[1], "rb"))
REC = {r["record_id"]: r for r in D["records"]}
ACTIONS = ["observe_all", "set_x_neg", "set_x_zero", "set_x_pos", "set_y_neg", "set_y_zero", "set_y_pos", "set_xy", "STOP"]


def cmd_name(c):
    o = json.loads(c)
    return "STOP" if o.get("stop") else o["action_id"]


steps = []
for i in range(279):
    comp = json.loads((TR / f"step-{i:05d}.complete.json").read_text())
    loss = json.loads((TR / f"step-{i:05d}.loss.json").read_text())
    assert comp["loss"] == loss["loss"] and loss["finite"]
    steps.append(comp)
S = pd.DataFrame(steps)
print("steps:", len(S), " stages:", S.stage.value_counts().to_dict())
print("all optimizer steps returned:", bool(S.optimizer_step_returned.all()), " rows per step:", S.groupby("stage").rows.unique().to_dict())
print("clip limit: 1.0; gradient norm before clip quantiles by stage:")
print(S.groupby("stage").gradient_norm_before_clip.describe().to_string())
print("steps where the gradient was clipped (norm_before > 1):", S.groupby("stage").apply(lambda g: f"{int((g.gradient_norm_before_clip > 1).sum())}/{len(g)}").to_dict())
print("input tokens per step by stage:", S.groupby("stage").input_tokens.describe()[["min", "50%", "max"]].to_dict())

sft = S[S.stage == "sft"].reset_index(drop=True)
dpo = S[S.stage == "dpo"].reset_index(drop=True)
sft["target"] = [("STOP" if REC[r]["supervised_action_id"] is None else REC[r]["supervised_action_id"]) for r in sft.record_id]
sft["snapshot"] = [REC[r]["snapshot"] for r in sft.record_id]
sft["world_index"] = [REC[r]["world_index"] for r in sft.record_id]
sft["action_tokens"] = sft.action_tokens
print("\n== SFT loss trajectory (single pass, roster order; each example seen exactly once BEFORE its own update)")
print("is the SFT order exactly the roster order?", list(sft.record_id) == D["roster"])
print("loss definition check: action_tokens values", sorted(sft.action_tokens.unique()))
for lo in range(0, 160, 20):
    blk = sft.iloc[lo:lo + 20]
    print(f"  steps {lo:3d}-{lo + 19:3d}: mean loss {blk.loss.mean():.4f}  median {blk.loss.median():.4f}  min {blk.loss.min():.4f}  max {blk.loss.max():.4f}")
print("  first 10 losses:", [round(x, 3) for x in sft.loss[:10]])
print("  last 10 losses :", [round(x, 3) for x in sft.loss[-10:]])
print("\nSFT loss by target command (online loss = -log-likelihood measure of the target before the update):")
g = sft.groupby("target").loss.agg(["count", "mean", "median", "min", "max"]).reindex(ACTIONS)
print(g.round(4).to_string())
print("\nSFT loss by target, second half of the pass only (steps 80-159):")
g = sft.iloc[80:].groupby("target").loss.agg(["count", "mean", "median", "min", "max"]).reindex(ACTIONS)
print(g.round(4).to_string())
print("\nSFT loss by snapshot index:")
print(sft.groupby("snapshot").loss.agg(["count", "mean", "median"]).round(4).to_string())
print("SFT loss by snapshot, second half:")
print(sft.iloc[80:].groupby("snapshot").loss.agg(["count", "mean", "median"]).round(4).to_string())

# Is loss mean-per-token or sum? compare to final scores later.
print("\n== DPO loss trajectory (119 single-pair updates; beta=0.1; loss at reference == ln2 = 0.693147)")
for lo in range(0, 119, 20):
    blk = dpo.iloc[lo:lo + 20]
    print(f"  dpo steps {lo:3d}-{min(lo + 19, 118):3d}: mean loss {blk.loss.mean():.5f}  min {blk.loss.min():.5f}  max {blk.loss.max():.5f}")
print("  first 10:", [round(x, 5) for x in dpo.loss[:10]])
print("  last 10 :", [round(x, 5) for x in dpo.loss[-10:]])
print("  DPO losses below ln2:", int((dpo.loss < np.log(2)).sum()), "/119; above:", int((dpo.loss > np.log(2)).sum()),
      " overall mean:", round(dpo.loss.mean(), 5), " min:", round(dpo.loss.min(), 5), " max:", round(dpo.loss.max(), 5))
# implied preference logit margin: loss = -log sigmoid(beta * delta) -> delta = logit(exp(-loss))/beta
beta = 0.1
dpo["implied_delta"] = [np.log(np.exp(-l) / (1 - np.exp(-l))) / beta for l in dpo.loss]
print("  implied DPO margin delta=(dlogp_chosen - dlogp_rejected) vs reference BEFORE each update: quantiles",
      {k: round(float(v), 3) for k, v in dpo.implied_delta.quantile([0, .25, .5, .75, 1]).items()})
print("  share of DPO steps with positive implied margin before update:", round(float((dpo.implied_delta > 0).mean()), 4))

# Reference (post-SFT) scores for the 119 selected pairs and final scores on all commands
ref = json.loads((TR / "reference-scores.json").read_text())["scores"]
fin = json.loads((FIT / "final-before-reload.json").read_text())
fin_scores, fin_cmds = fin["scores"], fin["commands"]
plan = json.loads((TR / "plan.json").read_text())
print("\nplan.json keys:", list(plan.keys())[:30])

rows = []
for i, st in dpo.iterrows():
    rid, pi = st.record_id, int(st.pair_index)
    key = json.dumps([rid, pi], separators=(",", ":"))
    lc, lr = ref[key]
    rec = REC[rid]
    # strict pairs in record order
    ex = None
    rows.append({"step": int(st["index"]), "record_id": rid, "pair_index": pi, "ref_logp_chosen": lc, "ref_logp_rejected": lr,
                 "dpo_loss": st.loss})
Rf = pd.DataFrame(rows)

EXPOSURE = json.load(open("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/"
                          "terminal01/remote/cpu/preparation/session/worker/artifacts/exposure.json"))
sel = {}
for rec in EXPOSURE["records"]:
    for row in rec["pairs"]:
        if row["status"] == "selected":
            sel[rec["record_id"]] = (row["pair_index"], cmd_name(row["chosen_command"]), cmd_name(row["rejected_command"]))
Rf["chosen"] = [sel[r][1] for r in Rf.record_id]
Rf["rejected"] = [sel[r][2] for r in Rf.record_id]
assert all(sel[r][0] == p for r, p in zip(Rf.record_id, Rf.pair_index))
print("DPO step order follows roster order of records with a strict pair:",
      list(Rf.record_id) == [r for r in D["roster"] if r in sel])


def fin_lp(rid, action):
    cmds = [cmd_name(c) for c in fin_cmds[rid]]
    return fin_scores[rid][cmds.index(action)]


Rf["fin_logp_chosen"] = [fin_lp(r, a) for r, a in zip(Rf.record_id, Rf.chosen)]
Rf["fin_logp_rejected"] = [fin_lp(r, a) for r, a in zip(Rf.record_id, Rf.rejected)]
Rf["ref_margin"] = Rf.ref_logp_chosen - Rf.ref_logp_rejected
Rf["fin_margin"] = Rf.fin_logp_chosen - Rf.fin_logp_rejected
Rf["margin_change"] = Rf.fin_margin - Rf.ref_margin
print("\n== SFT-checkpoint (reference) vs final DPO-checkpoint log-likelihoods on the 119 trained pairs")
print("SFT checkpoint already ranks the DPO-preferred command above the rejected one in",
      int((Rf.ref_margin > 0).sum()), "/119 pairs  (i.e. SFT policy agrees with measured preference; 59.5 = chance)")
print("Final DPO checkpoint ranks preferred above rejected in", int((Rf.fin_margin > 0).sum()), "/119 pairs")
print("margin (logp chosen - logp rejected): SFT mean", round(Rf.ref_margin.mean(), 3), "median", round(Rf.ref_margin.median(), 3),
      "| final mean", round(Rf.fin_margin.mean(), 3), "median", round(Rf.fin_margin.median(), 3))
print("margin change (final - SFT) quantiles:", {k: round(float(v), 3) for k, v in Rf.margin_change.quantile([0, .1, .25, .5, .75, .9, 1]).items()},
      " pairs whose margin increased:", int((Rf.margin_change > 0).sum()), "/119")
print("\nPer-command log-likelihood at SFT checkpoint vs final checkpoint, over the (record, command) points in the 119 trained pairs (238 points):")
long = pd.concat([
    Rf[["record_id", "chosen", "ref_logp_chosen", "fin_logp_chosen"]].rename(columns={"chosen": "cmd", "ref_logp_chosen": "sft", "fin_logp_chosen": "final"}).assign(role="preferred"),
    Rf[["record_id", "rejected", "ref_logp_rejected", "fin_logp_rejected"]].rename(columns={"rejected": "cmd", "ref_logp_rejected": "sft", "fin_logp_rejected": "final"}).assign(role="rejected")])
long["change"] = long.final - long.sft
t = long.groupby("cmd").agg(n=("sft", "size"), sft_mean=("sft", "mean"), sft_median=("sft", "median"), final_mean=("final", "mean"),
                            final_median=("final", "median"), change_mean=("change", "mean")).reindex(ACTIONS[:-1])
print(t.round(3).to_string())
print("\nsame split by role:")
print(long.groupby(["cmd", "role"]).agg(n=("sft", "size"), sft_mean=("sft", "mean"), final_mean=("final", "mean"), change_mean=("change", "mean")).round(3).to_string())

# SFT checkpoint: does it prefer set_x_neg over the alternative in pairs that involve set_x_neg?
inv = Rf[(Rf.chosen == "set_x_neg") | (Rf.rejected == "set_x_neg")]
xneg_lp = [a if c == "set_x_neg" else b for a, b, c in zip(inv.ref_logp_chosen, inv.ref_logp_rejected, inv.chosen)]
oth_lp = [b if c == "set_x_neg" else a for a, b, c in zip(inv.ref_logp_chosen, inv.ref_logp_rejected, inv.chosen)]
print(f"\nSFT checkpoint on the {len(inv)} trained pairs involving set_x_neg: set_x_neg scored higher than the other command in "
      f"{sum(x > o for x, o in zip(xneg_lp, oth_lp))}/{len(inv)}; mean logp(set_x_neg) {np.mean(xneg_lp):.3f} vs other {np.mean(oth_lp):.3f}")

print("\n== FINAL (DPO) checkpoint: full command score table on the 160 training states")
tab = []
for rid in D["roster"]:
    cmds = [cmd_name(c) for c in fin_cmds[rid]]
    sc = np.array(fin_scores[rid])
    order = np.argsort(-sc)
    rec = REC[rid]
    row = {"record_id": rid, "world_index": rec["world_index"], "snapshot": rec["snapshot"], "graph": rec["graph"], "cost_menu": rec["cost_menu"],
           "target": "STOP" if rec["supervised_action_id"] is None else rec["supervised_action_id"],
           "n_cmds": len(cmds), "winner": cmds[order[0]], "winner_logp": sc[order[0]],
           "runner_up": cmds[order[1]] if len(cmds) > 1 else None,
           "margin_to_runner_up": sc[order[0]] - sc[order[1]] if len(cmds) > 1 else np.nan,
           "legal_mass": float(np.exp(sc).sum())}
    for c, s in zip(cmds, sc):
        row["lp_" + c] = s
    tab.append(row)
F = pd.DataFrame(tab)
nt = F[F.n_cmds == 9]
print("command order in saved score vectors:", [cmd_name(c) for c in fin_cmds[D['roster'][0]]])
print("argmax-scored command on the 128 nonterminal training states:", nt.winner.value_counts().to_dict())
print("runner-up on those states:", nt.runner_up.value_counts().to_dict())
print("margin (winner - runner-up, nats) quantiles:", {k: round(float(v), 3) for k, v in nt.margin_to_runner_up.quantile([0, .05, .25, .5, .75, .95, 1]).items()})
print("P(winner)=exp(logp) quantiles:", {k: round(float(np.exp(v)), 4) for k, v in nt.winner_logp.quantile([0, .05, .25, .5, .75, .95, 1]).items()})
print("total probability mass on the 9 legal commands, quantiles:", {k: round(float(v), 4) for k, v in nt.legal_mass.quantile([0, .25, .5, .75, 1]).items()})
print("\nmean / sd / min / max of final log-likelihood per command across the 128 nonterminal training states:")
cols = ["lp_" + a for a in ACTIONS]
print(nt[cols].agg(["mean", "std", "min", "max"]).T.round(3).to_string())
print("\nfinal-checkpoint argmax == supervised (teacher) target on nonterminal states:", int((nt.winner == nt.target).sum()), "/128")
print("terminal states (only {stop} legal): logp(stop) quantiles", {k: round(float(v), 3) for k, v in F[F.n_cmds == 1].lp_STOP.quantile([0, .25, .5, .75, 1]).items()})
print("\nfinal checkpoint: mean logp by snapshot index (nonterminal):")
print(nt.groupby("snapshot")[cols].mean().round(3).to_string())
print("\nfinal checkpoint: mean logp by graph x cost menu (nonterminal):")
print(nt.groupby(["graph", "cost_menu"])[cols].mean().round(3).to_string())
print("\nstate sensitivity of the final policy: between-state SD of logp(set_x_neg) =", round(nt.lp_set_x_neg.std(), 4),
      "; between-state SD of the winner margin =", round(nt.margin_to_runner_up.std(), 4))
print("states where the measured-best action (4-replicate mean) == final argmax:",
      sum(1 for rid, w in zip(nt.record_id, nt.winner) if max(REC[rid]["feedback"]["mean_gains_per_cost"], key=REC[rid]["feedback"]["mean_gains_per_cost"].get) == w), "/128")
F.to_pickle(sys.argv[2])

print("\n== Saved greedy generations on the training states")
for stage in ("sft", "final"):
    gens = []
    for i in range(161):
        p = FIT / f"{stage}-generation-{i:02d}.json"
        if not p.exists():
            p = FIT / f"{stage}-generation-{i}.json"
        g = json.loads(p.read_text())
        gens.append(g)
    print(f"\n[{stage}] generation files: {len(gens)}; keys: {sorted(gens[0].keys())}")
    G = pd.DataFrame([{"ordinal": g["ordinal"], "record_id": g["record_id"], "completion": cmd_name(g["completion"]),
                       "legal_commands": g["legal_commands"]} for g in gens])
    G["target"] = [("STOP" if REC[r]["supervised_action_id"] is None else REC[r]["supervised_action_id"]) for r in G.record_id]
    G["snapshot"] = [REC[r]["snapshot"] for r in G.record_id]
    G["graph"] = [REC[r]["graph"] for r in G.record_id]
    G["cost_menu"] = [REC[r]["cost_menu"] for r in G.record_id]
    first160 = G.iloc[:160]
    print("  ordinal 160 revisits record:", G.iloc[160].record_id, " completion:", G.iloc[160].completion, " same as ordinal 0:", G.iloc[160].completion == G.iloc[0].completion)
    print("  first 160 cover the roster in order:", list(first160.record_id) == D["roster"])
    ntg = first160[first160.legal_commands == 9]
    print("  completions on the 128 nonterminal training states:", ntg.completion.value_counts().to_dict())
    print("  completions on the 32 terminal states:", first160[first160.legal_commands == 1].completion.value_counts().to_dict())
    print("  completion == supervised target: nonterminal", int((ntg.completion == ntg.target).sum()), "/128 ; all", int((first160.completion == first160.target).sum()), "/160")
    print("  nonterminal completions by snapshot:")
    print(pd.crosstab(ntg.snapshot, ntg.completion).to_string())
    print("  nonterminal completions by graph x cost:")
    print(pd.crosstab([ntg.graph, ntg.cost_menu], ntg.completion).to_string())
    print("  nonterminal completion vs supervised target:")
    print(pd.crosstab(ntg.target, ntg.completion).to_string())
    G.to_pickle(sys.argv[3] + "." + stage)
