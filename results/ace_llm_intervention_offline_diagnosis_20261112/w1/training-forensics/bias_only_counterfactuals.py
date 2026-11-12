"""Toy-model counterfactuals (INFERENCE ONLY): the same state-independent bias model, driven by other pair selections.
Offline arithmetic; reads saved JSON and my own scratch pickle (trusted). No weights."""
import json, math, pickle
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

FIT = Path("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit")
EXPOSURE = json.load(open("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/cpu/preparation/session/worker/artifacts/exposure.json"))
def cmd_name(c):
    o = json.loads(c); return "STOP" if o.get("stop") else o["action_id"]
PATH = {"observe_all": (0, None, None), "set_x_neg": (1, 0, 0), "set_x_pos": (1, 0, 1), "set_x_zero": (1, 0, 2),
        "set_y_neg": (1, 1, 0), "set_y_pos": (1, 1, 1), "set_y_zero": (1, 1, 2), "set_xy": (1, 2, None)}
l2 = np.log([0.368, 0.632]); l3 = np.log([0.752, 0.212, 0.036]); ls = np.log([0.884, 0.096, 0.02])
A, MOM = 0.9, 0.5
def lsm(v): return v - np.log(np.exp(v).sum())
def logp(theta, cmd):
    a, b, c = PATH[cmd]; out = lsm(theta[0])[a]
    if b is not None: out += lsm(theta[1])[b]
    if c is not None: out += lsm(theta[2])[c]
    return out
def grad(theta, cmd):
    a, b, c = PATH[cmd]; g = [np.zeros(2), np.zeros(3), np.zeros(3)]
    e = np.zeros(2); e[a] = 1; g[0] = e - np.exp(lsm(theta[0]))
    if b is not None: e = np.zeros(3); e[b] = 1; g[1] = e - np.exp(lsm(theta[1]))
    if c is not None: e = np.zeros(3); e[c] = 1; g[2] = e - np.exp(lsm(theta[2]))
    return g
def run(pairs, a=A, mom=MOM, init=None):
    theta = [x.copy() for x in (init or [l2, l3, ls])]; vel = [np.zeros(2), np.zeros(3), np.zeros(3)]
    for c, r in pairs:
        gc, gr = grad(theta, c), grad(theta, r); g = [x - y for x, y in zip(gc, gr)]
        n = math.sqrt(sum(float((x ** 2).sum()) for x in g))
        for k in range(3):
            vel[k] = mom * vel[k] + (1 - mom) * g[k] / n; theta[k] = theta[k] + a * vel[k]
    lp = {cmd: logp(theta, cmd) for cmd in PATH}
    return max(lp, key=lp.get), lp

strict_by_rec = {}
for rec in EXPOSURE["records"]:
    rows = [(cmd_name(row["chosen_command"]), cmd_name(row["rejected_command"]), row["status"]) for row in rec["pairs"] if row["status"] != "tied"]
    if rows: strict_by_rec[rec["record_id"]] = rows
actual = [next((c, r) for c, r, s in rows if s == "selected") for rows in strict_by_rec.values()]
w, lp = run(actual)
print("actual 119 selected pairs, actual order -> bias-only argmax:", w, {k: round(v, 2) for k, v in lp.items()})
rng = np.random.default_rng(20261112)
# (a) same 119 pairs, random order
wins = Counter(run([actual[i] for i in rng.permutation(len(actual))])[0] for _ in range(2000))
print("(a) same 119 pairs, shuffled order (2000 draws):", dict(wins.most_common()))
# (b) a different one-strict-pair-per-record selection (what another hash/seed would have picked), roster order
wins = Counter()
for _ in range(2000):
    pairs = [rows[rng.integers(len(rows))][:2] for rows in strict_by_rec.values()]
    wins[run(pairs)[0]] += 1
print("(b) random one-strict-pair-per-record selection, roster order (2000 draws):", dict(wins.most_common()))
# (c) all 687 strict pairs, roster order, with step size scaled so total movement is comparable, and unscaled
allp = [(c, r) for rows in strict_by_rec.values() for c, r, _ in rows]
print("(c) all", len(allp), "strict pairs in roster order: argmax with a=0.9:", run(allp)[0], "; with a=0.9*119/687:", run(allp, a=A * 119 / len(allp))[0])
wins = Counter(run([allp[i] for i in rng.permutation(len(allp))], a=A * 119 / len(allp))[0] for _ in range(500))
print("    all strict pairs, shuffled order, a scaled (500 draws):", dict(wins.most_common()))
# (d) uniform initial node probabilities (no SFT head start), actual pairs and order
w, lp = run(actual, init=[np.log([.5, .5]), np.log([1/3]*3), np.log([1/3]*3)])
print("(d) actual pairs+order but UNIFORM initial biases (no SFT head start):", w, {k: round(v, 2) for k, v in lp.items()})
# (e) sensitivity to step size for the actual pairs
print("(e) actual pairs, argmax by step size:", {a_: run(actual, a=a_)[0] for a_ in (0.1, 0.25, 0.5, 0.9, 1.5, 2.5)})
print("    with a=0.1 (10x smaller, close to an anchored update):", {k: round(v, 2) for k, v in run(actual, a=0.1)[1].items()})
