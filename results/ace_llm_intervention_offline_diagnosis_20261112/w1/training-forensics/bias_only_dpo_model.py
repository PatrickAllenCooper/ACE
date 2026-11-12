"""Can a STATE-INDEPENDENT token-bias model reproduce the saved online DPO losses?

Each DPO pair was seen exactly once, before its own update, so the saved online loss of step t measures only
what transferred from pairs 0..t-1. A model with no state input at all (8 free logits: observe/set, _x/_y/_xy,
_neg/_pos/_zero shared across X and Y) is updated with unit-norm steps on each pair's margin (all 119 real
steps were clipped to norm 1). One free parameter: step size a. Offline arithmetic only; no weights loaded.
"""
import json, math, pickle
from pathlib import Path
import numpy as np

FIT = Path("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit")
TR = FIT / "training"
EXPOSURE = json.load(open("/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/cpu/preparation/session/worker/artifacts/exposure.json"))
BETA = 0.1


def cmd_name(c):
    o = json.loads(c)
    return "STOP" if o.get("stop") else o["action_id"]


PATH = {"observe_all": (0, None, None), "set_x_neg": (1, 0, 0), "set_x_pos": (1, 0, 1), "set_x_zero": (1, 0, 2),
        "set_y_neg": (1, 1, 0), "set_y_pos": (1, 1, 1), "set_y_zero": (1, 1, 2), "set_xy": (1, 2, None)}
sel = [(rec["record_id"], row["pair_index"], cmd_name(row["chosen_command"]), cmd_name(row["rejected_command"]))
       for rec in EXPOSURE["records"] for row in rec["pairs"] if row["status"] == "selected"]
losses = [json.loads((TR / f"step-{160 + i:05d}.loss.json").read_text())["loss"] for i in range(119)]
implied = np.array([math.log(math.exp(-l) / (1 - math.exp(-l))) / BETA for l in losses])
ref = json.loads((TR / "reference-scores.json").read_text())["scores"]
fin = json.loads((FIT / "final-before-reload.json").read_text())

# reference node probabilities estimated from the mean SFT-checkpoint command log-likelihoods on the pair states
pts = {}
for rid, pi, c, r in sel:
    lc, lr = ref[json.dumps([rid, pi], separators=(",", ":"))]
    pts.setdefault(c, []).append(lc)
    pts.setdefault(r, []).append(lr)
p = {k: math.exp(np.mean(v)) for k, v in pts.items()}
p_set = sum(v for k, v in p.items() if k != "observe_all")
l2 = np.log(np.array([p["observe_all"], p_set]) / (p["observe_all"] + p_set))
px, py, pxy = p["set_x_neg"] + p["set_x_pos"] + p["set_x_zero"], p["set_y_neg"] + p["set_y_pos"] + p["set_y_zero"], p["set_xy"]
l3 = np.log(np.array([px, py, pxy]) / (px + py + pxy))
suf = np.array([p["set_x_neg"] + p["set_y_neg"], p["set_x_pos"] + p["set_y_pos"], p["set_x_zero"] + p["set_y_zero"]])
ls = np.log(suf / suf.sum())
print("reference node probabilities (from mean SFT log-likelihoods): observe/set", np.exp(l2).round(3), " _x/_y/_xy", np.exp(l3).round(3), " _neg/_pos/_zero", np.exp(ls).round(3))


def lsm(v):
    return v - np.log(np.exp(v).sum())


def logp(theta, cmd):
    a, b, c = PATH[cmd]
    out = lsm(theta[0])[a]
    if b is not None:
        out += lsm(theta[1])[b]
    if c is not None:
        out += lsm(theta[2])[c]
    return out


def grad(theta, cmd):
    a, b, c = PATH[cmd]
    g = [np.zeros(2), np.zeros(3), np.zeros(3)]
    e = np.zeros(2); e[a] = 1; g[0] = e - np.exp(lsm(theta[0]))
    if b is not None:
        e = np.zeros(3); e[b] = 1; g[1] = e - np.exp(lsm(theta[1]))
    if c is not None:
        e = np.zeros(3); e[c] = 1; g[2] = e - np.exp(lsm(theta[2]))
    return g


def run(a, weighted=False, momentum=0.0):
    theta0 = [l2.copy(), l3.copy(), ls.copy()]
    theta = [x.copy() for x in theta0]
    vel = [np.zeros(2), np.zeros(3), np.zeros(3)]
    pred = []
    for rid, pi, c, r in sel:
        d = (logp(theta, c) - logp(theta0, c)) - (logp(theta, r) - logp(theta0, r))
        pred.append(d)
        gc, gr = grad(theta, c), grad(theta, r)
        g = [x - y for x, y in zip(gc, gr)]
        n = math.sqrt(sum(float((x ** 2).sum()) for x in g))
        w = 1 / (1 + math.exp(BETA * d)) if weighted else 1.0
        for k in range(3):
            vel[k] = momentum * vel[k] + (1 - momentum) * w * g[k] / (n if not weighted else 1.0)
            theta[k] = theta[k] + a * vel[k]
    final_shift = {cmd: logp(theta, cmd) - logp(theta0, cmd) for cmd in PATH}
    return np.array(pred), final_shift


best = None
for mom in (0.0, 0.5, 0.9):
    for a in np.concatenate([np.arange(0.05, 1.0, 0.05), np.arange(1.0, 4.01, 0.25)]):
        pred, fs = run(a, momentum=mom)
        sse = float(((pred - implied) ** 2).sum())
        r = float(np.corrcoef(pred, implied)[0, 1]) if pred.std() > 0 else float("nan")
        if best is None or sse < best[0]:
            best = (sse, a, mom, r, pred, fs)
sse, a, mom, r, pred, fs = best
sst0 = float((implied ** 2).sum())
print(f"\nBest unit-norm bias-only model: step a={a:.2f} logit units per pair, momentum={mom}; corr(predicted, implied pre-update margin) = {r:.3f}; "
      f"R^2 vs the 'no transfer' prediction (margin 0 at every step) = {1 - sse / sst0:.3f}")
print("sign agreement between predicted and implied margins:", int((np.sign(pred) == np.sign(implied)).sum()), "/ 119 (excluding the first step, which is exactly 0)")
for mom in (0.0, 0.9):
    rows = []
    for a_ in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        pr, _ = run(a_, momentum=mom)
        rows.append((a_, round(float(np.corrcoef(pr, implied)[0, 1]), 3), round(1 - float(((pr - implied) ** 2).sum()) / sst0, 3)))
    print(f"  momentum {mom}: (a, corr, R^2)", rows)
obs_shift = {}
for rid, pi, c, r_ in sel:
    lc, lr = ref[json.dumps([rid, pi], separators=(",", ":"))]
    cmds = [cmd_name(x) for x in fin["commands"][rid]]
    obs_shift.setdefault(c, []).append(fin["scores"][rid][cmds.index(c)] - lc)
    obs_shift.setdefault(r_, []).append(fin["scores"][rid][cmds.index(r_)] - lr)
print("\nfinal per-command log-likelihood shift: bias-only model vs observed mean (final - SFT):")
for cmd in PATH:
    print(f"   {cmd:12s} model {fs[cmd]:+7.2f}   observed {np.mean(obs_shift[cmd]):+7.2f}")
print("corr(model shift, observed shift) over the 8 commands:", round(float(np.corrcoef([fs[c] for c in PATH], [np.mean(obs_shift[c]) for c in PATH])[0, 1]), 3))
# label-permutation control inside the toy model: random directions
rng = np.random.default_rng(7)
wins = []
for _ in range(2000):
    flips = rng.random(119) < 0.5
    theta0 = [l2.copy(), l3.copy(), ls.copy()]; theta = [x.copy() for x in theta0]; vel = [np.zeros(2), np.zeros(3), np.zeros(3)]
    for (rid, pi, c, r_), f in zip(sel, flips):
        if f: c, r_ = r_, c
        gc, gr = grad(theta, c), grad(theta, r_); g = [x - y for x, y in zip(gc, gr)]
        n = math.sqrt(sum(float((x ** 2).sum()) for x in g))
        for k in range(3):
            vel[k] = mom * vel[k] + (1 - mom) * g[k] / n; theta[k] = theta[k] + a * vel[k]
    lp = {cmd: logp(theta, cmd) for cmd in PATH}
    wins.append(max(lp, key=lp.get))
from collections import Counter
print(f"\nToy control (inference only): with the SAME pairs but coin-flip directions, same step size, the bias-only model ends with argmax:", dict(Counter(wins).most_common()))
