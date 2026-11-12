"""Model-internal preposterior distribution of the DECLARED endpoint (capped first crossing of realized NMSE <= 0.04)
for fixed action SEQUENCES, from the saved post-initialization public states. 'Truth' in each fantasy draw is a
parameter draw from the saved public posterior; no private world descriptor and no environment response is used.
Purpose: predicted signal-to-noise of the declared endpoint for design differences of known size."""
import math, json
import numpy as np, pandas as pd
from acelib import *
import static_designs as sd
K = 4000
sd.K = K
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 400)
IDX = sd.IDX

def paths(heads, parents, noise, seq, cost, seed):
    """Return per-draw arrays: capped first-crossing acquisition cost, final realized NMSE, final posterior risk."""
    b = sd.Batch(heads, parents, noise, seed)
    # truth structural means per query per draw
    truth = []
    for q in QUERY_CLAMPS:
        vals = {}
        for n in NODES:
            if n in q: vals[n] = np.full(K, q[n]); continue
            phi = np.stack([np.ones(K)] + [vals[p] for p in parents[n]], 1)
            vals[n] = np.einsum("kd,kd->k", phi, b.theta[n])
        truth.append(vals["Z"])
    truth = np.stack(truth, 1)
    def nm():
        m = {n: np.linalg.solve(b.P[n], b.r[n][:, :, None])[:, :, 0] for n in NODES}
        preds = []
        for q in QUERY_CLAMPS:
            vals = {}
            for n in NODES:
                if n in q: vals[n] = np.full(K, q[n]); continue
                phi = np.stack([np.ones(K)] + [vals[p] for p in parents[n]], 1)
                vals[n] = np.einsum("kd,kd->k", phi, m[n])
            preds.append(vals["Z"])
        return (((np.stack(preds, 1) - truth) / SCALE) ** 2).mean(1)
    first = np.full(K, 24.0)
    e = nm(); hit = e <= 0.04; first[hit] = 0; crossed = hit.copy()
    spent = 0
    for a in seq:
        b.add(a); spent += action_cost(a, cost)
        e = nm(); hit = (e <= 0.04) & ~crossed
        first[hit] = spent; crossed |= hit
    return first, e, b.risk()

plan = pd.read_pickle("static_designs.pkl").set_index("world")
S, SEQ = {}, {}
for wi in range(24):
    arm = load_arm(0, wi)
    hist = arm["terminal"]["final_state"]["history"]
    parents = graph_parents(arm["learner"]); noise = arm["learner"]["noise_variances"]
    S[wi] = (posterior(parents, noise, 1.0, hist[:8]), parents, noise)
    SEQ[wi] = {pol: [h["action_id"] for h in load_arm(pi, wi)["terminal"]["final_state"]["history"][8:]] for pi, pol in enumerate(POLICIES)}
res = {}
names = ["TRseq", "LIseq", "coverage", "randomseq", "rankerseq", "obs24(frozen)", "xneg24(DPO)"]
for wi in range(24):
    cost = meta(wi)[2]
    seqs = {"TRseq": SEQ[wi]["target_risk"], "LIseq": SEQ[wi]["local_information"], "coverage": SEQ[wi]["menu_coverage"],
            "randomseq": SEQ[wi]["uniform_random"], "rankerseq": SEQ[wi]["learned_ranker"],
            "obs24(frozen)": ["observe_all"] * 24, "xneg24(DPO)": ["set_x_neg"] * 24}
    for name, seq in seqs.items():
        res[(wi, name)] = paths(*S[wi], seq, cost, 4242 + wi)   # common random numbers across designs within a system
    print(wi, {n: round(float(res[(wi, n)][0].mean()) + 8, 2) for n in names}, flush=True)

print("\n== Model-expected declared endpoint (mean capped total cost over 24 systems), per fixed sequence ==")
out = []
for n in names:
    cap = np.mean([res[(wi, n)][0].mean() + 8 for wi in range(24)])
    p0 = np.mean([(res[(wi, n)][0] == 0).mean() for wi in range(24)])
    pcap = np.mean([(res[(wi, n)][0] == 24).mean() for wi in range(24)])
    fin_ok = np.sum([(res[(wi, n)][1] <= 0.04).mean() for wi in range(24)])
    ever = np.sum([(res[(wi, n)][0] < 24).mean() + ((res[(wi, n)][0] == 24) & (res[(wi, n)][1] <= 0.04)).mean() for wi in range(24)])
    fin = np.mean([res[(wi, n)][1].mean() for wi in range(24)])
    # per-system sd of a single-episode endpoint
    sd1 = math.sqrt(np.mean([res[(wi, n)][0].var(ddof=1) for wi in range(24)]))
    out.append(dict(design=n, exp_capped_total=cap, frac_cross_at_init=p0, frac_capped=pcap, exp_ever_of_24=ever, exp_final_ok_of_24=fin_ok, exp_final_nmse=fin, within_system_sd_cost=sd1))
print(pd.DataFrame(out).round(4).to_string(index=False))

print("\n== Predicted paired contrast on the declared endpoint with ONE episode per system (model-internal) ==")
print("   (a) coupled = same exogenous noise/parameter draw for both arms (not what the screen did);")
print("   (b) independent acquisition noise per arm, shared system and shared initial rows (what the screen did, approximated).")
rng = np.random.default_rng(99)
for a, b in [("TRseq", "coverage"), ("TRseq", "randomseq"), ("TRseq", "xneg24(DPO)"), ("TRseq", "obs24(frozen)"), ("TRseq", "LIseq"), ("xneg24(DPO)", "obs24(frozen)"), ("xneg24(DPO)", "randomseq")]:
    means, var_c, var_i = [], [], []
    for wi in range(24):
        A, B = res[(wi, a)][0], res[(wi, b)][0]
        means.append((A - B).mean())
        var_c.append((A - B).var(ddof=1))
        # independent-noise version: in each fantasy the two arms share theta (the system) and initial rows but have independent
        # acquisition noise. Approximate var by total variance decomposition using theta-shared pairing unavailable -> upper bound:
        var_i.append(A.var(ddof=1) + B.var(ddof=1) - 2 * 0.0)
    mean = np.mean(means)
    se_c = math.sqrt(np.sum(var_c)) / 24
    se_i = math.sqrt(np.sum(var_i)) / 24
    print(f"  {a:14s} - {b:14s}: expected mean diff {mean:+.3f} cost units ; predicted SE over 24 systems: coupled {se_c:.3f} (z={mean/se_c:+.2f}) ; uncoupled upper bound {se_i:.3f} (z={mean/se_i:+.2f})")
pd.to_pickle({k: (v[0].mean(), v[0].std(ddof=1), v[1].mean(), v[2].mean()) for k, v in res.items()}, "endpoint_power_model.pkl")
