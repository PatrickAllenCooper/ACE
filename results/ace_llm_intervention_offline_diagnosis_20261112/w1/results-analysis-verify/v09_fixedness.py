"""Verifier step 9: how fixed are target_risk / local_information action sequences?
Leave-one-out agreement with a per-cell positional modal sequence, under several cell definitions and with and without
sign normalization (far/near X extreme relative to the initial X-mean estimate), plus a within-run shuffle baseline.
Also: variants of the 'opens with the far X extreme' claim. Fresh code."""
import pandas as pd, numpy as np, pickle, collections, itertools
OUT = "/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/results-analysis-verify/"
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
curves = pd.read_csv(D + "curves.csv")
meta = pd.read_csv(OUT + "out_meta_mux.csv", index_col=0)
seqs = pickle.load(open(OUT + "out_seqs.pkl", "rb"))
W = meta.index.tolist()
curves["mux"] = (curves.prediction_0 - curves.prediction_2) / (curves.prediction_3 - curves.prediction_2)

# ---------- 'opens with far extreme' variants ----------
print("=== variants of 'opens with X at the farther extreme' ===")
for p in ("target_risk", "local_information"):
    v_strict_pure = v_strict_xy = v_firstX_any = v_firstX_pure = v_firstX_pure_current = 0
    exc = []
    for w in W:
        g = curves[(curves.world_id == w) & (curves.policy_id == p)].reset_index(drop=True)
        acts = g.action_id.tolist()[1:]
        mux_before = g.mux.tolist()[:-1]           # estimate in force when each action was chosen
        far0 = "set_x_neg" if meta.loc[w, "mux0"] > 0 else "set_x_pos"
        xval = {"set_x_neg": -1, "set_x_pos": 1, "set_xy": -1}
        farv0 = -1 if meta.loc[w, "mux0"] > 0 else 1
        v_strict_pure += acts[0] == far0
        v_strict_xy += xval.get(acts[0], 0) == farv0
        k_any = next((k for k, a in enumerate(acts) if a in xval), None)
        k_pure = next((k for k, a in enumerate(acts) if a in ("set_x_neg", "set_x_pos")), None)
        v_firstX_any += (k_any is not None and xval[acts[k_any]] == farv0)
        ok_pure = (k_pure is not None and acts[k_pure] == far0)
        v_firstX_pure += ok_pure
        if k_pure is not None:
            farcur = "set_x_neg" if mux_before[k_pure] > 0 else "set_x_pos"
            v_firstX_pure_current += acts[k_pure] == farcur
        if not ok_pure:
            exc.append((f"w{int(str(w)[-2:]):02d}", round(meta.loc[w, "mux0"], 4), acts[:4]))
    print(f"{p}: strict first action == far pure X extreme: {v_strict_pure}/24 | strict first action has X at far extreme (set_xy counted as X=-1): {v_strict_xy}/24 | "
          f"first X-setting action (incl. set_xy) at far extreme: {v_firstX_any}/24 | first pure X-extreme action is far (init estimate): {v_firstX_pure}/24 | same, using the estimate in force at that step: {v_firstX_pure_current}/24")
    print("    exceptions to 'first pure X-extreme action is far':", exc)

# ---------- positional agreement ----------
def loo_agreement(S, cells):
    """S: dict world -> list of tokens; cells: dict world -> cell label. Returns pooled fraction of positions where the
    held-out run's token equals the positional mode of the other runs in its cell (ties broken by first-seen order among sorted tokens)."""
    hit = tot = 0
    for w in S:
        others = [S[v] for v in S if v != w and cells[v] == cells[w]]
        if not others:
            continue
        for k, tok in enumerate(S[w]):
            col = [o[k] for o in others if len(o) > k]
            if not col:
                continue
            cnt = collections.Counter(col)
            best = max(cnt.values())
            mode = sorted([t for t, c in cnt.items() if c == best])[0]
            hit += (tok == mode); tot += 1
    return hit / tot, tot

def norm_sign(w, seq):
    far = "set_x_neg" if meta.loc[w, "mux0"] > 0 else "set_x_pos"
    near = "set_x_pos" if far == "set_x_neg" else "set_x_neg"
    return [("x_far" if a == far else "x_near" if a == near else a) for a in seq]

celldefs = {
    "graph x cost menu (6 cells of 4)": lambda w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"]),
    "graph (3 cells of 8)": lambda w: meta.loc[w, "graph_stratum"],
    "cost menu (2 cells of 12)": lambda w: meta.loc[w, "cost_stratum"],
    "graph x regime (12 cells of 2)": lambda w: (meta.loc[w, "graph_stratum"], meta.loc[w, "regime"]),
    "graph x cost x coefficient law (12 cells of 2)": lambda w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"], meta.loc[w, "regime"].split("_")[0]),
    "graph x cost x noise (12 cells of 2)": lambda w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"], meta.loc[w, "regime"].split("_")[1]),
    "all systems one cell (1 cell of 24)": lambda w: 0,
}
rng = np.random.default_rng(20261112)
NSH = 2000
print("\n=== leave-one-out positional agreement with a per-cell modal sequence ===")
res = []
for p in ("target_risk", "local_information", "menu_coverage", "uniform_random", "sft_llm"):
    for signnorm in (False, True):
        S = {w: (norm_sign(w, seqs[(w, p)]) if signnorm else list(seqs[(w, p)])) for w in W}
        for cname, cf in celldefs.items():
            cells = {w: cf(w) for w in W}
            a, tot = loo_agreement(S, cells)
            sh = []
            for _ in range(NSH if p in ("target_risk", "local_information") else 200):
                Sp = {w: list(rng.permutation(S[w])) for w in W}
                sh.append(loo_agreement(Sp, cells)[0])
            sh = np.array(sh)
            res.append(dict(policy=p, sign_normalized=signnorm, cell=cname, agreement=a, n_positions=tot, shuffle_mean=sh.mean(), shuffle_p95=np.quantile(sh, .95),
                            p_perm=(np.sum(sh >= a) + 1) / (len(sh) + 1)))
R = pd.DataFrame(res)
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
print(R.round(4).to_string(index=False))
R.to_csv(OUT + "out_fixedness.csv", index=False)

# ---------- additional descriptive measures for TR ----------
print("\n=== additional measures ===")
for p in ("target_risk", "local_information"):
    S = {w: seqs[(w, p)] for w in W}
    uniq = len({tuple(s) for s in S.values()})
    # pairwise positional agreement within graph x cost cells vs across
    within, across = [], []
    for a, b in itertools.combinations(W, 2):
        n = min(len(S[a]), len(S[b]))
        ag = np.mean([S[a][k] == S[b][k] for k in range(n)])
        same = (meta.loc[a, "graph_stratum"], meta.loc[a, "cost_stratum"]) == (meta.loc[b, "graph_stratum"], meta.loc[b, "cost_stratum"])
        (within if same else across).append(ag)
    # first-k prefix identity
    pref = {}
    for k in (1, 2, 3, 4):
        pref[k] = len({tuple(norm_sign(w, S[w])[:k]) for w in W})
    print(f"{p}: distinct full sequences {uniq}/24; mean pairwise positional agreement within graph x cost cells {np.mean(within):.3f} (n={len(within)}), across cells {np.mean(across):.3f} (n={len(across)});"
          f" distinct sign-normalized prefixes of length 1..4: {pref}")
    # agreement by step position (sign-normalized, graph x cost): early vs late
    Sn = {w: norm_sign(w, S[w]) for w in W}
    cells = {w: (meta.loc[w, "graph_stratum"], meta.loc[w, "cost_stratum"]) for w in W}
    for lo, hi in ((0, 3), (3, 8), (8, 16), (16, 24)):
        St = {w: Sn[w][lo:hi] for w in W}
        print(f"    sign-normalized LOO agreement, steps {lo}..{hi-1}: {loo_agreement(St, cells)[0]:.3f}")
