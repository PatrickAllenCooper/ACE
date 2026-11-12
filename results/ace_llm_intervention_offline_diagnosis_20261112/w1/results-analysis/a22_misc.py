import numpy as np, pandas as pd
from common import *
cells, curves, meta = load()
S = wide_steps(curves)
ini = S['dpo_llm'][:, 0]; inf = ini > THRESH
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
def capped_at(arr, cap):
    out = np.full(arr.shape[0], float(cap))
    for w in range(arr.shape[0]):
        idx = np.where(arr[w, :cap - INIT + 1] <= THRESH)[0]
        if len(idx): out[w] = INIT + idx[0]
    return out
c18 = {p: capped_at(S[p], 18) for p in POL}
A18 = {p: np.log10(S[p][:, 1:11]).mean(axis=1) for p in POL}
rng = np.random.default_rng(11)
def boot_ratio(tw, other, B=20000):
    idx = rng.integers(0, 24, size=(B, 24))
    num = (tw[idx] ** 2).mean(axis=1); den = other[idx].var(axis=1, ddof=1)
    r = num / den
    return np.nanquantile(r, [.025, .5, .975]), np.quantile(np.sqrt(num / 2), [.025, .5, .975])
for name, tw, oth in [('cap-18 cost', c18['dpo_llm'] - c18['sft_llm'], c18['dpo_llm'] - c18['target_risk']),
                      ('log-area over costs 9..18', A18['dpo_llm'] - A18['sft_llm'], A18['dpo_llm'] - A18['target_risk'])]:
    (lo, md, hi), (slo, smd, shi) = boot_ratio(tw, oth)
    print('%s: twin variance E[d^2] = %.3f ; (DPO - target_risk) variance = %.3f ; ratio %.2f (bootstrap over systems 95%% %.2f-%.2f) ; constant-policy replicate SD %.3f (95%% %.3f-%.3f) ; half-ratio (DPO arm noise alone) %.2f'
          % (name, (tw ** 2).mean(), oth.var(ddof=1), (tw ** 2).mean() / oth.var(ddof=1), lo, hi, np.sqrt((tw ** 2).mean() / 2), slo, shi, (tw ** 2).mean() / 2 / oth.var(ddof=1)))
# first action is x- for both DPO and SFT in all 24 systems: step-1 log error difference is pure noise
e1 = curves[curves.step == 1].pivot(index='w', columns='policy_id', values='normalized_mse')
a1 = curves[curves.step == 1].pivot(index='w', columns='policy_id', values='action_id')
print('\nDPO and SFT first action identical in %d of 24 systems (%s)' % ((a1.dpo_llm == a1.sft_llm).sum(), a1.dpo_llm.unique()))
d = np.log10(e1.dpo_llm) - np.log10(e1.sft_llm)
print('step-1 log10 error difference between the two identical first actions: SD %.3f ; |diff| median %.3f ; max %.3f ; crossing after step 1: DPO %d systems, SFT %d systems, both %d (informative only)'
      % (d.std(ddof=1), d.abs().median(), d.abs().max(), ((e1.dpo_llm <= THRESH) & inf).sum(), ((e1.sft_llm <= THRESH) & inf).sum(), ((e1.dpo_llm <= THRESH) & (e1.sft_llm <= THRESH) & inf).sum()))
# distribution of acquisition cost-to-target on informative systems
print('\ncapped acquisition cost (cost - 8) on the 14 informative systems, sorted:')
for p in POL:
    v = np.sort((cw[p].values - 8)[inf])
    print('  %-12s %s median %.1f ; crossed with the first response on %d ; within 3 units on %d' % (SHORT[p], v.tolist(), np.median(v), (v <= 1).sum(), (v <= 3).sum()))
# area envelope
aulc = {p: np.log10(S[p][:, 1:]).mean(axis=1) for p in POL}
M = np.array([aulc[p] for p in POL])
print('\nlog-area: target_risk mean %.4f ; best-of-8 envelope mean %.4f ; gap %.4f log10 units (x%.2f in geometric-mean error) ; target_risk is the per-system best on %d of 24 ; each policy best count: %s'
      % (aulc['target_risk'].mean(), M.min(axis=0).mean(), (aulc['target_risk'] - M.min(axis=0)).mean(), 10 ** (aulc['target_risk'] - M.min(axis=0)).mean(),
         (M.argmin(axis=0) == POL.index('target_risk')).sum(), {SHORT[POL[i]]: int((M.argmin(axis=0) == i).sum()) for i in range(8)}))
# count of systems on which each policy has the lowest final error
F = np.array([S[p][:, -1] for p in POL])
print('lowest final error counts:', {SHORT[POL[i]]: int((F.argmin(axis=0) == i).sum()) for i in range(8)})
# median final error by policy on all and informative
print('median final error all-24 / informative-14:', {SHORT[p]: (round(float(np.median(S[p][:, -1])), 5), round(float(np.median(S[p][inf, -1])), 5)) for p in POL})
