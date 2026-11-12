# Supporting calculations for the advantage-theory agenda. Pure arithmetic, no project code, no data.
import math
from statistics import NormalDist
N = NormalDist()
def z(p): return N.inv_cdf(p)

print("== 1. Paired SDs implied by saved t statistics (n = 24 systems) ==")
def sd_from(mean_log, t, n=24): return abs(mean_log) * math.sqrt(n) / abs(t)
rows = [("design quality, local info x1.02, t=2.8", math.log(1.02), 2.8),
        ("design quality, coverage x1.35, t=8.6", math.log(1.35), 8.6),
        ("design quality, random x1.42, t=7.3", math.log(1.42), 7.3),
        ("design quality, DPO x2.10, t=7.8", math.log(2.10), 7.8),
        ("final posterior risk, TR-localinfo -0.055, t=-2.6", 0.055, 2.6),
        ("final posterior risk, TR-coverage -0.319, t=-8.1", 0.319, 8.1),
        ("final posterior risk, TR-random -0.347, t=-6.6", 0.347, 6.6),
        ("final posterior risk, TR-ranker -0.523, t=-9.2", 0.523, 9.2)]
for name, m, t in rows:
    print(f"  {name:55s} mean {m:.4f}  SD {sd_from(m,t):.4f}  CV {sd_from(m,t)/m:.2f}")

print("\n== 2. Systems needed, paired two-sided test ==")
def n_needed(effect, sd, alpha=0.05, power=0.8, k=1):
    za = z(1 - alpha / (2 * k)); zb = z(power)
    return ((za + zb) * sd / effect) ** 2
for eff_pct in (2, 5, 7.5, 10, 15, 20):
    eff = math.log(1 + eff_pct / 100)
    out = []
    for sd in (0.035, 0.10, 0.15, 0.20, 0.25):
        out.append(f"SD{sd}:{math.ceil(n_needed(eff, sd))}")
    print(f"  effect {eff_pct:4.1f}% (log {eff:.4f}) 80% power alpha .05: " + "  ".join(out))
print("  Holm k=3, 90% power:")
for eff_pct in (5, 7.5, 10):
    eff = math.log(1 + eff_pct / 100)
    print(f"   effect {eff_pct}%: " + "  ".join(f"SD{sd}:{math.ceil(n_needed(eff, sd, power=0.9, k=3))}" for sd in (0.10, 0.15, 0.20, 0.25)))

print("\n== 3. Omission term: simple linear head y = a + b x, balanced design at +-s, prior ignored ==")
# prediction variance at x_q: sigma^2/n * (1 + x_q^2 / s^2)
for s in (1.0, 2.0):
    v = {xq: 1 + xq**2 / s**2 for xq in (-1, 0, 1)}
    print(f"  design at +-{s}: relative prediction variance at x=-1,0,1: {v[-1]:.3f} {v[0]:.3f} {v[1]:.3f}; mean {sum(v.values())/3:.3f}")
print("  ratio of mean prediction variance (+-2 vs +-1):", round((1+0.25*2/3)/(1+2/3), 3))

print("\n== 4. Prior-mean shift: squared-bias share for one slope, n rows of unit second moment ==")
# posterior variance ~ 1/(1/tau2 + n s2/sigma2); bias = V*delta/tau2; ratio bias^2/var = V*delta^2/tau2^2
for sigma2 in (0.25, 1.0):
    for n in (0, 2, 8, 16, 32):
        V = 1 / (1 + n / sigma2)
        for delta in (0.75, 1.5, 3.0):
            ratio = V * delta**2
            pass
        print(f"  sigma2={sigma2} n={n:2d}: V={V:.4f}  bias^2/var at delta 0.75,1.5,3.0 = " +
              ", ".join(f"{V*d*d:.3f}" for d in (0.75, 1.5, 3.0)))

print("\n== 5. Absent-edge (sparsity) information, design-only, shared Gaussian learner ==")
# two coefficients b1 (present) and b2 (truly 0) for a query c=(1,1); budget n split k on b1-probing actions, n-k on b2.
# Probing b1 with x2 clamped at 0 gives info only on b1; probing b2 with x1 clamped at 0 only on b2.
def risk_true(k, n, sigma2=1.0, tau2=1.0):
    v1 = 1 / (1 / tau2 + k / sigma2)
    v2 = 1 / (1 / tau2 + (n - k) / sigma2)
    # true error on b2 when b2=0 and prior mean 0: only noise part, V2^2 * (n-k)/sigma2
    e2 = v2**2 * (n - k) / sigma2
    return v1 + e2, v1 + v2
for sigma2 in (0.25, 1.0):
    for n in (4, 8, 16, 24):
        best_unif = min(range(n + 1), key=lambda k: risk_true(k, n, sigma2)[1])
        best_inf = min(range(n + 1), key=lambda k: risk_true(k, n, sigma2)[0])
        ru = risk_true(best_unif, n, sigma2)[0]; ri = risk_true(best_inf, n, sigma2)[0]
        print(f"  sigma2={sigma2} n={n:2d}: uninformed splits k={best_unif}, informed k={best_inf}; true risk {ru:.4f} vs {ri:.4f}; ratio {ri/ru:.3f}")

print("\n== 6. Confirmation arithmetic ==")
for n_sys, reps, llm_arms, dec in ((120, 2, 2, 24), (120, 4, 2, 24), (96, 2, 2, 24)):
    for t in (0.4, 0.7):
        s = n_sys * reps * llm_arms * dec * t
        print(f"  {n_sys} systems x {reps} reps x {llm_arms} LLM arms x {dec} decisions x {t}s = {s:.0f} s = {s/3600:.2f} h")
