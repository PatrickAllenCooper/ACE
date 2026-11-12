import math
from statistics import NormalDist
N = NormalDist()
z = lambda p: N.inv_cdf(p)
# 1. Power for a Spearman/Pearson correlation (Fisher z), two-sided alpha
def n_corr(rho, alpha, power):
    return ((z(1-alpha/2)+z(power))/math.atanh(rho))**2 + 3
for rho in (0.2, 0.3):
    for a, p in ((0.01, 0.8), (0.01, 0.85), (0.05, 0.9)):
        print(f"n for rho={rho} alpha={a} power={p}: {n_corr(rho,a,p):.0f}")
def power_corr(rho, n, alpha):
    se = 1/math.sqrt(n-3); return 1-N.cdf(z(1-alpha/2) - math.atanh(rho)/se)
print("power rho=0.2 n=300 alpha=.01:", round(power_corr(0.2,300,0.01),3))
se = 1/math.sqrt(297)
print("CI half-width near 0, n=300:", round(math.tanh(1.96*se),3))
print("lower CI when r=0.30, n=300:", round(math.tanh(math.atanh(0.3)-1.96*se),3))
# 2. Test-world sample sizes
for sd, hw in ((0.15,0.02),(0.20,0.02)):
    print(f"worlds for regret CI half-width {hw} at world-SD {sd}: {(1.96*sd/hw)**2:.0f}")
# paired log J ratio SDs implied by 24-system design-quality t values (numerics-protocol 7B)
for name, ratio, t in (("local_info",1.02,2.8),("coverage",1.35,8.6),("random",1.42,7.3),("ranker",1.59,8.6),("DPO",2.10,7.8)):
    se24 = math.log(ratio)/t; sd = se24*math.sqrt(24)
    print(f"{name}: SD(log ratio) ~ {sd:.3f}; 95% half-width at n=256: {1.96*sd/16:.4f}; at n=64: {1.96*sd/8:.4f}")
print("log(1.05) =", round(math.log(1.05),4))
# 3. Planner relabel cost from the verifier log: 4203 states in 499 s at 16384 draws (+5x256)
per_state = 499/4203
print("s/state at 16k draws:", round(per_state,4))
for nw in (32,128,512,2048):
    states = nw*65
    print(f"N={nw}: states~{states}, label CPU-h at 16k draws: {states*per_state/3600:.2f}")
valtest = 512*65
print("val+test states", valtest, "CPU-h", round(valtest*per_state/3600,2))
# 4. LLM training steps
for nw, spw, ep in ((512,24,2),(256,24,3),(128,24,3),(32,48,6)):
    steps = nw*spw*ep
    print(f"LLM corpus N={nw} x {spw} states x {ep} ep = {steps} steps -> {steps*0.05/3600:.2f}-{steps*0.2/3600:.2f} h at 0.05-0.2 s/step")
# 5. Forward-only E1/E2 counts
tables = 1858*3; edits = 300*12*3
tot = tables+edits
print("E1/E2 forwards:", tables, edits, tot, "Mac h at 0.3-1.5 s:", round(tot*0.3/3600,1), round(tot*1.5/3600,1), " CURC h at 0.12-0.4 s:", round(tot*0.12/3600,2), round(tot*0.4/3600,2))
# 6. tie rule false-strict rate check (t with 3 df, |t|>2)
# numeric integration of t3 tail
def t3_pdf(x): 
    return 2/(math.pi*math.sqrt(3)*(1+x*x/3)**2)
import itertools
s=0; dx=1e-4; x=2.0
while x<400: s+=t3_pdf(x)*dx; x+=dx
print("P(|t3|>2) ~", round(2*s,4))
# 7. Gibbs soft target: E_p[r] bound (K-1)*tau/e
for tau in (0.005,0.01,0.02):
    print(f"tau={tau}: worst-case E[r] <= {(7*tau/math.e):.4f}")
# 8. LoRA parameter counts for Qwen3-0.6B dims (hidden 1024, q out 2048, kv out 1024, mlp 3072, 28 layers)
def lora(r, mods):
    dims = {'q':(1024,2048),'k':(1024,1024),'v':(1024,1024),'o':(2048,1024),'gate':(1024,3072),'up':(1024,3072),'down':(3072,1024)}
    return 28*sum(r*(a+b) for m,(a,b) in dims.items() if m in mods)
print("rank8 q/v:", lora(8,{'q','v'}), " rank16 all:", lora(16,set('q k v o gate up down'.split())), " rank32 all:", lora(32,set('q k v o gate up down'.split())))
