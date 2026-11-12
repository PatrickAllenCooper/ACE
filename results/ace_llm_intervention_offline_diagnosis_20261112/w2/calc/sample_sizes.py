"""Sample-size and operating-characteristic arithmetic for the evaluation agenda.
Pure standard library. Reads one saved table (read-only) for planning parameters."""
import csv, math, statistics as st, random
from statistics import NormalDist
N = NormalDist()
z = lambda p: N.inv_cdf(p)
SRC = ('/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/'
       'scratchpad/w1/numerics-protocol/out_endpoints_cpu_arms.csv')

def planning_table():
    rows = list(csv.DictReader(open(SRC)))
    d = {(r['policy'], int(r['world'])): r for r in rows}
    arms = ['local_information', 'menu_coverage', 'uniform_random', 'learned_ranker']
    specs = [('declared first crossing, realized 0.04 (capped total)', 'capped_real', float),
             ('cost to learner posterior risk <= 0.04 (capped total)', 'capped_post', float),
             ('log final posterior risk', 'final_post', lambda x: math.log(float(x))),
             ('mean log posterior risk over cost grid', 'auc_logpost', float),
             ('log final realized NMSE', 'final_nmse', lambda x: math.log(float(x))),
             ('mean log realized NMSE over cost grid', 'auc_lognmse', float)]
    print('Planning parameters, other arm minus target risk, 24 dev systems, one episode, independent noise')
    for label, key, tf in specs:
        tr = [tf(d[('target_risk', w)][key]) for w in range(24)]
        out = []
        for a in arms:
            o = [tf(d[(a, w)][key]) for w in range(24)]
            diff = [x - y for x, y in zip(o, tr)]
            m, s = st.mean(diff), st.stdev(diff)
            out.append('%s: mean %+.3f sd %.3f d %.2f' % (a[:6], m, s, m / s))
        print(' -', label, '|', '; '.join(out))
    random.seed(1)
    print('Ratio of arithmetic means of final error (other/TR), bootstrap SE of log ratio x sqrt(24) = implied per-system SD')
    for key in ('final_post', 'final_nmse'):
        tr = [float(d[('target_risk', w)][key]) for w in range(24)]
        for a in arms:
            o = [float(d[(a, w)][key]) for w in range(24)]
            bs = []
            for _ in range(4000):
                idx = [random.randrange(24) for _ in range(24)]
                bs.append(math.log(sum(o[i] for i in idx) / sum(tr[i] for i in idx)))
            print('   %-11s %-18s ratio %.3f implied SD %.3f' % (key, a, st.mean(o) / st.mean(tr), st.stdev(bs) * math.sqrt(24)))

def sizes():
    print('\nSystems needed (normal approximation, +2 for t), standardized paired effect d')
    for d_ in (0.2, 0.3, 0.35, 0.5):
        sup80 = (z(0.975) + z(0.8)) ** 2 / d_ ** 2 + 2
        sup90 = (z(0.975) + z(0.9)) ** 2 / d_ ** 2 + 2
        tost = (z(0.95) + z(0.9)) ** 2 / d_ ** 2      # true difference 0, alpha .05 each side, 80% power
        holm5 = (z(1 - 0.025 / 5) + z(0.8)) ** 2 / d_ ** 2 + 2
        print('  d=%.2f superiority(1-sided .025) 80%%: %.0f  90%%: %.0f | TOST(+-d) 80%%: %.0f | Holm worst step (m=5) 80%%: %.0f' % (d_, sup80, sup90, tost, holm5))
    for S, a, lab in ((96, 0.025 / 5, 'E7 Holm worst step, S=96'), (48, 0.025 / 5, 'E7 Holm worst step, S=48'),
                      (72, 0.025 / 2, 'E8 Holm over 2 families, S=72'), (48, 0.025, 'E9 stage A two-sided .05, S=48'),
                      (30, 0.025, 'E9 stage B, 30 blocks')):
        print('  detectable d at 80%%, %s: %.2f' % (lab, (z(1 - a) + z(0.8)) / math.sqrt(S)))

def promotion():
    se = 1 / math.sqrt(48)
    thr = -0.15
    print('\nE5 promotion rule (S=48 validation systems, promote if d_hat <= -0.15 and 80%% upper bound < 0), SE %.3f' % se)
    for true in (-0.3, -0.15, 0.0):
        p = N.cdf((thr - true) / se)
        print('  true d %+.2f: P(promote one candidate) %.2f' % (true, p))
    p0 = N.cdf((thr - 0) / se)
    print('  global null, best of 3 independent candidates: P(any promoted) %.2f' % (1 - (1 - p0) ** 3))

def seeds():
    print('\nSeed replication: P(at least 4 of 5 seeds favourable), true d = 0.3, per-seed SE 1/sqrt(96)')
    for tau in (0.0, 0.2, 0.3):
        sd = math.sqrt(1 / 96 + tau ** 2)
        p = N.cdf(0.3 / sd)
        print('  seed SD %.1f: P(seed favourable) %.3f, P(>=4/5) %.2f' % (tau, p, p ** 5 + 5 * p ** 4 * (1 - p)))

def gpu():
    t = 0.6  # s per LLM decision, brief: 0.4-0.7 s; no reasoning tokens
    T = 24
    items = {
        'E5 per iteration': 3 * 3 * 48 * T * t + 2 * 3 * 48 * T * t + 48 * T * t,
        'E6': 5 * 96 * T * t + 5 * 24 * T * t + 96 * T * t + 2 * 96 * T * t,
        'E7 (non-semantic)': 2 * 2 * 96 * T * t + 2 * 48 * T * t + 2 * 48 * T * t,
        'E7 semantic ladder extra': 5 * 2 * 48 * T * t,
        'E8 (2 families, prompts x1.5)': 2 * 72 * 3 * T * t * 1.5,
        'E9 stage A (16 decisions)': 48 * 4 * 3 * 16 * t,
    }
    print('\nLLM inference GPU-slice-seconds at 0.6 s per decision')
    tot = 0
    for k, v in items.items():
        tot += v
        print('  %-32s %7.0f s = %.2f h' % (k, v, v / 3600))
    print('  total (one E5 iteration)          %7.0f s = %.1f h' % (tot, tot / 3600))

if __name__ == '__main__':
    planning_table(); sizes(); promotion(); seeds(); gpu()
