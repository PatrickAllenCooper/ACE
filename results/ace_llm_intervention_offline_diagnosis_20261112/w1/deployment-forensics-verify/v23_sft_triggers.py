"""What does the supervised adapter's x+ / y- / stop depend on: remaining budget, decision index, prompt length, posterior? Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
from scipy import stats
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb')); ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
sft = sorted([r for r in rows if r['arm'] == 'sft_llm'], key=lambda r: (r['world'], r['d']))
rem = lambda r: r['envelope']['state']['remaining_units']
for act in ['set_x_pos', 'set_y_neg', 'observe_all', 'STOP']:
    R = [r for r in sft if r['action'] == act]
    pt = np.array([r['prompt_tokens'] for r in R]); dd = np.array([r['d'] for r in R]); rr = np.array([rem(r) for r in R])
    print('%-5s n=%2d | d: mean %.1f sd %.2f range %d-%d | remaining: mean %.1f sd %.2f range %d-%d | prompt tokens: mean %.0f sd %.0f range %d-%d' % (
        SHORT[act], len(R), dd.mean(), dd.std(), dd.min(), dd.max(), rr.mean(), rr.std(), rr.min(), rr.max(), pt.mean(), pt.std(), pt.min(), pt.max()))
# Compare dispersion: coefficient of variation of d, remaining, tokens at first x+ per world
first = {}
for r in sft:
    if r['action'] == 'set_x_pos' and r['world'] not in first: first[r['world']] = r
print('first x+ per world: (world, cost, d, remaining, tokens, #x- so far)')
for w in sorted(first):
    r = first[w]; hist = r['envelope']['state']['history']
    print('  ', w, strata[w]['cost_stratum'][:4], strata[w]['graph_stratum'][:4], r['d'], rem(r), r['prompt_tokens'], sum(h['action_id'] == 'set_x_neg' for h in hist), sum(h['action_id'] == 'set_y_neg' for h in hist))
# is x+ timing explained by remaining budget or by decision index? uniform vs heterogeneous menus separate them only when y- (cost 2) was bought
# logistic-style tabulation: P(non x-) by remaining and by cost stratum
tab = collections.defaultdict(lambda: [0, 0])
for r in sft:
    if r['action'] == 'STOP': continue
    k = (rem(r)); tab[k][0] += r['action'] == 'set_x_pos'; tab[k][1] += 1
print('P(x+) by remaining:', {k: '%d/%d' % tuple(v) for k, v in sorted(tab.items()) if v[0]})
tabd = collections.defaultdict(lambda: [0, 0])
for r in sft:
    if r['action'] == 'STOP': continue
    tabd[r['d']][0] += r['action'] == 'set_x_pos'; tabd[r['d']][1] += 1
print('P(x+) by decision index:', {k: '%d/%d' % tuple(v) for k, v in sorted(tabd.items()) if v[0]})
# posterior-related features at x+ vs at matched non-x+ decisions (same remaining 11-13)
def feats(r):
    st = r['envelope']['state']; sc = ref[r['envelope_sha256']]
    S = {a: v[0] for a, v in sc['ref'].items()}
    return dict(risk=sc['before'], gain_xpos=S['set_x_pos'] - S['set_x_neg'], best_is_xpos=max(S, key=S.get) == 'set_x_pos', tokens=r['prompt_tokens'])
A = [feats(r) for r in sft if 11 <= rem(r) <= 13 and r['action'] == 'set_x_pos']
B = [feats(r) for r in sft if 11 <= rem(r) <= 13 and r['action'] == 'set_x_neg']
print('at remaining 11-13: x+ chosen n=%d, x- chosen n=%d' % (len(A), len(B)))
for k in ['risk', 'gain_xpos', 'tokens']:
    a = np.array([f[k] for f in A]); b = np.array([f[k] for f in B])
    print('   %-10s x+ chosen: mean %.5f ; x- chosen: mean %.5f ; Mann-Whitney p=%.3f' % (k, a.mean(), b.mean(), stats.mannwhitneyu(a, b).pvalue))
print('   reference TR prefers x+ over x- : when SFT chose x+ %d/%d ; when SFT chose x- %d/%d' % (sum(f['gain_xpos'] > 0 for f in A), len(A), sum(f['gain_xpos'] > 0 for f in B), len(B)))
# overall: how often does TR prefer x+ over x- on SFT states by block (base rate)
for lo, hi in [(0, 3), (4, 9), (10, 14), (15, 23)]:
    R = [r for r in sft if lo <= r['d'] <= hi and r['action'] != 'STOP']
    g = [feats(r)['gain_xpos'] > 0 for r in R]
    print('   SFT states d %d-%d: reference TR prefers x+ over x- in %d/%d' % (lo, hi, sum(g), len(g)))
# DPO: same base rate on its states
dpo = [r for r in rows if r['arm'] == 'dpo_llm']
for lo, hi in [(0, 3), (4, 9), (10, 14), (15, 23)]:
    R = [r for r in dpo if lo <= r['d'] <= hi]
    g = [(ref[r['envelope_sha256']]['ref']['set_x_pos'][0] - ref[r['envelope_sha256']]['ref']['set_x_neg'][0]) > 0 for r in R]
    print('   DPO states d %d-%d: reference TR prefers x+ over x- in %d/%d' % (lo, hi, sum(g), len(g)))
