"""Detailed checks on own-trajectory regret claims. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
from scipy import stats
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
ev_ref = pickle.load(open(SCR + '/ev_ref.pkl', 'rb'))
ev_256 = pickle.load(open(SCR + '/ev_256.pkl', 'rb'))
nm = pickle.load(open(SCR + '/nmse_by_state.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
def block(d): return 0 if d <= 3 else 1 if d <= 7 else 2 if d <= 15 else 3
for label, ev in [('ref16384', ev_ref)] + [('rep256-%d' % k, e) for k, e in enumerate(ev_256)]:
    print('\n##', label)
    E = {a: [e for e in ev if e['arm'] == a] for a in ARMS}
    # DPO vs random per world
    pw = {a: np.array([np.mean([e['rel'] for e in E[a] if e['world'] == w]) for w in range(24)]) for a in ARMS}
    worse = int((pw['dpo_llm'] > pw['uniform_random']).sum())
    p = stats.wilcoxon(pw['dpo_llm'], pw['uniform_random']).pvalue
    print('DPO mean rel regret > random in %d/24 worlds; Wilcoxon p=%.5f' % (worse, p))
    for a2 in ['frozen_llm', 'sft_llm', 'learned_ranker', 'menu_coverage']:
        print('   DPO worse than %s in %d/24 worlds (Wilcoxon p=%.4f)' % (a2, int((pw['dpo_llm'] > pw[a2]).sum()), stats.wilcoxon(pw['dpo_llm'], pw[a2]).pvalue))
    # by decision block
    for a in ['dpo_llm', 'uniform_random', 'frozen_llm', 'sft_llm', 'learned_ranker', 'menu_coverage', 'local_information', 'target_risk']:
        bl = [np.nanmean([e['rel'] for e in E[a] if block(e['d']) == b]) for b in range(4)]
        print('   %-18s rel regret by block d0-3, 4-7, 8-15, 16-23: %s' % (a, ' '.join('%.3f' % x for x in bl)))
    # x- optimal on DPO states by decision index
    byd = collections.Counter(e['d'] for e in E['dpo_llm'] if e['xneg_best'])
    print('x- optimal on DPO states: total %d of %d; by d: %s' % (sum(byd.values()), len(E['dpo_llm']), [(d, byd[d]) for d in sorted(byd)]))
    rk = collections.Counter(e['xneg_rank'] for e in E['dpo_llm'])
    full = [e for e in E['dpo_llm'] if e['n'] == 8]
    print('x- rank on DPO states (all 576):', sorted(rk.items()), '| rank 7 or 8 of 8:', sum(1 for e in full if e['xneg_rank'] >= 7), 'of', len(full), 'full-menu states;',
          'last-or-second-last incl. restricted menus:', sum(1 for e in E['dpo_llm'] if e['xneg_rank'] >= e['n'] - 1))
    # same for SFT states
    byd2 = collections.Counter(e['d'] for e in E['sft_llm'] if e['xneg_best'])
    print('x- optimal on SFT states: total %d of %d' % (sum(byd2.values()), len(E['sft_llm'])))
# TR arm: saved-score gap below 2 se
tr = [r for r in rows if r['arm'] == 'target_risk']
below = 0; below_alt = 0
for r in tr:
    pm = r['meta']['planning_metadata']; sc = pm['risk_reduction_per_cost']; se = pm['monte_carlo_standard_errors']
    order = sorted(sc, key=lambda a: (-sc[a], a))
    if len(order) < 2: continue
    gap = sc[order[0]] - sc[order[1]]
    below += gap < 2 * (se[order[0]] ** 2 + se[order[1]] ** 2) ** 0.5
    below_alt += gap < 2 * max(se[order[0]], se[order[1]])
print('\nTR arm saved 32-draw scores: top gap < 2 combined s.e. in %d of %d = %.1f%% (alt max-se def: %.1f%%)' % (below, len(tr), 100 * below / len(tr), 100 * below_alt / len(tr)))
# posterior risk vs private NMSE
pairs = [(ref[r['envelope_sha256']]['before'], nm[(r['arm'], r['world'], r['d'])]) for r in rows]
rb = np.array([p[0] for p in pairs]); nn = np.array([p[1] for p in pairs])
print('all 4371 decision states: mean posterior risk %.5f, mean private NMSE %.5f, ratio %.3f; median ratio of per-state risk/NMSE %.2f; Spearman %.3f' % (rb.mean(), nn.mean(), rb.mean() / nn.mean(), np.median(rb / nn), stats.spearmanr(rb, nn)[0]))
for a in ARMS:
    sel = [i for i, r in enumerate(rows) if r['arm'] == a]
    print('   %-18s ratio mean risk / mean NMSE = %.3f' % (a, rb[sel].mean() / nn[sel].mean()))
u = {}
for r in rows: u[r['envelope_sha256']] = (ref[r['envelope_sha256']]['before'], nm[(r['arm'], r['world'], r['d'])])
ub = np.array([v[0] for v in u.values()]); un = np.array([v[1] for v in u.values()])
print('unique states (4203): ratio %.3f' % (ub.mean() / un.mean()))
# Spearman of arm-level regret vs endpoints
cost = dict(uniform_random=14.250, menu_coverage=12.458, local_information=11.958, target_risk=11.625, learned_ranker=13.583, frozen_llm=13.625, sft_llm=14.625, dpo_llm=15.167)
fin = dict(uniform_random=0.04113, menu_coverage=0.01974, local_information=0.01800, target_risk=0.01722, learned_ranker=0.03572, frozen_llm=0.02008, sft_llm=0.02708, dpo_llm=0.03642)
ever = dict(uniform_random=21, menu_coverage=22, local_information=23, target_risk=23, learned_ranker=21, frozen_llm=22, sft_llm=21, dpo_llm=19)
reg = {a: np.nanmean([e['rel'] for e in ev_ref if e['arm'] == a]) for a in ARMS}
print('Spearman(arm rel regret, mean capped cost) = %.3f; vs mean final NMSE = %.3f; vs ever-attained (neg) = %.3f' % (
    stats.spearmanr([reg[a] for a in ARMS], [cost[a] for a in ARMS])[0], stats.spearmanr([reg[a] for a in ARMS], [fin[a] for a in ARMS])[0],
    stats.spearmanr([reg[a] for a in ARMS], [-ever[a] for a in ARMS])[0]))
