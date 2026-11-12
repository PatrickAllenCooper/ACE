"""Ranker contrast vs TR contrast; quality of SFT deviations and stops; constant-action baselines. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
from scipy import stats
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
ref = pickle.load(open(SCR + '/ref_scores.pkl', 'rb'))
nm = pickle.load(open(SCR + '/nmse_by_state.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json')); strata = [s['strata'] for s in camp['sources']]
S = lambda r, k='ref': {a: v[0] for a, v in (ref[r['envelope_sha256']][k] if k == 'ref' else ref[r['envelope_sha256']]['rep256'][k]).items()}
# ---- ranker contrast
rk = sorted([r for r in rows if r['arm'] == 'learned_ranker'], key=lambda r: (r['world'], r['d']))
def rcon(r):
    ids, sc = r['meta']['action_ids'], r['meta']['scores']
    return sc[ids.index('set_x_neg')] - sc[ids.index('set_x_pos')]
for k in ['ref', 0, 1]:
    tcon = np.array([S(r, k)['set_x_neg'] - S(r, k)['set_x_pos'] for r in rk]); rc = np.array([rcon(r) for r in rk]); dd = np.array([r['d'] for r in rk])
    m0 = dd == 0
    print('[TR scores: %s] ranker x-/x+ contrast vs TR contrast at d=0: Spearman %.3f (n=%d), sign agreement %d/24' % (k, stats.spearmanr(rc[m0], tcon[m0])[0], m0.sum(), (np.sign(rc[m0]) == np.sign(tcon[m0])).sum()))
    for lo, hi in [(0, 0), (1, 3), (4, 7), (8, 11), (12, 23)]:
        m = (dd >= lo) & (dd <= hi)
        print('     d %2d-%2d: sign agreement %3d/%3d ; Spearman %.3f ; TR prefers x- in %d ; ranker prefers x- in %d' % (lo, hi, (np.sign(rc[m]) == np.sign(tcon[m])).sum(), m.sum(), stats.spearmanr(rc[m], tcon[m])[0], (tcon[m] > 0).sum(), (rc[m] > 0).sum()))
    m = dd >= 12
    nz = m & (np.abs(tcon) > 0)
    print('     d>=12: n=%d, TR contrast >0 in %d, ==0 in %d; ranker contrast>0 in %d' % (m.sum(), (tcon[m] > 0).sum(), (tcon[m] == 0).sum(), (rc[m] > 0).sum()))
# ---- SFT deviations
sft = sorted([r for r in rows if r['arm'] == 'sft_llm'], key=lambda r: (r['world'], r['d']))
dev = [r for r in sft if r['action'] not in ('set_x_neg', 'STOP')]
better = sum(S(r)[r['action']] > S(r)['set_x_neg'] for r in dev)
print('\nSFT deviations from x- (n=%d): chosen action has higher reference TR score than x- in %d; chosen is the TR optimum in %d' % (len(dev), better, sum(max(S(r), key=S(r).get) == r['action'] for r in dev)))
for act in ['set_x_pos', 'set_y_neg', 'observe_all']:
    dd_ = [r for r in dev if r['action'] == act]
    gain = [S(r)[act] - S(r)['set_x_neg'] for r in dd_]
    print('   %-4s n=%2d better than x- in %2d; mean score gain vs x- %.5f; mean rel regret of chosen %.3f vs of x- %.3f' % (SHORT[act], len(dd_), sum(g > 0 for g in gain), np.mean(gain),
          np.mean([(max(S(r).values()) - S(r)[act]) / max(S(r).values()) for r in dd_]), np.mean([(max(S(r).values()) - S(r)['set_x_neg']) / max(S(r).values()) for r in dd_])))
stops = [r for r in sft if r['action'] == 'STOP']
print('SFT stops n=%d: best available reference score >0 in %d; mean best score %.5f; NMSE at stop > 0.04 in %d; posterior risk at stop mean %.4f' % (
    len(stops), sum(max(S(r).values()) > 0 for r in stops), np.mean([max(S(r).values()) for r in stops]),
    sum(nm[(r['arm'], r['world'], r['d'])] > 0.04 for r in stops), np.mean([ref[r['envelope_sha256']]['before'] for r in stops])))
print('   stop-state posterior risk for stops with NMSE>0.04 vs <=0.04:', np.round([ref[r['envelope_sha256']]['before'] for r in stops if nm[(r['arm'], r['world'], r['d'])] > 0.04], 4),
      '|', np.round(sorted(ref[r['envelope_sha256']]['before'] for r in stops if nm[(r['arm'], r['world'], r['d'])] <= 0.04), 4))
# ---- constant-action baselines on each arm's own states and on TR's states
print('\nmean relative regret of constant actions evaluated on TR-arm states (560) and on pooled unique states:')
trs = [r for r in rows if r['arm'] == 'target_risk']
uniq = {}
for r in rows: uniq.setdefault(r['envelope_sha256'], r)
for label, RR in [('TR states', trs), ('pooled unique states', list(uniq.values()))]:
    out = []
    for act in ACTIONS:
        vals = [(max(S(r).values()) - S(r)[act]) / max(S(r).values()) for r in RR if act in S(r) and max(S(r).values()) > 0]
        out.append('%s %.3f' % (SHORT[act], np.mean(vals)))
    print('  ', label, '|', ' '.join(out))
    best = collections.Counter(SHORT[max(S(r), key=lambda a: (S(r)[a], -ACTIONS.index(a)))] for r in RR)
    print('      optimum distribution', dict(best.most_common()))
# TR optimum distribution by decision block on TR states
for b, (lo, hi) in enumerate([(0, 3), (4, 7), (8, 15), (16, 23)]):
    c = collections.Counter(SHORT[max(S(r), key=S(r).get)] for r in trs if lo <= r['d'] <= hi)
    print('   TR states d %d-%d optimum:' % (lo, hi), dict(c.most_common()))
# frozen: observe optimal how often on its own states by block
fr = [r for r in rows if r['arm'] == 'frozen_llm']
for lo, hi in [(0, 3), (4, 7), (8, 15), (16, 23)]:
    F = [r for r in fr if lo <= r['d'] <= hi]
    print('   frozen states d %d-%d: observe is reference optimum in %d/%d; observe rank mean %.2f' % (lo, hi, sum(max(S(r), key=S(r).get) == 'observe_all' for r in F), len(F),
          np.mean([sorted(S(r), key=lambda a: -S(r)[a]).index('observe_all') + 1 for r in F])))
