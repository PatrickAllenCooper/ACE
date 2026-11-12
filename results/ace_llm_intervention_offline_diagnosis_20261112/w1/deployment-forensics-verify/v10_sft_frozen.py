"""SFT and frozen behaviour checks. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections
import numpy as np
from scipy import stats
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
camp = json.load(open(CAMP + '/campaign.json'))
strata = [s['strata'] for s in camp['sources']]
sft = sorted([r for r in rows if r['arm'] == 'sft_llm'], key=lambda r: (r['world'], r['d']))
acq = [r for r in sft if r['action'] != 'STOP']
non = [r for r in acq if r['action'] != 'set_x_neg']
print('SFT decisions', len(sft), 'acquisitions', len(acq), 'non-x-', len(non), {SHORT[k]: v for k, v in collections.Counter(r['action'] for r in non).items()})
print('non-x- decision index: min', min(r['d'] for r in non), 'distribution', sorted(collections.Counter(r['d'] for r in non).items()))
print('non-x- by remaining_units', sorted(collections.Counter(r['envelope']['state']['remaining_units'] for r in non).items()))
print('all SFT acquisitions by remaining_units', sorted(collections.Counter(r['envelope']['state']['remaining_units'] for r in acq).items()))
rem = lambda r: r['envelope']['state']['remaining_units']
a = sum(1 for r in non if 11 <= rem(r) <= 13); b = sum(1 for r in acq if 11 <= rem(r) <= 13)
c = len(non) - a; d = len(acq) - b
print('non-x- at remaining 11-13: %d/%d ; elsewhere %d/%d' % (a, b, c, d), 'Fisher p', stats.fisher_exact([[a, b - a], [c, d - c]])[1])
# by action and graph
for act in ['set_x_pos', 'set_y_neg', 'observe_all']:
    ws = collections.Counter(r['world'] for r in non if r['action'] == act)
    bygraph = collections.Counter(strata[w]['graph_stratum'] for w in ws)
    print(SHORT[act], 'worlds with >=1:', len(ws), 'by graph (worlds)', dict(bygraph), 'count by graph (acts)', dict(collections.Counter(strata[r['world']]['graph_stratum'] for r in non if r['action'] == act)),
          'by cost', dict(collections.Counter(strata[r['world']]['cost_stratum'] for r in non if r['action'] == act)),
          'by regime', dict(collections.Counter(strata[r['world']]['regime'] for r in non if r['action'] == act)),
          'remaining', sorted(collections.Counter(rem(r) for r in non if r['action'] == act).items()),
          'd', sorted(collections.Counter(r['d'] for r in non if r['action'] == act).items()))
# per-world sequences
for w in range(24):
    seq = [(r['d'], rem(r), SHORT[r['action']]) for r in sft if r['world'] == w]
    s = ' '.join('%s' % x[2] for x in seq)
    print(w, strata[w]['graph_stratum'][:4], strata[w]['cost_stratum'][:4], strata[w]['regime'], '|', s)
# stops
stops = [r for r in sft if r['action'] == 'STOP']
print('stops', len(stops), 'remaining at stop', sorted(collections.Counter(rem(r) for r in stops).items()), 'worlds', sorted(r['world'] for r in stops))
allrem = collections.Counter(rem(r) for r in sft)
for k in sorted(allrem):
    s = sum(1 for r in stops if rem(r) == k)
    if k <= 6: print('  remaining', k, 'stops', s, 'of', allrem[k], 'decisions')
print('  decisions at remaining>=4:', sum(v for k, v in allrem.items() if k >= 4), 'stops there', sum(1 for r in stops if rem(r) >= 4))
print('world without stop:', [w for w in range(24) if w not in {r['world'] for r in stops}])
# legal commands at stop decisions, terminal flag
print('terminal flags at stops', collections.Counter(r['envelope']['state']['terminal'] for r in stops), 'legal_commands at stops', collections.Counter(r['legal_commands'] for r in stops))
# cost sensitivity: action mix by cost stratum
for cs in ['uniform', 'heterogeneous']:
    c = collections.Counter(r['action'] for r in acq if strata[r['world']]['cost_stratum'] == cs)
    print(cs, {SHORT[k]: v for k, v in c.most_common()}, 'n', sum(c.values()))
tab = [[sum(1 for r in acq if strata[r['world']]['cost_stratum'] == cs and (r['action'] in ('set_y_neg',)) == yy) for yy in (True, False)] for cs in ['uniform', 'heterogeneous']]
print('y- by cost stratum table [[y-,other] uniform; heterogeneous]', tab, 'Fisher p', stats.fisher_exact(tab)[1])
tab2 = [[sum(1 for r in acq if strata[r['world']]['cost_stratum'] == cs and (r['action'] != 'set_x_neg') == yy) for yy in (True, False)] for cs in ['uniform', 'heterogeneous']]
print('non-x- by cost stratum', tab2, 'Fisher p', stats.fisher_exact(tab2)[1])
# spent units for SFT
spent = sum(r['join']['charged_units'] for r in sft)
print('SFT acquisition charged units', spent, '+ init 192 =', spent + 192)
# frozen
fr = [r for r in rows if r['arm'] == 'frozen_llm' and r['action'] != 'observe_all']
print('frozen non-observe:', [(r['world'], r['d'], r['ordinal'], r['action'], strata[r['world']], rem(r)) for r in fr])
