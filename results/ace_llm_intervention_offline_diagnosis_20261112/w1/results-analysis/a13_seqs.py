import numpy as np, pandas as pd
from collections import Counter
from common import *
cells, curves, meta = load()
for p in ['learned_ranker', 'frozen_llm', 'menu_coverage']:
    print('---', p)
    for w in range(24):
        r = curves[(curves.policy_id == p) & (curves.w == w)].sort_values('step')
        print('  w%02d %-8s %-13s %-12s %s' % (w, meta.graph_stratum[w], meta.cost_stratum[w], meta.regime[w], ' '.join(ASHORT[a] for a in r.action_id.values[1:])))
# SFT: position of non-x- actions
pos = {'x+': [], 'y-': [], 'obs': []}
for w in range(24):
    r = curves[(curves.policy_id == 'sft_llm') & (curves.w == w)].sort_values('step')
    for i, a in enumerate(r.action_id.values[1:]):
        if a != 'set_x_neg':
            pos[ASHORT[a]].append((w, i + 1, int(r.acquisition_cost_units.values[i]), meta.graph_stratum[w]))  # acquisition cost BEFORE the action
print('\nSFT non-x- actions: (world, step index, acquisition cost already paid before the action, graph)')
for k, v in pos.items():
    print(' ', k, len(v), v)
    print('     paid-before distribution:', dict(sorted(Counter(x[2] for x in v).items())), ' by graph:', dict(Counter(x[3] for x in v)))
# ranker: when does it switch sign
print('\nranker: step positions of x- (acquisition cost paid before action) pooled:')
c = Counter()
for w in range(24):
    r = curves[(curves.policy_id == 'learned_ranker') & (curves.w == w)].sort_values('step')
    for i, a in enumerate(r.action_id.values[1:]):
        if a == 'set_x_neg': c[i + 1] += 1
print('   x- count by step index:', dict(sorted(c.items())))
