"""Action structure of the strong numerical policies (and the others for reference)."""
import numpy as np, pandas as pd
from collections import Counter
from common import *
cells, curves, meta = load()
pd.set_option('display.width', 250)
acq = curves[curves.step > 0].copy()
acq = acq.join(meta[['graph_stratum', 'cost_stratum', 'regime', 'noise', 'prior']], on='w')
seqs = {(w, p): list(r.sort_values('step').action_id.values) for (w, p), r in acq.groupby(['w', 'policy_id'])}

print('=== full sequences, target_risk and local_information (one row per system)')
for p in ['target_risk', 'local_information']:
    print('---', p)
    for g in ['chain', 'fork', 'collider']:
        for c in ['uniform', 'heterogeneous']:
            for w in meta.index[(meta.graph_stratum == g) & (meta.cost_stratum == c)]:
                print('  %-8s %-13s %-12s w%02d n=%2d | %s' % (g, c, meta.regime[w], w, len(seqs[(w, p)]), ' '.join(ASHORT[a] for a in seqs[(w, p)])))

def dist(df, by):
    t = pd.crosstab([df[b] for b in by] if isinstance(by, list) else df[by], df.action_id)
    t = t.reindex(columns=ACTIONS, fill_value=0)
    t.columns = [ASHORT[a] for a in ACTIONS]
    t['n'] = t.sum(axis=1)
    return t
for p in ['target_risk', 'local_information']:
    d = acq[acq.policy_id == p]
    print('\n=== %s action counts by graph' % p); print(dist(d, 'graph_stratum').to_string())
    print('=== %s action counts by cost menu' % p); print(dist(d, 'cost_stratum').to_string())
    print('=== %s action counts by regime' % p); print(dist(d, 'regime').to_string())
    print('=== %s action counts by graph x cost' % p); print(dist(d, ['graph_stratum', 'cost_stratum']).to_string())
    print('=== %s action shares (row %%) by graph x cost' % p)
    t = dist(d, ['graph_stratum', 'cost_stratum']); print((t.drop(columns='n').div(t.n, axis=0) * 100).round(1).to_string())
    # cost-unit shares (budget spent per action)
    cu = d.copy(); cu['cost'] = cu.groupby(['w']).total_cost_units.diff().fillna(cu.total_cost_units - 8)
    print('=== %s action counts by step-index block' % p)
    d2 = d.copy(); d2['block'] = pd.cut(d2.step, [0, 2, 4, 8, 12, 16, 24], labels=['1-2', '3-4', '5-8', '9-12', '13-16', '17-24'])
    print(dist(d2, 'block').to_string())
    print('=== %s first action by graph x cost (4 systems each)' % p)
    f = d[d.step == 1]; print(dist(f, ['graph_stratum', 'cost_stratum']).drop(columns='n').to_string())

def modal_agreement(p, keys, maxstep=24, loo=False):
    """Mean over (cell, step) of the share of systems in the cell choosing the cell's modal action at that step.
    Only steps where at least 2 systems in the cell are still acting are counted. loo=True: predict each system's
    action from the modal action of the OTHER systems in the cell at that step (ties broken toward the alphabetically
    first action; no data from the predicted system)."""
    hits = tot = 0
    per_cell = {}
    grp = meta.groupby(keys).groups
    for cell, ws in grp.items():
        ch = ct = 0
        for s in range(maxstep):
            acts = [(w, seqs[(w, p)][s]) for w in ws if len(seqs[(w, p)]) > s]
            if len(acts) < 2: continue
            if not loo:
                c = Counter(a for _, a in acts)
                ch += c.most_common(1)[0][1]; ct += len(acts)
            else:
                for w, a in acts:
                    c = Counter(b for v, b in acts if v != w)
                    top = max(c.values()); pred = sorted([k for k, v in c.items() if v == top])[0]
                    ch += (pred == a); ct += 1
        per_cell[cell] = (ch, ct)
        hits += ch; tot += ct
    return hits / tot, per_cell, tot

print('\n=== closeness to a fixed sequence: share of decisions equal to the cell-and-step modal action')
print('(in-sample modal share has a floor of 1/k for k systems in a cell; leave-one-out (LOO) has no such floor)')
rows = []
for p in POL:
    r = {'policy': SHORT[p]}
    for name, keys in [('graph', ['graph_stratum']), ('graph_x_cost', ['graph_stratum', 'cost_stratum']),
                       ('graph_x_cost_x_noise', ['graph_stratum', 'cost_stratum', 'noise']),
                       ('graph_x_cost_x_prior', ['graph_stratum', 'cost_stratum', 'prior'])]:
        a, _, n = modal_agreement(p, keys); b, _, _ = modal_agreement(p, keys, loo=True)
        r[name + '_modal'] = a; r[name + '_loo'] = b
    # constant-action predictability: share of the modal action per graph x cost cell, ignoring step
    d = acq[acq.policy_id == p]
    t = pd.crosstab([d.graph_stratum, d.cost_stratum], d.action_id)
    r['graphcost_constant_action_share'] = t.max(axis=1).sum() / t.values.sum()
    r['global_constant_action_share'] = d.action_id.value_counts().iloc[0] / len(d)
    rows.append(r)
R = pd.DataFrame(rows).set_index('policy')
print(R.round(3).to_string())
R.to_csv('fixed_sequence_agreement.csv')

for p in ['target_risk', 'local_information']:
    a, pc, _ = modal_agreement(p, ['graph_stratum', 'cost_stratum']); b, pcl, _ = modal_agreement(p, ['graph_stratum', 'cost_stratum'], loo=True)
    print('\n%s per graph x cost cell: in-sample modal share / LOO agreement' % p)
    for cell in pc:
        print('   %-22s modal %3d/%3d = %.3f ; LOO %3d/%3d = %.3f' % (cell, pc[cell][0], pc[cell][1], pc[cell][0] / pc[cell][1], pcl[cell][0], pcl[cell][1], pcl[cell][0] / pcl[cell][1]))
    # per step
    print('  per-step modal share over the six graph x cost cells (steps 1..16):')
    line = []
    for s in range(16):
        h = t_ = 0
        for cell, ws in meta.groupby(['graph_stratum', 'cost_stratum']).groups.items():
            acts = [seqs[(w, p)][s] for w in ws if len(seqs[(w, p)]) > s]
            if len(acts) < 2: continue
            h += Counter(acts).most_common(1)[0][1]; t_ += len(acts)
        line.append('%d:%.2f' % (s + 1, h / t_))
    print('   ', ' '.join(line))
    # modal sequences
    print('  modal action per step, by graph x cost (ties shown with /):')
    for cell, ws in meta.groupby(['graph_stratum', 'cost_stratum']).groups.items():
        out = []
        for s in range(14):
            acts = [seqs[(w, p)][s] for w in ws if len(seqs[(w, p)]) > s]
            c = Counter(acts); top = max(c.values())
            out.append('/'.join(sorted(ASHORT[k] for k, v in c.items() if v == top)) + '(%d/%d)' % (top, len(acts)))
        print('   %-24s %s' % (cell, ' '.join(out)))

# multiset view: the conjugate posterior ignores order, so compare action-count vectors (first 12 cost units) across systems of a cell
print('\n=== action COUNT vectors within the first 12 acquisition cost units (order ignored), target_risk and local_information')
for p in ['target_risk', 'local_information']:
    for cell, ws in meta.groupby(['graph_stratum', 'cost_stratum']).groups.items():
        print('  ', p, cell)
        for w in ws:
            r = acq[(acq.policy_id == p) & (acq.w == w) & (acq.acquisition_cost_units <= 12)]
            c = Counter(r.action_id)
            print('      w%02d %-12s %s' % (w, meta.regime[w], ' '.join('%s:%d' % (ASHORT[a], c[a]) for a in ACTIONS if c[a])))
