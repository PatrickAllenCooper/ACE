import itertools, numpy as np, scipy, pandas as pd
from common import *
print('scipy', scipy.__version__, 'numpy', np.__version__, 'pandas', pd.__version__)
rng = np.random.default_rng(1)
for trial in range(6):
    k = rng.integers(3, 15)
    d = np.round(rng.normal(0.5, 2, size=k), 0 if trial % 2 else 3)
    nz = d[d != 0]
    obs = abs(nz.sum())
    cnt = 0; tot = 0
    for signs in itertools.product([1, -1], repeat=len(nz)):
        tot += 1
        if abs(np.dot(signs, nz)) >= obs - 1e-9: cnt += 1
    brute = cnt / tot if len(nz) else 1.0
    print(trial, len(nz), 'brute %.6f' % brute, 'fast %.6f' % signflip_exact(d))
# brute force on a real 24-vector with 11 nonzeros
cells, curves, meta = load()
cw = cells.pivot(index='w', columns='policy_id', values='capped_total_cost')
d = (cw['dpo_llm'] - cw['target_risk']).values.astype(float)
nz = d[d != 0]; obs = abs(nz.sum()); cnt = 0
for signs in itertools.product([1, -1], repeat=len(nz)):
    if abs(np.dot(signs, nz)) >= obs - 1e-9: cnt += 1
print('dpo - target_risk: nonzero', len(nz), 'brute', cnt / 2 ** len(nz), 'fast', signflip_exact(d), 'differences', d.astype(int).tolist())
