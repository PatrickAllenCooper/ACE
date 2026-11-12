import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections, math
import numpy as np
rows = pickle.load(open(SCR + '/all_rows.pkl', 'rb'))
def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values() if v)
for a in ARMS:
    c_all = collections.Counter(r['action'] for r in rows if r['arm'] == a)
    c_acq = collections.Counter(r['action'] for r in rows if r['arm'] == a and r['action'] != 'STOP')
    print('%-18s entropy incl. stop %.3f bits ; acquisitions only %.3f bits' % (a, H(c_all), H(c_acq)))
# per-arm tokens-per-response increment
by = collections.defaultdict(dict)
for r in rows:
    if 'prompt_tokens' in r: by[(r['arm'], r['world'])][r['d']] = r['prompt_tokens']
inc = collections.defaultdict(list)
for (arm, w), v in by.items():
    ds = sorted(v)
    for x, y in zip(ds, ds[1:]):
        if y == x + 1: inc[arm].append(v[y] - v[x])
for arm, v in inc.items(): print(arm, 'tokens added per response: mean %.2f median %.0f' % (np.mean(v), np.median(v)))
# latency vs prompt tokens: DPO slower? compare cuda ms per token
for arm in ['frozen_llm', 'sft_llm', 'dpo_llm']:
    R = [r for r in rows if r['arm'] == arm]
    c = np.array([r['cuda_ms'] for r in R]); t = np.array([r['prompt_tokens'] for r in R]); n = np.array([len(r['comp_ids']) for r in R])
    print(arm, 'median cuda ms %.1f ; median generated tokens %.0f ; median ms per generated token %.1f ; total cuda s %.1f' % (np.median(c), np.median(n), np.median(c / n), c.sum() / 1000))
