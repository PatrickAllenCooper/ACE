"""Prompt length, prompt composition, latency and command token tree. Tokenizer file only; no model weights."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import pickle, collections, glob
import numpy as np
from tokenizers import Tokenizer
snap = glob.glob('/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/')[0]
tok = Tokenizer.from_file(snap + 'tokenizer.json')
rows = pickle.load(open(SCR + '/llm_rows.pkl', 'rb'))
pt = np.array([r['prompt_tokens'] for r in rows])
print('prompt tokens: min %d max %d median %.0f mean %.1f  (n=%d)' % (pt.min(), pt.max(), np.median(pt), pt.mean(), len(pt)))
for arm in ['frozen_llm', 'sft_llm', 'dpo_llm']:
    a = np.array([r['prompt_tokens'] for r in rows if r['arm'] == arm])
    c = np.array([r['cuda_ms'] for r in rows if r['arm'] == arm]); e = np.array([r['elapsed'] for r in rows if r['arm'] == arm])
    print(arm, 'tokens min %d max %d median %.0f | cuda_ms median %.1f mean %.1f p90 %.1f sum %.0f | elapsed median %.3f' % (a.min(), a.max(), np.median(a), np.median(c), c.mean(), np.percentile(c, 90), c.sum(), np.median(e)))
# tokens per additional response: slope of prompt tokens vs number of history rows
hist = np.array([len(r['envelope']['state']['history']) for r in rows])
print('history rows range', hist.min(), hist.max())
sl = np.polyfit(hist, pt, 1); print('OLS tokens = %.2f * rows + %.1f' % (sl[0], sl[1]))
# within-episode increments
incs = []
by = collections.defaultdict(dict)
for r in rows: by[(r['arm'], r['world'])][r['d']] = r['prompt_tokens']
for k, v in by.items():
    ds = sorted(v)
    for a, b in zip(ds, ds[1:]):
        if b == a + 1: incs.append(v[b] - v[a])
incs = np.array(incs); print('per-response token increment: mean %.2f median %.0f min %d max %d (n=%d)' % (incs.mean(), np.median(incs), incs.min(), incs.max(), len(incs)))
# composition: token share of scm summary vs history, at d=0 and at the last decision
def share(r):
    st = r['envelope']['state']
    n_scm = len(tok.encode(canonical(st['scm']), add_special_tokens=False).ids)
    n_hist = len(tok.encode(canonical(st['history']), add_special_tokens=False).ids)
    n_menu = len(tok.encode(canonical(st['menu']), add_special_tokens=False).ids)
    n_task = len(tok.encode(canonical(r['envelope']['task']), add_special_tokens=False).ids)
    return n_scm, n_hist, n_menu, n_task, r['prompt_tokens']
first = [share(r) for r in rows if r['d'] == 0]
last = [share(r) for r in rows if r['d'] == 23]
for label, S in [('d=0', first), ('d=23', last)]:
    S = np.array(S, float)
    print(label, 'n=%d' % len(S), 'scm share %.3f history share %.3f menu share %.3f task share %.3f; mean tokens scm %.0f hist %.0f menu %.0f task %.0f prompt %.0f' % (
        (S[:, 0] / S[:, 4]).mean(), (S[:, 1] / S[:, 4]).mean(), (S[:, 2] / S[:, 4]).mean(), (S[:, 3] / S[:, 4]).mean(), S[:, 0].mean(), S[:, 1].mean(), S[:, 2].mean(), S[:, 3].mean(), S[:, 4].mean()))
# one 2x2 covariance token count
covs = []
for r in rows[:400]:
    for s in r['envelope']['state']['scm']:
        if len(s['parameter_mean']) == 2:
            covs.append(len(tok.encode(canonical(s['parameter_covariance']), add_special_tokens=False).ids))
print('tokens for one 2x2 covariance: median %.0f mean %.1f min %d max %d (n=%d)' % (np.median(covs), np.mean(covs), min(covs), max(covs), len(covs)))
ex = rows[0]['envelope']['state']['scm'][1]['parameter_covariance']; print('example', canonical(ex), len(tok.encode(canonical(ex), add_special_tokens=False).ids))
# command token tree
cmds = [canonical({'action_id': a}) for a in ACTIONS] + [canonical({'stop': True})]
enc = {c: tok.encode(c, add_special_tokens=False) for c in cmds}
for c in cmds: print(c, enc[c].ids, enc[c].tokens)
def tree(seqs, depth=0, prefix=()):
    nxt = collections.defaultdict(list)
    for s in seqs:
        if len(s) > depth: nxt[s[depth]].append(s)
    if len(nxt) > 1:
        print('branch after prefix', [tok.decode([t]) for t in prefix], '->', {tok.decode([t]): len(v) for t, v in nxt.items()})
    for t, v in nxt.items():
        tree(v, depth + 1, prefix + (t,))
tree([tuple(enc[c].ids) for c in cmds])
print('legal_commands distribution', collections.Counter((r['arm'], r['legal_commands']) for r in rows))
print('longest_command_tokens', collections.Counter(r['longest_command_tokens'] for r in rows))
