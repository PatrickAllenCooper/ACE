"""Verifier script 01: enumerate every JSON key path in every worker/controller/launch file of the three LLM
service directories, and search for any key or string value suggesting per-command scores."""
import json, os, re, sys, collections
GPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu'
pat = re.compile(r'score|logit|logprob|log_prob|prob|margin|likelihood|rank|entropy|nll|loss', re.I)

def walk(o, path, acc, vals):
    if isinstance(o, dict):
        for k, v in o.items():
            p = path + '.' + k
            acc[p] += 1
            walk(v, p, acc, vals)
    elif isinstance(o, list):
        # only descend into list elements if they are containers
        for i, v in enumerate(o):
            if isinstance(v, (dict, list)):
                walk(v, path + '[]', acc, vals)
    else:
        if isinstance(o, str) and pat.search(o) and len(o) < 400:
            vals[path].add(o[:200])

all_keys = collections.Counter()
per_kind = collections.defaultdict(collections.Counter)
vals = collections.defaultdict(set)
nfiles = 0
bad = []
for root, dirs, files in os.walk(GPU):
    for fn in files:
        if not fn.endswith('.json') and not fn.endswith('.jsonl'):
            print('non-json file', os.path.join(root, fn)); continue
        fp = os.path.join(root, fn)
        kind = re.sub(r'\d{3,}', 'N', os.path.relpath(fp, GPU))
        try:
            if fn.endswith('.jsonl'):
                objs = [json.loads(l) for l in open(fp) if l.strip()]
            else:
                objs = [json.load(open(fp))]
        except Exception as e:
            bad.append((fp, str(e))); continue
        nfiles += 1
        for o in objs:
            acc = collections.Counter()
            walk(o, '', acc, vals)
            for k in acc:
                per_kind[kind][k] += 1
                all_keys[k] += 1
print('files parsed', nfiles, 'bad', len(bad))
for b in bad[:10]: print('BAD', b)
print('\n=== distinct key paths matching pattern, by file kind ===')
for kind in sorted(per_kind):
    hits = [k for k in per_kind[kind] if pat.search(k.split('.')[-1])]
    if hits:
        print(kind)
        for h in sorted(hits): print('    ', h, per_kind[kind][h])
print('\n=== string values matching pattern ===')
for p in sorted(vals):
    print(p, '->', sorted(vals[p])[:8])
print('\n=== key paths per LLM worker record kind ===')
for kind in sorted(per_kind):
    if '/worker/' in kind and ('generation' in kind or 'generated' in kind or 'request' in kind or 'projection' in kind or 'terminal' in kind):
        print(kind, sum(1 for _ in per_kind[kind]))
        for k in sorted(per_kind[kind]): print('    ', k, per_kind[kind][k])
