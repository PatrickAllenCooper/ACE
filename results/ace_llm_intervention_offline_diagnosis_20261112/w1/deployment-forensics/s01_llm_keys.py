"""Enumerate every key present in every LLM service worker record, to establish
exactly what is and is not recorded per decision (question 1)."""
import json, os, re, collections, sys
GPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu'
for arm in ['frozen_llm', 'sft_llm', 'dpo_llm']:
    w = os.path.join(GPU, 'services', arm, 'worker')
    names = sorted(os.listdir(w))
    kinds = collections.defaultdict(list)
    for n in names:
        k = re.sub(r'\d{3,4}', 'N', n)
        kinds[k].append(n)
    print('=====', arm, 'files', len(names))
    for k, v in sorted(kinds.items()):
        keysets = collections.Counter()
        extra = collections.Counter()
        for n in v:
            p = os.path.join(w, n)
            if not n.endswith('.json'):
                keysets['<non-json>'] += 1
                continue
            d = json.load(open(p))
            if isinstance(d, dict):
                keysets[tuple(sorted(d.keys()))] += 1
                if 'phase' in d:
                    extra[d['phase']] += 1
            else:
                keysets['<%s>' % type(d).__name__] += 1
        if len(v) > 3:
            print('  ', k, len(v))
            for ks, c in keysets.items():
                print('      ', c, 'x', ks)
            if extra:
                print('       phases:', dict(extra))
    # any key anywhere containing score/logit/logprob/likelihood/prob?
    hits = collections.Counter()
    pat = re.compile(r'score|logit|logprob|likelihood|prob|margin|entropy', re.I)
    def walk(o, path=''):
        if isinstance(o, dict):
            for kk, vv in o.items():
                if pat.search(str(kk)):
                    hits[kk] += 1
                walk(vv, path + '/' + str(kk))
        elif isinstance(o, list):
            for vv in o[:50]:
                walk(vv, path)
    for n in names:
        if n.endswith('.json') and not n.startswith(('runtime-', 'gpu-mapping')):
            walk(json.load(open(os.path.join(w, n))))
    print('   keys matching score/logit/logprob/likelihood/prob/margin/entropy in worker records (excluding runtime inventories):', dict(hits))
