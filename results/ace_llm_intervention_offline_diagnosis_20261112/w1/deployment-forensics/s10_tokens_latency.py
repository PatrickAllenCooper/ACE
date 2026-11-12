"""Prompt length, tokens, latency per decision for LLM arms; command token tree;
input-side sensitivity (how much of the prompt changes between decisions)."""
import pickle, collections, json, os, hashlib, warnings
warnings.filterwarnings('ignore')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(HERE + '/out/decisions.pkl', 'rb'))
rows, worlds, pids = D['rows'], D['worlds'], D['policy_ids']
TOK = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json')
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOK)
cmds = ['{"action_id":"%s"}' % a for a in ['observe_all', 'set_x_neg', 'set_x_pos', 'set_x_zero', 'set_xy', 'set_y_neg', 'set_y_pos', 'set_y_zero']] + ['{"stop":true}']
print('Command tokenizations (pinned tokenizer, add_special_tokens=False); EOS 151645 appended at deployment:')
enc = {}
for c in cmds:
    e = tok.encode(c, add_special_tokens=False)
    enc[c] = e.ids
    print('  %-30s %2d tokens  %s  %s' % (c, len(e.ids), e.ids, [tok.decode([i]) for i in e.ids]))
# prefix tree branching
print('\nBranch points of the constrained greedy decode (prefix -> allowed next tokens):')
prefixes = collections.defaultdict(set)
for c, ids in enc.items():
    ids = ids + [151645]
    for i in range(len(ids)):
        prefixes[tuple(ids[:i])].add(ids[i])
for p, nxt in sorted(prefixes.items(), key=lambda kv: len(kv[0])):
    if len(nxt) > 1:
        print('  after %-40s -> %s' % (repr(tok.decode(list(p))), [tok.decode([i]) for i in sorted(nxt)]))
for arm in pids[5:]:
    R = [r for r in rows if r['arm'] == arm]
    pt = np.array([r['prompt_tokens'] for r in R]); pc = np.array([r['prompt_chars'] for r in R])
    el = np.array([r['elapsed_seconds'] for r in R]); cu = np.array([r['cuda_interval_ms'] for r in R])
    nn = collections.Counter(r['n_new_tokens'] for r in R)
    print('\n=====', arm, 'n', len(R))
    print('  prompt_tokens: min %d p25 %d median %d mean %.1f p75 %d max %d' % (pt.min(), np.percentile(pt, 25), np.median(pt), pt.mean(), np.percentile(pt, 75), pt.max()))
    print('  prompt_chars : min %d median %d mean %.1f max %d ; chars/token %.3f' % (pc.min(), np.median(pc), pc.mean(), pc.max(), pc.sum() / pt.sum()))
    print('  elapsed_seconds: min %.3f median %.3f mean %.3f p95 %.3f max %.3f ; total %.1f s' % (el.min(), np.median(el), el.mean(), np.percentile(el, 95), el.max(), el.sum()))
    print('  cuda_interval_ms: min %.1f median %.1f mean %.1f p95 %.1f max %.1f ; total %.1f ms' % (cu.min(), np.median(cu), cu.mean(), np.percentile(cu, 95), cu.max(), cu.sum()))
    print('  new tokens generated (incl EOS):', dict(nn), ' legal_commands:', dict(collections.Counter(r['legal_commands'] for r in R)), ' longest_command_tokens:', dict(collections.Counter(r['longest_command_tokens'] for r in R)))
    print('  peak_gpu_allocated MiB: min %.0f max %.0f' % (min(r['peak_gpu_allocated_bytes'] for r in R) / 2**20, max(r['peak_gpu_allocated_bytes'] for r in R) / 2**20))
    # tokens by decision index
    bd = collections.defaultdict(list); bl = collections.defaultdict(list)
    for r in R: bd[r['decision_index']].append(r['prompt_tokens']); bl[r['decision_index']].append(r['cuda_interval_ms'])
    print('  mean prompt tokens by decision index 0,5,10,15,20,23:', [round(float(np.mean(bd[k])), 0) for k in (0, 5, 10, 15, 20, 23) if k in bd])
    print('  mean cuda ms by decision index 0,1,5,10,15,20,23:', [round(float(np.mean(bl[k])), 0) for k in (0, 1, 5, 10, 15, 20, 23) if k in bl])
    sl = np.polyfit(pt, cu, 1)
    print('  latency ~ tokens linear fit: %.4f ms/token + %.1f ms ; corr %.3f' % (sl[0], sl[1], np.corrcoef(pt, cu)[0, 1]))
    print('  first request of session (ordinal 0) cuda ms: %.1f' % [r['cuda_interval_ms'] for r in R if r['ordinal'] == 0][0])
    print('  distinct prompts (by prompt-id hash): %d of %d ; distinct envelopes: %d' % (len({r['prompt_ids_sha'] for r in R}), len(R), len({r['envelope_sha256'] for r in R})))
# tokens per added history row
print('\nprompt token growth per additional accepted response (consecutive decisions in the same world):')
for arm in pids[5:]:
    inc = []
    for w in range(24):
        s = sorted((r for r in rows if r['arm'] == arm and r['world_idx'] == w), key=lambda r: r['decision_index'])
        inc += [b['prompt_tokens'] - a['prompt_tokens'] for a, b in zip(s, s[1:])]
    inc = np.array(inc)
    print('  %-11s mean %.1f sd %.1f min %d max %d' % (arm, inc.mean(), inc.std(), inc.min(), inc.max()))
# share of the prompt occupied by sections at first and last decision
r0 = next(r for r in rows if r['arm'] == 'dpo_llm' and r['world_idx'] == 0 and r['decision_index'] == 0)
r23 = next(r for r in rows if r['arm'] == 'dpo_llm' and r['world_idx'] == 0 and r['decision_index'] == 23)
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
for name, r in (('decision 0', r0), ('decision 23', r23)):
    st = r['envelope']['state']
    parts = {'history': st['history'], 'scm': st['scm'], 'menu': st['menu'], 'graph': st['graph'], 'task': r['envelope']['task'],
             'domain_notes+objective': [st['domain_notes'], st['objective']], 'counters': {k: st[k] for k in ('remaining_units', 'requests', 'returned_responses', 'spent_units', 'terminal')}}
    tot = r['prompt_tokens']
    print('\n  token budget by section, DPO world 0', name, '(total prompt tokens %d):' % tot)
    for k, v in parts.items():
        n = len(tok.encode(canonical(v), add_special_tokens=False).ids)
        print('     %-24s ~%5d tokens (%.1f%%)' % (k, n, 100 * n / tot))
print('\nexample scm JSON as the model sees it:', canonical(r0['envelope']['state']['scm'])[:700])
print('\nexample remaining/requests JSON fragment:', canonical({k: r0['envelope']['state'][k] for k in ('remaining_units', 'requests', 'returned_responses')}))
ids = tok.encode(canonical(r0['envelope']['state']['scm'][1]['parameter_covariance']), add_special_tokens=False)
print('one 2x2 covariance matrix =', len(ids.ids), 'tokens:', [tok.decode([i]) for i in ids.ids][:60])
