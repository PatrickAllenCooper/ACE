"""Build one unified per-decision dataset for all 8 arms x 24 systems from the
saved deployment records. Read-only on project data. No model weights.

Numerical arms: CPU campaign decision-NNNN.{input,process}.json
LLM arms: GPU service generation-NNNN.json + controller/<arm>-evidence.json joins,
          with the public envelope reconstructed by decoding the saved prompt_ids
          using the pinned tokenizer vocabulary (verified by two SHA256 checks).
"""
import json, os, sys, hashlib, pickle, collections, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
CPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_cpu_20261108/terminal02/remote/launch/cpu/worker/artifacts'
GPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu'
TOK = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json')
TOK_PIN = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'

def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

camp = json.load(open(CPU + '/campaign/campaign.json'))
policy_ids = camp['policy_ids']
worlds = []
for i, s in enumerate(camp['sources']):
    p = s['plan']
    worlds.append({'world_idx': i, 'world_id': p['world']['world_id'], 'graph': s['strata']['graph_stratum'],
                   'cost': s['strata']['cost_stratum'], 'regime': s['strata']['regime'],
                   'noise': p['learner']['noise_variances']['Z'],
                   'menu_costs': {a['action_id']: a['cost_units'] for a in p['menu']},
                   'mechanisms': p['world']['mechanisms'], 'plan_sha256': s['plan_sha256']})
wid2idx = {w['world_id']: w['world_idx'] for w in worlds}

rows = []
issues = []
# ---------------- numerical arms
for pi, arm in enumerate(policy_ids[:5]):
    for wi in range(24):
        ep = '%s/campaign/policy-%02d/world-%02d/episode' % (CPU, pi, wi)
        names = sorted(n for n in os.listdir(ep) if n.startswith('decision-') and n.endswith('.input.json'))
        term = json.load(open(ep + '/terminal.json'))
        journal = [json.loads(l) for l in open(ep + '/journal.jsonl')]
        stops = [e for e in journal if e.get('event') == 'stop']
        props = {e['proposal_id']: e for e in journal if e.get('event') == 'proposal'}
        stop_kind = props[stops[-1]['proposal_id']]['provenance'].get('kind') if stops else None
        if term['policy_decisions'] != len(names):
            issues.append(('decision count mismatch', arm, wi, term['policy_decisions'], len(names)))
        for di, n in enumerate(names):
            raw = open(ep + '/' + n).read()
            env = json.loads(raw)
            proc = json.load(open(ep + '/' + n.replace('.input.json', '.process.json')))
            if sha(canonical(env)) != proc['input_sha256']:
                issues.append(('input sha mismatch', arm, wi, di))
            out = json.loads(proc['stdout'])
            meta = json.loads(proc['stderr']) if proc['stderr'].strip() else None
            row = {'arm': arm, 'world_idx': wi, 'decision_index': di, 'envelope': env,
                   'envelope_sha256': proc['input_sha256'],
                   'action': out.get('action_id') if 'action_id' in out else 'STOP',
                   'elapsed_seconds': proc['elapsed_seconds'], 'meta': meta, 'returncode': proc['returncode'],
                   'episode_stop_kind': stop_kind}
            rows.append(row)

# ---------------- LLM arms
raw_tok = open(TOK, 'rb').read()
assert hashlib.sha256(raw_tok).hexdigest() == TOK_PIN
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOK)
PRE = '<|im_start|>user\n'
POST = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
preambles = collections.Counter()
for arm in policy_ids[5:]:
    ev = json.load(open('%s/controller/%s-evidence.json' % (GPU, arm)))
    for j in ev['joins']:
        g = json.load(open('%s/services/%s/worker/generation-%04d.json' % (GPU, arm, j['ordinal'])))
        if g['envelope_sha256'] != j['envelope_sha256'] or g['ordinal'] != j['ordinal']:
            issues.append(('join mismatch', arm, j['ordinal']))
        text = tok.decode(g['prompt_ids'], skip_special_tokens=False)
        if not (text.startswith(PRE) and text.endswith(POST)):
            issues.append(('chat template mismatch', arm, j['ordinal'])); continue
        user = text[len(PRE):-len(POST)]
        if sha(user) != g['prompt_sha256']:
            issues.append(('prompt sha mismatch', arm, j['ordinal']))
        head, rest = user.split('\n', 1)
        preambles[head] += 1
        assert rest.endswith('\nDecision:\n')
        env = json.loads(rest[:-len('\nDecision:\n')])
        if sha(canonical(env)) != g['envelope_sha256']:
            issues.append(('envelope sha mismatch', arm, j['ordinal']))
        comp = json.loads(g['completion'])
        action = comp.get('action_id') if 'action_id' in comp else 'STOP'
        if (action == 'STOP') != (j['outcome'] == 'policy_stop'):
            issues.append(('stop/outcome mismatch', arm, j['ordinal']))
        rows.append({'arm': arm, 'world_idx': wid2idx[j['world_id']], 'decision_index': j['decision_index'],
                     'envelope': env, 'envelope_sha256': g['envelope_sha256'], 'action': action,
                     'elapsed_seconds': g['elapsed_seconds'], 'cuda_interval_ms': g['cuda_interval_ms'],
                     'prompt_tokens': g['prompt_tokens'], 'prompt_chars': len(user),
                     'legal_commands': g['legal_commands'], 'longest_command_tokens': g['longest_command_tokens'],
                     'completion_token_ids': g['sequence_ids'][len(g['prompt_ids']):],
                     'n_new_tokens': len(g['sequence_ids']) - len(g['prompt_ids']),
                     'peak_gpu_allocated_bytes': g['peak_gpu_allocated_bytes'],
                     'ordinal': j['ordinal'], 'charged_units': j['charged_units'], 'outcome': j['outcome'],
                     'meta': None, 'prompt_ids_sha': hashlib.sha256(json.dumps(g['prompt_ids']).encode()).hexdigest()})
print('preamble variants:', len(preambles), list(preambles.values()))
print('rows', len(rows), 'issues', len(issues), issues[:10])
by = collections.Counter(r['arm'] for r in rows)
print(dict(by))
pickle.dump({'rows': rows, 'worlds': worlds, 'policy_ids': policy_ids}, open(HERE + '/out/decisions.pkl', 'wb'))
