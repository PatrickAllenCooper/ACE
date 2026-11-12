"""Build decision table for the three LLM arms by decoding saved prompt token ids. Verifier's own code."""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import glob, pickle, collections, re
from tokenizers import Tokenizer
snap = glob.glob('/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/')[0]
tok = Tokenizer.from_file(snap + 'tokenizer.json')
PRE = '<|im_start|>user\n'
POST = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
HEAD = ("Choose the next experimental intervention using this public SCM, paid history and prediction task. "
        "The task queries and weights describe where accurate predictions are useful. They contain no observed answers. "
        "Predict the noise-disabled structural response. Error is squared error divided by scale squared. "
        "The tolerance is a goal, not evidence that it has been achieved. Never use an unseen evaluation result to stop. "
        "Every environment response costs the listed units. Return exactly one JSON object containing an affordable "
        "action_id from state.menu, or {\"stop\":true}. Do not invent measurements or actions.\n")
TAIL = "\nDecision:\n"
camp = json.load(open(CAMP + '/campaign.json'))
wid2idx = {s['plan']['world']['world_id']: i for i, s in enumerate(camp['sources'])}
rows = []
stats = collections.Counter()
for arm in ['frozen_llm', 'sft_llm', 'dpo_llm']:
    ev = json.load(open(f'{GPU}/controller/{arm}-evidence.json'))
    joins = {j['ordinal']: j for j in ev['joins']}
    wd = f'{GPU}/services/{arm}/worker'
    gens = sorted(f for f in os.listdir(wd) if re.fullmatch(r'generation-\d{4}\.json', f))
    assert len(gens) == len(joins), (arm, len(gens), len(joins))
    for f in gens:
        g = json.load(open(f'{wd}/{f}'))
        o = g['ordinal']
        assert o == int(f[11:15])
        j = joins[o]
        text = tok.decode(g['prompt_ids'], skip_special_tokens=False)
        ok_wrap = text.startswith(PRE) and text.endswith(POST)
        prompt = text[len(PRE):-len(POST)]
        ok_psha = hashlib.sha256(prompt.encode()).hexdigest() == g['prompt_sha256']
        ok_head = prompt.startswith(HEAD) and prompt.endswith(TAIL)
        env_text = prompt[len(HEAD):-len(TAIL)]
        env = json.loads(env_text)
        ok_canon = canonical(env) == env_text
        ok_esha = digest(env) == g['envelope_sha256'] == j['envelope_sha256']
        ok_len = len(g['prompt_ids']) == g['prompt_tokens']
        seq = g['sequence_ids']
        ok_prefix = seq[:len(g['prompt_ids'])] == g['prompt_ids']
        comp_ids = seq[len(g['prompt_ids']):]
        comp_text = tok.decode(comp_ids[:-1], skip_special_tokens=False)
        ok_comp = comp_text == g['completion'] and comp_ids[-1] == 151645
        # unaccepted + raw files must agree
        gu = json.load(open(f'{wd}/generated-{o:04d}.unaccepted.json'))
        raw = json.load(open(f'{wd}/raw-generation-{o:04d}.json'))
        ok_dup = gu['completion'] == g['completion'] and gu['prompt_ids'] == g['prompt_ids'] and gu['sequence_ids'] == seq and raw['token_prefix'] == seq
        for name, ok in [('wrap', ok_wrap), ('prompt_sha', ok_psha), ('head', ok_head), ('canon', ok_canon), ('env_sha', ok_esha),
                         ('len', ok_len), ('prefix', ok_prefix), ('comp', ok_comp), ('dup', ok_dup)]:
            stats[(arm, name, ok)] += 1
        c = json.loads(g['completion'])
        action = 'STOP' if c.get('stop') is True else c['action_id']
        st = env['state']
        rows.append(dict(arm=arm, world=wid2idx[j['world_id']], world_id=j['world_id'], d=j['decision_index'], ordinal=o,
                         envelope=env, envelope_sha256=g['envelope_sha256'], action=action, meta=None,
                         prompt_tokens=g['prompt_tokens'], seq_tokens=len(seq), legal_commands=g['legal_commands'],
                         longest_command_tokens=g['longest_command_tokens'],
                         elapsed=g['elapsed_seconds'], cuda_ms=g['cuda_interval_ms'], join=j,
                         comp_ids=comp_ids))
for k in sorted(stats): print(k, stats[k])
print('rows', len(rows))
for arm in ['frozen_llm', 'sft_llm', 'dpo_llm']:
    c = collections.Counter(r['action'] for r in rows if r['arm'] == arm)
    print(arm, len([r for r in rows if r['arm'] == arm]), {SHORT[k]: v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}, 'acquired', sum(v for k, v in c.items() if k != 'STOP'))
    per = collections.Counter(r['world'] for r in rows if r['arm'] == arm)
    print('   per world', [per[w] for w in range(24)])
    jo = collections.Counter((r['join']['accepted'], r['join']['outcome'], r['join']['charged_units']) for r in rows if r['arm'] == arm)
    print('   joins', dict(jo))
pickle.dump(rows, open(SCR + '/llm_rows.pkl', 'wb'))
