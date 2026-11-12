"""Prompt token statistics and token-level label tallies from the saved prepared tokens (no tokenizer or model is loaded)."""
import json, os
from collections import Counter
OUT=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(OUT+'/records.json'))
TG='/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108'
PREP=TG+'/terminal01/remote/cpu/preparation/session/worker/artifacts'
d=json.load(open(PREP+'/llm-prepared.json'))
tp=d['training_plan']
recs=tp['records']
print('prepared records',len(recs),'stored_tokens',tp['stored_tokens'],'max_tokens',tp['numeric_recipe']['max_tokens'])
pl=[len(r['prompt_ids']) for r in recs]
print('prompt tokens min %d max %d mean %.1f'%(min(pl),max(pl),sum(pl)/len(pl)))
print('prompt_tokens metadata agrees:',all(r['token_metadata']['prompt_tokens']==len(r['prompt_ids']) for r in recs))
print('max prompt+max_new_tokens',max(pl)+tp['numeric_recipe']['max_new_tokens'],'<= max_tokens',max(pl)+tp['numeric_recipe']['max_new_tokens']<=tp['numeric_recipe']['max_tokens'])
byrid={r['rid']:r for r in R}
nt_pl=[len(r['prompt_ids']) for r in recs if not byrid[r['record_id']]['terminal']]
print('nonterminal prompt tokens min %d max %d'%(min(nt_pl),max(nt_pl)))
for s in range(5):
    x=[len(r['prompt_ids']) for r in recs if byrid[r['record_id']]['snapshot']==s]
    print('  snap',s,'min',min(x),'max',max(x),'mean %.0f'%(sum(x)/len(x)))
# common suffix / prefix
def common_suffix(seqs):
    n=0
    while all(len(s)>n for s in seqs) and len(set(s[-1-n] for s in seqs))==1: n+=1
    return n
def common_prefix(seqs):
    n=0
    while all(len(s)>n for s in seqs) and len(set(s[n] for s in seqs))==1: n+=1
    return n
allp=[r['prompt_ids'] for r in recs]
print('common suffix over all 160 prompts:',common_suffix(allp),'tokens ; common prefix:',common_prefix(allp))
ntp=[r['prompt_ids'] for r in recs if not byrid[r['record_id']]['terminal']]
print('common suffix over 128 nonterminal prompts:',common_suffix(ntp),' ; common prefix:',common_prefix(ntp))
for cm in ['uni','het']:
    sub=[r['prompt_ids'] for r in recs if (byrid[r['record_id']]['menu_costs']['set_y_neg']==2)==(cm=='het')]
    print('  common suffix within cost menu',cm,common_suffix(sub),'n',len(sub))
# where does the state-specific content sit relative to the end? distance from end to last differing token
# commands and trie
cmds=set()
for r in recs:
    for k,v in r['commands'].items(): cmds.add((k,tuple(v)))
print('distinct (command, token ids):',len(cmds))
for k,v in sorted(cmds): print('  ',k,list(v))
print('legal command counts',Counter(len(r['commands']) for r in recs))
print('terminal records commands',Counter(tuple(sorted(r['commands'])) for r in recs if byrid[r['record_id']]['terminal']))
# token decision tallies over nonterminal supervised labels
tok={k:list(v) for k,v in cmds}
nt=[r for r in recs if not byrid[r['record_id']]['terminal']]
labs=[tok[r['supervised_command']] for r in nt]
# trie walk along the set_x_neg path
path=tok['{"action_id":"set_x_neg"}']
print('\nGreedy path to set_x_neg, label mass at each branching decision (nonterminal labels):')
for i in range(len(path)):
    cons=[l for l in labs if l[:i]==path[:i]]
    nxt=Counter(l[i] for l in cons if len(l)>i)
    if len(nxt)>1 or i in (1,):
        print('  position %d: prefix-consistent labels %d ; next-token counts %s ; path token %d share %.3f'%(i,len(cons),dict(nxt),path[i],nxt[path[i]]/len(cons)))
# pooled third-token (suffix) tally used by the earlier agent
suf=Counter()
for r in nt:
    a=json.loads(r['supervised_command'])['action_id']
    if a.startswith('set_x_') or a.startswith('set_y_'): suf[a.split('_')[2]]+=1
print('pooled suffix over set_x_*/set_y_* labels:',dict(suf),'total',sum(suf.values()))
# legal first-branch tokens at a nonterminal state include stop
print('stop is a legal command at nonterminal states:',all('{"stop":true}' in r['commands'] for r in nt))
# training action tokens
print('strict_pairs total in prepared:',sum(len(r['strict_pairs']) for r in recs))
