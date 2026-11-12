import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import collections, glob
T = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108/terminal01/remote/launch/training/session/worker/artifacts/fit/training'
plan = json.load(open(T + '/plan.json'))
print('tokenizer_sha256 pins in training plan records:', collections.Counter(r['tokenizer_sha256'] for r in plan['records']))
r0 = plan['records'][0]
print('token_metadata', json.dumps(r0['token_metadata'])[:600])
print('commands', json.dumps(r0['commands'])[:900])
print('strict_pairs[0:2]', json.dumps(r0['strict_pairs'][:2])[:500])
snap = glob.glob('/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/')[0]
print('local tokenizer.json sha256', hashlib.sha256(open(snap + 'tokenizer.json', 'rb').read()).hexdigest())
# training prompt lengths and whether training prompts decode consistent with records
from tokenizers import Tokenizer
tok = Tokenizer.from_file(snap + 'tokenizer.json')
lens = [len(r['prompt_ids']) for r in plan['records']]
import numpy as np
print('training prompt tokens: min %d max %d median %.0f' % (min(lens), max(lens), np.median(lens)))
byrem = collections.defaultdict(list)
for r in plan['records']:
    raw = json.loads(r['raw_record_utf8']); byrem[raw['envelope']['state']['remaining_units']].append(len(r['prompt_ids']))
print('training prompt tokens by remaining budget:', {k: (len(v), int(np.median(v))) for k, v in sorted(byrem.items())})
# history composition of training states: what actions fill training histories?
hist = collections.Counter(); hist_by_snap = collections.defaultdict(collections.Counter)
for r in plan['records']:
    raw = json.loads(r['raw_record_utf8']); st = raw['envelope']['state']
    for h in st['history'][8:]:
        hist[h['action_id']] += 1; hist_by_snap[r['record_id'].split('/')[-1]][SHORT[h['action_id']]] += 1
print('training-state acquisition histories (post-initialisation rows):', {SHORT[k]: v for k, v in hist.most_common()})
for s in sorted(hist_by_snap): print('   ', s, dict(hist_by_snap[s].most_common()))
# max run of identical action in training histories
def maxrun(seq):
    best = cur = 0; prev = None
    for x in seq:
        cur = cur + 1 if x == prev else 1; prev = x; best = max(best, cur)
    return best
runs = [maxrun([h['action_id'] for h in json.loads(r['raw_record_utf8'])['envelope']['state']['history'][8:]]) for r in plan['records']]
print('longest run of one repeated acquisition in any training history:', max(runs), '| distribution', sorted(collections.Counter(runs).items()))
