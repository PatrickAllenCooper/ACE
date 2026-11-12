import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/deployment-forensics-verify')
from vcommon import *
import glob
snap = glob.glob('/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/')[0]
print('snapshot', snap)
for f in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'config.json', 'generation_config.json']:
    b = open(snap + f, 'rb').read()
    print(f, len(b), hashlib.sha256(b).hexdigest())
from tokenizers import Tokenizer
tok = Tokenizer.from_file(snap + 'tokenizer.json')
g = json.load(open(GPU + '/services/dpo_llm/worker/generation-0000.json'))
text = tok.decode(g['prompt_ids'], skip_special_tokens=False)
print(repr(text[:700]))
print('...')
print(repr(text[-300:]))
full = tok.decode(g['sequence_ids'], skip_special_tokens=False)
print(repr(full[-200:]))
print('completion tokens:', g['sequence_ids'][len(g['prompt_ids']):], [tok.decode([t], skip_special_tokens=False) for t in g['sequence_ids'][len(g['prompt_ids']):]])
