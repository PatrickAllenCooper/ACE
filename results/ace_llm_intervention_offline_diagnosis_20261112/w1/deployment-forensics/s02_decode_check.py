"""Verify the local tokenizer.json matches the pinned SHA256, decode one saved
prompt_ids list (tokenizer vocabulary only, NO model weights), and verify the
decoded user content against the saved prompt_sha256 and envelope_sha256."""
import json, hashlib, os, sys, warnings
warnings.filterwarnings('ignore')
TOK = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json')
PIN = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
raw = open(TOK, 'rb').read()
print('tokenizer.json sha256', hashlib.sha256(raw).hexdigest(), 'matches pin:', hashlib.sha256(raw).hexdigest() == PIN)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOK)
GPU = '/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu'
g = json.load(open(GPU + '/services/dpo_llm/worker/generation-0000.json'))
text = tok.decode(g['prompt_ids'], skip_special_tokens=False)
print(repr(text[:900]))
print('...')
print(repr(text[-400:]))
print('completion tokens:', g['sequence_ids'][len(g['prompt_ids']):], repr(tok.decode(g['sequence_ids'][len(g['prompt_ids']):], skip_special_tokens=False)))
pre = '<|im_start|>user\n'
post = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
assert text.startswith(pre) and text.endswith(post), (text[:40], text[-80:])
user = text[len(pre):-len(post)]
print('prompt sha256 match:', hashlib.sha256(user.encode()).hexdigest() == g['prompt_sha256'])
head, rest = user.split('\n', 1)
body = rest[:-len('\nDecision:\n')]
assert rest.endswith('\nDecision:\n')
env = json.loads(body)
canon = json.dumps(env, sort_keys=True, separators=(',', ':'), allow_nan=False)
print('canonical roundtrip equal:', canon == body)
print('envelope sha256 match:', hashlib.sha256(canon.encode()).hexdigest() == g['envelope_sha256'])
print('preamble:', head)
print('prompt chars', len(user), 'tokens', g['prompt_tokens'])
