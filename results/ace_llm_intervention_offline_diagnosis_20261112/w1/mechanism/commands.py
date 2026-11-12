import json, os, collections, warnings
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer
TOK="/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
tok=Tokenizer.from_file(TOK)
P="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/inputs/prepared.json"
d=json.load(open(P))
recs=d["training_plan"]["records"]
# command token tables: are they identical across records?
tables=collections.Counter(json.dumps(r["commands"],sort_keys=True) for r in recs)
print("distinct command tables across 160 records:", len(tables))
for t,c in tables.most_common():
    tb=json.loads(t)
    print("count",c,"commands",len(tb))
tb=json.loads(tables.most_common(1)[0][0])
for cmd,ids in sorted(tb.items(), key=lambda kv: kv[1]):
    pieces=[tok.decode([i], skip_special_tokens=False) for i in ids]
    print(f"{cmd:32s} n_tokens_incl_eos={len(ids)}  ids={ids}  pieces={pieces}")
# verify local tokenizer reproduces
for cmd,ids in tb.items():
    enc=tok.encode(cmd, add_special_tokens=False).ids
    assert enc==ids[:-1], (cmd, enc, ids)
print("local tokenizer reproduces all saved command tokenizations: True")
