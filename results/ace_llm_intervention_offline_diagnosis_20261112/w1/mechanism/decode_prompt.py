"""Offline decode of saved prompt token IDs with the pinned local tokenizer vocabulary.
No model weights, no network. Reads only saved worker generation records."""
import os, sys, json, hashlib, warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
from tokenizers import Tokenizer
TOK="/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
tok=Tokenizer.from_file(TOK)
GPU="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services"
OUT="/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/mechanism"
def load(arm, n):
    return json.load(open(f"{GPU}/{arm}/worker/generation-{n:04d}.json"))
def decode(ids):
    return tok.decode(ids, skip_special_tokens=False)
if __name__=="__main__":
    arm=sys.argv[1]; n=int(sys.argv[2])
    g=load(arm,n)
    full=decode(g["prompt_ids"])
    pre="<|im_start|>user\n"; post="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    print("starts_with_user_header:", full.startswith(pre), "ends_with_assistant_think:", full.endswith(post))
    user=full[len(pre):-len(post)]
    print("recorded prompt_sha256:", g["prompt_sha256"])
    print("decoded  user sha256  :", hashlib.sha256(user.encode()).hexdigest())
    print("prompt_tokens:", g["prompt_tokens"], "len(prompt_ids):", len(g["prompt_ids"]))
    print("user chars:", len(user), "user utf8 bytes:", len(user.encode()), "full templated chars:", len(full))
    print("completion:", g["completion"], "completion ids:", g["sequence_ids"][len(g["prompt_ids"]):])
    open(f"{OUT}/{arm}-{n:04d}.templated.txt","w").write(full)
    open(f"{OUT}/{arm}-{n:04d}.user.txt","w").write(user)
