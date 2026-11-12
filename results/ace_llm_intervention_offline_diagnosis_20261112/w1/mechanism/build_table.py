"""Decode every saved LLM-arm deployment prompt (frozen/sft/dpo), verify the recorded prompt hash,
parse the public envelope out of the decoded text, and write one row per decision.
Pure offline: pinned tokenizer vocabulary + saved generation records. No model weights."""
import os, json, hashlib, warnings, re, sys, glob, pickle
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"]="false"
from tokenizers import Tokenizer
TOK="/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
tok=Tokenizer.from_file(TOK)
GPU="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services"
OUT="/private/tmp/claude-501/-Users-pat--claude-spares-spare-Yk5OYw/579351c9-7517-4194-a931-08cc9167484d/scratchpad/w1/mechanism"
PRE="<|im_start|>user\n"; POST="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
HEAD_END="Do not invent measurements or actions.\n"; TAIL="\nDecision:\n"
rows=[]; bad=0
for arm in ("frozen_llm","sft_llm","dpo_llm"):
    files=sorted(glob.glob(f"{GPU}/{arm}/worker/generation-*.json"))
    for f in files:
        g=json.load(open(f))
        full=tok.decode(g["prompt_ids"], skip_special_tokens=False)
        assert full.startswith(PRE) and full.endswith(POST), f
        user=full[len(PRE):-len(POST)]
        ok = hashlib.sha256(user.encode()).hexdigest()==g["prompt_sha256"]
        if not ok: bad+=1
        i=user.index(HEAD_END)+len(HEAD_END)
        assert user.endswith(TAIL)
        env_text=user[i:-len(TAIL)]
        env=json.loads(env_text)
        canon=json.dumps(env, sort_keys=True, separators=(",",":"), allow_nan=False)
        assert canon==env_text, "canonical round trip differs"
        assert hashlib.sha256(canon.encode()).hexdigest()==g["envelope_sha256"], "envelope sha differs"
        st=env["state"]
        parents={n["name"]:n["parents"] for n in st["graph"]}
        graph = "chain" if parents["Z"]==["Y"] and parents["Y"]==["X"] else "fork" if parents["Z"]==["X"] else "collider" if parents["Z"]==["X","Y"] else "?"
        costs={a["action_id"]:a["cost_units"] for a in st["menu"]}
        rows.append(dict(arm=arm, ordinal=g["ordinal"], prompt_tokens=g["prompt_tokens"], user_chars=len(user),
            env_chars=len(env_text), completion=g["completion"], n_new=len(g["sequence_ids"])-len(g["prompt_ids"]),
            legal=g["legal_commands"], remaining=st["remaining_units"], requests=st["requests"], spent=st["spent_units"],
            graph=graph, cost="heterogeneous" if costs["set_y_neg"]==2 else "uniform", noise=st["scm"][0]["noise_variance"],
            hist_actions=[r["action_id"] for r in st["history"]], hash_ok=ok,
            envelope_sha256=g["envelope_sha256"], task_sha256=g["task_sha256"], elapsed=g["elapsed_seconds"],
            task=json.dumps(env["task"],sort_keys=True), scm=st["scm"], env=env))
print("rows",len(rows),"hash mismatches",bad)
pickle.dump(rows, open(f"{OUT}/decisions.pkl","wb"))
