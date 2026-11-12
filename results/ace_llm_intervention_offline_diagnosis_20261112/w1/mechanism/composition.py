"""Token/char composition of real deployment prompts. Classifies every character of the user content as
C  constant in every prompt of the study (instructions, key names, punctuation, domain notes, objective, task block, menu ids/targets/measure)
W  varies by system only (graph parents, menu cost values, noise variance, scm terms)
S  varies by step (history rows incl. their repeated keys, counters, eligible_rows, posterior means/covariances)
and reports numeric-literal share. Uses the pinned tokenizer for offsets; verifies ids equal the saved prompt ids."""
import json, pickle, re, warnings, collections, statistics as S
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer
TOK="/Users/pat/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
tok=Tokenizer.from_file(TOK)
GPU="/Users/pat/code/ACE/results/ace_llm_intervention_scientific_campaign_gpu_20261108/terminal02/remote/launch/gpu/services"
rows=pickle.load(open("decisions.pkl","rb"))
HEAD=("Choose the next experimental intervention using this public SCM, paid history and prediction task. "
 "The task queries and weights describe where accurate predictions are useful. They contain no observed answers. "
 "Predict the noise-disabled structural response. Error is squared error divided by scale squared. "
 "The tolerance is a goal, not evidence that it has been achieved. Never use an unseen evaluation result to stop. "
 "Every environment response costs the listed units. Return exactly one JSON object containing an affordable "
 "action_id from state.menu, or {\"stop\":true}. Do not invent measurements or actions.\n")
C=lambda v: json.dumps(v, sort_keys=True, separators=(",",":"), allow_nan=False)
def segments(env):
    """Yield (text, klass, section) pieces whose concatenation equals canonical(envelope)."""
    st=env["state"]; out=[]
    def add(t,k,sec): out.append((t,k,sec))
    add('{"state":{"domain_notes":'+C(st["domain_notes"])+',"graph":[',"C","graph")
    for i,n in enumerate(st["graph"]):
        add(("," if i else "")+'{"intervention_bounds":'+C(n["intervention_bounds"])+',"name":'+C(n["name"])+',"parents":',"C","graph")
        add(C(n["parents"]),"W","graph")
        add(',"unit":'+C(n["unit"])+'}',"C","graph")
    add('],"history":[',"C","history")
    for i,r in enumerate(st["history"]):
        # the whole row is new step-dependent content; split boilerplate keys from values for a finer view
        add(("," if i else "")+'{"action_id":',"Sk","history"); add(C(r["action_id"]),"S","history")
        add(',"cost_units":',"Sk","history"); add(C(r["cost_units"]),"S","history")
        add(',"kind":',"Sk","history"); add(C(r["kind"]),"S","history")
        add(',"request_id":',"Sk","history"); add(C(r["request_id"]),"S","history")
        add(',"targets":',"Sk","history"); add(C(r["targets"]),"S","history")
        add(',"values":{',"Sk","history")
        for j,(k,v) in enumerate(sorted(r["values"].items())):
            add(("," if j else "")+C(k)+":","Sk","history"); add(C(v),"Sn","history")
        add('}}',"Sk","history")
    add('],"menu":[',"C","menu")
    for i,a in enumerate(st["menu"]):
        add(("," if i else "")+'{"action_id":'+C(a["action_id"])+',"cost_units":',"C","menu"); add(C(a["cost_units"]),"W","menu")
        add(',"kind":'+C(a["kind"])+',"measure":'+C(a["measure"])+',"targets":'+C(a["targets"])+'}',"C","menu")
    add('],"objective":'+C(st["objective"])+',"remaining_units":',"C","counters"); add(C(st["remaining_units"]),"S","counters")
    add(',"requests":',"C","counters"); add(C(st["requests"]),"S","counters")
    add(',"returned_responses":',"C","counters"); add(C(st["returned_responses"]),"S","counters")
    add(',"scm":[',"C","scm")
    for i,s in enumerate(st["scm"]):
        add(("," if i else "")+'{"eligible_rows":',"C","scm"); add(C(s["eligible_rows"]),"S","scm")
        add(',"node":'+C(s["node"])+',"noise_variance":',"C","scm"); add(C(s["noise_variance"]),"W","scm")
        add(',"parameter_covariance":',"C","scm"); add(C(s["parameter_covariance"]),"Sn","scm")
        add(',"parameter_mean":',"C","scm"); add(C(s["parameter_mean"]),"Sn","scm")
        add(',"terms":',"C","scm"); add(C(s["terms"]),"W","scm"); add('}',"C","scm")
    add('],"spent_units":',"C","counters"); add(C(st["spent_units"]),"S","counters")
    add(',"terminal":',"C","counters"); add(C(st["terminal"]),"S","counters")
    add(',"version":'+C(st["version"])+'},"task":',"C","state_tail")
    add(C(env["task"]),"C","task")
    add(',"version":'+C(env["version"])+'}',"C","task")
    return out
def analyse(r, verbose=False):
    g=json.load(open(f"{GPU}/{r['arm']}/worker/generation-{r['ordinal']:04d}.json"))
    env=r["env"]; segs=[(HEAD,"C","instructions")]+segments(env)+[("\nDecision:\n","C","tail")]
    user="".join(t for t,_,_ in segs)
    assert user==HEAD+C(env)+"\nDecision:\n"
    enc=tok.encode(user, add_special_tokens=False)
    ids=g["prompt_ids"]
    assert ids[:3]==[151644,872,198] and ids[3:3+len(enc.ids)]==enc.ids and len(ids)==3+len(enc.ids)+9, "token mismatch"
    # char class map
    klass=[]; sect=[]
    for t,k,s in segs: klass.extend([k]*len(t)); sect.extend([s]*len(t))
    ck=collections.Counter(klass); cs=collections.Counter(sect)
    tk=collections.Counter(); ts=collections.Counter()
    for (a,b) in enc.offsets:
        # assign token to the class/section of its first non-space char
        tk[klass[a]]+=1; ts[sect[a]]+=1
    ntok=len(enc.ids)+12  # + chat wrapper: 3 prefix + 9 suffix special/format tokens
    res=dict(chars=len(user), tokens=len(ids), char_class=dict(ck), tok_class=dict(tk), char_sect=dict(cs), tok_sect=dict(ts), wrapper_tokens=12)
    return res, enc, user, klass
def show(r,label):
    res,enc,user,klass=analyse(r)
    n=res["tokens"]; c=res["chars"]
    print(f"--- {label}: arm={r['arm']} ordinal={r['ordinal']} history_rows={r['requests']} graph={r['graph']} cost={r['cost']} noise={r['noise']}")
    print(f"    user chars={c} total prompt tokens (incl. 12 chat-template tokens)={n}")
    print("    tokens by section:", {k:(v, f"{100*v/n:.1f}%") for k,v in sorted(res["tok_sect"].items(), key=lambda kv:-kv[1])}, "chat_wrapper:",(12,f"{100*12/n:.1f}%"))
    tc=res["tok_class"]; cc=res["char_class"]
    step=tc.get("S",0)+tc.get("Sn",0); stepk=tc.get("Sk",0)
    print(f"    tokens: constant C={tc.get('C',0)+12} ({100*(tc.get('C',0)+12)/n:.1f}%), system-only W={tc.get('W',0)} ({100*tc.get('W',0)/n:.1f}%), step-varying values S+Sn={step} ({100*step/n:.1f}%) of which numeric literals Sn={tc.get('Sn',0)} ({100*tc.get('Sn',0)/n:.1f}%), repeated history key boilerplate Sk={stepk} ({100*stepk/n:.1f}%)")
    step_c=cc.get("S",0)+cc.get("Sn",0)
    print(f"    chars : constant C={cc.get('C',0)} ({100*cc.get('C',0)/c:.1f}%), W={cc.get('W',0)} ({100*cc.get('W',0)/c:.1f}%), step values={step_c} ({100*step_c/c:.1f}%) [numeric {cc.get('Sn',0)} = {100*cc.get('Sn',0)/c:.1f}%], history keys Sk={cc.get('Sk',0)} ({100*cc.get('Sk',0)/c:.1f}%)")
    return res,enc,user,klass
dpo=sorted([r for r in rows if r["arm"]=="dpo_llm"], key=lambda r:r["ordinal"])
fro=sorted([r for r in rows if r["arm"]=="frozen_llm"], key=lambda r:r["ordinal"])
res,enc,user,klass=show(dpo[0],"first decision (reproduced prompt)")
show(dpo[23],"last decision of episode 0 (DPO)")
longest=max(rows, key=lambda r:r["prompt_tokens"]); show(longest,"longest prompt in campaign")
# posterior block position
i_scm=user.index(',"scm":['); i_task=user.index('"task":'); 
tok_pos=lambda ch: next(k for k,(a,b) in enumerate(enc.offsets) if b>ch)+3
print("first-decision prompt: token index where scm block starts:", tok_pos(i_scm), "task block starts:", tok_pos(i_task), "of", res["tokens"], "| tokens from end of scm numbers to answer position:", res["tokens"]-tok_pos(i_task))
# number tokenization
for s in ['1.2873731568992153','-0.6175376044829376','0.030303030303030304','-1.0','0.25','0.125','24']:
    e=tok.encode(s, add_special_tokens=False); print("   number",s,"->",len(e.ids),"tokens:",[tok.decode([i]) for i in e.ids])
e=tok.encode('"values":{"X":1.2873731568992153,"Y":-0.6175376044829376,"Z":-0.270917213687895}', add_special_tokens=False)
print("   in-context history values piece:", [tok.decode([i]) for i in e.ids])
# aggregate over all 1698 prompts
agg=collections.defaultdict(list)
for r in rows:
    rs,_,_,_=analyse(r); n=rs["tokens"]; tc=rs["tok_class"]
    agg["const"].append((tc.get("C",0)+12)/n); agg["W"].append(tc.get("W",0)/n); agg["stepvals"].append((tc.get("S",0)+tc.get("Sn",0))/n)
    agg["numeric"].append(tc.get("Sn",0)/n); agg["histkeys"].append(tc.get("Sk",0)/n); agg["scm_num_tokens"].append(sum(1 for (a,b) in [] ) )
    agg["hist_tokens"].append(rs["tok_sect"].get("history",0)/n); agg["scm_tokens"].append(rs["tok_sect"].get("scm",0)/n); agg["scm_abs"].append(rs["tok_sect"].get("scm",0))
    agg["task_abs"].append(rs["tok_sect"].get("task",0)); agg["instr_abs"].append(rs["tok_sect"].get("instructions",0)); agg["menu_abs"].append(rs["tok_sect"].get("menu",0))
print("ALL 1698 prompts token-id reconstruction verified. Fractions of prompt tokens (min / mean / max):")
for k in ("const","W","stepvals","numeric","histkeys","hist_tokens","scm_tokens"):
    v=agg[k]; print(f"   {k:12s} {min(v):.3f} / {S.mean(v):.3f} / {max(v):.3f}")
for k in ("scm_abs","task_abs","instr_abs","menu_abs"):
    v=agg[k]; print(f"   {k:12s} tokens min {min(v)} mean {S.mean(v):.1f} max {max(v)}")
