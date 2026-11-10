"""Extract display/CSV data from accepted saved reports. No model or experiment code."""
from pathlib import Path
import csv, hashlib, json, math, sys
from collections import Counter
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/ace_llm_intervention_scientific_finalization_20261108"
SOURCE = BASE / "report-retrieval01"
DEST = Path(sys.argv[1]).resolve()
DEST.mkdir(exist_ok=False)
PINS = {
 "handoff.json":"f95d744d29ad1fa4200b8b654e087cbcf5d7768013580b140c987fe6ebff8db9",
 "operation/receipt.json":"52f9cee616a67dec67a1ed6218b958dc4cf01399e1e8511edd4396a5f8c57e5b",
 "inputs/report-members.json":"012e5fc3f4994ff46b1776b2f50054084e080d946c4ad0780ae0b3dcde803ff4",
 "inputs/acceptance.json":"8ae40bd7952694271be0890590972250a5d8777f6856261b983b330fd999cb97",
}
LABELS = dict(zip(
 ["uniform_random","menu_coverage","local_information","target_risk","learned_ranker","frozen_llm","sft_llm","dpo_llm"],
 ["Uniform random","Menu coverage","Local information","Target risk","Numerical ranker","Frozen LLM","Supervised LLM","DPO LLM"]))
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read_pinned(path, pin):
 raw=path.read_bytes()
 assert sha(raw)==pin, str(path)
 return json.loads(raw)
def write_json(name,value):
 (DEST/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def write_csv(name,rows):
 assert rows
 with (DEST/name).open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
authenticated={name:read_pinned(SOURCE/name,pin) for name,pin in PINS.items()}
assert authenticated["inputs/acceptance.json"]["execution_qualified"] is True
members=authenticated["inputs/report-members.json"]
assert len(members)==242
raws={}
for name,info in members.items():
 raw=(SOURCE/"opaque"/name).read_bytes()
 assert len(raw)==info["bytes"] and sha(raw)==info["sha256"],name
 raws[name]=raw
prefix="cpu/cpu/worker/artifacts/campaign/private/"
assert sha(raws[prefix+"summary.json"])=="3f17abf3428411c44095dadcc3d80515267734d3729a0dbaef0989746dd7be0a"
s=json.loads(raws[prefix+"summary.json"])
assert s["cell_count"]==192 and len(s["policy_pairs"])==28
assert s["roster"]["policy_ids"]==list(LABELS)
worlds=[dict(id=w["world_id"], index=i, **w["strata"]) for i,w in enumerate(s["roster"]["worlds"])]
assert len(worlds)==24
reports={}
for i,w in enumerate(worlds):
 for j,policy in enumerate(LABELS):
  name=prefix+f"world-{i:02d}/arm-{j:02d}/cost-report.json"
  report=json.loads(raws[name]);assert (report["world_id"],report["policy_id"])==(w["id"],policy)
  reports[w["id"],policy]=report
cells=[];curves=[];action_rows=[]
for row in s["rows"]:
 key=row["world_id"],row["policy_id"];r=reports[key]
 assert row["expected_report_sha256"]==sha(raws[prefix+f"world-{key[0]-202611070100:02d}/arm-{list(LABELS).index(key[1]):02d}/cost-report.json"])
 cell=dict(world_id=key[0],policy_id=key[1],**row["strata"],
  capped_total_cost=row["capped_total_cost_units"],capped_acquisition_cost=row["capped_acquisition_cost_units"],
  final_error=row["final_error"],ever_attained=row["attained"],final_attained=row["final_attained"],
  failed=row["failed"],unresolved=row["cell_status"]=="unresolved",
  actual_total_cost=None if row["charges"] is None else row["charges"]["actual_total_cost_units"],
  termination_reason=row["termination_reason"], first_crossing_total=row["first_crossing_total_cost_units"])
 cells.append(cell)
 for point in r["curve"]:
  curve=dict(world_id=key[0],policy_id=key[1],total_cost_units=8+point["acquisition_cost_units"],
   acquisition_cost_units=point["acquisition_cost_units"],normalized_mse=point["normalized_mse"],
   action_id=point["action_id"],accepted_receipts=point["accepted_receipts"])
  curve.update({f"prediction_{i}":x for i,x in enumerate(point["predictions"])})
  curves.append(curve)
for policy in LABELS:
 counts=Counter(p["action_id"] for (w,pid),r in reports.items() if pid==policy for p in r["curve"] if p["kind"]=="accepted_acquisition")
 for action,count in sorted(counts.items()): action_rows.append(dict(policy_id=policy,action_id=action,accepted_acquisitions=count))
def compact_policy(p,ids):
 selected=[r for r in cells if r["policy_id"]==p["policy_id"] and r["world_id"] in ids]
 return dict(policy_id=p["policy_id"],label=LABELS[p["policy_id"]],world_count=len(ids),
  mean_capped_total_cost=p["mean_capped_total_cost"],mean_capped_acquisition_cost=p["mean_capped_acquisition_cost"],
  ever_attained=p["attained"],final_attained=sum(r["final_attained"] is True for r in selected),
  mean_final_error=p["all_world_mean_final_error"],failed=p["failed"],unresolved=p["unresolved"],
  actual_total_cost=p["logical_per_arm_actual_charges"]["total"]["all_world_sum"],
  stopped=sum(r["termination_reason"]=="policy_stop" for r in selected))
def compact_pair(p):
 return dict(left=p["left"],right=p["right"],mean_difference=p["mean_total_cost_difference"],
  wins=p["left_wins_lower_cost"],ties=p["ties"],losses=p["left_losses_higher_cost"],
  world_ids=p["world_ids"],differences=p["total_cost_differences"])
groups=[dict(id="all",label="All 24 systems",world_ids=[w["id"] for w in worlds],
 policies=[compact_policy(p,[w["id"] for w in worlds]) for p in s["policy_summaries"]],
 pairs=[compact_pair(p) for p in s["policy_pairs"]])]
for st in s["strata"]:
 groups.append(dict(id=st["field"]+"/"+st["value"],label=st["field"]+": "+st["value"],world_ids=st["world_ids"],
  policies=[compact_policy(p,st["world_ids"]) for p in st["policy_summaries"]],
  pairs=[compact_pair(p) for p in st["policy_pairs"]]))
data=dict(scope="24 synthetic development systems; descriptive, not confirmation",source_summary_sha256=sha(raws[prefix+"summary.json"]),
 source_members_sha256=PINS["inputs/report-members.json"],finalizer_acceptance_sha256=PINS["inputs/acceptance.json"],
 threshold=.04,initial_cost=8,total_cap=32,worlds=worlds,groups=groups,cells=cells,curves=curves,actions=action_rows)
write_json("display-data.json",data)
write_csv("policies.csv",groups[0]["policies"])
write_csv("cells.csv",cells)
write_csv("curves.csv",curves)
write_csv("actions.csv",action_rows)
write_csv("pairs.csv",[{**{k:v for k,v in p.items() if k not in ('world_ids','differences')},
 "world_ids":json.dumps(p['world_ids']),"differences":json.dumps(p['differences'])} for p in groups[0]["pairs"]])
write_csv("strata.csv",[dict(stratum=g["label"],**p) for g in groups[1:] for p in g["policies"]])
outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in DEST.iterdir() if p.is_file()}
write_json("derivation.json",dict(schema="ace-saved-report-display-extraction-v1",script_sha256=sha(Path(__file__).read_bytes()),
 inputs=PINS,summary_sha256=sha(raws[prefix+"summary.json"]),source_members=members,
 output_files=outputs,cells=len(cells),curves=len(curves),groups=len(groups),pairs=28,
 new_models=0,new_responses=0,new_evaluations=0,scientific_acceptance=False,
 scope="Saved-report display extraction. Independent numeric review is separate."))
print(json.dumps(dict(cells=len(cells),curves=len(curves),groups=len(groups),outputs=outputs)))
