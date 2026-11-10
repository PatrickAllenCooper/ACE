from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).parent;R=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108');C=R/'monitor-collection-control01';pins={}
def load(p):
 b=p.read_bytes();pins[str(p)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return json.loads(b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=load(C/'completion.json');run=load(C/'collector01/run.json');command=load(C/'collector01.command.json');gate=load(C/'terminal-collection-gate.json');failure=load(R/'launch-control01/failure-disposition.json');monitor=load(C/'monitor06/run.json')
# Continue after prior inspection's dict-vs-string expectation error; earlier passed predicates unchanged.
for n,h in x['capsule_pins'].items():assert sha(R/'terminal01'/n)==h['sha256'] and (R/'terminal01'/n).stat().st_size==h['bytes']
assert x['accounting']==load(R/'terminal01/disposition.json')['accounting']
collector=Path(command['command'][4]);assert sha(collector)==gate['collector_sha256']==x['capsule_pins']['pull_terminal.py']['sha256']
assert datetime.datetime.fromisoformat(x['scheduler_end_utc'].replace('Z','+00:00'))<datetime.datetime.fromisoformat(run['started_utc'])<datetime.datetime.fromisoformat(run['ended_utc'])
assert x['locks_released_on_process_exit'] is True and x['further_remote_operations'] is False
out={'status':'actual_control_lineage_pass','pins':pins,'prior_local_failure_no_allocation':True,'monitor_limit_failure_retained':True,'sole_collector_after_terminal':True,'collector_whole_invocation_cost':run,'terminal_gate':gate,'reviewer_attempts':[{'source':'close_control.py','status':'inspector_shape_error_after_initial_control_checks','issue':'handoff capsule_pins entries are bytes/sha256 dicts, not SHA strings; no evidence contradiction'},{'source':'close_control02.py','status':'remaining_checks_pass'}],'notes':['No remote observation was repeated. Current remote accessibility is outside this saved-evidence review.','One exploratory metadata-view command encountered an AttributeError when a command-list was formatted as a dict; original tool transcript retained. Both main independent audit scripts passed their first invocation without modification.']}
(O/'control-lineage.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print('control lineage pass')
