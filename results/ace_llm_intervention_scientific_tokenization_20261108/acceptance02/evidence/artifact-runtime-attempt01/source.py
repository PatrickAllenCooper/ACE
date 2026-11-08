from pathlib import Path
import hashlib,json,importlib.util,shutil,time,traceback,sys
ROOT=Path(__file__).resolve().parent
T=Path('/Users/pat/code/ACE/results/ace_llm_intervention_scientific_tokenization_20261108/terminal02')
R=T/'remote/launch';P=R/'preparation';A=P/'session/worker/artifacts'
AID=Path('/private/tmp/ace-token02-audit-aid-r1-r2-z69bc7os')
REVIEW=Path('/Users/pat/code/ACE/docs/development/guidance/reviews/ace_llm_intervention_scientific_token02_audit_aid_review_2026-11-08.md')
AID_PIN='b3804801e460b9aff44465683dc0fb5cb627af299f883f666ec773c94ab0cf8c'
REVIEW_PIN='9f6e0926e7ec71d36f93dbfb3f33f2093eced9f6abe7b1acb1041844aca5cc2e'

def sha(x):return hashlib.sha256(x).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def j(p):return json.loads(p.read_bytes())

def main(out):
 need(j(ROOT/'custody-attempt01/receipt.json')['status']=='custody_source_history_checks_passed','whole custody prerequisite')
 need(j(ROOT/'source-scheduler-attempt03/receipt.json')['status']=='source_scheduler_original_clock_checks_passed','source/absolute clock prerequisite')
 need(sha(REVIEW.read_bytes())==REVIEW_PIN and sha((AID/'audit_saved.py').read_bytes())==AID_PIN,'independent cleared aid/review pins')
 shutil.copyfile(REVIEW,out/'aid-review.md');(out/'aid').mkdir()
 for n in ('audit_saved.py','reference-pins.json','HANDOFF.md','acceptance-contract.json'):shutil.copyfile(AID/n,out/'aid'/n)
 shutil.copytree(AID/'source',out/'aid/source')
 pins=j(AID/'reference-pins.json')
 for n,v in pins.items():
  need(sha((AID/n).read_bytes())==v['sha256'] and (AID/n).stat().st_size==v['bytes'],'cleared source/reference manifest '+n)
 spec=importlib.util.spec_from_file_location('cleared_token02_saved_aid',out/'aid/audit_saved.py');aid=importlib.util.module_from_spec(spec);spec.loader.exec_module(aid)
 seal=j(T/'custody-inventory.json');allfiles=seal['copies'][str(T)]['files']
 def read(path):
  raw=path.read_bytes();n=path.relative_to(T).as_posix();v=allfiles[n]
  need(sha(raw)==v['sha256'] and len(raw)==v['bytes'],'post-custody changed '+n);return raw
 packet_raw=read(R/'inputs/packet.json');packet=aid.decode(packet_raw)
 rawmap={n[7:]:v for n,v in allfiles.items() if n.startswith('remote/')}
 collectionroot=T/'remote/common/collection/session/worker/task/collection'
 cprefix='common/collection/session/worker/task/collection/'
 cfiles={n[len(cprefix):]:v for n,v in rawmap.items() if n.startswith(cprefix)}
 need({n:v['sha256'] for n,v in cfiles.items()}==packet['collection']['files'],'whole accepted collection map unchanged')
 cget=lambda n:read(collectionroot/aid.safe_relative(n))
 batch=aid.accepted_batch(packet,aid.decode(read(R/'inputs/proposal.json')),aid.decode(read(R/'inputs/collection-acceptance.json')),cget)
 getraw=lambda n:read(A/aid.safe_relative(n))
 get=lambda n:aid.decode(getraw(n))
 auth=aid.audit_publication(getraw,packet,batch)
 tree=dict(bytes=sum(v['bytes'] for v in cfiles.values()),files=len(cfiles),files_sha256=aid.digest(packet['collection']['files']))
 expected_auth=dict(packet_sha256=aid.INNER,source_sha256={n:packet['files'][n] for n in packet['source_names']},
  tokenizer_files={n:rawmap['launch/tokenizer/'+n] for n in packet['tokenizer_files']},tokenizer_sha256=aid.digest(packet['model_files']),collection=tree)
 need(auth==expected_auth,'full exact authenticated input/tree/tokenizer metadata')
 prepared=getraw('llm-prepared.json');numerical=getraw('numerical-plan.json');exposure_raw=getraw('exposure.json')
 facts=aid.audit_artifacts(packet_raw,prepared,numerical,exposure_raw,batch)
 stage_raw=read(P/'stage.json');provenance=aid.decode(read(P/'independent-provenance-sources.json'))
 runtime=aid.audit_runtime(get,stage_raw,provenance,packet)
 phases=[aid.decode(read(p)) for p in sorted(A.glob('phase-*.json'))]
 lifecycle=aid.audit_lifecycle(aid.decode(read(P/'terminal.json')),get('terminal.json'),aid.decode(read(P/'session/worker.cleanup.json')),
  aid.decode(read(P/'session/worker.resources.json')),aid.decode(read(P/'session/worker.started.json')),aid.decode(read(P/'session/worker/work-start.json')),
  phases,worker_raw=getraw('terminal.json'),stage_raw=stage_raw,facts=facts)
 exposure=aid.decode(exposure_raw);llm=aid.decode(prepared);num=aid.decode(numerical)
 # Independent public reporting summaries of the already checked complete schedule.
 command_lengths=sorted({len(v) for r in llm['training_plan']['records'] for v in r['commands'].values()})
 prompt_lengths=[len(r['prompt_ids']) for r in llm['training_plan']['records']]
 need(not any(n in sys.modules for n in ('torch','numpy','transformers','tokenizers')),'heavy package imported in saved-evidence audit')
 report=dict(status='saved_artifacts_runtime_lifecycle_checks_passed',aid_sha256=AID_PIN,aid_review_sha256=REVIEW_PIN,
  facts=facts,runtime=runtime,lifecycle=lifecycle,exposure_counts=exposure['counts'],
  numerical_width=num['feature_width'],numerical_parameter_count=num['parameter_count'],
  llm_adapter_parameters=llm['training_plan']['declared_adapter_parameters'],llm_adapter_shapes=len(llm['training_plan']['expected_adapter_shapes']),
  command_token_lengths=command_lengths,prompt_tokens_total=sum(prompt_lengths),
  one_pass_sft=llm['training_plan']['one_pass_sft'],one_pass_dpo=llm['training_plan']['one_pass_dpo'],
  selected_pairs_by_record={r['record_id']:[p['preference_index'] for p in r['pairs'] if p['exposure_count']] for r in exposure['records']},
  original_collection_accounting=batch['accounting'],zero_new_responses=True,zero_training_updates=True,
  exact_cleanup_duration_measured=False,node_local_runtime_deleted=False,feature_tensors_regenerated=False,tokenizer_rerun=False,
  allocation_clock_check_receipt_sha256=sha((ROOT/'source-scheduler-attempt03/receipt.json').read_bytes()),acceptance=False)
 (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='selected_pairs_by_record'},indent=2))
 return dict(status=report['status'],report_sha256=sha((out/'report.json').read_bytes()),facts=facts,acceptance=False)

if __name__=='__main__':
 out=ROOT/'artifact-runtime-attempt01';out.mkdir(mode=0o700,exist_ok=False);(out/'source.py').write_bytes(Path(__file__).read_bytes())
 (out/'intent.json').write_text(json.dumps(dict(aid_sha256=AID_PIN,aid_review_sha256=REVIEW_PIN,actual_saved_evidence_only=True,model_tensor_tokenizer_execution=False),indent=2)+'\n')
 start=time.monotonic();cpu=time.process_time()
 try:r=main(out);code=0
 except BaseException as exc:r=dict(status='failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),acceptance=False);code=1
 r.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,cost_scope='stdlib saved arithmetic and metadata checks; no model/tokenizer/tensor execution')
 (out/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(code)
