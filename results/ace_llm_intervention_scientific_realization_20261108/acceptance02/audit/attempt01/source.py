"""Offline saved-custody audit. No project imports, network, model or regeneration.

Only stdout is written. A passing report is not itself acceptance. Independent
review/publication must bind this source, its inputs and retained run evidence.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath as P
import re
import stat
import subprocess

HERE = Path(__file__).parent
REPO = Path('/Users/pat/code/ACE')
PACKET = 'ae077ef888671701eaa29415248dc7442656eebe2e5693df14ea6a22d661d8d7'
FREEZE = '15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
COMMIT = 'a9346022f145cc29439325b6f291597b2e6ec037'
PACKET_COMMIT = '47524a99829f490d6a4446d47cc745cadf51aa60'
JOB = '35406081'
INTENT = '0ccc743d339adccc11f865952a946f42a574595f94d5b978be40d3f4f5da05d6'
RESULT = 'c1b22b2883d9da5530650b2a1cf9a20ad6caef5d7305770f255cb1963d432e8b'
OLD_CLAIM = '58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29'
VERSION = 'ace-scientific-cpu-supervisor-v1'

def need(v, code):
    if not v: raise ValueError(code)  # Never include descriptor values in errors.

def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def pairs(rows):
    d = {}
    for k,v in rows:
        need(k not in d, 'duplicate_key'); d[k] = v
    return d
def decode(b):
    return json.loads(b, object_pairs_hook=pairs, parse_constant=lambda _: need(False, 'nonfinite_json'))
def read(p, pin=None):
    p = Path(p); s = p.lstat()
    need(stat.S_ISREG(s.st_mode) and s.st_size <= 64*1024**2, 'regular_bounded_file')
    b = p.read_bytes(); t = p.lstat()
    need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns), 'read_changed')
    need(pin is None or sha(b)==pin, 'file_pin'); return b
def load(p, pin=None): return decode(read(p,pin))
def inv(root):
    need(root.is_dir() and not root.is_symlink(), 'root_directory')
    files={}; dirs=[]; total=0
    for p in sorted(root.rglob('*')):
        need(not p.is_symlink(), 'custody_symlink')
        name=p.relative_to(root).as_posix()
        if p.is_dir(): dirs.append(name); continue
        b=read(p); total+=len(b)
        need(p.stat().st_nlink==1 and total<=2*1024**3 and len(files)<8192, 'independent_bounded_custody')
        files[name]={'bytes':len(b),'sha256':sha(b)}
    return {'files':files,'directories':dirs}
def git(commit,path):
    return subprocess.check_output(['git','--no-replace-objects','show',commit+':'+path],cwd=REPO)
def inside(p,root):
    p,r=P(p),P(root);return p==r or r in p.parents
def normalize(rows):
    out=[]
    for r in rows:
        need(len(r)>=8 and not any(r[8:]),'accounting_fields')
        out.append([r[0],r[1].strip(),r[2].strip(),r[3].strip(),int(r[4]),int(r[5]),r[6].strip(),sorted(tres(r[7]).items())])
    need(len({r[0] for r in out})==len(out),'duplicate_job')
    return sorted(out,key=lambda r:int(r[0]))
def tres(s): return pairs(x.split('=',1) for x in s.split(',') if x)
def cpu_tres(s):
    t=tres(s)
    need(t.get('cpu')=='2' and t.get('node')=='1' and t.get('mem') in ('7680M','7.50G','7.5G') and not any(k.startswith('gres/') for k in t),'cpu_tres')
def transport(root,name):
    t=load(root/name);need(t['returncode']==0,'transport_failed');return decode(t['stdout'])
def source_constant(data,name):
    # Only string constants needed here; never execute a module/AST function.
    for n in ast.parse(data).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
            return ast.literal_eval(n.value)
    raise ValueError('missing_static_constant')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--terminal',required=True);ap.add_argument('--mirror',required=True)
    a=ap.parse_args(); root=Path(a.terminal).absolute(); mirror=Path(a.mirror).absolute()
    # Custody and authenticated source gates precede all descriptor decoding.
    first=inv(root); need(first==inv(mirror),'two_complete_copies')
    ci=load(root/'custody-inventory.json'); trimmed={**first,'files':{k:v for k,v in first['files'].items() if k!='custody-inventory.json'}}
    need(ci['identical'] is True and ci['acceptance'] is False and len(ci['copies'])==2 and all(v==trimmed for v in ci['copies'].values()),'custody_inventory')
    d=load(root/'disposition.json'); need(d['status']=='terminal_custody_review_pending' and d['errors']==[] and d['input_pin_mismatches']==[] and d['current_lineage_verified'] is True,'custody_disposition')
    before=transport(root,'inventory_transport.json'); after=transport(root,'inventory_after_transport.json')
    raw=inv(root/'remote')
    for obs in (before,after):
        need(obs.get('terminal_gate') is True and not obs.get('observation_error'),'remote_observation')
        need({'files':obs['files'],'directories':obs['directories']}==raw,'remote_inventory')
    need(before['utc']<after['utc'] and before['journal_pins']==after['journal_pins'] and normalize(before['related_accounting'])==normalize(after['related_accounting']),'two_observations')
    need(len(raw['files'])==331 and sum(v['bytes'] for v in raw['files'].values())==5292431,'concrete_raw_roster')
    launch=root/'remote/launch'; captured=HERE/'captured-launch'
    outer=load(launch/'packet.json',PACKET); pinned={**outer['new_input_files'],'packet.json':PACKET}
    need(len(outer['new_input_files'])==44,'concrete_input_roster')
    for name,h in pinned.items():
        b=read(launch/name,h);need(b==read(captured/name,h),'preexecution_capture')
        need(b==git(PACKET_COMMIT,'results/ace_llm_intervention_scientific_realization_20261108/launch02/'+name),'committed_packet')
        for label in ('local-launch','external-launch'):
            need(b==read(root/'coordinator'/label/name,h),'coordinator_packet')
    inner=load(launch/'inputs/packet.json'); ip=pinned['inputs/packet.json']; admission=pinned['inputs/admission.json']
    need(outer['source_commit']==inner['source_commit']==COMMIT and inner['freeze_sha256']==FREEZE and inner['mode']=='realize','fixed_source_freeze')
    for path,h in outer['source_pins'].items():
        b=read(launch/'inputs'/Path(path).name,h);need(b==git(COMMIT,path),'committed_source')
    for name,h in inner['files'].items(): read(launch/'inputs'/name,h)
    need(set(inner['files'])|{'packet.json','admission.json'}=={p.name for p in (launch/'inputs').iterdir()},'flat_input_roster')
    freeze=load(launch/'freeze.json',FREEZE); ns=load(launch/'namespace-admission.json',inner['namespace_admission_sha256'])
    need(ns['freeze_sha256']==FREEZE and ns['retry_allowed'] is False and ns['replacement_worlds'] is False,'namespace_descriptor')
    base=load(launch/'accepted-accounting.json',outer['baseline_sha256'])
    need(base['terminal_jobs']==39 and base['submission_journals']==80 and base['closed_cpu_seconds']==11048 and base['closed_gpu_slice_seconds']==1601 and base['outstanding_reservation']==0,'accepted_baseline')
    authority=load(launch/'recovery-authority.json'); failure=load(launch/'failure-acceptance.json')
    need(failure['status']=='accepted_failed_before_draw' and failure['scientific_worlds_generated']==0 and failure['environment_responses']==0 and failure['realizer_called'] is False,'predecessor_no_draw')
    need(authority['successor_freeze_sha256']==FREEZE and authority['failed_job_id']=='35402725' and authority['original_namespace_claim_sha256']==OLD_CLAIM and authority['automatic_retry_allowed'] is False,'recovery_authority')
    failed=load(launch/'failed-raw-inventory.json')
    oldfiles={('namespace.json' if n=='original-namespace.json' else 'launch/'+n.removeprefix('failed01/')):v for n,v in raw['files'].items() if n=='original-namespace.json' or n.startswith('failed01/')}
    olddirs=sorted('launch' if n=='failed01' else 'launch/'+n.removeprefix('failed01/') for n in raw['directories'] if n=='failed01' or n.startswith('failed01/'))
    need(oldfiles==failed['files'] and olddirs==sorted(failed['directories']) and len(oldfiles)==41,'original_failed_custody')
    for p in (root/'remote/original-namespace.json',root/'namespaces/original-local.namespace.json'):read(p,OLD_CLAIM)
    result=load(launch/'submission.result.json',RESULT); intent=load(launch/'submission.intent.json',INTENT)
    need(result['job_id']==JOB and result['returncode']==0 and result['ambiguous'] is False and result['intent_sha256']==INTENT,'once_only_result')
    need(intent['packet_commit']==PACKET_COMMIT and intent['packet_sha256']==PACKET and intent['pins']==pinned and intent['source_commit']==COMMIT,'once_only_intent')
    for label in ('local-launch','external-launch'):
        coord=root/'coordinator'/label; local=load(coord/'submission.intent.json')
        need(read(coord/'submission.result.json',RESULT)==read(launch/'submission.result.json'),'coordinator_result')
        need(set(intent)==set(local)|{'remote_utc','fresh_readiness','remote_namespace_claim_sha256'} and all(intent[k]==v for k,v in local.items()),'augmented_intent')
        controller=load(coord/'controller.json')
        need({Path(p).name for p in controller['held']}=={'ace-delivery-controller.lock','ace-foundation-controller.lock'} and controller['pid']==local['controller_pid_local'],'both_locks')
    claimpin=intent['local_namespace_claim_sha256']; claim=load(root/'remote/namespace.json',claimpin)
    need(read(root/'namespaces/local.namespace.json',claimpin)==read(root/'remote/namespace.json'),'recovery_claim_copies')
    need(intent['remote_namespace_claim_sha256']==claimpin and claim['packet_sha256']==PACKET and claim['packet_commit']==PACKET_COMMIT and claim['freeze_sha256']==FREEZE and claim['namespace_admission_sha256']==inner['namespace_admission_sha256'],'recovery_claim_binding')
    need(claim['world_ids']==ns['world_ids'] and claim['claim_kind']=='explicit_pre_draw_recovery02' and claim['retry_allowed'] is False and claim['replacement_worlds'] is False,'claim_roster')
    for k in ('failed_job_id','failure_acceptance_sha256','no_draw_proof_sha256','original_namespace_claim_sha256'):
        need(claim[k]==authority[k] and intent[k]==authority[k],'claim_predecessor')
    need(claim['recovery_authority_sha256']==pinned['recovery-authority.json'],'claim_authority')
    for ready in (load(launch/'readiness.json'),intent['fresh_readiness']):
        need(normalize(ready['related_accounting'])==normalize(base['related_accounting']) and ready['journal_pins']==base['journal_pins'],'prior_twice_reconciled')
        need(not any('|ace' in line.lower() for line in ready['queue'].splitlines()),'prelaunch_queue')
    journals={**base['journal_pins'],outer['remote_root']+'/submission.intent.json':INTENT,outer['remote_root']+'/submission.result.json':RESULT}
    need(len(journals)==82 and journals==before['journal_pins'],'82_exact_journals')
    rows=before['related_accounting'];need(len(rows)==40 and normalize([r for r in rows if r[0]!=JOB])==normalize(base['related_accounting']),'40_exact_jobs')
    row=next(r for r in rows if r[0]==JOB)
    need(row[1:5]==['ace_llm_scientific_realize02','COMPLETED','0:0','2'] and row[5]=='43' and row[6]=='ucb736_asc2','actual_success');cpu_tres(row[7])
    for obs in (before,after):
        for name in ('details','steps','queue'):need(obs['results'][name]['returncode']==0,'scheduler_query')
        details=[l.split('|') for l in obs['results']['details']['stdout'].splitlines() if l.strip()]
        need(len(details)==1 and details[0][:4]==[JOB,'acpu','cpu-normal','5'] and details[0][5]=='paco0228','scheduler_requested');cpu_tres(details[0][4])
        parents=[l.split('|')[:8] for l in obs['results']['steps']['stdout'].splitlines() if l.startswith(JOB+'|')]
        need(normalize(parents)==normalize([row]),'scheduler_step_parent')
        need(not any('|ace' in l.lower() or l.startswith(JOB+'|') for l in obs['results']['queue']['stdout'].splitlines()),'terminal_queue')
    out=launch/'realization'; session=out/'session'; worker=session/'worker'; task=worker/'task'
    need(not any(any(t in p.name for t in ('failure','interruption')) for p in out.rglob('*')),'no_failure_markers')
    allocation=load(out/'allocation.json');need(allocation['returncode']==0,'live_allocation')
    fields=pairs(re.findall(r'([^\s=]+)=(\S+)',allocation['stdout']))
    expected={'JobId':JOB,'JobState':'RUNNING','Account':'ucb736_asc2','Partition':'acpu','QOS':'cpu-normal','NumCPUs':'2','NumTasks':'1','NumNodes':'1','TimeLimit':'00:05:00','Requeue':'0','Restarts':'0'}
    need(all(fields.get(k)==v for k,v in expected.items()) and fields['UserId'].startswith('paco0228('),'live_identity');cpu_tres(fields['AllocTRES']);cpu_tres(fields['ReqTRES'])
    accelerators=load(out/'accelerators.json');need(accelerators['devices']==[] and not any(accelerators['environment'].values()),'no_accelerators')
    parent=load(out/'terminal.json','f75f6358f2a95ba47cd8470a45031005afb058935f35a091c9897220d4320974')
    wt=load(worker/'terminal.json','e484098f96e57929d327eca56114d2fd1cc79ee6789854c202febd56989793bc')
    for directory,value,status in ((out,parent,'complete'),(worker,wt,'worker_complete')):
        need(read(directory/'terminal.json')==read(directory/'terminal-candidate.json'),'terminal_publication')
        need(value['schema']==VERSION and value['status']==status and value['mode']=='realize' and value['packet_sha256']==ip and value['admission_sha256']==admission,'terminal_bindings')
    stage=load(out/'stage.json',parent['stage_sha256']);need(wt['stage_sha256']==parent['stage_sha256'],'worker_stage')
    prefix=stage['staged_prefix'];need(prefix==f'/scratch/local/{JOB}/ace-scientific-realize-{JOB}/runtime','node_local_prefix')
    need(stage['version']=='ace-runtime-archive-v1' and stage['archive_sha256']==inner['archive']['sha256'] and stage['manifest_sha256']==inner['manifest']['sha256'] and stage['objects']==49966 and stage['file_bytes']==8397575073 and stage['model_loaded'] is False and stage['runtime_qualified'] is False,'stage_receipt')
    need(parent['job_id']==JOB and parent['scientific_qualified'] is False and parent['retry_allowed'] is False and parent['worker_terminal_sha256']==sha(read(worker/'terminal.json')) and 0<parent['elapsed_seconds']<290,'outer_success')
    start=load(session/'worker.started.json');handoff=load(session/'handoff.json');marker=load(worker/'work-start.json');resources=load(session/'worker.resources.json');cleanup=load(session/'worker.cleanup.json')
    origin=handoff['started_monotonic'];pid=start['pid']
    need(handoff['job_id']==JOB and handoff['packet_sha256']==ip and handoff['stage_sha256']==parent['stage_sha256'] and start['started_monotonic']==origin and start['pgid']==pid,'owned_handoff')
    need(start['startup_deadline']==origin+120 and start['deadline_monotonic']==origin+260 and start['supervisor_deadline_monotonic']==origin+290,'absolute_deadlines')
    need(marker['packet_sha256']==ip and marker['pid']==pid and origin<=marker['monotonic']<origin+120,'startup_observed')
    need(resources['startup_observed'] is True and resources['returncode']==0 and resources['errors']==[] and resources['elapsed_including_cleanup_seconds']<260,'supervision_result')
    need(cleanup==resources['cleanup']==parent['cleanup'] and cleanup['pid']==cleanup['pgid']==pid and cleanup['group_still_exists'] is False and cleanup['kill_sent'] is False and cleanup['returncode']==0 and cleanup['errors']==[],'natural_group_cleanup')
    command=start['command']; diag=read(launch/'inputs/qualify_llm_intervention_runtime_metadata.py')
    expected_command=[prefix+'/bin/python','-I','-B','-c',source_constant(diag,'BOOTSTRAP'),outer['remote_root']+'/inputs/supervise_llm_intervention_scientific_cpu.py',inner['files']['supervise_llm_intervention_scientific_cpu.py'],'--packet',outer['remote_root']+'/inputs/packet.json','--packet-sha256',ip,'--source-commit',COMMIT,'--admission',outer['remote_root']+'/inputs/admission.json','--admission-sha256',admission,'--output',inner['output']+'/session/worker','--worker','--handoff-sha256',sha(read(session/'handoff.json'))]
    need(command==expected_command,'exact_authenticated_child')
    bindings=[]
    for directory,logical in ((out,inner['output']),(worker,inner['output']+'/session/worker')):
        s=load(directory/'start.json');need(s['packet_sha256']==ip and s['resources']==inner['resources'] and s['deadlines']=={k:origin+v for k,v in inner['deadlines'].items()},'start_binding')
        need([b['logical'] for b in s['output_path_bindings']]==[logical,inner['output']],'logical_outputs')
        for b in s['output_path_bindings']:
            need(b['mount_alias'] is True and b['resolved']==b['logical'].replace('/scratch/alpine/','/gpfs/alpine1/scratch/',1) and type(b['root_device']) is int and type(b['root_inode']) is int,'observed_mount_binding');bindings.append(b)
    need(len({(b['root_device'],b['root_inode']) for b in bindings})==1,'same_mount_identity')
    expected_projects={Path(n).stem:inner['files'][n] for n in inner['source_names'] if n!='supervise_llm_intervention_scientific_cpu.py'}
    before_runtime=load(worker/'runtime-before.json'); controls={'CUDA_VISIBLE_DEVICES':'','LD_LIBRARY_PATH':prefix+'/lib','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','MKL_THREADING_LAYER':'GNU'}
    for phase in ('before','before_realization','after_realization'):
        record=load(worker/('runtime-'+phase+'.json'));snap=record['snapshot']
        need(record['environment']==controls and record['accelerators']==accelerators,'runtime_controls')
        need(snap['python_version']=='3.10.19' and snap['prefix']==snap['base_prefix']==prefix and inside(snap['executable'],prefix),'runtime_interpreter')
        need(snap['sys_path']==before_runtime['snapshot']['sys_path'] and all(inside(p,prefix) for p in snap['sys_path']),'runtime_search_paths')
        need(not snap['generated_provenance']['present'] and not snap['mixed_module_paths'] and not snap['mixed_modules'] and not snap['mixed_native_paths'],'full_cpu_runtime_guard')
        need(all(inside(p,prefix) for p in snap['module_paths'].values()),'loaded_module_paths')
        for line in snap['process_maps'].splitlines():
            fields=line.split(None,5)
            if len(fields)==6 and fields[5].startswith('/'):
                p=fields[5].removesuffix(' (deleted)');need(inside(p,prefix) or any(inside(p,r) for r in ('/lib','/lib64','/usr/lib','/usr/lib64')),'native_mapping')
        need(not any(n.split('.')[0] in {'torch','transformers','tokenizers','scipy','safetensors'} for n in snap['module_paths']),'no_model_packages')
        need(all(not v.get('version_error') for v in snap['packages'].values()) and all(snap['packages'][n]['loaded'] is False for n in ('torch','transformers','scipy')),'forbidden_packages')
        if phase=='before':need(not record['captured_project_modules'] and snap['packages']['numpy']['loaded'] is False,'preimport_gate')
        else:
            need({n:v['sha256'] for n,v in record['captured_project_modules'].items()}==expected_projects,'live_source_closure')
            need(all(v['loader']=='authenticated_captured_bytes' and v['path']==outer['remote_root']+'/inputs/'+n+'.py' for n,v in record['captured_project_modules'].items()),'captured_loader_paths')
            need(snap['packages']['numpy']['version']==inner['runtime']['imported_versions']['numpy'] and any(n.startswith('numpy.') and p.endswith('.so') for n,p in snap['module_paths'].items()),'actual_numpy')
    need(wt['summary']=={'worlds_generated':56,'environment_response_attempts':0,'environment_cost_units':0} and all(wt[k]==0 for k in ('model_loads','training_updates','tokenizer_calls')),'zero_nonrealization_work')
    tree=load(worker/'task-inventory.json',wt['task_inventory_sha256']);actual=inv(task)
    need(tree=={'files':{k:v['sha256'] for k,v in actual['files'].items()},'bytes':sum(v['bytes'] for v in actual['files'].values())},'complete_task_inventory')
    taskpin='3b332e969422cf528d23b780721a6eff8bbb1bec171eab26346e7767de878e22'; terminal=load(task/'terminal.json',taskpin)
    need(read(task/'terminal.json')==read(task/'terminal.candidate.json'),'task_publication')
    proposal=load(launch/'inputs/proposal.json');need(load(task/'proposal.json')==proposal,'task_proposal')
    need(sha(canonical(proposal))==ns['proposal_semantic_sha256'],'proposal_semantic')
    common={'version':'ace-first-screen-descriptor-realization-v1','proposal_sha256':ns['proposal_semantic_sha256'],'source_pins':inner['protocol_source_pins'],'freeze_sha256':FREEZE,'namespace_admission_sha256':inner['namespace_admission_sha256'],'retry_allowed':False,'environment_responses':0,'policy_updates':0,'scientific_execution_qualified':False}
    taskintent=load(task/'intent.json')
    for obj in (terminal,taskintent):need(all(type(obj[k]) is type(v) and obj[k]==v for k,v in common.items()),'task_scope_binding')
    need(terminal['status']=='complete' and terminal['worlds_generated']==56,'task_success')
    rows=proposal['training_worlds']+proposal['development_worlds'];need(len(rows)==len(terminal['entries'])==56,'56_roster')
    need([r['world_id'] for r in rows[:32]]==ns['world_ids']['policy_train'] and [r['world_id'] for r in rows[32:]]==ns['world_ids']['development'],'32_24_namespaces')
    expected_files={'intent.json','proposal.json','terminal.json','terminal.candidate.json'}; descriptor_pins=[]
    for i,(r,e) in enumerate(zip(rows,terminal['entries'])):
        name=f'world-{i:03d}.json'; intentname=f'world-{i:03d}.intent.json';expected_files|={name,intentname}
        need(set(e)=={'scope','world_id','path','status','descriptor_sha256','file_sha256'} and e['scope']==r['scope'] and e['world_id']==r['world_id'] and e['path']==name and e['status']=='complete','ordered_descriptor_entry')
        need(taskintent['entries'][i]=={'scope':r['scope'],'world_id':r['world_id'],'path':name,'status':'unstarted'},'initial_draw_roster')
        need(load(task/intentname)=={'scope':r['scope'],'world_id':r['world_id']},'per_world_intent')
        # Decode private values only after custody/source/runtime gates; never print them.
        value=load(task/name,e['file_sha256']);need(sha(canonical(value))==e['descriptor_sha256'],'descriptor_semantic')
        need(set(value)=={'version','scope','world_id','graph','mechanisms'} and value['version']=='ace-linear-gaussian-world-v1' and value['scope']==r['scope'] and value['world_id']==r['world_id'] and value['graph']==r['public']['graph'],'descriptor_schema')
        need(len(value['mechanisms'])==len(value['graph']),'mechanism_count')
        for n,m in zip(value['graph'],value['mechanisms']):
            need(set(m)=={'node','intercept','coefficients','noise_variance'} and m['node']==n['name'] and [c[0] for c in m['coefficients']]==n['parents'],'mechanism_shape')
            numbers=[m['intercept'],m['noise_variance']]+[c[1] for c in m['coefficients']]
            need(all(type(v) in (int,float) and math.isfinite(v) for v in numbers) and m['noise_variance']>=0,'mechanism_finite')
        descriptor_pins.append({k:e[k] for k in ('scope','world_id','path','file_sha256','descriptor_sha256')})
    need(set(actual['files'])==expected_files and not actual['directories'],'exact_task_files')
    need(inv(root)==first and inv(mirror)==first,'final_complete_readback')
    report={'schema':'ace-realization02-offline-audit-v1','status':'audit_passed_pending_independent_acceptance','required_findings':0,'job_id':JOB,'packet_sha256':PACKET,'source_commit':COMMIT,'packet_commit':PACKET_COMMIT,'submission_intent_sha256':INTENT,'submission_result_sha256':RESULT,'freeze_sha256':FREEZE,'namespace_admission_sha256':inner['namespace_admission_sha256'],'realization_terminal_sha256':taskpin,'parent_terminal_sha256':sha(read(out/'terminal.json')),'worker_terminal_sha256':sha(read(worker/'terminal.json')),'stage_sha256':parent['stage_sha256'],'complete_custody_verified':True,'raw_files':len(raw['files']),'raw_bytes':sum(v['bytes'] for v in raw['files'].values()),'custody_files':len(first['files']),'custody_bytes':sum(v['bytes'] for v in first['files'].values()),'custody_inventory_sha256':sha(read(root/'custody-inventory.json')),'descriptor_pins':descriptor_pins,'worlds_generated':56,'training_descriptors':32,'development_descriptors':24,'environment_responses':0,'model_loads':0,'training_updates':0,'tokenizer_calls':0,'runtime_qualified_for_this_execution':True,'owned_process_cleanup_verified':True,'scientific_qualified':False,'allocated_cpu_seconds':86,'gpu_slice_seconds':0,'mount_binding_verified':True,'recovery_claim_sha256':claimpin,'original_namespace_claim_sha256':OLD_CLAIM,'terminal_jobs':40,'submission_journals':82,'closed_cpu_seconds':11134,'closed_gpu_slice_seconds':1601,'cpu_admission_after_release':14734,'outstanding_reservation':0,'related_accounting':before['related_accounting'],'journal_pins':journals}
    print(json.dumps(report,sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__': main()
