"""Independent saved-byte/JSON/Git/AST failure audit; no target imports or replay."""
from pathlib import Path
import ast,hashlib,json,os,re,stat,subprocess
R=Path('/Users/pat/code/ACE');E=Path(__file__).parent
L=R/'results/ace_llm_intervention_scientific_realization_20261108/terminal01'
X=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-realization-terminal-20261108-01')
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_realization_20261108_01';JOB='35402725'
OUTER='8ddbde39dd6538d5f45931f1eafd628a6d12415d199cb2f88b81799e306d37c1';INNER='dc9a181d45a630b83b97027fa03307143faff533de0200dfaa7ee8aa783aee21'
CLAIM='58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29';BASELINE='c5eb497d133a803f12b92ff7af07420cf4856d24e174162b9f634c4e0e27631f'
ENTRY='supervise_llm_intervention_scientific_cpu.py';ENTRY_SHA='794fd63af304ad7093c943f57dece7668ddd6bd5102a8c5db55cea19ee5054f9';HELPER='finalize_llm_intervention_campaign_cpu.py';HELPER_SHA='88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f'
COMMIT='f9b3197d37ecf987553c4c5741a16e9f5cac342a';inputs={}
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_nlink==1,str(p)
 b=p.read_bytes();h=sha(b)
 if str(p) in inputs:assert inputs[str(p)]['sha256']==h
 inputs[str(p)]={'bytes':len(b),'sha256':h};return b

def decode(b):
 def pairs(rows):
  d={}
  for k,v in rows:assert k not in d;d[k]=v
  return d
 return json.loads(b,object_pairs_hook=pairs)
def load(p):return decode(read(p))
def tree(root,exclude=()):
 files={};dirs=[]
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  rel=str(p.relative_to(root))
  if p.is_dir():dirs.append(rel)
  elif rel not in exclude:b=read(p);files[rel]={'bytes':len(b),'sha256':sha(b)}
 return dict(files=files,directories=dirs)

def normalized(rows):
 return sorted([[r[0],r[1].strip(),r[2].strip(),r[3].strip(),int(r[4]),int(r[5]),r[6].strip(),sorted(x.split('=',1) for x in r[7].split(',') if x)] for r in rows],key=lambda r:int(r[0]))
ci=load(L/'custody-inventory.json');assert read(L/'custody-inventory.json')==read(X/'custody-inventory.json')
assert ci['excludes']=='custody-inventory.json itself' and ci['identical'] is True
lt=tree(L,('custody-inventory.json',));xt=tree(X,('custody-inventory.json',));assert lt==xt==ci['copies'][str(L)]==ci['copies'][str(X)]
observations=[]
for n in ['inventory_transport.json','inventory_after_transport.json']:
 t=load(L/n);assert t['returncode']==0 and t['stderr']=='';v=decode(t['stdout']);observations.append(v)
 assert all(r['returncode']==0 and not r['stderr'] for r in v['results'].values())
first,last=observations
assert first['utc']<last['utc']
for k in ['files','directories','journal_pins','original_hardlink_groups','parent_accounting','related_accounting','root_present','namespace_present','terminal_gate']:assert first[k]==last[k],k
assert first['root_present'] is True and first['namespace_present'] is True and first['terminal_gate'] is True
raw=tree(L/'remote');assert raw==tree(X/'remote')
assert raw['files']==first['files'] and raw['directories']==first['directories'] and first['original_hardlink_groups']==[]
assert len(raw['files'])==41 and sum(v['bytes'] for v in raw['files'].values())==1940531
assert raw['directories']==['launch','launch/inputs','launch/logs']
assert not any(n.startswith('launch/realization') for n in raw['files'])
assert not (L/'remote/launch/realization').exists()
packet=load(L/'remote/launch/packet.json');assert sha(read(L/'remote/launch/packet.json'))==OUTER
assert sha(read(L/'remote/launch/inputs/packet.json'))==INNER
for n,h in dict(packet['new_input_files'],**{'packet.json':OUTER}).items():
 b=read(L/'remote/launch'/n);assert sha(b)==h
 for sub in ['coordinator/local-launch','coordinator/external-launch']:assert read(L/sub/n)==b
 gb=subprocess.check_output(['git','--no-replace-objects','show',COMMIT+':results/ace_llm_intervention_scientific_realization_20261108/launch01/'+n],cwd=R);assert b==gb
result=load(L/'remote/launch/submission.result.json');intent=load(L/'remote/launch/submission.intent.json')
assert result['job_id']==JOB and result['returncode']==0 and result['ambiguous'] is False and result['stdout'].strip()==JOB
assert result['intent_sha256']==sha(read(L/'remote/launch/submission.intent.json'))
for sub in ['coordinator/local-launch','coordinator/external-launch']:assert load(L/sub/'submission.result.json')==result
local_intent=load(L/'coordinator/local-launch/submission.intent.json');assert local_intent==load(L/'coordinator/external-launch/submission.intent.json')
assert all(intent[k]==v for k,v in local_intent.items())
assert intent['packet_sha256']==OUTER and intent['packet_commit']==COMMIT and intent['remote_root']==ROOT and intent['source_commit']==packet['source_commit']
claim=load(L/'remote/namespace.json');assert sha(read(L/'remote/namespace.json'))==CLAIM
for n in ['namespaces/local.namespace.json','coordinator/local-launch/namespace-claim.json','coordinator/external-launch/namespace-claim.json']:assert read(L/n)==read(L/'remote/namespace.json')
assert sha(read(claim['local_registry']))==CLAIM
assert intent['local_namespace_claim_sha256']==intent['remote_namespace_claim_sha256']==CLAIM
assert claim['freeze_sha256']==intent['freeze_sha256']==packet['new_input_files']['freeze.json']
assert claim['namespace_admission_sha256']==intent['namespace_admission_sha256']==packet['new_input_files']['namespace-admission.json']
assert claim['packet_commit']==COMMIT and claim['packet_sha256']==OUTER and claim['realization_root']==ROOT and claim['retry_allowed'] is False and claim['replacement_worlds'] is False
baseline=load(L/'remote/launch/accepted-accounting.json');assert sha(read(L/'remote/launch/accepted-accounting.json'))==BASELINE
rows=first['related_accounting'];assert len(rows)==39 and len({r[0] for r in rows})==39
current=[r for r in rows if r[0]==JOB];assert current==[first['parent_accounting']]
assert current[0][:7]==[JOB,'ace_llm_scientific_realize01','FAILED','1:0','2','4','ucb736_asc2']
assert normalized([r for r in rows if r[0]!=JOB])==normalized(baseline['related_accounting'])
assert all(r[2].split()[0] in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'] for r in rows)
journals=first['journal_pins'];assert len(journals)==80
new={ROOT+'/submission.intent.json':sha(read(L/'remote/launch/submission.intent.json')),ROOT+'/submission.result.json':sha(read(L/'remote/launch/submission.result.json'))}
assert journals==dict(baseline['journal_pins'],**new)
assert sum(int(r[4])*int(r[5]) for r in rows)==11048
assert sum(int(r[5]) for r in rows if 'gres/gpu:' in r[7])==1601
for ob in observations:
 res=ob['results'];fields=dict(re.findall(r'(\S+?)=(\S+)',res['job']['stdout']))
 expected=dict(JobId=JOB,JobName='ace_llm_scientific_realize01',Account='ucb736_asc2',QOS='cpu-normal',Partition='acpu',JobState='FAILED',ExitCode='1:0',Requeue='0',Restarts='0',NumCPUs='2',NumNodes='1',NumTasks='1',RunTime='00:00:04',TimeLimit='00:05:00',Command=ROOT+'/job.sbatch')
 assert all(fields[k]==v for k,v in expected.items())
 for k in ['ReqTRES','AllocTRES']:
  tres=dict(s.split('=',1) for s in fields[k].split(','));assert tres==dict(cpu='2',mem='7.50G',node='1',billing='2')
 assert res['details']['stdout'].strip()==JOB+'|acpu|cpu-normal|5|billing=2,cpu=2,mem=7.50G,node=1|paco0228'
 q=[l.split('|') for l in res['queue']['stdout'].splitlines() if l];assert not any(r[0]==JOB or r[1].lower().startswith('ace') for r in q)
 steps=[l.split('|') for l in res['steps']['stdout'].splitlines() if l]
 assert {r[0] for r in steps}=={JOB,JOB+'.batch',JOB+'.extern'}
 assert all(r[2] in ['FAILED','COMPLETED'] for r in steps)
 assert [r for r in steps if r[0]==JOB+'.batch'][0][2:4]==['FAILED','1:0']
# Verify diagnostic bytes and source-order statements, never call target methods.
err=read(L/'remote/launch/logs'/f'{JOB}.err');assert sha(err)=='aeadbd7365dcc616f86ea825b90ddca30551bf1ef56165de94b62d56429c3e62'
assert err.decode().endswith('ValueError: resolved output required\n') and 'line 407, in main' in err.decode() and 'line 51, in need' in err.decode()
assert read(L/'remote/launch/logs'/f'{JOB}.out')==b''
source=read(L/'remote/launch/inputs'/ENTRY);helper=read(L/'remote/launch/inputs'/HELPER)
assert sha(source)==ENTRY_SHA and sha(helper)==HELPER_SHA
lines=source.decode().splitlines();assert 'claimed=False' in lines[402] and 'args.owned_child=None' in lines[398]
assert "'resolved output required'" in lines[406] and 'out.mkdir' in lines[420] and 'claimed=True' in lines[420]
assert 'ctx=authenticate(args)' in lines[423] and 'else:parent(args,ctx)' in lines[425]
assert 'if claimed:' in lines[427] and 'raise'==lines[433].strip() and 'if args.owned_child is not None:' in lines[435]
assert 'realization.realize_screen_worlds' in lines[302]
ht=ast.parse(helper);assert all(isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef,ast.Assign,ast.Expr,ast.If)) for n in ht.body)
assert all(not n.decorator_list for n in ht.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
assert all(n.module in ('__future__','datetime','pathlib') for n in ht.body if isinstance(n,ast.ImportFrom))
assert all(all(not a.name.startswith('llm_intervention') for a in n.names) for n in ht.body if isinstance(n,ast.Import))
assert "m=types.ModuleType('_scientific_supervisor_base')" in source.decode()
guard=ht.body[-1]
assert isinstance(guard,ast.If) and ast.unparse(guard.test)=="__name__ == '__main__'" and len(guard.body)==1 and ast.unparse(guard.body[0])=='main()'
# Stable inventories again; parent may create later acceptance elsewhere, never inside raw custody.
assert tree(L,('custody-inventory.json',))==lt and tree(X,('custody-inventory.json',))==xt
for path,info in inputs.items():assert sha(Path(path).read_bytes())==info['sha256']
(E/'input-pins.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
(E/'observed-accounting.json').write_text(json.dumps(dict(related_accounting=rows,journal_pins=journals),indent=2,sort_keys=True)+'\n')
summary=dict(schema='ace-independent-realization-failure-audit-v1',status='failure_custody_and_pre_draw_boundary_verified',required_findings=0,job_id=JOB,packet_commit=COMMIT,source_commit=packet['source_commit'],outer_packet_sha256=OUTER,inner_packet_sha256=INNER,entry_sha256=ENTRY_SHA,base_helper_sha256=HELPER_SHA,stderr_sha256=sha(err),namespace_claim_sha256=CLAIM,raw_files=41,raw_bytes=1940531,capsule_files_including_inventory=len(lt['files'])+1,capsule_bytes_including_inventory=sum(d['bytes'] for d in lt['files'].values())+len(read(L/'custody-inventory.json')),two_complete_copies_match=True,two_remote_inventory_file_maps_match=True,remote_inventory_utc=[first['utc'],last['utc']],scheduler_state='FAILED',scheduler_exit_code='1:0',elapsed_seconds=4,allocated_cpus=2,memory_mib=7680,gpus=0,wall_seconds=300,new_allocated_cpu_seconds=8,new_gpu_slice_seconds=0,terminal_jobs=39,submission_journals=80,closed_cpu_seconds=11048,closed_gpu_slice_seconds=1601,preparation_allowance_cpu_seconds=3600,cpu_admission_after_release=14648,outstanding_reservation=0,output_claim_reached=False,scientific_parent_reached=False,owned_worker_spawn_reached=False,runtime_stage_reached=False,realizer_call_reached=False,scientific_worlds_generated=0,environment_responses=0,realization_accepted=False,runtime_qualified_for_this_execution=False,owned_worker_cleanup_applicable=False,alias_target_established=False,registry_preserved=True,automatic_retry_authorized=False,remote_action_by_reviewer=False,project_code_executed_by_reviewer=False)
(E/'audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
