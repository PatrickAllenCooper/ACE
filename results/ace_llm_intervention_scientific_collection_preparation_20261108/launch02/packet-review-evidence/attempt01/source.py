"""Independent local byte/JSON/Git inspection only. No project imports/calls."""
from pathlib import Path
import ast, hashlib, json, shlex, stat, subprocess

R=Path('/Users/pat/code/ACE'); E=Path(__file__).parent
L=R/'results/ace_llm_intervention_scientific_collection_20261108/launch01'
X=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-20261108-01')
C='a9346022f145cc29439325b6f291597b2e6ec037'
ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_collection_20261108_01'
T=R/'results/ace_llm_intervention_scientific_realization_20261108/terminal02'
A=R/'results/ace_llm_intervention_scientific_realization_20261108/acceptance02'
def need(ok,label):
    if not ok: raise ValueError(label)
def sha(b):return hashlib.sha256(b).hexdigest()
def pairs(items):
    d={}
    for k,v in items:
        need(k not in d,'duplicate JSON key');d[k]=v
    return d
def read(p):
    need(stat.S_ISREG(p.lstat().st_mode) and p.stat().st_nlink==1,'regular independent file')
    return p.read_bytes()
def obj(p):return json.loads(read(p),object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def files(root):
    result={}
    for p in root.rglob('*'):
        need(not p.is_symlink(),'symlink')
        if p.is_file():result[p.relative_to(root).as_posix()]=sha(read(p))
    return result
def git(path):return subprocess.check_output(['git','--no-replace-objects','show',C+':'+path],cwd=R)
pins=files(L); need(pins==files(X) and len(pins)==80,'exact80 two copies')
outer=obj(L/'packet.json');stage=obj(L/'inputs/packet.json');inner=obj(L/'inputs/collection-packet.json')
need(pins['packet.json']=='aaf0f21110c0906ba0a6a0ba23faa9e0b5ffc6098f1802682901d47e28302961','outer pin')
need(pins['inputs/packet.json']=='fddda2e1fe9c2110ed0ca9c770d4db85605e527af504778dff5957a62c990cfc','stage pin')
need(pins['inputs/collection-packet.json']=='a86dfd705dba2ca213c3ac25884c3e3deca177e00d8ce8be72732a3baf878c35','inner pin')
need({k:v for k,v in pins.items() if k!='packet.json'}==outer['new_input_files'],'whole roster')
need(outer['source_commit']==stage['source_commit']==C and outer['remote_root']==ROOT and outer['local_root']==str(L) and outer['external_root']==str(X),'source roots')
need(subprocess.check_output(['git','--no-replace-objects','cat-file','-t',C],cwd=R).strip()==b'commit','commit type')
for n in ('build_packet.py','launch_once.py','monitor_once.py'):
    need(read(L/n)==read(E/'authoring'/n),'reviewed authoring')
fixed={'freeze.json':'15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d',
       'realization-acceptance.json':'5e28e1aa16512eca3230154526dab65df65330f6920fcc094548c320736a65c1',
       'baseline.json':'69cc97784637b0ec522da6e00e00a1f08bdfabe9712e2104a43010ee19f2035f',
       'predecessor-baseline.json':'63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727',
       'namespace-admission.json':'5027bca99d1c1cdfc9cf89dd5639593308e8eb37f7f7ad2fb8aacb6ded5c0e36',
       'namespace-claim.json':'40c0937c623d9473a413347125d3bc35648955132c4cf2394f055d4b228fed6a',
       'realization-disposition.json':'57346878495095977983da9685a3ce468478c5652b293d818d005e95acdbccf4',
       'realization-inventory.json':'88af54fed84094045d8701164e34a84801dc538a87df42e2da5d3103f716a0af',
       'realization-packet.json':'ae077ef888671701eaa29415248dc7442656eebe2e5693df14ea6a22d661d8d7',
       'realization-result.json':'c1b22b2883d9da5530650b2a1cf9a20ad6caef5d7305770f255cb1963d432e8b'}
need(all(pins[n]==h for n,h in fixed.items()),'independent metadata pins')
for n,p in [('realization-acceptance.json',A/'acceptance.json'),('baseline.json',A/'accounting.json'),('realization-disposition.json',T/'disposition.json'),('realization-inventory.json',T/'inventory_transport.json'),('namespace-claim.json',T/'namespaces/local.namespace.json')]:
    need(read(L/n)==read(p),'accepted original bytes')
freeze=obj(L/'freeze.json'); accepted=obj(L/'realization-acceptance.json');base=obj(L/'baseline.json');ns=obj(L/'namespace-admission.json');claim=obj(L/'namespace-claim.json')
need(accepted['status']=='accepted' and type(accepted['required_findings']) is int and accepted['required_findings']==0 and accepted['complete_custody_verified'] is True,'accepted realization')
need(accepted['inventory_transport_sha256']==pins['realization-inventory.json'] and accepted['disposition_sha256']==pins['realization-disposition.json'] and accepted['namespace_claim_sha256']==pins['namespace-claim.json'] and accepted['realization_packet_sha256']==pins['realization-packet.json'],'actual joins')
need(base['input_pins']['predecessor']==pins['predecessor-baseline.json'] and base['input_pins']['acceptance']==pins['realization-acceptance.json'] and base['input_pins']['disposition']==pins['realization-disposition.json'],'baseline joins')
need(base['terminal_jobs']==40 and base['submission_journals']==82 and len(base['related_accounting'])==40 and len(base['journal_pins'])==82 and base['outstanding_reservation']==0,'complete baseline')
need(sum(int(r[4])*int(r[5]) for r in base['related_accounting'])==11134,'actual CPU baseline')
resources=dict(account='ucb736_asc2',cpus=2,gpus=0,memory_mib=7680,partition='acpu',qos='cpu-normal',requeue=False,wall_seconds=900)
deadlines=dict(startup=120,work=850,cleanup=875,hard=890)
need(stage['resources']==outer['resources']==resources and stage['deadlines']==outer['deadlines']==deadlines,'bounded resources deadlines')
fstage=next(s for s in freeze['stages'] if s['stage']=='collect')
need(all(fstage[k]==v for k,v in resources.items()) and fstage['deadlines']==deadlines and fstage['remote_root']==ROOT,'frozen resources')
need(outer['admission']==dict(candidate_cpu_reservation=1800,candidate_gpu_reservation=0,ceiling_cpu_seconds=57600,ceiling_gpu_seconds=7200,closed_cpu_seconds=11134,closed_gpu_slice_seconds=1601,preparation_allowance_cpu_seconds=3600,prior_jobs=40,prior_journals=82,total_cpu_seconds=16534,total_gpu_seconds=1601),'actual admission')
need(outer['baseline_sha256']==pins['baseline.json'] and outer['realization_job_id']=='35406081','accepted baseline job')
need(all(outer[k] is False for k in ('submitted','new_namespace_claimed','realization_repeated')) and outer['development_payloads_staged']==0,'no execution flags')
need((outer['planned_records'],outer['planned_pairs'],outer['max_response_attempts'],outer['max_reserved_cost_units'])==(160,3584,9216,10240),'plan limits')
closure=freeze['stage_closure']['stages']['collect'];need(len(outer['source_pins'])==23 and len(freeze['source_pins'])==67,'source rosters')
for p,h in freeze['source_pins'].items():need(sha(git(p))==h,'full source commit')
for p,h in outer['source_pins'].items():need(sha(read(L/'inputs'/Path(p).name))==h and read(L/'inputs'/Path(p).name)==git(p),'source bytes')
need({Path(p).name:h for p,h in outer['source_pins'].items()}==closure['input_source_files'],'source map')
need(stage['source_names']==closure['source_names'],'worker source names')
input_names={n.removeprefix('inputs/') for n in pins if n.startswith('inputs/')}
need(input_names==set(stage['files'])|{'packet.json','admission.json'} and len(stage['files'])==60,'exact supervisor input names')
need(all(pins['inputs/'+n]==h for n,h in stage['files'].items()),'supervisor input pins')
need(stage['output']==ROOT+'/collection' and stage['mode']=='collect' and stage['schema']=='ace-scientific-cpu-supervisor-v1','outer worker contract')
proposal=obj(L/'inputs/proposal.json');protocol=proposal['source_pins']
need(len(protocol)==12 and protocol==stage['protocol_source_pins'] and all(freeze['source_pins'][p]==h for p,h in protocol.items()),'original12 unchanged')
need(digest(proposal)=='585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8','original proposal')
need(stage['runtime']==freeze['runtime'] and stage['runtime_source']==freeze['runtime_source'] and stage['archive']==dict(path=freeze['runtime_archive_path'],sha256=freeze['runtime_archive_sha256']) and stage['manifest']==dict(path=freeze['runtime_manifest_path'],sha256=freeze['runtime_manifest_sha256']),'exact runtime')
need(stage['freeze_sha256']==pins['freeze.json'] and stage['namespace_admission_sha256']==pins['namespace-admission.json'],'frozen namespace')
admission=obj(L/'inputs/admission.json')
need(admission==dict(schema=stage['schema']+'/admission',mode='collect',status='admitted',packet_sha256=pins['inputs/packet.json'],output=stage['output'],freeze_sha256=pins['freeze.json'],namespace_admission_sha256=pins['namespace-admission.json'],scientific_protocol_frozen=True,namespace_owned=True,source_runtime_scheduler_qualified=True,accounting_admitted=True,resources=resources,deadlines=deadlines),'outer exact admission')
ids=list(range(202611070000,202611070032));terminal=obj(L/'inputs/realization-terminal.json')
need(pins['inputs/realization-terminal.json']==accepted['realization_terminal_sha256']=='3b332e969422cf528d23b780721a6eff8bbb1bec171eab26346e7767de878e22','actual task terminal')
need(len(terminal['entries'])==56 and terminal['worlds_generated']==56 and terminal['status']=='complete' and terminal['freeze_sha256']==stage['freeze_sha256'] and terminal['namespace_admission_sha256']==stage['namespace_admission_sha256'],'complete realization metadata')
need(inner['schema']=='ace-scientific-training-collection-cpu-v1' and inner['scope']=='policy_train' and inner['world_roster']==ids and inner['roster_sha256']==digest(ids),'training roster')
need(set(inner['files'])=={'proposal.json','realization-terminal.json'}|{f'descriptor-{i:03d}.json' for i in range(32)},'no development payload')
need(all(stage['files'][n]==h for n,h in inner['files'].items()),'inner input pins')
need(inner['source_pins']=={p:h for p,h in outer['source_pins'].items() if Path(p).name in closure['work_sources']},'inner source closure')
need(inner['proposal_sha256']==digest(proposal) and inner['freeze_sha256']==stage['freeze_sha256'] and inner['namespace_admission_sha256']==stage['namespace_admission_sha256'],'inner identity')
inventory=json.loads(obj(L/'realization-inventory.json')['stdout'])['files'];configs={};records=paircount=0
for i,row in enumerate(proposal['training_worlds']):
    need(row['world_id']==ids[i] and row['scope']=='policy_train','frozen training identity')
    name=f'descriptor-{i:03d}.json';entry=terminal['entries'][i];b=read(L/'inputs'/name)
    need(sha(b)==entry['file_sha256']==inventory[f'launch/realization/session/worker/task/world-{i:03d}.json']['sha256'],'original descriptor bytes')
    need(b==read(T/f'remote/launch/realization/session/worker/task/world-{i:03d}.json'),'accepted training descriptor')
    value=json.loads(b);need(value['scope']=='policy_train' and value['world_id']==ids[i] and digest(value)==inner['world_pins'][str(ids[i])]==entry['descriptor_sha256'],'descriptor semantic')
    # Independently project the reviewed typed assembly_recipe and _plain; no calls.
    public=row['public'];training=row['training']
    ap={k:public[k] for k in ('graph','menu','task','initial_actions','objective','domain_notes')}
    ap.update(scope='policy_train',expected_task_sha256=public['task_sha256'],total_budget=32)
    ap.update({k:training[k] for k in ('behavior_method','replicate_ids','planning_draws','tie_multiplier','minimum_gain_gap','structural_scores','request_namespace')})
    ap['snapshots']=[{k:s[k] for k in ('acquisition_cost_units','teacher_method','candidates')} for s in training['snapshots']]
    config=dict(execution_authorized=False,runtime_bindings_pending=['world','expected_world_sha256','learner','output'],learner_parameters=dict(graph=public['graph'],noise_variances=public['learner']['noise_variances'],prior_variance=1.0),assembly_parameters=ap)
    configs[str(ids[i])]=digest(config)
    records+=len(ap['snapshots']);paircount+=sum(len(s['candidates'])*(len(s['candidates'])-1)//2 for s in ap['snapshots'])
need(configs==inner['configuration_pins'] and records==160 and paircount==3584,'all numerical recipe pins')
inner_a=obj(L/'inputs/collection-admission.json')
need(inner_a==dict(schema=inner['schema']+'/admission',stage='policy_train_collection',status='admitted',packet_sha256=pins['inputs/collection-packet.json'],output=ROOT+'/collection/session/worker/task',freeze_sha256=stage['freeze_sha256'],namespace_admission_sha256=stage['namespace_admission_sha256'],realization_terminal_sha256=pins['inputs/realization-terminal.json'],source_runtime_scheduler_qualified=True,namespace_owned=True,accounting_admitted=True,max_response_attempts=9216,max_reserved_cost_units=10240,gpus=0),'inner exact charging admission')
need(pins['inputs/realization-acceptance.json']==pins['realization-acceptance.json'],'outer acceptance input')
batch=read(L/'job.sbatch').decode();directives=[l for l in batch.splitlines() if l.startswith('#SBATCH')]
need(directives==['#SBATCH --job-name=ace_llm_scientific_collect01','#SBATCH --account=ucb736_asc2','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','#SBATCH --nodes=1','#SBATCH --ntasks=1','#SBATCH --cpus-per-task=2','#SBATCH --mem=7680M','#SBATCH --time=00:15:00','#SBATCH --no-requeue','#SBATCH --output='+ROOT+'/logs/%j.out','#SBATCH --error='+ROOT+'/logs/%j.err'],'batch profile')
diag=ast.parse(read(L/'inputs/qualify_llm_intervention_runtime_metadata.py'))
bootstrap=next(ast.literal_eval(n.value) for n in diag.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))
command=shlex.split(batch.split('\nexec ',1)[1])
expected=[stage['runtime_source']+'/bin/python','-I','-B','-c',bootstrap,ROOT+'/inputs/supervise_llm_intervention_scientific_cpu.py',stage['files']['supervise_llm_intervention_scientific_cpu.py'],'--packet',ROOT+'/inputs/packet.json','--packet-sha256',pins['inputs/packet.json'],'--source-commit',C,'--admission',ROOT+'/inputs/admission.json','--admission-sha256',pins['inputs/admission.json'],'--output',ROOT+'/collection']
need(command==expected,'bootstrap exact argv')
build=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-preparation-20261108-02/build01')
receipt=obj(build/'receipt.json');need(receipt['returncode']==0,'successful retained build')
for n in ('build_packet.py','launch_once.py','monitor_once.py'):need(read(build/n)==read(L/n),'build source captures')
need(files(L)==files(X)==pins,'final packet readback')
result=dict(scope='authoring and exact unsubmitted collection01 packet only',required_findings=0,packet_sha256=pins['packet.json'],stage_packet_sha256=pins['inputs/packet.json'],inner_packet_sha256=pins['inputs/collection-packet.json'],source_commit=C,files=len(pins),bytes=sum((L/n).stat().st_size for n in pins),source_files=23,supervisor_files=60,private_training_descriptors=32,development_payloads=0,configuration_pins_verified=32,records=records,pairs=paircount,max_response_attempts=9216,max_reserved_cost_units=10240,admission=outer['admission'],build_receipt=receipt,scientific_execution=False,submission=False)
with (E/'packet-pins.json').open('x') as f:json.dump(pins,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(result,indent=2,sort_keys=True))
