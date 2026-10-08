"""Hash-authenticated, exclusive, inference-only launch; no fitting or responses."""
import argparse,base64,hashlib,json,os,re,subprocess,sys,time,types
from datetime import datetime,timezone
from pathlib import Path

def utc():return datetime.now(timezone.utc).isoformat()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path,expected):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular input required: '+str(p))
    raw=p.read_bytes();require(digest(raw)==expected,'digest mismatch: '+str(p));return raw
def mem_bytes(v):
    m=re.fullmatch(r'([0-9]+)([MG])',v);require(m is not None,'unsupported scheduler memory format')
    return int(m[1])*2**(20 if m[2]=='M' else 30)
def allocation(job,reg):
    require(job.isdigit(),'numeric Slurm job ID required')
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,timeout=10)
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S+)',raw))
    require(fields.get('JobId')==job,'scheduler identity mismatch')
    for key,want in [('Account',reg['account']),('Partition',reg['partition']),('QOS',reg['qos']),('NumNodes','1'),('NumCPUs','1'),('NumTasks','1'),('CPUs/Task','1')]:
        require(fields.get(key)==want,'allocated '+key+' differs')
    require(fields.get('TimeLimit')=='00:15:00','allocated wall limit differs')
    require(mem_bytes(fields.get('MinMemoryNode',''))==reg['rss_bytes'],'allocated node memory differs')
    tres_raw=fields.get('AllocTRES','')
    require(bool(tres_raw) and tres_raw not in ('(null)','N/A','Unknown'),'complete allocated-resource evidence required')
    tres={}
    for component in tres_raw.split(','):
        parts=component.split('=')
        require(len(parts)==2 and bool(parts[0]) and bool(parts[1]) and parts[0] not in tres,'ambiguous allocated-resource evidence')
        tres[parts[0]]=parts[1]
    require(tres.get('cpu')=='1' and tres.get('node')=='1','allocated TRES CPU/node differs')
    require(mem_bytes(tres.get('mem',''))==reg['rss_bytes'],'allocated TRES memory differs')
    require(not any(k.lower().startswith('gres') for k in tres),'generic accelerator allocation forbidden')
    require('gpu' not in ' '.join(fields.get(k,'').lower() for k in ('TresPerNode','TresPerTask','Gres')),'GPU allocation forbidden')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==1 and int(os.environ.get('SLURM_NTASKS','0'))==1 and int(os.environ.get('SLURM_JOB_NUM_NODES','0'))==1,'task/node environment differs')
    require(fields.get('JobState')=='RUNNING','allocation not running')
    return {'raw_scontrol':raw,'raw_scontrol_sha256':digest(raw.encode()),'verified_fields':{k:fields.get(k) for k in ('JobId','Account','Partition','QOS','NumNodes','NumCPUs','NumTasks','CPUs/Task','TimeLimit','MinMemoryNode','AllocTRES','TresPerNode','TresPerTask','Gres')}}

CHILD_BOOTSTRAP="""import base64,sys,types
from pathlib import Path
root,pin,receipt,cli64,verifier64=sys.argv[1:]
verifier_path=str(Path(root)/'verify_delivery_release.py')
v=types.ModuleType('verify_delivery_release');v.__file__=verifier_path;sys.modules[v.__name__]=v
exec(compile(base64.b64decode(verifier64),verifier_path,'exec',dont_inherit=True),v.__dict__)
cli_path=str(Path(root)/'replay_delivery_prospective_release.py')
sys.argv=[cli_path,'--root',root,'--expected-manifest-sha256',pin,'--receipt',receipt]
exec(compile(base64.b64decode(cli64),cli_path,'exec',dont_inherit=True),{'__name__':'__main__','__file__':cli_path})
"""

def validate_receipt(raw,reg):
    d=json.loads(raw)
    require(d.get('full_supplemental_replay') is True,'full replay not qualified')
    for k,w in [('checkpoints_replayed',640),('primary_cells_recomputed',240),('cached_responses_checked',48000),('new_optimizer_updates',0),('new_responses',0)]:
        require(type(d.get(k)) is int and d[k]==w,'invalid receipt '+k)
    require(d.get('runtime')==reg['exact_dependencies'],'actual runtime differs')
    require(d.get('integrity',{}).get('manifest_sha256')==reg['manifest_sha256'],'receipt manifest differs')
    require(d['integrity'].get('files_verified')==3890 and d['integrity'].get('bindings_verified')==5065,'receipt inventory differs')
    return digest(raw)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    require(sys.flags.optimize==0,'optimized Python forbidden')
    raw=read(a.freeze,a.freeze_sha256);reg=json.loads(raw);root=Path(reg['output']);started=time.monotonic()
    require(reg['account']=='ucb736_asc1' and reg['cpus']==1 and reg['rss_bytes']==3*2**30 and reg['wall_seconds']==900,'resource freeze differs')
    require(reg['new_fits']==reg['new_responses']==0,'inference-only freeze required')
    job=os.environ.get('SLURM_JOB_ID','');require(job.isdigit(),'Slurm allocation required')
    for name in ('replay_started.json','replay_execution.json','replay_receipt.json','replay.log'):
        require(not (root/name).exists(),'existing attempt artifact: '+name)
    with (root/'replay_started.json').open('x') as f:json.dump({'freeze_sha256':a.freeze_sha256,'job_id':job,'at':utc()},f)
    result={'status':'wrapper_failure','exit_code':None,'job_id':job,'freeze_sha256':a.freeze_sha256,'manifest_sha256':reg['manifest_sha256'],'requested_account':reg['account'],'stage':'allocation_preflight'}
    try:
        require(digest(Path(__file__).read_bytes())==reg['supervision_worker_sha256'],'worker bytes differ from authenticated launch')
        result['verified_allocation']=allocation(job,reg)
        result['account']=result['verified_allocation']['verified_fields']['Account']
        result['stage']='executable_authentication'
        package=Path(reg['package']);manifest=json.loads(read(package/'manifest.json',reg['manifest_sha256']))
        entries={f['path']:f for f in manifest['files']};require(len(entries)==3890,'unique manifest membership differs')
        cli=read(package/'replay_delivery_prospective_release.py',entries['replay_delivery_prospective_release.py']['sha256'])
        verifier=read(package/'verify_delivery_release.py',entries['verify_delivery_release.py']['sha256'])
        source=Path(reg['original_supervisor']);captured=read(source,reg['original_supervisor_sha256'])
        m=types.ModuleType('ace_authenticated_original_supervision');m.__file__=str(source);exec(compile(captured,str(source),'exec',dont_inherit=True),m.__dict__)
        result['authenticated_executables']={n:digest(b) for n,b in [('CLI',cli),('verifier',verifier),('supervisor',captured)]}
        result['stage']='child_supervision'
        command=[sys.executable,'-I','-B','-c',CHILD_BOOTSTRAP,str(package),reg['manifest_sha256'],str(root/'replay_receipt.json'),base64.b64encode(cli).decode(),base64.b64encode(verifier).decode()]
        child=m.supervise(command,started+850,reg['rss_bytes'],root/'replay.log',env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
        result['child_execution']=child
        require(child['status']=='complete' and child['exit_code']==0,'child replay failed')
        result['stage']='receipt_validation'
        require((root/'replay_receipt.json').is_file(),'child receipt missing')
        result['replay_receipt_sha256']=validate_receipt((root/'replay_receipt.json').read_bytes(),reg)
        result.update(status='complete',exit_code=0,stage='qualified')
    except BaseException as exc:
        result.update(status='wrapper_failure',exit_code=1,failure_type=type(exc).__name__,failure_reason=str(exc))
    result.update(finished_at=utc(),wrapper_elapsed_seconds=time.monotonic()-started,new_fits=0,new_responses=0)
    with (root/'replay_execution.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    if result['status']!='complete':raise SystemExit(1)

if __name__=='__main__':main()
