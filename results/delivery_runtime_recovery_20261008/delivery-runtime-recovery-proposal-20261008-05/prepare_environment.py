"""Single isolated CPU runtime preparation; captured source, no science."""
import argparse,base64,ctypes,hashlib,json,os,re,signal,subprocess,sys,time,resource
from datetime import datetime,timezone
from pathlib import Path
EXACT={'torch':'2.9.1','numpy':'2.2.6','scipy':'1.15.3','pandas':'2.3.3','sympy':'1.14.0','PyYAML':'6.0.3'}
def require(ok,msg):
    if not ok:raise ValueError(msg)

def digest(raw):return hashlib.sha256(raw).hexdigest()

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
    require(fields.get('TimeLimit')==reg['slurm_wall_limit'],'allocated wall limit differs')
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



def write_fd(fd,name,value):
    require(type(fd) is int and Path(name).name==name,'explicit directory descriptor/basename required')
    handle=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
    with os.fdopen(handle,'w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def write(path,value):
    path=Path(path);fd=globals().get('__output_dir_fd__');logical=globals().get('__logical_output__')
    require(type(fd) is int and type(logical) is str and logical and path.parent==Path(logical),'output descriptor binding required')
    write_fd(fd,path.name,value)
def read_fd(fd,name):
    require(type(fd) is int and Path(name).name==name,'explicit directory descriptor/basename required')
    handle=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    with os.fdopen(handle,'rb') as stream:return stream.read()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def base_identity(reg):
    requested=Path(reg['base_python']);expected=Path(reg['base_python_resolved'])
    actual=requested.resolve(strict=True);expected_actual=expected.resolve(strict=True)
    actual_sha=sha(actual);running_sha=sha('/proc/self/exe')
    return {'node':os.uname().nodename,'requested_path':str(requested),'frozen_resolved_path':str(expected),'node_resolved_requested':str(actual),'node_resolved_expected':str(expected_actual),'actual_sha256':actual_sha,'frozen_sha256':reg['base_python_sha256'],'running_elf_sha256':running_sha,'python_version':list(sys.version_info[:2]),'path_matches':actual==expected_actual,'byte_match':actual_sha==reg['base_python_sha256'],'running_byte_match':running_sha==reg['base_python_sha256']}
def validate_base_identity(record):
    require(record['path_matches'] and record['byte_match'] and record['running_byte_match'] and record['python_version']==[3,11],'base interpreter differs: '+json.dumps(record))
def approved(reg):
    require(reg['exact_dependencies']==EXACT,'six authorized dependency pins required')
    require(reg['account']=='ucb736_asc1' and reg['partition']=='acpu' and reg['qos']=='cpu-normal','authorized account/queue required')
    require(type(reg['cpus']) is int and reg['cpus']==1 and type(reg['rss_bytes']) is int and reg['rss_bytes']==3*2**30,'authorized CPU/memory cap required')
    require(type(reg['wall_seconds']) is int and reg['wall_seconds']==1800 and reg['slurm_wall_limit']=='00:30:00','authorized wall cap required')
    require(reg['environment']=='/scratch/alpine/paco0228/ACE/envs/delivery_replay_py311_20261008_02' and reg['output']=='/scratch/alpine/paco0228/ACE/results/delivery_runtime_preparation_20261008_02','approved new paths required')
def terminate_group(proc):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        try:os.killpg(proc.pid,sig)
        except ProcessLookupError:pass
        if sig==signal.SIGTERM:
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
    proc.wait(timeout=5)
    deadline=time.monotonic()+5
    while True:
        try:
            pid,_=os.waitpid(-proc.pid,os.WNOHANG)
            if pid:continue
            if time.monotonic()>deadline:raise RuntimeError('child-group reap deadline')
            time.sleep(.05)
        except ChildProcessError:break

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);start=time.monotonic();job=os.environ.get('SLURM_JOB_ID','');log=None;trusted_output=False
    result={'status':'started','job_id':job,'freeze_sha256':args.freeze_sha256,'at':datetime.now(timezone.utc).isoformat(),'new_fits':0,'new_responses':0,'commands':[]}
    try:
        # Output is separately claimed/staged before launch, with frozen inode/uid.
        require(type(globals().get('__logical_output__')) is str and str(out)==__logical_output__,'explicit logical output binding required')
        fd=globals().get('__output_dir_fd__');require(type(fd) is int,'verified output directory descriptor required')
        info=os.fstat(fd);require(info.st_ino==__output_inode__ and info.st_uid==os.geteuid() and info.st_mode&0o077==0,'output descriptor identity differs')
        claim=read_fd(fd,'output_claim.json');require(digest(claim)==__output_claim_sha256__,'output claim authentication failed')
        trusted_output=True
        write(out/'preparation_started.json',result)
        require(Path(args.freeze)==out/'freeze.json','explicit frozen basename required')
        raw=read_fd(fd,'freeze.json');require(digest(raw)==args.freeze_sha256,'freeze digest differs');reg=json.loads(raw);approved(reg)
        require(str(out)==reg['output'] and info.st_ino==reg['output_inode'],'output identity differs')
        require(globals().get('__executed_source_sha256__')==reg['worker_sha256'],'captured executed worker digest differs')
        require(not sys.flags.optimize,'optimized preparation forbidden')
        result['executed_source_sha256']=__executed_source_sha256__
        result['allocation']=allocation(job,reg)
        result['base_identity']=base_identity(reg)
        validate_base_identity(result['base_identity'])
        require(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,'Linux child-subreaper required')
        prefix=Path(reg['environment']);prefix.mkdir(mode=0o700,exist_ok=False)
        prefix_fd=os.open(prefix,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:write_fd(prefix_fd,'.preparation-owner.json',{'job_id':job,'freeze_sha256':args.freeze_sha256,'inode':os.fstat(prefix_fd).st_ino})
        finally:os.close(prefix_fd)
        env=os.environ.copy()
        for n in list(env):
            if n.startswith(('PIP_','PYTHON')):env.pop(n,None)
        env.update({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PIP_CONFIG_FILE':'/dev/null','PIP_DISABLE_PIP_VERSION_CHECK':'1','PIP_NO_INPUT':'1','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':'/proc/self/fd/'+str(fd)+'/tmp'})
        os.mkdir('tmp',mode=0o700,dir_fd=fd)
        log=os.fdopen(os.open('preparation.log',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd),'w')
        def run(command):
            remain=reg['wall_seconds']-30-(time.monotonic()-start);require(remain>0,'preparation deadline reached')
            item={'at':datetime.now(timezone.utc).isoformat(),'command':command};result['commands'].append(item);log.write(json.dumps(item)+'\n');log.flush()
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd='/proc/self/fd/'+str(fd),start_new_session=True,pass_fds=(fd,))
            try:rc=proc.wait(timeout=remain)
            except BaseException:
                try:terminate_group(proc)
                except BaseException as cleanup:item['cleanup_failure']=str(cleanup)
                raise
            item.update(exit_code=rc,finished_at=datetime.now(timezone.utc).isoformat())
            try:terminate_group(proc)
            except BaseException as cleanup:
                item['cleanup_failure']=str(cleanup)
                if rc==0:raise
            require(rc==0,'preparation command failed: '+json.dumps(command))
        run([reg['base_python'],'-I','-B','-m','venv','--without-pip','--copies',str(prefix)])
        python=str(prefix/'bin/python')
        os.mkdir('seed',mode=0o700,dir_fd=fd);seed=[]
        for wheel in reg['bundled_seed_wheels']:
            raw=Path(wheel['path']).read_bytes();require(digest(raw)==wheel['sha256'],'bundled seed wheel changed')
            name=Path(wheel['path']).name;dest='/proc/self/fd/'+str(fd)+'/seed/'+name
            with open(dest,'xb') as target:target.write(raw)
            seed.append(dest)
        pip_wheels=[w for w in seed if Path(w).name.startswith('pip-')];require(len(pip_wheels)==1,'single bundled pip wheel required')
        seed_bootstrap="import sys,runpy; wheel,*packages=sys.argv[1:];sys.path.insert(0,wheel);sys.argv=['pip','install','--no-index','--no-deps','--no-compile',*packages];runpy.run_module('pip',run_name='__main__')"
        run([python,'-I','-B','-c',seed_bootstrap,pip_wheels[0],*seed])
        run([python,'-I','-B','-m','pip','install','--no-input','--no-cache-dir','--only-binary=:all:','--index-url','https://pypi.org/simple','--report','/proc/self/fd/'+str(fd)+'/install_report.json',*[n+'=='+v for n,v in EXACT.items()]])
        run([python,'-I','-B','-m','pip','check'])
        qpath=out/'qualify_runtime.py';qraw=Path('/proc/self/fd/'+str(fd)+'/qualify_runtime.py').read_bytes();require(digest(qraw)==reg['qualification_worker_sha256'],'qualification source differs')
        bootstrap="import sys,base64,hashlib; source,path,*args=sys.argv[1:];raw=base64.b64decode(source);sys.argv=[path,*args];context=vars(sys.modules['__main__']);context.update({'__name__':'__main__','__file__':path,'__executed_source_sha256__':hashlib.sha256(raw).hexdigest(),'__output_dir_fd__':int(__import__('os').environ['ACE_PREPARATION_OUTPUT_FD'])});exec(compile(raw,path,'exec'),context)"
        env['ACE_PREPARATION_OUTPUT_FD']=str(fd)
        run([python,'-I','-B','-c',bootstrap,base64.b64encode(qraw).decode(),str(qpath),'--freeze','/proc/self/fd/'+str(fd)+'/freeze.json','--freeze-sha256',args.freeze_sha256])
        evidence=json.loads(Path('/proc/self/fd/'+str(fd)+'/runtime_qualification.json').read_text());require(evidence['qualified'] is True,'runtime not qualified')
        result.update(status='complete',exit_code=0,runtime_qualification_sha256=sha('/proc/self/fd/'+str(fd)+'/runtime_qualification.json'),install_report_sha256=sha('/proc/self/fd/'+str(fd)+'/install_report.json'))
    except BaseException as e:result.update(status='failed',exit_code=1,failure_type=type(e).__name__,failure_reason=str(e))
    finally:
        if log is not None:
            try:log.close()
            except BaseException as e:result.update(status='failed',exit_code=1,log_close_failure=str(e))
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,child_process_cpu_seconds=usage.ru_utime+usage.ru_stime,peak_reaped_child_rss_KiB=usage.ru_maxrss)
        if trusted_output:write(out/'preparation_execution.json',result)
        else:print(json.dumps(result),file=sys.stderr)
    return result['exit_code']
if __name__=='__main__':sys.exit(main())
