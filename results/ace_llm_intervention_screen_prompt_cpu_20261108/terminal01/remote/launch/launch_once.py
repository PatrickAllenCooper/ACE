#!/usr/bin/env python3
"""PREPARED ONLY. Future once-only CPU capacity launcher; never retry ambiguity.

Requires a built/committed packet and independently pinned, committed concrete
packet review naming its full SHA256. Both existing controller locks cover live
SSH/readiness, exact 36-job/74-journal reconciliation, transfer and one sbatch.
Submission is not qualification. Parent owns early/periodic monitoring, complete
terminal custody and independent result review. No automatic retries or repairs.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shlex, subprocess

REPO = Path('/Users/pat/code/ACE')
LOCAL = REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/launch01'
EXT = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-prompt-cpu-20261108-01')
ROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_screen_prompt_cpu_20261108_01'
FINALROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_finalization_20261108_01'
CPUROOT = '/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_campaign_artificial_20261108_03'
ENTRY = 'qualify_llm_intervention_screen_prompt_cpu.py'
FINALIZER = 'finalize_llm_intervention_campaign_cpu.py'
HELPER = '/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'
SOCKET = '/tmp/defab-curc.sock'
RESOURCES = dict(account='ucb736_asc2', partition='acpu', qos='cpu-normal', cpus=2,
                 memory_mib=7680, wall_seconds=300, gpus=0, requeue=False)
ADMISSION = dict(closed_cpu_seconds=10368, closed_gpu_slice_seconds=1483,
    cumulative_preparation_cpu_allowance=3600, new_cpu_reservation=600, new_gpu_slice_reservation=0,
    cpu_admission=14568, gpu_slice_admission=1483, cpu_ceiling=57600, gpu_slice_ceiling=7200)
MARKERS = ('controller.json', 'submission.intent.json', 'submission.result.json',
           'launcher_failure.json', 'ssh_master.json', 'readiness_transport.json')

def need(ok, message):
    if not ok: raise ValueError(message)
def sha(value): return hashlib.sha256(value).hexdigest()
def raw(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def text(value): return value.decode(errors='replace') if isinstance(value, bytes) else value
def save(name, value):
    for root in (EXT, LOCAL):
        with (root/name).open('xb') as handle: handle.write(raw(value)); handle.flush(); os.fsync(handle.fileno())
def unclaimed():
    for root in (EXT, LOCAL):
        need(root.is_dir() and not root.is_symlink(), 'built packet missing')
        need(not any((root/n).exists() or (root/n).is_symlink() for n in MARKERS), 'prior attempt; reconcile, never retry')

# Passed as captured text to the original runtime's isolated stdlib Python on
# the login node. No runtime staging, tokenization, model load or numerical work.
REMOTE_COMMON = r'''
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,types
root=Path(ROOT)
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(value):return hashlib.sha256(value).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(command):
 p=subprocess.run(command,capture_output=True,text=True,timeout=20)
 need(p.returncode==0,json.dumps(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
 return p.stdout
def write(name,value):
 with (root/name).open('x') as h:json.dump(value,h,sort_keys=True,indent=2,allow_nan=False);h.write(chr(10));h.flush();os.fsync(h.fileno())
need(sha(BASE_SOURCE.encode())==META['input_files'][FINALIZER],'captured supervisor helper differs')
outer=types.ModuleType('screen_readiness_bytes');outer.__file__=str(root/'inputs'/FINALIZER)
exec(compile(BASE_SOURCE,outer.__file__,'exec'),vars(outer))
need(sha(INVENTORY_SOURCE.encode())==outer.GPU_SHA,'accepted custody helper differs')
api=outer.module('screen_readiness_inventory',root/'custody_inventory.py',INVENTORY_SOURCE.encode())
def tres(value):
 d={}
 for row in value.split(',') if value else []:
  k,sep,v=row.partition('=');need(sep and k not in d and v,'malformed TRES');d[k]=v
 return d
def gpu_count(d,cpus,elapsed):
 g={k:v for k,v in d.items() if k.startswith('gres/')}
 need(all(k=='gres/gpu' or k.startswith('gres/gpu:') for k in g) and all(v.isdigit() for v in g.values()),'unknown GPU resource')
 typed=sum(int(v) for k,v in g.items() if k.startswith('gres/gpu:'))
 generic=int(g['gres/gpu']) if 'gres/gpu' in g else None
 need(generic is None or not any(k.startswith('gres/gpu:') for k in g) or generic==typed,'GPU count conflict')
 need(d or cpus==elapsed==0,'missing allocation inventory')
 return generic if generic is not None else typed
def queue():
 value=run(['squeue','-u','paco0228','-h','-o','%i|%j|%T|%R|%a'])
 need(not any('|ace_llm' in line for line in value.splitlines()),'related active job; do not submit')
 return value
def upstream():
 need(not Path(PACKET['output']).exists() and not Path(PACKET['output']).is_symlink(),'capacity output already claimed')
 for label,path in META['predecessor_roots'].items():
  expected={n[len(label)+1:]:v['sha256'] for n,v in DISPOSITION['files'].items() if n.startswith(label+'/')}
  need(expected and api.inventory(path)==expected,'closed predecessor custody changed: '+label)
def reconcile():
 started=time.monotonic();hostname=run(['hostname']).strip();q=queue()
 value=run(['sacct','-X','-u','paco0228','-S','2026-11-02','--format',
  'JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,Account,AllocTRES%180','-n','-P'])
 rows=[line.split('|')[:8] for line in value.splitlines() if '|ace_llm' in line]
 need(len(rows)==36 and all(len(r)==8 and r[0].isdigit() and r[4].isdigit() and r[5].isdigit()
  and r[2].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows),'terminal census changed')
 rows=sorted(rows,key=lambda r:int(r[0]))
 need(len({r[0] for r in rows})==36 and rows==META['prior_accounting'],'prior scheduler records changed')
 cpu=sum(int(r[4])*int(r[5]) for r in rows)
 gpu=sum(gpu_count(tres(r[7]),int(r[4]),int(r[5]))*int(r[5]) for r in rows)
 need(cpu==10368==DISPOSITION['closed_cpu_seconds'] and gpu==1483==DISPOSITION['closed_gpu_slice_seconds'],'closed resources differ')
 a=META['admission']
 need(a==ADMISSION and cpu+3600+600==a['cpu_admission']==14568<=57600
  and gpu==a['gpu_slice_admission']==1483<=7200,'admission differs')
 journals={str(p):sha(outer.read(p)) for d in root.parent.glob('ace_llm_intervention*') for p in d.glob('submission.*.json')}
 need(len(journals)==74 and journals==META['prior_journals'],'74-journal census changed')
 details=run(['sacct','-X','-j','35389391','-n','-P','--format=Partition,QOS,TimelimitRaw,ReqTRES%180,User'])
 dr=[line.split('|')[:5] for line in details.splitlines() if line.strip()]
 need(len(dr)==1 and dr[0][:3]==['acpu','cpu-normal','5'] and dr[0][4]=='paco0228','final01 scheduler identity differs')
 request=tres(dr[0][3]);need(request.get('cpu')=='2' and request.get('mem') in ('7680M','7.5G','7.50G')
  and gpu_count(request,2,61)==0,'final01 request differs')
 partition=run(['scontrol','show','partition','acpu','-o'])
 limits=dict(part.split('=',1) for part in partition.split() if '=' in part);cap=limits.get('MaxMemPerCPU','')
 need(limits.get('PartitionName')=='acpu' and limits.get('State')=='UP' and cap.isdigit() and int(cap)>=3840,'CPU partition capacity changed')
 for key,want in [('AllowAccounts','ucb736_asc2'),('AllowQos','cpu-normal')]:
  allow=limits.get(key,'ALL');need(allow=='ALL' or want in allow.split(','),'partition excludes '+want)
 need('ucb736_asc2' not in limits.get('DenyAccounts','').split(',') and 'cpu-normal' not in limits.get('DenyQos','').split(','),'partition denies request')
 assoc=run(['sacctmgr','-n','-P','show','assoc','where','account=ucb736_asc2','user=paco0228','format=Cluster,Account,User,Partition,MaxJobs,QOS'])
 ar=[line.split('|') for line in assoc.splitlines() if line.strip()]
 need(any(len(r)>=6 and r[:3]==['alpine','ucb736_asc2','paco0228'] and r[3] in ('','acpu') and r[4]!='0'
  and 'cpu-normal' in r[5].split(',') for r in ar),'account/QOS unavailable')
 upstream()
 outer.read(PACKET['manifest']['path'],PACKET['manifest']['sha256'])
 archive=Path(PACKET['archive']['path'])
 need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==8438650880,'runtime archive metadata differs')
 model=Path(PACKET['model_directory'])
 need(model.is_dir() and not model.is_symlink() and {p.name for p in model.iterdir()}==set(PACKET['model_files']),'model directory differs')
 for name,pin in PACKET['model_files'].items():
  if name!='model.safetensors':outer.read(model/name,pin)
 weights=model/'model.safetensors'
 need(weights.is_file() and not weights.is_symlink() and weights.stat().st_size>0,'opaque model file absent')
 need(root.parent.is_dir() and not root.parent.is_symlink() and root.parent.stat().st_uid==os.getuid(),'scratch owner differs')
 fs=os.statvfs(root.parent);need(fs.f_bavail*fs.f_frsize>2*1024**3,'insufficient output scratch')
 final_queue=queue();need(time.monotonic()-started<120,'readiness stale')
 return dict(utc=now(),hostname=hostname,queue=q,queue_before_mutation=final_queue,prior_accounting=rows,
  prior_journals=journals,closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,final01_details=details,
  partition=partition,association=assoc,scratch_free_bytes=fs.f_bavail*fs.f_frsize,
  bulk_archive_and_weight_byte_authentication='allocated CPU checks; no weight deserialization')
def verify_packet():
 for name,pin in PINS.items():outer.read(root/name,pin)
 need(api.inventory(root/'inputs')==META['input_files'],'exact 264 inputs plus packet differs')
 source=outer.read(root/'inputs'/ENTRY,META['worker_sha256'])
 worker=outer.module('screen_cpu_request_checks',root/'inputs'/ENTRY,source)
 args=types.SimpleNamespace(packet=str(root/'inputs/packet.json'),packet_sha256=META['screen_packet_sha256'],
  source_commit=META['source_commit'],output=PACKET['output'])
 context=worker.authenticate(args)
 need(context['manifest_pins']=={k:META[k] for k in ('fixture_manifest_file_sha256','fixture_manifest_canonical_sha256')},
  'byte/canonical manifest binding differs')
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('commit', 'review', 'review-sha256'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); unclaimed()
    revision = subprocess.check_output(['git', 'rev-parse', args.commit+'^{commit}'], cwd=REPO, text=True).strip()
    meta = json.loads((LOCAL/'packet-metadata.json').read_bytes())
    need(meta['remote_root'] == ROOT and meta['input_directory'] == ROOT+'/inputs' and meta['output'] == ROOT+'/cpu01'
        and meta['resources'] == RESOURCES and meta['admission'] == ADMISSION
        and meta['deadlines'] == dict(startup=120, work=260, cleanup=280, hard=290)
        and meta['expected_prior_jobs'] == 36 and meta['expected_prior_submission_journals'] == 74
        and meta['predecessor_roots'] == {'launch': FINALROOT, 'common': CPUROOT}, 'fixed packet identity/admission differs')
    names = ['inputs/'+n for n in meta['input_files']] + list(meta['root_files']) + ['packet-metadata.json']
    need(len(meta['input_files']) == 265 and len(names) == len(set(names))
        and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'bounded flat packet roster differs')
    pins = {}
    for name in names:
        value = (LOCAL/name).read_bytes()
        need(not (LOCAL/name).is_symlink() and not (EXT/name).is_symlink() and value == (EXT/name).read_bytes(), 'custody copies differ')
        need(subprocess.check_output(['git', 'show', revision+':'+str((LOCAL/name).relative_to(REPO))], cwd=REPO) == value,
             'packet commit does not bind '+name)
        pins[name] = sha(value)
    for name, pin in meta['input_files'].items(): need(pins['inputs/'+name] == pin, 'input pin differs')
    for name, pin in meta['root_files'].items(): need(pins[name] == pin, 'root artifact pin differs')
    for name, origin in meta['source_origins'].items():
        need(subprocess.check_output(['git', 'show', meta['source_commit']+':'+origin], cwd=REPO)
             == (LOCAL/'inputs'/name).read_bytes(), 'eventual source commit differs: '+name)
    need(Path(__file__).read_bytes() == (LOCAL/'launch_once.py').read_bytes(), 'running launcher differs from committed packet')
    review = Path(args.review).resolve(strict=True); review_raw = review.read_bytes()
    need(review != LOCAL/'source-review.txt' and review_raw.strip() and sha(review_raw) == args.review_sha256
        and meta['screen_packet_sha256'] in review_raw.decode(), 'independent concrete packet review naming full pin required')
    need(subprocess.check_output(['git', 'show', revision+':'+str(review.relative_to(REPO))], cwd=REPO) == review_raw,
         'concrete review not committed')
    packet = json.loads((LOCAL/'inputs/packet.json').read_bytes())
    need(pins['inputs/packet.json'] == meta['screen_packet_sha256'] and packet['source_commit'] == meta['source_commit']
        and packet['output'] == ROOT+'/cpu01' and packet['resources'] == RESOURCES
        and len(packet['files']) == 264 and packet['files'] == {n:p for n,p in meta['input_files'].items() if n != 'packet.json'}
        and meta['worker_sha256'] == packet['files'][ENTRY]
        and packet['fixture_manifest_sha256'] == meta['fixture_manifest_file_sha256']
        and packet['runtime_source'] == '/projects/paco0228/miniconda3/envs/ace', 'worker packet/source binding differs')
    disposition = json.loads((LOCAL/'final-disposition.json').read_bytes())
    intent = dict(utc=utc(), packet_commit=revision, source_commit=meta['source_commit'], pins=pins,
        screen_packet_sha256=meta['screen_packet_sha256'], worker_sha256=meta['worker_sha256'],
        fixture_manifest_file_sha256=meta['fixture_manifest_file_sha256'],
        fixture_manifest_canonical_sha256=meta['fixture_manifest_canonical_sha256'],
        review_path=str(review.relative_to(REPO)), review_sha256=sha(review_raw), review_text=review_raw.decode(),
        launcher_sha256=sha(Path(__file__).read_bytes()), remote_root=ROOT, output=packet['output'],
        controller_pid_local=os.getpid(), resources=RESOURCES, admission=ADMISSION, blind_retry_forbidden=True,
        scientific_qualified=False)
    values = dict(ROOT=ROOT, ENTRY=ENTRY, FINALIZER=FINALIZER, META=meta, PACKET=packet, PINS=pins,
        ADMISSION=ADMISSION, DISPOSITION=disposition,
        BASE_SOURCE=(LOCAL/'inputs'/FINALIZER).read_text(), INVENTORY_SOURCE=(LOCAL/'custody_inventory.py').read_text())
    common = '\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+REMOTE_COMMON
    command = shlex.join([packet['runtime_source']+'/bin/python', '-I', '-B', '-'])
    transport_env = dict(os.environ, CURC_SOCKET=SOCKET)
    def remote(name, source):
        try:
            result = subprocess.run(['bash', HELPER, 'run', command], env=transport_env,
                input=source, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            save(name, dict(python_source=source, ambiguous=True, utc=utc(), stdout=text(exc.stdout), stderr=text(exc.stderr)))
            raise
        save(name, dict(python_source=source, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, utc=utc()))
        need(result.returncode == 0, name+' failed; retain/reconcile, never retry')
        return json.loads(result.stdout)
    locks, claimed = [], False
    try:
        for name in ('ace-delivery-controller.lock', 'ace-foundation-controller.lock'):
            handle = open(EXT.parent/name, 'r+')
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException: handle.close(); raise
            locks.append(handle)
        unclaimed(); claimed = True
        save('controller.json', dict(held=[h.name for h in locks], pid=os.getpid(), utc=utc()))
        control = subprocess.run(['bash', HELPER, 'check'], env=transport_env, capture_output=True, text=True, timeout=20)
        save('ssh_master.json', dict(returncode=control.returncode, stdout=control.stdout, stderr=control.stderr, utc=utc()))
        need(control.returncode == 0, 'shared SSH unavailable; no reconnect or retry')
        ready = remote('readiness_transport.json', common+'''
need(not root.exists() and not root.is_symlink(),'launch root already exists; no retry')
r=reconcile()
root.mkdir(mode=0o700);(root/'logs').mkdir();(root/'inputs').mkdir()
p=root/'write-read-probe'
with p.open('xb') as h:h.write(b'ACE screen prompt cpu01 readiness');h.flush();os.fsync(h.fileno())
need(p.read_bytes()==b'ACE screen prompt cpu01 readiness','scratch readback failed');p.unlink()
r['scratch_write_read_cleanup']=True;write('readiness.json',r);print(json.dumps(r))
''')
        save('readiness.json', ready)
        for group in ('inputs', 'root'):
            selected = [n for n in names if n.startswith('inputs/') == (group == 'inputs')]
            target = ROOT+('/inputs/' if group == 'inputs' else '/')
            result = subprocess.run(['scp', '-o', 'ControlPath='+SOCKET, '-o', 'ControlMaster=no', '-o', 'BatchMode=yes',
                '-o', 'PasswordAuthentication=no', '-o', 'KbdInteractiveAuthentication=no', '-o', 'PubkeyAuthentication=no',
                *[str(LOCAL/n) for n in selected], 'paco0228@login.rc.colorado.edu:'+target],
                capture_output=True, text=True, timeout=180)
            save('transfer-'+group+'.json', dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, utc=utc()))
            need(result.returncode == 0, 'partial transfer retained; no retry')
        verified = remote('verification_transport.json', common+'\nverify_packet()\nprint(json.dumps(dict(utc=now(),verified=PINS)))\n')
        save('remote_verified.json', verified); save('submission.intent.json', intent)
        result = remote('submission_transport.json', common+'\nintent='+repr(intent)+'''
need(not (root/'submission.intent.json').exists() and not (root/'submission.result.json').exists(),'existing submission journal')
verify_packet();intent.update(reconcile())
for name,pin in PINS.items():outer.read(root/name,pin)
queue();intent['remote_utc']=now();write('submission.intent.json',intent)
r=dict(utc=now(),intent_sha256=sha((root/'submission.intent.json').read_bytes()),job_id=None,
       capacity_qualified=False,scientific_qualified=False)
try:
 env={k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
 p=subprocess.run(['sbatch','--parsable',str(root/'job.sbatch')],env=env,capture_output=True,text=True,timeout=25)
 job=p.stdout.strip().split(';')[0]
 r.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,job_id=job if p.returncode==0 and job.isdigit() else None)
except BaseException as exc:r.update(ambiguous=True,exception_type=type(exc).__name__,message=str(exc))
write('submission.result.json',r);print(json.dumps(r))
''')
        save('submission.result.json', result)
        need(result['job_id'] is not None, 'failed/ambiguous submission; never retry')
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        if claimed: save('launcher_failure.json', dict(utc=utc(), type=type(exc).__name__, message=str(exc), blind_retry_forbidden=True))
        raise
    finally:
        for handle in reversed(locks): fcntl.flock(handle, fcntl.LOCK_UN); handle.close()
        if claimed: save('controller_released.json', dict(pid=os.getpid(), utc=utc(), released=True))

if __name__ == '__main__': main()
