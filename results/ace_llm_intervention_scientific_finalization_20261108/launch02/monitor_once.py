"""One read-only scientific finalization snapshot, never submission or acceptance.
Reads scheduler status and bounded opaque supervisor/publication markers only. Private
campaign journals, descriptors, requests and generation token records stay opaque.
"""
from pathlib import Path
import argparse,subprocess,sys
sys.dont_write_bytecode=True
import build_packet as b
HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'

REMOTE=r'''
from pathlib import Path
import os,stat,hashlib,json,subprocess,datetime,signal
root=Path(ROOT)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def timeout(signum,frame):raise TimeoutError('bounded monitor remote deadline')
signal.signal(signal.SIGALRM,timeout);signal.alarm(45)
def read(p,cap=1024*1024):
 need(not p.is_symlink(),'evidence symlink forbidden')
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and s.st_size<=cap,'bounded ordinary evidence required')
  value=f.read(cap+1)
 need(len(value)<=cap,'oversize evidence');return value
def mark(p):
 if not p.exists():return dict(present=False)
 data=read(p);return dict(present=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),mtime_ns=p.stat().st_mtime_ns)
need(hashlib.sha256(read(root/'submission.intent.json')).hexdigest()==INTENT_SHA,'original remote intent differs')
observations=[]
for command in [['squeue','-j',JOB,'-h','-o','%i|%j|%T|%M|%R'],
 ['sacct','-j',JOB,'--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,ReqMem,AllocTRES%180,MaxRSS','-n','-P'],
 ['scontrol','show','job',JOB,'-o']]:
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=12)
  observations.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 except subprocess.TimeoutExpired as exc:
  txt=lambda x:x.decode(errors='replace') if isinstance(x,bytes) else x
  observations.append(dict(command=command,timeout=True,stdout=txt(exc.stdout),stderr=txt(exc.stderr)))
paths=['finalization/start.json','finalization/attempt.json','finalization/failure.json','finalization/allocation.json',
 'finalization/stage.json','finalization/handoff.json','finalization/worker.started.json','finalization/worker.cleanup.json',
 'finalization/worker.resources.json','finalization/worker/start.json','finalization/worker/work-start.json',
 'finalization/worker/runtime-before.json','finalization/worker/runtime-ready.json','finalization/worker/runtime-after.json',
 'finalization/worker/terminal.json','finalization/worker/attempt.json','finalization/worker/failure.json']
evidence={name:mark(root/name) for name in paths}
# Publication may precede a failing outer cleanup: presence is never acceptance or permission to retry.
campaign=Path(CAMPAIGN)
for name in ('finalization-intent.json','finalization.candidate.json','finalization.json'):
 evidence['campaign/'+name]=mark(campaign/name)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
 observations=observations,evidence=evidence,acceptance=False,scientific_qualified=False,
 note='Bounded opaque marker hashes only. Preserve failed/partial reports and all original campaign outputs; terminal custody and independent audit required.')))

'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('job-id','packet-sha256','submission-result-sha256','packet-directory','evidence-directory'):ap.add_argument('--'+n,required=True)
    a=ap.parse_args();local=Path(a.packet_directory).resolve(strict=True)
    p=b.decode(b.read(local/'packet.json',a.packet_sha256,maximum=128*1024**2));r=b.decode(b.read(local/'submission.result.json',a.submission_result_sha256))
    b.need(local==b.LOCAL and p['schema']==b.OUTER_VERSION and p['remote_root']==b.ROOT and p['resources']==b.RESOURCES and p['deadlines']==b.DEADLINES,'exact reviewed scientific finalizer packet required')
    b.need(a.job_id.isdigit() and r['job_id']==a.job_id and r['returncode']==0 and r['ambiguous'] is False and b.pin(r['intent_sha256']),'exact unambiguous original job required')
    intent=b.decode(b.read(local/'submission.intent.json'))
    b.need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==b.ROOT,'original local intent differs')
    for n in ('monitor_once.py','build_packet.py'):
        b.need(b.read(Path(__file__).resolve().with_name(n))==b.read(local/n,p['new_input_files'][n]),'monitor/helper differs from captured packet')
    out=Path(a.evidence_directory).absolute();out.mkdir(mode=0o700,parents=True,exist_ok=False)
    source='ROOT='+repr(b.ROOT)+'\nJOB='+repr(a.job_id)+'\nINTENT_SHA='+repr(r['intent_sha256'])+'\nCAMPAIGN='+repr(b.CPU_ROOT+'/cpu/worker/artifacts/campaign')+'\n'+REMOTE
    b.write(out/'remote_source.py',source.encode());b.write(out/'monitor_once.py',Path(__file__).read_bytes())
    b.write(out/'intent.json',b.raw(dict(job_id=a.job_id,packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
        monitor_sha256=b.sha(Path(__file__).read_bytes()),acceptance=False)))
    try:
        with b.coordinator_locks():
            run=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=50)
        b.write(out/'stdout.log',run.stdout.encode());b.write(out/'stderr.log',run.stderr.encode())
        b.write(out/'observation.json',b.raw(dict(returncode=run.returncode,utc=b.utc(),job_id=a.job_id,acceptance=False)))
        b.need(run.returncode==0,'observation failed; retained, never retry submission');print(run.stdout)
    except subprocess.TimeoutExpired as exc:
        data=lambda v:v if isinstance(v,bytes) else (v or '').encode()
        b.write(out/'stdout.log',data(exc.stdout));b.write(out/'stderr.log',data(exc.stderr))
        b.write(out/'failure.json',b.raw(dict(type=type(exc).__name__,message=str(exc),timeout=True,job_id=a.job_id,retry_submission=False)));raise
    except BaseException as exc:
        b.write(out/'failure.json',b.raw(dict(type=type(exc).__name__,message=str(exc),retry_submission=False)));raise

if __name__=='__main__':main()
