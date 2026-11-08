"""One read-only scheduler/outer-evidence observation; never retry or accept a run.

Only after an unambiguous submission result. Does not read private world payloads.
Parent owns polling decisions and the complete terminal collector.
"""
from pathlib import Path
import argparse
import subprocess
import sys
sys.dont_write_bytecode=True
import build_packet as b

HELPER='/Users/pat/.codex/skills/curc-access/scripts/curc-access.sh'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--packet-sha256',required=True)
    ap.add_argument('--submission-result-sha256',required=True)
    ap.add_argument('--packet-directory',required=True)
    ap.add_argument('--evidence-directory',required=True,help='new exclusive local directory for this observation')
    a=ap.parse_args()
    local=Path(a.packet_directory).resolve(strict=True)
    packet=b.decode(b.read(local/'packet.json',a.packet_sha256));remote_root=packet['remote_root']
    b.need(packet['schema']=='ace-scientific-collection-launch-v1' and packet['local_root']==str(local),
           'current pinned collection packet required')
    result=b.decode(b.read(local/'submission.result.json',a.submission_result_sha256))
    job=result.get('job_id');b.need(type(job) is str and job.isdigit() and not result.get('ambiguous'),
                                 'unambiguous original submission required; otherwise reconcile manually')
    intent=b.decode(b.read(local/'submission.intent.json'))
    b.need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==remote_root,'original intent binding differs')
    destination=Path(a.evidence_directory).absolute();destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    source='ROOT='+repr(remote_root)+'\nJOB='+repr(job)+'\nINTENT_SHA='+repr(result['intent_sha256'])+'\n'+r'''
import pathlib,hashlib,json,subprocess,datetime
root=pathlib.Path(ROOT)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(root/'submission.intent.json')!=INTENT_SHA:raise RuntimeError('remote intent differs')
commands=[['squeue','-j',JOB,'-h','-o','%i|%j|%T|%M|%R'],
 ['sacct','-j',JOB,'--format','JobIDRaw,JobName%80,State,ExitCode,AllocCPUS,ElapsedRaw,ReqMem,AllocTRES%180,MaxRSS','-n','-P'],
 ['scontrol','show','job',JOB,'-o']]
observations=[]
for command in commands:
 try:
  p=subprocess.run(command,capture_output=True,text=True,timeout=12)
  observations.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 except subprocess.TimeoutExpired as exc:
  def txt(v):return v.decode(errors='replace') if isinstance(v,bytes) else v
  observations.append(dict(command=command,timeout=True,message=str(exc),stdout=txt(exc.stdout),stderr=txt(exc.stderr)))
paths=['collection/terminal.json','collection/failure.json','collection/publication-interruption.json',
       'collection/session/worker/terminal.json','collection/session/worker/failure.json',
       'collection/session/worker/task/terminal.json','collection/session/worker/task/failure.json',
       'collection/session/worker/task/collection/terminal.json','collection/session/worker/task/collection/failure.json']
evidence={}
for name in paths:
 p=root/name
 if p.is_symlink():raise RuntimeError('evidence symlink')
 evidence[name]=dict(present=p.exists(),sha256=sha(p) if p.is_file() else None)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=JOB,
    observations=observations,evidence=evidence,acceptance=False)))
'''
    b.write(destination/'remote_source.py',source.encode())
    try:
        p=subprocess.run(['bash',HELPER,'run','python3 -'],input=source,capture_output=True,text=True,timeout=50)
        b.write(destination/'stdout.log',p.stdout.encode());b.write(destination/'stderr.log',p.stderr.encode())
        b.write(destination/'observation.json',b.raw(dict(returncode=p.returncode,utc=b.utc(),job_id=job,
            packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,acceptance=False)))
        b.need(p.returncode==0,'observation failed; retained, no submission retry')
        print(p.stdout)
    except subprocess.TimeoutExpired as exc:
        def data(v):return v if isinstance(v,bytes) else (v or '').encode()
        b.write(destination/'stdout.log',data(exc.stdout));b.write(destination/'stderr.log',data(exc.stderr))
        b.write(destination/'failure.json',b.raw(dict(type=type(exc).__name__,message=str(exc),
            command=['bash',HELPER,'run','python3 -'],timeout=True,job_id=job,
            packet_sha256=a.packet_sha256,submission_result_sha256=a.submission_result_sha256,
            retry_submission=False)))
        raise
    except BaseException as exc:
        b.write(destination/'failure.json',b.raw(dict(type=type(exc).__name__,message=str(exc),retry_submission=False)))
        raise


if __name__=='__main__':main()
