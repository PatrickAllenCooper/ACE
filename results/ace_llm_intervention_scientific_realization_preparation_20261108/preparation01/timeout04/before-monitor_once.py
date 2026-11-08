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
    ap.add_argument('--evidence-directory',required=True,help='new exclusive local directory for this observation')
    a=ap.parse_args()
    b.read(b.LOCAL/'packet.json',a.packet_sha256)
    result=b.decode(b.read(b.LOCAL/'submission.result.json',a.submission_result_sha256))
    job=result.get('job_id');b.need(type(job) is str and job.isdigit() and not result.get('ambiguous'),
                                 'unambiguous original submission required; otherwise reconcile manually')
    intent=b.decode(b.read(b.LOCAL/'submission.intent.json'))
    b.need(intent['packet_sha256']==a.packet_sha256 and intent['remote_root']==b.ROOT,'original intent binding differs')
    destination=Path(a.evidence_directory).absolute();destination.mkdir(mode=0o700,parents=True,exist_ok=False)
    source='ROOT='+repr(b.ROOT)+'\nJOB='+repr(job)+'\nINTENT_SHA='+repr(result['intent_sha256'])+'\n'+r'''
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
  observations.append(dict(command=command,timeout=True,message=str(exc)))
paths=['realization/terminal.json','realization/failure.json','realization/publication-interruption.json',
       'realization/session/worker/terminal.json','realization/session/worker/failure.json',
       'realization/session/worker/task/terminal.json','realization/session/worker/task/failure.json']
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
    except BaseException as exc:
        b.write(destination/'failure.json',b.raw(dict(type=type(exc).__name__,message=str(exc),retry_submission=False)))
        raise


if __name__=='__main__':main()
