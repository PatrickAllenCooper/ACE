"""Independent saved-byte TOKEN01 failure custody audit. Stdlib only, no project imports."""
from pathlib import Path
import os,stat,json,hashlib,tarfile,subprocess,time
HERE=Path(__file__).resolve().parent
REPO=Path('/Users/pat/code/ACE');TERM=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/terminal01'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-tokenization-terminal-20261108-01')
JOB='35415334';ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_tokenization_20261108_01'
PINS={'disposition.json':'2f55409dce3d44e7ee2e5eed6a359ffa1b46cdec03a7672e23eaca73f52fd679',
 'custody-inventory.json':'b02ecbbd83a3d9f48a3e7248fbb2512be66ffadac723a05bb2fb038465035cc9',
 'inventory_transport.json':'cd6da887c378ac54c83e4fb6732bd249141e9a2629e63d7d2ba6bac899c8e751',
 'inventory_after_transport.json':'d473e204b199c8efbd25f9afe562a0cf33d29fe353b3f485712b2c3c84fcac61',
 'stream-attempt/capture.json':'274879344b62d18e553e12cbf7289ea2a69e600b28296918dc57ddbbe5da88fe',
 'stream-attempt/materialization.json':'4437a8ee5d6085c2a5d99cdb0fd3c93157c88ca57f72f60fe6d9ddc6fed7fdc3'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p,h=None):
 b=Path(p).read_bytes()
 if h:need(sha(b)==h,'independent pin mismatch '+str(p))
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'duplicate JSON key');d[k]=v
  return d
 def bad(x):raise ValueError('nonfinite JSON')
 return json.loads(b,object_pairs_hook=pairs,parse_constant=bad)
def fp(p):
 a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_nlink==1,'ordinary independent file required '+str(p));h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 z=p.lstat();need((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns),'file changed')
 return dict(bytes=z.st_size,sha256=h.hexdigest())
def inventory(root):
 files={};dirs=[]
 for d,children,names in os.walk(root,followlinks=False):
  for n in children:
   p=Path(d)/n;need(stat.S_ISDIR(p.lstat().st_mode),'nonordinary directory');dirs.append(p.relative_to(root).as_posix())
  for n in names:
   p=Path(d)/n;files[p.relative_to(root).as_posix()]=fp(p)
 return dict(files=files,directories=sorted(dirs))
def normalized(rows):
 out=[]
 for r in rows:
  need(len(r)>=8 and not any(r[8:]) and r[0].isdigit() and r[4].isdigit() and r[5].isdigit(),'invalid accounting')
  pairs=[x.split('=',1) for x in r[7].split(',') if x]
  need(all(len(x)==2 for x in pairs) and len({x[0] for x in pairs})==len(pairs),'duplicate/malformed TRES')
  out.append([r[0],r[1].strip(),r[2].strip(),r[3].strip(),int(r[4]),int(r[5]),r[6].strip(),sorted(pairs)])
 need(len({r[0] for r in out})==len(out),'duplicate job')
 return sorted(out,key=lambda r:int(r[0]))
def cost(rows):
 cpu=gpu=0
 for r in normalized(rows):
  tres=dict(r[7]);accelerators={k:v for k,v in tres.items() if k.startswith('gres/')}
  need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in accelerators.items()),'unknown accelerator amount')
  typed=sum(int(v) for k,v in accelerators.items() if k.startswith('gres/gpu:'))
  n=int(accelerators.get('gres/gpu',typed));need(not('gres/gpu' in accelerators and any(k.startswith('gres/gpu:') for k in accelerators)) or n==typed,'GPU quantity conflict')
  cpu+=r[4]*r[5];gpu+=n*r[5]
 return cpu,gpu

def main():
 started=time.monotonic();cpu=time.process_time()
 for n,h in PINS.items():load(TERM/n,h);need((TERM/n).read_bytes()==(EXT/n).read_bytes(),'pinned mirror differs')
 d=load(TERM/'disposition.json');seal=load(TERM/'custody-inventory.json')
 need(seal['identical'] is True and seal['acceptance'] is False and seal['excludes']=='custody-inventory.json itself','seal contract')
 need(set(seal['copies'])=={str(TERM),str(EXT)},'seal roots')
 snapshots=[]
 for root in (TERM,EXT):
  actual=inventory(root);sealed=dict(files={n:v for n,v in actual['files'].items() if n!='custody-inventory.json'},directories=actual['directories'])
  need(sealed==seal['copies'][str(root)],'full capsule mismatch '+str(root));snapshots.append(actual)
 need(snapshots[0]==snapshots[1],'complete mirror mismatch')
 for n in snapshots[0]['files']:need(not os.path.samestat((TERM/n).stat(),(EXT/n).stat()),'same inode in independent mirrors')
 transport=[load(TERM/n) for n in ('inventory_transport.json','inventory_after_transport.json')]
 for t in transport:need(t['returncode']==0 and not t['stderr'],'inventory transport failed')
 obs,after=[json.loads(t['stdout']) for t in transport]
 for k in ('files','directories','related_accounting','parent_accounting','journal_pins','upstream_pins','original_hardlink_groups'):
  need(obs[k]==after[k],'before/after mismatch '+k)
 need(obs['terminal_gate'] is after['terminal_gate'] is True and not obs['upstream_errors'] and not after['upstream_errors'],'terminal/upstream gate')
 raw=dict(files={n[len('remote/'):]:v for n,v in snapshots[0]['files'].items() if n.startswith('remote/')},directories=[n[len('remote/'):] for n in snapshots[0]['directories'] if n.startswith('remote/')])
 need(raw['files']==obs['files']==d['remote_files'] and raw['directories']==sorted(obs['directories'])==sorted(d['remote_directories']),'complete raw inventory mismatch')
 need(len(raw['files'])==31039 and sum(v['bytes'] for v in raw['files'].values())==218721958,'raw membership/count mismatch')
 need(d['errors']==d['input_pin_mismatches']==[] and d['custody_bytes_verified'] is d['current_lineage_verified'] is True,'disposition conflicts')
 need(d['status']=='terminal_failed_outputs_preserved' and d['job_id']==JOB and d['remote_root']==ROOT,'failed disposition identity')
 # Independently hash each USTAR member without extraction or payload decoding.
 stream=load(TERM/'stream-transport.json');archive=TERM/'stream-attempt/archive.tar'
 need(fp(archive)==stream['archive'],'retained archive pin differs')
 tfiles={};tdirs=[];seen=set()
 with tarfile.open(archive,'r|') as t:
  for member in t:
   n=member.name;need(n and not n.startswith('/') and all(x not in ('','.','..') for x in n.split('/')) and n not in seen,'unsafe/duplicate archive path');seen.add(n)
   need(not member.pax_headers,'unexpected archive extension')
   if member.isdir():tdirs.append(n)
   else:
    need(member.isfile() and not member.islnk() and not member.issym() and n in raw['files'],'archive special/unknown member')
    h=hashlib.sha256();count=0
    with t.extractfile(member) as f:
     for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk);count+=len(chunk)
    tfiles[n]=dict(bytes=count,sha256=h.hexdigest());need(count==member.size,'truncated archive member')
 need(tfiles==raw['files'] and sorted(tdirs)==raw['directories'],'full archive membership/bytes differ')
 capture=load(TERM/'stream-attempt/capture.json');mat=load(TERM/'stream-attempt/materialization.json')
 need(capture['returncode']==0 and capture['cleanup']['verified'] is True and capture['cleanup']['errors']==[] and capture['cleanup']['leader_reaped'] is capture['cleanup']['group_absent'] is True,'transport cleanup failed')
 need(mat['files']==len(tfiles) and mat['bytes']==sum(x['bytes'] for x in tfiles.values()) and mat['archive']==stream['archive'],'materialization differs')
 # Authenticate original submission and committed captured input roster.
 outer=load(TERM/'remote/launch/packet.json',d['packet_sha256']);result=load(TERM/'remote/launch/submission.result.json',d['submission_result_sha256'])
 remote_intent=load(TERM/'remote/launch/submission.intent.json',result['intent_sha256'])
 local_intent=load(TERM/'coordinator/local-launch/submission.intent.json')
 need(set(remote_intent)==set(local_intent)|{'remote_utc','fresh_readiness'} and all(remote_intent[k]==v for k,v in local_intent.items()),'original local/remote intent mismatch')
 need(result['returncode']==0 and result['ambiguous'] is False and result['job_id']==JOB,'submission identity')
 pins={**outer['new_input_files'],'packet.json':d['packet_sha256']};need(len(pins)==87 and local_intent['pins']==pins,'exact87 submission pins')
 for n,h in pins.items():
  need(raw['files']['launch/'+n]['sha256']==h,'raw input mismatch '+n)
  value=(TERM/'remote/launch'/n).read_bytes()
  need(value==(TERM/'coordinator/local-launch'/n).read_bytes(),'original captured input differs')
  path='results/ace_llm_intervention_scientific_tokenization_20261108/launch01/'+n
  committed=subprocess.check_output(['git','--no-replace-objects','show',local_intent['packet_commit']+':'+path],cwd=REPO)
  need(committed==value,'packet Git binding differs')
 # Independent accepted original collection/failure trees, hash joins only.
 prior_root=REPO/'results/ace_llm_intervention_scientific_collection_20261108/terminal02'
 prior_disp=load(prior_root/'disposition.json','9ab4b1f7fd8aff0e29ec3a8dc2849483d27a3da4071b397d60b95ee46b4e79a9')
 for oldlabel,newlabel in [('launch','common'),('common','failed01')]:
  wanted={n[len(oldlabel)+1:]:v for n,v in prior_disp['remote_files'].items() if n.startswith(oldlabel+'/')}
  actual={n[len(newlabel)+1:]:v for n,v in raw['files'].items() if n.startswith(newlabel+'/')}
  need(wanted==actual,'complete preserved prior root changed '+newlabel)
  wd={n[len(oldlabel)+1:] for n in prior_disp['remote_directories'] if n.startswith(oldlabel+'/')}
  ad={n[len(newlabel)+1:] for n in raw['directories'] if n.startswith(newlabel+'/')};need(wd==ad,'preserved original directories changed')
 acceptance=load(TERM/'remote/launch/collection-acceptance.json','e217199099b81480f0e85c09f0432b60dd5edb2aad1b6b30c8ed86ca736e34e5')
 cp='common/collection/session/worker/task/collection/'
 need({n[len(cp):]:v['sha256'] for n,v in raw['files'].items() if n.startswith(cp)}==acceptance['collection']['files'],'accepted collection changed')
 need(raw['files']['namespace.json']['sha256']==local_intent['existing_namespace_claim_sha256'] and raw['files']['original-namespace.json']['sha256']==local_intent['original_namespace_claim_sha256'],'registry mismatch')
 # Exact complete rows and journals; failure charge never dropped.
 base=load(TERM/'remote/launch/baseline.json','ab8efc04b63d1598b5416a1c62f2c346b3fef5ab46842119b3a62ed89c3de62b')
 rows=obs['related_accounting'];current=[r for r in rows if r[0]==JOB];old=[r for r in rows if r[0]!=JOB]
 need(len(rows)==43 and len(current)==1 and normalized(old)==normalized(base['related_accounting']),'exact42 predecessor rows changed')
 row=current[0];need(row==obs['parent_accounting'] and row[2:8]==['FAILED','1:0','3','129','ucb736_asc2','billing=3,cpu=3,mem=11520M,node=1'],'actual failed allocation differs')
 need(cost(current)==(387,0) and cost(old)==(12986,1601) and cost(rows)==(13373,1601),'actual charges mismatch')
 jp={**base['journal_pins'],ROOT+'/submission.intent.json':result['intent_sha256'],ROOT+'/submission.result.json':d['submission_result_sha256']}
 need(len(jp)==88 and jp==obs['journal_pins'],'exact86 predecessor journals plus new2 differ')
 for record in (obs,after):
  for name in ('job','steps','details','queue','related_accounting'):need(record['results'][name]['returncode']==0,'scheduler observation failed')
  queue=[l.split('|') for l in record['results']['queue']['stdout'].splitlines() if l.strip()]
  need(all(len(r)==5 and not r[1].lower().startswith('ace') for r in queue),'active ACE reservation')
 fields=dict(x.split('=',1) for x in obs['results']['job']['stdout'].split() if '=' in x)
 for k,v in dict(JobId=JOB,Account='ucb736_asc2',QOS='cpu-normal',JobState='FAILED',Requeue='0',Restarts='0',ExitCode='1:0',RunTime='00:02:09',TimeLimit='00:10:00',Partition='acpu',NumCPUs='3',NumNodes='1',NumTasks='1',AllocTRES='cpu=3,mem=11520M,node=1,billing=3').items():need(fields[k]==v,'scheduler field '+k)
 out=dict(status='complete_saved_custody_and_failure_accounting_verified',acceptance=False,job_id=JOB,
   raw_files=len(raw['files']),raw_bytes=sum(v['bytes'] for v in raw['files'].values()),raw_directories=len(raw['directories']),
   capsule_files=len(snapshots[0]['files']),capsule_bytes=sum(v['bytes'] for v in snapshots[0]['files'].values()),
   artifact_pins=PINS,packet_sha256=d['packet_sha256'],packet_commit=local_intent['packet_commit'],submission_result_sha256=d['submission_result_sha256'],remote_intent_sha256=result['intent_sha256'],
   original_input_pins_verified=len(pins),two_independent_full_copies_verified=True,two_remote_inventories_verified=True,retained_archive_all_members_verified=True,
   complete_collection_and_failed_predecessor_bytes_unchanged=True,collection_files=len(acceptance['collection']['files']),
   terminal_jobs=43,submission_journals=88,closed_cpu_seconds=13373,closed_gpu_slice_seconds=1601,cpu_admission_seconds=16973,outstanding_reservation=0,
   failed_allocation_cpu_seconds=387,failed_allocation_gpu_seconds=0,related_accounting=rows,journal_pins=jp,
   scheduler_steps=obs['results']['steps'],scheduler_details=obs['results']['details'],node=fields['NodeList'],
   local_audit_wall_seconds=time.monotonic()-started,local_audit_process_cpu_seconds=time.process_time()-cpu,
   no_private_payload_decoding=True,no_scientific_execution=True)
 (HERE/'custody-result.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ('journal_pins','related_accounting','scheduler_steps','scheduler_details','artifact_pins')}))
if __name__=='__main__':main()
