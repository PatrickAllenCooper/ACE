"""Independent saved metadata inspection only. No project imports or payload arithmetic."""
from pathlib import Path
import json, hashlib, os, stat, subprocess, time
ROOT=Path('/Users/pat/code/ACE'); BASE=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
R=ROOT/'results/ace_llm_intervention_scientific_finalization_20261108'; T=R/'terminal01'; L=R/'launch02'
E=BASE/'ACE-LLM-intervention-scientific-finalization-terminal-20261108-01'
OUT=Path(__file__).parent; pins={}; checks=[]; facts={}
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(x):return (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def need(x,label):
 if not x:raise ValueError(label)
 checks.append(label)
def read(p,expected=None):
 p=Path(p);s=p.lstat();need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'regular sole-link '+str(p))
 b=p.read_bytes();h=sha(b);pins[str(p)]={'sha256':h,'bytes':len(b)}
 if expected:need(h==expected,'pin '+str(p))
 return b
def j(p,expected=None):return json.loads(read(p,expected))
def inv(p,exclude=()):
 d=[];f={}
 for root,ds,fs in os.walk(p,followlinks=False):
  for n in ds:
   q=Path(root)/n;need(not q.is_symlink(),'no directory link '+str(q));d.append(q.relative_to(p).as_posix())
  for n in fs:
   q=Path(root)/n;k=q.relative_to(p).as_posix()
   if k in exclude:continue
   b=read(q);f[k]={'bytes':len(b),'sha256':sha(b)}
 return dict(directories=sorted(d),files=dict(sorted(f.items())))
def normalized(x):
 need(type(x['directories']) is list and len(x['directories'])==len(set(x['directories'])),'unique directory list')
 need(all(n and not n.startswith('/') and all(s not in ('','.','..') for s in n.split('/')) for n in list(x['files'])+x['directories']),'safe inventory names')
 return dict(files=x['files'],directories=sorted(x['directories']))
def rowsnorm(rows):
 need(len(rows)==len({r[0] for r in rows}),'unique job IDs'); out=[]
 for r in rows:
  need(len(r)==8 and r[4].isdigit() and r[5].isdigit(),'exact numeric accounting row')
  p=[v.split('=',1) for v in r[7].split(',')];need(len(p)==len({k for k,v in p}),'unique TRES')
  out.append(r[:7]+[','.join(k+'='+v for k,v in sorted(p))])
 return sorted(out)
def githash(commit,path):
 return sha(subprocess.run(['git','show',commit+':'+path],cwd=ROOT,check=True,capture_output=True).stdout)
def audit():
 completion=j(R/'monitor-collection-control01/completion.json','bfc35146bd6eeb776fd61e5113335d07035096699a042f511b87ab93bf0ebe11')
 seal=j(T/'custody-inventory.json','8b05d07c16d9702107cdd984c8d4c0d0d8603c123ec011b7072d4cc08b3b6d62')
 need(seal['identical'] is True and seal['copies'][str(T)]==seal['copies'][str(E)],'sealed exact two-copy roster')
 for p in (T,E):need(inv(p,('custody-inventory.json',))==seal['copies'][str(p)],'actual whole capsule matches '+str(p))
 need(read(T/'custody-inventory.json')==read(E/'custody-inventory.json'),'seal copies equal')
 need(all((T/n).stat().st_ino!=(E/n).stat().st_ino or (T/n).stat().st_dev!=(E/n).stat().st_dev for n in seal['copies'][str(T)]['files']),'independent copy inodes')
 before=j(T/'inventory_transport.json','54e2035dd8ef619ff98de7f9a78d1609bd5c38facc89f7860a17cf2a4c7f3c30');after=j(T/'inventory_after_transport.json','4b790ec12fb4c4767c4db2b32cf92433fda66390b94dcd0df115517a49cf5774')
 obs=[]
 for record in (before,after):
  need(record['returncode']==0,'observation transport success');x=json.loads(record['stdout']);obs.append(x)
  need('observation_error' not in x and not x['upstream_errors'] and x['terminal_gate'] is True,'complete terminal observation')
  progress=[json.loads(n) for n in record['stderr'].splitlines() if n];need(all(n['phase']=='terminal_inventory_verification' and n['files_read']==1000*(i+1) for i,n in enumerate(progress)),'bounded fingerprint progress only')
 b,a=obs;rc=j(T/'remote-custody.json','3afa194bb489d6a31467e0ad10c1322e8ddf7c199fcc3908ef039d0330250ae2');d=j(T/'disposition.json','9d047cfaed3a8eae7fcf9dbc01e65e09063020ca5f6fff81acdf9c15872c89e2')
 full=normalized(rc['full_inventory']);need(normalized(b)==normalized(a)==full,'before after archive full inventory exact')
 need(rc['full_backup_verified'] is True and rc['full_local_copy'] is False and rc['acceptance'] is False,'honest full remote compact local custody')
 need(rc['original_before_inventory_sha256']==rc['original_after_inventory_sha256']==rc['project_readback_inventory_sha256']==sha(raw(full)),'three independent full inventory pins')
 need(rc['source_observation_sha256']==sha(raw(b)),'source observation serialization pin')
 need(rc['helper_sha256']=='4161aa5829d49b6b43e77bb0c3a23b4cfef9c9c5f7e56290d8b94b2444b7df9f','reviewed helper pin');read(T/'remote_custody.py',rc['helper_sha256'])
 need(d['errors']==[] and d['input_pin_mismatches']==[] and d['custody_bytes_verified'] is True and d['current_lineage_verified'] is True,'collector complete zero conflicts')
 need(d['remote_files']==full['files'] and sorted(d['remote_directories'])==full['directories'],'disposition whole archive map')
 need(len(full['files'])==107284 and sum(v['bytes'] for v in full['files'].values())==3308319856,'full raw totals')
 local=normalized(rc['local_inventory']);need(inv(T/'remote')==local,'selected bytes exact subset')
 need(all(full['files'].get(k)==v for k,v in local['files'].items()),'selected pins exact full map')
 need(len(local['files'])==379 and sum(v['bytes'] for v in local['files'].values())==306334884,'selected totals')
 need(set(rc['omitted_files'])==set(full['files'])-set(local['files']),'explicit omission membership')
 need(sum(v['bytes'] for k,v in full['files'].items() if k not in local['files'])==rc['omitted_bytes']==3001984972,'explicit omitted bytes')
 # Reuse accepted predecessor, compare current raw maps without re-auditing old science.
 GP=ROOT/'results/ace_llm_intervention_scientific_campaign_gpu_20261108'; ga=j(GP/'acceptance01/acceptance.json','510ceadd2bdf831fd43c27567dfa799df91efb8f42aa11141f14d294870534c6');base=j(GP/'acceptance01/baseline.json','40e9ec4c91bf174214e8788306b3900ba5f92ec32d3a01a024bc2b0558f50897');gr=j(GP/'acceptance01/recovery-acceptance.json','1c0329023225c0f97e63ef0291743695ea930af85ec8f790e9dae902155afe2a')
 prior=j(GP/'terminal02/remote-custody.json','e59bca378f54b0225eb8c74b73bfeebdad830faf5d3fd62fedfc56593de8e27c')
 mapping={k:next(n for n,p in rc['roots'].items() if p==v) for k,v in prior['roots'].items()}
 def remap(n):
  h,sep,tail=n.partition('/');return mapping[h]+(sep+tail if sep else '')
 oldfiles={remap(n):v for n,v in prior['full_inventory']['files'].items()}
 need(all(full['files'].get(n)==v for n,v in oldfiles.items()),'all accepted upstream files preserved')
 need(set(remap(n) for n in prior['full_inventory']['directories'])<=set(full['directories']),'all upstream directories preserved')
 added=set(full['files'])-set(oldfiles);cpre='cpu/cpu/worker/artifacts/campaign/'
 need(all(n.startswith('launch/') or n.startswith(cpre+'private/') or n in {cpre+s for s in ('finalization-intent.json','finalization.json','finalization.candidate.json')} for n in added),'only new finalizer control or reporting files')
 campaign={n[len(cpre):]:v['sha256'] for n,v in full['files'].items() if n.startswith(cpre)}
 need(all(campaign.get(n)==h for n,h in ga['campaign_files'].items()),'accepted campaign retained whole')
 # Packet, actual source, clock adoption and submission joins.
 outer=j(L/'packet.json','dcd34d5c040b7a9bf2769736a8ee5fa103090c4ef1af3f7c663cd961eabbc433');inner=j(L/'inputs/packet.json',outer['finalizer_packet_sha256']);ci=j(T/'collector-intent.json','61067cfb1e4e5ade8f26bd41c36bf8e0efd2ece70712e122004122bf2ea1aba7')
 for n,h in inner['files'].items():
  read(L/'inputs'/n,h);need(full['files']['launch/inputs/'+n]['sha256']==h,'actual input '+n)
 for n,h in outer['new_input_files'].items():
  read(L/n,h);need(full['files']['launch/'+n]['sha256']==h,'actual transferred object '+n)
 need(a['upstream_pins']==b['upstream_pins']==outer['original_input_pins'],'all original input pins at both observations')
 it=j(L/'submission.intent.json','849c9610467de284b38daea5edd6fe27372269d22278d2f853058d22c05f121b');res=j(L/'submission.result.json','044b96cf3225439ecbb4716823a4b243391f5b2fdfeb8b7b4e952ce7d47d130d');ri=j(T/'remote/launch/submission.intent.json',res['intent_sha256'])
 need(set(ri)==set(it)|{'remote_utc','fresh_readiness'} and {k:ri[k] for k in it}==it,'canonical local remote intent join')
 need(res['job_id']=='35533604' and res['returncode']==0 and res['ambiguous'] is False and res['stdout']=='35533604\n','once canonical unambiguous submission')
 need(full['files']['launch/submission.result.json']['sha256']==sha(raw(res)),'remote result exact')
 need(ci['packet_commit']==it['packet_commit']=='fc57d6d796b30a02e12b3c7fc32d9f6394638912','original actual packet commit')
 for n in ['packet.json','inputs/packet.json','job.sbatch','launch_once.py','build_packet.py']:
  need(githash(it['packet_commit'],str((L/n).relative_to(ROOT)))==sha(read(L/n)),'packet Git '+n)
 read(T/'pull_terminal.py',ci['collector_sha256']);need(ci['collector_sha256']=='b1994191042e910923ffe23e504f2ecd9af6d5242ad6743a5537a2aa4e73e5c3','actual reviewed collector')
 adoption=j(L/'finalizer-clock-adoption.json',inner['clock_adoption_sha256']);manifest=j(L/'finalizer-source-manifest.json',adoption['source_manifest_sha256'])
 entry='finalize_llm_intervention_scientific_campaign_cpu.py';need(adoption['entry_sha256']==inner['files'][entry]==manifest['entry']['sha256']=='8a26ada2fc4778981237ca547fa437c620e6510080e93ab334e9762ebee2cf6f','adopted final source join')
 need(githash(manifest['source_commit'],manifest['entry']['repository_path'])==inner['files'][entry],'operational source Git binding')
 for file,key in [('gpu-baseline.json','baseline_sha256'),('gpu-acceptance.json','gpu_acceptance_sha256'),('gpu-recovery-acceptance.json','gpu_recovery_acceptance_sha256'),('finalizer-source-review.md','source_review_sha256'),('finalizer-timing-review.md','timing_review_sha256')]:read(L/file,adoption[key])
 need(adoption['status']=='adopted' and adoption['required_findings']==0 and adoption['resources']==outer['resources']==inner['resources'] and adoption['deadlines']==outer['deadlines'],'actual adopted profile')
 # Exact accounting and journal extension.
 need(rowsnorm(b['related_accounting'])==rowsnorm(a['related_accounting']),'accounting stable observations');rows=a['related_accounting'];new=[x for x in rows if x[0]=='35533604'];need(len(new)==1 and new[0]==a['parent_accounting'],'one added actual job')
 need(rowsnorm([x for x in rows if x[0]!='35533604'])==rowsnorm(base['related_accounting']),'all 48 predecessor accounting rows unchanged')
 need(len(rows)==49 and new[0][2:7]==['COMPLETED','0:0','2','1596','ucb736_asc2'],'finalizer actual allocation row')
 cp=sum(int(x[4])*int(x[5]) for x in rows);gp=sum(int(x[5])*sum(int(v) for k,v in (z.split('=',1) for z in x[7].split(',')) if k.startswith('gres/gpu:')) for x in rows)
 need((cp,gp)==(40265,4458) and cp-base['closed_cpu_seconds']==3192 and gp==base['closed_gpu_slice_seconds'],'actual cumulative costs')
 journals=a['journal_pins'];need(journals==b['journal_pins'] and len(journals)==100,'100 stable journals');need(all(journals.get(k)==v for k,v in base['journal_pins'].items()),'all 98 predecessor journals unchanged')
 extra={k:v for k,v in journals.items() if k not in base['journal_pins']};need(extra=={outer['remote_root']+'/submission.intent.json':res['intent_sha256'],outer['remote_root']+'/submission.result.json':sha(raw(res))},'exact two new journals')
 need(rowsnorm(ri['fresh_readiness']['related_accounting'])==rowsnorm(base['related_accounting']) and ri['fresh_readiness']['journal_pins']==base['journal_pins'],'actual pre-submit exact baseline')
 for k in ('related_accounting','steps','details','queue'):need(a['results'][k]['returncode']==0 and not a['results'][k]['stderr'],'saved scheduler query '+k)
 step=[n.split('|') for n in a['results']['steps']['stdout'].splitlines()];need({n[0] for n in step}=={'35533604','35533604.batch','35533604.extern'} and all(n[2:4]==['COMPLETED','0:0'] for n in step),'all scheduler steps clean')
 need(a['results']['details']['stdout'].strip().split('|')[:4]==['35533604','acpu','cpu-normal','70'],'actual partition qos wall')
 need(not any(n.split('|')[1].startswith('ace_llm') for n in a['results']['queue']['stdout'].splitlines()),'no active campaign allocation')
 facts.update(full_raw_files=len(full['files']),full_raw_bytes=sum(v['bytes'] for v in full['files'].values()),selected_files=len(local['files']),selected_bytes=sum(v['bytes'] for v in local['files'].values()),capsule_files=len(seal['copies'][str(T)]['files']),upstream_preserved_files=len(oldfiles),new_files=len(added),source_input_files=len(inner['files']),source_names=len(inner['source_names']),accounting=dict(terminal_jobs=49,submission_journals=100,allocated_cpu_seconds=3192,closed_cpu_seconds=cp,closed_gpu_slice_seconds=gp,preparation_allowance_cpu_seconds=3600,cpu_admission_seconds=cp+3600,outstanding_reservation=0),scheduler_steps=step,roots=rc['roots'],archive=dict(path=rc['backup_archive_path'],bytes=rc['backup_archive_bytes'],sha256=rc['backup_archive_sha256']))
 (OUT/'accounting-material.json').write_bytes(raw(dict(related_accounting=rows,journal_pins=journals)))
 (OUT/'campaign-file-pins.json').write_bytes(raw(campaign))
 # Fixed 242 report membership only; no report content decode.
 names=[cpre+'private/'+n for n in ('summary.json','report-pins.json')]
 for w in range(24):
  pre=cpre+f'private/world-{w:02d}/';names.extend(pre+n for n in ('summary.json','report-pins.json'));names.extend(pre+f'arm-{k:02d}/cost-report.json' for k in range(8))
 need(len(names)==len(set(names))==242 and all(n in full['files'] for n in names),'complete fixed 242 report membership')
 members={n:full['files'][n] for n in sorted(names)};need(max(v['bytes'] for v in members.values())<=32*1024**2 and sum(v['bytes'] for v in members.values())<=512*1024**2,'reviewed batch per-member and total bounds')
 (OUT/'report-members.json').write_bytes(raw(members));facts['report_members']=dict(files=242,bytes=sum(v['bytes'] for v in members.values()),max_bytes=max(v['bytes'] for v in members.values()),sha256=sha(raw(members)),content_decoded=False)
 return facts
if __name__=='__main__':
 started=time.monotonic()
 try:
  audit();(OUT/'facts.json').write_bytes(raw(facts));print(json.dumps({'status':'saved_metadata_checks_pass','checks':len(checks),'facts':facts},sort_keys=True))
 except BaseException as e:
  (OUT/'inspection-failure.json').write_bytes(raw(dict(type=type(e).__name__,message=str(e),checks_completed=len(checks))));raise
 finally:
  (OUT/'read-pins.json').write_bytes(raw(pins));(OUT/'checks.json').write_bytes(raw(checks));(OUT/'inspection-cost.json').write_bytes(raw(dict(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time())))
