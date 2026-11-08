from pathlib import Path
import os,stat,hashlib,json,time,resource,traceback,ast
OUT=Path(__file__).resolve().parents[1];cfg=json.loads((OUT/'authority.json').read_bytes());start=time.monotonic();cpu=time.process_time()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
 return h.hexdigest()
def dump(n,v):
 with (OUT/'attempts'/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def fingerprint(p):
 s=p.lstat();need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'non-independent regular file: '+str(p));h=sha(p);a=p.lstat()
 need((s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns),'changed file')
 return dict(bytes=s.st_size,sha256=h)
try:
 roots=[Path(cfg[k]) for k in ('repo','external')]
 for root in roots:
  need((root/'custody-inventory.json').stat().st_size==cfg['custody_seal_bytes'],'seal size')
  for n,h in cfg['pins'].items():need(sha(root/n)==h,'independent input pin: '+n)
 need(cfg['custody_seal_bytes']<96*1024**2,'specific bounded seal reader96MiB')
 seal=json.loads((roots[0]/'custody-inventory.json').read_bytes());need(seal['identical'] is True and len(seal['copies'])==2,'two sealed copies')
 report=[];raw_views=[]
 for root in roots:
  expected=next(v for k,v in seal['copies'].items() if Path(k).resolve()==root.resolve());files={};dirs=[]
  for parent,dnames,fnames in os.walk(root,followlinks=False):
   for n in sorted(dnames):
    p=Path(parent)/n;need(not p.is_symlink(),'directory symlink');dirs.append(p.relative_to(root).as_posix())
   for n in sorted(fnames):
    p=Path(parent)/n;rel=p.relative_to(root).as_posix()
    if rel!='custody-inventory.json':files[rel]=fingerprint(p)
  need(files==expected['files'] and sorted(dirs)==sorted(expected['directories']),'sealed whole capsule differs: '+str(root))
  raw={n[7:]:v for n,v in files.items() if n.startswith('remote/')};raw_dirs=sorted(n[7:] for n in dirs if n.startswith('remote/'))
  need(len(raw)==cfg['raw_files'] and sum(v['bytes'] for v in raw.values())==cfg['raw_bytes'],'raw counts')
  raw_views.append((raw,raw_dirs));report.append(dict(root=str(root),files=len(files),directories=len(dirs),all_hashes_matched=True,raw_files=len(raw),raw_bytes=sum(v['bytes'] for v in raw.values())))
 need(raw_views[0]==raw_views[1],'two raw independent copies disagree')
 t=[json.loads((roots[0]/n).read_bytes()) for n in ('inventory_transport.json','inventory_after_transport.json')];obs=[]
 for tr in t:
  need(tr['returncode']==0,'inventory transport status');v=json.loads(tr['stdout']);obs.append(v)
  need(v['terminal_gate'] is True and v['root_present'] is True and not v.get('observation_error'),'terminal gate')
  need(v['files']==raw_views[0][0] and sorted(v['directories'])==raw_views[0][1],'original remote/raw inventory differs')
  assignments={}
  for node in ast.parse(tr['python_source']).body:
   if isinstance(node,ast.Assign):
    for target in node.targets:
     if isinstance(target,ast.Name) and target.id in ('ROOT','JOB'):assignments[target.id]=ast.literal_eval(node.value)
  need(assignments==dict(ROOT='/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_scientific_training_gpu_20261108_01',JOB='35420219'),'inventory root/job')
 for k in ('files','directories','original_hardlink_groups','parent_accounting','journal_pins','upstream_pins','upstream_errors'):
  need(obs[0][k]==obs[1][k],'observations differ: '+k)
 need(sorted(obs[0]['related_accounting'])==sorted(obs[1]['related_accounting']),'accounting observations differ')
 disp=json.loads((roots[0]/'disposition.json').read_bytes())
 need(disp['errors']==disp['input_pin_mismatches']==[] and disp['current_lineage_verified'] is disp['custody_bytes_verified'] is True,'collector conflicts')
 need(disp['remote_files']==raw_views[0][0] and disp['observed_related_jobs']==45 and disp['observed_submission_journals']==92,'disposition raw/history')
 need((roots[0]/'stream-attempt/archive.tar').stat().st_size==cfg['archive_bytes'],'archive size')
 dump('custody01.json',dict(status='verified_complete_original_custody',copies=report,observations_matched=True,parent_row=obs[0]['parent_accounting'],related_rows=len(obs[0]['related_accounting']),journals=len(obs[0]['journal_pins']),archive_sha256=cfg['pins']['stream-attempt/archive.tar'],specific_seal_read_bound_bytes=96*1024**2,seal_bytes=cfg['custody_seal_bytes'],qualification_flags_emitted=False,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 print('custody01 verified',flush=True)
except BaseException as exc:
 dump('custody01-failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise
