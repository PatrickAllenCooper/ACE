from pathlib import Path
import os,stat,json,hashlib,time,traceback,subprocess,resource
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/pat/code/ACE')
LOCAL=REPO/'results/ace_llm_intervention_scientific_tokenization_20261108/terminal02'
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-tokenization-terminal-20261108-02')
COMMIT='6f29023422b01db07412f609e12659fef065bf6c'
JOB='35419195'
ANCHORS={'disposition.json':'8d83f99a69e33d7444c4f8eb7559ebb59a06127622912af609fc84446553dfd6',
'custody-inventory.json':'f0dbfeb471ec2afc4132596519376ba4abade6b9d17772922f7a315720019d88',
'inventory_transport.json':'e48caaddc6fd313f6306a989c6f64820b84d131b9be0eeea963f30c7e17f41bf',
'inventory_after_transport.json':'47109f79a0849a15d30441e4e52440b4bf23c88bb93de2577b338dd5637d4a33',
'stream-transport.json':'fb7bd425a33a135cef6b613574c5df766f199f825cd8b3d2881bb6c36790ef10',
'pull_terminal.py':'68d19c2037fa54968483e718d64a11d3eb1f8d7aec9ae09219b96ecd3613c7a0',
'collector-review.md':'39bbebe31a4feca11945032310ef47d37ef31513d78d1b92ad2ec2e5621b9413',
'stream_custody.py':'fab01ad840e4e73ef0fc2136fd2eeb12b1033256e80a8f96a392afb2d50b758a',
'stream_custody_review.md':'5058426e9dccaaf210eb2969a8623a31123f13dfefaa43c57e975e5800819371'}
OUTER='69aaaccdf9f20b11665f4708638db9ba621df4c9881befc1d4bb4225a46f912f'
INNER='cab514d59959dd64c080d1414bd1dea6a47da59a93c55d3d23fe3b2fc338ee1c'
RESULT='6efcd47104ead43413043697846595bb831774e793f5ad5923e47cc9731bfa73'
LOCAL_INTENT='69bb4ea194ea9b7d972caeea8fd30c9bef394ae3f4970fbbb1023e7ef89eb027'
REMOTE_INTENT='6436ef782b8198037d09ca97ed74efb8c96650d6d054865b4af79b91979bcc15'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def decode(raw):
 def unique(rows):
  out={}
  for k,v in rows:need(k not in out,'duplicate JSON');out[k]=v
  return out
 def reject(x):raise ValueError('nonfinite '+x)
 out=json.loads(raw,object_pairs_hook=unique,parse_constant=reject);canonical(out);return out
def read(p,h=None):
 b=p.read_bytes()
 if h:need(sha(b)==h,'independent pin '+str(p))
 return b
def tree(root,expected):
 actual={};dirs=[];identities={}
 for base,ds,fs in os.walk(root,followlinks=False):
  for n in ds:
   p=Path(base)/n;s=p.lstat();need(stat.S_ISDIR(s.st_mode),'directory/symlink '+str(p));dirs.append(p.relative_to(root).as_posix())
  for n in fs:
   p=Path(base)/n;rel=p.relative_to(root).as_posix()
   with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
    s=os.fstat(f.fileno());need(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'regular independent custody '+rel)
    h=hashlib.sha256()
    for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    after=os.fstat(f.fileno())
   identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_nlink)
   need(identity(s)==identity(after)==identity(p.lstat()),'changed '+rel)
   identities[rel]=(s.st_dev,s.st_ino)
   if rel=='custody-inventory.json':need(h.hexdigest()==ANCHORS[rel],'seal changed');continue
   actual[rel]={'sha256':h.hexdigest(),'bytes':s.st_size}
 need(actual==expected['files'] and sorted(dirs)==expected['directories'],'whole capsule roster/hash/size mismatch '+str(root))
 return actual,identities

def main():
 seal=decode(read(LOCAL/'custody-inventory.json',ANCHORS['custody-inventory.json']))
 need(seal['identical'] is True and seal['acceptance'] is False and seal['excludes']=='custody-inventory.json itself','seal flags')
 need(set(seal['copies'])=={str(LOCAL),str(EXT)},'two exact independent roots')
 for root in (LOCAL,EXT):
  for n,h in ANCHORS.items():read(root/n,h)
 need(seal['copies'][str(LOCAL)]==seal['copies'][str(EXT)],'declared copy maps differ')
 print('Authenticated all independent metadata anchors; hashing whole copies',flush=True)
 maps=[];identities=[]
 for root in (LOCAL,EXT):
  files,ids=tree(root,seal['copies'][str(root)]);maps.append(files);identities.append(ids)
  print('Verified complete copy',str(root),len(files),sum(v['bytes'] for v in files.values()),flush=True)
 need(maps[0]==maps[1] and all(identities[0][n]!=identities[1][n] for n in identities[0]),'copies alias or differ')
 d=decode(read(LOCAL/'disposition.json'))
 first_t=decode(read(LOCAL/'inventory_transport.json'));last_t=decode(read(LOCAL/'inventory_after_transport.json'))
 need(first_t['returncode']==last_t['returncode']==0,'transport status')
 first,last=decode(first_t['stdout']),decode(last_t['stdout'])
 for obs in (first,last):need(obs['terminal_gate'] is True and not obs.get('observation_error') and obs['root_present'] and obs['failed_token_present'] and obs['common_present'] and obs['failed_present'],'full terminal roots')
 for k in ('files','directories','original_hardlink_groups','parent_accounting','related_accounting','journal_pins','upstream_pins','upstream_errors'):
  need(first[k]==last[k],'before/after changed '+k)
 rawmap={n[7:]:v for n,v in maps[0].items() if n.startswith('remote/')}
 rawdirs=[n[7:] for n in seal['copies'][str(LOCAL)]['directories'] if n.startswith('remote/')]
 need(rawmap==first['files']==d['remote_files'] and rawdirs==sorted(first['directories'])==sorted(d['remote_directories']),'complete remote files/directories')
 need(len(rawmap)==d['raw_files']==31608 and sum(v['bytes'] for v in rawmap.values())==d['raw_bytes']==336832982,'raw count/byte joins')
 need(d['errors']==d['input_pin_mismatches']==[] and d['custody_bytes_verified'] is d['current_lineage_verified'] is True and d['acceptance'] is False,'disposition agrees')
 launch=LOCAL/'remote/launch';outer=decode(read(launch/'packet.json',OUTER));inner=decode(read(launch/'inputs/packet.json',INNER))
 result=decode(read(launch/'submission.result.json',RESULT));intent=decode(read(launch/'submission.intent.json',REMOTE_INTENT))
 need(result['job_id']==JOB and result['returncode']==0 and result['ambiguous'] is False and result['intent_sha256']==REMOTE_INTENT,'original submitted job/result')
 need(intent['packet_sha256']==OUTER and intent['packet_commit']==COMMIT,'intent packet/commit')
 # Find retained original local intent by its independently supplied byte pin.
 local_intents=[n for n,v in maps[0].items() if n.startswith('coordinator/') and n.endswith('/submission.intent.json') and v['sha256']==LOCAL_INTENT]
 need(local_intents,'pinned local intent absent')
 local_intent=decode(read(LOCAL/local_intents[0],LOCAL_INTENT))
 need(set(intent)==set(local_intent)|{'remote_utc','fresh_readiness'} and all(canonical(intent[k])==canonical(v) for k,v in local_intent.items()),'remote/local semantic intent join')
 need(intent['pins']==dict(outer['new_input_files'],**{'packet.json':OUTER}),'intent input roster')
 for n,h in intent['pins'].items():need(rawmap['launch/'+n]['sha256']==h,'current original input '+n)
 for n,h in inner['files'].items():need(rawmap['launch/inputs/'+n]['sha256']==h,'inner input '+n)
 for n,h in inner['tokenizer_files'].items():need(rawmap['launch/tokenizer/'+n]['sha256']==h,'original tokenizer bytes '+n)
 need(inner['source_commit']==outer['source_commit']=='a9346022f145cc29439325b6f291597b2e6ec037','scientific source commit')
 # Byte-only full committed packet verification; no source callable executes.
 local_prefix='results/ace_llm_intervention_scientific_tokenization_20261108/launch02/'
 for n,h in intent['pins'].items():
  g=subprocess.run(['git','--no-replace-objects','show',COMMIT+':'+local_prefix+n],cwd=REPO,capture_output=True)
  need(g.returncode==0 and sha(g.stdout)==h,'committed exact packet member '+n)
 ci=decode(read(LOCAL/'collector-intent.json'))
 need(ci['job_id']==JOB and ci['packet_commit']==COMMIT and ci['local_intent_sha256']==LOCAL_INTENT and ci['submission_result_sha256']==RESULT,'collector independent invocation')
 g=subprocess.run(['git','--no-replace-objects','show',COMMIT+':'+ci['collector_repository_path']],cwd=REPO,capture_output=True)
 need(g.returncode==0 and sha(g.stdout)==ANCHORS['pull_terminal.py'],'committed collector source')
 need(ci['collector_sha256']==ANCHORS['pull_terminal.py'] and ci['collector_review_sha256']==ANCHORS['collector-review.md'],'collector review/source')
 beforebase=decode(read(launch/'token-failure-baseline.json','7a80a2300afde1ef71f8b303815b51fc2bb347515b78c40d972b4ec10138daf5'))
 rows=first['related_accounting'];prior=[r for r in rows if r[0]!=JOB];current=[r for r in rows if r[0]==JOB]
 need(len(rows)==44 and len({r[0] for r in rows})==44 and sorted(prior)==sorted(beforebase['related_accounting']),'exact43 original plus actual44th row')
 need(current==[first['parent_accounting']] and current[0][:8]==[JOB,'ace_llm_scientific_token02','COMPLETED','0:0','3','272','ucb736_asc2','billing=3,cpu=3,mem=11520M,node=1'],'actual bounded CPU allocation row')
 remote_root=outer['remote_root']
 journal=dict(beforebase['journal_pins'],**{remote_root+'/submission.intent.json':REMOTE_INTENT,remote_root+'/submission.result.json':RESULT})
 need(len(journal)==90 and journal==first['journal_pins'],'all88 original journals plus2 original')
 cpu=gpu=0
 for r in rows:
  need(r[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'},'nonterminal predecessor')
  pairs=[x.split('=',1) for x in r[7].split(',') if x];t=dict(pairs);need(len(t)==len(pairs),'TRES duplicates')
  gs={k:v for k,v in t.items() if k.startswith('gres/')};need(all(v.isdigit() and (k=='gres/gpu' or k.startswith('gres/gpu:')) for k,v in gs.items()),'unknown accelerator cost')
  typed=sum(int(v) for k,v in gs.items() if k.startswith('gres/gpu:'));n=int(gs.get('gres/gpu',typed))
  need('gres/gpu' not in gs or not any(k.startswith('gres/gpu:') for k in gs) or n==typed,'GPU charge mismatch')
  need(t or int(r[4])==int(r[5])==0,'missing allocation')
  cpu+=int(r[4])*int(r[5]);gpu+=n*int(r[5])
 need(cpu==14189==beforebase['closed_cpu_seconds']+816 and gpu==1601==beforebase['closed_gpu_slice_seconds'],'all actual costs')
 need(d['closed_cpu_seconds']==cpu and d['closed_gpu_slice_seconds']==gpu and d['cpu_admission_after_release']==17789 and d['preparation_allowance_cpu_seconds']==3600,'disposition actual costs')
 # Whole original failed capsule is sealed independently and preserved unchanged.
 failed=LOCAL/'upstream-token01-custody';fs=decode(read(failed/'custody-inventory.json','b02ecbbd83a3d9f48a3e7248fbb2512be66ffadac723a05bb2fb038465035cc9'))
 views=list(fs['copies'].values());need(fs['identical'] is True and len(views)==2 and views[0]==views[1],'failed capsule seal')
 nested={n[len('upstream-token01-custody/'):]:v for n,v in maps[0].items() if n.startswith('upstream-token01-custody/') and n!='upstream-token01-custody/custody-inventory.json'}
 nestdirs=[n[len('upstream-token01-custody/'):] for n in seal['copies'][str(LOCAL)]['directories'] if n.startswith('upstream-token01-custody/')]
 need(nested==views[0]['files'] and nestdirs==views[0]['directories'],'entire failed original capsule including archives')
 stream=decode(read(LOCAL/'stream-transport.json'));need(stream['helper_sha256']==ANCHORS['stream_custody.py'] and stream['helper_review_sha256']==ANCHORS['stream_custody_review.md'] and stream['acceptance'] is False,'stream helper/review')
 for n,h in [('capture.json',stream['capture_sha256']),('materialization.json',stream['materialization_sha256'])]:read(LOCAL/'stream-attempt'/n,h)
 need(maps[0]['stream-attempt/archive.tar']==stream['archive'],'complete retained archive identity')
 print('Full custody, original history, current source packet and costs verified',flush=True)
 return dict(status='custody_source_history_checks_passed',raw_files=31608,raw_bytes=336832982,capsule_files_excluding_seal=len(maps[0]),capsule_bytes_excluding_seal=sum(v['bytes'] for v in maps[0].values()),
  source_commit=inner['source_commit'],packet_commit=COMMIT,source_names=len(inner['source_names']),inner_inputs=len(inner['files']),terminal_jobs=44,submission_journals=90,
  closed_cpu_seconds=cpu,closed_gpu_slice_seconds=gpu,cpu_admission_seconds=cpu+3600,current_allocated_cpu_seconds=816,
  full_original_failed_token01_capsule_verified=True,artifact_runtime_lifecycle_audit_pending=True,acceptance=False,independent_input_pins=ANCHORS)

if __name__=='__main__':
 out=ROOT/'custody-attempt01';out.mkdir(mode=0o700,exist_ok=False)
 (out/'audit_custody.py').write_bytes(Path(__file__).read_bytes())
 (out/'intent.json').write_text(json.dumps(dict(anchors=ANCHORS,job_id=JOB,packet_commit=COMMIT,source_sha256=sha(Path(__file__).read_bytes()),actual_audit=True,models=False,remote=False),indent=2)+'\n')
 started=time.monotonic();cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 try:
  result=main();code=0
 except BaseException as exc:
  result=dict(status='failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),acceptance=False);code=1
 finally:
  after=resource.getrusage(resource.RUSAGE_CHILDREN)
  result.update(wall_seconds=time.monotonic()-started,checker_process_cpu_seconds=time.process_time()-cpu,git_child_cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,cost_scope='this checker plus local Git children only; no allocated job cost')
  (out/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 print(json.dumps(result,indent=2));raise SystemExit(code)
