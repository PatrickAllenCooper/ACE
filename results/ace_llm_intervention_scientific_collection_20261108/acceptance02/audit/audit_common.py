"""Independent saved-custody audit. Stdlib bytes/JSON/AST/Git only; no project imports."""
import ast, hashlib, json, os, stat, subprocess, sys, time
from pathlib import Path
REPO=Path('/Users/pat/code/ACE')
REL='results/ace_llm_intervention_scientific_collection_20261108/terminal02'
LOCAL=REPO/REL
EXT=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-collection-terminal-20261108-02')
JOB='35409614'
COMMIT='75e3b8221f36a82d2ac4183354ba6a76fdc1fdd4'
SOURCE='a9346022f145cc29439325b6f291597b2e6ec037'
PACKET='a5c36ddfd72de17965bc828d870fc352d1d40dbe86b1d473c8a67fd0bea4ae90'
BASELINE='eeec362db1990d5cf1fb7945708293842dc9b9283a2777697e6840ad5a2c147b'
INTENT='a9ddf90cf92515ff9670c61c904290226b4d9e1a38a00cee50f3526aa843c87c'
FAILURE='9b34ca42142fd4ce8fb79f1a3319f2a2efde8df0bb90f5ad32d18dcf4c237d59'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(p):
 need(not p.is_symlink(),'symlink file')
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as f:
  a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode),'not regular');b=f.read();z=os.fstat(f.fileno())
 identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(identity(a)==identity(z)==identity(p.lstat()),'changed during read')
 return b

def decode(b):
 def pairs(rows):
  d={}
  for k,v in rows:need(k not in d,'duplicate key');d[k]=v
  return d
 def reject(_):raise ValueError('nonfinite')
 v=json.loads(b,object_pairs_hook=pairs,parse_constant=reject)
 json.dumps(v,allow_nan=False)
 return v

def read(p):return decode(raw(p))
def save(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def inventory(root):
 files={};dirs=[]
 need(root.is_dir() and not root.is_symlink(),'ordinary root required')
 for p in sorted(root.rglob('*')):
  need(not p.is_symlink(),'symlink in capsule')
  n=p.relative_to(root).as_posix()
  if p.is_dir():dirs.append(n)
  else:
   need(p.stat().st_nlink==1,'independent regular custody required')
   b=raw(p);files[n]={'bytes':len(b),'sha256':sha(b)}
 return dict(files=files,directories=dirs)
def normalize(rows):
 out=[];ids=set()
 for row in rows:
  r=list(row)
  while r and r[-1]=='':r.pop()
  need(len(r)==8 and r[0] not in ids and r[4].isdigit() and r[5].isdigit(),'invalid accounting row');ids.add(r[0])
  pairs=[s.split('=',1) for s in r[7].split(',') if s]
  need(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'invalid TRES')
  r[7]=','.join(k+'='+v for k,v in sorted(pairs));out.append(r)
 return sorted(out)
def git(commit,path):
 return subprocess.run(['git','--no-replace-objects','show',commit+':'+path],cwd=REPO,capture_output=True,check=True).stdout


def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return sha(canonical(v))
