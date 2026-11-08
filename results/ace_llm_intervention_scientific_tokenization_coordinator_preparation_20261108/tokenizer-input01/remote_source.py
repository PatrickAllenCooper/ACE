import pathlib,hashlib,os,stat,sys
p=pathlib.Path('/scratch/alpine/paco0228/ACE/results/ace_llm_intervention_20261102_01/model/generation_config.json')
assert p==p.resolve() and not p.is_symlink()
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
 s=os.fstat(f.fileno());assert stat.S_ISREG(s.st_mode) and s.st_size<65536
 b=f.read(65536)
assert hashlib.sha256(b).hexdigest()=='2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2'
sys.stdout.buffer.write(b)
