from pathlib import Path
r=Path(__file__).resolve().parent;p=r/'audit_saved_gpu72.py';s=p.read_text()
s=s.replace("('before','inventory_before_transport.json')", "('before','inventory_transport.json')")
pos=s.index('def telemetry_facts(')
s=s[:pos]+'''def saved_inventory(files,directories):
 """Validate the original directory container, then make a semantic comparison view.
 Never mutate an authenticated observation or change its original digest.
 """
 need(type(directories) is list,'original directory inventory must be a list')
 need(all(type(n) is str for n in directories),'directory names must be strings')
 value=dict(files=files,directories=sorted(directories))
 aid.validate_inventory(value)
 return value

def cpu_ancestor_inventory(previous,current):
 """Exact original CPU roots/pins, with only the existing three GPU append roots."""
 mapping={'launch':'cpu','cpu':'token','training':'training','common':'common',
          'failed01':'failed01','failed-token01':'failed-token01',
          'namespace.json':'namespace.json','original-namespace.json':'original-namespace.json'}
 need(type(previous['roots']) is dict and type(current['roots']) is dict
      and set(previous['roots'])==set(mapping)
      and set(current['roots'])==set(mapping.values())|{'launch'},'exact original ancestor root labels required')
 for old,new in mapping.items():
  path=previous['roots'][old]
  need(type(path) is str and PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
       and current['roots'][new]==path,'original ancestor physical root changed: '+old)
 old=saved_inventory(previous['full_inventory']['files'],previous['full_inventory']['directories'])
 now=saved_inventory(current['full_inventory']['files'],current['full_inventory']['directories'])
 def mapped(name):
  head,sep,tail=name.partition('/')
  need(head in mapping,'ancestor member outside original root map')
  return mapping[head]+sep+tail
 files={mapped(n):v for n,v in old['files'].items()};dirs={mapped(n) for n in old['directories']}
 need(all(now['files'].get(n)==v for n,v in files.items()),'whole original ancestor/failure bytes changed')
 need(dirs<=set(now['directories']),'original ancestor directory missing')
 append={C+f'policy-{i:02d}' for i in (5,6,7)}
 def allowed(name):
  return name=='launch' or name.startswith('launch/') or any(name==a or name.startswith(a+'/') for a in append)
 need(all(allowed(n) for n in set(now['files'])-set(files))
      and all(allowed(n) for n in set(now['directories'])-dirs),'ancestor changed outside three-policy append/current launch')

''' + s[pos:]
s=s.replace('memory=[];previous=None;first_seen=None','memory=[];previous=None;first_seen=None;registered=False')
s=s.replace("  if sample is None:need(i==0 and t['registered_pid'] is None and n==0,'only pre-registration observation can omit sample')\n  else:\n   need(sample['returncode']", """  if sample is None:
   need(not registered and t['registered_pid'] is None and n==0
        and all(t[k] is None for k in ('first_verified_monotonic','last_gpu_monotonic','last_generation_monotonic'))
        and t['uncertainties']==[],'only contiguous unregistered zero-progress prefix can omit samples')
  else:
   need(t['registered_pid']==pid,'sample requires original registered process')
   registered=True
   need(sample['returncode']""")
s=s.replace('  observations=[]\n','  observations=[];inventories=[]\n',1)
s=s.replace('   observations.append(v)\n',"   inventories.append(saved_inventory(v['files'],v['directories']))\n   observations.append(v)\n",1)
s=s.replace("  for key in ('files','directories','original_hardlink_groups'", "  for key in ('files','original_hardlink_groups'",1)
s=s.replace("   need(self.before[key]==after[key],'original second observation differs: '+key)\n", "   need(self.before[key]==after[key],'original second observation differs: '+key)\n  need(inventories[0]==inventories[1],'original second directory inventory differs')\n",1)
s=s.replace("need(full==dict(files=self.before['files'],directories=sorted(self.before['directories']))", "need(full==inventories[0]")
s=s.replace("and previous_d['remote_files']==previous['full_inventory']['files'] and sorted(previous_d['remote_directories'])==previous['full_inventory']['directories']", "and saved_inventory(previous_d['remote_files'],previous_d['remote_directories'])==previous['full_inventory']")
s=s.replace("  for n,v in previous['full_inventory']['files'].items():\n   target='cpu/'+n[len('launch/'):] if n.startswith('launch/') else n\n   need(self.full.get(target)==v,'whole original ancestor/failure bytes changed')", "  cpu_ancestor_inventory(previous,self.receipt)")
p.write_text(s)
c=r/'contract.md';c.write_text(c.read_text().replace('inventory_before_transport.json','inventory_transport.json'))
