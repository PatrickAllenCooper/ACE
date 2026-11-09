from pathlib import Path
import ast,hashlib,json,os,shutil,stat,time
ROOT=Path('/tmp/ace-downstream-consolidation-current.txt').read_text().strip();ROOT=Path(ROOT)
TMP=Path('/var/folders/rp/5fry8hzd6r79lbs65w92n8b00000gn/T')
SOURCES=[
 ('lineage/00-initial-source-capture',TMP/'ace-downstream-compact-custody-w48246u8'),
 ('lineage/01-compact-adapter',Path('/tmp/ace-downstream-compact-adapter-6r23szeq')),
 ('lineage/02-d1d2-correction',Path('/tmp/ace-downstream-compact-d1d2-1d_57cvu')),
 ('lineage/03-missing-parent-correction',Path('/tmp/ace-downstream-compact-parent-fix-j795b8og')),
 ('lineage/04-metadata-deadline',Path('/tmp/ace-downstream-metadata-deadline-98ir1xyd')),
 ('lineage/05-deadline-propagation',Path('/tmp/ace-downstream-deadline-propagation-zrnq581f')),
 ('reviews/01-initial-two-findings',TMP/'ace-downstream-compact-independent-review-xfrmbshh'),
 ('reviews/02-d1d2-closed-parent-finding',TMP/'ace-downstream-d1d2-independent-recheck-r5nry3ti'),
 ('reviews/03-parent-fix-closed',TMP/'ace-downstream-d3-independent-recheck-lp33n8e8'),
 ('reviews/04-initial-deadline-finding',TMP/'ace-downstream-metadata-deadline-review-fpt7x36a'),
 ('reviews/05-final-deadline-closed',TMP/'ace-downstream-metadata-deadline-final-review-8o1llrti'),
 ('final-implementation',Path('/tmp/ace-downstream-deadline-propagation-zrnq581f/implementation')),
 ('dependencies/backend-committed-preparation',Path('/Users/pat/code/ACE/results/ace_llm_intervention_remote_custody_20261109/preparation01/backend')),
 ('dependencies/storage-final-review',Path('/Users/pat/code/ACE/results/ace_llm_intervention_remote_custody_20261109/preparation01/final-review')),
]
def raw(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(path,data):
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 files={};dirs=[]
 for p in sorted(root.rglob('*')):
  s=p.lstat();n=p.relative_to(root).as_posix()
  if stat.S_ISLNK(s.st_mode):raise ValueError('source symlink: '+str(p))
  if stat.S_ISDIR(s.st_mode):dirs.append(n)
  elif stat.S_ISREG(s.st_mode):
   h=digest(p);after=p.lstat()
   if (s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino):raise ValueError('local source mutated')
   files[n]={'bytes':s.st_size,'sha256':h}
  else:raise ValueError('special source file: '+str(p))
 return {'files':files,'directories':dirs}
started=time.monotonic();origin=[]
try:
 for relative,source in SOURCES:
  source=source.resolve(strict=True);before=inventory(source);target=ROOT/relative;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copytree(source,target,copy_function=shutil.copy2)
  actual=inventory(target);after=inventory(source)
  if not before==actual==after:raise ValueError('copy/original inventory differs: '+str(source))
  if any(os.path.samestat((source/n).stat(),(target/n).stat()) for n in before['files']):raise ValueError('copy aliases original inode')
  origin.append({'source':str(source),'destination':relative,'source_inventory_before':before,'source_after_and_copy_equal':True,'independent_file_inodes':True})
 authority=Path('/Users/pat/code/ACE/results/ace_llm_intervention_remote_custody_20261109/preparation01/storage-authority.json')
 authority_pin='02a7a56bcd563433b356cef708d88841f5af30b4f91ddae4d3820394f9b7bbe1'
 if digest(authority)!=authority_pin:raise ValueError('authority pin differs')
 shutil.copy2(authority,ROOT/'dependencies/storage-authority.json')
 if digest(authority)!=digest(ROOT/'dependencies/storage-authority.json'):raise ValueError('authority copy changed')
 write(ROOT/'origin-inventories.json',raw(origin))
 write(ROOT/'consolidate-local.py',Path(__file__).read_bytes())
 expected=json.loads((ROOT/'lineage/05-deadline-propagation/handoff.json').read_bytes())['source_pins']
 finalpins={n:digest(ROOT/'final-implementation'/n) for n in expected}
 if finalpins!=expected:raise ValueError('final implementation does not match narrow reviewed handoff')
 write(ROOT/'final-source-pins.json',raw(finalpins))
 # Read constant argument declarations via AST only: never import/execute a coordinator.
 argumentmap={}
 for stage in ('gpu','finalizer'):
  files={}
  for name in ('build_packet.py','launch_once.py','monitor_once.py','pull_terminal.py'):
   tree=ast.parse((ROOT/'final-implementation'/stage/name).read_bytes())
   main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
   flags=[]
   if name=='build_packet.py':
    defs={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    fixed=ast.literal_eval(defs['FIXED']);prefixes=[ast.literal_eval(n.elts[0]) for n in defs['STAGES'].elts]
    first=defs['METADATA'];extras=ast.literal_eval(first.left.right);kinds=ast.literal_eval(first.right.args[0].generators[1].iter)
    names=list(fixed)+list(extras)+[prefix+'-'+kind for prefix in prefixes for kind in kinds]+list(ast.literal_eval(defs['TOKEN_METADATA']))+['token-packet']
    if len(names)!=len(set(names)):raise ValueError('duplicate metadata CLI')
    for key in names:flags.extend(['--'+key,'--'+key+'-sha256'])
   for node in main.body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='parse_args':break
    if isinstance(node,ast.For) and isinstance(node.iter,ast.Tuple) and all(isinstance(n,ast.Constant) and isinstance(n.value,str) for n in node.iter.elts):
     flags.extend('--'+n.value for n in node.iter.elts)
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='add_argument':
     arg=node.value.args[0]
     if isinstance(arg,ast.Constant):flags.append(arg.value)
   if len(flags)!=len(set(flags)):raise ValueError('duplicate flags')
   files[name]={'source_sha256':finalpins[stage+'/'+name],'required_flags':flags,'all_values':'independently pinned actual values; no supplied defaults'}
  argumentmap[stage]=files
 write(ROOT/'CLI_ARGUMENTS.json',raw(argumentmap))
 commands=[]
 for stage,files in argumentmap.items():
  for name,info in files.items():
   role='reviewed-authoring-directory' if name=='build_packet.py' else ('committed-standalone-collector-path' if name=='pull_terminal.py' else 'committed-concrete-packet-directory')
   target=f'<{stage}-{role}>'+('' if name=='pull_terminal.py' else '/'+name)
   commands.append('# '+stage+' '+name+' — TEMPLATE ONLY, not executed')
   commands.append("python3 '"+target+"' \\")
   for j,flag in enumerate(info['required_flags']):commands.append("  "+flag+" '<required:"+flag[2:]+">'"+(' \\' if j<len(info['required_flags'])-1 else ''))
   commands.append('')
 write(ROOT/'CLI_TEMPLATES.txt',('\n'.join(commands)+'\n').encode())
 authority_data=json.loads((ROOT/'dependencies/storage-authority.json').read_bytes())
 helper_matches=[];review_matches=[]
 for p in (ROOT/'dependencies').rglob('*'):
  if p.is_file():
   h=digest(p)
   if h==authority_data['backend_sha256']:helper_matches.append(str(p.relative_to(ROOT)))
   if h==authority_data['review_sha256']:review_matches.append(str(p.relative_to(ROOT)))
 if not helper_matches or not review_matches:raise ValueError('captured accepted backend/authority review bytes missing')
 write(ROOT/'dependency-pins.json',raw({'authority_sha256':authority_pin,'backend_sha256':authority_data['backend_sha256'],'backend_matching_files':helper_matches,'authority_review_sha256':authority_data['review_sha256'],'authority_review_matching_files':review_matches}))
 finalreview='reviews/05-final-deadline-closed/review.md'
 if digest(ROOT/finalreview)!='47f3accb75fa16e3b13dc0c514ea2ae22bdba4bbc8b91b0a84aa6b5fb1feb933':raise ValueError('final narrow review pin mismatch')
 write(ROOT/'consolidation.json',raw({'schema':'ace-downstream-external-preparation-v1','status':'source_lineage_consolidated','required_findings':0,'final_review':finalreview,'final_review_sha256':digest(ROOT/finalreview),'final_source_pins_sha256':digest(ROOT/'final-source-pins.json'),'origin_inventories_sha256':digest(ROOT/'origin-inventories.json'),'cli_argument_map_sha256':digest(ROOT/'CLI_ARGUMENTS.json'),'authority_sha256':authority_pin,'copy_sources':len(SOURCES),'copied_source_files':sum(len(v['source_inventory_before']['files']) for v in origin),'copied_source_payload_bytes':sum(sum(f['bytes'] for f in v['source_inventory_before']['files'].values()) for v in origin),'elapsed_seconds_before_final_documentation':time.monotonic()-started,'remote_actions':0,'locks_acquired':0,'tests_run':0,'builds_run':0,'commits':0,'scientific_execution':False,'originals_changed':False,'execution_acceptance':False}))
 print(json.dumps({'root':str(ROOT),'origins':len(SOURCES),'final_source_pins_sha256':digest(ROOT/'final-source-pins.json'),'cli_counts':{k:{n:len(v['required_flags']) for n,v in f.items()} for k,f in argumentmap.items()},'dependency_matches':{'backend':helper_matches,'review':review_matches}},sort_keys=True))
except BaseException as exc:
 write(ROOT/'consolidation-failure.json',raw({'type':type(exc).__name__,'message':str(exc),'originals_preserved':True,'retry_allowed':False}));raise
