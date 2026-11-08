"""Read saved JSON/bytes and Git objects only; never import project code."""
from pathlib import Path
import ast, copy, hashlib, json, subprocess

R = Path('/Users/pat/code/ACE')
E = Path(__file__).parent
GROUP = R / 'results/ace_llm_intervention_scientific_contract_20261108'
OLD = GROUP / 'candidate01'
NEW = GROUP / 'candidate03'
EXT = Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-scientific-contract-20261108-candidate03')
COMMIT = 'a9346022f145cc29439325b6f291597b2e6ec037'
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_bytes())
def inventory(root):
    paths = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in paths if p.is_file()}
def git(p):
    return subprocess.check_output(['git', '--no-replace-objects', 'show', COMMIT + ':' + p], cwd=R)

old, new = read(OLD/'scientific-freeze.json'), read(NEW/'scientific-freeze.json')
assert sha((OLD/'scientific-freeze.json').read_bytes()) == '57a3e6dd27ded5cec007be235b799a2bec711fc2c5630ff16fef87d74273cd28'
assert sha((NEW/'scientific-freeze.json').read_bytes()) == '15602614ef9d7e9a3e21027e7495f891c98ba32a2b6b0ee9857e5b71d5e5e66d'
assert subprocess.check_output(['git','--no-replace-objects','cat-file','-t',COMMIT],cwd=R).strip() == b'commit'
inv = inventory(NEW)
assert inv == inventory(EXT) and len(inv) == 101
assert {k:v for k,v in inv.items() if k != 'inventory.json'} == read(NEW/'inventory.json')['files']
oldinv = inventory(OLD)
assert {k:v for k,v in oldinv.items() if k != 'inventory.json'} == read(OLD/'inventory.json')['files']
assert oldinv['inventory.json'] == '4af9daa7c51b24c1581990fbe908cd834ca6b5bfe3e3688d1fd96dea7e057df6'
assert new['source_commit'] == COMMIT
assert set(new['source_pins']) == set(old['source_pins']) and len(new['source_pins']) == 67
for p,h in new['source_pins'].items():
    b=(NEW/'source'/p).read_bytes()
    assert sha(b)==h and b==git(p)==(R/p).read_bytes()
changes={p:{'before':old['source_pins'][p],'after':h} for p,h in new['source_pins'].items() if h != old['source_pins'][p]}
expected_names={'supervise_llm_intervention_scientific_cpu.py','collect_llm_intervention_scientific_training_cpu.py','prepare_llm_intervention_scientific_campaign_cpu.py','qualify_llm_intervention_scientific_campaign_gpu.py'}
assert {Path(p).name for p in changes} == expected_names
assert changes == new['operational_recovery']['source_changes']
for p,h in new['component_reviews'].items():
    b=(NEW/'reviews'/Path(p).name).read_bytes()
    assert sha(b)==h and b==git(p)
assert read(NEW/'review-pins.json')==new['component_reviews']
review='docs/development/guidance/reviews/ace_llm_intervention_scientific_mount_alias_review_2026-11-08.md'
assert new['component_reviews']==dict(old['component_reviews'],**{review:'7964e4d6874d14b4bfa372b47ed6db36da2ad6401d85893c31e8966bf8656b48'})
guide=new['operational_recovery']['correction_guide']
assert sha((NEW/'operational-amendment.md').read_bytes())==guide['sha256']
assert (NEW/'operational-amendment.md').read_bytes()==git(guide['path'])
for name, h in [('proposal.json',old['original_proposal']['sha256']),('design.md',old['design']['sha256']),('predecessor-freeze.json',new['operational_recovery']['predecessor_freeze_sha256']),('failure-acceptance.json','5da56ce061706f29a23c9a37a898bcbac128d16421f6cfd30731655208021a8d'),('accounting.json','63cd36dfd9c0dcda4ef8ceec7b99d91ecfa696676cd1d99149256c6a0fdb3727'),('original-namespace-claim.json','58033e812390731ad08c83b7236e6e47893de3a3488cf4602d5c42809a51fa29')]:
    assert inv[name]==h
assert (NEW/'proposal.json').read_bytes()==(OLD/'proposal.json').read_bytes()
proposal=read(NEW/'proposal.json')
assert len(proposal['source_pins'])==12
assert all(new['source_pins'][p]==h for p,h in proposal['source_pins'].items())
assert sha(Path(new['operational_recovery']['original_local_registry']).read_bytes())==inv['original-namespace-claim.json']
assert new['stage_closure']==read(NEW/'closure-map.json')
assert read(OLD/'closure-map.json')==old['stage_closure']
assert new['stage_closure']['metadata_predecessor']['sha256']==oldinv['closure-map.json']
replacements={d['before']:d['after'] for d in changes.values()}
def replace(v):
    if isinstance(v,dict): return {k:replace(x) for k,x in v.items()}
    if isinstance(v,list): return [replace(x) for x in v]
    return replacements.get(v,v) if isinstance(v,str) else v
for k in ['source_pins','stages']:
    assert new['stage_closure'][k]==replace(old['stage_closure'][k])
unchanged_fields=set(old)-{'utc','source_commit','source_pins','stage_closure','component_reviews','stages','admission_accounting'}
assert set(new)==set(old)|{'operational_recovery'}
assert all(new[k]==old[k] for k in unchanged_fields)
assert len(new['stages'])==len(old['stages'])==7
for a,b in zip(old['stages'],new['stages']):
    expected=copy.deepcopy(a)
    expected['entry_sha256']=new['source_pins']['scripts/research/'+a['entry']]
    if a['stage']=='realize':
        assert a['remote_root'].endswith('_01') and a['exclusive_external_root'].endswith('-01')
        expected['remote_root']=a['remote_root'][:-2]+'02'
        expected['exclusive_external_root']=a['exclusive_external_root'][:-2]+'02'
    assert b==expected
account=copy.deepcopy(old['admission_accounting'])
account.update(baseline_sha256=inv['accounting.json'],closed_cpu_seconds=11048,max_projected_cpu_seconds=55568)
assert new['admission_accounting']==account
assert 11048+3600+sum(s['cpus']*s['wall_seconds'] for s in new['stages'])==55568
assert 1601+sum(s['gpus']*s['wall_seconds'] for s in new['stages'])==7181
prep=Path('/tmp/ace-scientific-mount-alias-preparation-current.txt').read_text().strip()
build=Path(prep)/'candidate-build02'
assert (NEW/'build_recovery_contract.py').read_bytes()==(build/'source.py').read_bytes()
assert inv['build_recovery_contract.py']=='2360f46dd61f28a410da2c9da281cdefe7baf942ff0b57a3674d9b6a50886220'
stale={}
for p in new['source_pins']:
    if Path(p).name not in new['stage_closure']['helper_metadata']: continue
    tree=ast.parse((NEW/'source'/p).read_bytes())
    actual=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    row=new['stage_closure']['helper_metadata'][Path(p).name]
    assert row['sha256']==new['source_pins'][p]
    assert row['top_level_classes']==[n.name for n in tree.body if isinstance(n,ast.ClassDef)]
    reported=row['top_level_functions']
    if actual!=reported:stale[p]={'missing_functions':[n for n in actual if n not in reported]}
assert not stale
assert (NEW/'predecessor-closure-map.json').read_bytes()==(OLD/'closure-map.json').read_bytes()
assert new['stage_closure']['predecessor_metadata_semantics']==old['stage_closure']['metadata_semantics']
old_meta=old['stage_closure']; new_meta=new['stage_closure']
allowed_meta={'source_pins','stages','helper_metadata','metadata_semantics','observed_head','git_binding_final','metadata_predecessor','source_delta','scope','status','resource_contract_status','unresolved_authorization'}
assert set(new_meta)==set(old_meta)|{'operational_amendment','predecessor_metadata_semantics'}
assert all(new_meta[k]==v for k,v in old_meta.items() if k not in allowed_meta)
assert new_meta['observed_head']==COMMIT and new_meta['git_binding_final'] is True
assert new_meta['operational_amendment']==new['operational_recovery']
assert set(new_meta['helper_metadata'])==set(old_meta['helper_metadata'])
for name,row in new_meta['helper_metadata'].items():
    prior=old_meta['helper_metadata'][name]
    assert set(row)==set(prior)
    assert all(row[k]==v for k,v in prior.items() if k not in {'sha256','top_level_functions','top_level_classes'})
assert inventory(GROUP/'candidate02')==read(E.parent/'candidate02-audit01/candidate-file-pins.json')
assert read(build/'receipt.json')['returncode']==0
command=read(build/'command.json')
assert command[command.index('--source-commit')+1]==COMMIT
assert command[command.index('--review-sha256')+1]==new['component_reviews'][review]
assert read(NEW/'inventory.json')['source_commit']==COMMIT

result=dict(schema='ace-recovery-contract-candidate-static-audit-v1',freeze_sha256=inv['scientific-freeze.json'],source_commit=COMMIT,files=101,source_files=67,review_files=len(new['component_reviews']),mirrors_equal=True,inventory_verified=True,predecessor_inventory_unchanged=True,exact_scientific_fields_unchanged=True,stage_deltas_verified=True,source_git_bytes_verified=True,source_changes=changes,proposal_and_predecessor_closure_exact=True,metadata_semantics=new['stage_closure']['metadata_semantics'],stale_function_metadata=stale,build_receipt=read(build/'receipt.json'),candidate_required_findings=0,builder_required_findings=0,accepted_for_adoption=True,scientific_qualified=False)
for name,value in [('receipt.json',result),('candidate-file-pins.json',inv),('predecessor-file-pins.json',oldinv)]:
    with (E/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(result,sort_keys=True,indent=2))
