from pathlib import Path
import ast,json,hashlib,tempfile,shutil,os
repo=Path('/Users/pat/code/ACE');base=Path('/Users/pat/ACE_Study_Results/2026-10-peter-baseline')
root=repo/'results/ace_llm_intervention_scientific_tokenization_recovery_preparation_20261108/amendment01';ext=base/'ACE-LLM-intervention-scientific-token02-operational-amendment-20261108-01'
audit=Path(tempfile.mkdtemp(prefix='ace-token02-exact-adoption-review-',dir='/tmp'))
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def load(p):
 def pairs(xs):
  d={}
  for k,v in xs:need(k not in d,'duplicate key');d[k]=v
  return d
 return json.loads(Path(p).read_bytes(),object_pairs_hook=pairs)
files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in root.rglob('*') if p.is_file()}
need(set(files)=={'operational-amendment.json','adoption.json','diagnosis.json','operational-guide.md','operational-review.md'},'exact five-file capsule')
need({str(p.relative_to(ext)) for p in ext.rglob('*') if p.is_file()}==set(files),'mirror membership')
for n,v in files.items():
 need(not (root/n).is_symlink() and not (ext/n).is_symlink() and (root/n).read_bytes()==(ext/n).read_bytes(),'mirror bytes')
 need(not os.path.samestat((root/n).stat(),(ext/n).stat()),'independent mirrors')
need(files['operational-amendment.json']['sha256']=='a77f63dab7f495bee1c00f81027d9cc6c5743b2c3c9df9341901bbe54a019c84','amendment pin')
need(files['adoption.json']['sha256']=='6c2047fe10188385571e1ba972ee09698263af30615928ab1ca93c8ce2ff7802','adoption pin')
am=load(root/'operational-amendment.json');ad=load(root/'adoption.json');pins=ad['actual_input_pins']
for n,h in ad['files'].items():need(files[n]['sha256']==h,'adoption captured dependency pin')
need(ad['actual_packet_built'] is ad['submitted'] is False and ad['exact_adoption_review_pending'] is True,'metadata-only record')
fa=repo/'results/ace_llm_intervention_scientific_tokenization_20261108/failure-acceptance01'
consumer=load(fa/'consumer-arguments.json')
for n,item in consumer.items():
 value=Path(item['path']).read_bytes();need(sha(value)==item['sha256']==pins[n],'actual independent dependency '+n)
# Join to already accepted records only; no raw custody re-audit.
a=load(fa/'acceptance.json');proof=load(fa/'no-learnedwork-proof.json');account=load(fa/'accounting.json')
need(a['status']=='accepted_failed_before_token_worker' and a['required_findings']==0 and a['complete_custody_verified'] is True,'accepted failure')
need(proof['failure_acceptance_sha256']==pins['token-failure-acceptance'] and proof['accounting_sha256']==pins['token-failure-baseline'],'proof lineage')
need(account['input_pins']['predecessor']==pins['baseline'] and account['input_pins']['acceptance']==pins['token-failure-acceptance'] and account['input_pins']['disposition']==pins['token-failure-disposition'],'baseline lineage')
need((account['terminal_jobs'],account['submission_journals'],account['closed_cpu_seconds'],account['closed_gpu_slice_seconds'],account['cpu_admission_seconds'],account['outstanding_reservation'])==(43,88,13373,1601,16973,0),'accepted accounting scalars')
old=repo/'results/ace_llm_intervention_scientific_tokenization_20261108/launch01'
for name in ('freeze','namespace-admission','namespace-claim','collection-acceptance','baseline'):
 need(sha((old/(name+'.json')).read_bytes())==pins[name],'unchanged original authority '+name)
source=Path('/tmp/ace-scientific-token02-authoring-8tbddys1/build_packet.py').read_bytes();tree=ast.parse(source)
defs={target.id:x.value for x in tree.body if isinstance(x,ast.Assign) for target in x.targets if isinstance(target,ast.Name)}
def resolve(node):
 if isinstance(node,ast.Constant):return node.value
 if isinstance(node,ast.Name):return resolve(defs[node.id])
 if isinstance(node,ast.Dict):return {resolve(k):resolve(v) for k,v in zip(node.keys,node.values)}
 if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='pins':return pins[resolve(node.slice)]
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:return {k.arg:resolve(k.value) for k in node.keywords}
 raise ValueError('nonliteral expression')
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='token_operational_binding')
expected=next(n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
contract=load('/tmp/ace-scientific-token02-authoring-8tbddys1/operational-amendment-fields.json')['exact_required_amendment_fields_with_source_expressions']
need(set(am)==set(contract) and am==resolve(expected),'exact consumer contract fields and values')
need(am=={k:resolve(ast.parse(v,mode='eval').body) for k,v in contract.items()},'description/source agreement')
need(type(am['required_findings']) is int and am['required_findings']==0,'integer finding count')
need(am['scientific_inputs_unchanged'] is am['other_stages_unchanged'] is am['one_submission_only'] is True and am['automatic_retry_allowed'] is am['replacement_worlds_allowed'] is False,'boolean authority flags')
need(am['original_deadlines']==dict(startup=120,work=550,cleanup=570,hard=590) and am['deadlines']==dict(startup=300,work=550,cleanup=570,hard=590),'exact deadline-only delta')
shutil.copytree(root,audit/'amendment');shutil.copyfile('/tmp/ace-scientific-token02-authoring-8tbddys1/operational-amendment-fields.json',audit/'consumer-fields.json')
(audit/'consumer-builder.py').write_bytes(source);shutil.copyfile(__file__,audit/'check-adoption.py')
receipt=dict(required_findings=0,scope='Exact five-file metadata adoption and consumer interface only; no raw audit, source function or test execution.',amendment_sha256=files['operational-amendment.json']['sha256'],adoption_sha256=files['adoption.json']['sha256'],
 files=files,exact_fields=len(am),consumer_builder_sha256=sha(source),consumer_description_sha256=sha((audit/'consumer-fields.json').read_bytes()),actual_acceptance_sha256=pins['token-failure-acceptance'],proof_sha256=pins['token-no-learnedwork-proof'],accounting_sha256=pins['token-failure-baseline'],design_review_sha256=pins['token-operational-review'],original_claim_and_freeze_preserved=True,tests_run=0,remote_actions=0,packet_built=False)
(audit/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'audit':str(audit),'receipt_sha256':sha((audit/'receipt.json').read_bytes()),**receipt}))
