"""One-off stdlib/Git artifact audit; does not import/execute project code."""
import ast, hashlib, json, os, pathlib, subprocess
P=pathlib.Path
REPO=P('/Users/pat/code/ACE')
BASE=REPO/'results/ace_llm_intervention_screen_training_shape_20261108/preparation01'
EXT=P('/Users/pat/ACE_Study_Results/2026-10-peter-baseline/ACE-LLM-intervention-screen-training-shape-preparation-20261108-01')
OUT=P(__file__).parent
A=BASE/'artifact'
UP=REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/terminal01/remote/launch'
COMMIT='b070fe03dcc6c3ab32fefb961e2562417aaf8dcc'
TOKEN='fb663a035eef20420fe461bc4dfa4fa6afa8462b38e32e1262c5661f666ff72c'
PACKET='0979cb607643b4b5d01fbf56c6e701a3e474d4d2bbda1e1c9f4e78e91251ccfa'
ACCEPT='69446fb29e24c34cff070d2f9f699a5d1d8d3e9a478c9d438adc01be7a290e59'
BUILDER='fcd2a905286ff190e0ac01584209066f311987470acaa502f2b08be78eec7298'
CORE={'llm_intervention_canonical_scoring.py':'795141ceba33a04e82f7fe67a0e55554979fd2464710bc72c142bf094b980169','llm_intervention_training_engine.py':'1e25ceb9b2de9b84666d99cdb6c02087000b9c2bfe6ea493d13dcff1a8fe5a49','llm_intervention_training_plan.py':'a62662b3ee323ea0ae30685391fab94b00c4e6bb057e710bd799a38317a66aee'}
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return sha(canonical(x).encode())
def unique(pairs):
 d={}
 for k,v in pairs:
  assert k not in d, ('duplicate JSON key',k)
  d[k]=v
 return d
def decode(b):return json.loads(b,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def read(p):
 assert p.is_file() and not p.is_symlink(),p
 return p.read_bytes()
def obj(p):return decode(read(p))
def git(rel):return subprocess.check_output(['git','show',COMMIT+':'+rel],cwd=REPO,env={**os.environ,'GIT_NO_REPLACE_OBJECTS':'1'})
def inventory(root):
 result={}
 assert root.is_dir() and not root.is_symlink()
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink(),p
  if p.is_file():
   data=read(p);result[str(p.relative_to(root))]={'sha256':sha(data),'bytes':len(data)}
  else:assert p.is_dir(),p
 return result
first=inventory(BASE);external=inventory(EXT);assert first==external,'repository/external roster or bytes mismatch'
for rel in first:
 assert (BASE/rel).stat().st_ino != (EXT/rel).stat().st_ino or (BASE/rel).stat().st_dev != (EXT/rel).stat().st_dev,('shared inode',rel)
(OUT/'original-inventory-attempt02.json').write_text(json.dumps(first,indent=2)+'\n')
assert sha(read(A/'builder.py'))==BUILDER and read(A/'builder.py')==read(BASE/'builder.py')
prepared=obj(A/'prepared.json');plan=prepared['training_plan'];terminal=obj(A/'terminal.json');intent=obj(A/'intent.json')
assert terminal['prepared_sha256']==sha(read(A/'prepared.json'))
assert terminal['plan_sha256']==prepared['training_plan_sha256']==digest(plan)
assert read(A/'prepared.json')==(canonical(prepared)+'\n').encode()
packet=obj(A/'capacity-packet.json');tokens=obj(A/'tokenization.json');accept=obj(A/'capacity-acceptance.json')
for file,pin,original in [('capacity-packet.json',PACKET,UP/'inputs/packet.json'),('tokenization.json',TOKEN,UP/'cpu01/worker/artifacts/tokenization.json'),('capacity-acceptance.json',ACCEPT,REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/acceptance01/acceptance.json')]:
 assert sha(read(A/file))==pin and read(A/file)==read(original),(file,'upstream mismatch')
assert accept['status']=='accepted' and accept['job_id']=='35391340' and accept['required_findings']==0
assert accept['packet_sha256']==PACKET and accept['tokenization_sha256']==TOKEN
for k in ['capacity_qualified','complete_custody_verified','runtime_qualified_for_this_cpu_execution','owned_process_cleanup_verified']:assert accept[k] is True
for k in ['scientific_qualified','training_execution_qualified']:assert accept[k] is False
assert packet['source_commit']=='21b187887ecb62bdc4b3515f3d27b79b22769d2b'
sourcepins={p.name:sha(read(p)) for p in (A/'source').iterdir()};assert len(sourcepins)==18
assert sourcepins==intent['source_sha256']==prepared['source_sha256']==terminal['source_sha256']
assert prepared['source_commit']==intent['source_commit']==COMMIT
for name,pin in sourcepins.items():assert read(A/'source'/name)==git('scripts/research/'+name),(name,'Git blob mismatch')
for name,pin in CORE.items():assert sourcepins[name]==pin
core_review='docs/development/guidance/reviews/ace_llm_intervention_unpadded_training_review_2026-11-08.md'
review_bytes=git(core_review);assert sha(review_bytes)=='0c70846b59e621ce8c04b5d925dc0fee452a243f87870ed140418f3ec4469787'
for pin in CORE.values():assert pin.encode() in review_bytes
for name in sourcepins:
 for node in ast.walk(ast.parse(read(A/'source'/name))):
  ds=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
  for dep in ds:
   if dep.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_','supervise_llm_intervention_')):assert dep+'.py' in sourcepins,(name,dep)
expected_files={'builder.py','command.json','execution.json','stdout.log','stderr.log'}|{'artifact/'+n for n in ['builder.py','capacity-acceptance.json','capacity-packet.json','tokenization.json','intent.json','terminal.json','prepared.json']}|{'artifact/source/'+n for n in sourcepins}|{'artifact/shape-'+str(i)+suffix for i in range(4) for suffix in ('.record.json','.receipt.json')}
assert set(first)==expected_files,(set(first)-expected_files,expected_files-set(first))
command=obj(BASE/'command.json');execution=obj(BASE/'execution.json')
assert command=={'command':[str(REPO/'.venv_test/bin/python'),'-B',str(BASE/'builder.py'),'--source-commit',COMMIT,'--acceptance',str(REPO/'results/ace_llm_intervention_screen_prompt_cpu_20261108/acceptance01/acceptance.json'),'--acceptance-sha256',ACCEPT,'--output',str(A)],'cwd':'/tmp'}
assert execution['exit_code']==0 and read(BASE/'stderr.log')==b'' and decode(read(BASE/'stdout.log'))==terminal
assert terminal['status']=='prepared_only' and terminal['model_loaded'] is False and terminal['training_execution_qualified'] is False
for k in ['scientific_worlds','environment_responses']:assert terminal[k]==0
for k in ['scientific_worlds','environment_responses','tokenizer_calls','inference_calls','optimizer_updates']:assert intent[k]==0
assert intent['source_commit']==COMMIT
rows={r['fixture_id']:r for r in tokens['rows']};assert len(rows)==len(tokens['rows'])==120 and all(r['status']=='fits' for r in rows.values())
manifest_raw=read(UP/'inputs/fixture-manifest.json');assert sha(manifest_raw)==packet['files']['fixture-manifest.json']
manifest=decode(manifest_raw);assert sha(manifest_raw)==packet['fixture_manifest_sha256']==accept['fixture_manifest_sha256']
assert set(rows)=={r['fixture_id'] for r in manifest['fixtures']}
LONG='collider-uniform-noise0.25-long_spelling-last_all_actions';STOP='collider-uniform-noise0.25-long_spelling-full_history_terminal'
specs=[(LONG,'observe_all',['observe_all','set_x_neg']),(LONG,'set_x_neg',['set_x_neg','observe_all']),(STOP,None,None),(LONG,'set_xy',['set_xy','observe_all'])]
# Rebuild full prompt text directly from the reviewed public format, without calling task/matched_views.
prefix=('Choose the next experimental intervention using this public SCM, paid history and prediction task. '
 'The task queries and weights describe where accurate predictions are useful. They contain no observed answers. '
 'Predict the noise-disabled structural response. Error is squared error divided by scale squared. '
 'The tolerance is a goal, not evidence that it has been achieved. Never use an unseen evaluation result to stop. '
 'Every environment response costs the listed units. Return exactly one JSON object containing an affordable '
 'action_id from state.menu, or {"stop":true}. Do not invent measurements or actions.\n')
def cmd(a):return canonical({'stop':True} if a is None else {'action_id':a})
expected_records=[];selected=[]
for i,(name,label,pair) in enumerate(specs):
 rid='shape-'+str(i);row=rows[name];eraw=read(UP/'inputs'/(name+'.json'));praw=read(UP/'inputs'/(name+'.prompt.txt'))
 assert sha(eraw)==packet['files'][name+'.json'] and sha(praw)==packet['files'][name+'.prompt.txt']
 env=decode(eraw);assert praw==(prefix+canonical(env)+'\nDecision:\n').encode()
 assert digest(env)==row['metadata']['envelope_sha256'] and digest(env['task'])==row['metadata']['task_sha256'] and sha(praw)==row['metadata']['prompt_sha256']
 receipt={'scope':'artificial','record_id':rid,'world_id':990100+i,'fixture_id':name,'envelope_sha256':digest(env),'tokenization_sha256':TOKEN,'capacity_acceptance_sha256':ACCEPT,'labels':'fabricated arithmetic qualification only','environment_responses':0}
 receipt_raw=(canonical(receipt)+'\n').encode();assert read(A/(rid+'.receipt.json'))==receipt_raw
 record={'version':'ace-matched-intervention-supervision-v1','scope':'artificial','world_id':990100+i,'envelope':env,'supervised_action_id':label,'preferences':[] if pair is None else [{'left':pair[0],'right':pair[1],'tie':False,'chosen':pair[0],'rejected':pair[1]}],'supervision_receipt_sha256':sha(receipt_raw)}
 record_raw=canonical(record).encode();assert read(A/(rid+'.record.json'))==record_raw
 state=env['state'];legal=[a['action_id'] for a in state['menu'] if not state['terminal'] and a['cost_units']<=state['remaining_units']]+[None]
 assert label in legal and (pair is None or all(a in legal for a in pair))
 assert set(row['commands'])=={cmd(a) for a in legal}
 assert len(row['prompt_ids'])==row['metadata']['prompt_tokens'] and row['metadata']['max_tokens']==8192 and row['metadata']['max_new_tokens']==32
 assert len(row['prompt_ids'])+32<=8192
 assert all(0<len(ids)<=32 and ids[-1]==151645 and all(type(x)is int and x>=0 for x in ids) for ids in row['commands'].values())
 rec={'version':'ace-bounded-training-engine-v1','record_id':rid,'scope':'artificial','raw_record_utf8':record_raw.decode(),'record_sha256':sha(record_raw),'supervision_receipt_sha256':sha(receipt_raw),'tokenizer_sha256':digest(packet['model_files']),'token_metadata':row['metadata'],'prompt_ids':row['prompt_ids'],'commands':row['commands'],'pad_token_id':151643,'eos_token_id':151645,'supervised_command':cmd(label),'strict_pairs':[] if pair is None else [[cmd(a) for a in pair]],'tied_pairs':[],'provenance_qualified':False}
 expected_records.append(rec)
 selected.append({'record_id':rid,'fixture_id':name,'envelope_raw_sha256':sha(eraw),'envelope_semantic_sha256':digest(env),'prompt_sha256':sha(praw),'record_sha256':sha(record_raw),'receipt_sha256':sha(receipt_raw),'prepared_record_semantic_sha256':digest(rec),'prompt_tokens':len(row['prompt_ids']),'supervised_command_tokens':len(rec['commands'][cmd(label)]),'pair_lengths':[] if pair is None else [len(rec['commands'][cmd(a)]) for a in pair]})
assert plan['records']==expected_records
# Independently reconstruct the complete plan rather than trusting its summary counters.
schedule=[{'stage':'sft','record_id':'shape-'+str(i),'pair_index':None} for i in range(4)]+[{'stage':'dpo','record_id':'shape-'+str(i),'pair_index':0} for i in range(2)]
shapes={f'model.layers.{i}.self_attn.{part}.{side}':shape for i in range(28) for part,width in [('q_proj',2048),('v_proj',1024)] for side,shape in [('A',[8,1024]),('B',[width,8])]}
def literal_assignment(path,name):
 for n in ast.parse(read(path)).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
 raise AssertionError(name)
gpu=literal_assignment(A/'source/qualify_llm_intervention_checkpoint_gpu.py','RECIPE');keys=literal_assignment(A/'source/llm_intervention_training_engine.py','NUMERIC_KEYS')
recipe={k:gpu[k] for k in keys};recipe.update(optimizer='AdamW',weight_decay=0.,adam_betas=[.9,.999],adam_eps=1e-8,amsgrad=False,foreach=None,fused=None,maximize=False,capturable=False,differentiable=False,clip_gradient_norm=1.,optimizer_state='continuous_across_sft_dpo',scoring_path='canonical_selected_unpadded')
exposures={'shape-'+str(i):{'sft':1,'dpo_pairs':[] if i==2 else [1 if i<2 else 0],'tied_pairs_retained':0} for i in range(4)}
expected_plan={'version':'ace-bounded-training-engine-v1','scope':'artificial','execution_kind':'qwen_screen_singletons','numeric_recipe':recipe,'records':expected_records,'record_pins':{r['record_id']:digest(r) for r in expected_records},'schedule':schedule,'expected_adapter_shapes':shapes,'max_trainable_parameters':1146880,'declared_adapter_parameters':sum(a*b for a,b in shapes.values()),'checkpoint_steps':[3,5],'outer_provenance_receipt_sha256':sha(read(A/'intent.json')),'exposures':exposures,'sft_steps':4,'dpo_steps':2,'sft_rows':4,'dpo_rows':4,'stored_tokens':sum(len(r['prompt_ids'])+sum(map(len,r['commands'].values())) for r in expected_records),'one_pass_sft':True,'one_pass_dpo':False,'provenance_qualified':False,'retry_allowed':False}
assert plan==expected_plan and digest(expected_plan)==terminal['plan_sha256']
expected_outer={'schema':'ace-screen-training-shape-preparation-v1','scope':'artificial','training_plan':expected_plan,'training_plan_sha256':digest(expected_plan),'capacity_acceptance_sha256':ACCEPT,'capacity_packet_sha256':PACKET,'tokenization_sha256':TOKEN,'tokenizer_sha256':digest(packet['model_files']),'source_sha256':sourcepins,'source_commit':COMMIT,'scientific_qualified':False,'training_execution_qualified':False,'labels':'fabricated only','environment_responses':0,'tokenizer_calls':0}
assert prepared==expected_outer and sha((canonical(expected_outer)+'\n').encode())==terminal['prepared_sha256']
assert terminal['prompt_tokens']==[x['prompt_tokens'] for x in selected] and terminal['dpo_command_lengths']==[x['pair_lengths'] for x in selected[:2]] and terminal['sft_steps']==4 and terminal['dpo_steps']==2
assert inventory(BASE)==first and inventory(EXT)==external,'originals changed during review'
report={'status':'checks_passed','required_findings':0,'method':'stdlib JSON/hash/AST and Git blob reads only; no builder/project-module/test execution','source_commit':COMMIT,'source_sha256':sourcepins,'reviewed_core_sha256':CORE,'core_review_sha256':sha(review_bytes),'builder_sha256':BUILDER,'prepared_sha256':sha(read(A/'prepared.json')),'training_plan_sha256':digest(plan),'tokenizer_sha256':digest(packet['model_files']),'capacity_acceptance_sha256':ACCEPT,'capacity_packet_sha256':PACKET,'tokenization_sha256':TOKEN,'fixture_manifest_byte_sha256':sha(manifest_raw),'fixture_manifest_semantic_sha256':digest(manifest),'all_token_rows':120,'selected_records':selected,'schedule':schedule,'exposures':exposures,'original_file_count':len(first),'original_bytes':sum(x['bytes'] for x in first.values()),'repo_external_exact_roster_and_bytes':True,'copies_distinct_inodes':True,'originals_unchanged_second_inventory':True,'original_inventory_sha256':sha(read(OUT/'original-inventory-attempt02.json')),'execution':execution,'child_terminal_resources':{k:terminal[k] for k in ('process_cpu_seconds','elapsed_seconds','peak_rss_bytes')},'all_28_command_sequences_preserved':sum(len(r['commands']) for r in expected_records)==28,'complete_prepared_object_independently_reconstructed':True,'scientific_qualified':False,'training_execution_qualified':False,'gpu_execution_qualified':False,'scope':'These saved artificial qualification input bytes only; not worker acceptance, submission authorization or scientific training'}
(OUT/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','required_findings','prepared_sha256','training_plan_sha256','original_file_count','original_bytes','repo_external_exact_roster_and_bytes','complete_prepared_object_independently_reconstructed']},indent=2))
