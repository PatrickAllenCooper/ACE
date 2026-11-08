from pathlib import Path
import ast, hashlib, json, types, subprocess, os
R=Path('/Users/pat/code/ACE');S=R/'scripts/research'
OUT=Path('/tmp/ace-scientific-source-closure-20261108-01')
assert not os.path.lexists(OUT)
cache={};trees={};resolved={};bindings={};derivations=[]
def raw(x):return (json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):
    if n not in cache:
        p=S/n;assert p.is_file() and not p.is_symlink();cache[n]=p.read_bytes();trees[n]=ast.parse(cache[n],filename=str(p))
    return cache[n]
def evaluate(n,node,extras=None):
    read(n);env=dict(bindings.get(n,{}));env.update(extras or {})
    safe={'frozenset':frozenset,'set':set,'dict':dict,'tuple':tuple,'range':range,'sorted':sorted}
    bound={a.id for a in ast.walk(node) if isinstance(a,ast.Name) and isinstance(a.ctx,ast.Store)}
    for a in ast.walk(node):
        if isinstance(a,ast.Call):
            assert (isinstance(a.func,ast.Name) and a.func.id in safe) or (isinstance(a.func,ast.Attribute) and a.func.attr=='split'),ast.dump(a)
        assert not isinstance(a,(ast.Lambda,ast.NamedExpr,ast.Await,ast.Yield))
    for a in ast.walk(node):
        if isinstance(a,ast.Name) and isinstance(a.ctx,ast.Load) and a.id not in bound and a.id not in env and a.id not in safe:
            env[a.id]=constant(n,a.id)
    env.update(safe);env['__builtins__']={}
    # Only inspected constant expressions: no module bodies, function definitions or project calls.
    return eval(compile(ast.Expression(node),str(S/n),'eval'),env,{})
def constant(n,key):
    read(n)
    if (n,key) in resolved:return resolved[n,key]
    if key in bindings.get(n,{}):return bindings[n][key]
    for a in trees[n].body:
        if isinstance(a,ast.Assign):
            for target in a.targets:
                if isinstance(target,ast.Name) and target.id==key:
                    v=evaluate(n,a.value);resolved[n,key]=v;return v
                if isinstance(target,(ast.Tuple,ast.List)):
                    for i,item in enumerate(target.elts):
                        if isinstance(item,ast.Name) and item.id==key:
                            v=evaluate(n,a.value)[i];resolved[n,key]=v;return v
    raise ValueError((n,key))
def expr(n,func,var):
    read(n)
    f=next(x for x in trees[n].body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==func)
    a=next(x for x in ast.walk(f) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==var for t in x.targets))
    derivations.append(dict(source=n,function=func,variable=var,line=a.lineno,expression=ast.unparse(a.value)))
    return a.value
outer='supervise_llm_intervention_scientific_cpu.py';tok='prepare_llm_intervention_scientific_training_cpu.py'
train='supervise_llm_intervention_scientific_training_gpu.py';cpu='prepare_llm_intervention_scientific_campaign_cpu.py'
gpu='qualify_llm_intervention_scientific_campaign_gpu.py';fin='finalize_llm_intervention_scientific_campaign_cpu.py'
base='prepare_llm_intervention_campaign_cpu.py';service='llm_intervention_model_service.py';diagnostic='qualify_llm_intervention_runtime_metadata.py'
service_names=set(constant(service,'SOURCE_NAMES'))
bindings[cpu]={'B':types.SimpleNamespace(**{k:constant(base,k) for k in ('SOURCE_NAMES','NUMERIC_SOURCE_NAMES','HELPER_PINS','LLM_POLICIES','CPU_POLICIES','NUMERIC_POLICIES')})}
bindings[tok]={'service':types.SimpleNamespace(SOURCE_NAMES=service_names)}
probe_path='results/ace_llm_intervention_campaign_artificial_20261108/cpu03/inputs/probe.py'
probe=(R/probe_path).read_bytes();assert sha(probe)=='8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951'
cache['probe.py']=probe;trees['probe.py']=ast.parse(probe)
def filepin(n):return sha(read(n))
def pinmap(names):return {n:filepin(n) for n in sorted(names)}
def fixed(n,key):return {k:v for k,v in constant(n,key).items() if k.endswith('.py')}
def stage(entry,names,inputs,helpers,loaded,loader=None,notes=''):
    names=set(names);inputs=set(inputs);loaded=set(loaded);loader=loaded if loader is None else set(loader)
    assert names<=inputs and loaded<=inputs and loader<=inputs
    for n,h in helpers.items():assert n in inputs and filepin(n)==h,(entry,n,h,filepin(n))
    return dict(entrypoint=entry,entrypoint_sha256=filepin(entry),
        input_source_files=pinmap(inputs),source_names=sorted(names),
        worker_loader_source_files=pinmap(loader),loaded_work_source_roster=sorted(loaded),
        loaded_work_source_pins=pinmap(loaded),loaded_roster_kind='required by source, not observed execution',
        helpers_outside_source_names=pinmap(inputs-names),helper_pins=helpers,notes=notes)
stages={}
protocol=set(constant(outer,'PROTOCOL_NAMES'))
read(outer);f=next(n for n in trees[outer].body if isinstance(n,ast.FunctionDef) and n.name=='work_sources')
ret=next(n.value for n in f.body if isinstance(n,ast.Return))
for mode in ('realize','collect'):
    work=set(evaluate(outer,ret,{'mode':mode}));names=work|{outer};h=fixed(outer,'HELPER_PINS')
    loaded=work-{constant(outer,'COLLECT')}
    stages[mode]=stage(outer,names,names|set(h),h,loaded,notes='Outer entry runs as main; helpers load under private names. Collection entry is separately authenticated and module-loaded under _admitted_collection, excluded from captured_project_modules. Runtime requires exact work_sources(mode) minus collection entry.')
    stages[mode]['work_sources']=sorted(work)
    stages[mode]['separately_loaded_work_entries']=sorted(work-loaded)
    stages[mode]['protocol_source_pins']={'scripts/research/'+n:filepin(n) for n in sorted(protocol)}
    derivations.append(dict(source=outer,function='work_sources',mode=mode,line=f.lineno,expression=ast.unparse(ret)))
tnames=set(evaluate(tok,expr(tok,'packet_shape','sources')));th=fixed(tok,'HELPER_PINS')
stages['token']=stage(tok,tnames,tnames|set(th),th,tnames-{tok},notes='service.SOURCE_NAMES union EXTRA_SOURCES. Private old helper ENTRY is changed to current token entry in child before constructing loader; source bytes are unchanged. import_cpu_closure requires all loader sources.')
names=set(constant(train,'SOURCE_NAMES'));extra={n for n in constant(train,'EXTRA_DATA') if n.endswith('.py')};h=fixed(train,'INDEPENDENT')
h['llm_intervention_runtime_archive.py']=constant(diagnostic,'HELPER_SHA')
stages['train']=stage(train,names,names|extra,h,names-{train},notes='Exact 29-source worker roster plus archive helper input. No probe input in this stage. PREPARATION_SOURCES below is a prepared-artifact authentication map, not an additional packet roster.')
stages['train']['prepared_artifact_source_roster']=sorted(constant(train,'PREPARATION_SOURCES'))
stages['train']['prepared_artifact_source_pins']=pinmap(constant(train,'PREPARATION_SOURCES'))
cnames=set(constant(cpu,'SOURCE_NAMES'));ch=fixed(cpu,'HELPER_PINS');cinputs=cnames|set(ch)
stages['CPUcampaign']=stage(cpu,cnames,cinputs,ch,cnames-{cpu},notes='Inherited old CPU SOURCE_NAMES plus scientific entry/lineage/builder/owned helper. Old CPU entry remains in worker roster and is required by the scientific module _base() binding. No old artificial data roster is inherited.')
stages['CPUcampaign']['numeric_subprocesses']={}
for kind,numeric in constant(cpu,'NUMERIC_SOURCE_NAMES').items():
    stages['CPUcampaign']['numeric_subprocesses'][kind]=dict(input_source_files=pinmap(set(numeric)|{cpu,base}),worker_loader_source_files=pinmap(numeric),entrypoint=cpu,extra_data=['numeric-bindings.json'],notes='numeric_child captures numeric roster plus scientific and old CPU entries; its loader excludes scientific entry. Numeric ranker additionally authenticates export/receipt files from accepted CPU output.')
gnames=cnames|set(constant(gpu,'EXTRA_SOURCES'));ginputs=cinputs|set(constant(gpu,'EXTRA_SOURCES'));gh=dict(ch)
for filekey,pinkey in [('CPU_ENTRY','CPU_ENTRY_SHA'),('CPU_OWNED','CPU_OWNED_SHA'),('GPU_HELPER','GPU_HELPER_SHA'),('PROCESS_HELPER','PROCESS_HELPER_SHA'),('DIAGNOSTIC','DIAGNOSTIC_SHA'),('LINEAGE','LINEAGE_SHA')]:gh[constant(gpu,filekey)]=constant(gpu,pinkey)
gh['llm_intervention_scientific_gpu_phase.py']=constant(gpu,'PHASE_SHA');gh['llm_intervention_scientific_service_evidence.py']=constant(gpu,'EVIDENCE_SHA')
stages['GPUcampaign']=stage(gpu,gnames,ginputs,gh,gnames-{gpu},notes='Exact inherited CPU inputs/source_names plus EXTRA_SOURCES; preserve all original CPU bytes. Full controller roster imported by import_cpu_closure. Persistent model services use their separately pinned service roster.')
stages['GPUcampaign']['service_source_files']=pinmap(service_names)
fnames=gnames|{fin,constant(fin,'GPU_ENTRY')};finputs=ginputs|{fin,constant(fin,'GPU_ENTRY')};fh=dict(gh)
for name,key in [(constant(fin,'CPU_ENTRY'),'CPU_SHA'),(constant(fin,'GPU_ENTRY'),'GPU_SHA'),('llm_intervention_service_runtime.py','RUNTIME_SHA'),('llm_intervention_campaign.py','CAMPAIGN_SHA')]:fh[name]=constant(fin,key)
stages['finalizer']=stage(fin,fnames,finputs,fh,set(),loader=fnames-{constant(fin,'CPU_ENTRY')},notes='Exact inherited GPU files/source_names plus scientific finalizer and old GPU helper. Uses old CPU CapturedSources: excludes OLD CPU entry, not scientific finalizer entry. Only campaign reporting is explicitly imported; no import_cpu_closure and no exact all-source runtime membership assertion. Loaded work listed below is static explicit import, not an exhaustive observed module set. Old GPU helper is used for admission metadata only; its historical campaign execution functions are not called.')
stages['finalizer']['loaded_roster_kind']='not statically exact; actual loaded mapping must come from eventual runtime receipts'
stages['finalizer']['loaded_work_source_roster']=None;stages['finalizer']['loaded_work_source_pins']=None
stages['finalizer']['explicit_worker_project_imports']=pinmap({'llm_intervention_campaign.py'})
stages['finalizer']['privately_loaded_helpers']=pinmap({constant(fin,'CPU_ENTRY'),constant(fin,'GPU_ENTRY'),constant(base,'DIAGNOSTIC'),constant(base,'SERVICE'),constant(base,'TRAINING'),'llm_intervention_service_runtime.py'})
# Cross-stage bindings are source contracts, not executions or artifact acceptances.
collection='collect_llm_intervention_scientific_training_cpu.py'
assert {Path(n).name for n in constant(collection,'SOURCE_PATHS')}==set(stages['collect']['work_sources'])
assert protocol <= set().union(*(set(s['input_source_files']) for s in stages.values()))
union=set().union(*(set(s['input_source_files']) for s in stages.values()))
source_pins={('scripts/research/'+n if n!='probe.py' else probe_path):filepin(n) for n in sorted(union)}
proposal=json.loads((R/'results/ace_llm_intervention_screen_protocol_preparation_20261107/test-attempt01/fixtures/prospective-proposal.json').read_bytes())
assert len(proposal['source_pins'])==12 and all(source_pins[k]==v for k,v in proposal['source_pins'].items())
# Static imports are classified separately from helper-only dormant paths; never widen a contract silently.
for s in stages.values():
    missing=[]
    for n in s['source_names']:
        for a in ast.walk(trees[n]):
            deps=[a.module] if isinstance(a,ast.ImportFrom) else [x.name for x in a.names] if isinstance(a,ast.Import) else []
            for dep in deps:
                if dep and dep.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_','supervise_llm_intervention_','collect_llm_intervention_','finalize_llm_intervention_')) and dep.split('.')[0]+'.py' not in s['input_source_files']:
                    missing.append(dict(source=n,line=a.lineno,import_name=dep))
    s['lexical_imports_outside_input_roster']=missing
    s['lexical_import_note']='AST includes dormant function bodies; listed imports are not automatically runtime dependencies. No silent roster expansion.'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
doc='docs/development/guidance/ace_llm_intervention_scientific_contract_2026-11-08.md'
review='docs/development/guidance/reviews/ace_llm_intervention_scientific_contract_science_review_2026-11-08.md'
docbytes=(R/doc).read_bytes();reviewbytes=(R/review).read_bytes()
result=dict(status='prospective_source_closure_proposal_only',source_pins=source_pins,stages=stages,
    original_protocol_source_pins=proposal['source_pins'],probe=dict(repo_relative_path=probe_path,sha256=sha(probe),production_module=False,original_static_artifact=True),
    repository=str(R),observed_head=head,git_binding_final=False,
    science_contract=dict(path=doc,sha256=sha(docbytes)),science_review=dict(path=review,sha256=sha(reviewbytes),required_findings=0,companion_copy='science-review.md'),
    derivations=derivations,source_count=len(union),stage_order=list(stages),
    scope='Read-only source metadata and AST constant expressions. No project module body/function/worker/engine/main executed; no tests/runtime/model/tokenizer/scientific draws/remote actions. Source_pins is the union of exact per-stage production inputs, not all repository production files. Parent owns complete Git/source binding and future combined freeze.',
    unresolved_authorization='No concrete packet, acceptance, namespace, runtime, resource or submission authorization is created.',
    acquisition_method='AST on-demand constant-expression evaluation with whitelist of stdlib container/range/sorted constructors and literal split; no project callable invocation.',
    helper_metadata={n:{'sha256':filepin(n),'top_level_functions':[a.name for a in trees[n].body if isinstance(a,ast.FunctionDef)],'top_level_classes':[a.name for a in trees[n].body if isinstance(a,ast.ClassDef)]} for n in sorted(union)})
# Recheck snapshot bytes immediately before publishing the exclusive proposal.
for n in union:assert ((R/probe_path).read_bytes() if n=='probe.py' else (S/n).read_bytes())==cache[n]
assert (R/doc).read_bytes()==docbytes and (R/review).read_bytes()==reviewbytes
OUT.mkdir()
(OUT/'proposal.json').write_bytes(raw(result));(OUT/'science-review.md').write_bytes(reviewbytes);(OUT/'contract.md').write_bytes(docbytes)
print(json.dumps(dict(path=str(OUT/'proposal.json'),sha256=sha(raw(result)),source_count=len(union),observed_head=head,science_review_sha256=sha(reviewbytes),stages={k:dict(input_sources=len(v['input_source_files']),source_names=len(v['source_names']),loaded_work=None if v['loaded_work_source_roster'] is None else len(v['loaded_work_source_roster']),lexical_outside=v['lexical_imports_outside_input_roster']) for k,v in stages.items()}),indent=2))
