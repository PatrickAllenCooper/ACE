"""Static-only supplemental source/Git joins for the independent review."""
import ast
import json
import os
import subprocess
import sys
import time
import audit as a

a.PINS.update(a.decode((a.WORK/'source-pins.json').read_bytes()))
gitpins=[]
def blob(commit,path,expected):
    env=dict(os.environ,GIT_NO_REPLACE_OBJECTS='1',GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
    r=subprocess.run(['git','--no-replace-objects','cat-file','blob',commit+':'+path],cwd=a.REPO,env=env,capture_output=True,check=False)
    a.check(r.returncode==0 and a.sha(r.stdout)==expected,'local Git blob '+commit+':'+path)
    gitpins.append(dict(commit=commit,repository_path=path,bytes=len(r.stdout),sha256=a.sha(r.stdout),replacement_objects_disabled=True,lazy_fetch_disabled=True))
    return r.stdout

intent=a.j(a.LAUNCH/'submission.intent.json')
for name,pin in intent['pins'].items():
    blob(intent['packet_commit'],str((a.LAUNCH/name).relative_to(a.REPO)),pin)

names=['qualify_llm_intervention_scientific_campaign_gpu.py','llm_intervention_model_service.py','qualify_llm_intervention_model_service.py','supervise_llm_intervention_checkpoint_gpu.py','qualify_llm_intervention_runtime_metadata.py']
stdlib_roots={'__future__','argparse','ast','contextlib','datetime','gc','hashlib','importlib','io','json','marshal','math','os','pathlib','re','resource','signal','stat','struct','subprocess','sys','tempfile','threading','time','traceback','types'}
modules={}
for name in names:
    source=a.read(a.LAUNCH/'inputs'/name)
    blob(intent['source_commit'],'scripts/research/'+name,a.sha(source))
    tree=ast.parse(source)
    imports=[];calls=[];guards=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            roots=[x.name.split('.')[0] for x in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]]
            a.check(all(r in stdlib_roots for r in roots),'stdlib top-level imports '+name+':'+str(node.lineno))
            imports.append(dict(line=node.lineno,statement=ast.unparse(node)))
        elif isinstance(node,ast.If) and ast.unparse(node.test)=="__name__ == '__main__'":
            guards.append(dict(start=node.lineno,end=node.end_lineno,statement=ast.unparse(node)))
        elif not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            for child in ast.walk(node):
                if isinstance(child,ast.Call): calls.append(dict(line=child.lineno,call=ast.unparse(child)))
    a.check(all(ast.parse(c['call'],mode='eval').body.func.id in {'tuple','range','dict','frozenset'} for c in calls),'only constant construction top-level calls '+name)
    modules[name]=dict(sha256=a.sha(source),imports=imports,top_level_calls_outside_main_guard=calls,main_guards=guards)

entry=a.read(a.LAUNCH/'inputs'/names[0]).decode().splitlines()
ranges=[(91,104),(141,151),(181,223),(281,347),(438,460),(475,486),(631,650),(653,700)]
excerpts=[dict(start=s,end=e,text='\n'.join(entry[s-1:e])) for s,e in ranges]
expected={98:'value = handle.read(maximum+1)',151:'return {str(p.relative_to(root)):sha(read(p)) for p in files}',198:"need(inventory(paths['cpu_output'],exclude=excluded)==acceptance['cpu_output_files'],",346:"bundle = admit_cpu(acceptance,packet,old,decode(data['prepared.json']))",444:'allocation_pin = save(out/\'allocation.json\',allocation(scheduler,job,helper,packet[\'resources\']))',673:"raise TimeoutError('campaign supervisor signal '+str(sig))",675:'signal.setitimer(signal.ITIMER_REAL,max(.01,args.started+240-time.monotonic()))',677:'ctx = authenticate(args.packet,args.packet_sha256,args.admission,args.admission_sha256)',681:'out.mkdir(mode=0o700,exist_ok=False)',684:'stage_and_exec(args,ctx)'}
for lineno,text in expected.items():a.check(entry[lineno-1].strip()==text,'trace/source exact line '+str(lineno))

launch=a.read(a.LAUNCH/'launch_once.py').decode().splitlines()
collector=a.read(a.TERM/'pull_terminal.py').decode().splitlines()
backend=a.read(a.TERM/'remote_custody.py').decode().splitlines()
proof=dict(schema='ace-gpu01-independent-static-source-boundary-v1',
    method='Read-only AST parsing, byte hashing and local Git blob joins; no original code imported, compiled or executed.',
    git_objects=gitpins,modules=modules,entry_excerpts=excerpts,
    exclusive_submission_source=dict(path=str(a.LAUNCH/'launch_once.py'),sha256=a.sha(a.read(a.LAUNCH/'launch_once.py')),ranges=[dict(start=124,end=143,text='\n'.join(launch[123:143])),dict(start=187,end=188,text='\n'.join(launch[186:188]))],
        finding='Exclusive intent/result and absent GPU output preconditions; xb intent write precedes the sole sbatch call; no retry loop. Exclusive here refers to submission artifacts, not a whole-node exclusive GPU allocation.'),
    collector_interface=dict(path=str(a.TERM/'pull_terminal.py'),sha256=a.sha(a.read(a.TERM/'pull_terminal.py')),start=163,end=194,text='\n'.join(collector[162:194]),
        finding='Collector accepts its own GPU allocation.json or its current scontrol result for no-requeue; it does not ingest the earlier failure-observation scontrol. Both accepted interfaces were unavailable after pre-output failure and later job removal from scontrol.'),
    backup_method=dict(path=str(a.TERM/'remote_custody.py'),sha256=a.sha(a.read(a.TERM/'remote_custody.py')),ranges=[dict(start=326,end=329,text='\n'.join(backend[325:329])),dict(start=362,end=382,text='\n'.join(backend[361:382])),dict(start=442,end=470,text='\n'.join(backend[441:470]))],
        finding='Saved source verifies every named USTAR member digest, canonical header/order, no extra/truncated members, padding and whole-archive hash. Receipt binds before, after and project archive readback. This review verifies saved records locally and performs no fresh remote archive read.'),
    failure_boundary=dict(trace=[dict(function=f,line=n) for f,n in [('main',677),('authenticate',346),('admit_cpu',198),('inventory',151),('read',98),('interrupted',673)]],
        first_cpu_inventory_incomplete=True,campaign_inventory_not_reached=True,private_campaign_bundle_not_decoded=True,training_lineage_admission_not_reached=True,
        output_creation_not_reached=True,allocation_json_creation_not_reached=True,stage_not_reached=True,controller_not_reached=True,policy_service_not_reached=True,
        model_loading_not_reached=True,new_response_generation_not_reached=True,world_realization_not_reached=True,
        alarm_seconds=240,signal=14,individual_file_or_latency_cause_known=False,
        note='Opaque adapter/input byte reads may precede failure. Those byte reads are not tensor decoding or model loading. No claim of measured GPU utilization or node-local scratch cleanup.'),
    checks=a.CHECKS,required_findings=0,original_source_executions=0,models_imported=0,tests_run=0,remote_actions=0)
(a.WORK/'source-boundary.json').write_bytes(a.canon(proof))
(a.WORK/'source-pins.json').write_bytes(a.canon(a.PINS))
print(json.dumps(dict(git_blobs_verified=len(gitpins),source_modules=len(modules),static_checks=len(a.CHECKS),required_findings=0)))
