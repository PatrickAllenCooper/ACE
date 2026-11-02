#!/usr/bin/env python3
"""Target CPU qualification of shared-task tokens and selected-action scoring.

Uses tiny random Qwen instances, never pretrained weights or scientific worlds.
Authenticates the saved GPU04 adapter inventory without deploying that model.
Real full-size adapter inference and long-context GPU demand remain separate.
"""
from __future__ import annotations
import argparse
import copy
import faulthandler
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

SOURCES = {'prepare_llm_intervention_task_cpu.py','llm_intervention_action_scoring.py',
 'llm_intervention_language_policy.py','llm_intervention_learning.py','llm_intervention_contract.py',
 'llm_intervention_grammar.py','llm_intervention_task.py','llm_intervention_scm.py',
 'llm_intervention_episode.py','llm_intervention_numerical_policy.py','llm_intervention_target_risk.py',
 'supervise_llm_intervention_gpu.py'}
PACKAGES = {'torch','transformers','numpy','scipy','safetensors','huggingface-hub','tokenizers','jinja2'}
MODELS = {'model.safetensors','config.json','generation_config.json','tokenizer.json',
          'tokenizer_config.json','merges.txt','vocab.json'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def validate_allocation(raw,job):
    from supervise_llm_intervention_gpu import scheduler_fields,memory_mib
    f=scheduler_fields(raw)
    for k,v in {'JobId':job,'JobState':'RUNNING','Requeue':'0','Restarts':'0',
                'Account':'ucb736_asc1','Partition':'acpu','QOS':'cpu-normal',
                'NumNodes':'1','NumTasks':'1','TimeLimit':'00:10:00'}.items():
        if f.get(k)!=v:raise ValueError('CPU allocation mismatch: '+k)
    def tres(text):
        pairs=[x.split('=') for x in text.split(',')]
        if any(len(x)!=2 for x in pairs) or len({x[0] for x in pairs})!=len(pairs):
            raise ValueError('invalid resource fields')
        return dict(pairs)
    allocated,requested=tres(f['AllocTRES']),tres(f['ReqTRES'])
    if (allocated.get('cpu') not in ('1','2') or f.get('NumCPUs')!=allocated['cpu']
        or f.get('CPUs/Task')!=allocated['cpu'] or allocated.get('node')!='1'
        or memory_mib(allocated.get('mem',''))!=4096 or requested.get('cpu')!='1'
        or memory_mib(requested.get('mem',''))!=4096
        or any(k.startswith('gres/') for k in set(allocated)|set(requested))):
        raise ValueError('CPU/memory/GRES allocation differs')
    return {'fields':f,'allocated':allocated,'requested':requested}

def qwen_scoring_checks():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from llm_intervention_learning import (install_adapters,pack_actions,supervised_loss,
        sequence_log_probs,snapshot_reference,reference_policy,preference_loss)
    from llm_intervention_action_scoring import (supervised_loss_selected,sequence_log_probs_selected,
                                               projection_profile)
    from llm_intervention_language_policy import restore_adapter
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device='cpu').manual_seed(203).get_state())
        config=Qwen3Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,
            attention_dropout=0.,tie_word_embeddings=False)
        config._attn_implementation='eager'
        model=Qwen3ForCausalLM(config).float()
        base=copy.deepcopy(model)
        names=install_adapters(model,rank=2,alpha=4.)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n.endswith('.B'):p.fill_(.015)
        snapshot_reference(model)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n.endswith('.B'):p.add_(.005)
    batch=pack_actions([2,3,4,5,6,7,8,9,10,11],[[12,13,1],[14,1]],0,64)
    checks={}
    for stage in ('sft','preference'):
        def objective(scorer,supervised):
            if stage=='sft':return supervised(model,batch)
            policy=scorer(model,batch).reshape(1,2)
            with reference_policy(model):reference=scorer(model,batch).reshape(1,2)
            return preference_loss(policy,reference)
        values,gradients=[],[]
        for scorer,supervised in ((sequence_log_probs,supervised_loss),
                                 (sequence_log_probs_selected,supervised_loss_selected)):
            model.zero_grad(set_to_none=True)
            value=objective(scorer,supervised);value.backward()
            values.append(float(value.detach()))
            gradients.append({n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad})
        if abs(values[0]-values[1])>2e-6:raise ValueError('Qwen loss equivalence failed')
        maxdiff=0.
        for n,g in gradients[0].items():
            torch.testing.assert_close(g,gradients[1][n],rtol=5e-4,atol=2e-6)
            maxdiff=max(maxdiff,float((g-gradients[1][n]).abs().max()))
        if not any(g.abs().sum()>0 for g in gradients[1].values()):raise ValueError('all tiny Qwen gradients zero')
        checks[stage]={'full_loss':values[0],'selected_loss':values[1],
                       'gradient_max_absolute_difference':maxdiff,'parameters_compared':len(gradients[0])}
    stream=io.BytesIO()
    torch.save({n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad},stream)
    raw=stream.getvalue()
    reload=restore_adapter(base,raw,adapter_sha256=hashlib.sha256(raw).hexdigest(),
        training_receipt_sha256='a'*64,expected_projection_names=names,rank=2,alpha=4.)
    with torch.no_grad():
        original=sequence_log_probs_selected(model,batch)
        restored=sequence_log_probs_selected(base,batch)
    torch.testing.assert_close(original,restored,rtol=0,atol=0)
    checks['reload']={'score_max_absolute_difference':float((original-restored).abs().max()),
        'projection_names':list(names),'fabricated_receipt_pin':True,'updates':0}
    checks['projection_profile']=projection_profile(batch,31)
    return checks

def run(args):
    out=Path(args.output);out.mkdir(exist_ok=False)
    started,cpu=time.monotonic(),time.process_time();number=0
    def phase(name,**extra):
        nonlocal number
        save(out/f'phase-{number:03}.json',{'phase':name,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'elapsed_seconds':time.monotonic()-started,'process_cpu_seconds':time.process_time()-cpu,**extra});number+=1
    def expired(signum,frame):raise TimeoutError('CPU qualification deadline560seconds')
    signal.signal(signal.SIGALRM,expired);signal.alarm(560)
    stack=(out/'runtime_stacks.log').open('x');faulthandler.dump_traceback_later(60,repeat=True,file=stack)
    try:
        phase('authentication_start')
        p=Path(args.freeze).resolve();source=p.parent
        if sha(p)!=args.freeze_sha256:raise ValueError('freeze differs')
        freeze=json.loads(p.read_bytes())
        if (freeze['schema']!='ace-shared-task-cpu-v1' or set(freeze['sources'])!=SOURCES
            or set(freeze['runtime'])!=PACKAGES or set(freeze['model_files'])!=MODELS
            or {x.name for x in source.glob('*.py')}!=SOURCES
            or list(sys.version_info[:3])!=freeze['python_version']
            or Path(__file__).resolve()!=source/'prepare_llm_intervention_task_cpu.py'
            or freeze['max_context_tokens']!=32768 or freeze['max_new_tokens']!=32
            or set(freeze['fixtures'])!={f'case{i}.json' for i in range(5)}):
            raise ValueError('source/runtime/protocol closure differs')
        for name,pin in freeze['sources'].items():
            if sha(source/name)!=pin:raise ValueError('source differs: '+name)
        for name,pin in freeze['fixtures'].items():
            if sha(source/name)!=pin:raise ValueError('fixture differs: '+name)
        if sha(source/'accepted_adapter.pt')!=freeze['accepted_adapter_sha256']:
            raise ValueError('accepted GPU04 adapter bytes differ')
        model_path=Path(freeze['model_path'])
        if {x.name for x in model_path.iterdir()}!=MODELS:raise ValueError('model membership differs')
        for name,pin in freeze['model_files'].items():
            if sha(model_path/name)!=pin:raise ValueError('model bytes differ: '+name)
        versions={n:importlib.metadata.version(n) for n in PACKAGES}
        if versions!=freeze['runtime']:raise ValueError('runtime versions differ')
        sys.path.insert(0,str(source))
        job=os.environ.get('SLURM_JOB_ID','')
        if not job.isdigit():raise ValueError('Slurm job identity required')
        raw=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,check=True,timeout=20).stdout
        save(out/'allocation.json',{'raw':raw,**validate_allocation(raw,job)})
        os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        phase('authentication_complete',packages=versions)
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from transformers import AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
        from llm_intervention_language_policy import prepare_task_tokens,generate_task_decision,training_action_batch
        phase('imports_complete')
        tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        model_config=json.loads((model_path/'config.json').read_bytes())
        adapter=torch.load(source/'accepted_adapter.pt',map_location='cpu',weights_only=True)
        expected={f'model.layers.{i}.self_attn.{head}_proj.{part}':shape
            for i in range(model_config['num_hidden_layers']) for head in ('q','v')
            for part,shape in [('A',(8,model_config['hidden_size'])),('B',(
                model_config['head_dim']*model_config['num_attention_heads' if head=='q' else 'num_key_value_heads'],8))]}
        if type(adapter) is not dict or set(adapter)!=set(expected):raise ValueError('actual adapter inventory differs')
        for name,shape in expected.items():
            t=adapter[name]
            if type(t)is not torch.Tensor or tuple(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():
                raise ValueError('actual adapter tensor differs: '+name)
        save(out/'accepted_adapter_inventory.json',{'sha256':freeze['accepted_adapter_sha256'],
             'tensors':len(adapter),'parameters':sum(t.numel() for t in adapter.values()),
             'actual_full_model_restoration_executed':False})
        profiles=[];envelopes=[]
        for name in sorted(freeze['fixtures']):
            envelope=json.loads((source/name).read_bytes());envelopes.append(envelope)
            grammar,meta=prepare_task_tokens(envelope,tokenizer,max_tokens=32768,max_new_tokens=32)
            batch,training_meta=training_action_batch(envelope,tokenizer,('{"stop":true}',),max_tokens=32768,max_new_tokens=32)
            if meta!=training_meta or batch.input_ids[0,:len(grammar.prompt_ids)].tolist()!=list(grammar.prompt_ids):
                raise ValueError('training and deployment prefix differ')
            profiles.append({'fixture':name,**meta})
        save(out/'task_token_profiles.json',profiles);phase('task_tokens_complete',cases=len(profiles))
        checks=qwen_scoring_checks();save(out/'tiny_qwen_scoring_checks.json',checks)
        phase('tiny_qwen_scoring_reload_complete')
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(204).get_state())
            config=Qwen3Config(vocab_size=model_config['vocab_size'],hidden_size=16,intermediate_size=32,
                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
                max_position_embeddings=32768,attention_dropout=0.,tie_word_embeddings=True)
            config._attn_implementation='eager'
            tiny=Qwen3ForCausalLM(config).float().requires_grad_(False).eval()
        generations=[]
        for i in range(2):
            completion,record=generate_task_decision(tiny,tokenizer,envelopes[i],max_tokens=32768,max_new_tokens=32)
            generations.append({'fixture':f'case{i}.json',**record})
        save(out/'tiny_qwen_generations.json',generations);phase('tiny_actual_generation_complete',decisions=2)
        save(out/'terminal.json',{'status':'complete','freeze_sha256':args.freeze_sha256,'source_pins':freeze['sources'],
            'package_versions':versions,'task_profiles':5,'tiny_qwen_generation_decisions':2,
            'accepted_adapter_sha256':freeze['accepted_adapter_sha256'],
            'tiny_scoring_reload_checks_sha256':sha(out/'tiny_qwen_scoring_checks.json'),
            'scope':'target-runtime tiny random Qwen integration, no pretrained weights or scientific utility',
            'pretrained_weight_loads':0,'optimizer_updates':0,'environment_responses':0,'scientific_worlds':0,
            'process_cpu_seconds':time.process_time()-cpu,'elapsed_seconds':time.monotonic()-started,
            'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    except BaseException as exc:
        save(out/'failure.json',{'status':'failed','exception_type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'process_cpu_seconds':time.process_time()-cpu,
            'elapsed_seconds':time.monotonic()-started});raise
    finally:
        phase('worker_function_exit');faulthandler.cancel_dump_traceback_later();stack.close();signal.alarm(0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',required=True)
    parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--output',required=True)
    run(parser.parse_args())
