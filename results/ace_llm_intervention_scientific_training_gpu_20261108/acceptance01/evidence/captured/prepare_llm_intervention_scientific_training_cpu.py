"""Accepted complete policy_train collection -> matched CPU preparation only.

Outer caller supplies independent packet/entry pins to the existing diagnostic
BOOTSTRAP, exact source_commit, accepted collection custody/freeze/namespace
bindings, six tokenizer files, and measured CPU resources/deadlines. No defaults
constitute admission. The original collection tree stays immutable; all failures
belong to a new exclusive output. No model loading, fitting or response calls.

Collection acceptance schema: ace-scientific-collection-acceptance-v1, status
accepted, required_findings=0, scope=policy_train, complete_custody_verified and
accounting_complete true, and collection exactly equal to packet['collection'].
That object contains root, files (complete original relative-path byte-pin map),
freeze_sha256, realization_acceptance_sha256, world_pins, configuration_pins,
source_pins (original protocol twelve plus collection helper), namespaces.
The independently reviewed acceptance, not this preparer, establishes collection
execution and journal accounting. No development payload is an input here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time
import traceback
import types

ENTRY = 'prepare_llm_intervention_scientific_training_cpu.py'
VERSION = 'ace-scientific-training-preparation-cpu-v1'
OLD = 'prepare_llm_intervention_training_cpu.py'
FINALIZER = 'finalize_llm_intervention_campaign_cpu.py'
CPU = 'qualify_llm_intervention_service_cpu.py'
DIAGNOSTIC = 'qualify_llm_intervention_runtime_metadata.py'
SERVICE = 'llm_intervention_model_service.py'
OWNED = 'llm_intervention_cpu_owned_supervision.py'
HELPER_PINS = {
    OWNED: '1c08423bb8fdd69ad637fbf5366978ea4a630c7ef206cc1b89ab30e3617a14e8',
    OLD: 'b2b1425bcd010d5ddc5deeda0dc3bd89817c9fe4cae6c4cc6465ec9f7f998f43',
    FINALIZER: '88102baf56395e7ef06fc7e96aae3ed2d7b8eef4ecb8b71f77acfc7e4b8e813f',
    CPU: 'd658bedd192d5c3e92f8fc22a052c67d6524df296623ef74e2c5f38bf02fbeb5',
    DIAGNOSTIC: '31bb109e2ce9603e72fe97e1cb3fcfdff2b1a72b3f7eaf74431ca92f1cd7c329',
    SERVICE: 'a7735dd35f56d7653a5ff42520db7fd67b98395e62333cccce31583317f7134e',
    'llm_intervention_service_runtime.py': '117dc22bd56889906612bff0c975095918e062974fa537757b3e93c981dbfbc0',
    'llm_intervention_runtime_archive.py': '7664772074cc84bdc9d668f5a99aae475a9e54aafc0c85486e23bf9b88ba747e',
    'probe.py': '8d6f9e2761922802712a8bf2e77df6bdec3708ce1e08272864bebbf76f0d1951',
}
PROPOSAL_BYTE_SHA = '8e3214958f671dfee5ef0a2d7adc70e8f5cdc26edf0fbeff02e7925921ff13da'
PROPOSAL_SHA = '585536b1408054ecebc7b7bb0e80055fa153b5bc7a699d8f2927db25ee745ea8'
WORLD_IDS = tuple(range(202611070000, 202611070032))
ROSTER = tuple(f'policy_train/{w}/snapshot-{i:03d}' for w in WORLD_IDS for i in range(5))
TOKENIZER_NAMES = {'config.json', 'generation_config.json', 'tokenizer_config.json',
                   'tokenizer.json', 'merges.txt', 'vocab.json'}
EXTRA_SOURCES = {ENTRY, OLD, CPU, 'llm_intervention_training_plan.py',
    'llm_intervention_training_engine.py', 'llm_intervention_training_exposure.py',
    'llm_intervention_supervision.py', 'llm_intervention_checkpointing.py',
    'qualify_llm_intervention_checkpoint_gpu.py', 'llm_intervention_gradient_probe.py', 'llm_intervention_ranker.py',
    'llm_intervention_ranker_training.py'}
PACKET_KEYS = {'schema', 'scope', 'source_commit', 'files', 'source_names', 'protocol_source_pins',
    'collection', 'archive', 'manifest', 'runtime_source', 'runtime', 'model_directory',
    'model_files', 'tokenizer_files', 'resources', 'deadlines'}
MAX_TREE_FILES, MAX_TREE_BYTES = 65536, 2*1024**3
MAX_RECORD_BYTES, MAX_BATCH_BYTES, MAX_OUTPUT_BYTES = 1024**2, 64*1024**2, 256*1024**2
FLAGS = dict(model_weights_loaded=False, explicit_cuda_api_called=False, inference_calls=0,
    environment_responses=0, training_updates=0, training_execution_qualified=False,
    scientific_qualified=False, retry_allowed=False)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return sha(canonical(value).encode())


def decode(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    def reject(x):
        raise ValueError('nonfinite JSON constant: '+x)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    canonical(value)
    return value


def read(path, expected=None, maximum=MAX_OUTPUT_BYTES):
    with os.fdopen(os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK), 'rb') as f:
        a = os.fstat(f.fileno())
        need(stat.S_ISREG(a.st_mode) and a.st_size <= maximum, 'bounded regular input required')
        raw = f.read(maximum+1); b = os.fstat(f.fileno())
    need(len(raw) <= maximum and (a.st_ino, a.st_size, a.st_mtime_ns) ==
         (b.st_ino, b.st_size, b.st_mtime_ns), 'input changed during capture')
    need(expected is None or pin(expected) and sha(raw) == expected, 'independent byte pin differs')
    return raw


def write(path, value):
    raw = value if type(value) is bytes else (canonical(value)+'\n').encode()
    need(len(raw) <= MAX_OUTPUT_BYTES, 'bounded output required')
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return sha(raw)


def load_module(name, path, raw):
    mod = types.ModuleType(name); mod.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), vars(mod))
    return mod


def safe_relative(name):
    need(type(name) is str and name and not Path(name).is_absolute()
         and all(x not in ('', '.', '..') for x in name.split('/')), 'unsafe collection path')
    return name


def separated(*paths):
    resolved = [Path(p).resolve() for p in paths]
    need(all(a != b and a not in b.parents and b not in a.parents
             for i, a in enumerate(resolved) for b in resolved[i+1:]), 'input/output trees overlap')


def tree_pins(root, expected):
    root = Path(root)
    need(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'ordinary collection root required')
    need(type(expected) is dict and 0 < len(expected) <= MAX_TREE_FILES
         and all(pin(v) for v in expected.values()), 'bounded complete collection file pins required')
    for n in expected:
        safe_relative(n)
    observed = {}; total = 0
    for p in sorted(root.rglob('*')):
        need(not p.is_symlink(), 'collection symlink forbidden')
        if p.is_dir():
            continue
        raw = read(p, maximum=64*1024**2); total += len(raw)
        need(total <= MAX_TREE_BYTES and len(observed) < MAX_TREE_FILES, 'collection custody bound exceeded')
        observed[p.relative_to(root).as_posix()] = sha(raw)
    need(observed == expected, 'complete original collection custody differs')
    return {'files': len(observed), 'bytes': total, 'files_sha256': digest(observed)}


def resource_contract(resources, deadlines):
    need(type(resources) is dict and set(resources) == {'account','partition','qos','cpus','memory_mib','wall_seconds','gpus','requeue'}
         and resources['account'] == 'ucb736_asc2' and resources['partition'] == 'acpu'
         and resources['qos'] == 'cpu-normal' and type(resources['gpus']) is int and resources['gpus'] == 0
         and resources['requeue'] is False, 'explicit CPU-only resource contract required')
    for key, low, high in [('cpus',1,64), ('memory_mib',1024,262144), ('wall_seconds',60,86400)]:
        need(type(resources[key]) is int and low <= resources[key] <= high, 'bounded caller resource required: '+key)
    need(type(deadlines) is dict and set(deadlines) == {'startup','work','cleanup','hard'}
         and all(type(v) is int for v in deadlines.values())
         and 0 < deadlines['startup'] < deadlines['work'] < deadlines['cleanup'] < deadlines['hard'] < resources['wall_seconds'],
         'ordered caller-measured absolute deadlines required')


def allocation(raw, job, resources):
    need(re.fullmatch('[1-9][0-9]*', job or '') and not os.environ.get('SLURM_JOB_GPUS')
         and not os.environ.get('SLURM_STEP_GPUS'), 'CPU Slurm allocation required')
    fields = dict(re.findall(r'([^\s=]+)=(\S+)', raw))
    expected = {'JobId':job, 'JobState':'RUNNING', 'Account':resources['account'],
        'Partition':resources['partition'], 'QOS':resources['qos'], 'NumCPUs':str(resources['cpus']),
        'NumTasks':'1', 'NumNodes':'1', 'Requeue':'0', 'Restarts':'0'}
    need(all(fields.get(k) == v for k,v in expected.items()) and fields.get('UserId','').startswith('paco0228('), 'allocation identity differs')
    def tres(value):
        rows = [v.split('=',1) for v in value.split(',')]
        need(all(len(v)==2 for v in rows) and len(dict(rows))==len(rows), 'malformed TRES')
        result = dict(rows)
        need(not any(k.startswith('gres/') for k in result), 'accelerators forbidden')
        return result
    for name in ('AllocTRES','ReqTRES'):
        r = tres(fields.get(name,'')); memory = r.get('mem','')
        m = re.fullmatch(r'(\d+(?:\.\d+)?)([MG])', memory)
        need(m is not None and float(m[1])*(1024 if m[2]=='G' else 1) == resources['memory_mib']
             and r.get('cpu') == str(resources['cpus']) and r.get('node') == '1', 'CPU/memory request or allocation differs')
    limit = fields.get('TimeLimit',''); m = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', limit)
    need(m is not None and int(m[1] or 0)*86400+int(m[2])*3600+int(m[3])*60+int(m[4]) == resources['wall_seconds'], 'wall-time limit differs')
    return fields


def packet_shape(packet, service, diagnostic):
    need(type(packet) is dict and set(packet) == PACKET_KEYS and packet['schema'] == VERSION
         and packet['scope'] == 'policy_train' and re.fullmatch('[0-9a-f]{40}',packet['source_commit']), 'scientific preparation packet differs')
    resource_contract(packet['resources'], packet['deadlines'])
    sources = set(service.SOURCE_NAMES) | EXTRA_SOURCES
    needed = sources | set(HELPER_PINS) | {'proposal.json','collection-acceptance.json'}
    need(packet['source_names'] == sorted(sources) and set(packet['files']) == needed
         and all(pin(v) for v in packet['files'].values()), 'exact captured input/source closure required')
    need(all(packet['files'][n] == h for n,h in HELPER_PINS.items())
         and packet['files']['proposal.json'] == PROPOSAL_BYTE_SHA, 'closed helper or original proposal differs')
    need(packet['runtime'] == service.RUNTIME and packet['runtime_source'] == diagnostic.RUNTIME_SOURCE, 'accepted runtime identity differs')
    for key,h in [('archive',diagnostic.ARCHIVE_SHA),('manifest',diagnostic.MANIFEST_SHA)]:
        r = packet[key]
        need(set(r) == {'path','sha256'} and r['sha256'] == h and Path(r['path']).is_absolute(), 'runtime archive/manifest differs')
    need(set(packet['model_files']) == TOKENIZER_NAMES | {'model.safetensors'}
         and all(pin(v) for v in packet['model_files'].values())
         and packet['tokenizer_files'] == {n:packet['model_files'][n] for n in TOKENIZER_NAMES}
         and Path(packet['model_directory']).is_absolute(), 'six staged tokenizer files and seven-file identity required')


def collection_binding(packet, proposal, acceptance, get):
    """Pure accepted-byte binding; get reads only paths inside the pinned tree."""
    c = packet['collection']
    need(set(c) == {'root','files','freeze_sha256','realization_acceptance_sha256','world_pins',
                   'configuration_pins','source_pins','namespaces'}, 'exact collection binding required')
    need(digest(proposal) == PROPOSAL_SHA and packet['protocol_source_pins'] == proposal['source_pins']
         and len(proposal['source_pins']) == 12, 'original proposal/protocol pins differ')
    need(tuple(w['world_id'] for w in proposal['training_worlds']) == WORLD_IDS, 'complete ordered policy_train worlds required')
    need(acceptance.get('schema') == 'ace-scientific-collection-acceptance-v1'
         and acceptance.get('status') == 'accepted' and type(acceptance.get('required_findings')) is int
         and acceptance['required_findings'] == 0 and acceptance.get('scope') == 'policy_train'
         and acceptance.get('complete_custody_verified') is True and acceptance.get('accounting_complete') is True
         and acceptance.get('collection') == c, 'independent complete collection acceptance required')
    need(pin(c['freeze_sha256']) and pin(c['realization_acceptance_sha256']), 'original freeze/realization acceptance required')
    keys = {str(w) for w in WORLD_IDS}
    need(set(c['world_pins']) == set(c['configuration_pins']) == keys
         and all(pin(v) for m in (c['world_pins'],c['configuration_pins']) for v in m.values())
         and c['namespaces'] == {str(w['world_id']):w['training']['request_namespace'] for w in proposal['training_worlds']},
         'descriptor/configuration/namespace membership differs')
    need(set(c['source_pins']) == set(proposal['source_pins']) | {'scripts/research/llm_intervention_training_collection.py'}
         and all(c['source_pins'][k] == v for k,v in proposal['source_pins'].items())
         and all(pin(v) for v in c['source_pins'].values()), 'original collection source contract differs')
    data = {n:get(n) for n in ('terminal.json','inventory.json','plan.json','intent.json')}
    term, inv, plan, intent = [decode(data[n]) for n in ('terminal.json','inventory.json','plan.json','intent.json')]
    need(intent['proposal_sha256'] == PROPOSAL_SHA and intent['expected_source_pins'] == c['source_pins']
         and intent['world_pins'] == c['world_pins'] and intent['configuration_pins'] == c['configuration_pins']
         and intent['world_roster'] == list(WORLD_IDS) and intent['expected_roster_sha256'] == digest(WORLD_IDS)
         and plan['intent_sha256'] == sha(data['intent.json']) and plan['source_file_sha256'] == c['source_pins'], 'original collection intent/source/freeze lineage differs')
    need(term['status'] == 'complete' and term['exposure_ready'] is True and term['accounting_complete'] is True
         and term['scope'] == inv['scope'] == 'policy_train' and term['record_roster'] == inv['record_roster'] == list(ROSTER)
         and term['inventory_sha256'] == sha(data['inventory.json']) and inv['collection_plan_sha256'] == sha(data['plan.json'])
         and term['world_roster'] == plan['world_roster'] == list(WORLD_IDS)
         and term['planned_records'] == plan['records'] and len(inv['records']) == 160, 'complete collection identity differs')
    for k in ('partial_records_authorized_for_training','complete_case_filtering_allowed','policy_training_performed','scientific_evaluation_performed'):
        need(term[k] is False, 'collection cannot already be filtered/trained/evaluated')
    account = term['accounting']; worlds = term['worlds']
    need([w['world_id'] for w in worlds] == list(WORLD_IDS) and all(w['status']=='complete' for w in worlds), 'incomplete world accounting')
    fields = ('reserved_response_attempts','known_returns','executor_accepted_responses','unknown_return_attempts','total_reserved_cost_units')
    for w in worlds:
        a = w['accounting']
        need(a['accounting_complete'] is True and a['accounting_errors'] == [] and a['unknown_return_attempts'] == 0
             and a['reserved_response_attempts'] == a['known_returns'] == a['executor_accepted_responses'], 'unknown/incomplete world returns')
    need(all(type(account[k]) is int and account[k] >= 0
             and all(type(w['accounting'][k]) is int and w['accounting'][k]>=0 for w in worlds)
             and account[k] == sum(w['accounting'][k] for w in worlds) for k in fields)
         and account['unknown_return_attempts'] == 0
         and account['reserved_response_attempts'] == account['known_returns'] == account['executor_accepted_responses']
         and 0 < account['reserved_response_attempts'] <= 9216
         and 0 < account['total_reserved_cost_units'] <= 10240
         and account['reserved_response_attempts'] <= plan['maximum_response_attempts']
         and account['total_reserved_cost_units'] <= plan['maximum_reserved_cost_units'], 'actual accounting not reconciled')
    records, receipts = {}, {}; retained = 0; size = 0
    need(set(inv['record_pins']) == set(inv['supervision_receipt_pins']) == set(ROSTER), 'complete record/receipt pins required')
    need(len(plan['records']) == 160, 'complete planned records required')
    for i,(rid,planned,row) in enumerate(zip(ROSTER,plan['records'],inv['records'])):
        wid,si = WORLD_IDS[i//5],i%5
        need(planned['record_id'] == row['record_id'] == rid and row['status']=='complete'
             and all(row.get(k)==v for k,v in planned.items()) and planned['world_id']==wid
             and planned['snapshot_index']==si and planned['scope']=='policy_train', 'original ordered row differs')
        path = safe_relative(row['path'])
        records[rid],receipts[rid] = get(path+'/record.json'),get(path+'/receipt.json')
        size += len(records[rid])+len(receipts[rid])
        need(len(records[rid]) <= MAX_RECORD_BYTES and size <= MAX_BATCH_BYTES, 'complete batch byte bound exceeded')
        need(sha(records[rid]) == inv['record_pins'][rid] == row['record_sha256']
             and sha(receipts[rid]) == inv['supervision_receipt_pins'][rid] == row['supervision_receipt_sha256'], 'original row byte pins differ')
        rec, receipt = decode(records[rid]),decode(receipts[rid]); env = rec['envelope']
        candidates = proposal['training_worlds'][i//5]['training']['snapshots'][si]['candidates']
        pairs = [[a,b] for j,a in enumerate(candidates) for b in candidates[j+1:]]
        need(len(candidates) == (8 if si<4 else 0) and planned['candidate_ids']==receipt['candidate_ids']==candidates
             and planned['pairs']==[[p['left'],p['right']] for p in rec['preferences']]==pairs
             and len(pairs)==(28 if si<4 else 0) and receipt['pair_count']==len(pairs)
             and rec['world_id']==receipt['world_id']==wid and rec['scope']==receipt['scope']=='policy_train'
             and receipt['snapshot_index']==si and receipt['task_sha256']==digest(env['task'])
             and receipt['envelope_file_sha256']==sha((canonical(env)+'\n').encode())
             and rec['supervision_receipt_sha256']==sha(receipts[rid]), 'matched pair/state/receipt association differs')
        need(si<4 or rec['supervised_action_id'] is None and env['state']['terminal'] is True, 'terminal stop record required')
        retained += len(pairs)
    need(retained==3584, 'all 3584 retained comparisons required')
    return dict(data=data,records=records,receipts=receipts,record_pins=inv['record_pins'],
                receipt_pins=inv['supervision_receipt_pins'],accounting=account)


def tokenizer_inventory(packet):
    root = Path(packet['model_directory'])
    need(root.is_dir() and not root.is_symlink() and {p.name for p in root.iterdir()} == TOKENIZER_NAMES,
         'exact six-file tokenizer-only directory required; weights/adapters forbidden')
    return {n:{'sha256':sha(read(root/n,p,32*1024**2)),'bytes':(root/n).stat().st_size}
            for n,p in packet['tokenizer_files'].items()}


def prepare_pipeline(packet, batch, packet_pin, journal, *, exposure_api, bridge_api,
                     ranker_api, tokenizer_factory):
    """Glue seam for fabricated tests; production supplies captured closed APIs."""
    exposure = exposure_api(records=batch['records'],record_pins=batch['record_pins'],
        supervision_receipt_pins=batch['receipt_pins'],record_roster=ROSTER,scope='policy_train')
    journal.save('exposure.json',exposure)
    c = exposure['counts']; d = c['dpo_steps']
    need(c['records']==160 and c['retained_pairs']==3584 and c['sft_steps']==160
         and type(d) is int and 0<=d<=128, 'fixed complete exposure counts differ')
    need(d>0 and exposure['has_both_training_stages'] is True, 'D=0: retain failure; never recollect or change recipe')
    checkpoints = (159,159+d)
    journal.event('cpu_tokenizer_load_start')
    tokenizer = tokenizer_factory()
    need(tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643, 'accepted tokenizer special IDs differ')
    result = bridge_api(terminal_raw=batch['data']['terminal.json'],terminal_pin=sha(batch['data']['terminal.json']),
        inventory_raw=batch['data']['inventory.json'],inventory_pin=sha(batch['data']['inventory.json']),
        collection_plan_raw=batch['data']['plan.json'],collection_plan_pin=sha(batch['data']['plan.json']),
        records=batch['records'],receipts=batch['receipts'],expected_roster=ROSTER,expected_scope='policy_train',
        tokenizer=tokenizer,tokenizer_pin=digest(packet['model_files']),outer_provenance_pin=packet_pin,
        checkpoint_steps=checkpoints,execution_kind='qwen_screen_singletons')
    journal.save('llm-prepared.candidate.json',result)
    need(result['exposure']==exposure and result['exposure_sha256']==digest(exposure), 'bridge recomputed a different exposure')
    numerical = ranker_api.prepare_plan(records=batch['records'],record_pins=batch['record_pins'],
        supervision_receipt_pins=batch['receipt_pins'],record_roster=ROSTER,scope='policy_train',
        exposure=exposure,exposure_sha256=digest(exposure),recipe=ranker_api.numeric_recipe(artificial=False),
        checkpoint_steps=checkpoints,outer_provenance_receipt_sha256=packet_pin)
    journal.save('numerical-plan.candidate.json',numerical)
    validate_plans(result,numerical,exposure,packet,batch,packet_pin)
    journal.save('llm-prepared.json',result); journal.save('numerical-plan.json',numerical)
    journal.event('cpu_tokenization_complete',records=160,sft_steps=160,dpo_steps=d)
    return {'records':160,'retained_pairs':3584,'sft_steps':160,'dpo_steps':d,
        'training_plan_sha256':result['training_plan_sha256'],'numerical_plan_sha256':digest(numerical),
        'exposure_sha256':digest(exposure)}


def validate_plans(llm, numerical, exposure, packet, batch, packet_pin):
    plan = llm['training_plan']; d = exposure['counts']['dpo_steps']
    need(0<d<=128 and llm['training_plan_sha256']==digest(plan) and llm['exposure']==exposure
         and llm['exposure_sha256']==digest(exposure) and plan['scope']=='policy_train'
         and plan['execution_kind']=='qwen_screen_singletons' and plan['numeric_recipe']['scoring_path']=='canonical_selected_unpadded'
         and plan['schedule']==exposure['schedule'] and plan['sft_steps']==160 and plan['dpo_steps']==d
         and plan['checkpoint_steps']==[159,159+d] and plan['outer_provenance_receipt_sha256']==packet_pin
         and plan['declared_adapter_parameters']==plan['max_trainable_parameters']==1146880
         and plan['expected_adapter_shapes']=={f'model.layers.{i}.self_attn.{part}.{side}':shape
             for i in range(28) for part,width in (('q_proj',2048),('v_proj',1024))
             for side,shape in (('A',[8,1024]),('B',[width,8]))}
         and plan['sft_rows']==160 and plan['dpo_rows']==2*d
         and plan['numeric_recipe']['max_tokens']==8192 and plan['numeric_recipe']['max_new_tokens']==32
         and [r['record_id'] for r in plan['records']]==list(ROSTER), 'LLM plan/exposure identity differs')
    count=0
    for r in plan['records']:
        rid=r['record_id']; prompt=r['prompt_ids']; commands=r['commands']; meta=r['token_metadata']
        need(r['raw_record_utf8'].encode()==batch['records'][rid] and r['record_sha256']==batch['record_pins'][rid]
             and r['supervision_receipt_sha256']==batch['receipt_pins'][rid] and plan['record_pins'][rid]==digest(r)
             and r['tokenizer_sha256']==llm['tokenizer_sha256']==digest(packet['model_files'])
             and r['pad_token_id']==151643 and r['eos_token_id']==151645, 'prepared record/tokenizer pin differs')
        need(prompt and len(prompt)+32<=8192 and all(type(x)is int and x>=0 for x in prompt)
             and commands and all(ids and len(ids)<=32 and ids[-1]==151645 and all(type(x)is int and x>=0 for x in ids) for ids in commands.values())
             and meta['prompt_tokens']==len(prompt) and meta['max_tokens']==8192 and meta['max_new_tokens']==32,
             'untruncated complete token sequences required')
        count+=len(prompt)+sum(map(len,commands.values()))
    need(plan['stored_tokens']==count<=4194304 and llm['new_environment_responses']==llm['training_updates']==0
         and llm['scientific_qualified'] is llm['outer_execution_qualified'] is plan['provenance_qualified'] is False,
         'plan token count or qualification differs')
    need(numerical['scope']=='policy_train' and numerical['record_roster']==list(ROSTER)
         and numerical['raw_records_utf8']=={k:v.decode() for k,v in batch['records'].items()}
         and numerical['record_pins']==batch['record_pins'] and numerical['supervision_receipt_pins']==batch['receipt_pins']
         and numerical['exposure']==exposure and numerical['exposure_sha256']==digest(exposure)
         and numerical['checkpoint_steps']==[159,159+d] and numerical['outer_provenance_receipt_sha256']==packet_pin
         and numerical['recipe']['artificial_control'] is False and numerical['recipe']['hidden']==128
         and numerical['recipe']['device']=='cpu' and numerical['recipe']['dtype']=='float32'
         and numerical['recipe']['seed']==109 and numerical['recipe']['optimizer']=='Adam'
         and numerical['recipe']['learning_rate']==.001 and numerical['recipe']['beta']==.1
         and numerical['recipe']['optimizer_state']=='continuous_across_sft_dpo'
         and numerical['scientific_provenance_qualified'] is False, 'matched numerical plan differs')


def authenticate(args):
    root=Path(args.packet).absolute().parent
    raw=read(args.packet,args.packet_sha256,16*1024**2); packet=decode(raw)
    need(type(packet.get('files')) is dict and len(packet['files'])<=128
         and all(type(n)is str and Path(n).name==n and n.endswith(('.json','.py')) for n in packet['files']), 'flat bounded input roster required')
    data={n:read(root/n,h,16*1024**2) for n,h in packet['files'].items()}
    for n,h in HELPER_PINS.items():need(sha(data[n])==h,'closed helper pin differs: '+n)
    mods={n:load_module('_scientific_'+str(i),root/n,data[n]) for i,n in enumerate((OLD,FINALIZER,CPU,DIAGNOSTIC,SERVICE))}
    b,f,cpu,d,service=[mods[n] for n in (OLD,FINALIZER,CPU,DIAGNOSTIC,SERVICE)]
    packet_shape(packet,service,d)
    need(packet['source_commit']==args.source_commit and Path(__file__).absolute()==root/ENTRY,'entry/source commit identity differs')
    sources={n:data[n] for n in packet['source_names']}
    for name,value in sources.items():
        for node in ast.walk(ast.parse(value)):
            deps=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            need(all(dep+'.py' in sources for dep in deps if dep.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_'))), 'captured project closure incomplete')
    separated(root,packet['collection']['root'],packet['model_directory'],args.output)
    tree=tree_pins(packet['collection']['root'],packet['collection']['files'])
    def get(name):
        safe_relative(name)
        return read(Path(packet['collection']['root'])/name,packet['collection']['files'][name],64*1024**2)
    batch=collection_binding(packet,decode(data['proposal.json']),decode(data['collection-acceptance.json']),get)
    tokens=tokenizer_inventory(packet)
    return dict(root=root,packet=packet,data=data,base=b,finalizer=f,cpu=cpu,diagnostic=d,service=service,batch=batch,tree=tree,tokens=tokens,
                packet_path=Path(args.packet).absolute(),packet_sha256=args.packet_sha256)


def recheck(ctx):
    p=ctx['packet']
    read(ctx['packet_path'],ctx['packet_sha256'],16*1024**2)
    for n,h in p['files'].items():read(ctx['root']/n,h,16*1024**2)
    need(tree_pins(p['collection']['root'],p['collection']['files'])==ctx['tree'], 'collection changed after preparation')
    need(tokenizer_inventory(p)==ctx['tokens'], 'staged tokenizer files changed')


def publish(out, value):
    write(out/'terminal-candidate.json',value)
    os.link(out/'terminal-candidate.json',out/'terminal.json')


def failure(out, exc):
    name='publication-interruption.json' if (out/'terminal.json').exists() else 'failure.json'
    if not (out/name).exists():
        write(out/name,dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),**FLAGS))


def child(args,ctx):
    b,p,root=ctx['base'],ctx['packet'],ctx['root']; out=Path(args.output)
    b.ENTRY=ENTRY  # Configure only this isolated verified helper object, not its source.
    captured={n:ctx['data'][n] for n in p['source_names']}
    need(not any(n.startswith(('llm_intervention_','qualify_llm_intervention_','prepare_llm_intervention_')) for n in sys.modules), 'project imports preceded captured worker')
    class Loader(b.CapturedSources):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.startswith(('prepare_llm_intervention_','collect_llm_intervention_')) and fullname not in self.sources:
                raise ImportError('uncaptured preparation module: '+fullname)
            return super().find_spec(fullname,path,target)
    loader=Loader(captured,root);sys.meta_path.insert(0,loader)
    journal=ctx['service'].Journal(out/'artifacts')
    try:
        from llm_intervention_service_runtime import ServiceRuntime
        observer=ServiceRuntime(p,captured,loader,root/ENTRY,globals(),args.runtime_stage,
            args.runtime_stage_sha256,journal,ctx['service'].read_pinned,ctx['service'].decode)
        journal.save('authenticated.json',dict(packet_sha256=args.packet_sha256,source_sha256={n:p['files'][n] for n in p['source_names']},
            tokenizer_files=ctx['tokens'],tokenizer_sha256=digest(p['model_files']),collection=ctx['tree']))
        original=journal.output/'originals';original.mkdir()
        for n,raw in ctx['batch']['data'].items():write(original/n,raw)
        for i,rid in enumerate(ROSTER):
            write(original/f'record-{i:03d}.json',ctx['batch']['records'][rid]);write(original/f'receipt-{i:03d}.json',ctx['batch']['receipts'][rid])
        observer.import_cpu_closure()
        write(out/'work-start.json',dict(packet_sha256=args.packet_sha256,pid=os.getpid(),monotonic=time.monotonic()))
        from transformers import AutoTokenizer
        from llm_intervention_training_exposure import propose_exposure
        from llm_intervention_training_plan import prepare_complete_collection
        import llm_intervention_ranker_training as ranker
        summary=prepare_pipeline(p,ctx['batch'],args.packet_sha256,journal,exposure_api=propose_exposure,
            bridge_api=prepare_complete_collection,ranker_api=ranker,
            tokenizer_factory=lambda:AutoTokenizer.from_pretrained(p['model_directory'],local_files_only=True,trust_remote_code=False))
        recheck(ctx);observer.observe('after-cpu-tokenization')
        publish(journal.output,dict(schema=VERSION,status='cpu_preparation_complete',packet_sha256=args.packet_sha256,
            runtime_stage_sha256=args.runtime_stage_sha256,**summary,
            llm_prepared_file_sha256=sha(read(journal.output/'llm-prepared.json')),
            numerical_plan_file_sha256=sha(read(journal.output/'numerical-plan.json')),**FLAGS))
    except BaseException as exc:
        failure(journal.output,exc);raise
    finally:
        sys.meta_path.remove(loader)


def validate_worker(ctx,args,stage,stage_pin,provenance,checks):
    out=Path(args.output)/'session/worker/artifacts';p=ctx['packet']
    def get(n):return decode(read(out/n))
    need(not (out/'failure.json').exists() and not (out/'publication-interruption.json').exists(), 'failed worker cannot publish success')
    llm,num,exposure=get('llm-prepared.json'),get('numerical-plan.json'),get('exposure.json')
    validate_plans(llm,num,exposure,p,ctx['batch'],args.packet_sha256)
    for name in ('llm-prepared','numerical-plan'):
        need(read(out/(name+'.json'))==read(out/(name+'.candidate.json')), 'candidate/success bytes differ')
    a=get('authenticated.json')
    need(a==dict(packet_sha256=args.packet_sha256,source_sha256={n:p['files'][n] for n in p['source_names']},
        tokenizer_files=ctx['tokens'],tokenizer_sha256=digest(p['model_files']),collection=ctx['tree']), 'worker input authentication differs')
    ctx['cpu'].validate_runtime_records(get,stage,stage_pin,ctx['service'],ctx['diagnostic'],checks,provenance)
    need(get('runtime-stage-binding.json')['gpu_mapping_enabled'] is False,'GPU mapping exemption forbidden')
    for phase in ('after-service-imports','after-cpu-tokenization'):
        projects=get('runtime-'+phase+'.json')['captured_project_modules']
        expected={Path(n).stem:p['files'][n] for n in p['source_names'] if n!=ENTRY}
        need(set(expected)<=set(projects)<=set(expected)|{'__main__','__mp_main__'}
             and all(projects[n]['sha256']==h for n,h in expected.items()), 'captured namespace/source pins differ')
    ctx['base'].validate_phases([get(n.name) for n in sorted(out.glob('phase-*.json'))])
    t=get('terminal.json')
    need(t['schema']==VERSION and t['status']=='cpu_preparation_complete' and t['packet_sha256']==args.packet_sha256
         and t['runtime_stage_sha256']==stage_pin and all(t[k]==v for k,v in FLAGS.items())
         and t['training_plan_sha256']==llm['training_plan_sha256'] and t['numerical_plan_sha256']==digest(num)
         and t['exposure_sha256']==digest(exposure) and t['records']==160 and t['retained_pairs']==3584
         and t['sft_steps']==160 and t['dpo_steps']==exposure['counts']['dpo_steps']
         and t['llm_prepared_file_sha256']==sha(read(out/'llm-prepared.json'))
         and t['numerical_plan_file_sha256']==sha(read(out/'numerical-plan.json')), 'worker terminal differs')
    return sha(read(out/'terminal.json'))


def parent(args,ctx):
    p,d,b,out=ctx['packet'],ctx['diagnostic'],ctx['base'],Path(args.output)
    ownership=load_module('_scientific_cpu_owned',ctx['root']/OWNED,ctx['data'][OWNED])
    args.owned_child=ownership.OwnedChild(args.started+p['deadlines']['hard'])
    write(out/'packet.json',p)
    job=os.environ.get('SLURM_JOB_ID','')
    q=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
    write(out/'allocation.json',dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr))
    need(q.returncode==0,'allocation query failed');allocation(q.stdout,job,p['resources'])
    scratch=Path(os.environ.get('SLURM_SCRATCH',''))
    need(scratch.is_absolute(),'absolute node scratch required');scratch=scratch.resolve(strict=True)
    need(scratch.is_dir() and scratch.stat().st_uid==os.getuid(),'owned node scratch required')
    local=scratch/('ace-scientific-preparation-'+job);local.mkdir(mode=0o700,exist_ok=False)
    archive=load_module('_scientific_archive',ctx['root']/'llm_intervention_runtime_archive.py',ctx['data']['llm_intervention_runtime_archive.py'])
    probe=load_module('_scientific_cleanup',ctx['root']/'probe.py',ctx['data']['probe.py'])
    checks=load_module('_scientific_runtime',ctx['root']/'llm_intervention_service_runtime.py',ctx['data']['llm_intervention_service_runtime.py'])
    event=d.events(out,args.started,args.process_started,512*1024**2);event('stage_start')
    stage=d.stage_runtime(p,archive,local/'runtime',event);stage_pin=write(out/'stage.json',stage)
    prefix=Path(stage['staged_prefix']);event('stage_complete',staged_prefix=str(prefix))
    provenance=d.provenance_contract(prefix);write(out/'independent-provenance-sources.json',provenance)
    session=out/'session';session.mkdir(mode=0o700,exist_ok=False)
    command=[str(prefix/'bin/python'),'-I','-B','-c',d.BOOTSTRAP,str(ctx['root']/ENTRY),p['files'][ENTRY],
        '--packet',str(Path(args.packet).absolute()),'--packet-sha256',args.packet_sha256,
        '--source-commit',args.source_commit,'--output',str(session/'worker'),'--worker',
        '--runtime-stage',str(out/'stage.json'),'--runtime-stage-sha256',stage_pin,
        '--parent-started',str(args.started)]
    session_result=ownership.supervise(command,b.cpu_environment(prefix,os.environ),session,
        started=args.started,startup_deadline=args.started+p['deadlines']['startup'],
        work_deadline=args.started+p['deadlines']['work'],packet_sha256=args.packet_sha256,
        owned=args.owned_child,kill_owned=probe.kill_owned)
    cleanup=session_result['cleanup']
    terminal_pin=validate_worker(ctx,args,stage,stage_pin,provenance,checks)
    recheck(ctx)
    need(time.monotonic()<args.started+p['deadlines']['hard'],'hard deadline exhausted before publication')
    publish(out,dict(schema=VERSION,status='cpu_preparation_complete',packet_sha256=args.packet_sha256,
        worker_terminal_sha256=terminal_pin,stage_sha256=stage_pin,cleanup=cleanup,
        elapsed_seconds=time.monotonic()-args.started,**FLAGS))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('packet','packet-sha256','source-commit','output'):ap.add_argument('--'+n,required=True)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--runtime-stage');ap.add_argument('--runtime-stage-sha256')
    ap.add_argument('--parent-started',type=float)
    args=ap.parse_args();args.started,args.process_started=time.monotonic(),time.process_time();args.owned_child=None
    def interrupted(sig,frame):raise InterruptedError('scientific preparation signal '+str(sig))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,interrupted)
    out=Path(args.output).absolute()
    # Reject overlap before claiming output, even when authentication later fails.
    preliminary=decode(read(args.packet,args.packet_sha256,16*1024**2))
    resource_contract(preliminary['resources'],preliminary['deadlines'])
    separated(out,Path(args.packet).absolute().parent,preliminary['collection']['root'],preliminary['model_directory'])
    out.mkdir(mode=0o700,exist_ok=False)
    try:
        origin=args.parent_started if args.worker else args.started
        need(type(origin) is float and origin<=time.monotonic(),'parent monotonic origin required')
        end=origin+preliminary['deadlines']['work' if args.worker else 'hard']
        need(end>time.monotonic(),'preauthentication deadline expired');signal.setitimer(signal.ITIMER_REAL,end-time.monotonic())
        write(out/'start.json',dict(schema=VERSION,packet_sha256=args.packet_sha256,resources=preliminary['resources'],deadlines=preliminary['deadlines'],**FLAGS))
        ctx=authenticate(args)
        if args.worker:
            need(args.runtime_stage and pin(args.runtime_stage_sha256),'independently pinned runtime stage required');child(args,ctx)
        else:parent(args,ctx)
    except BaseException as exc:
        try:failure(out,exc)
        except BaseException as publication:
            if args.owned_child is not None:args.owned_child.error('failure_publication',publication)
            try:print('Failure publication:',repr(publication),'Original:',repr(exc),file=sys.stderr)
            except BaseException:pass
        raise
    finally:
        if args.owned_child is not None:
            args.owned_child.hold_until_closed()
            if args.owned_child.errors:
                try:print('Lifecycle diagnostics:',args.owned_child.errors,file=sys.stderr)
                except BaseException:pass
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
